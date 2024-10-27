import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers as KL
from tensorflow.keras.layers import Layer

from pose_model.utils.utils import pairwise_distance


class SinusoidalPositionalEmbedding(Layer):
    def __init__(self, d_model, **kwargs):
        super(SinusoidalPositionalEmbedding, self).__init__(**kwargs)
        if d_model % 2 != 0:
            raise ValueError(f'Sinusoidal positional encoding with odd d_model: {d_model}')
        self.d_model = d_model
        self.div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * (-tf.math.log(10000.0) / d_model))

    @tf.function
    def call(self, emb_indices):
        omegas = tf.reshape(emb_indices, (-1, 1, 1)) * self.div_term
        sin_embeddings = tf.sin(omegas)
        cos_embeddings = tf.cos(omegas)
        embeddings = tf.concat([sin_embeddings, cos_embeddings], axis=-1)
        embeddings = tf.reshape(embeddings, tf.concat([tf.shape(emb_indices), [self.d_model]], axis=0))

        return embeddings


@keras.utils.register_keras_serializable()
class GeometricStructureEmbedding(Layer):
    def __init__(self, config, **kwargs):
        super(GeometricStructureEmbedding, self).__init__(**kwargs)
        self.config = config
        self.sigma_d = self.config['sigma_d']
        self.sigma_a = self.config["sigma_a"]
        self.factor_a = 180.0 / (self.sigma_a * np.pi)
        self.angle_k = self.config['angle_k']
        self.embedding = SinusoidalPositionalEmbedding(self.config['embed_dim'], name="Sinusoidal_positional_embedding")
        self.proj_d = KL.Dense(self.config['embed_dim'], name="GeometricStructureEmbedding_dense_1")
        self.proj_a = KL.Dense(self.config['embed_dim'], name="GeometricStructureEmbedding_dense_2")
        self.reduction_a = self.config['reduction_a']
        if self.reduction_a not in ['max', 'mean']:
            raise ValueError(f'Unsupported reduction mode: {self.reduction_a}.')

    @tf.function
    def get_embedding_indices(self, points):
        """
            Compute the indices of pair-wise distance embedding and triplet-wise angular embedding.

            Args:
                points: torch.Tensor (B, N, 3), input point cloud

            Returns:
                d_indices: torch.FloatTensor (B, N, N), distance embedding indices
                a_indices: torch.FloatTensor (B, N, N, k), angular embedding indices
        """
        num_point = tf.shape(points)[1]
        dist_map = tf.sqrt(pairwise_distance(points, points))  # [B, N, N]

        d_indices = dist_map / self.sigma_d  # [B, N, N]
        neg_dist_map = -dist_map
        values, indices = tf.nn.top_k(neg_dist_map, k=self.angle_k + 1, sorted=True)
        knn_indices = indices[:, :, 1:]  # [B, N, K] 每个点与其最近的k个点的索引

        expanded_points = tf.tile(tf.expand_dims(points, axis=1),
                                  [1, num_point, 1, 1])  # [B, N, N, 3] 将每个点的坐标复制到与点云中其他所有点的对比中

        knn_points = tf.gather(expanded_points, knn_indices, batch_dims=2, axis=2)  # (B, N, k, 3) 收集每个点K最近邻的点坐标

        ref_vectors = knn_points - tf.expand_dims(points, axis=2)  # (B, N, k, 3) 每个点到其k最近邻的向量

        anc_vectors = tf.expand_dims(points, axis=1) - tf.expand_dims(points,
                                                                      axis=2)  # (B, N, N, 3) 点到所有其他点的向量

        # 扩展 ref_vectors 和 anc_vectors
        ref_vectors = tf.tile(tf.expand_dims(ref_vectors, axis=2), [1, 1, num_point, 1, 1])  # (B, N, N, k, 3)
        anc_vectors = tf.tile(tf.expand_dims(anc_vectors, axis=3), [1, 1, 1, self.angle_k, 1])  # (B, N, N, k, 3)

        sin_values = tf.norm(tf.linalg.cross(ref_vectors, anc_vectors), axis=-1)  # (B, N, N, k)
        cos_values = tf.reduce_sum(ref_vectors * anc_vectors, axis=-1)  # (B, N, N, k)
        angles = tf.math.atan2(sin_values, cos_values)  # (B, N, N, k)
        a_indices = angles * self.factor_a  # (B, N, N, k)

        return d_indices, a_indices

    def call(self, points):
        d_indices, a_indices = self.get_embedding_indices(points)
        d_embeddings = self.embedding(d_indices)  # [B, N, N, embed_dim]
        d_embeddings = self.proj_d(d_embeddings)  # [B, N, N, embed_dim]
        a_embeddings = self.embedding(a_indices)  # [B, N, N, k, embed_dim]
        a_embeddings = self.proj_a(a_embeddings)  # [B, N, N, k, embed_dim]

        if self.reduction_a == 'max':
            a_embeddings = tf.reduce_max(a_embeddings, axis=3)  # 对第四个维度即k维度，代表每个点的k个最近邻的值进行最大化操作
        else:
            a_embeddings = tf.reduce_mean(a_embeddings, axis=3)

        embeddings = d_embeddings + a_embeddings
        return embeddings

    def get_config(self):
        config = super(GeometricStructureEmbedding, self).get_config()
        config.update({
            'config': self.config
        })
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
