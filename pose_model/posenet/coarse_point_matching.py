import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers as KL
from tensorflow.keras.layers import Layer

from pose_model.utils.utils import compute_feature_similarity
from pose_model.backbone.vit import GeometricTransformer


@keras.utils.register_keras_serializable()
class AddFeatureBgTokenLayer(Layer):
    def __init__(self, hidden_dim, **kwargs):
        super(AddFeatureBgTokenLayer, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.share_sparse_feature_dense = KL.Dense(hidden_dim, name="share_sparse_feature_dense")
        self.bg_token = self.add_weight(
            shape=(1, 1, hidden_dim),
            initializer=tf.random_normal_initializer(mean=0., stddev=0.02),
            trainable=True,
            name="sparse_bg_token"
        )

    def call(self, f1, f2):
        f1 = self.share_sparse_feature_dense(f1)
        f2 = self.share_sparse_feature_dense(f2)
        f1 = tf.concat([tf.tile(self.bg_token, [tf.shape(f1)[0], 1, 1]), f1], axis=1)
        f2 = tf.concat([tf.tile(self.bg_token, [tf.shape(f2)[0], 1, 1]), f2], axis=1)

        return f1, f2

    def get_config(self):
        config = super(AddFeatureBgTokenLayer, self).get_config()
        config.update({
            'hidden_dim': self.hidden_dim,
        })
        return config


@keras.utils.register_keras_serializable()
class CoarsePointMatching(Layer):
    """
    用于计算点云间的初始变换（旋转和平移）
    """

    def __init__(self, config, return_feat=False, **kwargs):
        super(CoarsePointMatching, self).__init__(**kwargs)
        self.config = config
        self.return_feat = return_feat
        self.nblock = self.config['nblock']
        self.out_proj = KL.Dense(self.config['out_dim'], name="CoarsePointMatching_dense")
        self.transformers = [GeometricTransformer(blocks=['self', 'cross'],
                                                  d_model=self.config['hidden_dim'],
                                                  num_heads=4,
                                                  dropout=None,
                                                  activation_fn='ReLU',
                                                  return_attention_scores=False,
                                                  name="CoarsePointMatching_Geo_block{}".format(idx)
                                                  ) for idx in range(self.nblock)]

    def call(self, f1, geo1, f2, geo2):
        atten_list = []
        for idx, transformer in enumerate(self.transformers):
            f1, f2 = transformer(f1, geo1, f2, geo2)
            atten_list.append(compute_feature_similarity(
                self.out_proj(f1),
                self.out_proj(f2),
                self.config['sim_type'],
                self.config['temp'],
                self.config['normalize_feat']
            ))

        if self.return_feat:
            return atten_list, self.out_proj(f1), self.out_proj(f2)
        else:
            return atten_list

    def get_config(self):
        config = super(CoarsePointMatching, self).get_config()
        config.update({
            'config': self.config,  # Ensure this contains serializable entries
            'return_feat': self.return_feat,
        })
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
