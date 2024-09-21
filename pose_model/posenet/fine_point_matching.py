import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers as KL
from tensorflow.keras.layers import Layer

from pose_model.utils.utils import compute_feature_similarity
from pose_model.backbone.vit import SparseToDenseTransformer
from tf_ops.tf_grouping import query_ball_point, group_point


@keras.utils.register_keras_serializable()
class SharedMLP(tf.keras.Sequential):
    def __init__(self, args, bn=False, activation='relu', preact=False, first=False, name=None):
        super(SharedMLP, self).__init__(name=name)
        for i in range(len(args)):
            if preact and (not first or i != 0):
                if bn:
                    self.add(tf.keras.layers.BatchNormalization())
                self.add(tf.keras.layers.Activation(activation))
            self.add(tf.keras.layers.Conv2D(args[i], (1, 1), use_bias=not bn,
                                            name=f"{name}_conv{i}"))
            if not preact or first and i == 0:
                if bn:
                    self.add(tf.keras.layers.BatchNormalization())
                self.add(tf.keras.layers.Activation(activation))


class QueryAndGroup(Layer):
    """
        Groups with a ball query of radius
        Args:
            radius (float): 查询球体的半径。
            nsample (int): 每个球内要考虑的最大样本数。
            use_xyz (bool): 是否将 xyz 坐标作为特征的一部分。
            ret_grouped_xyz (bool): 是否返回分组的 xyz 坐标。
            normalize_xyz (bool): 是否对分组的 xyz 坐标进行归一化处理。
            sample_uniformly (bool): 是否在查询内均匀采样。
            ret_unique_cnt (bool): 是否返回每个查询区域中唯一点的计数。
    """

    def __init__(self, radius, nsample, use_xyz=True, ret_grouped_xyz=False, normalize_xyz=False,
                 sample_uniformly=False, ret_unique_cnt=False, name=None):
        super(QueryAndGroup, self).__init__(name=name)
        self.radius = radius
        self.nsample = nsample
        self.use_xyz = use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.sample_uniformly = sample_uniformly
        self.ret_unique_cnt = ret_unique_cnt

        if self.ret_unique_cnt:
            assert self.sample_uniformly

    def call(self, xyz, new_xyz, features=None):
        """
        Args:
            xyz : tf.Tensor xyz coordinates of the features (B, N, 3)
            new_xyz : tf.Tensor centriods (B, npoint, 3)
            features : torch.Tensor Descriptors of the features (B, N, C)

        Returns
            new_features : torch.Tensor (B, npoint, nsample, C+3) tensor
        """
        idx, pts_cnt = query_ball_point(self.radius, self.nsample, xyz, new_xyz)  # (batch_size, npoint, nsample)

        grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, nsample, 3)

        grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, self.nsample, 1])  # translation normalization

        if features is not None:
            grouped_features = group_point(features, idx)  # (batch_size, npoint, nsample, channel)
            if self.use_xyz:
                new_features = tf.concat(
                    [grouped_xyz, grouped_features], axis=-1
                )  # (batch_size, npoint, nample, 3+channel)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        ret = [new_features]
        if self.ret_grouped_xyz:
            ret.append(grouped_xyz)
        if self.ret_unique_cnt:
            ret.append(pts_cnt)
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)


class PositionalEncoding(Layer):
    def __init__(self, out_dim, r1=0.1, r2=0.2, nsample1=32, nsample2=64, use_xyz=True, bn=True, name=None):
        super(PositionalEncoding, self).__init__(name=name)
        self.group1 = QueryAndGroup(r1, nsample1, use_xyz=use_xyz, name=name+"_queryAndgroup1")
        self.group2 = QueryAndGroup(r2, nsample2, use_xyz=use_xyz, name=name+"_queryAndgroup2")

        self.mlp1 = SharedMLP([32, 64, 128], bn=bn, name=name+"_mlp1")
        self.mlp2 = SharedMLP([32, 64, 128], bn=bn, name=name+"_mlp2")
        self.mlp3 = tf.keras.layers.Conv1D(out_dim, 1, activation=None, use_bias=not bn, name=name+"_mlp3")

    def call(self, pts1, pts2=None):
        """
        Returns:
            feat: Tensor descriptors of the features (B, npoint, out_dim)

        """
        if pts2 is None:
            pts2 = pts1

        # scale1
        feat1 = self.group1(pts1, pts2, pts1)
        feat1 = self.mlp1(feat1)
        feat1 = tf.reduce_max(feat1, axis=-1, keepdims=True)
        # scale2
        feat2 = self.group2(pts1, pts2, pts1)
        feat2 = self.mlp2(feat2)
        feat2 = tf.reduce_max(feat2, axis=-1, keepdims=True)

        feat = tf.concat([feat1, feat2], axis=2)

        feat = tf.squeeze(feat, axis=-1)
        feat = self.mlp3(feat)
        return feat


@keras.utils.register_keras_serializable()
class AddPEAndBgTokenLayer(Layer):
    def __init__(self, config, **kwargs):
        super(AddPEAndBgTokenLayer, self).__init__(**kwargs)
        self.config = config
        self.hidden_dim = config['hidden_dim']
        self.pe_radius1 = config['pe_radius1']
        self.pe_radius2 = config['pe_radius2']
        self.share_dense_feature_dense = KL.Dense(self.hidden_dim, name="share_dense_feature_dense")
        self.PE = PositionalEncoding(self.hidden_dim, r1=self.pe_radius1, r2=self.pe_radius2,
                                     name="positional_encoding")
        # self.bg_token = tf.Variable(tf.random.normal([1, 1, self.hidden_dim]) * 0.02, name="dense_bg_token")
        self.bg_token = self.add_weight(
            shape=(1, 1, self.hidden_dim),
            initializer=tf.random_normal_initializer(mean=0., stddev=0.02),
            trainable=True,
            name="dense_bg_token"
        )

    def call(self, init_R, init_t, p1, f1, p2, f2):
        p1_ = tf.linalg.matmul((p1 - tf.expand_dims(init_t, axis=1)), init_R)
        f1 = self.share_dense_feature_dense(f1) + self.PE(p1_)
        f1 = tf.concat([tf.tile(self.bg_token, [tf.shape(f1)[0], 1, 1]), f1], axis=1)

        f2 = self.share_dense_feature_dense(f2) + self.PE(p2)
        f2 = tf.concat([tf.tile(self.bg_token, [tf.shape(f2)[0], 1, 1]), f2], axis=1)

        return f1, f2

    def get_config(self):
        config = super(AddPEAndBgTokenLayer, self).get_config()
        config.update({
            'config': self.config,
        })
        return config


@keras.utils.register_keras_serializable()
class FinePointMatching(Layer):
    def __init__(self, config, return_feat=False, **kwargs):
        super(FinePointMatching, self).__init__(**kwargs)
        self.config = config
        self.return_feat = return_feat
        self.nblock = self.config['nblock']
        self.out_proj = KL.Dense(self.config['out_dim'], name="FinePointMatching_dense")
        self.transformers = [SparseToDenseTransformer(d_model=self.config['hidden_dim'],
                                                      sparse_blocks=['self', 'cross'],
                                                      num_heads=4,
                                                      dropout=None,
                                                      activation='relu',
                                                      focusing_factor=self.config['focusing_factor'],
                                                      with_bg_token=True,
                                                      replace_bg_token=True,
                                                      name="FinePointMatching_STD_block{}".format(idx)
                                                      ) for idx in range(self.nblock)]

    def call(self,f1, geo1, fps_idx1,f2, geo2, fps_idx2):

        atten_list = []
        for idx in range(self.nblock):
            f1, f2 = self.transformers[idx](f1, geo1, fps_idx1, f2, geo2, fps_idx2)
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
        config = super(FinePointMatching, self).get_config()
        config.update({
            'config': self.config,
            'return_feat': self.return_feat,
        })
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)
