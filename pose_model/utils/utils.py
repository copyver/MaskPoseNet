import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer

from tf_ops.tf_sampling import farthest_point_sample


@tf.function
def get_chosen_pixel_feats(img, choose):
    """
    从图像张量中抽取特定的像素特征
    :param img: [B, H, W, C]  C=256=vit_outdim
    :param choose: [B, num_pixels]
    :return: x: [B, num_pixels, C]
    """
    shape = tf.shape(img)
    B, H, W, C = shape[0], shape[1], shape[2], shape[3]
    img = tf.reshape(img, [B, H * W, C])
    choose_expanded = tf.cast(tf.tile(choose, [B, 1]), tf.int32)
    x = tf.gather(img, choose_expanded, axis=1, batch_dims=1)

    return x


def sample_pts_feats(pts, feats, npoint=2048, return_index=False):
    """
    :param pts: B*N*3
    :param feats: B*N*C
    :param npoint: number of sampled points
    :param return_index: is to return index of sampled points
    :return: pts_sampled [B, N ,3]
             feats_sampled [B, N ,C]  N=config.Coarse_NPOINT
    """
    pts = tf.cast(pts, tf.float32)

    sample_idx = farthest_point_sample(npoint, pts)  # 执行最远点采样

    pts_sampled = tf.gather(pts, sample_idx, batch_dims=1, axis=1)  # 根据索引收集点坐标

    feats_sampled = tf.gather(feats, sample_idx, batch_dims=1, axis=1)  # 根据索引收集点特征

    if return_index:
        return pts_sampled, feats_sampled, sample_idx
    else:
        return pts_sampled, feats_sampled


def compute_feature_similarity(feat1, feat2, type='cosine', temp=1.0, normalize_feat=True):
    """
    Computes the similarity between feature sets.
    Args:
        feat1 (tf.Tensor): (B, N, C)
        feat2 (tf.Tensor): (B, M, C)
        type (str): Type of similarity to compute ('cosine' or 'L2').
        temp (float): Temperature for scaling similarities.
        normalize_feat (bool): If True, normalize features to unit length.

    Returns:
        tf.Tensor: Similarity matrix (B, N, M).
    """
    if normalize_feat:
        feat1 = tf.nn.l2_normalize(feat1, axis=2)
        feat2 = tf.nn.l2_normalize(feat2, axis=2)

    if type == 'cosine':
        atten_mat = tf.matmul(feat1, feat2, transpose_b=True)
    elif type == 'L2':
        # Expanding dimensions for broadcasting subtraction
        # 反向传播进行梯度计算的时候可能会遇到在0处求导的情况，这也是loss突然变为nan的原因，在sqrt()添加一个极小数之后得到解决
        atten_mat = tf.sqrt(tf.add(pairwise_distance(feat1, feat2), 1e-10))
    else:
        raise ValueError("Unsupported similarity type: {}".format(type))

    atten_mat = atten_mat / temp

    return atten_mat


def aug_pose_noise(gt_r, gt_t, std_rots=[15, 10, 5, 1.25, 1], max_rot=45, sel_std_trans=[0.2, 0.2, 0.2], max_trans=0.8):
    B = tf.shape(gt_r)[0]
    std_rot = np.random.choice(std_rots)
    angles = tf.random.normal(shape=(B, 3), mean=0, stddev=std_rot)
    angles = tf.clip_by_value(angles, -max_rot, max_rot)

    zeros = tf.zeros((B, 1, 1))
    ones = tf.ones((B, 1, 1))

    a1 = tf.reshape(angles[:, 0], [B, 1, 1]) * np.pi / 180.0
    a1 = tf.concat([
        tf.concat([tf.cos(a1), -tf.sin(a1), zeros], axis=2),
        tf.concat([tf.sin(a1), tf.cos(a1), zeros], axis=2),
        tf.concat([zeros, zeros, ones], axis=2)
    ], axis=1)

    a2 = tf.reshape(angles[:, 1], [B, 1, 1]) * np.pi / 180.0
    a2 = tf.concat([
        tf.concat([ones, zeros, zeros], axis=2),
        tf.concat([zeros, tf.cos(a2), -tf.sin(a2)], axis=2),
        tf.concat([zeros, tf.sin(a2), tf.cos(a2)], axis=2)
    ], axis=1)

    a3 = tf.reshape(angles[:, 2], [B, 1, 1]) * np.pi / 180.0
    a3 = tf.concat([
        tf.concat([tf.cos(a3), zeros, tf.sin(a3)], axis=2),
        tf.concat([zeros, ones, zeros], axis=2),
        tf.concat([-tf.sin(a3), zeros, tf.cos(a3)], axis=2)
    ], axis=1)

    rand_rot = a1 @ a2 @ a3
    rand_rot = gt_r @ rand_rot

    rand_trans = tf.random.normal(shape=(B, 3), mean=0.0, stddev=sel_std_trans)
    rand_trans = tf.clip_by_value(rand_trans, -max_trans, max_trans)
    rand_trans = gt_t + tf.cast(rand_trans, dtype=tf.float32)
    indices = tf.reshape(tf.range(tf.shape(rand_trans)[0]), [-1, 1])  # 创建行索引
    indices = tf.concat([indices, tf.fill((tf.shape(rand_trans)[0], 1), 2)], axis=1)  # 添加列索引为2
    updated_values = tf.clip_by_value(rand_trans[:, 2], clip_value_min=1e-6, clip_value_max=1e+6)
    rand_trans = tf.tensor_scatter_nd_update(rand_trans, indices, updated_values)

    return rand_rot, rand_trans


def svd_wrapper(H):
    # Convert the TensorFlow tensor to a numpy array
    # Perform SVD using numpy
    U, S, Vt = np.linalg.svd(H)
    # Convert the results back to TensorFlow tensors
    return tf.convert_to_tensor(U), tf.convert_to_tensor(S), tf.convert_to_tensor(Vt)


def weighted_procrustes(src_points, ref_points, weights=None, weight_thresh=0.0, eps=1e-5, return_transform=False,
                        src_centroid=None, ref_centroid=None):
    """
    加权的奇异值分解 (SVD) 计算从源点集 (src_points) 到参考点集 (ref_points) 的刚性变换
    Args:
        src_points: torch.Tensor (B, N, 3) or (N, 3)
        ref_points: torch.Tensor (B, N, 3) or (N, 3)
        weights: torch.Tensor (B, N) or (N,) (default: None)
        weight_thresh: float (default: 0.)
        eps: float (default: 1e-5)
        return_transform: bool (default: False)

    Returns:
        R: torch.Tensor (B, 3, 3) or (3, 3)
        t: torch.Tensor (B, 3) or (3, )
        transform: torch.Tensor (B, 4, 4) or (4, 4)
    """
    if len(src_points.shape) == 2:
        src_points = tf.expand_dims(src_points, axis=0)
        ref_points = tf.expand_dims(ref_points, axis=0)
        if weights is not None:
            weights = tf.expand_dims(weights, axis=0)
        squeeze_first = True
    else:
        squeeze_first = False

    # batch_size = src_points.shape[0]
    batch_size = tf.shape(src_points)[0]  #

    if weights is None:
        weights = tf.ones_like(src_points[:, :, 0])
    weights = tf.where(weights < weight_thresh, tf.zeros_like(weights), weights)
    weights = weights / (tf.reduce_sum(weights, axis=1, keepdims=True) + eps)
    weights = tf.expand_dims(weights, axis=2)

    if src_centroid is None:
        src_centroid = tf.reduce_sum(src_points * weights, axis=1, keepdims=True)
    elif len(src_centroid.shape) == 2:
        src_centroid = tf.expand_dims(src_centroid, axis=1)
    src_points_centered = src_points - src_centroid

    if ref_centroid is None:
        ref_centroid = tf.reduce_sum(ref_points * weights, axis=1, keepdims=True)
    elif len(ref_centroid.shape) == 2:
        ref_centroid = tf.expand_dims(ref_centroid, axis=1)
    ref_points_centered = ref_points - ref_centroid

    H = tf.linalg.matmul(src_points_centered, ref_points_centered * weights, transpose_a=True)

    # U, S, Vt = tf.numpy_function(svd_wrapper, [H], [tf.float32, tf.float32, tf.float32])
    # V = tf.transpose(Vt, perm=[0, 2, 1])
    # Ut = tf.transpose(U, perm=[0, 2, 1])
    with tf.device('/CPU:0'):
        S, U, V = tf.linalg.svd(H)
        Ut = tf.transpose(U, perm=[0, 2, 1])
        det = tf.linalg.det(tf.matmul(V, Ut))

    eye = tf.eye(3, batch_shape=[batch_size])
    indices = tf.stack([tf.range(batch_size), tf.fill([batch_size], 2), tf.fill([batch_size], 2)], axis=1)
    eye = tf.tensor_scatter_nd_update(eye, indices=indices, updates=tf.sign(det))
    R = tf.matmul(tf.matmul(V, eye), Ut)
    t = tf.transpose(ref_centroid, perm=[0, 2, 1]) - tf.linalg.matmul(R, tf.transpose(src_centroid, perm=[0, 2, 1]))
    t = tf.squeeze(t, axis=-1)
    if return_transform:
        transform = tf.eye(4, batch_shape=[batch_size])
        transform = tf.tensor_scatter_nd_update(transform, [[0, 0], [1, 1], [2, 2]], R)
        transform = tf.tensor_scatter_nd_update(transform, [[0, 3], [1, 3], [2, 3]], t)
        if squeeze_first:
            transform = tf.squeeze(transform, axis=0)
        return transform
    else:
        if squeeze_first:
            R = tf.squeeze(R, axis=0)
            t = tf.squeeze(t, axis=0)
        return R, t


class WeightedProcrustes(Layer):
    def __init__(self, weight_thresh=0.5, eps=1e-5, return_transform=False):
        super(WeightedProcrustes, self).__init__()
        self.weight_thresh = weight_thresh
        self.eps = eps
        self.return_transform = return_transform

    def call(self, src_points, tgt_points, weights=None, src_centroid=None, ref_centroid=None):
        return weighted_procrustes(
            src_points,
            tgt_points,
            weights=weights,
            weight_thresh=self.weight_thresh,
            eps=self.eps,
            return_transform=self.return_transform,
            src_centroid=src_centroid,
            ref_centroid=ref_centroid
        )

    def get_config(self):
        config = super(WeightedProcrustes, self).get_config()
        config.update({
            'weight_thresh': self.weight_thresh,
            'eps': self.eps,
            'return_transform': self.return_transform
        })
        return config


def compute_coarse_Rt(atten, pts1, pts2, model_pts=None, n_proposal1=6000, n_proposal2=300):
    WSVD = WeightedProcrustes()

    B, N1 = tf.shape(pts1)[0], tf.shape(pts1)[1]
    N2 = tf.shape(pts2)[1]

    if model_pts is None:
        model_pts = pts2
    expand_model_pts = tf.reshape(tf.repeat(model_pts[:, tf.newaxis, :, :], n_proposal2, axis=1),
                                  [B * n_proposal2, -1, 3])

    # 计算软分配矩阵
    pred_score = tf.nn.softmax(atten, axis=2) * tf.nn.softmax(atten, axis=1)
    pred_label1 = tf.math.argmax(pred_score[:, 1:, :], axis=2)
    pred_label2 = tf.math.argmax(pred_score[:, :, 1:], axis=1)
    weights1 = tf.cast(pred_label1 > 0, tf.float32)
    weights2 = tf.cast(pred_label2 > 0, tf.float32)

    pred_score = pred_score[:, 1:, 1:]
    pred_score = pred_score * tf.expand_dims(weights1, axis=2) * tf.expand_dims(weights2, axis=1)
    pred_score = tf.reshape(pred_score, [B, N1 * N2]) ** 1.5

    # 采样姿态假设
    cumsum_weights = tf.cumsum(pred_score, axis=1)
    cumsum_weights /= (tf.expand_dims(cumsum_weights[:, -1], axis=1) + 1e-8)
    idx = tf.searchsorted(cumsum_weights, tf.random.uniform([B, n_proposal1 * 3]))
    idx1, idx2 = idx // N2, idx % N2
    idx1 = tf.clip_by_value(idx1, 0, N1 - 1)
    idx2 = tf.clip_by_value(idx2, 0, N2 - 1)

    p1 = tf.reshape(tf.gather(pts1, idx1, axis=1, batch_dims=1), [B, n_proposal1, 3, 3])
    p2 = tf.reshape(tf.gather(pts2, idx2, axis=1, batch_dims=1), [B, n_proposal1, 3, 3])
    p1 = tf.reshape(p1, [B * n_proposal1, 3, 3])
    p2 = tf.reshape(p2, [B * n_proposal1, 3, 3])
    pred_rs, pred_ts = WSVD(p2, p1, None)  # 需要cpu计算，可能不兼容4090
    pred_rs = tf.reshape(pred_rs, [B, n_proposal1, 3, 3])
    pred_ts = tf.reshape(pred_ts, [B, n_proposal1, 1, 3])

    p1 = tf.reshape(p1, [B, n_proposal1, 3, 3])
    p2 = tf.reshape(p2, [B, n_proposal1, 3, 3])

    transformed_p1 = tf.linalg.matmul(p1-pred_ts, pred_rs)  # Apply rotation and translation
    dis = tf.norm(transformed_p1 - p2, axis=3)
    mean_dis = tf.reduce_mean(dis, axis=2)
    # Get indices of the smallest distances
    _, idx = tf.math.top_k(-mean_dis, k=n_proposal2)  # top_k returns largest by default, hence -mean_dis
    # Gather the best rotations and translations based on indices
    pred_rs = tf.gather(pred_rs, idx, batch_dims=1)
    pred_ts = tf.gather(pred_ts, idx, batch_dims=1)

    # Pose selection
    transformed_pts = tf.linalg.matmul(tf.expand_dims(pts1, 1) - pred_ts, pred_rs)
    transformed_pts = tf.reshape(transformed_pts, [B * n_proposal2, -1, 3])

    # Calculate distance to the expanded model points
    dis = tf.sqrt(pairwise_distance(transformed_pts, expand_model_pts))
    min_dis = tf.reshape(tf.reduce_min(dis, axis=2), [B, n_proposal2, -1])

    # Compute scores based on weights and distances
    weights1_expanded = tf.expand_dims(weights1, 1)
    scores = tf.reduce_sum(weights1_expanded, axis=2) / (tf.reduce_sum(min_dis * weights1_expanded, axis=2) + 1e-8)
    _, idx_final = tf.math.top_k(scores, k=1, sorted=True)

    # Select the best rotation and translation
    pred_R = tf.squeeze(tf.gather(pred_rs, idx_final, batch_dims=1), axis=1)
    pred_t = tf.squeeze(tf.gather(pred_ts, idx_final, batch_dims=1), axis=1)
    pred_t = tf.squeeze(pred_t, axis=1)
    # pred_R.shape (None, 3, 3) pred_t.shape (None, 3)

    return pred_R, pred_t


def compute_fine_Rt(atten, pts1, pts2, model_pts=None, dis_thres=0.15):
    """
    Returns:
        预测的旋转矩阵、平移向量和得分
    """
    if model_pts is None:
        model_pts = pts2

    B, N1 = tf.shape(pts1)[0], tf.shape(pts1)[1]
    N2 = tf.shape(model_pts)[1]
    # compute pose
    WSVD = WeightedProcrustes(weight_thresh=0.0)
    assginment_mat = tf.nn.softmax(atten, axis=2) * tf.nn.softmax(atten, axis=1)
    label1 = tf.math.argmax(assginment_mat[:, 1:, :], axis=2)
    label2 = tf.math.argmax(assginment_mat[:, :, 1:], axis=1)
    weights1 = tf.cast(label1 > 0, tf.float32)
    weights2 = tf.cast(label2 > 0, tf.float32)

    assginment_mat = assginment_mat[:, 1:, 1:]
    assginment_mat = assginment_mat * tf.expand_dims(weights1, axis=2) * tf.expand_dims(weights2, axis=1)
    pred_score = tf.reshape(assginment_mat, [B, N1 * N2]) ** 1.5

    # 采样姿态假设
    cumsum_weights = tf.cumsum(pred_score, axis=1)
    cumsum_weights /= (tf.expand_dims(cumsum_weights[:, -1], axis=1) + 1e-8)

    n_proposal = 6000
    idx = tf.searchsorted(cumsum_weights, tf.random.uniform([B, n_proposal*N1]))
    idx1, idx2 = idx // N2, idx % N2
    idx1 = tf.clip_by_value(idx1, 0, N1 - 1)
    idx2 = tf.clip_by_value(idx2, 0, N2 - 1)

    p1 = tf.reshape(tf.gather(pts1, idx1, axis=1, batch_dims=1), [B, n_proposal, N1, 3])
    p2 = tf.reshape(tf.gather(model_pts, idx2, axis=1, batch_dims=1), [B, n_proposal, N1, 3])
    p1 = tf.reshape(p1, [B*n_proposal, N1, 3])
    p2 = tf.reshape(p2, [B*n_proposal, N1, 3])
    assginment_score = tf.reduce_sum(assginment_mat, axis=2)
    pred_rs, pred_ts = WSVD(p2, p1, assginment_score)  # 需要cpu计算，可能不兼容4090

    n_proposal2 = 300
    pred_rs = tf.reshape(pred_rs, [B, n_proposal, 3, 3])
    pred_ts = tf.reshape(pred_ts, [B, n_proposal, 1, 3])

    p1 = tf.reshape(p1, [B, n_proposal, N1, 3])
    p2 = tf.reshape(p2, [B, n_proposal, N1, 3])

    transformed_p1 = tf.linalg.matmul(p1 - pred_ts, pred_rs)  # Apply rotation and translation
    dis = tf.norm(transformed_p1 - p2, axis=3)
    mean_dis = tf.reduce_mean(dis, axis=2)
    # Get indices of the smallest distances
    _, idx = tf.math.top_k(-mean_dis, k=n_proposal2)  # top_k returns largest by default, hence -mean_dis
    # Gather the best rotations and translations based on indices
    pred_rs = tf.gather(pred_rs, idx, batch_dims=1)
    pred_ts = tf.gather(pred_ts, idx, batch_dims=1)

    # Pose selection
    transformed_pts = tf.linalg.matmul(tf.expand_dims(pts1, 1) - pred_ts, pred_rs)
    transformed_pts = tf.reshape(transformed_pts, [B * n_proposal2, -1, 3])

    expand_model_pts = tf.reshape(tf.repeat(model_pts[:, tf.newaxis, :, :], n_proposal2, axis=1),
                                  [B * n_proposal2, -1, 3])

    # Calculate distance to the expanded model points
    dis = tf.sqrt(pairwise_distance(transformed_pts, expand_model_pts))
    min_dis = tf.reshape(tf.reduce_min(dis, axis=2), [B, n_proposal2, -1])

    # Compute scores based on weights and distances
    weights1_expanded = tf.expand_dims(weights1, 1)
    scores = tf.reduce_sum(weights1_expanded, axis=2) / (tf.reduce_sum(min_dis * weights1_expanded, axis=2) + 1e-8)
    _, idx_final = tf.math.top_k(scores, k=1, sorted=True)

    # Select the best rotation and translation
    pred_R = tf.squeeze(tf.gather(pred_rs, idx_final, batch_dims=1), axis=1)
    pred_t = tf.squeeze(tf.gather(pred_ts, idx_final, batch_dims=1), axis=1)
    pred_t = tf.squeeze(pred_t, axis=1)

    # compute score
    pred_pts = tf.matmul((pts1 - tf.expand_dims(pred_t, 1)), pred_R)
    dis = tf.reduce_min(tf.sqrt(pairwise_distance(pred_pts, model_pts)), axis=2)
    mask = tf.cast((label1 > 0), tf.float32)
    pose_score = tf.cast((dis < dis_thres), tf.float32)
    pose_score = tf.reduce_sum(pose_score * mask, axis=1) / (tf.reduce_sum(mask, axis=1) + 1e-8)
    pose_score = pose_score * tf.reduce_mean(mask, axis=1)

    return pred_R, pred_t, pose_score


def pairwise_distance(
        x: tf.Tensor, y: tf.Tensor, normalized: bool = False, channel_first: bool = False
) -> tf.Tensor:
    """Pairwise distance of two (batched) point clouds in TensorFlow.

    Args:
        x (tf.Tensor): Shape (*, N, C) or (*, C, N)
        y (tf.Tensor): Shape (*, M, C) or (*, C, M)
        normalized (bool, optional): If the points are normalized, we use "d2 = 2 - 2xy". Defaults to False.
        channel_first (bool, optional): If True, the points shape is (*, C, N). Defaults to False.

    Returns:
        tf.Tensor: Shape (*, N, M) containing pairwise distances.
    """
    if channel_first:
        x = tf.transpose(x, perm=[0, 2, 1])  # Transpose to shape (*, N, C)
        xy = tf.linalg.matmul(x, y)  # x is (*, N, C), y is (*, C, M)
    else:
        y = tf.transpose(y, perm=[0, 2, 1])  # Transpose to shape (*, C, M)
        xy = tf.linalg.matmul(x, y)  # x is (*, N, C), y is (*, C, M)

    if normalized:
        sq_distances = 2.0 - 2.0 * xy
    else:
        x2 = tf.reduce_sum(x ** 2, axis=-1, keepdims=True)  # Shape (*, N, 1)
        y2 = tf.reduce_sum(y ** 2, axis=-2, keepdims=True)  # Shape (*, 1, M)
        sq_distances = x2 + y2 - 2 * xy

    return tf.maximum(sq_distances, 0.0)


def assert_normalized_quaternion(quaternion: tf.Tensor):
    with tf.control_dependencies(
        [
            tf.debugging.assert_near(
                tf.ones_like(quaternion[..., 0]),
                tf.linalg.norm(quaternion, axis=-1),
                message="Input quaternions are not normalized.",
            )
        ]
    ):
        return tf.identity(quaternion)


def assert_valid_rotation(rotation_matrix: tf.Tensor):
    r = rotation_matrix
    with tf.control_dependencies(
        [
            tf.debugging.assert_near(
                tf.ones_like(rotation_matrix[..., 0, 0]),
                tf.linalg.det(rotation_matrix),
                message="Invalid rotation matrix.",
            ),
            tf.debugging.assert_near(
                tf.linalg.matmul(r, r, transpose_a=True),
                tf.eye(3, batch_shape=tf.shape(r)[:-2], dtype=r.dtype),
                message="Invalid rotation matrix.",
            ),
        ]
    ):
        return tf.identity(r)


def quaternion_to_rotation_matrix(
    quaternion: tf.Tensor, assert_normalized: bool = False, normalize: bool = False
):
    assert quaternion.shape[-1] == 4
    if normalize:
        quaternion = tf.math.l2_normalize(quaternion, axis=-1)
    if assert_normalized:
        quaternion = assert_normalized_quaternion(quaternion)

    # aliases
    w = quaternion[..., 0]
    x = quaternion[..., 1]
    y = quaternion[..., 2]
    z = quaternion[..., 3]

    # rotation matrix from quaternion
    r11 = 1 - 2 * (y * y + z * z)
    r12 = 2 * (x * y - z * w)
    r13 = 2 * (x * z + y * w)

    r21 = 2 * (x * y + z * w)
    r22 = 1 - 2 * (x * x + z * z)
    r23 = 2 * (y * z - x * w)

    r31 = 2 * (x * z - y * w)
    r32 = 2 * (y * z + x * w)
    r33 = 1 - 2 * (x * x + y * y)

    tf_transform_0 = tf.stack([r11, r21, r31], axis=-1)
    tf_transform_1 = tf.stack([r12, r22, r32], axis=-1)
    tf_transform_2 = tf.stack([r13, r23, r33], axis=-1)
    return tf.stack([tf_transform_0, tf_transform_1, tf_transform_2], axis=-1)


def rotation_matrix_to_quaternion(
    rotation_matrix: tf.Tensor, assert_valid: bool = False
):
    r = rotation_matrix
    assert r.shape[-2:] == [3, 3]
    if assert_valid:
        r = assert_valid_rotation(r)
    # aliases
    r11, r12, r13 = r[..., 0, 0], r[..., 0, 1], r[..., 0, 2]
    r21, r22, r23 = r[..., 1, 0], r[..., 1, 1], r[..., 1, 2]
    r31, r32, r33 = r[..., 2, 0], r[..., 2, 1], r[..., 2, 2]

    nu1 = r11 + r22 + r33
    q1_a = 0.5 * tf.sqrt(1.0 + nu1)
    q1_b = 0.5 * tf.sqrt(
        (tf.square(r32 - r23) + tf.square(r13 - r31) + tf.square(r21 - r12))
        / (3.0 - nu1)
    )
    q1 = tf.where(nu1 > 0.0, q1_a, q1_b)
    nu2 = r11 - r22 - r33
    q2_a = 0.5 * tf.sqrt(1.0 + nu2)
    q2_b = 0.5 * tf.sqrt(
        (tf.square(r32 - r23) + tf.square(r12 + r21) + tf.square(r31 + r13))
        / (3.0 - nu2)
    )
    q2 = tf.where(nu2 > 0.0, q2_a, q2_b)
    nu3 = -r11 + r22 - r33
    q3_a = 0.5 * tf.sqrt(1.0 + nu3)
    q3_b = 0.5 * tf.sqrt(
        (tf.square(r13 - r31) + tf.square(r12 + r21) + tf.square(r23 + r32))
        / (3.0 - nu3)
    )
    q3 = tf.where(nu3 > 0.0, q3_a, q3_b)

    nu4 = -r11 - r22 + r33
    q4_a = 0.5 * tf.sqrt(1.0 + nu4)
    q4_b = 0.5 * tf.sqrt(
        (tf.square(r21 - r12) + tf.square(r31 + r13) + tf.square(r32 + r23))
        / (3.0 - nu4)
    )
    q4 = tf.where(nu4 > 0.0, q4_a, q4_b)

    pos = tf.ones_like(q1)
    neg = -tf.ones_like(q1)
    # assume q1 is positive
    q2_sign = tf.where((r32 - r23) > 0.0, pos, neg)
    q3_sign = tf.where((r13 - r31) > 0.0, pos, neg)
    q4_sign = tf.where((r21 - r12) > 0.0, pos, neg)
    q2 *= q2_sign
    q3 *= q3_sign
    q4 *= q4_sign
    return tf.stack([q1, q2, q3, q4], axis=-1)
