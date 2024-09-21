import tensorflow as tf

from pose_model.utils.utils import pairwise_distance
from tensorflow.keras import layers as KL

@tf.function
def compute_true_label(pts1, pts2, gt_R, gt_t, dis_thres=0.15):
    """
    根据源点云和目标点云以及相应的旋转和平移矩阵返回真实标签
    """
    gt_pts = tf.matmul(pts1 - tf.expand_dims(gt_t, 1), gt_R)
    dis_mat = tf.sqrt(pairwise_distance(gt_pts, pts2))

    dis1, label1 = tf.math.reduce_min(dis_mat, axis=2), tf.math.argmin(dis_mat, axis=2)
    fg_label1 = tf.cast(dis1 <= dis_thres, tf.float32)
    label1 = tf.cast(fg_label1 * (tf.cast(label1, tf.float32) + 1.0), tf.int64)

    dis2, label2 = tf.math.reduce_min(dis_mat, axis=1), tf.math.argmin(dis_mat, axis=1)
    fg_label2 = tf.cast(dis2 <= dis_thres, tf.float32)
    label2 = tf.cast(fg_label2 * (tf.cast(label2, tf.float32) + 1.0), tf.int64)

    return label1, label2


class ComputeTrueLabel(KL.Layer):
    def __init__(self, **kwargs):
        super(ComputeTrueLabel, self).__init__(**kwargs)

    def call(self, pts1, pts2, gt_R, gt_t, dis_thres=0.15):
        return compute_true_label(pts1, pts2, gt_R, gt_t, dis_thres)


@tf.function
def compute_correspondence_loss(label1, label2, atten_list):
    CE = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                       reduction=tf.keras.losses.Reduction.NONE)
    loss = 0
    for idx, atten in enumerate(atten_list):
        l1 = tf.reduce_mean(CE(label1, atten[:, 1:, :]), axis=1)
        l2 = tf.reduce_mean(CE(label2, tf.transpose(atten[:, :, 1:], [0, 2, 1])), axis=1)
        loss += 0.5 * (l1 + l2)

    return tf.reduce_mean(loss)


class ComputeCorrespondenceLoss(KL.Layer):
    def __init__(self, **kwargs):
        super(ComputeCorrespondenceLoss, self).__init__(**kwargs)

    def call(self, label1, label2, atten_list):
        return compute_correspondence_loss(label1, label2, atten_list)


def compute_pred_acc(label1, atten_list):
    atten_last = atten_list[-1][:, 1:, :]
    pred_label = tf.argmax(atten_last, axis=2)
    acc = tf.reduce_mean(tf.cast(tf.equal(pred_label, label1), tf.float32), axis=1)
    return acc


def compute_loss_rotation(pred_R, gt_R):
    """
    计算与姿态估计相关的旋转损失，旋转由3x3旋转矩阵表示。
    该函数计算预测旋转矩阵和目标旋转矩阵之间的几何距离。
    L = arccos( 0.5 * (Trace(R\tilde(R)^T) - 1)
    以弧度计算损失。
    """
    eps = 1e-6

    n_obj = tf.shape(gt_R)[0]

    product = tf.matmul(pred_R, tf.transpose(gt_R, perm=[0, 2, 1]))
    trace = tf.reduce_sum(tf.linalg.diag_part(product), axis=1)
    theta = tf.clip_by_value(0.5 * (trace - 1), -1 + eps, 1 - eps)
    rad = tf.acos(theta)

    loss = tf.reduce_sum(rad) / tf.cast(n_obj, rad.dtype)
    return loss
