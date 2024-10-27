import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers as KL


def apply_attention(input_tensor, attention_type, stage, block, reduction=16):
    if attention_type == 'SE':
        return se_block(input_tensor, reduction)
    elif attention_type == 'CBAM':
        return cbam_block(input_tensor, reduction)
    elif attention_type == 'PSA':
        return pyramid_squeeze_attention(input_tensor, stage, block, reduction)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")


def se_block(input_tensor, reduction=16):
    channels = input_tensor.shape[-1]
    se = KL.GlobalAveragePooling2D()(input_tensor)
    se = KL.Reshape((1, 1, channels))(se)
    se = KL.Dense(channels // reduction, activation='relu')(se)
    se = KL.Dense(channels, activation='sigmoid')(se)
    se = KL.multiply([input_tensor, se])
    return se


def cbam_block(input_tensor, reduction=16):
    channels = input_tensor.shape[-1]

    # Channel Attention
    avg_pool = KL.GlobalAveragePooling2D()(input_tensor)
    max_pool = KL.GlobalMaxPooling2D()(input_tensor)
    shared_layer_one = KL.Dense(channels // reduction, activation='relu', kernel_initializer='he_normal')
    shared_layer_two = KL.Dense(channels, kernel_initializer='he_normal')
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)
    cbam_feature = KL.Add()([avg_pool, max_pool])
    cbam_feature = KL.Activation('sigmoid')(cbam_feature)

    # Spatial Attention
    avg_pool = K.mean(input_tensor, axis=3, keepdims=True)
    max_pool = K.max(input_tensor, axis=3, keepdims=True)
    concat = KL.Concatenate(axis=3)([avg_pool, max_pool])
    cbam_feature = KL.Conv2D(1, (7, 7), padding='same', activation='sigmoid', kernel_initializer='he_normal',
                             use_bias=False)(concat)
    cbam_feature = KL.multiply([input_tensor, cbam_feature])

    return cbam_feature


def pyramid_squeeze_attention(input_tensor, stage, block, reduction=16):
    """
    Pyramid Squeeze Attention module.

    Arguments:
        input_tensor: Input feature map, tensor of shape (batch, height, width, channels).
        reduction: Reduction ratio for squeeze operation.

    Returns:
        output_tensor: Output feature map with attention applied.
    """
    conv_name_base = 'psa' + str(stage) + block + '_branch'
    channels = input_tensor.shape[-1]

    # Global Average Pooling
    gap = KL.GlobalAveragePooling2D()(input_tensor)
    gap = KL.Reshape((1, 1, channels))(gap)

    # Pyramid Squeeze
    pool1 = K.pool2d(input_tensor, pool_size=(1, 1), strides=(1, 1), padding='same', pool_mode='avg')
    pool2 = K.pool2d(input_tensor, pool_size=(2, 2), strides=(2, 2), padding='same', pool_mode='avg')
    pool4 = K.pool2d(input_tensor, pool_size=(4, 4), strides=(4, 4), padding='same', pool_mode='avg')

    pool1 = KL.Conv2D(channels // reduction, (1, 1), padding='same', activation='relu', name=conv_name_base+'a')(pool1)
    pool2 = KL.Conv2D(channels // reduction, (1, 1), padding='same', activation='relu', name=conv_name_base+'b')(pool2)
    pool4 = KL.Conv2D(channels // reduction, (1, 1), padding='same', activation='relu', name=conv_name_base+'c')(pool4)

    pool1 = KL.Reshape((1, 1, channels // reduction))(KL.GlobalAveragePooling2D()(pool1))
    pool2 = KL.Reshape((1, 1, channels // reduction))(KL.GlobalAveragePooling2D()(pool2))
    pool4 = KL.Reshape((1, 1, channels // reduction))(KL.GlobalAveragePooling2D()(pool4))

    concat = KL.Concatenate(axis=-1)([gap, pool1, pool2, pool4])
    concat = KL.Conv2D(channels, (1, 1), padding='same', activation='sigmoid', name=conv_name_base+'d')(concat)

    output_tensor = KL.multiply([input_tensor, concat])

    return output_tensor


class AttentionalFeatureFuse(KL.Layer):
    def __init__(self, channels=64, r=4, stage=1, **kwargs):
        super(AttentionalFeatureFuse, self).__init__(**kwargs)
        inter_channels = channels // r
        # Local attention layers
        self.local_att = tf.keras.Sequential([
            KL.Conv2D(inter_channels, kernel_size=1, strides=1, padding='valid'),
            KL.BatchNormalization(),
            KL.ReLU(),
            KL.Conv2D(channels, kernel_size=1, strides=1, padding='valid'),
            KL.BatchNormalization()],
            name='aff_local_{}'.format(stage)
        )

        # Global attention layers
        self.global_att = tf.keras.Sequential([
            KL.GlobalAveragePooling2D(),
            KL.Reshape((1, 1, channels)),
            KL.Conv2D(inter_channels, kernel_size=1, strides=1, padding='valid'),
            KL.BatchNormalization(),
            KL.ReLU(),
            KL.Conv2D(channels, kernel_size=1, strides=1, padding='valid'),
            KL.BatchNormalization()],
            name='aff_global_{}'.format(stage)
        )
        self.sigmoid = KL.Activation('sigmoid')

    def call(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = 2 * x * wei + 2 * residual * (1 - wei)

        return xo
