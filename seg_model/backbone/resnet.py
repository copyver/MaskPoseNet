from tensorflow.keras import layers as KL

from seg_model.block.attention import apply_attention


class BatchNorm(KL.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """

    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)


def identity_block(input_tensor, kernel_size, filters, stage, block,
                   use_bias=True, train_bn=True, attention=None):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
        特征图尺寸不变，维度增加,3层
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',
                  use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    if attention:
        x = apply_attention(x, attention, stage, block)

    x = KL.Add()([x, input_tensor])
    x = KL.Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True, train_bn=True, attention=None):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_bias: Boolean. To use or not use a bias in conv layers.
        train_bn: Boolean. Train or freeze Batch Norm layers
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    特征图尺寸减小(默认减半），维度增加, 4层
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = KL.Conv2D(nb_filter1, (1, 1), strides=strides,
                  name=conv_name_base + '2a', use_bias=use_bias)(input_tensor)
    x = BatchNorm(name=bn_name_base + '2a')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',
                  name=conv_name_base + '2b', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2b')(x, training=train_bn)
    x = KL.Activation('relu')(x)

    x = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base +
                                           '2c', use_bias=use_bias)(x)
    x = BatchNorm(name=bn_name_base + '2c')(x, training=train_bn)

    shortcut = KL.Conv2D(nb_filter3, (1, 1), strides=strides,
                         name=conv_name_base + '1', use_bias=use_bias)(input_tensor)
    shortcut = BatchNorm(name=bn_name_base + '1')(shortcut, training=train_bn)

    if attention:
        x = apply_attention(x, attention)

    x = KL.Add()([x, shortcut])
    x = KL.Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def resnet_graph(input_image, architecture, branch, stage5=False, train_bn=True, attention=None):
    """Build a ResNet graph.
        architecture: Can be resnet50 or resnet101
        stage5: Boolean. If False, stage5 of the network is not created
        train_bn: Boolean. Train or freeze Batch Norm layers
    """
    assert architecture in ["resnet50", "resnet101"]

    # Stage 1  尺寸1/4，通道数64
    x = KL.ZeroPadding2D((3, 3))(input_image)
    x = KL.Conv2D(64, (7, 7), strides=(2, 2), name='conv1' + branch, use_bias=True)(x)
    x = BatchNorm(name='bn_conv1' + branch)(x, training=train_bn)
    x = KL.Activation('relu')(x)
    C1 = x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # Stage 2  尺寸1/4 通道数256
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a' + branch, strides=(1, 1),
                   train_bn=train_bn, attention=None)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b' + branch, train_bn=train_bn, attention=None)
    C2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c' + branch, train_bn=train_bn, attention=attention)

    # Stage 3  尺寸1/8 通道数512
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a' + branch, train_bn=train_bn, attention=None)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b' + branch, train_bn=train_bn, attention=None)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c' + branch, train_bn=train_bn, attention=None)
    C3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d' + branch, train_bn=train_bn, attention=attention)

    # Stage 4  尺寸1/16 通道数1024
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a' + branch, train_bn=train_bn, attention=None)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(block_count - 1):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + i) + branch,
                           train_bn=train_bn, attention=None)

    x = identity_block(x, 3, [256, 256, 1024], stage=4, block=chr(98 + block_count) + branch,
                       train_bn=train_bn, attention=attention)
    C4 = x

    # Stage 5  尺寸1/32 通道数2048
    if stage5:
        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a' + branch, train_bn=train_bn, attention=None)
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b' + branch, train_bn=train_bn, attention=None)
        C5 = x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c' + branch,
                                train_bn=train_bn, attention=attention)
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]
