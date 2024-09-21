import tensorflow as tf
from tensorflow.keras import layers as KL

from seg_model.block.attention import AttentionalFeatureFuse as AFF


class ConcatFuseFpn(KL.Layer):
    def __init__(self, feature_size):
        super(ConcatFuseFpn, self).__init__()
        self.conv1 = KL.Conv2D(feature_size, (1, 1), name='fpn_c5p5')
        self.conv2 = KL.Conv2D(feature_size, (1, 1), name='fpn_c4p4')
        self.conv3 = KL.Conv2D(feature_size, (1, 1), name='fpn_c3p3')
        self.conv4 = KL.Conv2D(feature_size, (1, 1), name='fpn_c2p2')
        self.conv5 = KL.Conv2D(feature_size, (3, 3), padding="SAME", name="fpn_p2")
        self.conv6 = KL.Conv2D(feature_size, (3, 3), padding="SAME", name="fpn_p3")
        self.conv7 = KL.Conv2D(feature_size, (3, 3), padding="SAME", name="fpn_p4")
        self.conv8 = KL.Conv2D(feature_size, (3, 3), padding="SAME", name="fpn_p5")
        self.up1 = KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")
        self.up2 = KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")
        self.up3 = KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")
        self.pool1 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")

    def call(self, feature1, feature2):
        _, C2, C3, C4, C5 = feature1
        _, D2, D3, D4, D5 = feature2

        # Concatenate the corresponding feature maps
        C2D2 = tf.concat([C2, D2], axis=-1)
        C3D3 = tf.concat([C3, D3], axis=-1)
        C4D4 = tf.concat([C4, D4], axis=-1)
        C5D5 = tf.concat([C5, D5], axis=-1)

        # Top-down pathway
        P5 = self.conv1(C5D5)
        P4 = KL.Add(name="fpn_p4add")([self.up1(P5), self.conv2(C4D4)])
        P3 = KL.Add(name="fpn_p3add")([self.up2(P4), self.conv3(C3D3)])
        P2 = KL.Add(name="fpn_p2add")([self.up3(P3), self.conv4(C2D2)])

        # Apply 3x3 convolutions to the outputs
        P2 = self.conv5(P2)
        P3 = self.conv6(P3)
        P4 = self.conv7(P4)
        P5 = self.conv8(P5)

        # Generate P6
        P6 = self.pool1(P5)

        # Return feature maps
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        return rpn_feature_maps, mrcnn_feature_maps


class RGBBasedFuse(KL.Layer):
    def __init__(self, feature_size):
        super(RGBBasedFuse, self).__init__()
        self.aff1 = AFF(channels=feature_size, r=16, stage=1)
        self.aff2 = AFF(channels=feature_size, r=16, stage=2)
        self.aff3 = AFF(channels=feature_size, r=16, stage=3)
        self.aff4 = AFF(channels=feature_size, r=16, stage=4)
        self.conv1 = KL.Conv2D(feature_size, (1, 1), name='rbf_c5p5')
        # self.conv2 = KL.Conv2D(feature_size, kernel_size=(3, 3), strides=(2, 2), padding='same', name='rbf_d4p5')
        self.conv3 = KL.Conv2D(feature_size, (1, 1), name='rbf_c4p4')
        # self.conv4 = KL.Conv2D(feature_size, kernel_size=(3, 3), strides=(2, 2), padding='same', name='rbf_d3p4')
        self.conv5 = KL.Conv2D(feature_size, (1, 1), name='rbf_c3p3')
        # self.conv6 = KL.Conv2D(feature_size, kernel_size=(3, 3), strides=(2, 2), padding='same', name='rbf_d2p3')
        self.conv7 = KL.Conv2D(feature_size, (1, 1), name='rbf_c2p2')
        # self.conv8 = KL.Conv2D(feature_size, kernel_size=(1, 1), strides=(1, 1), padding='same', name='rbf_d1p2')
        self.pool1 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="rbf_p6")

        self.cb1 = tf.keras.Sequential([KL.Conv2D(feature_size, kernel_size=(1, 1), strides=(1, 1)),
                                        KL.BatchNormalization()])
        self.cb2 = tf.keras.Sequential([KL.Conv2D(feature_size, kernel_size=(1, 1), strides=(2, 2)),
                                        KL.BatchNormalization()])
        self.cb3 = tf.keras.Sequential([KL.Conv2D(feature_size, kernel_size=(1, 1), strides=(2, 2)),
                                        KL.BatchNormalization()])
        self.cb4 = tf.keras.Sequential([KL.Conv2D(feature_size, kernel_size=(1, 1), strides=(2, 2)),
                                        KL.BatchNormalization()])

        self.conv8 = KL.Conv2D(feature_size, (3, 3), padding="SAME", name="fpn_p2")
        self.conv9 = KL.Conv2D(feature_size, (3, 3), padding="SAME", name="fpn_p3")
        self.conv10 = KL.Conv2D(feature_size, (3, 3), padding="SAME", name="fpn_p4")
        self.conv11 = KL.Conv2D(feature_size, (3, 3), padding="SAME", name="fpn_p5")

    def call(self, feature1, feature2):
        _, C2, C3, C4, C5 = feature1
        D1, D2, D3, D4, _ = feature2

        P2 = self.aff1(self.conv1(C2), self.cb1(D1))
        P3 = self.aff2(self.conv3(C3), self.cb2(D2))
        P4 = self.aff3(self.conv5(C4), self.cb3(D3))
        P5 = self.aff4(self.conv7(C5), self.cb4(D4))

        # Apply 3x3 convolutions to the outputs
        P2 = self.conv8(P2)
        P3 = self.conv9(P3)
        P4 = self.conv10(P4)
        P5 = self.conv11(P5)

        P6 = self.pool1(P5)

        # Return feature maps
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        return rpn_feature_maps, mrcnn_feature_maps


class RGBDCrossFuse(KL.Layer):
    def __init__(self, feature_size):
        super(RGBDCrossFuse, self).__init__()
        self.aff2 = AFF(channels=feature_size, r=16, stage=2)
        self.aff3 = AFF(channels=feature_size, r=16, stage=3)
        self.aff4 = AFF(channels=feature_size, r=16, stage=4)
        self.conv1 = KL.Conv2D(feature_size, (1, 1), name='fpn_c5p5')
        self.conv2 = KL.Conv2D(feature_size, (1, 1), name='fpn_c4p4')
        self.conv3 = KL.Conv2D(feature_size, (1, 1), name='fpn_c3p3')
        self.conv4 = KL.Conv2D(feature_size, (1, 1), name='fpn_c2p2')
        self.conv5 = KL.Conv2D(feature_size, (3, 3), padding="SAME", name="fpn_p2")
        self.conv6 = KL.Conv2D(feature_size, (3, 3), padding="SAME", name="fpn_p3")
        self.conv7 = KL.Conv2D(feature_size, (3, 3), padding="SAME", name="fpn_p4")
        self.conv8 = KL.Conv2D(feature_size, (3, 3), padding="SAME", name="fpn_p5")

        self.up1 = KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")
        self.up2 = KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")
        self.up3 = KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")
        self.pool1 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")

    def call(self, feature1, feature2):
        _, C2, C3, C4, C5 = feature1
        _, D2, D3, D4, D5 = feature2

        C2D2 = tf.concat([C2, D2], axis=-1)
        C3D3 = tf.concat([C3, D3], axis=-1)
        C4D4 = tf.concat([C4, D4], axis=-1)
        C5D5 = tf.concat([C5, D5], axis=-1)

        P5 = self.conv1(C5D5)
        P4 = self.aff4(self.up1(P5), self.conv2(C4D4))
        P3 = self.aff3(self.up2(P4), self.conv3(C3D3))
        P2 = self.aff2(self.up3(P3), self.conv4(C2D2))

        # Apply 3x3 convolutions to the outputs
        P2 = self.conv5(P2)
        P3 = self.conv6(P3)
        P4 = self.conv7(P4)
        P5 = self.conv8(P5)

        # Generate P6
        P6 = self.pool1(P5)

        # Return feature maps
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        return rpn_feature_maps, mrcnn_feature_maps


class RGBDFuse(KL.Layer):
    def __init__(self, feature_size):
        super(RGBDFuse, self).__init__()
        self.aff2 = AFF(channels=256, r=16, stage=2)
        self.aff3 = AFF(channels=512, r=16, stage=3)
        self.aff4 = AFF(channels=1024, r=16, stage=4)
        self.aff5 = AFF(channels=2048, r=16, stage=5)
        self.conv1 = KL.Conv2D(1024, (1, 1), name='fpn_c5p5')
        self.conv2 = KL.Conv2D(feature_size, (1, 1), name='fpn_c4p4')
        self.conv3 = KL.Conv2D(feature_size, (1, 1), name='fpn_c3p3')
        self.conv4 = KL.Conv2D(feature_size, (1, 1), name='fpn_c2p2')
        self.conv5 = KL.Conv2D(feature_size, (3, 3), padding="SAME", name="fpn_p2")
        self.conv6 = KL.Conv2D(feature_size, (3, 3), padding="SAME", name="fpn_p3")
        self.conv7 = KL.Conv2D(feature_size, (3, 3), padding="SAME", name="fpn_p4")
        self.conv8 = KL.Conv2D(feature_size, (3, 3), padding="SAME", name="fpn_p5")

        self.up1 = KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")
        self.up2 = KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")
        self.up3 = KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")
        self.pool1 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")

    def call(self, feature1, feature2):
        _, C2, C3, C4, C5 = feature1
        _, D2, D3, D4, D5 = feature2

        P5 = self.aff5(C5, D5)
        P5 = self.conv1(P5)
        P4 = self.aff4(self.up1(P5), self.conv2(self.aff4(C4, D4)))
        P3 = self.aff3(self.up2(P4), self.conv3(self.aff3(C3, D3)))
        P2 = self.aff2(self.up3(P3), self.conv4(self.aff2(C2, D2)))

        # Apply 3x3 convolutions to the outputs
        P2 = self.conv5(P2)
        P3 = self.conv6(P3)
        P4 = self.conv7(P4)
        P5 = self.conv8(P5)

        # Generate P6
        P6 = self.pool1(P5)

        # Return feature maps
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]

        return rpn_feature_maps, mrcnn_feature_maps
