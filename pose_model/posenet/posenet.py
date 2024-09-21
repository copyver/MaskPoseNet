import datetime
import multiprocessing
import os
import re

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers as KL
from tensorflow.keras import models as KM

from pose_model.posenet.coarse_point_matching import CoarsePointMatching, AddFeatureBgTokenLayer
from pose_model.posenet.fine_point_matching import FinePointMatching, AddPEAndBgTokenLayer
from pose_model.block.geometric_embed import GeometricStructureEmbedding
from pose_model.utils.loss import compute_pred_acc, ComputeTrueLabel, ComputeCorrespondenceLoss
from pose_model.data.posenet_datagen import DataGenerator
from pose_model.utils.utils import (sample_pts_feats, compute_coarse_Rt, compute_fine_Rt)
from pose_model.backbone.vit import ViTEncoder


class PoseModel(object):
    """
    Encapsulates the PoseModel functionality.
    the actual Keras model is in the keras_model property.
    """
    def __init__(self, mode, config, model_dir, logger):
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.logger = logger
        self.vit_npoint = config.VIT_NPOINT
        self.coarse_npoint = config.COARSE_NPOINT
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)

    def build(self, mode, config):
        """
        Build PoseNet architecture.
        mode: Either "training" or "inference". The inputs and
              outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference']

        # 定义输入
        input_pts = KL.Input(shape=[config.TRAIN_DATASET['n_sample_observed_point'], 3], name="pts")
        input_rgb = KL.Input(shape=[config.IMAGE_DIM, config.IMAGE_DIM, 3], name="rgb")
        input_rgb_choose = KL.Input(shape=[config.TRAIN_DATASET['n_sample_observed_point'], ], name="rgb_choose")
        input_tem1_rgb = KL.Input(shape=[config.IMAGE_DIM, config.IMAGE_DIM, 3], name="tem1_rgb")
        input_tem1_choose = KL.Input(shape=[config.TRAIN_DATASET['n_sample_template_point'], ], name="tem1_choose")
        input_tem1_pts = KL.Input(shape=[config.TRAIN_DATASET['n_sample_template_point'], 3], name="tem1_pts")
        input_tem2_rgb = KL.Input(shape=[config.IMAGE_DIM, config.IMAGE_DIM, 3], name="tem2_rgb")
        input_tem2_choose = KL.Input(shape=[config.TRAIN_DATASET['n_sample_template_point'], ], name="tem2_choose")
        input_tem2_pts = KL.Input(shape=[config.TRAIN_DATASET['n_sample_template_point'], 3], name="tem2_pts")

        if mode == 'training':
            input_translation_label = KL.Input(shape=[3, ], name="translation_label")
            input_rotation_label = KL.Input(shape=[3, 3], name="rotation_label")

        # 利用ViT进行特征提取
        ViTLayer = ViTEncoder(config=config.VIT_CONFIG, npoint=self.vit_npoint, name="ViTEncoder")
        dense_pm, dense_fm, dense_po, dense_fo, radius = ViTLayer(input_rgb, input_rgb_choose, input_pts,
                                                                  input_tem1_rgb, input_tem1_choose, input_tem1_pts,
                                                                  input_tem2_rgb, input_tem2_choose, input_tem2_pts)
        dense_fm = tf.reshape(dense_fm, [-1, config.VIT_NPOINT, config.VIT_CONFIG['out_dim']])

        # 添加背景点
        bg_point = tf.ones((tf.shape(dense_pm)[0], 1, 3), dtype=tf.float32) * 100

        # 采样稀疏点云，稀疏图像特征
        sparse_pm, sparse_fm, fps_idx_m = sample_pts_feats(dense_pm, dense_fm, self.coarse_npoint,
                                                           return_index=True)
        sparse_fm = tf.reshape(sparse_fm, [-1, config.COARSE_NPOINT, config.VIT_CONFIG['out_dim']])
        sparse_po, sparse_fo, fps_idx_o = sample_pts_feats(dense_po, dense_fo, self.coarse_npoint,
                                                           return_index=True)

        # 几何结构特征嵌入
        share_geo_embedding = GeometricStructureEmbedding(config=config.GEOMETRIC_EMBED_CONFIG,
                                                          name="share_geo_embedding")
        geo_embedding_m = share_geo_embedding(tf.concat([bg_point, sparse_pm], axis=1))
        geo_embedding_o = share_geo_embedding(tf.concat([bg_point, sparse_po], axis=1))

        # 稀疏特征添加随机可学习的背景特征
        add_feature_bg_token = AddFeatureBgTokenLayer(hidden_dim=config.COARSE_POINT_MATCHING['hidden_dim'],
                                                      name="add_feature_bg_token")
        sparse_fm, sparse_fo = add_feature_bg_token(sparse_fm, sparse_fo)

        coarse_matching_layer = CoarsePointMatching(config=config.COARSE_POINT_MATCHING, name="CoarsePointMatching")
        coarse_atten_list = coarse_matching_layer(sparse_fm, geo_embedding_m, sparse_fo, geo_embedding_o)

        if mode == 'training':
            # 粗糙点云匹配
            gt_t = tf.cast(input_translation_label / (tf.reshape(radius, (-1, 1)) + 1e-6), dtype=tf.float32)
            gt_R = tf.cast(input_rotation_label, dtype=tf.float32)
            init_R, init_t = compute_coarse_Rt(coarse_atten_list[-1], sparse_pm, sparse_po,
                                               None,
                                               config.COARSE_POINT_MATCHING['nproposal1'],
                                               config.COARSE_POINT_MATCHING['nproposal2'])

        else:
            input_model_pts = KL.Input(shape=[None, 3], name="model_pts")
            init_R, init_t = compute_coarse_Rt(coarse_atten_list[-1], sparse_pm, sparse_po,
                                               None,
                                               config.COARSE_POINT_MATCHING['nproposal1'],
                                               config.COARSE_POINT_MATCHING['nproposal2'])

        # 密集特征添加随机可学习的背景特征和位置编码
        add_pe_and_bg_token = AddPEAndBgTokenLayer(config=config.FINE_POINT_MATCHING)
        dense_fm, dense_fo = add_pe_and_bg_token(init_R, init_t, dense_pm, dense_fm, dense_po, dense_fo)

        fine_matching_layer = FinePointMatching(config=config.FINE_POINT_MATCHING, name="FinePointMatching")
        fine_atten_list = fine_matching_layer(dense_fm, geo_embedding_m, fps_idx_m,
                                              dense_fo, geo_embedding_o, fps_idx_o)

        if mode == 'training':
            # 计算粗糙匹配损失
            compute_coarse_label = ComputeTrueLabel(name="compute_coarse_label")
            coarse_label1, coarse_label2 = compute_coarse_label(sparse_pm, sparse_po, gt_R, gt_t,
                                                                config.COARSE_POINT_MATCHING['loss_dis_thres'])
            compute_coarse_loss = ComputeCorrespondenceLoss(name="compute_coarse_loss")
            coarse_loss = compute_coarse_loss(coarse_label1, coarse_label2, coarse_atten_list)
            coarse_acc = compute_pred_acc(coarse_label1, coarse_atten_list)

            # 计算精细匹配损失
            compute_fine_label = ComputeTrueLabel(name="compute_fine_label")
            fine_label1, fine_label2 = compute_fine_label(dense_pm, dense_po, gt_R, gt_t,
                                                          config.FINE_POINT_MATCHING['loss_dis_thres'])
            compute_fine_loss = ComputeCorrespondenceLoss(name="compute_fine_loss")
            fine_loss = compute_fine_loss(fine_label1, fine_label2, fine_atten_list)
            fine_acc = compute_pred_acc(fine_label1, fine_atten_list)

            # 定义模型输入输出
            inputs = [input_pts, input_rgb, input_rgb_choose,
                      input_translation_label, input_rotation_label,
                      input_tem1_rgb, input_tem1_choose, input_tem1_pts,
                      input_tem2_rgb, input_tem2_choose, input_tem2_pts,
                      ]

            outputs = []

            model = KM.Model(inputs=inputs, outputs=outputs, name="PoseModel")
            model.add_loss(coarse_loss)
            model.add_loss(fine_loss)
            model.add_metric(coarse_loss, name="coarse_loss", aggregation='mean')
            model.add_metric(fine_loss, name="fine_loss", aggregation='mean')
            model.add_metric(coarse_acc, name="coarse_acc", aggregation='mean')
            model.add_metric(fine_acc, name="fine_acc", aggregation='mean')

        else:
            pred_R, pred_t, pred_pose_score = KL.Lambda(lambda x: compute_fine_Rt(*x), name="compute_fine_Rt")(
                [fine_atten_list[-1], dense_pm, dense_po, None])
            pred_t = pred_t * (tf.reshape(radius, [-1, 1]) + 1e-6)
            inputs = [input_pts, input_rgb, input_rgb_choose, input_model_pts,
                      input_tem1_rgb, input_tem1_choose, input_tem1_pts,
                      input_tem2_rgb, input_tem2_choose, input_tem2_pts,
                      ]
            outputs = [pred_R, pred_t, pred_pose_score]
            model = KM.Model(inputs=inputs, outputs=outputs, name="PoseModel")

        if config.GPU_COUNT > 1:
            from seg_model.mrcnn.parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)

        return model

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers, augmentation=None):
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        layer_regex = {
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = DataGenerator(train_dataset, self.config, shuffle=True, augmentation=augmentation)
        val_generator = DataGenerator(val_dataset, self.config, shuffle=True)
        self.logger.info("=> successfully create DataGenerator ...")

        # Create log_dir if it does not exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path, verbose=0, save_weights_only=True, save_freq='epoch'),
        ]

        self.logger.info("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        self.logger.info("Checkpoint Path: {}".format(self.checkpoint_path))

        self.set_trainable(layers)
        self.compile(learning_rate)

        if os.name == 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()  # Todo:多卡训练有误？

        self.keras_model.fit(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=1,
            use_multiprocessing=False,
        )
        save_model_path = os.path.join(self.log_dir, "mask_pose_model.keras")
        self.keras_model.save(save_model_path)
        self.epoch = max(self.epoch, epochs)

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):

        if verbose > 0 and keras_model is None:
            self.logger.info("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") \
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainable layer names
            if trainable and verbose > 0:
                self.logger.info("{}{:20}   ({})".format(" " * indent, layer.name,
                                                         layer.__class__.__name__))

    def compile(self, learning_rate):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0001,
            beta_1=0.5,
            beta_2=0.999,
            epsilon=0.000001,
            decay=0.0
        )

        self.keras_model.compile(optimizer=optimizer)

    def detect(self, pts, rgb, rgb_choose, model_pts,
               tem1_rgb, tem1_choose, tem1_pts,
               tem2_rgb, tem2_choose, tem2_pts):
        pts = np.expand_dims(pts, axis=0)
        rgb = np.expand_dims(rgb, axis=0)
        rgb_choose = np.expand_dims(rgb_choose, axis=0)
        model_pts = np.expand_dims(model_pts, axis=0)
        tem1_rgb = np.expand_dims(tem1_rgb, axis=0)
        tem1_choose = np.expand_dims(tem1_choose, axis=0)
        tem1_pts = np.expand_dims(tem1_pts, axis=0)
        tem2_rgb = np.expand_dims(tem2_rgb, axis=0)
        tem2_choose = np.expand_dims(tem2_choose, axis=0)
        tem2_pts = np.expand_dims(tem2_pts, axis=0)
        pred_R, pred_t, pred_pose_score = self.keras_model.predict([pts, rgb, rgb_choose, model_pts,
                                                                    tem1_rgb, tem1_choose, tem1_pts,
                                                                    tem2_rgb, tem2_choose, tem2_pts], verbose=0)
        results = {"pred_R": pred_R,
                   "pred_t": pred_t,
                   "pred_pose_score": pred_pose_score}
        return results

    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the corresponding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exclude: list of layer names to exclude
        """
        import h5py
        from tensorflow.python.keras.saving import hdf5_format

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        with h5py.File(filepath, mode='r') as f:
            if 'layer_names' not in f.attrs and 'model_weights' in f:
                f = f['model_weights']

            # In multi-GPU training, we wrap the model. Get layers
            # of the inner model because they have the weights.
            keras_model = self.keras_model
            layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
                else keras_model.layers

            # Exclude some layers
            if exclude:
                layers = filter(lambda l: l.name not in exclude, layers)

            if by_name:
                hdf5_format.load_weights_from_hdf5_group_by_name(f, layers)
            else:
                hdf5_format.load_weights_from_hdf5_group(f, layers)

        # Update the log directory
        self.set_log_dir(filepath)

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.
        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]mask\_rcnn\_[\w-]+(\d{4})\.h5"
            m = re.match(regex, str(model_path))
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                # Epoch number in file is 1-based, and in Keras code it's 0-based.
                # So, adjust for that then increment by one to start from the next epoch
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "posemodel_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")

    def summary(self):
        self.keras_model.summary()

    def print_weights(self):
        for layer in self.keras_model.layers:
            weights = layer.get_weights()
            print(f"Weights of {layer.name}: {weights}")
