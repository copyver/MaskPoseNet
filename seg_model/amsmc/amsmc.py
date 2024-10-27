import multiprocessing
import os
import re
import datetime
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K

from tensorflow.keras import layers as KL
from tensorflow.keras import models as KM

from seg_model import utils
from seg_model.backbone.resnet import resnet_graph
from seg_model.data.datagen import DataGenerator
from seg_model.block.fuse import ConcatFuseFpn, RGBBasedFuse, RGBDCrossFuse, RGBDFuse
from seg_model.mrcnn import model as mrcnn
from seg_model.mrcnn.utils import resize_image

fuse_classes = {
    'rgb_based_fuse': RGBBasedFuse,
    'rgbd_cross_fuse': RGBDCrossFuse,
    'rgbd_fuse': RGBDFuse,
    'concat_fuse': ConcatFuseFpn
}


def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  ".format(str(array.shape)))
        if array.size:
            text += ("min: {:10.5f}  max: {:10.5f}".format(array.min(), array.max()))
        else:
            text += ("min: {:10}  max: {:10}".format("", ""))
        text += "  {}".format(array.dtype)
    print(text)


class AMSMC(mrcnn.MaskRCNN):
    def __init__(self, mode, seg_cfg, com_seg, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = seg_cfg
        self.use_norm = seg_cfg.USE_NORM
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, seg_cfg=seg_cfg, com_cfg=com_seg)

    def build(self, mode, seg_cfg, com_cfg):
        """Build HFMask R-CNN architecture.
             input_shape: The shape of the input image.
             mode: Either "training" or "inference". The inputs and
                 outputs of the model differ accordingly.
         """
        assert mode in ['training', 'inference']

        # Image size must be dividable by 2 multiple times
        h, w = seg_cfg.IMAGE_SHAPE[:2]
        if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Inputs
        input_rgb = KL.Input(shape=[None, None, seg_cfg.IMAGE_SHAPE[2]], name='input_rgb')
        input_xyz = KL.Input(shape=[None, None, seg_cfg.XYZ_IMAGE_CHANNEL_COUNT], name='input_xyz')

        input_image_meta = KL.Input(shape=[seg_cfg.IMAGE_META_SIZE],
                                    name="input_image_meta")

        if mode == "training":
            # RPN GT
            input_rpn_match = KL.Input(
                shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            input_rpn_bbox = KL.Input(
                shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

            # Detection GT (class IDs, bounding boxes, and masks)
            # 1. GT Class IDs (zero padded)
            input_gt_class_ids = KL.Input(
                shape=[None], name="input_gt_class_ids", dtype=tf.int32)
            # 2. GT Boxes in pixels (zero padded)
            # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
            input_gt_boxes = KL.Input(
                shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)

            # Normalize coordinates
            gt_boxes = KL.Lambda(lambda x: mrcnn.norm_boxes_graph(
                x, K.shape(input_rgb)[1:3]))(input_gt_boxes)
            # 3. GT Masks (zero padded)
            # [batch, height, width, MAX_GT_INSTANCES]
            if seg_cfg.USE_MINI_MASK:
                input_gt_masks = KL.Input(
                    shape=[seg_cfg.MINI_MASK_SHAPE[0],
                           seg_cfg.MINI_MASK_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
            else:
                input_gt_masks = KL.Input(
                    shape=[seg_cfg.IMAGE_SHAPE[0], seg_cfg.IMAGE_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
        elif mode == "inference":
            # Anchors in normalized coordinates
            input_anchors = KL.Input(shape=[None, 4], name="input_anchors")

        # resnet
        attention = None if com_cfg.ATTENTION.TYPE == 'None' else com_cfg.ATTENTION.TYPE

        rgb_feature = resnet_graph(input_rgb, seg_cfg.BACKBONE, branch='_rgb',
                                   stage5=True, train_bn=seg_cfg.TRAIN_BN, attention=attention)
        xyz_feature = resnet_graph(input_xyz, seg_cfg.BACKBONE, branch='_xyz',
                                   stage5=True, train_bn=seg_cfg.TRAIN_BN, attention=attention)

        fuse_type = com_cfg.BACKBONE.get("FUSION")
        if fuse_type in fuse_classes:
            fuse_layer = fuse_classes[fuse_type](seg_cfg.TOP_DOWN_PYRAMID_SIZE)
        else:
            raise ValueError(f"Unknown fuse type: {fuse_type}")
        rpn_feature_maps, mrcnn_feature_maps = fuse_layer(rgb_feature, xyz_feature)

        # Anchors
        if mode == "training":
            anchors = self.get_anchors(seg_cfg.IMAGE_SHAPE)
            # Duplicate across the batch dimension because Keras requires it
            # TODO: can this be optimized to avoid duplicating the anchors?
            anchors = np.broadcast_to(anchors, (seg_cfg.BATCH_SIZE,) + anchors.shape)

            # A hack to get around Keras's bad support for constants
            # This class returns a constant layer
            class ConstLayer(tf.keras.layers.Layer):
                def __init__(self, x, name=None):
                    super(ConstLayer, self).__init__(name=name)
                    self.x = tf.Variable(x)

                def call(self, input):
                    return self.x

            anchors = ConstLayer(anchors, name="anchors")(input_rgb)
        else:
            anchors = input_anchors

        # RPN Model
        rpn = mrcnn.build_rpn_model(seg_cfg.RPN_ANCHOR_STRIDE,
                                    len(seg_cfg.RPN_ANCHOR_RATIOS), seg_cfg.TOP_DOWN_PYRAMID_SIZE)
        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        for p in rpn_feature_maps:
            layer_outputs.append(rpn([p]))
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [KL.Concatenate(axis=1, name=n)(list(o))
                   for o, n in zip(outputs, output_names)]

        rpn_class_logits, rpn_class, rpn_bbox = outputs

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = seg_cfg.POST_NMS_ROIS_TRAINING if mode == "training" \
            else seg_cfg.POST_NMS_ROIS_INFERENCE
        rpn_rois = mrcnn.ProposalLayer(
            proposal_count=proposal_count,
            nms_threshold=seg_cfg.RPN_NMS_THRESHOLD,
            name="ROI",
            config=seg_cfg)([rpn_class, rpn_bbox, anchors])

        if mode == "training":
            # Class ID mask to mark class IDs supported by the dataset the image
            # came from.
            active_class_ids = KL.Lambda(
                lambda x: mrcnn.parse_image_meta_graph(x)["active_class_ids"]
            )(input_image_meta)

            if not seg_cfg.USE_RPN_ROIS:
                # Ignore predicted ROIs and use ROIs provided as an input.
                input_rois = KL.Input(shape=[seg_cfg.POST_NMS_ROIS_TRAINING, 4],
                                      name="input_roi", dtype=np.int32)
                # Normalize coordinates
                target_rois = KL.Lambda(lambda x: mrcnn.norm_boxes_graph(
                    x, K.shape(input_rgb)[1:3]))(input_rois)
            else:
                target_rois = rpn_rois

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            rois, target_class_ids, target_bbox, target_mask = \
                mrcnn.DetectionTargetLayer(seg_cfg, name="proposal_targets")([
                    target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

            # Network Heads
            # TODO: verify that this handles zero padded ROIs
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = \
                mrcnn.fpn_classifier_graph(rois, mrcnn_feature_maps, input_image_meta,
                                           seg_cfg.POOL_SIZE, seg_cfg.NUM_CLASSES,
                                           train_bn=seg_cfg.TRAIN_BN,
                                           fc_layers_size=seg_cfg.FPN_CLASSIF_FC_LAYERS_SIZE)

            mrcnn_mask = mrcnn.build_fpn_mask_graph(rois, mrcnn_feature_maps,
                                                    input_image_meta,
                                                    seg_cfg.MASK_POOL_SIZE,
                                                    seg_cfg.NUM_CLASSES,
                                                    train_bn=seg_cfg.TRAIN_BN)

            # TODO: clean up (use tf.identify if necessary)
            output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

            # Losses
            rpn_class_loss = KL.Lambda(lambda x: mrcnn.rpn_class_loss_graph(*x), name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = KL.Lambda(lambda x: mrcnn.rpn_bbox_loss_graph(seg_cfg, *x), name="rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, rpn_bbox])
            class_loss = KL.Lambda(lambda x: mrcnn.mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
                [target_class_ids, mrcnn_class_logits, active_class_ids])
            bbox_loss = KL.Lambda(lambda x: mrcnn.mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
                [target_bbox, target_class_ids, mrcnn_bbox])
            mask_loss = KL.Lambda(lambda x: mrcnn.mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
                [target_mask, target_class_ids, mrcnn_mask])

            # Model
            inputs = [input_rgb, input_xyz, input_image_meta,
                      input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes, input_gt_masks]
            if not seg_cfg.USE_RPN_ROIS:
                inputs.append(input_rois)
            outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                       mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
                       rpn_rois, output_rois,
                       rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss]
            model = KM.Model(inputs, outputs, name='amsmc')

        else:
            # Network Heads
            # Proposal classifier and BBox regressor heads
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = \
                mrcnn.fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, input_image_meta,
                                           seg_cfg.POOL_SIZE, seg_cfg.NUM_CLASSES,
                                           train_bn=seg_cfg.TRAIN_BN,
                                           fc_layers_size=seg_cfg.FPN_CLASSIF_FC_LAYERS_SIZE)

            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in
            # normalized coordinates
            detections = mrcnn.DetectionLayer(seg_cfg, name="mrcnn_detection")(
                [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

            # Create masks for detections
            detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
            mrcnn_mask = mrcnn.build_fpn_mask_graph(detection_boxes, mrcnn_feature_maps,
                                                    input_image_meta,
                                                    seg_cfg.MASK_POOL_SIZE,
                                                    seg_cfg.NUM_CLASSES,
                                                    train_bn=seg_cfg.TRAIN_BN)

            model = KM.Model([input_rgb, input_xyz, input_image_meta, input_anchors],
                             [detections, mrcnn_class, mrcnn_bbox,
                              mrcnn_mask, rpn_rois, rpn_class, rpn_bbox],
                             name='amsmc')

        # Add multi-GPU support.
        if seg_cfg.GPU_COUNT > 1:
            from seg_model.mrcnn.parallel_model import ParallelModel
            model = ParallelModel(model, seg_cfg.GPU_COUNT)

        return model

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
              augmentation=None, custom_callbacks=None, no_augmentation_sources=None):

        assert self.mode == "training", "Create model in training mode."
        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(rgbd_cross_fuse)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(rgbd_cross_fuse)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(rgbd_cross_fuse)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers,
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = DataGenerator(train_dataset, self.config, shuffle=True,
                                        augmentation=augmentation)
        val_generator = DataGenerator(val_dataset, self.config, shuffle=True)

        # Create log_dir if it does not exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True, save_best_only=False),
        ]

        # Add custom callbacks to the list
        if custom_callbacks:
            callbacks += custom_callbacks

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        if os.name == 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        self.keras_model.fit(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=2,
            use_multiprocessing=False,
        )
        self.epoch = max(self.epoch, epochs)

    def mold_xyz(self, depth, cam_k, use_norm):
        if use_norm:
            imgX, imgY, imgZ, imgN = utils.process_depth_data(depth, cam_k, use_norm)
        else:
            imgX, imgY, imgZ = utils.process_depth_data(depth, cam_k, use_norm)

        # Stack XYZ channels
        XYZ = np.stack((imgX, imgY, imgZ), axis=-1)

        # Append normals if necessary
        if use_norm:
            XYZ = np.concatenate((XYZ, np.expand_dims(imgN, axis=-1)), axis=-1)

        return XYZ

    def mold_depths(self, depths, cam_k, use_norm):
        molded_depths = []
        for depth in depths:
            depth = self.mold_xyz(depth, cam_k, use_norm)
            molded_depth, _, _, _, _ = resize_image(
                depth,
                min_dim=self.config.IMAGE_MIN_DIM,
                min_scale=self.config.IMAGE_MIN_SCALE,
                max_dim=self.config.IMAGE_MAX_DIM,
                mode=self.config.IMAGE_RESIZE_MODE)
            molded_depths.append(molded_depth)
        molded_depths = np.stack(molded_depths)
        return molded_depths

    def detect(self, images, depth_images, cam_k, verbose=0):
        assert self.mode == "inference", "Create model in inference mode."
        assert len(
            images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)
            for depth_image in depth_images:
                log("depth image", depth_image)

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)
        molded_depth_images = self.mold_depths(depth_images, cam_k, self.use_norm)
        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, \
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("molded_depth_images", molded_depth_images)
            log("image_metas", image_metas)
            log("anchors", anchors)

        # Run object detection
        detections, _, _, mrcnn_mask, _, _, _ = \
            self.keras_model.predict([molded_images, molded_depth_images, image_metas, anchors], verbose=0)

        # Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks = \
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                       image.shape, molded_images[i].shape,
                                       windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            The path of the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            import errno
            raise FileNotFoundError(
                errno.ENOENT,
                "Could not find model directory under {}".format(self.model_dir))
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("amsmc"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            import errno
            raise FileNotFoundError(
                errno.ENOENT, "Could not find weight files in {}".format(dir_name))
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return checkpoint

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
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # \path\to\logs\coco20171029T2315\amsmc_coco_0001.h5 (Windows)
            # /path/to/logs/coco20171029T2315/amsmc_coco_0001.h5 (Linux)
            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]mask\_rcnn\_[\w-]+(\d{4})\.h5"
            # Use string for regex since we might want to use pathlib.Path as model_path
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
        self.checkpoint_path = os.path.join(self.log_dir, "amsmc_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")

    def summary(self, *args, **kwargs):
        self.keras_model.summary(*args, **kwargs)

    def count_params(self):
        return self.keras_model.count_params()