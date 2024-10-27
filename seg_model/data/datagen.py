import json
import time
from abc import ABC, abstractmethod

import numpy as np
from pycocotools import mask as cocomask
from tensorflow.keras import utils as KU

from seg_model import utils
from seg_model.mrcnn import model as mrcnn
from seg_model.mrcnn.utils import generate_pyramid_anchors


class Dataset(ABC):
    """Abstract base class for datasets."""

    def __init__(self):
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]

    @abstractmethod
    def load_data(self):
        """This method must be implemented in subclasses.
        It should load the dataset and return relevant data structures."""
        pass

    @abstractmethod
    def load_image(self, image_id):
        """This method must be implemented in subclasses.
        It should load and return the image corresponding to the given image ID."""
        pass

    @abstractmethod
    def load_depth_image(self, image_id, use_norm=False):
        """This method must be implemented in subclasses.
        It should load and return the depth image corresponding to the given image ID."""
        pass

    @abstractmethod
    def load_mask(self, image_id):
        """This method must be implemented in subclasses.
        It should load and return the mask corresponding to the given image ID."""
        pass

    @property
    def image_ids(self):
        return self._image_ids

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        # Mapping from source class and image IDs to internal IDs
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self.image_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

        print("Number of images: %d" % self.num_images)
        print("Number of classes: %d" % self.num_classes)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's available online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = cocomask.frPyObjects(segm, height, width)
            rle = cocomask.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = cocomask.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = cocomask.decode(rle)
        if len(m.shape) > 2:
            m = m[..., 0]
        return m


class DataGenerator(KU.Sequence):
    def __init__(self, dataset, config, shuffle=True, augmentation=None,
                 random_rois=0, detection_targets=False):
        self.image_ids = np.copy(dataset.image_ids)
        self.dataset = dataset
        self.config = config
        self.use_norm = config.USE_NORM
        self.backbone_shapes = utils.compute_backbone_shapes(config, config.IMAGE_SHAPE)
        self.anchors = generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                                config.RPN_ANCHOR_RATIOS,
                                                self.backbone_shapes,
                                                config.BACKBONE_STRIDES,
                                                config.RPN_ANCHOR_STRIDE)
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.random_rois = random_rois
        self.batch_size = self.config.BATCH_SIZE
        self.detection_targets = detection_targets

    def __len__(self):
        return int(np.ceil(len(self.image_ids) / float(self.batch_size)))

    def __getitem__(self, idx):
        b = 0
        image_index = -1

        while b < self.batch_size:
            image_index = (image_index + 1) % len(self.image_ids)
            if self.shuffle and image_index == 0:
                np.random.shuffle(self.image_ids)

            # Load RGB and XYZN images along with other data
            image_id = self.image_ids[image_index]
            image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                mrcnn.load_image_gt(self.dataset, self.config, image_id, augmentation=self.augmentation)
            xyz_image = utils.load_xyz_image(self.dataset, self.config, image_id,
                                             use_norm=self.use_norm)

            # Check for valid data
            if not np.any(gt_class_ids > 0):
                continue

            # RPN Targets
            rpn_match, rpn_bbox = mrcnn.build_rpn_targets(image.shape, self.anchors,
                                                          gt_class_ids, gt_boxes, self.config)

            # Mask R-CNN Targets
            if self.random_rois:
                rpn_rois = mrcnn.generate_random_rois(
                    image.shape, self.random_rois, gt_class_ids, gt_boxes)
                if self.detection_targets:
                    rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask = \
                        mrcnn.build_detection_targets(
                            rpn_rois, gt_class_ids, gt_boxes, gt_masks, self.config)

            # Init batch arrays
            if b == 0:
                batch_image_meta = np.zeros(
                    (self.batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                batch_rpn_match = np.zeros(
                    [self.batch_size, self.anchors.shape[0], 1], dtype=rpn_match.dtype)
                batch_rpn_bbox = np.zeros(
                    [self.batch_size, self.config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
                batch_images = np.zeros(
                    (self.batch_size,) + image.shape, dtype=np.float32)
                batch_xyz_images = np.zeros(
                    (self.batch_size,) + xyz_image.shape, dtype=np.float32)
                batch_gt_class_ids = np.zeros(
                    (self.batch_size, self.config.MAX_GT_INSTANCES), dtype=np.int32)
                batch_gt_boxes = np.zeros(
                    (self.batch_size, self.config.MAX_GT_INSTANCES, 4), dtype=np.int32)
                batch_gt_masks = np.zeros(
                    (self.batch_size, gt_masks.shape[0], gt_masks.shape[1],
                     self.config.MAX_GT_INSTANCES), dtype=gt_masks.dtype)
                if self.random_rois:
                    batch_rpn_rois = np.zeros(
                        (self.batch_size, rpn_rois.shape[0], 4), dtype=rpn_rois.dtype)
                    if self.detection_targets:
                        batch_rois = np.zeros(
                            (self.batch_size,) + rois.shape, dtype=rois.dtype)
                        batch_mrcnn_class_ids = np.zeros(
                            (self.batch_size,) + mrcnn_class_ids.shape, dtype=mrcnn_class_ids.dtype)
                        batch_mrcnn_bbox = np.zeros(
                            (self.batch_size,) + mrcnn_bbox.shape, dtype=mrcnn_bbox.dtype)
                        batch_mrcnn_mask = np.zeros(
                            (self.batch_size,) + mrcnn_mask.shape, dtype=mrcnn_mask.dtype)

            # If more instances than fits in the array, sub-sample from them.
            if gt_boxes.shape[0] > self.config.MAX_GT_INSTANCES:
                ids = np.random.choice(
                    np.arange(gt_boxes.shape[0]), self.config.MAX_GT_INSTANCES, replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:, :, ids]

            # Add to batch
            batch_image_meta[b] = image_meta
            batch_rpn_match[b] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[b] = rpn_bbox
            batch_images[b] = mrcnn.mold_image(image.astype(np.float32), self.config)
            batch_xyz_images[b] = xyz_image.astype(np.float32)
            batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
            batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
            batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks
            if self.random_rois:
                batch_rpn_rois[b] = rpn_rois
                if self.detection_targets:
                    batch_rois[b] = rois
                    batch_mrcnn_class_ids[b] = mrcnn_class_ids
                    batch_mrcnn_bbox[b] = mrcnn_bbox
                    batch_mrcnn_mask[b] = mrcnn_mask
            b += 1

        # along with keras model
        inputs = [batch_images, batch_xyz_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                  batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
        outputs = []

        if self.random_rois:
            inputs.extend([batch_rpn_rois])
            if self.detection_targets:
                inputs.extend([batch_rois])
                # Keras requires that attn_output and targets have the same number of dimensions
                batch_mrcnn_class_ids = np.expand_dims(
                    batch_mrcnn_class_ids, -1)
                outputs.extend(
                    [batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask])

        return inputs, outputs


class CameraData:
    """Load camera cam_K"""

    def __init__(self, annotation_file=None):
        self.dataset = {}
        self.cam = {}
        self.des = {}
        if annotation_file is not None:
            print('loading camera data into memory...')
            tic = time.time()
            with open(annotation_file, 'r') as f:
                dataset = json.load(f)
            if not isinstance(dataset, dict):
                raise ValueError('annotation file format {} not supported'.format(type(dataset)))
            self.dataset = dataset
            self.create_index()
            print('Done (t={:0.2f}s)'.format(time.time() - tic))

    def create_index(self):
        print('creating camera index...')
        for image_id, cam_data in self.dataset.items():
            self.cam[int(image_id)] = cam_data['cam_K']
            self.des[int(image_id)] = cam_data['depth_scale']
        print('camera index created')

    def __getitem__(self, image_id):
        image_id = int(image_id)
        return self.cam[image_id]
