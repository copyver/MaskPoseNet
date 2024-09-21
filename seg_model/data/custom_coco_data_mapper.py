"""custom standard dataset data mapper"""
import json
import os
import random

import cv2
import numpy as np
import skimage.io
from pycocotools.coco import COCO

import seg_model.utils as utils
from seg_model.data.datagen import Dataset


class CustomCOCODataset(Dataset):
    def __init__(self, dataset_dir, subset, use_depth=False, class_ids=None,
                 transforms=None, split_ratio=0.8, seed=42):
        """
        Initialize the Custom COCO Dataset.

        Parameters:
        - dataset_dir: Root directory of the dataset.
        - subset: 'train' or 'val'. If 'train', the data will be split into training and validation sets.
        - use_depth: Whether to use depth images.
        - class_ids: List of class IDs to include.
        - transforms: Transformations to apply.
        - split_ratio: Proportion of data to use for training (between 0 and 1).
        - seed: Random seed for reproducibility.
        """
        super().__init__()
        self.coco = None
        self.dataset_dir = dataset_dir
        self.subset = subset  # 'train' or 'val'
        self.use_depth = use_depth
        self.class_ids = class_ids
        self.transforms = transforms
        self.split_ratio = split_ratio
        self.seed = seed
        self._image_ids = []
        self.image_info = []
        self.source_class_ids = {}
        self.camera = {}
        self.load_data(class_ids)
        self.prepare()

    def load_data(self, class_ids=None):
        # Set random seed for reproducibility
        random.seed(self.seed)

        # Define paths
        subset_dir = os.path.join(self.dataset_dir, self.subset)
        coco_ann_path = os.path.join(subset_dir, "scene_instances_gt.json")
        image_dir = os.path.join(subset_dir, "images", "color_ims")
        depth_dir = os.path.join(subset_dir, "images", "depth_ims")
        camera_path = os.path.join(subset_dir, "scene_camera.json")

        # Load camera data if available
        if os.path.exists(camera_path):
            with open(camera_path, 'r') as f:
                camera_data = json.load(f)
                self.camera = camera_data

        # Initialize COCO API
        coco = COCO(coco_ann_path)
        self.coco = coco

        # Load all classes or a subset
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("handle", i, coco.loadCats(i)[0]["name"])

        # Collect all images and annotations
        all_image_info = []
        for i in image_ids:
            img_info = coco.imgs[i]
            image_id = img_info['id']
            image_path = os.path.join(image_dir, img_info['file_name'])
            depth_path = os.path.join(depth_dir, img_info['file_name']) if self.use_depth else None

            # Get annotations
            ann_ids = coco.getAnnIds(imgIds=[image_id], catIds=class_ids, iscrowd=None)
            anns = coco.loadAnns(ann_ids)

            # Store camera info if available
            camera_info = None
            if self.camera and str(image_id) in self.camera:
                camera_info = self.camera[str(image_id)]

            # Collect image information
            image_record = {
                "source": "handle",
                "id": image_id,
                "path": image_path,
                "depth_path": depth_path,
                "width": img_info["width"],
                "height": img_info["height"],
                "annotations": anns,
                "camera": camera_info
            }

            all_image_info.append(image_record)

        # Shuffle the data
        random.shuffle(all_image_info)

        # Split the data into training and validation sets
        split_index = int(len(all_image_info) * self.split_ratio)
        if self.subset == 'train':
            selected_images = all_image_info[:split_index]
        elif self.subset == 'val':
            selected_images = all_image_info[split_index:]
        else:
            selected_images = all_image_info

        # Add selected images to the dataset
        for image_info in selected_images:
            self.image_info.append(image_info)
            self._image_ids.append(image_info['id'])

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array."""
        image_info = self.image_info[image_id]
        image = skimage.io.imread(image_info['path'])
        # If grayscale, convert to RGB
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # Remove alpha channel if present
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_depth_image(self, image_id, use_norm=False):
        """Load the depth image and process it."""
        image_info = self.image_info[image_id]
        depth_path = image_info['depth_path']
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        camera_info = image_info['camera']

        if camera_info:
            cam_K = np.array(camera_info["cam_K"]).reshape(3, 3)
            depth_scale = camera_info.get("depth_scale", 1.0)
        else:
            cam_K = np.array([[1062.67, 0, 646.166],
                              [0, 1062.67, 474.238],
                              [0, 0, 1]])
            depth_scale = 1.0

        # Apply depth scale
        depth = depth * depth_scale / 1000

        # Process depth data (implement utils.process_depth_data as needed)
        if use_norm:
            imgX, imgY, imgZ, imgN = utils.process_depth_data(depth, cam_K, use_norm)
        else:
            imgX, imgY, imgZ = utils.process_depth_data(depth, cam_K, use_norm)
        XYZ = np.stack((imgX, imgY, imgZ), axis=-1)
        if use_norm:
            XYZ = np.concatenate((XYZ, np.expand_dims(imgN, axis=-1)), axis=-1)
        return XYZ

    def load_mask(self, image_id):
        """Generate instance masks for an image."""
        image_info = self.image_info[image_id]
        annotations = image_info["annotations"]
        height = image_info["height"]
        width = image_info["width"]

        instance_masks = []
        class_ids = []

        for annotation in annotations:
            class_id = self.map_source_class_id("handle.{}".format(annotation['category_id']))
            if class_id:
                # Convert annotation to mask
                mask = self.annToMask(annotation, height, width)
                if mask.max() < 1:
                    continue
                if annotation.get('iscrowd', 0):
                    # Handle crowd annotations if any
                    class_id *= -1
                    if mask.shape[0] != height or mask.shape[1] != width:
                        mask = np.ones([height, width], dtype=bool)
                instance_masks.append(mask)
                class_ids.append(class_id)

        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Return an empty mask
            mask = np.zeros((height, width, 0), dtype=bool)
            class_ids = np.array([], dtype=np.int32)
            return mask, class_ids


# class CustomCOCODataset(Dataset):
#     def __init__(self, dataset_dir, subset, use_depth=False, class_ids=None, transforms=None, return_coco=False):
#         super().__init__()
#         self.dataset_dir = dataset_dir
#         self.subset = subset
#         self.use_depth = use_depth
#         self.class_ids = class_ids
#         self.transforms = transforms
#         self._image_ids = []
#         self.image_info = []
#         self.source_class_ids = {}
#         self.camera = None
#         self.load_data(class_ids, return_coco)
#         self.prepare()
#
#     def load_image(self, image_id):
#         """Load the specified image and return a [H,W,3] Numpy array.
#         """
#         # Load image
#         image = skimage.io.imread(self.image_info[image_id]['path'])
#         # If grayscale. Convert to RGB for consistency.
#         if image.ndim != 3:
#             image = skimage.color.gray2rgb(image)
#         # If has an alpha channel, remove it for consistency
#         if image.shape[-1] == 4:
#             image = image[..., :3]
#         return image
#
#     def load_data(self, class_ids=None, return_coco=False):
#
#         coco = COCO("{}/{}/scene_instances_gt.json".format(self.dataset_dir, self.subset))
#         image_dir = "{}/{}/images/color_ims".format(self.dataset_dir, self.subset)
#
#         if self.use_depth:
#             depth_dir = "{}/{}/images/depth_ims".format(self.dataset_dir, self.subset)
#             camera_path = "{}/{}/scene_camera.json".format(self.dataset_dir, self.subset)
#             if os.path.exists(camera_path):
#                 self.camera = CameraData(camera_path)
#
#         # Load all classes or a subset?
#         if not class_ids:
#             # All classes
#             class_ids = sorted(coco.getCatIds())
#
#         # All images or a subset?
#         if class_ids:
#             image_ids = []
#             for id in class_ids:
#                 image_ids.extend(list(coco.getImgIds(catIds=[id])))
#             # Remove duplicates
#             image_ids = list(set(image_ids))
#         else:
#             # All images
#             image_ids = list(coco.imgs.keys())
#
#         # Add classes
#         for i in class_ids:
#             self.add_class("coco", i, coco.loadCats(i)[0]["name"])
#
#         # Add images
#         for i in image_ids:
#             self.add_image(
#                 "coco", image_id=i,
#                 path=os.path.join(image_dir, coco.imgs[i]['file_name']),
#                 depth_path=os.path.join(depth_dir, coco.imgs[i]['file_name']) if self.use_depth else None,
#                 width=coco.imgs[i]["width"],
#                 height=coco.imgs[i]["height"],
#                 annotations=coco.loadAnns(coco.getAnnIds(
#                     imgIds=[i], catIds=class_ids, iscrowd=None)))
#         if return_coco:
#             return coco
#
#     def load_mask(self, image_id):
#         # If not a COCO image, delegate to parent class.
#         image_info = self.image_info[image_id]
#
#         instance_masks = []
#         class_ids = []
#         annotations = self.image_info[image_id]["annotations"]
#         # Build mask of shape [height, width, instance_count] and list
#         # of class IDs that correspond to each channel of the mask.
#         for annotation in annotations:
#             class_id = self.map_source_class_id(
#                 "coco.{}".format(annotation['category_id']))
#             if class_id:
#                 m = self.annToMask(annotation, image_info["height"],
#                                    image_info["width"])
#                 if m.max() < 1:
#                     continue
#                 if annotation['iscrowd']:
#                     class_id *= -1
#                     if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
#                         m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
#                 instance_masks.append(m)
#                 class_ids.append(class_id)
#
#         # Pack instance masks into an array
#         if class_ids:
#             mask = np.stack(instance_masks, axis=2).astype(bool)
#             class_ids = np.array(class_ids, dtype=np.int32)
#             return mask, class_ids
#         else:
#             # Call super class to return an empty mask
#             return super(CustomCOCODataset, self).load_mask(image_id)
#
#     def load_depth_image(self, image_id, use_norm=False):
#         # Load depth image
#         depth = cv2.imread(self.image_info[image_id]['depth_path'], cv2.IMREAD_UNCHANGED)
#         if self.camera is not None:
#             cam_K = np.array(self.camera[image_id]).reshape(3, 3)
#             depth_scale = 5
#         else:
#             cam_K = np.array([[1062.67, 0, 646.166], [0, 1062.67, 474.238], [0, 0, 1]])
#             depth_scale = 1
#
#         # apply scale
#         depth = depth / 1000.0 * depth_scale
#
#         # Process depth data
#         if use_norm:
#             imgX, imgY, imgZ, imgN = utils.process_depth_data(depth, cam_K, use_norm)
#         else:
#             imgX, imgY, imgZ = utils.process_depth_data(depth, cam_K, use_norm)
#
#         # Stack XYZ channels
#         XYZ = np.stack((imgX, imgY, imgZ), axis=-1)
#
#         # Append normals if necessary
#         if use_norm:
#             XYZ = np.concatenate((XYZ, np.expand_dims(imgN, axis=-1)), axis=-1)
#
#         return XYZ
if __name__ == "__main__":
    dataser_dir = "../../datasets/datasets_handle_4000t_800v/"
    handle_dataset = CustomCOCODataset(dataser_dir, "test", use_depth=True)
    print("success")