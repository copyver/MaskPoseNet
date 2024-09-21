"""LineMod dataset data mapper"""
from seg_model.data.datagen import Dataset
from seg_model import utils
import os
import json
import numpy as np
import skimage.io
import skimage.color
import cv2
from pycocotools.coco import COCO
import random


class LMDataset(Dataset):
    def __init__(self, dataset_dir, subset, use_depth=False, class_ids=None,
                 transforms=None, split_ratio=0.8, seed=42):
        """
        Initialize the LineMOD dataset.

        Parameters:
        - dataset_dir: Root directory of the dataset.
        - subset: 'train' or 'test'. If 'train', the data will be split into training and validation sets.
        - use_depth: Whether to use depth images.
        - class_ids: List of class IDs to include.
        - transforms: Transformations to apply.
        - split_ratio: Proportion of data to use for training (between 0 and 1).
        - seed: Random seed for reproducibility.
        """
        super().__init__()
        self.coco = None
        self.dataset_dir = dataset_dir
        self.subset = subset  # 'train' or 'val' or 'test'
        self.use_depth = use_depth
        self.class_ids = class_ids
        self.transforms = transforms
        self._image_ids = []
        self.image_info = []
        self.source_class_ids = {}
        self.camera = {}
        self.split_ratio = split_ratio
        self.seed = seed
        self.load_data(class_ids)
        self.prepare()

    def load_data(self, class_ids=None):
        # Set random seed for reproducibility
        random.seed(self.seed)

        # Get list of object IDs (as strings)
        subset_dir = os.path.join(self.dataset_dir, 'test' if self.subset in ['train', 'val'] else self.subset)
        obj_dirs = [d for d in sorted(os.listdir(subset_dir))
                    if os.path.isdir(os.path.join(subset_dir, d))]

        # If class_ids are specified, filter obj_dirs
        if class_ids:
            obj_dirs = [str(obj_id) for obj_id in class_ids if str(obj_id) in obj_dirs]

        # Add classes
        for obj_id in obj_dirs:
            self.add_class("lm", int(obj_id), "object_{}".format(obj_id))

        # Initialize variables to collect all images and annotations
        all_image_info = []
        image_id_counter = 0  # To assign unique image IDs

        # Iterate over each object directory
        for obj_id in obj_dirs:
            obj_dir = os.path.join(subset_dir, obj_id)
            rgb_dir = os.path.join(obj_dir, "rgb")
            depth_dir = os.path.join(obj_dir, "depth")
            camera_path = os.path.join(obj_dir, "scene_camera.json")
            coco_ann_path = os.path.join(obj_dir, "scene_gt_coco.json")

            # Load camera information if available
            if os.path.exists(camera_path):
                with open(camera_path, 'r') as f:
                    camera_data = json.load(f)

            # Initialize COCO API for the current object's annotations
            coco = COCO(coco_ann_path)
            self.coco = coco

            # Load all images for this object
            image_ids = list(coco.imgs.keys())

            # Add images and annotations
            for img_id in image_ids:
                img_info = coco.imgs[img_id]

                # Handle the 'file_name' that includes 'rgb/' prefix
                image_filename = img_info['file_name']
                # Ensure the path does not duplicate 'rgb/' when constructing the full path
                if image_filename.startswith('rgb/'):
                    image_path = os.path.join(obj_dir, image_filename)
                else:
                    image_path = os.path.join(rgb_dir, image_filename)

                depth_path = os.path.join(depth_dir, os.path.basename(image_filename)) if self.use_depth else None

                # Assign a unique image ID
                unique_image_id = image_id_counter
                image_id_counter += 1

                # Get annotations for this image
                ann_ids = coco.getAnnIds(imgIds=[img_id])
                anns = coco.loadAnns(ann_ids)

                # Adjust annotations with the new unique image ID and category ID
                for ann in anns:
                    ann['image_id'] = unique_image_id
                    ann['category_id'] = int(obj_id)

                # Store camera info if available
                camera_info = None
                if os.path.exists(camera_path):
                    img_id_str = str(img_id)
                    if img_id_str in camera_data:
                        camera_info = camera_data[img_id_str]

                # Collect image information
                image_record = {
                    "source":"lm",
                    "id": unique_image_id,
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
            # For 'test' or other subsets, use all images
            selected_images = all_image_info

        # Add the selected images to the dataset
        for image_info in selected_images:
            self.image_info.append(image_info)
            self._image_ids.append(image_info['id'])

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array."""
        image_info = self.image_info[image_id]
        image = skimage.io.imread(image_info['path'])
        # If grayscale, convert to RGB for consistency
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
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
            depth_scale = camera_info["depth_scale"]
        else:
            # Default camera parameters if not available
            cam_K = np.array([[572.4114, 0, 325.2611],
                              [0, 573.57043, 242.04899],
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
            class_id = self.map_source_class_id("lm.{}".format(annotation['category_id']))
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


if __name__ =="__main__":
    dataset_dir = "/home/yhlever/DeepLearning/6D_object_pose_estimation/Datasets/lm"
    dataset_train = LMDataset(dataset_dir,"val", use_depth=True);
    print("success")

