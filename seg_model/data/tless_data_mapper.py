import os
import json


class TLESSDataset(object):
    def __init__(self, dataset_dir, subset, use_depth=False, class_ids=None, transforms=None):
        """
        Initialize the T-LESS dataset.

        :param dataset_dir: Root directory of the T-LESS dataset.
        :param subset: Subset to load: 'train' or 'val'.
        :param use_depth: Boolean to include depth data.
        :param class_ids: List of class IDs to include (None for all).
        :param transforms: Optional transformations to apply to the data.
        """
        super().__init__()
        self.dataset_dir = dataset_dir
        self.subset = subset
        self.use_depth = use_depth
        self.transforms = transforms

        # Load all data
        self.load_all_data(class_ids=class_ids)

        # Prepare the dataset
        self.prepare()

    def load_all_data(self, class_ids=None):
        """
        Load data from all 30 object folders.
        """
        object_folders = [f for f in os.listdir(self.dataset_dir) if os.path.isdir(os.path.join(self.dataset_dir, f))]
        object_folders = sorted(object_folders)  # Ensure consistent ordering

        # If class_ids is None, include all classes
        if class_ids is None:
            class_ids = list(range(1, 31))  # T-LESS has 30 classes

        for class_id in class_ids:
            object_folder = f"{class_id:06d}"  # Assuming folder names like obj_01, obj_02, ..., obj_30
            object_path = os.path.join(self.dataset_dir, object_folder)

            if not os.path.exists(object_path):
                print(f"Warning: Object folder {object_folder} does not exist.")
                continue

            # Add class
            self.add_class("tless", class_id, f"Object_{class_id:02d}")

            # Paths
            rgb_dir = os.path.join(object_path, "rgb")
            depth_dir = os.path.join(object_path, "depth")
            mask_dir = os.path.join(object_path, "mask")
            mask_visib_dir = os.path.join(object_path, "mask_visib")
            camera_file = os.path.join(object_path, "scene_camera.json")
            gt_file = os.path.join(object_path, "scene_gt.json")
            gt_info_file = os.path.join(object_path, "scene_gt_info.json")

            # Load camera data
            if os.path.exists(camera_file):
                with open(camera_file, 'r') as f:
                    camera_data = json.load(f)
                if not self.camera:
                    self.camera = camera_data
                else:
                    self.camera.update(camera_data)

            # Load annotations
            with open(gt_file, 'r') as f:
                gt_data = json.load(f)
            with open(gt_info_file, 'r') as f:
                gt_info_data = json.load(f)

            # Iterate through all images in the RGB directory
            image_files = sorted(os.listdir(rgb_dir))
            for img_file in image_files:
                image_id = int(os.path.splitext(img_file)[0])  # Assuming image names are like 000001.png

                # Load annotations for this image
                annotations = []
                if str(image_id) in gt_data:
                    for obj in gt_data[str(image_id)]:
                        obj_info = gt_info_data[str(image_id)][gt_data[str(image_id)].index(obj)]
                        annotation = {
                            "obj_id": obj["obj_id"],  # Object ID
                            "pose": obj["pose"],  # Pose information
                            "occluded": obj_info["occluded"],  # Occlusion info
                            # Add more fields if necessary
                        }
                        annotations.append(annotation)

                # Add image to the dataset
                self.add_image(
                    "tless",
                    image_id=image_id,
                    path=os.path.join(rgb_dir, img_file),
                    depth_path=os.path.join(depth_dir, img_file) if self.use_depth else None,
                    mask_dir=mask_dir,
                    mask_visib_dir=mask_visib_dir,
                    width=640,  # Adjust if necessary
                    height=480,  # Adjust if necessary
                    annotations=annotations
                )
