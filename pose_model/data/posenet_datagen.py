import itertools
import json
import os
import time
from collections import defaultdict

import cv2
import numpy as np
import open3d as o3d
import skimage.io
import tensorflow as tf
from pycocotools import mask as maskUtils
from tensorflow.keras import utils as KU


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


def convert_blender_to_pyrender(blender_nocs):
    # 交换 Y 和 Z 轴
    pyrender_nocs = blender_nocs[:, :, [0, 2, 1]]
    # 反转 Y 轴
    pyrender_nocs[:, :, 2] *= -1
    return pyrender_nocs


def visualize_point_cloud(pt1, pt2=None):

    # 将 NumPy 数组转换为 Open3D 的 PointCloud 对象
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pt1)
    pcd1.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0] for _ in range(pt1.shape[0])]))

    # 创建一个可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 将点云添加到可视化窗口
    vis.add_geometry(pcd1)

    if pt2 is not None:
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pt2)
        pcd2.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 1] for _ in range(pt2.shape[0])]))  #
        vis.add_geometry(pcd2)

    # 运行可视化
    vis.run()


def get_random_rotation():
    angles = np.random.rand(3) * 2 * np.pi
    rand_rotation = np.array([
        [1,0,0],
        [0,np.cos(angles[0]),-np.sin(angles[0])],
        [0,np.sin(angles[0]), np.cos(angles[0])]
    ]) @ np.array([
        [np.cos(angles[1]),0,np.sin(angles[1])],
        [0,1,0],
        [-np.sin(angles[1]), 0, np.cos(angles[1])]
    ]) @ np.array([
        [np.cos(angles[2]),-np.sin(angles[2]),0],
        [np.sin(angles[2]), np.cos(angles[2]),0],
        [0, 0, 1]
    ])
    return rand_rotation


def get_point_cloud_from_depth(depth, K, bbox=None):
    cam_fx, cam_fy, cam_cx, cam_cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    im_H, im_W = depth.shape
    xmap = np.array([[i for i in range(im_W)] for j in range(im_H)])
    ymap = np.array([[j for i in range(im_W)] for j in range(im_H)])

    if bbox is not None:
        rmin, rmax, cmin, cmax = bbox
        depth = depth[rmin:rmax, cmin:cmax].astype(np.float32)
        xmap = xmap[rmin:rmax, cmin:cmax].astype(np.float32)
        ymap = ymap[rmin:rmax, cmin:cmax].astype(np.float32)

    pt2 = depth.astype(np.float32)
    pt0 = (xmap.astype(np.float32) - cam_cx) * pt2 / cam_fx
    pt1 = (ymap.astype(np.float32) - cam_cy) * pt2 / cam_fy

    cloud = np.stack([pt0, pt1, pt2]).transpose((1, 2, 0))
    return cloud


def get_bbox(label):
    img_width, img_length = label.shape
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    c_b = cmax - cmin
    b = min(max(r_b, c_b), min(img_width, img_length))
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]

    rmin = center[0] - int(b / 2)
    rmax = center[0] + int(b / 2)
    cmin = center[1] - int(b / 2)
    cmax = center[1] + int(b / 2)

    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return [rmin, rmax, cmin, cmax]


def get_resize_rgb_choose(choose, bbox, img_size):
    rmin, rmax, cmin, cmax = bbox
    crop_h = rmax - rmin
    ratio_h = img_size / crop_h
    crop_w = cmax - cmin
    ratio_w = img_size / crop_w

    row_idx = choose // crop_h
    col_idx = choose % crop_h
    choose = (np.floor(row_idx * ratio_w) * img_size + np.floor(col_idx * ratio_h)).astype(np.int64)

    return choose


class PoseNetDataset(object):
    def __init__(self):
        self.scene_pose_gt_info = []
        self.scene_camera_info = []
        self.scene_instances_gt_info = []
        self._image_ids = []
        self.image_info = []
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]

    def add_class(self, source, class_id, class_name):
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
        image_info.update(kwargs)   # Todo: delete images which have no annotation
        self.image_info.append(image_info)

    def prepare(self):
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [c["name"] for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)
        print("Number of images: %d" % self.num_images)
        print("Number of classes: %d" % self.num_classes)



    @property
    def image_ids(self):
        return self._image_ids

    def load_data(self, dataset_dir, subset):
        image_dir = "{}/{}/images/color_ims".format(dataset_dir, subset)
        depth_dir = "{}/{}/images/depth_ims".format(dataset_dir, subset)
        self.scene_camera_info = self._load_anns(dataset_dir, subset, "scene_camera.json")
        self.scene_instances_gt_info = self._load_anns(dataset_dir, subset, "scene_instances_gt.json")
        self.scene_pose_gt_info = self._load_anns(dataset_dir, subset, "scene_pose_gt.json")
        self.createIndex()
        class_ids = self.getCatIds()
        image_ids = list(self.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("yhlever", i, self.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "yhlever", image_id=i,
                path=os.path.join(image_dir, self.imgs[i]['file_name']),
                depth_path=os.path.join(depth_dir, self.imgs[i]['file_name']),
                width=self.imgs[i]["width"],
                height=self.imgs[i]["height"],
                annotations=self.loadAnns(self.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))


    def _load_anns(self, dataset_dir, subset, annotation_key):
        assert annotation_key in ["scene_camera.json", "scene_instances_gt.json", "scene_pose_gt.json"], \
            'annotation file format {} not supported'.format(annotation_key)
        annotations_file = os.path.join(dataset_dir, subset, annotation_key)
        print('loading annotations into memory...')
        tic = time.time()
        with open(annotations_file, 'r') as f:
            dataset = json.load(f)
        assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
        print('Done (t={:0.2f}s)'.format(time.time() - tic))
        return dataset

    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns,catToImgs = defaultdict(list), defaultdict(list)
        if 'annotations' in self.scene_instances_gt_info:
            for ann in self.scene_instances_gt_info['annotations']:
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann

        if 'images' in self.scene_instances_gt_info:
            for img in self.scene_instances_gt_info['images']:
                imgs[img['id']] = img

        if 'categories' in self.scene_instances_gt_info:
            for cat in self.scene_instances_gt_info['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.scene_instances_gt_info and 'categories' in self.scene_instances_gt_info:
            for ann in self.scene_instances_gt_info['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])

        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

        print('index created!')

    def getCatIds(self):
        cats = self.scene_instances_gt_info['categories']
        ids = [cat['id'] for cat in cats]
        return ids

    def loadCats(self, ids):
        return [self.cats[ids]]

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def getAnnIds(self, imgIds=[], catIds=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(imgIds) == len(catIds) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds) == 0 else [ann for ann in anns if ann['category_id'] in catIds]

        if not iscrowd == None:
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]
        return ids

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image

    def load_depth_image(self, image_id):
        image = skimage.io.imread(self.image_info[image_id]['depth_path'])
        image = image[:, :, 0]
        return image

    def load_image_annotations(self, image_id):
        annotations = self.image_info[image_id]['annotations']
        return annotations

    def load_pose_Rt(self, image_id, annotation_id):
        # 转换image_id为字符串，因为字典的键是字符串形式
        image_id = str(image_id)
        # 确保数据存在于指定的image_id中
        if image_id in self.scene_pose_gt_info:
            # 遍历所有annotations
            for annotation in self.scene_pose_gt_info[image_id]:
                # 检查当前annotation的id是否匹配
                if annotation['annotation_id'] == annotation_id:
                    return annotation['cam_R_m2c'], annotation['cam_t_m2c']
        return None, None  # 如果没有找到匹配的id，返回None

    def load_camera_k(self, image_id):
        image_id = str(image_id)
        if image_id in self.scene_camera_info:
            return self.scene_camera_info[image_id]['cam_K']

    def load_mask(self, image_id, annotation_id):
        image_info = self.image_info[image_id]
        annotation = self.anns[annotation_id]
        m = self.annToMask(annotation, image_info["height"],
                           image_info["width"])
        if m.max() < 1:
            return None
        return m

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
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
        m = maskUtils.decode(rle)
        return m


class DataGenerator(KU.Sequence):
    def __init__(self, dataset, config, shuffle=True, augmentation=None):
        self.image_ids = np.copy(dataset.image_ids)
        self.dataset = dataset
        self.config = config
        self.shuffle = shuffle
        self.color_augmentor = augmentation
        self.batch_size = self.config.BATCH_SIZE
        self.rgb_mask_flag = config.TRAIN_DATASET["rgb_mask_flag"]
        self.n_sample_template_point = config.TRAIN_DATASET["n_sample_template_point"]
        self.img_size = config.TRAIN_DATASET["img_size"]
        self.file_base = config.TRAIN_DATASET["file_base"]
        self.dilate_mask = config.TRAIN_DATASET["dilate_mask"]
        self.n_sample_observed_point = config.TRAIN_DATASET["n_sample_observed_point"]
        self.shift_range = config.TRAIN_DATASET["shift_range"]
        self.on_epoch_end()


    def __len__(self):
        return int(np.ceil(len(self.image_ids) / float(self.batch_size)))

    def __getitem__(self, index):
        b = 0
        while b < self.batch_size:

            image_id = self.image_ids[index]

            annotations = self.dataset.load_image_annotations(image_id)  # annotations in image

            num_annotations = len(annotations)
            if num_annotations <= 0:
                index += 1
                continue
            valid_idx = np.random.randint(0, num_annotations)
            assert 0 <= valid_idx < len(
                annotations), f"Valid index {valid_idx} out of range for annotations of length {len(annotations)}"
            valid_annotation = annotations[valid_idx]  # select an annotation
            category_id = valid_annotation['category_id']  # set category_id to choose model
            annotation_id = valid_annotation['id']  # set annotation_id

            # target_R, target_t
            target_R, target_t = self.dataset.load_pose_Rt(image_id, annotation_id)
            target_R = np.array(target_R).reshape(3, 3).astype(np.float32)
            target_t = np.array(target_t).reshape(3).astype(np.float32)

            # camera_k
            camera_k = self.dataset.load_camera_k(image_id)
            camera_k = np.array(camera_k).reshape(3, 3).astype(np.float32)

            # template
            tem1_rgb, tem1_choose, tem1_pts = self.get_template(self.file_base, category_id, 1)
            tem2_rgb, tem2_choose, tem2_pts = self.get_template(self.file_base, category_id, 35)

            if tem1_rgb is None:
                continue
            # mask
            mask = self.dataset.load_mask(image_id, annotation_id)
            if np.sum(mask) == 0:
                continue

            if self.dilate_mask and np.random.rand() < 0.5:
                mask = np.array(mask > 0).astype(np.uint8)
                mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=4)
            mask = (mask * 255).astype(np.uint8)

            bbox = get_bbox(mask > 0)
            y1, y2, x1, x2 = bbox
            mask = mask[y1:y2, x1:x2]
            choose = mask.astype(np.float32).flatten().nonzero()[0]

            # depth
            depth = self.dataset.load_depth_image(image_id).astype(np.float32)
            depth = depth / 1000.0 * 5  # 5为depth_scale
            pts = get_point_cloud_from_depth(depth, camera_k, [y1, y2, x1, x2])
            pts = pts.reshape(-1, 3)[choose, :]

            target_pts = (pts - target_t[None, :]) @ target_R
            tem_pts = np.concatenate([tem1_pts, tem2_pts], axis=0)
            radius = np.max(np.linalg.norm(tem_pts, axis=1))
            target_radius = np.linalg.norm(target_pts, axis=1)
            flag = target_radius < radius * 1.2  # for outlier removal

            pts = pts[flag]
            choose = choose[flag]

            if len(choose) < 32:
                continue

            if len(choose) <= self.n_sample_observed_point:
                choose_idx = np.random.choice(np.arange(len(choose)), self.n_sample_observed_point)
            else:
                choose_idx = np.random.choice(np.arange(len(choose)), self.n_sample_observed_point, replace=False)
            choose = choose[choose_idx]
            pts = pts[choose_idx]

            # rgb
            rgb = self.dataset.load_image(image_id).astype(np.uint8)
            rgb = rgb[..., ::-1][y1:y2, x1:x2, :]
            if np.random.rand() < 0.8 and self.color_augmentor is not None:
                rgb = self.color_augmentor.augment_image(rgb)
            if self.rgb_mask_flag:
                rgb = rgb * (mask[:, :, None] > 0).astype(np.uint8)
            rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            rgb = self.transform(np.array(rgb))
            rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], self.img_size)

            # rotation aug
            rand_R = get_random_rotation()
            tem1_pts = tem1_pts @ rand_R
            tem2_pts = tem2_pts @ rand_R
            target_R = target_R @ rand_R

            # translation aug
            add_t = np.random.uniform(-self.shift_range, self.shift_range, (1, 3))
            target_t = target_t + add_t[0]
            add_t = add_t + 0.001 * np.random.randn(pts.shape[0], 3)
            pts = np.add(pts, add_t)

            # Init batch arrays
            if b == 0:
                batch_pts = np.zeros(
                    (self.batch_size,) + pts.shape, dtype=pts.dtype)
                batch_rgb = np.zeros(
                    (self.batch_size,) + rgb.shape, dtype=np.float32)
                batch_rgb_choose = np.zeros(
                    (self.batch_size,) + rgb_choose.shape, dtype=rgb_choose.dtype)
                batch_translation_label = np.zeros(
                    (self.batch_size,) + target_t.shape, dtype=np.float32)
                batch_rotation_label = np.zeros(
                    (self.batch_size,) + target_R.shape, dtype=np.float32)
                batch_tem1_rgb = np.zeros(
                    (self.batch_size,) + tem1_rgb.shape, dtype=np.float32)
                batch_tem1_choose = np.zeros(
                    (self.batch_size,) + tem1_choose.shape, dtype=tem1_choose.dtype)
                batch_tem1_pts = np.zeros(
                    (self.batch_size,) + tem1_pts.shape, dtype=tem1_pts.dtype)
                batch_tem2_rgb = np.zeros(
                    (self.batch_size,) + tem2_rgb.shape, dtype=np.float32)
                batch_tem2_choose = np.zeros(
                    (self.batch_size,) + tem2_choose.shape, dtype=tem2_choose.dtype)
                batch_tem2_pts = np.zeros(
                    (self.batch_size,) + tem2_pts.shape, dtype=tem2_pts.dtype)

            batch_pts[b] = pts
            batch_rgb[b] = rgb
            batch_rgb_choose[b] = rgb_choose
            batch_translation_label[b] = target_t
            batch_rotation_label[b] = target_R
            batch_tem1_rgb[b] = tem1_rgb
            batch_tem1_choose[b] = tem1_choose
            batch_tem1_pts[b] = tem1_pts
            batch_tem2_rgb[b] = tem2_rgb
            batch_tem2_choose[b] = tem2_choose
            batch_tem2_pts[b] = tem2_pts
            b += 1

        inputs = [batch_pts, batch_rgb, batch_rgb_choose, batch_translation_label, batch_rotation_label,
                  batch_tem1_rgb, batch_tem1_choose, batch_tem1_pts,
                  batch_tem2_rgb, batch_tem2_choose, batch_tem2_pts,
                  ]

        outputs = []

        return inputs, outputs

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.image_ids)

    def transform(self, image):
        image = tf.image.convert_image_dtype(image, tf.float32)

        mean = tf.constant([0.485, 0.456, 0.406])
        std = tf.constant([0.229, 0.224, 0.225])
        image = (image - mean) / std

        return image

    def get_template(self, file_base, category_id, tem_index=1):
        category_id = "obj_{}".format(category_id)
        rgb_path = os.path.join(file_base, category_id, 'rgb_' + str(tem_index) + '.png')
        xyz_path = os.path.join(file_base, category_id, 'xyz_' + str(tem_index) + '.npy')
        mask_path = os.path.join(file_base, category_id, 'mask_' + str(tem_index) + '.png')
        if not os.path.isfile(rgb_path):
            raise FileNotFoundError(f"The file '{rgb_path}' does not exist.")

        # mask
        mask = skimage.io.imread(mask_path).astype(np.uint8) == 255
        bbox = get_bbox(mask)
        y1, y2, x1, x2 = bbox
        mask = mask[y1:y2, x1:x2]

        # rgb
        rgb = skimage.io.imread(rgb_path).astype(np.uint8)[..., ::-1][y1:y2, x1:x2, :]
        if np.random.rand() < 0.8 and self.color_augmentor is not None:
            rgb = self.color_augmentor.augment_image(rgb)
        if self.rgb_mask_flag:
            rgb = rgb * (mask[:, :, None] > 0).astype(np.uint8)
        rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        rgb = self.transform(np.array(rgb))

        # xyz
        choose = mask.astype(np.float32).flatten().nonzero()[0]
        if len(choose) <= self.n_sample_template_point:
            choose_idx = np.random.choice(np.arange(len(choose)), self.n_sample_template_point)
        else:
            choose_idx = np.random.choice(np.arange(len(choose)), self.n_sample_template_point, replace=False)
        choose = choose[choose_idx]

        xyz = np.load(xyz_path).astype(np.float32)
        xyz = convert_blender_to_pyrender(xyz)[y1:y2, x1:x2, :]  # 需要转换坐标系
        xyz = xyz.reshape((-1, 3))[choose, :]
        choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], self.img_size)

        return rgb, choose, xyz
