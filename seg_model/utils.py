import math
import random
import time
from distutils.version import LooseVersion

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.color
import skimage.io
import skimage.transform
from pycocotools import mask as maskUtils
from pycocotools.cocoeval import COCOeval

from seg_model.mrcnn.utils import resize_image


def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        # New in 0.14: anti_aliasing. Default it to False for backward
        # compatibility with skimage 0.13.
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range, anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma)
    else:
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range)


def resize_depth(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    image_dtype = image.dtype
    crop = None
    h, w = image.shape[:2]
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]

    if mode == "none":
        return image
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))

    if min_scale and scale < min_scale:
        scale = min_scale

    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    if scale != 1:
        image = resize(image, (round(h * scale), round(w * scale)),
                       preserve_range=True)

    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        image = image[y:y + min_dim, x:x + min_dim]
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype)


def process_depth_data(depth, K, use_norm=False):
    if depth is None or not isinstance(depth, np.ndarray):
        raise ValueError("Input depth image is not valid.")
    if K is None or not isinstance(K, np.ndarray) or K.shape != (3, 3):
        raise ValueError("Camera intrinsic matrix K is not valid.")

    if len(depth.shape) == 3:
        depth = depth.astype(np.float32)
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)

    pt2 = depth
    m, n = np.indices(depth.shape)
    pt0 = (n - K[0, 2]) * pt2 / K[0, 0]
    pt1 = (m - K[1, 2]) * pt2 / K[1, 1]
    mPointCloud = np.dstack((pt0, pt1, pt2))

    # 计算表面法线
    if use_norm:
        x_next = np.roll(mPointCloud, -1, axis=1)
        x_prev = np.roll(mPointCloud, 1, axis=1)
        y_next = np.roll(mPointCloud, -1, axis=0)
        y_prev = np.roll(mPointCloud, 1, axis=0)
        u_vector = x_next - x_prev
        v_vector = y_next - y_prev
        normal = np.cross(v_vector, u_vector)
        norm = np.linalg.norm(normal, axis=2, keepdims=True)
        valid_mask = norm.squeeze() != 0
        normal[valid_mask] /= norm.squeeze()[valid_mask][:, np.newaxis]
        normal[0, :], normal[-1, :], normal[:, 0], normal[:, -1] = 0, 0, 0, 0
        return pt0, pt1, pt2, normal

    return pt0, pt1, pt2


def add_noise_by_channel(image, coord_std=0.01, depth_std=0.01, normal_std=0.01):
    """
    根据通道分别为XYZ图像的坐标、深度值和法向量添加噪声。

    参数:
    - image: 输入的图像数组，形状可以是（宽，高，3）或（宽，高，6）。
    - coord_std: 坐标通道的噪声标准差。
    - depth_std: 深度值通道的噪声标准差。
    - normal_std: 法向量通道的噪声标准差。

    返回:
    - 添加噪声后的图像。
    """
    dtype = image.dtype
    noisy_image = np.copy(image).astype(np.float32)
    channels = image.shape[-1]

    # 为坐标和深度值添加噪声
    if channels >= 3:
        # 坐标X, Y和深度Z
        noisy_image[..., 0] += np.random.normal(0, coord_std, image[..., 0].shape)
        noisy_image[..., 1] += np.random.normal(0, coord_std, image[..., 1].shape)
        noisy_image[..., 2] += np.random.normal(0, depth_std, image[..., 2].shape)

    # 如果存在，为法向量添加噪声并重新归一化
    if channels == 6:
        for i in range(3, 6):
            noisy_image[..., i] += np.random.normal(0, normal_std, image[..., i].shape)
        # 重新归一化法向量
        norms = np.linalg.norm(noisy_image[..., 3:6], axis=-1, keepdims=True)
        noisy_image[..., 3:6] /= norms

    noisy_image = np.clip(noisy_image, 0, 255).astype(dtype)
    return noisy_image


def load_xyz_image(dataset, config, image_id, use_norm=False, augmentation=None, add_noise=False,
                   coord_std=0.01, depth_std=0.01, normal_std=0.01):
    xyz = dataset.load_depth_image(image_id, use_norm)

    xyz, window, scale, padding, crop = resize_image(
        xyz,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)

    if augmentation:
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad",
                           "Affine", "PiecewiseAffine"]

        def hook(images, augmenter, parents, default):
            return augmenter.__class__.__name__ in MASK_AUGMENTERS

        # Store shapes before augmentation to compare
        xyz_shape = xyz.shape
        # Make augmenters deterministic to apply similarly to images and masks
        det = augmentation.to_deterministic()
        xyz = det.augment_image(xyz)
        # Change mask to np.uint8 because imgaug doesn't support np.bool
        # Verify that shapes didn't change
        assert xyz.shape == xyz_shape, "Augmentation shouldn't change image size"

    if add_noise:
        xyz = add_noise_by_channel(xyz, coord_std, depth_std, normal_std)

    return xyz


############################################################
#  COCO Evaluation
############################################################
def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []
    results = []
    for image_id in image_ids:
        source = dataset.image_info[image_id]["source"]
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, source),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None,
                  use_depth=False):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)
        if use_depth:
            image_info = dataset.image_info[image_id]
            depth_image = skimage.io.imread(image_info['depth_path'])
            camera_info = image_info['camera']
            if camera_info:
                cam_K = np.array(camera_info["cam_K"]).reshape(3, 3)
                depth_scale = camera_info.get("depth_scale", 1.0)
            depth_image = depth_image * depth_scale / 1000  # Todo:合成数据集depth scale不同于真实数据集

        # Run detection
        t = time.time()
        if use_depth:
            r = model.detect([image], [depth_image], cam_K, verbose=0)[0]
        else:
            r = model.detect([image], verbose=0)[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"],
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {} sec. Average {} sec/image".format(
        t_prediction, t_prediction / len(image_ids)))
    return cocoEval.stats


def show_cocoEval(stats, save_path=None):
    data = {
        'Metric': ['AP', 'AP (IoU=0.50)', 'AP (IoU=0.75)', 'AP (small)', 'AP (medium)', 'AP (large)',
                   'AR (maxDets=1)', 'AR (maxDets=10)', 'AR (maxDets=100)', 'AR (small)', 'AR (medium)', 'AR (large)'],
        'Value': stats
    }

    df = pd.DataFrame(data)

    # Remove metrics with invalid values (-1.000)
    df = df[df['Value'] >= 0]

    # Create bar plots
    plt.figure(figsize=(10, 8))
    bars = plt.barh(df['Metric'], df['Value'], color='skyblue')
    for bar in bars:
        plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{bar.get_width():.3f}',
                 va='center', ha='right', color='blue', fontsize=10)
    plt.xlabel('Metric Values')
    plt.title('COCO Evaluation Results')
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    # Display the plot
    plt.show()


def compute_backbone_shapes(config, image_shape):
    """Computes the width and height of each stage of the backbone network.

    Returns:
        [N, (height, width)]. Where N is the number of stages
    """
    if callable(config.BACKBONE):
        return config.COMPUTE_BACKBONE_SHAPE(image_shape)

    # Currently supports ResNet only
    assert config.BACKBONE in ["resnet50", "resnet101"]
    return np.array(
        [[int(math.ceil(image_shape[0] / stride)),
          int(math.ceil(image_shape[1] / stride))]
         for stride in config.BACKBONE_STRIDES])
