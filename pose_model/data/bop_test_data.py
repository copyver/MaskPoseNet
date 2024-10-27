"""
数据读取
blender和pyrender坐标系不同
"""
import os

import cv2
import numpy as np
import skimage.io
import trimesh

from pose_model.data.posenet_datagen import (get_bbox, get_point_cloud_from_depth,
                                             get_resize_rgb_choose, convert_blender_to_pyrender)
from pose_model.posenet_config import PoseNetConfig

BOP_DATASET = PoseNetConfig.TRAIN_DATASET


def get_template(file_base, category_id, img_size, n_sample_template_point, tem_index=1):
    rgb_path = os.path.join(file_base, f"obj_{category_id}", f"rgb_{tem_index}.png")
    xyz_path = os.path.join(file_base, f"obj_{category_id}", f"xyz_{tem_index}.npy")
    mask_path = os.path.join(file_base, f"obj_{category_id}", f"mask_{tem_index}.png")
    if not os.path.isfile(rgb_path):
        raise FileNotFoundError(f"The file '{rgb_path}' does not exist.")

    # mask
    mask = skimage.io.imread(mask_path).astype(np.uint8) == 255
    bbox = get_bbox(mask)
    y1, y2, x1, x2 = bbox
    mask = mask[y1:y2, x1:x2]

    # rgb
    rgb = skimage.io.imread(rgb_path).astype(np.uint8)[..., ::-1][y1:y2, x1:x2, :]
    rgb = rgb * (mask[:, :, None] > 0).astype(np.uint8)
    rgb = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    # xyz
    choose = mask.astype(np.float32).flatten().nonzero()[0]
    if len(choose) <= n_sample_template_point:
        choose_idx = np.random.choice(np.arange(len(choose)), n_sample_template_point)
    else:
        choose_idx = np.random.choice(np.arange(len(choose)), n_sample_template_point, replace=False)
    choose = choose[choose_idx]

    xyz = np.load(xyz_path).astype(np.float32)
    xyz = convert_blender_to_pyrender(xyz)[y1:y2, x1:x2, :]  # 需要转换坐标系
    xyz = xyz.reshape((-1, 3))[choose, :]
    choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], img_size)

    return rgb, choose, xyz


def read_bop_test_data(dataset, model_mesh, img_index, ann_index):
    img_size = BOP_DATASET['img_size']
    n_sample_template_point = BOP_DATASET['n_sample_template_point']
    n_sample_model_point = BOP_DATASET['n_sample_model_point']

    image_ids = np.copy(dataset.image_ids)

    image_id = image_ids[img_index]

    annotations = dataset.load_image_annotations(image_id)

    num_annotations = len(annotations)

    assert 0 <= ann_index < num_annotations, \
        f"Valid index {ann_index} out of range for annotations of length {num_annotations}"

    valid_annotation = annotations[ann_index]  # select an annotation
    category_id = valid_annotation['category_id']  # set category_id to choose model
    annotation_id = valid_annotation['id']

    target_R, target_t = dataset.load_pose_Rt(image_id, annotation_id)
    target_R = np.array(target_R).reshape(3, 3).astype(np.float32)
    target_t = np.array(target_t).reshape(3).astype(np.float32)
    print("gt_R: ", target_R)
    print("gt_t: ", target_t)

    # camera_k
    camera_k = dataset.load_camera_k(image_id)
    camera_k = np.array(camera_k).reshape(3, 3).astype(np.float32)

    # template
    tem1_rgb, tem1_choose, tem1_pts = get_template(BOP_DATASET['file_base'], category_id,
                                                   img_size, n_sample_template_point, 0)
    tem2_rgb, tem2_choose, tem2_pts = get_template(BOP_DATASET['file_base'], category_id,
                                                   img_size, n_sample_template_point,1)

    # mask
    mask = dataset.load_mask(image_id, annotation_id)
    mask = (mask * 255).astype(np.uint8)
    bbox = get_bbox(mask > 0)
    y1, y2, x1, x2 = bbox
    mask = mask[y1:y2, x1:x2]
    choose = mask.astype(np.float32).flatten().nonzero()[0]

    # depth
    depth = dataset.load_depth_image(image_id).astype(np.float32)
    depth = depth / 1000.0 * 5  #
    pts = get_point_cloud_from_depth(depth, camera_k, [y1, y2, x1, x2])
    pts = pts.reshape(-1, 3)[choose, :]

    n_sample_observed_point = BOP_DATASET['n_sample_observed_point']
    if len(choose) <= n_sample_observed_point:
        choose_idx = np.random.choice(np.arange(len(choose)), n_sample_observed_point)
    else:
        choose_idx = np.random.choice(np.arange(len(choose)), n_sample_observed_point, replace=False)
    choose = choose[choose_idx]
    pts = pts[choose_idx]

    # rgb
    rgb = dataset.load_image(image_id).astype(np.uint8)
    image = rgb.copy()
    rgb = rgb[..., ::-1][y1:y2, x1:x2, :]
    rgb = rgb * (mask[:, :, None] > 0).astype(np.uint8)
    rgb = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], img_size)

    # model_pts
    model_pts, _, _ = trimesh.sample.sample_surface(model_mesh, n_sample_model_point, sample_color=True)
    model_pts = model_pts.astype(np.float32)

    # intrinsic
    camera_k = np.expand_dims(camera_k, axis=0)

    return (pts, rgb, rgb_choose, model_pts,
            tem1_rgb, tem1_choose, tem1_pts,
            tem2_rgb, tem2_choose, tem2_pts,
            camera_k, image)


def read_custom_test_data(rgb, depth, best_mask, category_id, model_mesh, camera_k):
    img_size = BOP_DATASET['img_size']
    n_sample_template_point = BOP_DATASET['n_sample_template_point']
    n_sample_model_point = BOP_DATASET['n_sample_model_point']

    mask = (best_mask * 255).astype(np.uint8)
    bbox = get_bbox(mask > 0)
    y1, y2, x1, x2 = bbox
    mask = mask[y1:y2, x1:x2]
    choose = mask.astype(np.float32).flatten().nonzero()[0]

    # pts
    depth = depth[:, :, 0]
    depth = depth / 1000.0 * 5
    pts = get_point_cloud_from_depth(depth, camera_k, [y1, y2, x1, x2])
    pts = pts.reshape(-1, 3)[choose, :]

    n_sample_observed_point = BOP_DATASET['n_sample_observed_point']
    if len(choose) <= n_sample_observed_point:
        choose_idx = np.random.choice(np.arange(len(choose)), n_sample_observed_point)
    else:
        choose_idx = np.random.choice(np.arange(len(choose)), n_sample_observed_point, replace=False)
    choose = choose[choose_idx]
    pts = pts[choose_idx]

    # resized rgb
    rgb = rgb[..., ::-1][y1:y2, x1:x2, :]
    rgb = rgb * (mask[:, :, None] > 0).astype(np.uint8)
    rgb = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], img_size)

    # model_pts
    model_pts, _, _ = trimesh.sample.sample_surface(model_mesh, n_sample_model_point, sample_color=True)
    model_pts = model_pts.astype(np.float32)

    tem1_rgb, tem1_choose, tem1_pts = get_template(BOP_DATASET['file_base'], category_id,
                                                   img_size, n_sample_template_point,0)
    tem2_rgb, tem2_choose, tem2_pts = get_template(BOP_DATASET['file_base'], category_id,
                                                   img_size, n_sample_template_point,1)

    return (pts, rgb, rgb_choose, model_pts,
            tem1_rgb, tem1_choose, tem1_pts,
            tem2_rgb, tem2_choose, tem2_pts
            )


def read_real_world_data(rgb, depth, best_mask, category_id, camera_k):
    img_size = BOP_DATASET['img_size']
    n_sample_template_point = BOP_DATASET['n_sample_template_point']
    n_sample_model_point = BOP_DATASET['n_sample_model_point']

    mask = (best_mask * 255).astype(np.uint8)
    bbox = get_bbox(mask > 0)
    y1, y2, x1, x2 = bbox
    mask = mask[y1:y2, x1:x2]
    choose = mask.astype(np.float32).flatten().nonzero()[0]

    # pts
    pts = get_point_cloud_from_depth(depth, camera_k, [y1, y2, x1, x2])
    pts = pts.reshape(-1, 3)[choose, :]

    n_sample_observed_point = BOP_DATASET['n_sample_observed_point']
    if len(choose) <= n_sample_observed_point:
        choose_idx = np.random.choice(np.arange(len(choose)), n_sample_observed_point)
    else:
        choose_idx = np.random.choice(np.arange(len(choose)), n_sample_observed_point, replace=False)
    choose = choose[choose_idx]
    pts = pts[choose_idx]

    # resized rgb
    rgb = rgb[..., ::-1][y1:y2, x1:x2, :]
    rgb = rgb * (mask[:, :, None] > 0).astype(np.uint8)
    rgb = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    rgb_choose = get_resize_rgb_choose(choose, [y1, y2, x1, x2], img_size)

    # model_pts
    model_cad_path = os.path.join("templates", f"obj_{category_id}", f"obj_{category_id}.obj")
    model_mesh = trimesh.load(model_cad_path, force='mesh')
    model_pts, _ = trimesh.sample.sample_surface(model_mesh, n_sample_model_point, sample_color=False)
    model_pts = model_pts.astype(np.float32)

    tem1_rgb, tem1_choose, tem1_pts = get_template("templates", category_id,
                                                   img_size, n_sample_template_point, 0)
    tem2_rgb, tem2_choose, tem2_pts = get_template("templates", category_id,
                                                   img_size, n_sample_template_point, 1)

    return (pts, rgb, rgb_choose, model_pts,
            tem1_rgb, tem1_choose, tem1_pts,
            tem2_rgb, tem2_choose, tem2_pts
            )
