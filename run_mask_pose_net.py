import time

import cv2
import numpy as np
import tensorflow as tf

import argparse
from pose_model.data.bop_test_data import read_real_world_data
from pose_model.data.posenet_datagen import visualize_point_cloud
from pose_model.posenet.posenet import PoseModel
from pose_model.posenet_config import PoseNetConfig
from pose_model.utils.visualize import draw_detections
from seg_model.amsmc.amsmc import AMSMC
from seg_model.config import Config
from seg_model.mrcnn.visualize import display_instances
from seg_model.engine.run_seg_net import load_config


def parser_args():
    parser = argparse.ArgumentParser(description='Run MaskPoseNet.')
    parser.add_argument('--config',
                        required=True,
                        metavar="/path/to/net.yaml/",
                        help='YAML to set segmentation model config')
    parser.add_argument('--dataset',
                        required=False,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--seg_net_weights',
                        required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    parser.add_argument('--pose_net_weights',
                        required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    parser.add_argument('--logs',
                        required=True,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--color',
                        required=False,
                        metavar="<color image>",
                        help='Color Image use to detect')
    parser.add_argument('--depth',
                        required=False,
                        metavar="<depth image>",
                        help='Depth Image use to detect')
    args = parser.parse_args()
    return args


class CustomInstanceConfig(Config):
    NAME = "HandleIns"
    USE_NORM = False
    NUM_CLASSES = 1 + 1
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    DETECTION_MIN_CONFIDENCE = 0.9
    DETECTION_NMS_THRESHOLD = 0.4


class CustomPoseNetConfig(PoseNetConfig):
    NAME = 'HandlePose'
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    STEPS_PER_EPOCH = 4000
    VALIDATION_STEPS = 800


class ModelLoader:
    def __init__(self, args):
        self.seg_net_weights = args.seg_net_weights
        self.pose_net_weights = args.pose_net_weights
        self.com_cfg = load_config(args.config)
        self.model_dir = args.logs

        self.amsmc_seg_model_graph = tf.Graph()
        self.pose_model_graph = tf.Graph()

        self.amsmc_sess = tf.compat.v1.Session(graph=self.amsmc_seg_model_graph)
        self.pose_sess = tf.compat.v1.Session(graph=self.pose_model_graph)

        with self.amsmc_seg_model_graph.as_default():
            with self.amsmc_sess.as_default():
                self.amsmc_seg_model = self.load_amsmc_seg_model(self.seg_net_weights)

        with self.pose_model_graph.as_default():
            with self.pose_sess.as_default():
                self.pose_model = self.load_pose_model(self.pose_net_weights)

    def load_amsmc_seg_model(self, weights_path):
        custom_instance_config = CustomInstanceConfig()
        amsmc_seg_model = AMSMC(mode="inference", seg_cfg=custom_instance_config, com_seg=self.com_cfg.MODEL.SEG_MODEL,
                                model_dir=self.model_dir)
        amsmc_seg_model.load_weights(weights_path, by_name=True)
        return amsmc_seg_model

    def load_pose_model(self, weights_path):
        custom_pose_config = CustomPoseNetConfig()
        pose_model = PoseModel(mode="inference", config=custom_pose_config, model_dir=self.model_dir, logger=None)
        pose_model.load_weights(weights_path)
        return pose_model

    def detect_instance(self, color, depth, cam_k):
        with self.amsmc_seg_model_graph.as_default():
            with self.amsmc_sess.as_default():
                results = self.amsmc_seg_model.detect([color], [depth], cam_k, verbose=0)
        return results[0]

    def detect_pose(self, pts, rgb, rgb_choose, model_pts, tem1_rgb, tem1_choose, tem1_pts, tem2_rgb, tem2_choose,
                    tem2_pts):
        with self.pose_model_graph.as_default():
            with self.pose_sess.as_default():
                results = self.pose_model.detect(pts, rgb, rgb_choose, model_pts, tem1_rgb, tem1_choose, tem1_pts,
                                                 tem2_rgb, tem2_choose, tem2_pts)
        return results


def main():
    show_pts = True
    args = parser_args()
    model_loader = ModelLoader(args)
    start_time = time.time()

    cam_k = np.array([[1062.67, 0, 646.166], [0, 1062.67, 474.238], [0, 0, 1]])
    image = cv2.imread(args.color)
    depth_image = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)
    depth_image = depth_image / 1000
    class_names = ['bj', 'handle']

    r = model_loader.detect_instance(image, depth_image, cam_k)

    if r['scores'].size == 0:
        print("No masks...")
    else:
        print(f"Number: {len(r['scores'])}")
        print(f"Scores：{r['scores']}")
        display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], title="Predictions",
                          show_mask=False)

        best_mask = r['masks'][:, :, 0]
        cat_id = r['class_ids'][0]

        (pts, rgb, rgb_choose, model_pts,
         tem1_rgb, tem1_choose, tem1_pts,
         tem2_rgb, tem2_choose, tem2_pts) = read_real_world_data(image, depth_image, best_mask, cat_id, cam_k)
        cam_k = np.expand_dims(cam_k, axis=0)

        results = model_loader.detect_pose(pts, rgb, rgb_choose, model_pts, tem1_rgb, tem1_choose, tem1_pts, tem2_rgb,
                                           tem2_choose, tem2_pts)
        pred_R, pred_t, pred_score = results['pred_R'], results['pred_t'], results['pred_pose_score']
        print("pred_R: ", pred_R)
        print("pred_t: ", pred_t)
        print("pred_score: ", pred_score)
        draw_detections(image, pred_R, pred_t, model_pts, cam_k)
        if show_pts:
            pred_pts = np.matmul(np.expand_dims(pts, 0) - np.expand_dims(pred_t, 1), pred_R)
            visualize_point_cloud(pred_pts.squeeze(0), model_pts)

    print("total time: ", time.time() - start_time)


if __name__ == "__main__":
    main()
    print("success")
