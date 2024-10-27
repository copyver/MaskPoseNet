import os

import numpy as np
import tensorflow as tf

from pose_model.data.bop_test_data import read_real_world_data
from pose_model.posenet.posenet import PoseModel
from pose_model.posenet_config import PoseNetConfig
from pose_model.utils.visualize import draw_detections
from seg_model.amsmc.amsmc import AMSMC
from seg_model.config import Config
from seg_model.engine.run_seg_net import load_config
from seg_model.mrcnn.visualize import display_instances

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = "assets"
CONFIG_PATH = "configs/maskposenet_R50_crossfuse.yml"
SEG_NET_WEIGHTS = "assets/seg_model_weights/seg_model_handle_res101_rgbd_cross_fuse_pretrained.h5"
POSE_NET_WEIGHTS = "assets/pose_model_weights/pose_model_handle_pretrained.h5"


class CustomInstanceConfig(Config):
    NAME = "handle"
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
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self, amsmc_weights_path, pose_model_weights_path, com_cfg, logs_dir):
        if not hasattr(self, 'initialized'):
            self.seg_net_weights = amsmc_weights_path
            self.pose_net_weights = pose_model_weights_path
            self.com_cfg = com_cfg
            self.model_dir = logs_dir

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
            self.initialized = True

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


def process_images(color_image, depth_image):
    com_cfg = load_config(CONFIG_PATH)
    model_loader = ModelLoader(SEG_NET_WEIGHTS, POSE_NET_WEIGHTS, com_cfg, LOGS_DIR)

    cam_k = np.array([[1062.67, 0, 646.166], [0, 1062.67, 474.238], [0, 0, 1]])
    depth_image = depth_image / 1000
    class_names = ['bj', 'handle']

    r = model_loader.detect_instance(color_image, depth_image, cam_k)

    if r['scores'].size == 0:
        print("No masks...")
        return None, None, None
    else:
        print(f"Number: {len(r['scores'])}")
        print(f"Scores：{r['scores']}")
        display_instances(color_image, r['rois'], r['masks'], r['class_ids'],
                          class_names, r['scores'], title="Predictions",
                          show_mask=False)

        best_mask = r['masks'][:, :, 0]
        cat_id = r['class_ids'][0]

        (pts, rgb, rgb_choose, model_pts,
         tem1_rgb, tem1_choose, tem1_pts,
         tem2_rgb, tem2_choose, tem2_pts) = read_real_world_data(color_image, depth_image, best_mask, cat_id, cam_k)
        cam_k = np.expand_dims(cam_k, axis=0)
        results = model_loader.detect_pose(pts, rgb, rgb_choose, model_pts, tem1_rgb, tem1_choose, tem1_pts, tem2_rgb,
                                           tem2_choose, tem2_pts)
        pred_R, pred_t, pred_score = results['pred_R'], results['pred_t'], results['pred_pose_score']
        print("pred_R: ", pred_R)
        print("pred_t: ", pred_t)
        print("pred_score: ", pred_score)
        draw_detections(color_image, pred_R, pred_t, model_pts, cam_k)

    return pred_R, pred_t, pred_score
