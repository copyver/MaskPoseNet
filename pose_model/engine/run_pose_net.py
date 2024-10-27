import argparse
import logging
import os
import random
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
import trimesh

from pose_model.data.bop_test_data import read_bop_test_data
from pose_model.data.posenet_datagen import PoseNetDataset, visualize_point_cloud
from pose_model.posenet.posenet import PoseModel
from pose_model.posenet_config import PoseNetConfig
from pose_model.utils.visualize import draw_detections

SEED = 42


def parser_args():
    now = datetime.now()
    parser = argparse.ArgumentParser(
        description='Train posemodel')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS COCO or 'detect' on self image")
    parser.add_argument('--dataset',
                        required=True,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--weights',
                        required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    parser.add_argument('--logs',
                        required=True,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')

    args = parser.parse_args()
    return args


def setup_logging(args):
    log_dir = os.path.join(args.logs, "seg_model{}.log".format(datetime.now().strftime("%y%m%d%H%M")))
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_dir),
                            logging.StreamHandler()
                        ])
    return logging.getLogger('6DPoseModel_LOG')


class PoseConfig(PoseNetConfig):
    NAME = 'HandlePose'

    GPU_COUNT = 1

    IMAGES_PER_GPU = 1

    STEPS_PER_EPOCH = 4000

    VALIDATION_STEPS = 800


if __name__ == '__main__':
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    args = parser_args()
    logger = setup_logging(args)
    config = PoseConfig()

    # Create Model
    if args.command == "train":
        logger.info("=> creating model ...")
        model = PoseModel(mode='training', config=config, model_dir=args.logs, logger=logger)
    else:
        logger.info("=> loading model ...")
        model = PoseModel(mode="inference", config=config, model_dir=args.logs, logger=logger)
        if args.weights is not None:
            model.load_weights(args.weights)
            logger.info("=> loaded model weights '{}'".format(args.weights))

    # Train or evaluate
    if args.command == "train":
        dataset_train = PoseNetDataset()
        dataset_train.load_data(args.dataset, 'train')
        dataset_train.prepare()

        dataset_val = PoseNetDataset()
        dataset_val.load_data(args.dataset, 'val')
        dataset_val.prepare()

        logger.info("=> Training ...")
        tic = time.time()
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='all',
                    augmentation=None)
        consume_time = time.time() - tic
        logger.info('=> Successful Training Model, Total Time: {} min'.format(consume_time / 60))

    elif args.command == "inference":
        logger.info("=> starting inference ...")
        show_pts = True
        dataset_test = PoseNetDataset()
        dataset_test.load_data(args.dataset, 'test')
        dataset_test.prepare()
        model_cad_path = "/home/yhlever/DeepLearning/Datasets/model/handle_gt.obj"
        mesh = trimesh.load(model_cad_path, force='mesh')
        (pts, rgb, rgb_choose, model_pts,
         tem1_rgb, tem1_choose, tem1_pts,
         tem2_rgb, tem2_choose, tem2_pts,
         camera_k, image) \
            = read_bop_test_data(dataset_test, mesh, 42, 2)
        results = model.detect(pts, rgb, rgb_choose, model_pts,
                               tem1_rgb, tem1_choose, tem1_pts,
                               tem2_rgb, tem2_choose, tem2_pts)
        pred_R, pred_t, pred_score = results['pred_R'], results['pred_t'], results['pred_pose_score']
        draw_image_bbox = draw_detections(image, pred_R, pred_t, model_pts, camera_k)
        logger.info('=> Successful Inference')
        if show_pts:
            pred_pts = np.matmul(np.expand_dims(pts, 0) - np.expand_dims(pred_t, 1), pred_R)
            visualize_point_cloud(pred_pts.squeeze(0), model_pts)
        print("success")

    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
