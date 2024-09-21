import argparse
import logging
import os
import random
from datetime import datetime
import json
import cv2
import imgaug.augmenters as iaa
import numpy as np
import tensorflow as tf
import yaml
from addict import Dict

from seg_model.amsmc.amsmc import AMSMC as INS_SEG_MODEL
from seg_model.data.custom_coco_data_mapper import CustomCOCODataset
from seg_model.data.lm_data_mapper import LMDataset
from seg_model.mrcnn import visualize
from seg_model.utils import evaluate_coco
from seg_model.utils import show_cocoEval
from seg_model.config import LMConfig, HandleConfig

COCO_MODEL_PATH = "../../assets/seg_model_weights/mask_rcnn_coco.h5"
SEED = 42


def parser_args():
    parser = argparse.ArgumentParser(description='Train AMSMC on MS COCO.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS COCO or 'detect' on self image")
    parser.add_argument('--config',
                        required=True,
                        metavar="/path/to/net.yaml/",
                        help='YAML to set segmentation model config')
    parser.add_argument('--dataset',
                        required=False,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--dataset_type',
                        required=False,
                        default='coco',
                        choices=['handle', 'lm'],
                        metavar="<dataset type>",
                        help='Type of the dataset (default="coco")')
    parser.add_argument('--weights',
                        required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
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
    parser.add_argument('--limit',
                        required=False,
                        default=100,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=100)')
    args = parser.parse_args()
    return args


def load_config(file_path):
    """Load YAML configuration file."""
    try:
        with open(file_path, 'r') as config_file:
            config_dict = yaml.safe_load(config_file)
        return Dict(config_dict)
    except FileNotFoundError:
        logging.error("Configuration file not found.")
        raise
    except yaml.YAMLError as exc:
        logging.error(f"Error in configuration file: {exc}")
        raise


def _init_seg_model(args):
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    config = load_config(args.config)

    log_dir = os.path.join(args.logs, "seg_model{}.log".format(datetime.now().strftime("%y%m%d%H%M")))

    if args.command == 'train':
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            handlers=[
                                logging.FileHandler(log_dir),
                                logging.StreamHandler()
                            ])
        logger = logging.getLogger("InstanceSegModel_LOG")
        logger.info("Command arguments:")
        logger.info(args)
        return config, logger
    else:
        return config


if __name__ == '__main__':
    args = parser_args()
    if args.command == 'train':
        com_cfg, logger = _init_seg_model(args)
    else:
        com_cfg = _init_seg_model(args)

    # Select the appropriate configuration class based on the dataset type
    if args.dataset_type == 'handle':
        seg_cfg = HandleConfig()
    elif args.dataset_type == 'lm':
        seg_cfg = LMConfig()
    else:
        raise ValueError(f"Unsupported dataset type: {args.dataset_type}")

    if args.command != 'train':
        class InferenceConfig(seg_cfg.__class__):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0.9
            DETECTION_NMS_THRESHOLD = 0.4

        seg_cfg = InferenceConfig()
    else:
        seg_cfg.log_config(logger)

    # Create model
    if args.command == "train":
        model = INS_SEG_MODEL(mode="training", seg_cfg=seg_cfg, com_seg=com_cfg.MODEL.SEG_MODEL,
                              model_dir=args.logs)
        total_params = model.count_params()
        logger.info(f'Total number of parameters: {total_params}')
    else:
        model = INS_SEG_MODEL(mode="inference", seg_cfg=seg_cfg, com_seg=com_cfg.MODEL.SEG_MODEL,
                              model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.weights.lower() == "last":
        model_path = model.find_last()
    else:
        model_path = args.weights

    # Load weights
    print("Loading weights ", model_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching number of classes
        model.load_weights(model_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    elif args.weights.lower() == "none":
        pass
    else:
        model.load_weights(model_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        # Choose the appropriate dataset class based on args.dataset_type
        if args.dataset_type == 'handle':
            dataset_train = CustomCOCODataset(args.dataset, "train", use_depth=True)
            dataset_val = CustomCOCODataset(args.dataset, "val", use_depth=True)
        elif args.dataset_type == 'lm':
            dataset_train = LMDataset(args.dataset, "train", use_depth=True)
            dataset_val = LMDataset(args.dataset, "val", use_depth=True)
        else:
            raise ValueError(f"Unsupported dataset type: {args.dataset_type}")

        # Augmentation
        augmentation = iaa.Sequential([
            iaa.Sometimes(0.5, iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))),
            iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0.0, 3.0))),
            iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255))),
            iaa.Sometimes(0.5, iaa.Multiply((0.8, 1.2))),
            iaa.Sometimes(0.5, iaa.ContrastNormalization((0.75, 1.5)))
        ])

        model.train(dataset_train, dataset_val,
                    learning_rate=seg_cfg.LEARNING_RATE,
                    epochs=25,
                    layers='all',
                    augmentation=augmentation)

    elif args.command == "evaluate":
        print("Running evaluation on {} images.".format(args.limit))
        if args.dataset_type == 'handle':
            dataset_test = CustomCOCODataset(args.dataset, "test", use_depth=True)
        elif args.dataset_type == 'lm':
            dataset_test = LMDataset(args.dataset, "test", use_depth=True)
        else:
            raise ValueError(f"Unsupported dataset type: {args.dataset_type}")

        coco = dataset_test.coco
        stats = evaluate_coco(model, dataset_test, coco, "bbox", limit=int(args.limit), use_depth=True)
        show_cocoEval(stats, save_path="../../assets/evaluate_chart/seg_model_evaluate_{}_{}.png".
                      format(args.dataset_type, datetime.now().strftime("%y%m%d%H%M")))

    elif args.command == "inference":
        if args.dataset_type == 'handle':
            class_names = ['bg', 'handle']
        elif args.dataset_type == 'lm':
            # For LineMOD, class names are 'object_1', 'object_2', ..., etc.
            class_names = ['BG'] + ["object_{}".format(i) for i in range(1, 16)]
        else:
            raise ValueError(f"Unsupported dataset type: {args.dataset_type}")

        if args.dataset_type == 'handle':
            # syn test
            # cam_path = os.path.join(args.dataset, 'syn_test/scene_camera.json')
            # with open(cam_path, 'r') as f:
            #     data = json.load(f)
            # cam_K = np.array(data["455"]["cam_K"]).reshape((3, 3))

            # real test
            cam_K = np.array([[1062.67, 0, 646.166],
                              [0, 1062.67, 474.238],
                              [0, 0, 1]])
            depth_scale = 1  # syn 5

        elif args.dataset_type == 'lm':
            cam_K = np.array([[572.4114, 0, 325.2611],
                              [0, 573.57043, 242.04899],
                              [0, 0, 1]])
            depth_scale = 1
        else:
            raise ValueError(f"Unsupported dataset type: {args.dataset_type}")

        # Load the image and depth
        if args.color is not None and args.depth is not None:
            color = cv2.imread(args.color)
            depth = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)
            depth = depth * depth_scale / 1000
        else:
            print("Please provide both --color and --depth arguments for inference.")
            exit(1)

        results = model.detect([color], [depth], cam_K, verbose=1)
        r = results[0]
        if r['scores'].size == 0:
            print("No masks...")
        else:
            print(f"Number: {len(r['scores'])}")
            print(f"Scores：{r['scores']}")
            visualize.display_instances(color, r['rois'], r['masks'], r['class_ids'],
                                        class_names, r['scores'], title="Predictions", show_mask=False)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate' or detect".format(args.command))
