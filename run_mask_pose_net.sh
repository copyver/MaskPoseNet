#!/bin/bash

CONFIG_PATH="configs/maskposenet_R50_crossfuse.yml"
DATASET_DIR="datasets/datasets_handle_4000t_800v"
SEG_NET_WEIGHTS="assets/seg_model_weights/seg_model_handle_res101_rgbd_cross_fuse_pretrained.h5"
POSE_NET_WEIGHTS="assets/pose_model_weights/pose_model_handle_pretrained.h5"
LOGS_DIR="assets"
COLOR="/home/yhlever/CLionProjects/RobotGrasp/results/color.png"
DEPTH="/home/yhlever/CLionProjects/RobotGrasp/results/depth.png"

export PYTHONPATH=$(pwd)/../../:$PYTHONPATH

python3 run_mask_pose_net.py \
    --config=$CONFIG_PATH \
    --dataset=$DATASET_DIR \
    --seg_net_weights=$SEG_NET_WEIGHTS \
    --pose_net_weights=$POSE_NET_WEIGHTS \
    --logs=$LOGS_DIR \
    --color=$COLOR \
    --depth=$DEPTH