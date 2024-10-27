#!/bin/bash

COMMAND="inference"
CONFIG_PATH="../../configs/maskposenet_R50_crossfuse.yml"
DATASET_TYPE="handle"
WEIGHTS="../../assets/seg_model_weights/seg_model_handle_res101_rgbd_cross_fuse_pretrained.h5"
LOGS_DIR="../../assets/seg_model_weights"
COLOR="/home/yhlever/CLionProjects/RobotGrasp/results/color.png"
DEPTH="/home/yhlever/CLionProjects/RobotGrasp/results/depth.png"

export PYTHONPATH=$(pwd)/../../:$PYTHONPATH

python3 run_seg_net.py $COMMAND \
    --config=$CONFIG_PATH \
    --dataset_type=$DATASET_TYPE \
    --weights=$WEIGHTS \
    --logs=$LOGS_DIR \
    --color=$COLOR \
    --depth=$DEPTH