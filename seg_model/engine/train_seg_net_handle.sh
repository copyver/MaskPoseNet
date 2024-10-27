#!/bin/bash

COMMAND="train"
CONFIG_PATH="../../configs/maskposenet_R50_crossfuse.yml"
DATASET_DIR="../../datasets/datasets_handle_4000t_800v/"
DATASET_TYPE="handle"
WEIGHTS="../../assets/seg_model_weights/seg_model_handle_res101_rgbd_cross_fuse_pretrained.h5"
LOGS_DIR="../../assets/seg_model_weights"

export PYTHONPATH=$(pwd)/../../:$PYTHONPATH

python3 run_seg_net.py $COMMAND \
    --config=$CONFIG_PATH \
    --dataset=$DATASET_DIR \
    --dataset_type=$DATASET_TYPE \
    --weights=$WEIGHTS \
    --logs=$LOGS_DIR
