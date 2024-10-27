#!/bin/bash

COMMAND="train"
CONFIG_PATH="../../configs/maskposenet_R50_crossfuse.yml"
DATASET_DIR="/home/yhlever/DeepLearning/6D_object_pose_estimation/Datasets/lm"
DATASET_TYPE="lm"
WEIGHTS="../../assets/seg_model_weights/lm20240920T1737/amsmc_lm_0025.h5"
LOGS_DIR="../../assets/seg_model_weights"


export PYTHONPATH=$(pwd)/../../:$PYTHONPATH

python3 run_seg_net.py $COMMAND \
    --config=$CONFIG_PATH \
    --dataset=$DATASET_DIR \
    --dataset_type=$DATASET_TYPE \
    --weights=$WEIGHTS \
    --logs=$LOGS_DIR \
    --limit=$LIMIT