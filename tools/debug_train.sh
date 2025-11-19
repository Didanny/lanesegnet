#!/usr/bin/env bash
set -x

timestamp=`date +"%Y-%m-%d_%H-%M-%S"`

WORK_DIR=work_dirs/debug_lanesegnet
CONFIG=projects/configs/lanesegnet_r50_8x1_24e_olv2_subset_A.py

# Create the work directory if it doesn't exist
mkdir -p ${WORK_DIR}

GPUS=$1
PORT=${PORT:-28510}

python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    tools/train.py $CONFIG --launcher pytorch --work-dir ${WORK_DIR} ${@:2} \
    2>&1 | tee ${WORK_DIR}/train.${timestamp}.log
