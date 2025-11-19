#!/usr/bin/env bash
set -x

GPUS=$1
WORK_DIR=$2
CHECKPOINT_NAME=$3
CONFIG=projects/configs/lanesegnet_r50_8x1_24e_olv2_subset_A.py

# Construct full checkpoint path
CHECKPOINT=${WORK_DIR}/${CHECKPOINT_NAME}.pth

PORT=${PORT:-28510}

python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    tools/test.py $CONFIG $CHECKPOINT --launcher pytorch \
    --out-dir ${WORK_DIR}/test --eval openlane_v2 ${@:4} \
    2>&1 | tee ${WORK_DIR}/test.${CHECKPOINT_NAME}.log
