#!/bin/bash
# =============================================================================
# ChangeMamba MambaBDA — xBD Dataset Baseline Training
# Model: ChangeMambaBDA (optical pre + optical post, 5-class damage)
# =============================================================================
set -e

WORKDIR="/root/ChangeMamba"
PYTHON="/root/venv/bin/python"

CFG="$WORKDIR/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml"
PRETRAINED="$WORKDIR/pretrained_weights/vmamba_tiny_e292.pth"

XBD_ROOT="$WORKDIR/data/xbd"
TRAIN_LIST="$XBD_ROOT/xBD_list/train_all.txt"
VAL_LIST="$XBD_ROOT/xBD_list/val_all.txt"

LOG_DIR="$WORKDIR/logs/xbd"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  MambaBDA — xBD Baseline Training"
echo "  Config     : $CFG"
echo "  Pretrained : $PRETRAINED"
echo "  Data root  : $XBD_ROOT"
echo "  Log        : $LOG_DIR/train.log"
echo "============================================================"

export PYTHONPATH="$WORKDIR:$PYTHONPATH"

$PYTHON $WORKDIR/changedetection/script/train_MambaBDA.py \
    --cfg "$CFG" \
    --pretrained_weight_path "$PRETRAINED" \
    --dataset xBD \
    --train_dataset_path "$XBD_ROOT/train" \
    --test_dataset_path  "$XBD_ROOT/test" \
    --train_data_list_path "$TRAIN_LIST" \
    --test_data_list_path  "$VAL_LIST" \
    --batch_size 8 \
    --crop_size 512 \
    --max_iters 80000 \
    --learning_rate 1e-4 \
    --weight_decay 5e-3 \
    --model_type MambaBDA_xBD \
    --model_param_path "$WORKDIR/saved_models" \
    2>&1 | tee "$LOG_DIR/train.log"
