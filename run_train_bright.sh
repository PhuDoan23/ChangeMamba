#!/bin/bash
# =============================================================================
# ChangeMamba MambaBDA — BRIGHT Dataset Baseline Training
# Model: ChangeMambaMMBDA (multimodal, optical pre + SAR post)
# =============================================================================
set -e

WORKDIR="/root/ChangeMamba"
PYTHON="/root/venv/bin/python"

CFG="$WORKDIR/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml"
PRETRAINED="$WORKDIR/pretrained_weights/vmamba_tiny_e292.pth"

BRIGHT_ROOT="$WORKDIR/data/bright"
LIST_DIR="$BRIGHT_ROOT/changemamba_lists"
TRAIN_LIST="$LIST_DIR/train_list.txt"
VAL_LIST="$LIST_DIR/val_list.txt"
TEST_LIST="$LIST_DIR/test_list.txt"

LOG_DIR="$WORKDIR/logs/bright"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  MambaBDA — BRIGHT Baseline Training"
echo "  Config     : $CFG"
echo "  Pretrained : $PRETRAINED"
echo "  Data root  : $BRIGHT_ROOT"
echo "  Log        : $LOG_DIR/train.log"
echo "============================================================"

export PYTHONPATH="$WORKDIR:$PYTHONPATH"

$PYTHON $WORKDIR/changedetection/script/train_MambaBDA_bright.py \
    --cfg "$CFG" \
    --pretrained_weight_path "$PRETRAINED" \
    --dataset BRIGHT \
    --train_dataset_path "$BRIGHT_ROOT" \
    --val_dataset_path   "$BRIGHT_ROOT" \
    --test_dataset_path  "$BRIGHT_ROOT" \
    --train_data_list_path "$TRAIN_LIST" \
    --val_data_list_path   "$VAL_LIST" \
    --test_data_list_path  "$TEST_LIST" \
    --train_batch_size 8 \
    --crop_size 512 \
    --max_iters 80000 \
    --learning_rate 1e-4 \
    --weight_decay 5e-3 \
    --num_workers 4 \
    --model_type MambaBDA_BRIGHT \
    --model_param_path "$WORKDIR/saved_models" \
    2>&1 | tee "$LOG_DIR/train.log"
