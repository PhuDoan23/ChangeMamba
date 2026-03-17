#!/bin/bash
# =============================================================================
# ChangeMamba — Vast.ai Setup Script
# Runs MambaBDA baseline on BRIGHT and xBD datasets
# Usage: bash setup_vastai.sh
# =============================================================================
set -e

WORKDIR="/root/ChangeMamba"
PYTHON="/root/venv/bin/python"
PIP="/root/venv/bin/pip"

echo "============================================================"
echo "  ChangeMamba Baseline Setup"
echo "============================================================"

# ── 1. System packages ────────────────────────────────────────────────────── #
apt-get update -qq && apt-get install -y -qq git wget unzip libgl1 libglib2.0-0

# ── 2. Python venv ────────────────────────────────────────────────────────── #
python3 -m venv /root/venv
$PIP install --upgrade pip -q

# ── 3. PyTorch (CUDA 12.1) ────────────────────────────────────────────────── #
$PIP install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q

# ── 4. Clone repo ─────────────────────────────────────────────────────────── #
if [ ! -d "$WORKDIR" ]; then
    git clone https://github.com/ChenHongruixuan/ChangeMamba.git $WORKDIR
fi
cd $WORKDIR

# ── 5. Python dependencies ────────────────────────────────────────────────── #
$PIP install timm==0.4.12 yacs termcolor tensorboardX fvcore seaborn packaging chardet -q
$PIP install rasterio tifffile einops -q

# ── 6. Install selective scan CUDA kernel ─────────────────────────────────── #
echo "Building selective_scan kernel..."
cd $WORKDIR/kernels/selective_scan
$PIP install -e . -q
cd $WORKDIR

# ── 7. Make repo importable as `changedetection.*` ────────────────────────── #
# Scripts use `from changedetection.X import Y` with WORKDIR on PYTHONPATH.
# No extra steps needed — PYTHONPATH is set in each run script.

# ── 8. VMamba-Tiny pretrained weights ─────────────────────────────────────── #
mkdir -p $WORKDIR/pretrained_weights
if [ ! -f "$WORKDIR/pretrained_weights/vmamba_tiny_e292.pth" ]; then
    echo "Downloading VMamba-Tiny weights..."
    wget -q --show-progress -P $WORKDIR/pretrained_weights \
        https://zenodo.org/records/14037769/files/vmamba_tiny_e292.pth
fi

# ── 9. Dataset: BRIGHT ────────────────────────────────────────────────────── #
mkdir -p $WORKDIR/data/bright
if [ ! -d "$WORKDIR/data/bright/pre-event" ]; then
    echo ""
    echo "⚠  BRIGHT dataset not found."
    echo "   Upload your data to: $WORKDIR/data/bright/"
    echo "   Expected layout:"
    echo "     data/bright/pre-event/   *.tif"
    echo "     data/bright/post-event/  *.tif"
    echo "     data/bright/target/      *.tif"
    echo "     data/bright/splits/standard_ML/{train,val,test}_set.txt"
fi

# ── 10. Dataset: xBD ──────────────────────────────────────────────────────── #
mkdir -p $WORKDIR/data/xbd
if [ ! -d "$WORKDIR/data/xbd/train" ]; then
    echo ""
    echo "⚠  xBD dataset not found."
    echo "   Upload your data to: $WORKDIR/data/xbd/"
    echo "   Expected layout:"
    echo "     data/xbd/train/images/{pre,post}/  *.png"
    echo "     data/xbd/train/labels/             *.png"
    echo "     data/xbd/test/  (same structure)"
    echo "     data/xbd/xBD_list/{train,val}_all.txt"
fi

# ── 11. Generate BRIGHT split lists (ChangeMamba format) ──────────────────── #
BRIGHT_SPLIT=$WORKDIR/data/bright/splits/standard_ML
if [ -d "$BRIGHT_SPLIT" ]; then
    echo "Generating ChangeMamba-format BRIGHT split lists..."
    $PYTHON - <<'PYEOF'
import os
from pathlib import Path

split_dir = Path("/root/ChangeMamba/data/bright/splits/standard_ML")
out_dir   = Path("/root/ChangeMamba/data/bright/changemamba_lists")
out_dir.mkdir(parents=True, exist_ok=True)

for split in ["train", "val", "test"]:
    src = split_dir / f"{split}_set.txt"
    dst = out_dir   / f"{split}_list.txt"
    if src.exists():
        stems = [l.strip() for l in src.read_text().splitlines() if l.strip()]
        dst.write_text("\n".join(stems) + "\n")
        print(f"  {dst.name}: {len(stems)} samples")
PYEOF
fi

# ── 12. Saved models directory ────────────────────────────────────────────── #
mkdir -p $WORKDIR/saved_models

echo ""
echo "============================================================"
echo "  Setup complete. Run training with:"
echo "    bash run_train_bright.sh"
echo "    bash run_train_xbd.sh"
echo "============================================================"
