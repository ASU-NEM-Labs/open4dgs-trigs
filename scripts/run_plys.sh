#!/bin/bash
# TriGS PLY-sequence training — dense per-frame Gaussian PLYs in / checkpoint out.
#
# Usage:
#   bash scripts/run_plys.sh <input_dir> <output_dir> [keyframe_stride] [n_steps] [gpu_id]
set -euo pipefail

INPUT_DIR=$1
OUTPUT_DIR=$2
KEYFRAME_STRIDE=${3:-3}
N_STEPS=${4:-30000}
GPU_ID=${5:-0}

mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=$GPU_ID uv run --no-sync trigs-train-plys \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --config.keyframe-stride "$KEYFRAME_STRIDE" \
    --config.n-steps "$N_STEPS"
