#!/bin/bash
# TriGS multi-view COLMAP + ROMA training pipeline.
#
# Usage:
#   bash scripts/run_colmap.sh <roma_input_dir> <colmap_data_dir> <output_dir> \
#                              <start_frame> <end_frame> <keyframe_step> [gpu_id]
set -euo pipefail

ROMA_INPUT_DIR=$1
DATA_DIR=$2
OUTPUT_DIR=$3
START_FRAME=$4
END_FRAME=$5
KEYFRAME_STEP=$6
GPU_ID=${7:-0}

mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=$GPU_ID uv run --no-sync trigs-train-colmap \
    --data-dir "$DATA_DIR" \
    --roma-input-dir "$ROMA_INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --start-frame "$START_FRAME" \
    --end-frame "$END_FRAME" \
    --keyframe-step "$KEYFRAME_STEP"
