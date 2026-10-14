#!/usr/bin/env bash
# Shared, machine-local configuration checks for the paper reproduction scripts.

set -euo pipefail

PAPER_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PAPER_REPO_ROOT}"
export PYTHONPATH="${PAPER_REPO_ROOT}:${PAPER_REPO_ROOT}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

paper_require_variables() {
    local variable_name
    for variable_name in "$@"; do
        if [ -z "${!variable_name:-}" ]; then
            echo "[ERROR] ${variable_name} is not set. Source configs/paper.env first." >&2
            exit 2
        fi
    done
}

paper_require_stage2() {
    paper_require_variables \
        TRAIN_RGB_DEPTH_H5 VAL_DEPTH_DIR \
        FATHOMNET_YOLO_PATH TCUSOD_CHECKPOINT TCUSOD_REPO_ROOT \
        DINOV2_REPO_PATH
}

paper_require_variables SD_PATH ELIC_PATH TRAIN_H5 VAL_IMAGE_DIR OUTPUT_ROOT

PAPER_RATE_LAMBDAS=(32 24 16 12 8 6 4)
STAGE1_STEPS=30001
STAGE2_STEPS=21001
PAPER_SEED="${PAPER_SEED:-2026}"

