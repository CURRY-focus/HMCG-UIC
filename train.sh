#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/scripts/paper_common.sh"

accelerate launch --num_processes=1 src/train.py \
    --sd_path="${SD_PATH}" \
    --elic_path="${ELIC_PATH}" \
    --train_dataset="${TRAIN_H5}" \
    --test_dataset="${VAL_IMAGE_DIR}" \
    --output_dir="${OUTPUT_ROOT}/hmcg_uic/stage1" \
    --seed "${PAPER_SEED}" \
    --max_train_steps "${STAGE1_STEPS}" \
    --checkpointing_steps 1000 \
    --eval_freq 500 \
    --lambda_rate 0.5 \
    --lambda_structure 1.0 \
    --lambda_color 0.5 \
    --lambda_color_chroma 1.0 \
    --lambda_color_mean 0.2 \
    --lambda_color_vq 0.1 \
    --color_codebook_size 256 \
    --color_code_dim 64 \
    --color_global_tokens 4 \
    --color_residual_scale 0.05 \
    --lambda_semantic 1.0 \
    --lambda_semantic_attention 1.0 \
    --lambda_semantic_boundary 0.2 \
    --lambda_semantic_vq 0.1 \
    --semantic_codebook_size 256 \
    --semantic_code_dim 64 \
    --semantic_global_tokens 4 \
    --semantic_residual_scale 0.05

