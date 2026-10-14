#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/scripts/paper_common.sh"
paper_require_stage2

for lambda_rate in "${PAPER_RATE_LAMBDAS[@]}"; do
    accelerate launch --num_processes=1 src/finetune.py \
        --sd_path="${SD_PATH}" \
        --elic_path="${ELIC_PATH}" \
        --codec_path="${OUTPUT_ROOT}/hmcg_uic/stage1/checkpoints/hmcg_uic_stage1_30001.pkl" \
        --train_dataset="${TRAIN_RGB_DEPTH_H5}" \
        --test_dataset="${VAL_IMAGE_DIR}" \
        --test_depth_dir="${VAL_DEPTH_DIR}" \
        --output_dir="${OUTPUT_ROOT}/hmcg_uic/stage2/lambda_${lambda_rate}" \
        --seed "${PAPER_SEED}" \
        --max_train_steps "${STAGE2_STEPS}" \
        --checkpointing_steps 1000 \
        --eval_freq 500 \
        --lambda_rate "${lambda_rate}" \
        --learning_rate 5e-5 \
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
        --semantic_residual_scale 0.05 \
        --use_machine_task_loss \
        --lambda_machine_task 1.0 \
        --machine_task_preset underwater_yolo_tcusod
done

