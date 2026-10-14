#!/bin/bash
# UniME-V2 soft-distribution training on the verified Qwen2-VL reproduction.
# Optimizer, adapter, resolution, global batch and GradCache chunk settings
# intentionally match run_train_qwen2_inbatch.sh; only data/loss are UniME-V2.

set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-/tmp/ume-recipes/huggingface}"
export WANDB_MODE=disabled

export B3_DEBUG="${B3_DEBUG:-1}"
export B3_DEBUG_INTERVAL="${B3_DEBUG_INTERVAL:-1}"
export B3_DDP_TRACE_STEPS="${B3_DDP_TRACE_STEPS:-0}"
export B3_NUMERIC_DEBUG_STEPS="${B3_NUMERIC_DEBUG_STEPS:-0}"
export B3_FINGERPRINT="${B3_FINGERPRINT:-1}"
export B3_FINGERPRINT_CALLS="${B3_FINGERPRINT_CALLS:-2}"
export B3_FINGERPRINT_RANKS="${B3_FINGERPRINT_RANKS:-0}"

NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-2218}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-./models/Qwen2-VL-2B-Instruct}"
UNIME_V2_DATA_PATH="${UNIME_V2_DATA_PATH:-./data/UniME-V2-Training-Datasets/train_data_Qwen25VL_7B_scores.json}"
UNIME_V2_IMAGE_DIR="${UNIME_V2_IMAGE_DIR:-./data/MMEB_V1}"
DATASET_CONFIG="${DATASET_CONFIG:-configs/data_configs/unime_v2.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-./exp/training/unime_v2_qwen2_vl_2b}"
LOG_DIR="${LOG_DIR:-./logs}"

# Identical to the established Qwen2-VL in-batch control. Batch size counts
# original UniME rows; each row expands to five inputs on each encoder side.
BATCH_SIZE_PER_DEVICE="${BATCH_SIZE_PER_DEVICE:-128}"
MAX_STEPS="${MAX_STEPS:-2000}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
SAVE_STEPS="${SAVE_STEPS:-100}"
RESUME_FROM="${RESUME_FROM:-none}"

GC_Q_CHUNK_SIZE="${GC_Q_CHUNK_SIZE:-128}"
GC_P_CHUNK_SIZE="${GC_P_CHUNK_SIZE:-128}"
GC_DYNAMIC_LIMIT="${GC_DYNAMIC_LIMIT:-64}"
SEED="${SEED:-42}"
RESIZE_MIN_PIXELS="${RESIZE_MIN_PIXELS:-4096}"
RESIZE_MAX_PIXELS="${RESIZE_MAX_PIXELS:-2257920}"

LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-qkv_proj,o_proj,gate_up_proj,down_proj,k_proj,q_proj,out_proj,v_proj}"

mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_FILE:-${LOG_DIR}/train_qwen2_unime_v2_${TIMESTAMP}.log}"

echo "Starting Qwen2-VL-2B UniME-V2 training"
echo "Model: ${MODEL_NAME_OR_PATH}"
echo "Teacher-score data: ${UNIME_V2_DATA_PATH}"
echo "Image root: ${UNIME_V2_IMAGE_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${NUM_GPUS}; original rows/device: ${BATCH_SIZE_PER_DEVICE}"
echo "Expanded encoder inputs/device/side: $((BATCH_SIZE_PER_DEVICE * 5))"
echo "Candidates/original query: 9 (one positive + eight hard negatives)"
echo "LoRA/DoRA: r=${LORA_R}, alpha=${LORA_ALPHA}, dropout=${LORA_DROPOUT}"
echo "Log file: ${LOG_FILE}"

nohup torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    --max_restarts=0 \
    train_unime_v2.py \
    --output_dir "${OUTPUT_DIR}" \
    --lora \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --lora_target_modules "${LORA_TARGET_MODULES}" \
    --model_name "${MODEL_NAME_OR_PATH}" \
    --bf16 \
    --pooling eos \
    --normalize True \
    --qwen2_model_impl b3 \
    --qwen2_b3_attention_impl flash_attention_2 \
    --qwen2_b3_legacy_position_ids False \
    --temperature 0.02 \
    --dataloader_num_workers 0 \
    --dataset_config "${DATASET_CONFIG}" \
    --unime_v2_data_path "${UNIME_V2_DATA_PATH}" \
    --unime_v2_image_dir "${UNIME_V2_IMAGE_DIR}" \
    --unime_v2_num_hard_negatives 8 \
    --negative_strategy unime_v2 \
    --embedding_gradient_strategy info_nce \
    --pos_only False \
    --grad_cache True \
    --per_device_train_batch_size "${BATCH_SIZE_PER_DEVICE}" \
    --gc_q_chunk_size "${GC_Q_CHUNK_SIZE}" \
    --gc_p_chunk_size "${GC_P_CHUNK_SIZE}" \
    --gc_dynamic_limit "${GC_DYNAMIC_LIMIT}" \
    --optim adamw_torch \
    --lr_scheduler_type linear \
    --learning_rate "${LEARNING_RATE}" \
    --max_steps "${MAX_STEPS}" \
    --warmup_steps "${WARMUP_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --logging_steps 1 \
    --save_safetensors False \
    --remove_unused_columns False \
    --resume_from "${RESUME_FROM}" \
    --seed "${SEED}" \
    --gradient_accumulation_steps 1 \
    --resize_use_processor True \
    --resize_min_pixels "${RESIZE_MIN_PIXELS}" \
    --resize_max_pixels "${RESIZE_MAX_PIXELS}" \
    --interleave_batch_size 0 \
    --interleave_stopping_strategy all_exhausted \
    --ddp_timeout 14400 \
    --ddp_broadcast_buffers False \
    --ddp_find_unused_parameters False \
    --ignore_data_skip false \
    > "${LOG_FILE}" 2>&1 &

PID=$!
echo "Training started in background with PID: ${PID}"
echo "Monitor with: tail -f ${LOG_FILE}"

