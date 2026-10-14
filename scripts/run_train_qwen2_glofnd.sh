#!/bin/bash
# Qwen2-VL ordinary in-batch InfoNCE plus GloFND false-negative filtering.
# All backbone, data, batching, optimizer and GradCache settings match
# run_train_qwen2_inbatch.sh. Only the detached GloFND negative mask is new.

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

# Released GloFND defaults. Initialization is the agreed sparse-coverage
# adaptation: each anchor starts from its first global batch quantile.
export GLOFND_ALPHA="${GLOFND_ALPHA:-0.0005}"
export GLOFND_THRESHOLD_LR="${GLOFND_THRESHOLD_LR:-0.05}"
export GLOFND_BETA1="${GLOFND_BETA1:-0.9}"
export GLOFND_BETA2="${GLOFND_BETA2:-0.98}"
export GLOFND_EPS="${GLOFND_EPS:-1e-8}"
export GLOFND_DEBUG="${GLOFND_DEBUG:-1}"
export GLOFND_DEBUG_INTERVAL="${GLOFND_DEBUG_INTERVAL:-${B3_DEBUG_INTERVAL}}"

NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-2218}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-./models/Qwen2-VL-2B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-./exp/training/glofnd_qwen2_vl_2b}"
LOG_DIR="${LOG_DIR:-./logs}"

BATCH_SIZE_PER_DEVICE="${BATCH_SIZE_PER_DEVICE:-128}"
MAX_STEPS="${MAX_STEPS:-2000}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
SAVE_STEPS="${SAVE_STEPS:-100}"
RESUME_FROM="${RESUME_FROM:-none}"

GC_Q_CHUNK_SIZE="${GC_Q_CHUNK_SIZE:-128}"
GC_P_CHUNK_SIZE="${GC_P_CHUNK_SIZE:-128}"
GC_DYNAMIC_LIMIT="${GC_DYNAMIC_LIMIT:-64}"
INTERLEAVE_BATCH_SIZE="${INTERLEAVE_BATCH_SIZE:-64}"
SEED="${SEED:-42}"
RESIZE_MIN_PIXELS="${RESIZE_MIN_PIXELS:-4096}"
RESIZE_MAX_PIXELS="${RESIZE_MAX_PIXELS:-2257920}"
DATASET_CONFIG="${DATASET_CONFIG:-configs/data_configs/mmeb_new/mmeb20.yaml}"

LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-qkv_proj,o_proj,gate_up_proj,down_proj,k_proj,q_proj,out_proj,v_proj}"

mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_FILE:-${LOG_DIR}/train_qwen2_glofnd_${TIMESTAMP}.log}"
LOG_APPEND="${LOG_APPEND:-0}"
export GLOFND_LOG_PATH="${GLOFND_LOG_PATH:-${OUTPUT_DIR}/glofnd_steps.jsonl}"

mkdir -p "$(dirname "${LOG_FILE}")"
case "${LOG_APPEND}" in
    0)
        exec 3> "${LOG_FILE}"
        ;;
    1)
        exec 3>> "${LOG_FILE}"
        printf '\n[GLOFND_LOG_RESUME] time=%s resume_from=%s\n' \
            "$(date '+%Y-%m-%d %H:%M:%S')" "${RESUME_FROM}" >&3
        ;;
    *)
        echo "LOG_APPEND must be 0 or 1, got: ${LOG_APPEND}" >&2
        exit 2
        ;;
esac

echo "Starting Qwen2-VL-2B ordinary in-batch InfoNCE + GloFND"
echo "Model: ${MODEL_NAME_OR_PATH}"
echo "Data: ${DATASET_CONFIG}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${NUM_GPUS}; batch/device: ${BATCH_SIZE_PER_DEVICE}"
echo "GloFND: alpha=${GLOFND_ALPHA}, threshold_lr=${GLOFND_THRESHOLD_LR}, betas=(${GLOFND_BETA1},${GLOFND_BETA2})"
echo "Initialization: per-anchor lazy global-batch (1-alpha) quantile"
echo "Log file: ${LOG_FILE}"

nohup torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    --max_restarts=0 \
    train_glofnd.py \
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
    --negative_strategy vlm2vec_inbatch \
    --embedding_gradient_strategy info_nce \
    --pos_only True \
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
    --interleave_batch_size "${INTERLEAVE_BATCH_SIZE}" \
    --interleave_stopping_strategy all_exhausted \
    --ddp_timeout 14400 \
    --ddp_broadcast_buffers False \
    --ddp_find_unused_parameters False \
    --ignore_data_skip false \
    >&3 2>&1 &

PID=$!
exec 3>&-
echo "Training started in background with PID: ${PID}"
echo "Monitor with: tail -f ${LOG_FILE}"
