#!/bin/bash
# Train Qwen2-VL-2B with the verified B3 batch/data construction and EGA.
# This script intentionally mirrors run_train.sh. The only training-algorithm
# change is rectangular EGA over B3's global B x (6B) candidate matrix.

set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-/tmp/ume-recipes/huggingface}"
export WANDB_MODE=disabled

export B3_DEBUG="${B3_DEBUG:-1}"
export B3_DEBUG_INTERVAL="${B3_DEBUG_INTERVAL:-1}"
export B3_DDP_TRACE_STEPS="${B3_DDP_TRACE_STEPS:-0}"
export B3_NUMERIC_DEBUG_STEPS="${B3_NUMERIC_DEBUG_STEPS:-0}"
export B3_FINGERPRINT="${B3_FINGERPRINT:-0}"
export B3_FINGERPRINT_CALLS="${B3_FINGERPRINT_CALLS:-20}"
export B3_FINGERPRINT_RANKS="${B3_FINGERPRINT_RANKS:-0}"
export B3_EGA_RUN_SELF_TESTS="${B3_EGA_RUN_SELF_TESTS:-1}"

NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-2224}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-./models/Qwen2-VL-2B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-./exp/training/b3_ega_qwen2b_debug}"
LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "${LOG_DIR}"

# These values are kept identical to the successful Qwen2-VL B3 reproduction.
BATCH_SIZE_PER_DEVICE="${BATCH_SIZE_PER_DEVICE:-128}"
MAX_STEPS="${MAX_STEPS:-2000}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
SAVE_STEPS="${SAVE_STEPS:-100}"
RESUME_FROM="${RESUME_FROM:-none}"

GC_Q_CHUNK_SIZE="${GC_Q_CHUNK_SIZE:-128}"
GC_P_CHUNK_SIZE="${GC_P_CHUNK_SIZE:-128}"
GC_DYNAMIC_LIMIT="${GC_DYNAMIC_LIMIT:-64}"
CHUNK_SIZE="${CHUNK_SIZE:-32}"
DATASET_CONFIG="${DATASET_CONFIG:-configs/data_configs/mmeb_new/mmeb20_HNPS_bs16bi_30.130_30P.10.5_70.170_qwen2b.yaml}"
LORA_R="${LORA_R:-16}"
EGA_ALPHA="${EGA_ALPHA:-20.0}"

if [[ "${B3_EGA_RUN_SELF_TESTS}" == "1" ]]; then
    echo "Running B3+EGA rectangular formula and GradCache self-tests"
    python -m unittest tests.test_b3_ega -v
    if [[ "${NUM_GPUS}" -ge 2 ]]; then
        echo "Running B3+EGA two-rank distributed cache preflight"
        torchrun --standalone --nproc_per_node=2 "${SCRIPT_DIR}/tests/verify_b3_ega_distributed.py"
    fi
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_qwen2_b3_ega_${TIMESTAMP}.log"

echo "Starting Qwen2-VL-2B B3+EGA training"
echo "Model: ${MODEL_NAME_OR_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${NUM_GPUS}"
echo "Batch size per device: ${BATCH_SIZE_PER_DEVICE}"
echo "Max steps: ${MAX_STEPS}"
echo "Save steps: ${SAVE_STEPS}"
echo "LoRA rank: ${LORA_R}"
echo "EGA alpha: ${EGA_ALPHA}"
echo "B3 target layout: 1 positive + 5 explicit hard negatives"
echo "EGA negative scope: all global targets except each query's aligned positive"
echo "Log file: ${LOG_FILE}"

nohup torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    --max_restarts=0 \
    train.py \
    --output_dir "${OUTPUT_DIR}" \
    --lora \
    --lora_r "${LORA_R}" \
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
    --negative_strategy b3 \
    --grad_cache True \
    --per_device_train_batch_size "${BATCH_SIZE_PER_DEVICE}" \
    --gc_q_chunk_size "${GC_Q_CHUNK_SIZE}" \
    --gc_p_chunk_size "${GC_P_CHUNK_SIZE}" \
    --gc_dynamic_limit "${GC_DYNAMIC_LIMIT}" \
    --embedding_gradient_strategy b3_ega \
    --ega_alpha "${EGA_ALPHA}" \
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
    --resize_use_processor True \
    --interleave_batch_size 1 \
    --ddp_timeout 14400 \
    --ddp_broadcast_buffers False \
    --ddp_find_unused_parameters False \
    --ignore_data_skip true \
    --sdibn \
    --odibn \
    --chunk_size "${CHUNK_SIZE}" \
    > "${LOG_FILE}" 2>&1 &

PID=$!
echo "Training started in background with PID: ${PID}"
echo "To monitor training: tail -f ${LOG_FILE}"
echo "Training checkpoints: ${OUTPUT_DIR}"
