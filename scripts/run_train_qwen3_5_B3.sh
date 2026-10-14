#!/bin/bash
# Train Transformers-5.9 native Qwen3.5-2B with the reproduced B3 strategy.
# This is intentionally separate from run_train.sh, which remains the verified
# Qwen2-VL/B3 entry point.

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
# Qwen3.5's 18 Gated DeltaNet layers are prohibitively slow with the official
# PyTorch fallback. Fail early instead of looking like a distributed hang.
export QWEN3_5_REQUIRE_FAST_KERNELS="${QWEN3_5_REQUIRE_FAST_KERNELS:-1}"

NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-2216}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3.5-2B}"
OUTPUT_DIR="${OUTPUT_DIR:-./exp/training/b3_qwen3_5_2b_lora16}"
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
CHUNK_SIZE="${CHUNK_SIZE:-32}"

RESIZE_MIN_PIXELS="${RESIZE_MIN_PIXELS:-4096}"
configs/data_configs/mmeb_new/mmeb20_HNPS_bs16bi_30.130_30P.10.5_70.170_qwen2b.yaml
RESIZE_MAX_PIXELS="${RESIZE_MAX_PIXELS:-2257920}"
DATASET_CONFIG="${DATASET_CONFIG:-configs/data_configs/mmeb_new/mmeb20_HNPS_bs16bi_30.130_30P.10.5_70.170_qwen2b.yaml}"

LORA_R=16
LORA_ALPHA=64
# The successful Qwen2 B3 run adapts q/k/v/o plus down_proj. Qwen3.5's
# Gated DeltaNet replaces q/k/v/o with these in/out and gating projections;
# include all of them while keeping the MLP scope at down_proj.
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj,down_proj}"

mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_qwen3_5_b3_lora16_${TIMESTAMP}.log"

echo "Starting Qwen3.5-2B B3 training"
echo "Model: ${MODEL_NAME_OR_PATH}"
echo "Data: ${DATASET_CONFIG}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${NUM_GPUS}"
echo "Batch size per device: ${BATCH_SIZE_PER_DEVICE}"
echo "LoRA: r=${LORA_R}, alpha=${LORA_ALPHA}"
echo "Log file: ${LOG_FILE}"

nohup torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    --max_restarts=0 \
    train.py \
    --output_dir "${OUTPUT_DIR}" \
    --lora \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_target_modules "${LORA_TARGET_MODULES}" \
    --model_name "${MODEL_NAME_OR_PATH}" \
    --bf16 \
    --pooling eos \
    --normalize True \
    --temperature 0.02 \
    --dataloader_num_workers 0 \
    --dataset_config "${DATASET_CONFIG}" \
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
    --resize_use_processor True \
    --resize_min_pixels "${RESIZE_MIN_PIXELS}" \
    --resize_max_pixels "${RESIZE_MAX_PIXELS}" \
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
echo "Monitor with: tail -f ${LOG_FILE}"
