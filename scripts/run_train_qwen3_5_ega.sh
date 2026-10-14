#!/bin/bash
# Train Qwen3.5-2B with EGA over ordinary cross-device in-batch negatives.
# This intentionally matches run_train_qwen3_5_inbatch.sh in every model,
# prompt, data, batching and optimizer setting. Only the cached embedding
# gradient changes from ordinary InfoNCE autograd to EGA.

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
export QWEN3_5_REQUIRE_FAST_KERNELS="${QWEN3_5_REQUIRE_FAST_KERNELS:-1}"
export EGA_RUN_SELF_TESTS="${EGA_RUN_SELF_TESTS:-1}"

NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-2218}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3.5-2B}"
OUTPUT_DIR="${OUTPUT_DIR:-./exp/training/ega_qwen3_5_2b_lora16}"
LOG_DIR="${LOG_DIR:-./logs}"

BATCH_SIZE_PER_DEVICE="${BATCH_SIZE_PER_DEVICE:-128}"
MAX_STEPS="${MAX_STEPS:-2000}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
SAVE_STEPS="${SAVE_STEPS:-100}"
RESUME_FROM="${RESUME_FROM:-none}"
EGA_ALPHA="${EGA_ALPHA:-20.0}"

GC_Q_CHUNK_SIZE="${GC_Q_CHUNK_SIZE:-128}"
GC_P_CHUNK_SIZE="${GC_P_CHUNK_SIZE:-128}"
GC_DYNAMIC_LIMIT="${GC_DYNAMIC_LIMIT:-64}"
INTERLEAVE_BATCH_SIZE="${INTERLEAVE_BATCH_SIZE:-64}"

RESIZE_MIN_PIXELS="${RESIZE_MIN_PIXELS:-4096}"
RESIZE_MAX_PIXELS="${RESIZE_MAX_PIXELS:-2257920}"
DATASET_CONFIG="${DATASET_CONFIG:-configs/data_configs/mmeb_new/mmeb20.yaml}"

LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj,down_proj}"

if [[ "${EGA_RUN_SELF_TESTS}" == "1" ]]; then
    echo "Running EGA formula, GradCache and simulated-DDP self-tests"
    python -m unittest tests.test_ega -v
    if [[ "${NUM_GPUS:-8}" -ge 2 ]]; then
        echo "Running EGA two-rank distributed cache preflight"
        torchrun --standalone --nproc_per_node=2 "${SCRIPT_DIR}/tests/verify_ega_distributed.py"
    fi
fi


mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_qwen3_5_ega_lora16_${TIMESTAMP}.log"

echo "Starting Qwen3.5-2B EGA training"
echo "Model: ${MODEL_NAME_OR_PATH}"
echo "Data: ${DATASET_CONFIG}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${NUM_GPUS}"
echo "Batch size per device: ${BATCH_SIZE_PER_DEVICE}"
echo "LoRA: r=${LORA_R}, alpha=${LORA_ALPHA}, dropout=${LORA_DROPOUT}"
echo "EGA alpha: ${EGA_ALPHA}"
echo "VLM2Vec source block per device: ${INTERLEAVE_BATCH_SIZE}"
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
    --lora_dropout "${LORA_DROPOUT}" \
    --lora_target_modules "${LORA_TARGET_MODULES}" \
    --model_name "${MODEL_NAME_OR_PATH}" \
    --bf16 \
    --pooling eos \
    --normalize True \
    --temperature 0.02 \
    --dataloader_num_workers 0 \
    --dataset_config "${DATASET_CONFIG}" \
    --negative_strategy vlm2vec_inbatch \
    --embedding_gradient_strategy ega \
    --ega_alpha "${EGA_ALPHA}" \
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
    --resize_use_processor True \
    --resize_min_pixels "${RESIZE_MIN_PIXELS}" \
    --resize_max_pixels "${RESIZE_MAX_PIXELS}" \
    --interleave_batch_size "${INTERLEAVE_BATCH_SIZE}" \
    --ddp_timeout 14400 \
    --ddp_broadcast_buffers False \
    --ddp_find_unused_parameters False \
    --ignore_data_skip false \
    > "${LOG_FILE}" 2>&1 &

PID=$!
echo "Training started in background with PID: ${PID}"
echo "Monitor with: tail -f ${LOG_FILE}"
