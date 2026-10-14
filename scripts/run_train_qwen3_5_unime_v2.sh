#!/bin/bash
# Qwen3.5-2B UniME-V2 soft-distribution training. The released UniME-V2
# 5-pair expansion, teacher scores and symmetric-KL objective are unchanged;
# model/runtime/adapter settings match run_train_qwen3_5_inbatch.sh.

set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-/tmp/ume-recipes/huggingface}"
export WANDB_MODE=disabled

# TileLang/FLA JIT-compiles Qwen3.5 Gated DeltaNet kernels on the first
# backward pass. The home directory has a small quota on Nebula notebooks, so
# keep all compiler caches on node-local storage. These caches contain only
# compiled kernels and do not change model inputs, gradients, or optimization.
LOCAL_KERNEL_CACHE_ROOT="${LOCAL_KERNEL_CACHE_ROOT:-/tmp/ume-recipes/qwen3_5_unime_v2_kernel_cache}"
export TILELANG_CACHE_DIR="${TILELANG_CACHE_DIR:-${LOCAL_KERNEL_CACHE_ROOT}/tilelang}"
export TILELANG_TMP_DIR="${TILELANG_TMP_DIR:-${TILELANG_CACHE_DIR}/tmp}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${LOCAL_KERNEL_CACHE_ROOT}/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${LOCAL_KERNEL_CACHE_ROOT}/torch_extensions}"
mkdir -p \
    "${TILELANG_CACHE_DIR}" \
    "${TILELANG_TMP_DIR}" \
    "${TRITON_CACHE_DIR}" \
    "${TORCH_EXTENSIONS_DIR}"

export B3_DEBUG="${B3_DEBUG:-1}"
export B3_DEBUG_INTERVAL="${B3_DEBUG_INTERVAL:-1}"
export B3_DDP_TRACE_STEPS="${B3_DDP_TRACE_STEPS:-0}"
export B3_NUMERIC_DEBUG_STEPS="${B3_NUMERIC_DEBUG_STEPS:-0}"
export B3_FINGERPRINT="${B3_FINGERPRINT:-1}"
export B3_FINGERPRINT_CALLS="${B3_FINGERPRINT_CALLS:-2}"
export B3_FINGERPRINT_RANKS="${B3_FINGERPRINT_RANKS:-0}"
# Match the verified Qwen3.5 runs: FlashAttention-2 remains mandatory for
# full-attention layers; Gated DeltaNet may use Transformers' correct fallback.
export QWEN3_5_REQUIRE_FAST_KERNELS="${QWEN3_5_REQUIRE_FAST_KERNELS:-0}"

NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-2230}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3.5-2B}"
UNIME_V2_DATA_PATH="${UNIME_V2_DATA_PATH:-./data/UniME-V2-Training-Datasets/train_data_Qwen25VL_7B_scores.json}"
UNIME_V2_IMAGE_DIR="${UNIME_V2_IMAGE_DIR:-./data/MMEB_V1}"
DATASET_CONFIG="${DATASET_CONFIG:-configs/data_configs/unime_v2.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-./exp/training/unime_v2_qwen3_5_2b}"
LOG_DIR="${LOG_DIR:-./logs}"

# Match the established Qwen3.5 ordinary in-batch optimization profile.
# Batch size counts original UniME rows; each row expands to five inputs on
# each encoder side and yields one positive plus eight hard negatives.
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

LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj,down_proj}"

SCRIPT_DIR=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
cd "${SCRIPT_DIR}"

if [[ ! -f "${MODEL_NAME_OR_PATH}/config.json" || ! -f "${MODEL_NAME_OR_PATH}/tokenizer.json" ]]; then
    echo "Local Qwen3.5 backbone/processor is incomplete: ${MODEL_NAME_OR_PATH}" >&2
    exit 1
fi
if [[ ! -f "${UNIME_V2_DATA_PATH}" ]]; then
    echo "UniME-V2 teacher-score JSON is missing: ${UNIME_V2_DATA_PATH}" >&2
    exit 1
fi
if [[ ! -f "${DATASET_CONFIG}" ]]; then
    echo "UniME-V2 dataset config is missing: ${DATASET_CONFIG}" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_FILE:-${LOG_DIR}/train_qwen3_5_unime_v2_${TIMESTAMP}.log}"

echo "Starting Qwen3.5-2B UniME-V2 training"
echo "Model: ${MODEL_NAME_OR_PATH}"
echo "Teacher-score data: ${UNIME_V2_DATA_PATH}"
echo "Image root: ${UNIME_V2_IMAGE_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${NUM_GPUS}; original rows/device: ${BATCH_SIZE_PER_DEVICE}"
echo "Expanded encoder inputs/device/side: $((BATCH_SIZE_PER_DEVICE * 5))"
echo "Candidates/original query: 9 (one positive + eight hard negatives)"
echo "LoRA/DoRA: r=${LORA_R}, alpha=${LORA_ALPHA}, dropout=${LORA_DROPOUT}"
echo "GradCache chunks: query=${GC_Q_CHUNK_SIZE}, target=${GC_P_CHUNK_SIZE}"
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
