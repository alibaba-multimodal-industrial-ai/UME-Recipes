#!/bin/bash
# Qwen3.5-2B HOBIT-C: ordinary global InfoNCE with source-aware HOBIT batches.
# The model, processor, LoRA, data and optimization profile match the verified
# Qwen3.5 ordinary in-batch control; only global-batch construction is HOBIT-C.

set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export HF_HOME="${HF_HOME:-/tmp/ume-recipes/huggingface}"
export B3_DEBUG="${B3_DEBUG:-1}"
export B3_DEBUG_INTERVAL="${B3_DEBUG_INTERVAL:-1}"
export B3_DDP_TRACE_STEPS="${B3_DDP_TRACE_STEPS:-0}"
export B3_NUMERIC_DEBUG_STEPS="${B3_NUMERIC_DEBUG_STEPS:-0}"
export B3_FINGERPRINT="${B3_FINGERPRINT:-1}"
export B3_FINGERPRINT_CALLS="${B3_FINGERPRINT_CALLS:-2}"
export B3_FINGERPRINT_RANKS="${B3_FINGERPRINT_RANKS:-0}"
# Match the verified Qwen3.5 runs in this environment: full-attention layers
# still use FlashAttention-2, while Gated DeltaNet may use its correct PyTorch
# fallback when causal-conv1d is unavailable.
export QWEN3_5_REQUIRE_FAST_KERNELS="${QWEN3_5_REQUIRE_FAST_KERNELS:-0}"

NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT_CACHE="${MASTER_PORT_CACHE:-2378}"
MASTER_PORT_TRAIN="${MASTER_PORT_TRAIN:-2388}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3.5-2B}"
OUTPUT_DIR="${OUTPUT_DIR:-./exp/training/inbatch_qwen3_5_2b_hobit_c_lora16}"
LOG_DIR="${LOG_DIR:-./logs}"
DATASET_CONFIG="${DATASET_CONFIG:-configs/data_configs/mmeb_new/mmeb20.yaml}"

BATCH_SIZE_PER_DEVICE="${BATCH_SIZE_PER_DEVICE:-128}"
MAX_STEPS="${MAX_STEPS:-2000}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SEED="${SEED:-42}"
RESUME_FROM="${RESUME_FROM:-none}"
GC_Q_CHUNK_SIZE="${GC_Q_CHUNK_SIZE:-128}"
GC_P_CHUNK_SIZE="${GC_P_CHUNK_SIZE:-128}"
GC_DYNAMIC_LIMIT="${GC_DYNAMIC_LIMIT:-64}"
INTERLEAVE_BATCH_SIZE="${INTERLEAVE_BATCH_SIZE:-64}"
RESIZE_MIN_PIXELS="${RESIZE_MIN_PIXELS:-4096}"
RESIZE_MAX_PIXELS="${RESIZE_MAX_PIXELS:-2257920}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj,down_proj}"

# Preserve HOBIT's s/b=8/128 ratio at global batch 1024: each global batch is
# two independently constructed 512-case source blocks.
export HOBIT_SOURCE_BLOCK_SIZE="${HOBIT_SOURCE_BLOCK_SIZE:-512}"
export HOBIT_SEED_SIZE="${HOBIT_SEED_SIZE:-32}"
export HOBIT_CANDIDATE_K="${HOBIT_CANDIDATE_K:-128}"
export HOBIT_ALPHA="${HOBIT_ALPHA:-1.0}"
export HOBIT_TEMPERATURE="${HOBIT_TEMPERATURE:-0.02}"
export HOBIT_ENCODE_BATCH_SIZE="${HOBIT_ENCODE_BATCH_SIZE:-16}"
# Resolution and backbone are part of the default cache name so incompatible
# preprocessing cannot be silently reused.
export HOBIT_CACHE_DIR="${HOBIT_CACHE_DIR:-./exp/cache/qwen3_5_hobit_initial_student_r${RESIZE_MIN_PIXELS}_m${RESIZE_MAX_PIXELS}}"
export HOBIT_CACHE_STAGE_DIR="${HOBIT_CACHE_STAGE_DIR:-/tmp/qwen3_5_hobit_cache_build_$$}"
export HOBIT_RUNTIME_CACHE_DIR="${HOBIT_RUNTIME_CACHE_DIR:-/tmp/qwen3_5_hobit_runtime_$$}"
export HOBIT_DIAGNOSTICS_PATH="${HOBIT_DIAGNOSTICS_PATH:-${OUTPUT_DIR}/hobit_diagnostics.jsonl}"

SCRIPT_DIR=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
cd "${SCRIPT_DIR}"

if [[ "${NUM_GPUS}" != "8" || "${BATCH_SIZE_PER_DEVICE}" != "128" ]]; then
    echo "HOBIT comparison requires 8 GPUs and batch size 128 per GPU." >&2
    exit 1
fi
if [[ "${INTERLEAVE_BATCH_SIZE}" != "64" || "${HOBIT_SOURCE_BLOCK_SIZE}" != "512" ]]; then
    echo "HOBIT comparison requires two 512-case source blocks per global batch." >&2
    exit 1
fi
if [[ "${HOBIT_SEED_SIZE}" != "32" || "${HOBIT_CANDIDATE_K}" != "128" ]]; then
    echo "This controlled HOBIT run requires seed_size=32 and candidate_k=128." >&2
    exit 1
fi
if [[ ( "${HOBIT_ALPHA}" != "1.0" && "${HOBIT_ALPHA}" != "1" ) || "${HOBIT_TEMPERATURE}" != "0.02" ]]; then
    echo "This controlled HOBIT run requires alpha=1.0 and tau_h=0.02." >&2
    exit 1
fi
if [[ "${RESUME_FROM}" != "none" ]]; then
    echo "HOBIT-C resume is disabled because its rolling embedding cache is not checkpointed." >&2
    exit 1
fi
if [[ ! -f "${MODEL_NAME_OR_PATH}/config.json" || ! -f "${MODEL_NAME_OR_PATH}/tokenizer.json" ]]; then
    echo "Local Qwen3.5 backbone/processor is incomplete: ${MODEL_NAME_OR_PATH}" >&2
    exit 1
fi

if [[ "${HOBIT_WORKER_MODE:-0}" != "1" ]]; then
    if [[ -e "${OUTPUT_DIR}" ]]; then
        echo "Refusing to reuse existing output: ${OUTPUT_DIR}" >&2
        exit 1
    fi
    mkdir -p "${LOG_DIR}"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="${LOG_DIR}/train_qwen3_5_hobit_${TIMESTAMP}.log"
    nohup env HOBIT_WORKER_MODE=1 /bin/bash "$0" > "${LOG_FILE}" 2>&1 &
    echo "Qwen3.5 HOBIT-C pipeline started in background with PID: $!"
    echo "Monitor with: tail -f ${LOG_FILE}"
    exit 0
fi

if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "Refusing to reuse existing output: ${OUTPUT_DIR}" >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES="${HOBIT_PREFLIGHT_GPU:-0}" python ${SCRIPT_DIR}/verify/verify_qwen3_5_preflight.py \
    --model_name "${MODEL_NAME_OR_PATH}" \
    --min_pixels "${RESIZE_MIN_PIXELS}" \
    --max_pixels "${RESIZE_MAX_PIXELS}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --lora_target_modules "${LORA_TARGET_MODULES}"
CUDA_VISIBLE_DEVICES="${HOBIT_PREFLIGHT_GPU:-0}" python ${SCRIPT_DIR}/verify/verify_hobit_preflight.py

export DHBG_CACHE_DIR="${HOBIT_CACHE_DIR}"
export DHBG_CACHE_STAGE_DIR="${HOBIT_CACHE_STAGE_DIR}"
export DHBG_ENCODE_BATCH_SIZE="${HOBIT_ENCODE_BATCH_SIZE}"
export DHBG_DEFER_SEEDS_TO_V2=1

CACHE_ARGS=(
    --output_dir /tmp/qwen3_5_hobit_builder_state
    --model_name "${MODEL_NAME_OR_PATH}"
    --bf16 --pooling eos --normalize True
    --temperature 0.02
    --dataloader_num_workers 0
    --dataset_config "${DATASET_CONFIG}"
    --negative_strategy vlm2vec_inbatch
    --embedding_gradient_strategy info_nce
    --pos_only True
    --per_device_train_batch_size "${BATCH_SIZE_PER_DEVICE}"
    --max_steps "${MAX_STEPS}"
    --seed "${SEED}"
    --resize_use_processor True
    --resize_min_pixels "${RESIZE_MIN_PIXELS}"
    --resize_max_pixels "${RESIZE_MAX_PIXELS}"
    --interleave_batch_size "${INTERLEAVE_BATCH_SIZE}"
    --interleave_stopping_strategy all_exhausted
    --ddp_timeout 14400
)

if [[ ! -f "${HOBIT_CACHE_DIR}/manifest.json" ]]; then
    echo "Encoding initial Qwen3.5 HOBIT query/positive vectors with the student checkpoint"
    torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT_CACHE}" --max_restarts=0 \
        ${SCRIPT_DIR}/build/build_hobit_cache.py "${CACHE_ARGS[@]}"
fi

# The rolling cache is mutated throughout training. Keep it on node-local
# storage because NumPy mmap seek/write is unreliable on OSS/FUSE.
mkdir -p "${HOBIT_RUNTIME_CACHE_DIR}"
cp "${HOBIT_CACHE_DIR}/manifest.json" "${HOBIT_RUNTIME_CACHE_DIR}/manifest.json"
cp "${HOBIT_CACHE_DIR}/query_embeddings.npy" "${HOBIT_RUNTIME_CACHE_DIR}/query_embeddings.npy"
cp "${HOBIT_CACHE_DIR}/positive_embeddings.npy" "${HOBIT_RUNTIME_CACHE_DIR}/positive_embeddings.npy"

echo "Starting Qwen3.5-2B HOBIT-C training"
echo "Model: ${MODEL_NAME_OR_PATH}"
echo "Data: ${DATASET_CONFIG}"
echo "Output: ${OUTPUT_DIR}"
echo "Global batch: 1024 = two source-local blocks of 512"
echo "HOBIT: seed=32 candidate_k=128 alpha=1.0 tau_h=0.02"
echo "LoRA: r=${LORA_R}, alpha=${LORA_ALPHA}, dropout=${LORA_DROPOUT}"
echo "Initial cache: ${HOBIT_CACHE_DIR}"
echo "Rolling runtime cache: ${HOBIT_RUNTIME_CACHE_DIR}"

torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT_TRAIN}" --max_restarts=0 \
    train_hobit.py \
    --output_dir "${OUTPUT_DIR}" \
    --lora --lora_r "${LORA_R}" --lora_alpha "${LORA_ALPHA}" --lora_dropout "${LORA_DROPOUT}" \
    --lora_target_modules "${LORA_TARGET_MODULES}" \
    --model_name "${MODEL_NAME_OR_PATH}" \
    --bf16 --pooling eos --normalize True \
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
   