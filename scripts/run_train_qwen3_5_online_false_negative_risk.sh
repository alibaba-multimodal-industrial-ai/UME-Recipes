#!/bin/bash
# Isolated Qwen3.5-2B migration of the Qwen2-VL online false-negative-risk
# probability-deletion experiment. It has its own cache/output and does not
# alter the ordinary in-batch, B3, or HOBIT paths.

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
export QWEN3_5_REQUIRE_FAST_KERNELS="${QWEN3_5_REQUIRE_FAST_KERNELS:-0}"

NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT_CACHE="${MASTER_PORT_CACHE:-2278}"
MASTER_PORT_TRAIN="${MASTER_PORT_TRAIN:-2288}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3.5-2B}"
OUTPUT_DIR="${OUTPUT_DIR:-./exp/training/inbatch_qwen3_5_2b_online_fn_probdrop_l15_cap40_lora${LORA_R:-16}}"
LOG_DIR="${LOG_DIR:-./logs}"

# Never share this cache with Qwen2-VL: dimensions, tokenizer/processor and
# initial embedding geometry differ.
export FNRW_CACHE_DIR="${FNRW_CACHE_DIR:-./exp/cache/qwen3_5_inbatch_online_fn_per_dataset_v3_r${RESIZE_MIN_PIXELS:-4096}_m${RESIZE_MAX_PIXELS:-2257920}_pqm${FNRW_FAISS_PQ_M:-128}}"
export FNRW_CACHE_STAGE_DIR="${FNRW_CACHE_STAGE_DIR:-/tmp/qwen3_5_online_fn_cache_build_$$}"
export FNRW_LOG_PATH="${FNRW_LOG_PATH:-${OUTPUT_DIR}/online_fn_risk_steps.jsonl}"

export FNRW_TOP_K="${FNRW_TOP_K:-100}"
export FNRW_LAMBDA="${FNRW_LAMBDA:-1.5}"
export FNRW_MUTUAL_COEFFICIENT="${FNRW_MUTUAL_COEFFICIENT:-0.5}"
export FNRW_MAX_PAIR_DROP_PROB="${FNRW_MAX_PAIR_DROP_PROB:-0.95}"
export FNRW_MAX_DROP_RATIO="${FNRW_MAX_DROP_RATIO:-0.40}"
export FNRW_DEBUG="${FNRW_DEBUG:-${B3_DEBUG}}"
export FNRW_FAISS_NLIST="${FNRW_FAISS_NLIST:-4096}"
export FNRW_FAISS_NPROBE="${FNRW_FAISS_NPROBE:-32}"
export FNRW_FAISS_MIN_LIST_SIZE="${FNRW_FAISS_MIN_LIST_SIZE:-64}"
# Qwen3.5-2B emits 2048-dimensional embeddings. m=128 divides 2048;
# Qwen2's m=96 does not and must not be reused here.
export FNRW_FAISS_PQ_M="${FNRW_FAISS_PQ_M:-128}"
export FNRW_FAISS_PQ_BITS="${FNRW_FAISS_PQ_BITS:-8}"
export FNRW_FAISS_TRAIN_SIZE="${FNRW_FAISS_TRAIN_SIZE:-100000}"
export FNRW_FAISS_THREADS="${FNRW_FAISS_THREADS:-32}"
# Cache construction performs ordinary full forwards rather than GradCache.
# Keep this independent of the semantic training batch to avoid Qwen3.5 OOM.
export FNRW_ENCODE_BATCH_SIZE="${FNRW_ENCODE_BATCH_SIZE:-16}"

BATCH_SIZE_PER_DEVICE="${BATCH_SIZE_PER_DEVICE:-128}"
MAX_STEPS="${MAX_STEPS:-2000}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
SAVE_STEPS="${SAVE_STEPS:-100}"
GC_Q_CHUNK_SIZE="${GC_Q_CHUNK_SIZE:-128}"
GC_P_CHUNK_SIZE="${GC_P_CHUNK_SIZE:-128}"
GC_DYNAMIC_LIMIT="${GC_DYNAMIC_LIMIT:-64}"
# Preserve the Qwen2 online-FN method's one-source-per-global-batch contract.
INTERLEAVE_BATCH_SIZE="${INTERLEAVE_BATCH_SIZE:-128}"
SEED="${SEED:-42}"
export FNRW_MASK_SEED="${FNRW_MASK_SEED:-${SEED}}"
RESIZE_MIN_PIXELS="${RESIZE_MIN_PIXELS:-4096}"
RESIZE_MAX_PIXELS="${RESIZE_MAX_PIXELS:-2257920}"
DATASET_CONFIG="${DATASET_CONFIG:-configs/data_configs/mmeb_new/mmeb20.yaml}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj,down_proj}"

SCRIPT_DIR=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
cd "${SCRIPT_DIR}"

if [[ "${FNRW_WORKER_MODE:-0}" != "1" ]]; then
    if [[ -e "${OUTPUT_DIR}" ]]; then
        echo "Refusing to reuse existing output: ${OUTPUT_DIR}" >&2
        echo "Set OUTPUT_DIR to a new directory." >&2
        exit 1
    fi
    mkdir -p "${LOG_DIR}"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="${LOG_DIR}/train_qwen3_5_online_fn_probdrop_${TIMESTAMP}.log"
    nohup env FNRW_WORKER_MODE=1 /bin/bash "$0" > "${LOG_FILE}" 2>&1 &
    PID=$!
    echo "Qwen3.5 online FN-risk pipeline started in background with PID: ${PID}"
    echo "The same log covers preflight, cache construction and training."
    echo "Monitor with: tail -f ${LOG_FILE}"
    exit 0
fi

if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "Refusing to reuse existing output: ${OUTPUT_DIR}" >&2
    exit 1
fi


echo "Running Qwen3.5 checkpoint/processor/adapter/forward preflight"
CUDA_VISIBLE_DEVICES=0 python ${SCRIPT_DIR}/verify/verify_qwen3_5_preflight.py \
    --model_name "${MODEL_NAME_OR_PATH}" \
    --min_pixels "${RESIZE_MIN_PIXELS}" \
    --max_pixels "${RESIZE_MAX_PIXELS}" \
    --lora_r "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lora_dropout "${LORA_DROPOUT}" \
    --lora_target_modules "${LORA_TARGET_MODULES}"

echo "Starting isolated Qwen3.5 online false-negative-risk experiment"
echo "Cache: ${FNRW_CACHE_DIR}"
echo "Local cache-build stage: ${FNRW_CACHE_STAGE_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Method: K=${FNRW_TOP_K}, lambda=${FNRW_LAMBDA}, mutual=${FNRW_MUTUAL_COEFFICIENT}"
echo "Deletion: max_pair_probability=${FNRW_MAX_PAIR_DROP_PROB}, max_row_ratio=${FNRW_MAX_DROP_RATIO}, mask_seed=${FNRW_MASK_SEED}"
echo "FAISS: pq_m=${FNRW_FAISS_PQ_M}, pq_bits=${FNRW_FAISS_PQ_BITS}"
echo "Cache encode batch per rank: ${FNRW_ENCODE_BATCH_SIZE}"
echo "Training: steps=${MAX_STEPS}, lr=${LEARNING_RATE}, warmup=${WARMUP_STEPS}, GPUs=${NUM_GPUS}"
echo "LoRA: r=${LORA_R}, alpha=${LORA_ALPHA}, dropout=${LORA_DROPOUT}"
echo "Pixels: min=${RESIZE_MIN_PIXELS}, max=${RESIZE_MAX_PIXELS}"

if [[ ! -f "${FNRW_CACHE_DIR}/manifest.json" ]]; then
    echo "Building one-time Qwen3.5 full-corpus target cache and per-dataset Top-K table"
    torchrun \
        --nproc_per_node="${NUM_GPUS}" \
        --master_port="${MASTER_PORT_CACHE}" \
        --max_restarts=0 \
        ${SCRIPT_DIR}/build/build_online_false_negative_cache.py \
        --output_dir /tmp/qwen3_5_online_fn_cache_builder_state \
        --model_name "${MODEL_NAME_OR_PATH}" \
        --bf16 \
        --pooling eos \
        --normalize True \
        --temperature 0.02 \
        --dataloader_num_workers 0 \
        --dataset_config "${DATASET_CONFIG}" \
        --negative_strategy vlm2vec_inbatch \
        --embedding_gradient_strategy info_nce \
        --pos_only True \
        --per_device_train_batch_size "${BATCH_SIZE_PER_DEVICE}" \
        --max_steps "${MAX_STEPS}" \
        --seed "${SEED}" \
        --resize_use_processor True \
        --resize_min_pixels "${RESIZE_MIN_PIXELS}" \
        --resize_max_pixels "${RESIZE_MAX_PIXELS}" \
        --interleave_batch_size "${INTERLEAVE_BATCH_SIZE}" \
        --interleave_stopping_strategy all_exhausted \
        --ddp_timeout 14400
else
    echo "Reusing existing Qwen3.5 target/vector cache; compatibility is revalidated at trainer startup"
fi

echo "Starting ${MAX_STEPS}-step Qwen3.5 online FN-risk training"
torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT_TRAIN}" \
    --max_restarts=0 \
    train_online_false_negative_risk.py \
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
    --resume_from none \
    --seed "${SEED}" \
    --gradient_accumulation_steps 1 \
    --resize_use_processor True \
    --resize_min_pixels "${RESIZE_MIN_PIXELS}" \
    --resize_max_pixels "${RESIZE_MAX_PIXELS}" \
    --interleave_batch_size "${INTERLEAVE_BATCH_SIZE}" \
    --interleave_stopping_strategy all_exhausted \
    --ddp_timeout 14400 \
    --ddp_broadcast_buf