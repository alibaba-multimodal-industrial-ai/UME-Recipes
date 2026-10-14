#!/bin/bash
# Fully isolated Qwen2-VL online false-negative-risk probability-deletion experiment.
# Stage 1 builds the one-time full-corpus target cache and Top-K index.
# Stage 2 aggressively samples and removes high-risk in-batch negatives.

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
MASTER_PORT_CACHE="${MASTER_PORT_CACHE:-2248}"
MASTER_PORT_TRAIN="${MASTER_PORT_TRAIN:-2258}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-./models/Qwen2-VL-2B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-./exp/training/inbatch_qwen2_vl_2b_online_fn_probdrop_l15_cap40}"
LOG_DIR="${LOG_DIR:-./logs}"
export FNRW_CACHE_DIR="${FNRW_CACHE_DIR:-./exp/cache/qwen2_vl_inbatch_online_fn_per_dataset_v3}"
# NumPy mmap is unsupported on the OSS/FUSE mount. All build-time shards stay
# on node-local disk; only finalized runtime artifacts are streamed to OSS.
export FNRW_CACHE_STAGE_DIR="${FNRW_CACHE_STAGE_DIR:-/tmp/qwen2_vl_online_fn_cache_build_$$}"
export FNRW_LOG_PATH="${FNRW_LOG_PATH:-${OUTPUT_DIR}/online_fn_risk_steps.jsonl}"

# Aggressive probabilistic deletion: p_drop=1-exp(-lambda*energy), followed by
# a per-anchor budget cap and low-variance dependent Bernoulli sampling.
export FNRW_TOP_K="${FNRW_TOP_K:-100}"
export FNRW_LAMBDA="${FNRW_LAMBDA:-1.5}"
export FNRW_MUTUAL_COEFFICIENT="${FNRW_MUTUAL_COEFFICIENT:-0.5}"
export FNRW_MAX_PAIR_DROP_PROB="${FNRW_MAX_PAIR_DROP_PROB:-0.95}"
export FNRW_MAX_DROP_RATIO="${FNRW_MAX_DROP_RATIO:-0.40}"
export FNRW_DEBUG="${FNRW_DEBUG:-${B3_DEBUG}}"
export FNRW_FAISS_NLIST="${FNRW_FAISS_NLIST:-4096}"
export FNRW_FAISS_NPROBE="${FNRW_FAISS_NPROBE:-32}"
export FNRW_FAISS_MIN_LIST_SIZE="${FNRW_FAISS_MIN_LIST_SIZE:-64}"
export FNRW_FAISS_PQ_M="${FNRW_FAISS_PQ_M:-96}"
export FNRW_FAISS_PQ_BITS="${FNRW_FAISS_PQ_BITS:-8}"
export FNRW_FAISS_TRAIN_SIZE="${FNRW_FAISS_TRAIN_SIZE:-100000}"
export FNRW_FAISS_THREADS="${FNRW_FAISS_THREADS:-32}"

# Match run_train_qwen2_inbatch.sh except for the deliberate one-source-per-
# global-batch interleave setting below.
BATCH_SIZE_PER_DEVICE="${BATCH_SIZE_PER_DEVICE:-128}"
MAX_STEPS="${MAX_STEPS:-2000}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
SAVE_STEPS="${SAVE_STEPS:-100}"
GC_Q_CHUNK_SIZE="${GC_Q_CHUNK_SIZE:-128}"
GC_P_CHUNK_SIZE="${GC_P_CHUNK_SIZE:-128}"
GC_DYNAMIC_LIMIT="${GC_DYNAMIC_LIMIT:-64}"
# One 128-example source block per device = one 1024-example source per
# global batch on 8 GPUs. This keeps risk neighborhoods and in-batch pairs
# inside the same MMEB dataset for this experiment.
INTERLEAVE_BATCH_SIZE="${INTERLEAVE_BATCH_SIZE:-128}"
SEED="${SEED:-42}"
export FNRW_MASK_SEED="${FNRW_MASK_SEED:-${SEED}}"
RESIZE_MIN_PIXELS="${RESIZE_MIN_PIXELS:-4096}"
RESIZE_MAX_PIXELS="${RESIZE_MAX_PIXELS:-2257920}"
DATASET_CONFIG="${DATASET_CONFIG:-configs/data_configs/mmeb_new/mmeb20.yaml}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-qkv_proj,o_proj,gate_up_proj,down_proj,k_proj,q_proj,out_proj,v_proj}"

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
    LOG_FILE="${LOG_DIR}/train_qwen2_online_fn_probdrop_${TIMESTAMP}.log"
    nohup env FNRW_WORKER_MODE=1 /bin/bash "$0" > "${LOG_FILE}" 2>&1 &
    PID=$!
    echo "Online FN-risk pipeline started in background with PID: ${PID}"
    echo "The same log covers initial cache construction and 2000-step training."
    echo "Monitor with: tail -f ${LOG_FILE}"
    exit 0
fi

if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "Refusing to reuse existing output: ${OUTPUT_DIR}" >&2
    exit 1
fi

if ! python3 -c 'import faiss; print("[ONLINE_FN_DEPENDENCIES_OK] faiss", faiss.__version__)'; then
    echo "This isolated experiment requires FAISS in the active Qwen environment." >&2
    echo "Install with: pip install -r requirements_online_false_negative_risk.txt" >&2
    exit 1
fi

echo "Starting isolated online false-negative-risk probability-deletion experiment"
echo "Cache: ${FNRW_CACHE_DIR}"
echo "Local cache-build stage: ${FNRW_CACHE_STAGE_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Method: K=${FNRW_TOP_K}, lambda=${FNRW_LAMBDA}, mutual=${FNRW_MUTUAL_COEFFICIENT}"
echo "Deletion: max_pair_probability=${FNRW_MAX_PAIR_DROP_PROB}, max_row_ratio=${FNRW_MAX_DROP_RATIO}, mask_seed=${FNRW_MASK_SEED}"
echo "Training: steps=${MAX_STEPS}, lr=${LEARNING_RATE}, warmup=${WARMUP_STEPS}, GPUs=${NUM_GPUS}"

if [[ ! -f "${FNRW_CACHE_DIR}/manifest.json" ]]; then
    echo "Building one-time full-corpus target cache and Top-K table"
    torchrun \
        --nproc_per_node="${NUM_GPUS}" \
        --master_port="${MASTER_PORT_CACHE}" \
        --max_restarts=0 \
        ${SCRIPT_DIR}/build/build_online_false_negative_cache.py \
        --output_dir /tmp/qwen2_online_fn_cache_builder_state \
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
    echo "Reusing existing Qwen2-VL target/vector cache; compatibility is revalidated at trainer startup"
fi

echo "Starting ${MAX_STEPS}-step online FN-risk probability-deletion training"
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