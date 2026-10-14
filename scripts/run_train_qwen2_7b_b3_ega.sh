#!/bin/bash
# Qwen2-VL-7B B3+EGA training.
#
# This is an isolated extension of run_train_qwen2_7b_B3.sh: it preserves the
# validated 7B backbone, official qwen7b HNPS/Metis shards, B3 sampler, LoRA,
# optimizer, image-resolution and GradCache settings. The only algorithmic
# change is rectangular EGA over B3's global B x (6B) candidate matrix.
# Existing B3 preprocessing is validated and reused; it is never rebuilt here.

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
export B3_EGA_RUN_SELF_TESTS="${B3_EGA_RUN_SELF_TESTS:-1}"

NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-2421}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-./models/Qwen2-VL-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-./exp/training/b3_ega_qwen2_vl_7b_lora16}"
LOG_DIR="${LOG_DIR:-./logs/qwen2_vl_7b}"

# Keep these values identical to the validated Qwen2-VL-7B B3 run.
BATCH_SIZE_PER_DEVICE="${BATCH_SIZE_PER_DEVICE:-128}"
MAX_STEPS="${MAX_STEPS:-2000}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
SAVE_STEPS="${SAVE_STEPS:-100}"
RESUME_FROM="${RESUME_FROM:-none}"
GC_Q_CHUNK_SIZE="${GC_Q_CHUNK_SIZE:-64}"
GC_P_CHUNK_SIZE="${GC_P_CHUNK_SIZE:-64}"
GC_DYNAMIC_LIMIT="${GC_DYNAMIC_LIMIT:-64}"
CHUNK_SIZE="${CHUNK_SIZE:-32}"
RESIZE_MIN_PIXELS="${RESIZE_MIN_PIXELS:-4096}"
RESIZE_MAX_PIXELS="${RESIZE_MAX_PIXELS:-2257920}"
B3_TEACHER_MODEL="${B3_TEACHER_MODEL:-./models/VLM2Vec-Qwen2VL-7B}"
B3_PREPROCESSED_ROOT="${B3_PREPROCESSED_ROOT:-./MMEB-train}"
DATASET_CONFIG="${DATASET_CONFIG:-configs/data_configs/mmeb_new/mmeb20_HNPS_bs16bi_30.130_30P.10.5_70.170_qwen7b.yaml}"
LORA_R="${LORA_R:-16}"
EGA_ALPHA="${EGA_ALPHA:-20.0}"

SCRIPT_DIR=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
cd "${SCRIPT_DIR}"

if [[ ! -f "${MODEL_NAME_OR_PATH}/config.json" || ! -f "${MODEL_NAME_OR_PATH}/tokenizer.json" ]]; then
    echo "Qwen2-VL-7B backbone/processor is incomplete: ${MODEL_NAME_OR_PATH}" >&2
    echo "Set MODEL_NAME_OR_PATH to the complete local Qwen2-VL-7B-Instruct directory." >&2
    exit 1
fi
if [[ ! -f "${B3_TEACHER_MODEL}/adapter_config.json" ]]; then
    echo "VLM2Vec-Qwen2VL-7B teacher metadata is incomplete: ${B3_TEACHER_MODEL}" >&2
    exit 1
fi
if [[ ! -f "${DATASET_CONFIG}" ]]; then
    echo "B3 qwen7b dataset config does not exist: ${DATASET_CONFIG}" >&2
    exit 1
fi

# Bind the portable checked-in YAML to the existing official B3 cache. Keeping
# this runtime copy also records the exact data source next to the experiment.
RUNTIME_CONFIG_DIR="./exp/runtime_configs"
mkdir -p "${RUNTIME_CONFIG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUNTIME_DATASET_CONFIG="${RUNTIME_CONFIG_DIR}/qwen2_7b_B3_EGA_${TIMESTAMP}.yaml"
sed "s|dataset_name: ./MMEB-train|dataset_name: ${B3_PREPROCESSED_ROOT}|g" \
    "${DATASET_CONFIG}" > "${RUNTIME_DATASET_CONFIG}"

EXPECTED_SUBSETS=0
while IFS= read -r subset_name; do
    [[ -n "${subset_name}" ]] || continue
    EXPECTED_SUBSETS=$((EXPECTED_SUBSETS + 1))
    if [[ "${subset_name}" != *qwen7b ]]; then
        echo "Refusing non-qwen7b B3 shard: ${subset_name}" >&2
        exit 1
    fi
    subset_dir="${B3_PREPROCESSED_ROOT}/${subset_name}"
    if [[ ! -d "${subset_dir}" ]] || ! find "${subset_dir}" -type f -name '*.parquet' -print -quit | grep -q .; then
        echo "Missing official B3 qwen7b shard: ${subset_dir}" >&2
        exit 1
    fi
done < <(sed -n 's/^[[:space:]]*subset_name:[[:space:]]*//p' "${RUNTIME_DATASET_CONFIG}")
if [[ "${EXPECTED_SUBSETS}" -ne 20 ]]; then
    echo "Expected 20 B3 qwen7b subsets, found ${EXPECTED_SUBSETS} in ${DATASET_CONFIG}" >&2
    exit 1
fi

if [[ "${B3_EGA_RUN_SELF_TESTS}" == "1" ]]; then
    echo "Running B3+EGA rectangular formula and GradCache self-tests"
    python -m unittest tests.test_b3_ega -v
    if [[ "${NUM_GPUS}" -ge 2 ]]; then
        echo "Running B3+EGA two-rank distributed cache preflight"
        torchrun --standalone --nproc_per_node=2 "${SCRIPT_DIR}/tests/verify_b3_ega_distributed.py"
    fi
fi

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_qwen2_7b_B3_EGA_lora16_${TIMESTAMP}.log"

echo "Starting Qwen2-VL-7B B3+EGA training"
echo "Model: ${MODEL_NAME_OR_PATH}"
echo "Teacher provenance: ${B3_TEACHER_MODEL}"
echo "Official qwen7b shards: ${B3_PREPROCESSED_ROOT}"
echo "Data template: ${DATASET_CONFIG}"
echo "Resolved data config: ${RUNTIME_DATASET_CONFIG}"
echo "Preprocessing: skipped (validated ${EXPECTED_SUBSETS} existing qwen7b shards)"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${NUM_GPUS}"
echo "Global query batch: $((NUM_GPUS * BATCH_SIZE_PER_DEVICE))"
echo "B3 target layout: 1 positive + 5 explicit hard negatives per query"
echo "Global target batch: $((NUM_GPUS * BATCH_SIZE_PER_DEVICE * 6))"
echo "GradCache chunks: query=${GC_Q_CHUNK_SIZE}, target=${GC_P_CHUNK_SIZE}"
echo "LoRA rank: ${LORA_R}"
echo "EGA alpha: ${EGA_ALPHA}"
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
    --dataset_config "${RUNTIME_DATASET_CONFIG}" \
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
e
