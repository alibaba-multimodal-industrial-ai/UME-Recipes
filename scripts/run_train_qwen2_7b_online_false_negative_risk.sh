#!/bin/bash
# Qwen2-VL-7B scaling run for probabilistic online false-negative deletion.
# The full-corpus cache and FAISS index are rebuilt for the 3584-d 7B encoder.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-./models/Qwen2-VL-7B-Instruct}"
if [[ ! -f "${MODEL_NAME_OR_PATH}/config.json" || ! -f "${MODEL_NAME_OR_PATH}/tokenizer.json" ]]; then
    echo "Qwen2-VL-7B backbone/processor is incomplete: ${MODEL_NAME_OR_PATH}" >&2
    echo "Set MODEL_NAME_OR_PATH to the complete local Qwen2-VL-7B-Instruct directory." >&2
    exit 1
fi

export MODEL_NAME_OR_PATH
export OUTPUT_DIR="${OUTPUT_DIR:-./exp/training/inbatch_qwen2_vl_7b_online_fn_probdrop_l15_cap40}"
export LOG_DIR="${LOG_DIR:-./logs/qwen2_vl_7b}"
export MASTER_PORT_CACHE="${MASTER_PORT_CACHE:-2478}"
export MASTER_PORT_TRAIN="${MASTER_PORT_TRAIN:-2488}"
# Runtime memory knob; it does not change the logical FN-risk batch/objective.
export GC_Q_CHUNK_SIZE="${GC_Q_CHUNK_SIZE:-64}"
export GC_P_CHUNK_SIZE="${GC_P_CHUNK_SIZE:-64}"
export GC_DYNAMIC_LIMIT="${GC_DYNAMIC_LIMIT:-64}"
export BATCH_SIZE_PER_DEVICE="${BATCH_SIZE_PER_DEVICE:-128}"
export MAX_STEPS="${MAX_STEPS:-2000}"
export WARMUP_STEPS="${WARMUP_STEPS:-200}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export RESIZE_MIN_PIXELS="${RESIZE_MIN_PIXELS:-4096}"
export RESIZE_MAX_PIXELS="${RESIZE_MAX_PIXELS:-2257920}"
export LORA_R="${LORA_R:-16}"
export LORA_ALPHA="${LORA_ALPHA:-64}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
export FNRW_ENCODE_BATCH_SIZE="${FNRW_ENCODE_BATCH_SIZE:-16}"
export FNRW_CACHE_DIR="${FNRW_CACHE_DIR:-./exp/cache/qwen2_vl_7b_inbatch_online_fn_per_dataset_v3}"
export FNRW_CACHE_STAGE_DIR="${FNRW_CACHE_STAGE_DIR:-/tmp/qwen2_vl_7b_online_fn_cache_build_$$}"
# 3584 / 224 = 16 dimensions per PQ subquantizer, matching 1536 / 96 in 2B.
export FNRW_FAISS_PQ_M="${FNRW_FAISS_PQ_M:-224}"

exec /bin/bash "${SCRIPT_DIR}/scripts/run_train_qwen2_online_false_negative_risk.sh"
