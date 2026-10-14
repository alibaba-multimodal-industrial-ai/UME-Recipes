#!/bin/bash
# Qwen2-VL-7B scaling run for HOBIT-C. The HOBIT construction is unchanged;
# its initial/rolling caches are model-specific and therefore use 7B-only paths.

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
export OUTPUT_DIR="${OUTPUT_DIR:-./exp/training/inbatch_qwen2_vl_7b_hobit_c}"
export LOG_DIR="${LOG_DIR:-./logs/qwen2_vl_7b}"
export MASTER_PORT_CACHE="${MASTER_PORT_CACHE:-2458}"
export MASTER_PORT_TRAIN="${MASTER_PORT_TRAIN:-2468}"
# Runtime memory knobs; tune them to measured headroom without changing HOBIT.
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
# Initial-cache encoding only; lower on OOM or raise for throughput.
export HOBIT_ENCODE_BATCH_SIZE="${HOBIT_ENCODE_BATCH_SIZE:-8}"
export HOBIT_CACHE_DIR="${HOBIT_CACHE_DIR:-./exp/cache/qwen2_vl_7b_hobit_initial_student}"
export HOBIT_CACHE_STAGE_DIR="${HOBIT_CACHE_STAGE_DIR:-/tmp/qwen2_vl_7b_hobit_cache_build_$$}"
export HOBIT_RUNTIME_CACHE_DIR="${HOBIT_RUNTIME_CACHE_DIR:-/tmp/qwen2_vl_7b_hobit_runtime_$$}"

exec /bin/bash "${SCRIPT_DIR}/scripts/run_train_qwen2_hobit.sh"
