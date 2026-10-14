#!/bin/bash
# Qwen2-VL-7B scaling run for the established ordinary in-batch baseline.
# The global batch, optimizer, data, image resolution and LoRA hyperparameters
# remain identical to the 2B control. Only GradCache replay chunks are smaller.

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
export OUTPUT_DIR="${OUTPUT_DIR:-./exp/training/inbatch_qwen2_vl_7b_lora16}"
export LOG_DIR="${LOG_DIR:-./logs/qwen2_vl_7b}"
export MASTER_PORT="${MASTER_PORT:-2418}"
# Runtime memory knob: lower on OOM or raise when peak-memory headroom permits;
# it does not change the logical batch, negatives, loss, or optimizer step.
export GC_Q_CHUNK_SIZE="${GC_Q_CHUNK_SIZE:-64}"
export GC_P_CHUNK_SIZE="${GC_P_CHUNK_SIZE:-64}"
export GC_DYNAMIC_LIMIT="${GC_DYNAMIC_LIMIT:-32}"

export BATCH_SIZE_PER_DEVICE="${BATCH_SIZE_PER_DEVICE:-128}"
export MAX_STEPS="${MAX_STEPS:-2000}"
export WARMUP_STEPS="${WARMUP_STEPS:-200}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export RESIZE_MIN_PIXELS="${RESIZE_MIN_PIXELS:-4096}"
export RESIZE_MAX_PIXELS="${RESIZE_MAX_PIXELS:-2257920}"
export LORA_R="${LORA_R:-16}"
export LORA_ALPHA="${LORA_ALPHA:-64}"
export LORA_DROPOUT