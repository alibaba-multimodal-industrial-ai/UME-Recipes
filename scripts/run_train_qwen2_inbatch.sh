#!/bin/bash
# Train Qwen2-VL-2B with VLM2Vec's ordinary cross-device in-batch negatives.
# The proven Qwen2-VL B3-compatible model/processor/LoRA/GradCache path and
# optimization hyperparameters match run_train.sh. Only the original MMEB
# pairs, source interleaving, and negative construction differ.

set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-/tmp/ume-recipes/huggingface}"
export WANDB_MODE=disabled

# Keep the same per-step metrics as Qwen2-VL B3. The first two rank-0
# fingerprints additionally verify checkpoint tensors, adapter initialization,
# raw pairs, packed inputs, and early representations for this new branch.
export B3_DEBUG="${B3_DEBUG:-1}"
export B3_DEBUG_INTERVAL="${B3_DEBUG_INTERVAL:-1}"
export B3_DDP_TRACE_STEPS="${B3_DDP_TRACE_STEPS:-0}"
export B3_NUMERIC_DEBUG_STEPS="${B3_NUMERIC_DEBUG_STEPS:-0}"
export B3_FINGERPRINT="${B3_FINGERPRINT:-1}"
export B3_FINGERPRINT_CALLS="${B3_FINGERPRINT_CALLS:-2}"
export B3_FINGERPRINT_RANKS="${B3_FINGERPRINT_RANKS:-0}"

NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-2218}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-./models/Qwen2-VL-2B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-./exp/training/inbatch_qwen2_vl_2b_lora32}"
LOG_DIR="${LOG_DIR:-./logs}"

# Keep these equal to the successful Qwen2-VL+B3 run so the experiment isolates
# the ordinary in-batch-negative strategy instead of changing the optimizer.
BATCH_SIZE_PER_DEVICE="${BATCH_SIZE_PER_DEVICE:-128}"
MAX_STEPS="${MAX_STEPS:-2000}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
SAVE_STEPS="${SAVE_STEPS:-100}"
RESUME_FROM="${RESUME_FROM:-none}"

GC_Q_CHUNK_SIZE="${GC_Q_CHUNK_SIZE:-128}"
GC_P_CHUNK_SIZE="${GC_P_CHUNK_SIZE:-128}"
GC_DYNAMIC_LIMIT="${GC_DYNAMIC_LIMIT:-64}"
# VLM2Vec interprets 64 as examples per device from one selected source. With
# 8 ranks this becomes a 512-example source block in the global sampler.
INTERLEAVE_BATCH_SIZE="${INTERLEAVE_BATCH_SIZE:-64}"
SEED="${SEED:-42}"
RESIZE_MIN_PIXELS="${RESIZE_MIN_PIXELS:-4096}"
RESIZE_MAX_PIXELS="${RESIZE_MAX_PIXELS:-2257920}"

# Original MMEB query-positive pairs only. Never point this control branch at
# HNPS/HNIPS/Metis data; train.py also rejects those configurations.
DATASET_CONFIG="${DATASET_CONFIG:-configs/data_configs/mmeb_new/mmeb20.yaml}"

LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-qkv_proj,o_proj,gate_up_proj,down_proj,k_proj,q_proj,out_proj,v_proj}"

mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_qwen2_inbatch_${TIMESTAMP}.log"

echo "Starting Qwen2-VL-2B VLM2Vec ordinary in-batch training"
echo "Model: ${MODEL_NAME_OR_PATH}"
echo "Data: ${DATASET_CONFIG}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${NUM_GPUS}"
echo "Batch size per device: ${BATCH_SIZE_PER_DEVICE}"
echo "LoRA/DoRA: r=${LORA_R}, alpha=${LORA_ALPHA}, dropout=${LORA_DROPOUT}"
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
    --ignore_data_skip false \
    > "${LOG_FILE}" 2>&1 &

PID=$!
echo "Training started in background with PID: ${PID}"
echo "Monitor with: tail -f ${LOG_FILE}"
