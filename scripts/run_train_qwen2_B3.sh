#!/bin/bash
# B3 Model Training Script (LoRA + nohup)
# Reproduced from B3/run_train.sh for Qwen3-VL-Embedding-main
# Trains Qwen2-VL-2B with B3 strategy (LoRA + DoRA + GradCache)

set -e

# ==================== Configuration ====================

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/tmp/ume-recipes/huggingface
export WANDB_MODE=disabled

# B3 embedding monitor (zero extra forward pass)
export B3_DEBUG=1
export B3_DEBUG_INTERVAL="${B3_DEBUG_INTERVAL:-1}"
# Keep extra new-framework diagnostics off for a clean B3 log comparison.
# They can still be enabled explicitly when investigating a hang or NaN.
export B3_DDP_TRACE_STEPS="${B3_DDP_TRACE_STEPS:-0}"
export B3_NUMERIC_DEBUG_STEPS="${B3_NUMERIC_DEBUG_STEPS:-0}"
export B3_FINGERPRINT="${B3_FINGERPRINT:-0}"
export B3_FINGERPRINT_CALLS="${B3_FINGERPRINT_CALLS:-20}"
export B3_FINGERPRINT_RANKS="${B3_FINGERPRINT_RANKS:-0}"

# GPU settings
NUM_GPUS=8
MASTER_PORT=2215

# Model path (local) or HuggingFace model name
MODEL_NAME_OR_PATH="./models/Qwen2-VL-2B-Instruct"

# Output directory (resume from existing checkpoint)
# Keep this rank-local GradCache reproduction separate from the earlier run
# whose synchronized chunking changed the LoRA-dropout RNG trajectory.
OUTPUT_DIR="${OUTPUT_DIR:-./exp/training/b3_qwen2b_lora16}"

# Log directory
LOG_DIR="./logs"
mkdir -p ${LOG_DIR}

# Training hyperparameters
BATCH_SIZE_PER_DEVICE="${BATCH_SIZE_PER_DEVICE:-128}"
MAX_STEPS="${MAX_STEPS:-2000}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"

# Checkpoint saving interval (every N steps)
SAVE_STEPS="${SAVE_STEPS:-100}"
RESUME_FROM="${RESUME_FROM:-none}"

# Gradient cache settings
GC_Q_CHUNK_SIZE=128
GC_P_CHUNK_SIZE=128
GC_DYNAMIC_LIMIT=64

# B3 specific parameters
CHUNK_SIZE=32

# Data config
DATASET_CONFIG="configs/data_configs/mmeb_new/mmeb20_HNPS_bs16bi_30.130_30P.10.5_70.170_qwen2b.yaml"

# LoRA settings
LORA_R=16

# ==================== Training Command ====================

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_qwen2_B3_lora16_${TIMESTAMP}.log"

echo "Starting B3 training (LoRA mode)..."
echo "Model: ${MODEL_NAME_OR_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${NUM_GPUS}"
echo "Batch size per device: ${BATCH_SIZE_PER_DEVICE}"
echo "Max steps: ${MAX_STEPS}"
echo "Save steps: ${SAVE_STEPS}"
echo "LoRA rank: ${LORA_R}"
echo "Log file: ${LOG_FILE}"

nohup torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_port=${MASTER_PORT} \
    --max_restarts=0 \
    train.py \
    --output_dir ${OUTPUT_DIR} \
    --lora \
    --lora_r ${LORA_R} \
    --model_name ${MODEL_NAME_OR_PATH} \
    --bf16 \
    --pooling eos \
    --normalize True \
    --qwen2_model_impl b3 \
    --qwen2_b3_attention_impl flash_attention_2 \
    --qwen2_b3_legacy_position_ids False \
    --temperature 0.02 \
    --dataloader_num_workers 0 \
    --dataset_config ${DATASET_CONFIG} \
    --grad_cache True \
    --per_device_train_batch_size ${BATCH_SIZE_PER_DEVICE} \
    --gc_q_chunk_size ${GC_Q_CHUNK_SIZE} \
    --gc_p_chunk_size ${GC_P_CHUNK_SIZE} \
    --gc_dynamic_limit ${GC_DYNAMIC_LIMIT} \
    --optim adamw_torch \
    --lr_scheduler_type linear \
    --learning_rate ${LEARNING_RATE} \
    --max_steps ${MAX_STEPS} \
    --warmup_steps ${WARMUP_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --logging_steps 1 \
    --save_safetensors False \
    --remove_unused_columns False \
    --resume_from ${RESUME_FROM} \
    --resize_use_processor True \
    --interleave_batch_size 1 \
    --ddp_timeout 14400 \
    --ddp_broadcast_buffers False \
    --ddp_find_unused_parameters False \
    --ignore_data_skip true \
    --sdibn \
    --odibn \
    --chunk_size ${CHUNK_SIZE} \
    > ${LOG_FILE} 2>&1 &

PID=$!
echo "Training started in background with PID: ${PID}"
echo "To monitor training: tail -f ${LOG_FILE}"
echo "To check process: ps -p ${PID}"
echo ""
echo "Training is continuing in the background; checkpoints will be saved to: ${OUTPUT_DIR}"
