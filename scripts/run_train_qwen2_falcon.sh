#!/bin/bash
# Qwen2-VL FALCON: official quantile-policy batch construction with the agreed
# per-MMEB-source rotating safe-probe reward. Existing launchers are untouched.

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

export NUM_GPUS="${NUM_GPUS:-8}"
export MASTER_PORT_CACHE="${MASTER_PORT_CACHE:-2398}"
export MASTER_PORT_TRAIN="${MASTER_PORT_TRAIN:-2408}"
export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-./models/Qwen2-VL-2B-Instruct}"
export LOG_DIR="${LOG_DIR:-./logs}"
export DATASET_CONFIG="${DATASET_CONFIG:-configs/data_configs/mmeb_new/mmeb20.yaml}"

export BATCH_SIZE_PER_DEVICE="${BATCH_SIZE_PER_DEVICE:-128}"
export MAX_STEPS="${MAX_STEPS:-2000}"
export WARMUP_STEPS="${WARMUP_STEPS:-200}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export SAVE_STEPS="${SAVE_STEPS:-100}"
export SEED="${SEED:-42}"
export RESUME_FROM="${RESUME_FROM:-none}"
export GC_Q_CHUNK_SIZE="${GC_Q_CHUNK_SIZE:-128}"
export GC_P_CHUNK_SIZE="${GC_P_CHUNK_SIZE:-128}"
export GC_DYNAMIC_LIMIT="${GC_DYNAMIC_LIMIT:-64}"
export INTERLEAVE_BATCH_SIZE="${INTERLEAVE_BATCH_SIZE:-64}"
export RESIZE_MIN_PIXELS="${RESIZE_MIN_PIXELS:-4096}"
export RESIZE_MAX_PIXELS="${RESIZE_MAX_PIXELS:-2257920}"
export LORA_R="${LORA_R:-16}"
export LORA_ALPHA="${LORA_ALPHA:-64}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
export LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-qkv_proj,o_proj,gate_up_proj,down_proj,k_proj,q_proj,out_proj,v_proj}"

# Resolution-dependent artifacts must never share a path.  In particular, the
# old 3136/1003520 vectors are not valid for the 4096/2257920 experiment even
# if a recovered manifest was later edited to advertise the new pixel bounds.
export FALCON_RESOLUTION_TAG="${FALCON_RESOLUTION_TAG:-min${RESIZE_MIN_PIXELS}_max${RESIZE_MAX_PIXELS}}"
export OUTPUT_DIR="${OUTPUT_DIR:-./exp/training/inbatch_qwen2_vl_2b_falcon_r${LORA_R}_${FALCON_RESOLUTION_TAG}}"

# Paper-faithful actor and chain.  5632 is the nearest multiple of the local
# 512-case source block to the official 5664-case search space.
export FALCON_SOURCE_BLOCK_SIZE="${FALCON_SOURCE_BLOCK_SIZE:-512}"
export FALCON_SEARCH_SPACE_SIZE="${FALCON_SEARCH_SPACE_SIZE:-5632}"
export FALCON_NUM_DOWNSAMPLE="${FALCON_NUM_DOWNSAMPLE:-100}"
export FALCON_POLICY_HIDDEN_DIM="${FALCON_POLICY_HIDDEN_DIM:-256}"
export FALCON_POLICY_RESIDUAL_BLOCKS="${FALCON_POLICY_RESIDUAL_BLOCKS:-2}"
export FALCON_POLICY_LR="${FALCON_POLICY_LR:-1e-4}"
export FALCON_POLICY_UPDATE_FREQ="${FALCON_POLICY_UPDATE_FREQ:-10}"
export FALCON_PROBE_GLOBAL_SIZE="${FALCON_PROBE_GLOBAL_SIZE:-64}"
export FALCON_TEMPERATURE="${FALCON_TEMPERATURE:-0.02}"
export FALCON_ENCODE_BATCH_SIZE="${FALCON_ENCODE_BATCH_SIZE:-16}"

export FALCON_STUDENT_CACHE_DIR="${FALCON_STUDENT_CACHE_DIR:-./exp/cache/qwen2_vl_falcon_initial_student_${FALCON_RESOLUTION_TAG}}"
export FALCON_STUDENT_CACHE_STAGE_DIR="${FALCON_STUDENT_CACHE_STAGE_DIR:-/tmp/qwen2_vl_falcon_student_build_$$}"
export FALCON_STUDENT_RUNTIME_CACHE_DIR="${FALCON_STUDENT_RUNTIME_CACHE_DIR:-/tmp/qwen2_vl_falcon_student_runtime_$$}"

# Build or reuse resolution-tagged VLM2Vec teacher vectors and a B3-safe sparse
# graph. They are used only to remove likely false negatives from the probe;
# FALCON batch similarity always comes from the rolling student cache above.
export FALCON_TEACHER_MODEL="${FALCON_TEACHER_MODEL:-./models/VLM2Vec-V2.0}"
export FALCON_TEACHER_BASE_MODEL="${FALCON_TEACHER_BASE_MODEL:-${MODEL_NAME_OR_PATH}}"
export FALCON_TEACHER_CACHE_DIR="${FALCON_TEACHER_CACHE_DIR:-./exp/cache/qwen2_vl_falcon_teacher_vlm2vec_v2_${FALCON_RESOLUTION_TAG}}"
export FALCON_TEACHER_CACHE_STAGE_DIR="${FALCON_TEACHER_CACHE_STAGE_DIR:-/tmp/qwen2_vl_falcon_teacher_build_$$}"
export FALCON_PROBE_GRAPH_CACHE_DIR="${FALCON_PROBE_GRAPH_CACHE_DIR:-./exp/cache/qwen2_vl_falcon_probe_graph_vlm2vec_teacher_${FALCON_RESOLUTION_TAG}_r256_a0p03_kb3_s42_n2000}"
export FALCON_PROBE_GRAPH_STAGE_DIR="${FALCON_PROBE_GRAPH_STAGE_DIR:-/tmp/qwen2_vl_falcon_graph_build_$$}"
export FALCON_PROBE_GRAPH_RUNTIME_DIR="${FALCON_PROBE_GRAPH_RUNTIME_DIR:-/tmp/qwen2_vl_falcon_graph_runtime_$$}"
export FALCON_GRAPH_VECTOR_RUNTIME_DIR="${FALCON_GRAPH_VECTOR_RUNTIME_DIR:-/tmp/qwen2_vl_falcon_teacher_vectors_$$}"
export FALCON_B3_DATASET_CONFIG="${FALCON_B3_DATASET_CONFIG:-configs/data_configs/mmeb_new/mmeb20_HNPS_bs16bi_30.130_30P.10.5_70.170_qwen2b.yaml}"
export FALCON_DIAGNOSTICS_PATH="${FALCON_DIAGNOSTICS_PATH:-${OUTPUT_DIR}/falcon_diagnostics.jsonl}"

SCRIPT_DIR=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
cd "${SCRIPT_DIR}"

if [[ "${NUM_GPUS}" != "8" || "${BATCH_SIZE_PER_DEVICE}" != "128" ]]; then
    echo "Controlled FALCON comparison requires 8 GPUs and batch size 128 per GPU." >&2
    exit 1
fi
if [[ "${INTERLEAVE_BATCH_SIZE}" != "64" || "${FALCON_SOURCE_BLOCK_SIZE}" != "512" ]]; then
    echo "FALCON requires exactly two 512-case source blocks per global batch." >&2
    exit 1
fi
if [[ "${FALCON_SEARCH_SPACE_SIZE}" != "5632" || "${FALCON_NUM_DOWNSAMPLE}" != "100" ]]; then
    echo "Controlled FALCON reproduction requires search_space=5632 and state=100." >&2
    exit 1
fi
if [[ "${FALCON_POLICY_HIDDEN_DIM}" != "256" || "${FALCON_POLICY_RESIDUAL_BLOCKS}" != "2" || "${FALCON_POLICY_UPDATE_FREQ}" != "10" ]]; then
    echo "FALCON actor architecture/update frequency differs from the official implementation." >&2
    exit 1
fi
if [[ "${FALCON_POLICY_LR}" != "1e-4" && "${FALCON_POLICY_LR}" != "0.0001" ]]; then
    echo "FALCON policy AdamW learning rate must remain at the official 1e-4." >&2
    exit 1
fi
if [[ "${FALCON_PROBE_GLOBAL_SIZE}" != "64" || "${FALCON_TEMPERATURE}" != "0.02" ]]; then
    echo "Controlled MMEB probe requires global size 64 and temperature 0.02." >&2
    exit 1
fi
if [[ "${SEED}" != "42" || "${RESUME_FROM}" != "none" ]]; then
    echo "This first controlled FALCON run requires SEED=42 and RESUME_FROM=none." >&2
    exit 1
fi
if [[ ! -f "${MODEL_NAME_OR_PATH}/config.json" || ! -f "${MODEL_NAME_OR_PATH}/tokenizer.json" ]]; then
    echo "Local Qwen2-VL backbone/processor is incomplete: ${MODEL_NAME_OR_PATH}" >&2
    exit 1
fi
if [[ ! -f "${FALCON_TEACHER_MODEL}/adapter_config.json" ]]; then
    echo "VLM2Vec teacher adapter_config.json is missing: ${FALCON_TEACHER_MODEL}" >&2
    exit 1
fi
if ! compgen -G "${FALCON_TEACHER_MODEL}/adapter_model*.safetensors" >/dev/null \
    && ! compgen -G "${FALCON_TEACHER_MODEL}/adapter_model*.bin" >/dev/null; then
    echo "VLM2Vec teacher adapter weights are missing: ${FALCON_TEACHER_MODEL}" >&2
    exit 1
fi

if [[ "${FALCON_WORKER_MODE:-0}" != "1" ]]; then
    if [[ -e "${OUTPUT_DIR}" ]]; then
        echo "Refusing to reuse existing output: ${OUTPUT_DIR}" >&2
        exit 1
    fi
    mkdir -p "${LOG_DIR}"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="${LOG_DIR}/train_qwen2_falcon_${TIMESTAMP}.log"
    nohup env FALCON_WORKER_MODE=1 /bin/bash "$0" > "${LOG_FILE}" 2>&1 &
    echo "Qwen2-VL FALCON pipeline started in background with PID: $!"
    echo "Monitor with: tail -f ${LOG_FILE}"
    exit 0
fi

if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "Refusing to reuse existing output: ${OUTPUT_DIR}" >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES="${FALCON_PREFLIGHT_GPU:-0}" python ${SCRIPT_DIR}/verify/verify_falcon_preflight.py

export DHBG_CACHE_DIR="${FALCON_STUDENT_CACHE_DIR}"
export DHBG_CACHE_STAGE_DIR="${FALCON_STUDENT_CACHE_STAGE_DIR}"
export DHBG_ENCODE_BATCH_SIZE="${FALCON_ENCODE_BATCH_SIZE}"
export DHBG_DEFER_SEEDS_TO_V2=1

STUDENT_CACHE_ARGS=(
    --output_dir /tmp/qwen2_falcon_student_builder_state
    --model_name "${MODEL_NAME_OR_PATH}"
    --bf16 --pooling eos --normalize True
    --qwen2_model_impl b3
    --qwen2_b3_attention_impl flash_attention_2
    --qwen2_b3_legacy_position_ids False
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

if [[ ! -f "${FALCON_STUDENT_CACHE_DIR}/manifest.json" ]]; then
    echo "Encoding FALCON epoch-0 query/positive vectors with the student checkpoint"
    torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT_CACHE}" --max_restarts=0 \
        ${SCRIPT_DIR}/build/build_falcon_cache.py "${STUDENT_CACHE_ARGS[@]}"
fi

TEACHER_CACHE_ARGS=(
    --output_dir /tmp/qwen2_falcon_teacher_builder_state
    --model_name "${FALCON_TEACHER_MODEL}"
    --processor_name "${FALCON_TEACHER_BASE_MODEL}"
    --init_lora_model True --init_lora_base_model "${FALCON_TEACHER_BASE_MODEL}"
    --bf16 --pooling eos --normalize True
    --qwen2_model_impl b3
    --qwen2_b3_attention_impl flash_attention_2
    --qwen2_b3_legacy_position_ids False
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

if [[ ! -f "${FALCON_TEACHER_CACHE_DIR}/manifest.json" ]]; then
    echo "Encoding resolution-matched probe vectors with the VLM2Vec-V2.0 teacher"
    export DHBG_CACHE_DIR="${FALCON_TEACHER_CACHE_DIR}"
    export DHBG_CACHE_STAGE_DIR="${FALCON_TEACHER_CACHE_STAGE_DIR}"
    torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT_CACHE}" --max_restarts=0 \
        ${SCRIPT_DIR}/build/build_dynamic_hard_batch_cache.py "${TEACHER_CACHE_ARGS[@]}"
fi

if [[ ! -f "${FALCON_PROBE_GRAPH_CACHE_DIR}/manifest.json" ]]; then
    echo "Building the B3-safe sparse probe graph from resolution-matched VLM2Vec teacher vectors"
    mkdir -p "${FALCON_GRAPH_VECTOR_RUNTIME_DIR}"
    cp "${FALCON_TEACHER_CACHE_DIR}/manifest.json" "${FALCON_GRAPH_VECTOR_RUNTIME_DIR}/manifest.json"
    cp "${FALCON_TEACHER_CACHE_DIR}/query_embeddings.npy" "${FALCON_GRAPH_VECTOR_RUNTIME_DIR}/query_embeddings.npy"
    cp "${FALCON_TEACHER_CACHE_DIR}/positive_embeddings.npy" "${FALCON_GRAPH_VECTOR_RUNTIME_DIR}/positive_embeddings.npy"
    CUDA_VISIBLE_DEVICES="${FALCON_GRAPH_BUILD_GPU:-0}" python "${SCRIPT_DIR}/build/build_dynamic_hard_batch_v2_cache.py" \
        --source-cache "${FALCON_GRAPH_VECTOR_RUNTIME_DIR}" \
        --output-cache "${FALCON_PROBE_GRAPH_CACHE_DIR}" \
        --stage-dir "${FALCON_PROBE_GRAPH_STAGE_DIR}" \
        --dataset-config "${DATASET_CONFIG}" \
        --b3-dataset-config "${FALCON_B3_DATASET_CONFIG}" \
        --student-model "${MODEL_NAME_OR_PATH}" \
        --resize-min-pixels "${RESIZE_MIN_PIXELS}" \
        --resize-max-pixels "${RESIZE_MAX_PIXELS}" \
        --max-steps 2000 --global-batch-size 1024 \
        --rank-depth 256 --rank-alpha 0.03 --rank-query-chunk 128 \
        --seed "${SEED}"
    rm -f "${FALCON_GRAPH_VECTOR_RUNTIME_DIR}/manifest.json"
    rm -f "${FALCON_GRAPH_VECTOR_RUNTIME_DIR}/query_embeddings.npy"
    rm -f "${FALCON_GRAPH_VECTOR_RUNTIME_DIR}/positive_embeddings.npy"
    rmdir "${FALCON_GRAPH_VECTOR_RUNTIME_DIR}" 2>/dev/null || true
fi

# Both caches are read/written through mmap during training. OSS/FUSE does not
# provide the required semantics, so use run-specific node-local copies.
mkdir -p "${FALCON_STUDENT_RUNTIME_CACHE_DIR}"
cp "${FALCON_STUDENT_CACHE_DIR}/manifest.json" "${FALCON_STUDENT_RUNTIME_CACHE_DIR}/manifest.json"
cp "${FALCON_STUDENT_CACHE_DIR}/query_embeddings.npy" "${FALCON_STUDENT_RUNTIME_CACHE_DIR}/query_embeddings.npy"
cp "${FALCON_STUDENT_CACHE_DIR}/positive_embeddings.npy" "${FALCON_STUDENT_RUNTIME_CACHE_DIR}/positive_embeddings.npy"

mkdir -p "${FALCON_PROBE_GRAPH_RUNTIME_DIR}"
cp "${FALCON_PROBE_GRAPH_CACHE_DIR}/manifest.json" "${FALCON_PROBE_GRAPH_RUNTIME_DIR}/manifest.json"
find "${FALCON_PROBE_GRAPH_CACHE_DIR}" -maxdepth 1 -type f -name 'graph_dataset_*.npy' \
    -exec cp {} "${FALCON_PROBE_GRAPH_RUNTIME_DIR}/" \;

echo "Starting Qwen2-VL FALCON training"
echo "Main objective: unchanged cross-device InfoNCE; global batch 1024 = 2x512 source blocks"
echo "FALCON actor: official 100-256-residualx2 Beta policy; update steps 1,11,21,..."
echo "Reward: per-source rotating 64-case B3-safe probe loss improvement"
echo "Diagnostics: ${FALCON_DIAGNOSTICS_PATH}"

torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT_TRAIN}" --max_restarts=0 \
    train_falcon.py \
    --output_dir "${OUTPUT_DIR}" \
    --lora --lora_r "${LORA_R}" --lora_alpha "${LORA_ALPHA}" --lora_dropout "${LORA_DROPOUT}" \
    --lora_target_modules "${LORA_TARGET_MODULES}" \
    --model_name "${MODEL_NAME_OR_PATH}" \
    --bf16 --pooling eos --normalize True \
    --qwen2_model_impl b3 \
    --qwen2_b3_attention_impl flash_attention_2 \
    --qwen2_b3_legacy_position_ids False \
    --temperature 0.02 \
    --dataloader_num_workers 0 \
    --dataset_config "${DATASET_CONFIG}" \
    --negative_strategy vlm2vec_inbatch \
    --embedding_gradient_strategy info_nce \
    --pos_only True --grad_cache True \
    --per_device_train_batch_size "${BATCH_SIZE_PER_DEVICE}" \
    --gc_q_chunk_size "${GC_Q_CHUNK_SIZE}" \
    --gc_p_chunk_size "${GC_P_CHUNK_SIZE}" \
    --gc_dynamic_limit "${GC_DYNAMIC_LIMIT}" \
    --optim adamw_torch --lr_scheduler_type linear \
    --learning_rate "${LEARNING_RATE}" \
    --max_steps "${MAX_STEPS}" --warmup_steps "${WARMUP_STEPS}" \
    --save_steps "${SAVE_STEPS}" --logging_steps 1 \
    --save_safetensors False --remove_unused_columns False \
    --resume_from "${RESUME_FROM}" --seed "${SEED}" \
    --gradient_accumulation_steps 1 \
    --resize_use_processor True \
    --resize_min_pixels "${RESIZE_MIN_PIXELS}" \
    --resize_max_pixels "${RESIZE_MAX_PIXELS}" \
    --interleave_batch_size "${INTERLEAVE_BATCH_SIZE}" \
    --interleave_stopping_strategy all_exhausted \
    --ddp_timeout 14400 --ddp_broadcast_buffers False \
    --ddp_find_unused_paramet