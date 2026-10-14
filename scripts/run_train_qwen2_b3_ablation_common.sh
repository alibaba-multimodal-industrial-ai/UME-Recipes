#!/bin/bash
set -euo pipefail

ABLATION="${1:?usage: scripts/run_train_qwen2_b3_ablation_common.sh random_batch|no_fn_filter|no_instance_hn}"
case "${ABLATION}" in
  random_batch|no_fn_filter|no_instance_hn) ;;
  *) echo "Unknown B3 ablation: ${ABLATION}" >&2; exit 2 ;;
esac

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${HF_HOME:-/tmp/ume-recipes/huggingface}"
export WANDB_MODE=disabled
export B3_DEBUG="${B3_DEBUG:-1}"
export B3_DEBUG_INTERVAL="${B3_DEBUG_INTERVAL:-1}"
export B3_DDP_TRACE_STEPS="${B3_DDP_TRACE_STEPS:-0}"
export B3_NUMERIC_DEBUG_STEPS="${B3_NUMERIC_DEBUG_STEPS:-0}"
export B3_FINGERPRINT="${B3_FINGERPRINT:-0}"
export B3_FINGERPRINT_CALLS="${B3_FINGERPRINT_CALLS:-20}"
export B3_FINGERPRINT_RANKS="${B3_FINGERPRINT_RANKS:-0}"

NUM_GPUS="${NUM_GPUS:-8}"
MASTER_PORT="${MASTER_PORT:-2215}"
MASTER_PORT_PREPROCESS="${MASTER_PORT_PREPROCESS:-2216}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-./models/Qwen2-VL-2B-Instruct}"
BATCH_SIZE_PER_DEVICE="${BATCH_SIZE_PER_DEVICE:-128}"
MAX_STEPS="${MAX_STEPS:-2000}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
SAVE_STEPS="${SAVE_STEPS:-100}"
RESUME_FROM="${RESUME_FROM:-none}"
GC_Q_CHUNK_SIZE="${GC_Q_CHUNK_SIZE:-128}"
GC_P_CHUNK_SIZE="${GC_P_CHUNK_SIZE:-128}"
GC_DYNAMIC_LIMIT="${GC_DYNAMIC_LIMIT:-64}"
CHUNK_SIZE=32
LORA_R="${LORA_R:-8}"
BASELINE_CONFIG="configs/data_configs/mmeb_new/mmeb20_HNPS_bs16bi_30.130_30P.10.5_70.170_qwen2b.yaml"
SOURCE_CONFIG="${B3_NO_FN_SOURCE_CONFIG:-configs/data_configs/mmeb_new/mmeb20.yaml}"

SCRIPT_DIR=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
cd "${SCRIPT_DIR}"

case "${ABLATION}" in
  random_batch)
    OUTPUT_DIR="${OUTPUT_DIR:-./exp/training/b3_qwen2b_ablation_random_batch}"
    DATASET_CONFIG="${DATASET_CONFIG:-${BASELINE_CONFIG}}"
    SAMPLER_ARGS=(--sdibn)
    EXTRA_DATA_ARGS=()
    ;;
  no_instance_hn)
    OUTPUT_DIR="${OUTPUT_DIR:-./exp/training/b3_qwen2b_ablation_no_instance_hn}"
    DATASET_CONFIG="${DATASET_CONFIG:-${BASELINE_CONFIG}}"
    SAMPLER_ARGS=(--sdibn --odibn)
    EXTRA_DATA_ARGS=(--pos_only True)
    ;;
  no_fn_filter)
    OUTPUT_DIR="${OUTPUT_DIR:-./exp/training/b3_qwen2b_ablation_no_fn_filter}"
    NO_FN_CACHE_ROOT="${NO_FN_CACHE_ROOT:-./exp/cache/b3_qwen2_no_fn_filter/MMEB-train}"
    DATASET_CONFIG="${DATASET_CONFIG:-./exp/cache/b3_qwen2_no_fn_filter/mmeb20_b3_no_fn_filter.yaml}"
    B3_ROOT="${B3_ROOT:-./B3}"
    B3_NO_FN_TEACHER_MODEL="${B3_NO_FN_TEACHER_MODEL:-./models/VLM2Vec-V2.0}"
    B3_NO_FN_TEACHER_BASE_MODEL="${B3_NO_FN_TEACHER_BASE_MODEL:-./models/Qwen2-VL-2B-Instruct}"
    B3_NO_FN_VECTOR_CACHE="${B3_NO_FN_VECTOR_CACHE:-./exp/cache/b3_qwen2_no_fn_teacher_vectors_vlm2vec_v2_b3raw_p3136_1003520}"
    B3_NO_FN_VECTOR_STAGE="${B3_NO_FN_VECTOR_STAGE:-/tmp/b3_qwen2_no_fn_teacher_vectors_build_$$}"
    B3_NO_FN_VECTOR_RUNTIME="${B3_NO_FN_VECTOR_RUNTIME:-/tmp/b3_qwen2_no_fn_teacher_vectors_runtime_$$}"
    B3_NO_FN_RANK_CACHE="${B3_NO_FN_RANK_CACHE:-./exp/cache/b3_qwen2_no_fn_teacher_ranks_vlm2vec_v2_b3raw_p3136_1003520}"
    B3_NO_FN_RANK_STAGE="${B3_NO_FN_RANK_STAGE:-/tmp/b3_qwen2_no_fn_rank_build}"
    B3_NO_FN_RANK_RUNTIME="${B3_NO_FN_RANK_RUNTIME:-/tmp/b3_qwen2_no_fn_rank_runtime_$$}"
    B3_NO_FN_ENCODE_BATCH_SIZE="${B3_NO_FN_ENCODE_BATCH_SIZE:-32}"
    B3_NO_FN_RANK_QUERY_CHUNK="${B3_NO_FN_RANK_QUERY_CHUNK:-128}"
    B3_NO_FN_RESIZE_MIN_PIXELS="${B3_NO_FN_RESIZE_MIN_PIXELS:-3136}"
    B3_NO_FN_RESIZE_MAX_PIXELS="${B3_NO_FN_RESIZE_MAX_PIXELS:-1003520}"

    B3_NO_FN_FINAL_READY=0
    if [[ -f "${DATASET_CONFIG}" ]] && \
       python ${SCRIPT_DIR}/verify/verify_b3_ablation_inputs.py \
         --ablation no_fn_filter \
         --dataset-config "${DATASET_CONFIG}" \
         --baseline-config "${BASELINE_CONFIG}"; then
      B3_NO_FN_FINAL_READY=1
      B3_ABLATION_INPUTS_PREVERIFIED=1
      echo "Reusing fully verified B3 no-FN shards; teacher vectors/ranks are not needed"
    fi

    if [[ "${B3_NO_FN_FINAL_READY}" != "1" ]]; then
      if [[ -z "${B3_GENERAL_RANK_ROOT:-}" || -z "${B3_CLASSIFICATION_RANK_ROOT:-}" ]]; then
      B3_NO_FN_VECTOR_BUILT_LOCAL=0
      B3_NO_FN_ACTIVE_VECTOR_ROOT=""
      if [[ ! -f "${B3_NO_FN_RANK_CACHE}/manifest.json" ]]; then
        if [[ ! -f "${B3_NO_FN_VECTOR_CACHE}/manifest.json" ]]; then
          if [[ ! -f "${B3_NO_FN_TEACHER_MODEL}/adapter_config.json" ]]; then
            echo "B3 no-FN teacher adapter is missing: ${B3_NO_FN_TEACHER_MODEL}" >&2
            exit 1
          fi
          echo "B3 no-FN teacher vectors are absent; encoding the 20 raw sources once"
          export DHBG_CACHE_DIR="${B3_NO_FN_VECTOR_CACHE}"
          export DHBG_CACHE_STAGE_DIR="${B3_NO_FN_VECTOR_STAGE}"
          export DHBG_ENCODE_BATCH_SIZE="${B3_NO_FN_ENCODE_BATCH_SIZE}"
          export DHBG_DEFER_SEEDS_TO_V2=1
          torchrun \
            --nproc_per_node="${NUM_GPUS}" \
            --master_port="${MASTER_PORT_PREPROCESS}" \
            --max_restarts=0 \
            ${SCRIPT_DIR}/build/build_dynamic_hard_batch_cache.py \
            --output_dir /tmp/b3_qwen2_no_fn_teacher_builder_state \
            --model_name "${B3_NO_FN_TEACHER_MODEL}" \
            --processor_name "${B3_NO_FN_TEACHER_BASE_MODEL}" \
            --init_lora_model True \
            --init_lora_base_model "${B3_NO_FN_TEACHER_BASE_MODEL}" \
            --bf16 \
            --pooling eos \
            --normalize True \
            --qwen2_model_impl b3 \
            --qwen2_b3_attention_impl flash_attention_2 \
            --qwen2_b3_legacy_position_ids False \
            --temperature 0.02 \
            --dataloader_num_workers 0 \
            --dataset_config "${SOURCE_CONFIG}" \
            --negative_strategy b3 \
            --embedding_gradient_strategy info_nce \
            --pos_only True \
            --per_device_train_batch_size "${BATCH_SIZE_PER_DEVICE}" \
            --max_steps "${MAX_STEPS}" \
            --seed 42 \
            --resize_use_processor True \
            --resize_min_pixels "${B3_NO_FN_RESIZE_MIN_PIXELS}" \
            --resize_max_pixels "${B3_NO_FN_RESIZE_MAX_PIXELS}" \
            --interleave_batch_size 128 \
            --interleave_stopping_strategy all_exhausted \
            --ddp_timeout 14400
          # The fresh stage already contains the two final embedding arrays.
          # Consume it directly instead of reading the newly published ~6 GB
          # back from OSS/FUSE into a second local copy.
          cp "${B3_NO_FN_VECTOR_CACHE}/manifest.json" "${B3_NO_FN_VECTOR_STAGE}/manifest.json"
          B3_NO_FN_ACTIVE_VECTOR_ROOT="${B3_NO_FN_VECTOR_STAGE}"
          B3_NO_FN_VECTOR_BUILT_LOCAL=1
        else
          echo "Reusing B3 no-FN teacher vectors: ${B3_NO_FN_VECTOR_CACHE}"
        fi
      fi

      if [[ ! -f "${B3_NO_FN_RANK_CACHE}/manifest.json" ]]; then
        if [[ -z "${B3_NO_FN_ACTIVE_VECTOR_ROOT}" ]]; then
          echo "Staging teacher vectors on node-local storage for exact chunked Top-k"
          mkdir -p "${B3_NO_FN_VECTOR_RUNTIME}"
          cp "${B3_NO_FN_VECTOR_CACHE}/manifest.json" "${B3_NO_FN_VECTOR_RUNTIME}/manifest.json"
          cp "${B3_NO_FN_VECTOR_CACHE}/query_embeddings.npy" "${B3_NO_FN_VECTOR_RUNTIME}/query_embeddings.npy"
          cp "${B3_NO_FN_VECTOR_CACHE}/positive_embeddings.npy" "${B3_NO_FN_VECTOR_RUNTIME}/positive_embeddings.npy"
          B3_NO_FN_ACTIVE_VECTOR_ROOT="${B3_NO_FN_VECTOR_RUNTIME}"
        else
          echo "Using freshly encoded node-local teacher vectors: ${B3_NO_FN_ACTIVE_VECTOR_ROOT}"
        fi
        torchrun \
          --nproc_per_node="${NUM_GPUS}" \
          --master_port="${MASTER_PORT_PREPROCESS}" \
          --max_restarts=0 \
          ${SCRIPT_DIR}/build/build_b3_no_fn_rank_cache.py \
          --source-cache "${B3_NO_FN_ACTIVE_VECTOR_ROOT}" \
          --output-cache "${B3_NO_FN_RANK_CACHE}" \
          --stage-dir "${B3_NO_FN_RANK_STAGE}" \
          --source-config "${SOURCE_CONFIG}" \
          --baseline-config "${BASELINE_CONFIG}" \
          --query-chunk "${B3_NO_FN_RANK_QUERY_CHUNK}" \
          --resize-min-pixels "${B3_NO_FN_RESIZE_MIN_PIXELS}" \
          --resize-max-pixels "${B3_NO_FN_RESIZE_MAX_PIXELS}"
        if [[ "${B3_NO_FN_KEEP_LOCAL_VECTORS:-0}" != "1" && "${B3_NO_FN_ACTIVE_VECTOR_ROOT}" == "${B3_NO_FN_VECTOR_RUNTIME}" && "${B3_NO_FN_VECTOR_RUNTIME}" == /tmp/* ]]; then
          rm -f \
            "${B3_NO_FN_VECTOR_RUNTIME}/manifest.json" \
            "${B3_NO_FN_VECTOR_RUNTIME}/query_embeddings.npy" \
            "${B3_NO_FN_VECTOR_RUNTIME}/positive_embeddings.npy"
          rmdir "${B3_NO_FN_VECTOR_RUNTIME}" 2>/dev/null || true
        fi
        if [[ "${B3_NO_FN_KEEP_LOCAL_VECTORS:-0}" != "1" && "${B3_NO_FN_VECTOR_BUILT_LOCAL}" == "1" ]]; then
          case "${B3_NO_FN_VECTOR_STAGE}" in
            /tmp/b3_qwen2_no_fn_teacher_vectors_build_*)
              rm -rf -- "${B3_NO_FN_VECTOR_STAGE}"
              echo "Removed completed temporary teacher-vector stage: ${B3_NO_FN_VECTOR_STAGE}"
              ;;
            *)
              echo "Keeping nonstandard teacher-vector stage path: ${B3_NO_FN_VECTOR_STAGE}"
              ;;
          esac
        fi
      else
        echo "Reusing B3 no-FN teacher ranks: ${B3_NO_FN_RANK_CACHE}"
      fi

      if [[ -f "${B3_NO_FN_RANK_STAGE}/manifest.json" ]] && \
         cmp -s "${B3_NO_FN_RANK_STAGE}/manifest.json" "${B3_NO_FN_RANK_CACHE}/manifest.json"; then
        B3_NO_FN_ACTIVE_RANK_ROOT="${B3_NO_FN_RANK_STAGE}"
        echo "Using node-local freshly built rank cache: ${B3_NO_FN_ACTIVE_RANK_ROOT}"
      else
        echo "Staging compact teacher ranks on node-local storage"
        mkdir -p "${B3_NO_FN_RANK_RUNTIME}"
        cp "${B3_NO_FN_RANK_CACHE}/manifest.json" "${B3_NO_FN_RANK_RUNTIME}/manifest.json"
        for dataset_dir in "${B3_NO_FN_RANK_CACHE}"/*; do
          [[ -d "${dataset_dir}" ]] || continue
          dataset_name=$(basename "${dataset_dir}")
          mkdir -p "${B3_NO_FN_RANK_RUNTIME}/${dataset_name}"
          cp "${dataset_dir}"/*.npy "${B3_NO_FN_RANK_RUNTIME}/${dataset_name}/"
        done
        B3_NO_FN_ACTIVE_RANK_ROOT="${B3_NO_FN_RANK_RUNTIME}"
      fi
      B3_GENERAL_RANK_ROOT="${B3_GENERAL_RANK_ROOT:-${B3_NO_FN_ACTIVE_RANK_ROOT}}"
      B3_CLASSIFICATION_RANK_ROOT="${B3_CLASSIFICATION_RANK_ROOT:-${B3_NO_FN_ACTIVE_RANK_ROOT}}"
      fi
      BUILD_ARGS=(
        --b3-root "${B3_ROOT}"
        --output-root "${NO_FN_CACHE_ROOT}"
        --baseline-config "${BASELINE_CONFIG}"
        --config-output "${DATASET_CONFIG}"
      )
      BUILD_ARGS+=(--general-rank-root "${B3_GENERAL_RANK_ROOT}")
      BUILD_ARGS+=(--classification-rank-root "${B3_CLASSIFICATION_RANK_ROOT}")
      if [[ -n "${B3_SOURCE_ROOT:-}" ]]; then
        BUILD_ARGS+=(--source-root "${B3_SOURCE_ROOT}")
      fi
      python ${SCRIPT_DIR}/build/build_b3_no_fn_filter_shards.py "${BUILD_ARGS[@]}"
      if [[ "${B3_NO_FN_KEEP_LOCAL_RANKS:-0}" != "1" ]]; then
        case "${B3_NO_FN_ACTIVE_RANK_ROOT:-}" in
          /tmp/b3_qwen2_no_fn_rank_build|/tmp/b3_qwen2_no_fn_rank_runtime_*)
            rm -rf -- "${B3_NO_FN_ACTIVE_RANK_ROOT}"
            echo "Removed completed temporary teacher-rank stage: ${B3_NO_FN_ACTIVE_RANK_ROOT}"
            ;;
        esac
      fi
    fi
    SAMPLER_ARGS=(--sdibn --odibn)
    EXTRA_DATA_ARGS=()
    ;;
esac

if [[ ! -f "${DATASET_CONFIG}" ]]; then
  echo "B3 ablation dataset config is missing: ${DATASET_CONFIG}" >&2
  exit 1
fi

if [[ "${B3_ABLATION_INPUTS_PREVERIFIED:-0}" != "1" ]]; then
  python ${SCRIPT_DIR}/verify/verify_b3_ablation_inputs.py \
    --ablation "${ABLATION}" \
    --dataset-config "${DATASET_CONFIG}" \
    --baseline-config "${BASELINE_CONFIG}"
fi

LOG_DIR="${LOG_DIR:-./logs}"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_qwen2_b3_ablation_${ABLATION}_${TIMESTAMP}.log"

echo "Starting isolated Qwen2-VL B3 ablation"
echo "Ablation: ${ABLATION}"
echo "Model: ${MODEL_NAME_OR_PATH}"
echo "Dataset config: ${DATASET_CONFIG}"
echo "Output: ${OUTPUT_DIR}"
echo "Global query batch: $((NUM_GPUS * BATCH_SIZE_PER_DEVICE))"
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
  --dataset_config "${DATASET_CONFIG}" \
  --negative_strategy b3 \
  --b3_ablation "${ABLATION}" \
  --embedding_gradient_strategy info_nce \
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
  --resize_use_processor True \
  --interleave_batch_size 1 \
  --ddp_timeout 14400 \
  --ddp_broadcast_buffers False \
  --ddp_find_unused_parameters False \
  --ignore_data_skip true \
  "${SAMPLER_ARGS[@]}" \
  "${EXTRA_DATA_ARGS[@]}" \
  --chunk_size "${CHUNK_SIZE}" \
  > "${LOG_FILE}" 2>&1 &

PID=$!
echo "Training started in background with PID: ${PID}"
echo "Monitor with: tail -f ${LOG_FILE}"
