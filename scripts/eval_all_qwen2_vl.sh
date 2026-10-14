#!/bin/bash
# Evaluate a Qwen2-VL adapter trained by this repository with B3's 36 image
# tasks while keeping the training/evaluation backbone implementation aligned.

set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: bash eval_all.sh <base_model> <checkpoint> <output_dir>"
    echo "Example: bash eval_all_qwen2_vl.sh ./models/Qwen2-VL-7B-Instruct ./exp/training/b3_ega_qwen2_vl_7b_lora16/checkpoint-2000 ./exp/evaluation/qwen2-vl-7b-b3-ega-checkpoint-2000"
    exit 2
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# `datasets` creates lock files even for datasets loaded from a local
# directory. The notebook home cache can be read-only or backed by a
# filesystem that rejects file locking, so keep all Hugging Face runtime
# caches on node-local storage and share that location across torchrun ranks.
cache_uid="$(id -u)"
hf_cache_root="${B3_EVAL_HF_CACHE_ROOT:-/tmp/b3_eval_hf_cache_${cache_uid}}"
export HF_HOME="${hf_cache_root}"
export HF_DATASETS_CACHE="${hf_cache_root}/datasets"
export HF_MODULES_CACHE="${hf_cache_root}/modules"
export XDG_CACHE_HOME="${hf_cache_root}/xdg"
mkdir -p \
    "${HF_DATASETS_CACHE}" \
    "${HF_MODULES_CACHE}" \
    "${XDG_CACHE_HOME}"
if [ ! -w "${HF_DATASETS_CACHE}" ]; then
    echo "Error: Hugging Face datasets cache is not writable: ${HF_DATASETS_CACHE}"
    exit 2
fi

BASE_MODEL="$1"
CHECKPOINT="$2"
OUTPUT_DIR="$3"
IMAGE_OUTPUT_DIR="${OUTPUT_DIR}/image"

if [ ! -f "${CHECKPOINT}/adapter_config.json" ]; then
    echo "Error: adapter_config.json was not found in ${CHECKPOINT}"
    exit 2
fi
if [ ! -f "${CHECKPOINT}/adapter_model.bin" ] && [ ! -f "${CHECKPOINT}/adapter_model.safetensors" ]; then
    echo "Error: adapter weights were not found in ${CHECKPOINT}"
    exit 2
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    GPU_COUNT="${NUM_GPUS:-$(nvidia-smi --list-gpus | wc -l)}"
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT - 1)))
else
    GPU_COUNT=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
fi
export CUDA_VISIBLE_DEVICES

MASTER_PORT="${MASTER_PORT:-29500}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DATASET_CACHE="${DATASET_CACHE:-./data/MMEB-eval-cache}"
IMAGE_DIR="${IMAGE_DIR:-./data/vlm2vec_eval/image-tasks}"
RESIZE_MIN_PIXELS="${RESIZE_MIN_PIXELS:-4096}"
RESIZE_MAX_PIXELS="${RESIZE_MAX_PIXELS:-2257920}"

DATASETS=(
    ImageNet-1K N24News HatefulMemes SUN397 VOC2007 Place365 ImageNet-A
    ImageNet-R ObjectNet Country211 Wiki-SS-NQ VisDial CIRR VisualNews_t2i
    VisualNews_i2t MSCOCO_t2i MSCOCO_i2t NIGHTS WebQA OVEN FashionIQ EDIS
    OK-VQA A-OKVQA DocVQA InfographicsVQA ChartQA Visual7W ScienceQA GQA
    TextVQA VizWiz MSCOCO RefCOCO RefCOCO-Matching Visual7W-Pointing
)

echo "B3-consistent evaluation"
echo "Base model: ${BASE_MODEL}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output: ${OUTPUT_DIR}"
echo "GPUs: ${GPU_COUNT} (${CUDA_VISIBLE_DEVICES})"
echo "HF cache: ${hf_cache_root} (node-local, shared by all ranks)"

mkdir -p "${IMAGE_OUTPUT_DIR}"
echo "Loading the model once and evaluating ${#DATASETS[@]} datasets"
torchrun \
    --nproc_per_node="${GPU_COUNT}" \
    --master_port="${MASTER_PORT}" \
    --max_restarts=0 \
    eval_mmeb.py \
    --model_name "${BASE_MODEL}" \
    --checkpoint_path "${CHECKPOINT}" \
    --encode_output_path "${IMAGE_OUTPUT_DIR}" \
    --pooling eos \
    --normalize True \
    --qwen2_model_impl b3 \
    --qwen2_b3_attention_impl flash_attention_2 \
    --qwen2_b3_legacy_position_ids True \
    --lora \
    --lora_r 8 \
    --bf16 \
    --dataset_name "${DATASET_CACHE}" \
    --subset_name "${DATASETS[@]}" \
    --dataset_split test \
    --per_device_eval_batch_size "${BATCH_SIZE}" \
    --image_dir "${IMAGE_DIR}" \
    --tgt_prefix_mod \
    --model_backbone qwen2_vl \
    --resize_use_processor \
    --resize_min_pixels "${RESIZE_MIN_PIXELS}" \
    --resize_max_pixels "${RESIZE_MAX_PIXELS}"

python -m src.evaluation.mmeb_v2.gather_results \
    "${OUTPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}"

echo "Evaluation completed: ${OUTPUT_DIR}"
