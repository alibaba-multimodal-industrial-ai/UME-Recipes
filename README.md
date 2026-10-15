# UME-Recipes: What Really Works for Universal Multimodal Embeddings?

[📝 Paper](https://arxiv.org/abs/2610.15035) | [🤗 Hugging Face](https://huggingface.co/alibaba-multimodal-industrial-ai/UME-Recipes)

Official training and evaluation recipes for **What Really Works for Universal Multimodal Embeddings? A Controlled Study of Contrastive Training Strategies**.

UME-Recipes provides controlled comparisons of contrastive training strategies using aligned backbones, training data, optimization budgets, and MMEB evaluation.

| | |
|---|---|
| Backbones | Qwen2-VL-2B, Qwen2-VL-7B, and Qwen3.5-2B |
| Benchmark | MMEB, covering 36 image tasks |
| Focus | False-negative mitigation, hard-negative mining, batch construction, and gradient modulation |

---

## Supported training recipes

| Method | Qwen2-VL-2B | Qwen2-VL-7B | Qwen3.5-2B |
|---|:---:|:---:|:---:|
| VLM2Vec | ✓ | ✓ | ✓ |
| B3 | ✓ | ✓ | ✓ |
| EGA | ✓ | ✓ | ✓ |
| B3 + EGA | ✓ | ✓ | ✓ |
| Online false-negative mitigation | ✓ | ✓ | ✓ |
| HOBIT | ✓ | ✓ | ✓ |
| UniME-V2 | ✓ | ✓ | ✓ |
| GloFND / LLaVE / FALCON | ✓ | – | – |
| B3 ablations | ✓ | – | – |

---

## Repository structure

```text
scripts/       training and evaluation entry points
configs/       dataset configurations
src/           models, data, losses, trainers, and evaluation code
build/         preprocessing and cache builders
verify/        preflight checks
tests/         correctness tests
MMEB-train/    precomputed B3 metadata shards
```

Model weights, images, ordinary MMEB pairs, UniME-V2 teacher-score data, checkpoints, and logs are not included.

---

## Installation

The reference environment uses Python 3.11+, PyTorch 2.8, and Transformers 5.9.0.

```bash
git clone https://github.com/alibaba-multimodal-industrial-ai/UME-Recipes.git
cd UME-Recipes

uv sync
source .venv/bin/activate
```

Alternatively:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

FlashAttention-2 is recommended. Qwen3.5 also benefits from compatible `flash-linear-attention` and `causal-conv1d` kernels.

---

## Data layout

```text
models/
├── Qwen2-VL-2B-Instruct/
├── Qwen2-VL-7B-Instruct/
└── Qwen3.5-2B/

data/
├── MMEB_V1/
├── MMEB-eval-cache/
├── vlm2vec_eval/image-tasks/
└── UniME-V2-Training-Datasets/train_data_Qwen25VL_7B_scores.json

MMEB-train/
├── <ordinary MMEB training subsets>
└── <precomputed B3 subsets included in this repository>
```

The checked-in `MMEB-train/` contains B3 metadata shards but not images or ordinary MMEB subsets. Obtain those resources under their original licenses and place them at the paths above, or update the YAML files in `configs/`.

---

## Training

Run commands from the repository root. The default controlled recipes use 8 GPUs, a global batch size of 1,024, LoRA/DoRA rank 16, 2,000 steps, last-token pooling, and the same image-resolution range.

### VLM2Vec

```bash
bash scripts/run_train_qwen2_inbatch.sh
bash scripts/run_train_qwen2_7b_inbatch.sh
bash scripts/run_train_qwen3_5_inbatch.sh
```

The historical `inbatch` script names denote the VLM2Vec training recipe.

### B3

```bash
bash scripts/run_train_qwen2_B3.sh
bash scripts/run_train_qwen2_7b_B3.sh
bash scripts/run_train_qwen3_5_B3.sh
```

Other methods follow the same naming pattern in `scripts/`, for example:

```bash
bash scripts/run_train_qwen2_ega.sh
bash scripts/run_train_qwen2_b3_ega.sh
bash scripts/run_train_qwen2_glofnd.sh
bash scripts/run_train_qwen2_hobit.sh
bash scripts/run_train_qwen2_unime_v2.sh
```

Override paths or hyperparameters with environment variables:

```bash
MODEL_NAME_OR_PATH=/path/to/model \
OUTPUT_DIR=./exp/training/my_run \
MAX_STEPS=2000 \
bash scripts/run_train_qwen2_inbatch.sh
```

---

## MMEB evaluation

The evaluation scripts load a checkpoint once and evaluate all 36 MMEB image tasks.

```bash
bash scripts/eval_all_qwen2_vl.sh \
  ./models/Qwen2-VL-2B-Instruct \
  ./exp/training/my_run/checkpoint-2000 \
  ./exp/evaluation/my_run
```

```bash
bash scripts/eval_all_qwen3_5.sh \
  ./models/Qwen3.5-2B \
  ./exp/training/my_run/checkpoint-2000 \
  ./exp/evaluation/my_run
```

Set `DATASET_CACHE`, `IMAGE_DIR`, `BATCH_SIZE`, or `CUDA_VISIBLE_DEVICES` when the defaults do not match your environment. Results are summarized in `details.tsv` and `summary.tsv`.

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{li2026what,
  title={What Really Works for Universal Multimodal Embeddings? A Controlled Study of Contrastive Training Strategies},
  author={Li, Duo and Pan, Haojun and Yang, Zuhao and Wang, Xintong and Huo, Chengfu and Lu, Shijian and Ding, Liang},
  journal={arXiv preprint arXiv:2610.15035},
  year={2026}
}
```

## License

The code is released under the [Apache License 2.0](LICENSE). Dataset and model licenses remain with their respective owners.
