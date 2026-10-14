from dataclasses import dataclass, field
from transformers import TrainingArguments
from typing import List


@dataclass
class ModelArguments:
    model_name: str = field(metadata={"help": "huggingface model name or path"})
    model_type: str = field(default=None, metadata={"help": "model type, typically includes in config file, but sometimes needs mannually add"})
    processor_name: str = field(default=None, metadata={"help": "processor_name, huggingface model name or path"})
    model_backbone: str = field(default=None, metadata={"help": "HF model type"})
    checkpoint_path: str = field(default=None, metadata={"help": "a local model path, could be a LoRA version"})
    pooling: str = field(default='last', metadata={"help": "pooling method for encoder"})
    normalize: bool = field(default=False, metadata={"help": "normalize query and passage representations"})
    temperature: float = field(default=0.02, metadata={"help": "temperature for softmax"})
    lora: bool = field(default=False, metadata={"help": "do parameter-efficient fine-tuning with lora"})
    lora_r: int = field(default=16, metadata={"help": "lora r"})
    lora_alpha: int = field(default=64, metadata={"help": "lora alpha"})
    lora_dropout: float = field(default=0.1, metadata={"help": "lora dropout"})
    lora_target_modules: str = field(default="qkv_proj,o_proj,gate_up_proj,down_proj,k_proj,q_proj,out_proj,v_proj", metadata={"help": "lora target modules"})
    moe_lora: bool = field(
        default=False,
        metadata={
            "help": (
                "Use the isolated TSEmbed dense softmax MoE-DoRA architecture. "
                "This flag does not enable EANS or two-stage training."
            )
        },
    )
    moe_num_experts: int = field(
        default=4,
        metadata={"help": "Number of dense DoRA experts for --moe_lora."},
    )
    moe_router_temperature: float = field(
        default=1.0,
        metadata={
            "help": (
                "Softmax temperature for the MoE-DoRA router. The TSEmbed "
                "paper denotes this as tau' but does not report a value."
            )
        },
    )
    qwen3_5_bidirectional_attention: bool = field(
        default=False,
        metadata={
            "help": (
                "Isolated Qwen3.5 ablation: make checkpoint-backed standard "
                "full-attention layers bidirectional. The hybrid model's "
                "Gated DeltaNet layers remain causal. Intended for mean pooling."
            )
        },
    )
    qwen3_5_prompt_profile: str = field(
        default="legacy",
        metadata={
            "help": (
                "Qwen3.5 prompt profile. 'legacy' preserves all existing "
                "training/evaluation behavior; 'embedding_v2' enables the "
                "isolated role-aware final-token embedding prompt experiment."
            )
        },
    )
    qwen2_model_impl: str = field(
        default="native",
        metadata={
            "help": (
                "Qwen2-VL model implementation: 'native' uses Transformers "
                "5.9, while 'b3' uses the repository's B3-compatible "
                "Transformers-4.48-era implementation."
            )
        },
    )
    qwen2_b3_attention_impl: str = field(
        default="flash_attention_2",
        metadata={
            "help": (
                "Attention backend for the repository B3 Qwen2-VL model. "
                "The B3-compatible default is 'flash_attention_2'; 'sdpa' "
                "and 'eager' remain available for controlled diagnostics."
            )
        },
    )
    qwen2_b3_legacy_position_ids: bool = field(
        default=False,
        metadata={
            "help": (
                "Experimental replay-safe approximation of B3's stateful "
                "training-time position-id behavior. This is not exact and "
                "should remain disabled for the Transformers 5.9 baseline."
            )
        },
    )
    qwen2_embedding_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Isolated Qwen2-VL experiment: register one trainable special "
                "token, append it as the final active token of every query and "
                "target, and use the existing last-token pooling path."
            )
        },
    )
    qwen2_embedding_token_text: str = field(
        default="<embedding>",
        metadata={
            "help": (
                "Special-token spelling for --qwen2_embedding_token. The "
                "training checkpoint persists and validates this exact value."
            )
        },
    )
    qwen2_eval_token_pruning: bool = field(
        default=False,
        metadata={
            "help": (
                "Evaluation-only Qwen2-VL experiment: after decoder layer 0, "
                "retain tokens ranked highest by the last token's layer-0 "
                "attention. Disabled by default and rejected during training."
            )
        },
    )
    qwen2_eval_token_keep_ratio: float = field(
        default=0.1,
        metadata={"help": "Fraction of active tokens retained by Qwen2 evaluation pruning."},
    )
    qwen2_eval_token_prune_short_text_threshold: int = field(
        default=16,
        metadata={
            "help": (
                "Do not prune pure-text evaluation inputs with at most this "
                "many active tokens. Visual inputs are not exempt."
            )
        },
    )
    qwen2_eval_token_prune_min_tokens: int = field(
        default=2,
        metadata={
            "help": (
                "Minimum retained tokens for a pruned non-empty input, "
                "including the forced final pooling token."
            )
        },
    )
    qwen2_eval_token_prune_debug_interval: int = field(
        default=50,
        metadata={"help": "Print Qwen2 pruning diagnostics every N inference batches."},
    )
    num_crops: int = field(default=16, metadata={"help": "number of crops used in image encoder"})
    uigraph_use: bool = field(default=False, metadata={"help": "Enable ui graph for token selection"})
    uigraph_diff: int = field(default=1, metadata={"help": "Pixel difference used for constructing ui graph for token selection"})
    uigraph_rand: bool = field(default=False, metadata={"help": "Enable random graph construction for token selection"})
    uimask_ratio: float = field(default=0.5, metadata={"help": "Specify the percentage of patch tokens to skip per component for token selection"})
    uimask_rand: bool = field(default=False, metadata={"help": "Enable random token selection instead of uniform selection"})
    lm_skip_layer: str = field(default='[1,28,0]', metadata={"help": "Specify the layers of the language model to skip for token selection"})
    vis_skip_layer: str = field(default='[1,32,0]', metadata={"help": "Specify the layers of the vision model to skip for token selection"})
    #! new args
    init_lora_model: bool = field(default=False, metadata={"help": "initializing with lora model"})
    init_lora_base_model: str = field(
        default=None,
        metadata={
            "help": (
                "Optional local backbone override when loading an existing "
                "LoRA checkpoint. Used by offline teacher preprocessing."
            )
        },
    )


@dataclass
class DataArguments:
    dataset_config: str = field(default=None, metadata={"help": "yaml file with dataset configuration"})
    dataset_name: str = field(default=None, metadata={"help": "huggingface dataset name"})
    subset_name: List[str] = field(default=None, metadata={"help": "Useful for datasets with subsets"})
    dataset_split: str = field(default='train', metadata={"help": "dataset split"})
    num_sample_per_subset: int = field(default=None, metadata={"help": "number of training samples per subset"})
    image_dir: str = field(default=None, metadata={"help": "Image directory path"})
    encode_output_path: str = field(default=None, metadata={"help": "encode output path"})
    max_len: int = field(default=None, metadata={"help": "The maximum total input sequence length after tokenization. Use with caution, since it may truncate text prompts due to large image lengths."},)
    embedding_type: str = field(default="", metadata={"help": "embedding type"})
    image_resolution: str = field(default=None, metadata={"help": "for models i.e. LLaVA-next and Qwen, resize images first, none means using original image resolution. This is only works when `--resize_use_processor false`."})
    resize_use_processor: bool = field(default=False, metadata={"help": "Resize visual inputs insides processor, e.g. Qwen2VLImageProcessor, instead of by our code."})
    resize_min_pixels: int = field(default=28*28*4, metadata={"help": "The min pixels of the image to resize the image. This is only works when `--resize_use_processor true`."})
    resize_max_pixels: int = field(default=28*28*1280, metadata={"help": "The max pixels of the image to resize the image. This is only works when `--resize_use_processor true`."})
    image_decay_factor: float = field(default=None, metadata={"help": "The image decay factor for resizing temporal images"})
    num_hardneg: int = field(default=0, metadata={"help": "hard negative number"})
    #! new args
    sdibn: bool = field(default=False, metadata={"help": "huggingface model name"})
    odibn: bool = field(default=False, metadata={"help": "huggingface model name"})
    rdibn: bool = field(default=False, metadata={"help": "huggingface model name"})
    tgt_prefix_mod: bool = field(default=False, metadata={"help": "Modify the pos_prefix"})
    chunk_size: int = field(default=32, metadata={"help": "Cluster sizes in metis. Only used in odibn"})
    #!new args 2
    eval_dataset_name: str = field(default=None, metadata={"help": "Useful for datasets with subsets"})
    eval_subset_name: List[str] = field(default=None, metadata={"help": "Useful for datasets with subsets"})
    eval_image_dir: str = field(default=None, metadata={"help": "Eval Image directory path"})
    pos_only: bool = field(default=False, metadata={"help": "Only use positives"})
    negative_strategy: str = field(
        default="b3",
        metadata={
            "help": (
                "Training-negative construction. 'b3' preserves the existing "
                "SDIBN/ODIBN behavior; 'vlm2vec_inbatch' uses one positive per "
                "query and ordinary cross-device in-batch negatives; 'unime_v2' "
                "uses UniME-V2's per-query soft distribution over one positive "
                "and eight teacher-scored hard negatives."
            )
        },
    )
    b3_ablation: str = field(
        default="none",
        metadata={
            "help": (
                "Isolated Qwen2-VL B3 ablation contract. Supported values are "
                "'none', 'random_batch', 'no_fn_filter', and "
                "'no_instance_hn'. The default leaves every existing training "
                "and evaluation path unchanged."
            )
        },
    )
    unime_v2_data_path: str = field(
        default=None,
        metadata={"help": "Path to UniME-V2's train_data_*_scores.json file."},
    )
    unime_v2_image_dir: str = field(
        default=None,
        metadata={"help": "Root directory for image paths stored in UniME-V2 JSON."},
    )
    unime_v2_num_hard_negatives: int = field(
        default=8,
        metadata={"help": "Number of ordered UniME-V2 teacher-scored hard negatives."},
    )


@dataclass
class TrainingArguments(TrainingArguments):
    image_encoder_freeze: bool = field(default=False, metadata={"help": "huggingface model name"})
    output_dir: str = field(default=None, metadata={"help": "directory for saving trained models"})
    # Current Transformers no longer exposes this CLI field, while B3 and this project's
    # custom Trainer still use it when saving PEFT checkpoints. Keep it as a
    # project-level compatibility argument.
    save_safetensors: bool = field(
        default=False,
        metadata={"help": "Save adapter checkpoints with safetensors."},
    )
    resume_from: str = field(default="none", metadata={"help": "`auto` will detect if any previous checkpoints should be resumed. or specify specific step of the checkpoint."})
    project_name: str = field(default=None, metadata={"help": "project name"})
    logging_steps: int = field(default=1, metadata={"help": "logging steps"})
    num_train_epochs: int = field(default=1, metadata={"help": "number of training epochs"})
    grad_cache: bool = field(default=False, metadata={"help": "Use gradient cache update"})
    gc_q_chunk_size: int = field(default=128, metadata={"help": "query side subset size. Should be power of 2"})
    gc_p_chunk_size: int = field(default=128, metadata={"help": "target side subset size. Should be power of 2"})
    interleave_stopping_strategy: str = field(default="all_exhausted", metadata={"help": "all_exhausted or first_exhausted"})
    interleave_batch_size: float = field(default=0, metadata={"help": "Specify mini-batch size to interleave data from multi-sources, 0/None means random sampling by examples, 1 means full batch."})
    #!new args
    gc_dynamic_limit: int = field(default=125, metadata={"help": "gc_chunk default limit - (128, 125) sized matrices works for Qwen2b. gc_dynamic_limit would be 125 and gc_p|q_chunk_size would be 128"})
    embedding_gradient_strategy: str = field(
        default="info_nce",
        metadata={
            "help": (
                "Embedding-gradient rule used by GradCache. 'info_nce' keeps "
                "the existing autograd path; 'ega' enables Explicit Gradient "
                "Amplification for ordinary one-positive batches; 'b3_ega' "
                "applies its rectangular extension to B3's one-positive plus "
                "five-hard-negative layout; 'llave' enables the official bidirectional "
                "detached hardness-weighted contrastive objective."
            )
        },
    )
    ega_alpha: float = field(
        default=20.0,
        metadata={"help": "EGA hard-negative amplification coefficient."},
    )
    llave_alpha: float = field(
        default=9.0,
        metadata={
            "help": (
                "LLaVE negative hardness coefficient. The paper uses alpha=9; "
                "the superscript after 9 in the PDF is a footnote marker."
            )
        },
    )


@dataclass
class MTEBArguments:
    device: str = field(default="cuda", metadata={"help": "use cuda for single GPU inference, if multiple GPUs are available it will use DP automatically"})
    batch_size_per_device: int = field(default=16, metadata={"help": ""})
    max_length: int = field(default=512, metadata={"help": ""})
    eval_output_dir: str = field(default=None, metadata={"help": "directory for saving trained models"})
    task_types: List[str] = field(default=None, metadata={"help": ""})
    tasks: List[str] = field(default=None, metadata={"help": ""})
    prompt_family: List[str] = field(default=None, metadata={"help": ""})
