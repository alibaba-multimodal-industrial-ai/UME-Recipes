from dataclasses import dataclass, field
from transformers import TrainingArguments
from typing import List


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "huggingface model name or path"}
    )
    normalize: bool = field(
        default=True, 
        metadata={"help": "normalize query and passage representations"}
    )
    instruction: str = field(
        default="Represent the user's input.", 
        metadata={"help": "default instruction for the model"}
    )

@dataclass
class DataArguments:
    dataset_config: str = field(
        default=None, 
        metadata={"help": "yaml file with dataset configuration"}
    )
    data_basedir: str = field(
        default=None, 
        metadata={"help": "Expect an absolute path to the base directory of all datasets. If set, it will be prepended to each dataset path"}
    )
    encode_output_path: str = field(
        default=None, 
        metadata={"help": "encode output path"}
    )
    # (optional)
    rerank_output_path: str = field(
        default=None,
        metadata={"help": "Where to save rerank results. Default: `data_args.encode_output_path`/rerank_output"}
    )

@dataclass
class EvalArguments(TrainingArguments):
    pass

@dataclass
class RerankArguments:
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path or HF repo id for Qwen3-VL reranker."}
    )
    instruction: str = field(
        default="Given a search query, retrieve relevant candidates that answer the query.",
        metadata={"help": "Instruction passed to reranker."}
    )

    # topk setting
    topk: int = field(default=100, metadata={"help": "TopK from embedding retrieval to rerank."})
