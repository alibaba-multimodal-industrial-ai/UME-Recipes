import os
import time
import json
import yaml
import random
import pickle
import logging
import numpy as np
import torch
import torch.distributed as dist

from datetime import timedelta
from tqdm.auto import tqdm
from transformers import HfArgumentParser

from datasets import concatenate_datasets
from datasets.distributed import split_dataset_by_node

from .arguments import RerankArguments, DataArguments, EvalArguments
from .utils.basic_utils import print_rank, print_master
from .utils.eval_utils.metrics import RankingMetrics
from .data.datasets.base_eval_dataset import AutoEvalPairDataset, generate_cand_dataset

from ...models.qwen3_vl_reranker import Qwen3VLReranker

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def pad_dataset_to_divisible(dataset, world_size):
    num_samples = len(dataset)
    if num_samples % world_size == 0:
        return dataset, num_samples

    num_to_add = world_size - (num_samples % world_size)
    padded_size = num_samples + num_to_add

    padding_data = dataset.select([i % len(dataset) for i in range(num_to_add)])
    padded_dataset = concatenate_datasets([dataset, padding_data])
    return padded_dataset, padded_size

def build_corpus_lookup(cand_dataset):
    """
    Build mapping: cand_id(str) -> candidate_sample(dict)
    """
    lookup = {}

    if cand_dataset is not None:
        for i in range(len(cand_dataset)):
            it = cand_dataset[i]
            lookup[it["dataset_infos"]["cand_name"]] = it["cand_input"]

    return lookup


def load_topk_from_pred(pred_path: str, topk: int):
    """
    Load embedding retrieval TopK from *_pred.jsonl written by eval_embedding.
    """
    topk_ids = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            pred = obj.get("prediction", [])
            topk_ids.append(pred[:topk])
    return topk_ids

@torch.no_grad()
def rerank_topk_for_queries(
    reranker: Qwen3VLReranker,
    query_dataset,
    cand_lookup: dict,
    batch_size: int,
    full_dataset_len: int,
):
    """
    For each query, rerank its retrieved TopK candidates by Qwen3VLReranker.
    Return pred_dicts compatible with RankingMetrics.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    pred_dicts = []

    for query_idx, query_item in enumerate(tqdm(
        query_dataset, desc=f"Reranking (rank {rank})",
        disable=local_rank > 0, ncols=120,
    )):
        q_sample = query_item["query_input"]
        instruction = q_sample.pop("instruction", None)
        
        gt_dids = query_item["dataset_infos"]["label_name"]
        gt_dids = gt_dids if isinstance(gt_dids, list) else [gt_dids]
        rel_scores = query_item["dataset_infos"].get("rel_scores", None)

        topk_ids = query_item["topk_ids"]

        pred_dict = {
            "label": gt_dids,
            "rel_scores": rel_scores,
            "retrieved": topk_ids,
            "prediction": [],
            "rerank_scores": [],
        }
        pred_dicts.append(pred_dict)

        if len(topk_ids) == 0:
            print_rank(f"In query {query_idx}, cannot find any retrieved results.")
            continue

        # build docs mm contents
        docs = []
        valid_cand_ids = []
        for did in topk_ids:
            d_sample = cand_lookup.get(did, None)
            if d_sample is None:
                # if miss, skip it (avoid crashing)
                print_rank(f"In query {query_idx}, missing candidate for {did}")
                continue
            docs.append(d_sample)
            valid_cand_ids.append(did)

        if len(valid_cand_ids) == 0:
            print_rank(f"In query {query_idx}, cannot find any valid retrieved results.")
            continue
        
        # score docs with batching
        rerank_scores = []
        for s in range(0, len(docs), batch_size):
            batch_docs = docs[s:s + batch_size]
            inputs = {
                "instruction": instruction,
                "query": q_sample,
                "documents": batch_docs,
            }
            batch_scores = reranker.process(inputs)  # list[float]
            rerank_scores.extend(batch_scores)

        # sort by reranker score desc
        order = np.argsort(-np.asarray(rerank_scores)).tolist()
        reranked_ids = [valid_cand_ids[i] for i in order]
        reranked_scores = [rerank_scores[i] for i in order]

        pred_dict["prediction"] = reranked_ids
        pred_dict["rerank_scores"] = reranked_scores

    if dist.is_initialized():
        all_pred_dicts = [None for _ in range(world_size)]
        dist.all_gather_object(all_pred_dicts, pred_dicts)

        if rank == 0:
            final_pred_dicts = [ 
                pred_dict 
                for node_pred_dicts in all_pred_dicts 
                    for pred_dict in node_pred_dicts
            ]
            return final_pred_dicts[:full_dataset_len]
        else:
            return None
    else:
        return pred_dicts


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if "RANK" in os.environ and dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    print_master("=== Distributed Setup Initialized (Reranker Eval) ===")
    print_master(f"Master -> ADDR: {os.environ.get('MASTER_ADDR')}, PORT: {os.environ.get('MASTER_PORT')}")
    print_master(f"World Size: {world_size}")
    if torch.cuda.is_available():
        print_rank(f"Rank: {rank}, Local Rank: {local_rank} on {torch.cuda.get_device_name()}")
    
    parser = HfArgumentParser((RerankArguments, DataArguments, EvalArguments))
    model_args, data_args, eval_args = parser.parse_args_into_dataclasses()

    output_dir = data_args.rerank_output_path or \
        os.path.join(data_args.encode_output_path, 'rerank_output')
    os.makedirs(output_dir, exist_ok=True)

    # -------- Load reranker model (DDP-safe download) --------
    if rank == 0:
        print_master(f"[rank=0] Loading reranker from: {model_args.model_name_or_path}")
        reranker = Qwen3VLReranker(
            model_args.model_name_or_path,
            default_instruction=model_args.instruction,
            attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16,
        )

    if dist.is_initialized():
        dist.barrier()

    if rank != 0:
        print_rank("Loading reranker from cache...")
        time.sleep(random.randint(2 * rank, 3 * rank))
        reranker = Qwen3VLReranker(
            model_args.model_name_or_path,
            default_instruction=model_args.instruction,
            torch_dtype=torch.bfloat16,
        )

    with open(data_args.dataset_config, 'r') as yaml_file:
        dataset_configs = yaml.safe_load(yaml_file)

    # -------- Main loop over datasets --------
    for dataset_name, task_config in dataset_configs.items():
        if dist.is_initialized():
            dist.barrier()
        print_master(f"\n--- Reranker Evaluating {dataset_name} ---")

        # 0. Skip if already exists and valid
        score_path = os.path.join(output_dir, f"{dataset_name}_rerank_score.json")
        pred_path = os.path.join(output_dir, f"{dataset_name}_rerank_pred.jsonl")
        if os.path.exists(score_path) and os.path.exists(pred_path):
            try:
                with open(score_path, "r") as f:
                    sd = json.load(f)
                if "num_pred" in sd:
                    print_master(f"[{dataset_name}] Rerank results exist. Skip.")
                    formatted = {k: f"{v:.4f}" for k, v in sd.items() if isinstance(v, (int, float))}
                    print_master(f"Scores: {formatted}")
                    if dist.is_initialized():
                        dist.barrier()
                    continue
            except Exception as e:
                print_master(f"[{dataset_name}] Existing rerank cache corrupted ({e}), recompute...")

        # 1. Get retrieved TopK from embedding pred.jsonl
        try:
            embed_pred_path = os.path.join(data_args.encode_output_path, f"{dataset_name}_pred.jsonl")
            topk_ids = load_topk_from_pred(embed_pred_path, model_args.topk)
        except Exception as e:
            print_master(f"[{dataset_name}] Failed to read {embed_pred_path} ({e}), skip.")
            continue

        # 2. Load dataset again to get raw content for reranker
        if data_args.data_basedir is not None:
            for key in ["image_root", "video_root", "frame_root", "clip_root", "data_path"]:
                if task_config.get(key):
                    task_config[key] = os.path.join(data_args.data_basedir, task_config[key])
        try:
            full_eval_qry_dataset, corpus = AutoEvalPairDataset.instantiate(
                model_args=model_args, data_args=data_args, **task_config
            )
            full_eval_cand_dataset = generate_cand_dataset(full_eval_qry_dataset, corpus)
            full_eval_qry_dataset = full_eval_qry_dataset.add_column("topk_ids", topk_ids)
            cand_lookup = build_corpus_lookup(full_eval_cand_dataset)

            if dist.is_initialized():
                padded_qry_dataset, _ = pad_dataset_to_divisible(full_eval_qry_dataset, world_size)
                eval_qry_dataset = split_dataset_by_node(padded_qry_dataset, rank=rank, world_size=world_size)
            else:
                padded_qry_dataset = full_eval_qry_dataset
                eval_qry_dataset = full_eval_qry_dataset

        except Exception as e:
            print_rank(f"Failed to load dataset {dataset_name}: {e}")
            raise

        # 3. Get rerank results
        pred_dicts = rerank_topk_for_queries(
            reranker=reranker,
            query_dataset=eval_qry_dataset,
            cand_lookup=cand_lookup,
            batch_size=eval_args.per_device_eval_batch_size,
            full_dataset_len=len(full_eval_qry_dataset)
        )

        # -------- metric + save (rank 0 only) --------
        if rank == 0:
            metrics_to_report = task_config.get("metrics", ["hit", "ndcg", "precision", "recall", "f1", "map", "mrr"])
            metrics = RankingMetrics(metrics_to_report)
            score_dict = metrics.evaluate(pred_dicts)
            score_dict["num_pred"] = len(pred_dicts)
            score_dict["num_data"] = len(pred_dicts)
            score_dict["topk"] = model_args.topk
            score_dict["reranker_model"] = model_args.model_name_or_path

            with open(score_path, "w") as f:
                json.dump(score_dict, f, indent=4, ensure_ascii=False)

            with open(pred_path, "w", encoding="utf-8") as f:
                for pred in pred_dicts:
                    f.write(json.dumps(pred, ensure_ascii=False) + "\n")

            formatted = {k: f"{v:.4f}" for k, v in score_dict.items() if isinstance(v, (int, float))}
            print_master(f"[{dataset_name}] Final Rerank Score: {formatted}")
            print_master(f"Saved: {score_path}")
            print_master(f"Saved: {pred_path}")
            

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
