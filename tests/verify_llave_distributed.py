"""Two-process LLaVE cross-device loss and GradCache scaling preflight.

Run with:
  torchrun --standalone --nproc_per_node=2 tests/verify_llave_distributed.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.training.grad_cache.grad_cache import GradCache
from src.training.loss import LLaVEDistributedContrastiveLoss


def official_weighted_clip_reference(similarity, labels, temperature, alpha):
    logit_scale = 1.0 / temperature

    def one_direction(scores):
        logits = logit_scale * scores
        reward = torch.log(
            torch.exp(alpha * logits.detach() / logit_scale)
        )
        reward.scatter_(1, labels[:, None], 0.0)
        return F.cross_entropy(logits + reward, labels)

    return (one_direction(similarity) + one_direction(similarity.T)) / 2


def main():
    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError(f"This preflight requires exactly two ranks, got {world_size}")
    if use_cuda:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    local_batch = 4
    hidden_size = 11
    temperature = 0.02
    alpha = 9.0
    generator = torch.Generator(device="cpu").manual_seed(20260903)
    global_query = F.normalize(
        torch.randn(world_size * local_batch, hidden_size, generator=generator), dim=1
    )
    global_target = F.normalize(
        torch.randn(world_size * local_batch, hidden_size, generator=generator), dim=1
    )
    labels = torch.arange(world_size * local_batch)
    reference_q = global_query.clone().requires_grad_(True)
    reference_t = global_target.clone().requires_grad_(True)
    reference_loss = official_weighted_clip_reference(
        reference_q @ reference_t.T,
        labels,
        temperature,
        alpha,
    )
    reference_loss.backward()

    start = rank * local_batch
    stop = start + local_batch
    local_query = global_query[start:stop].to(device)
    local_target = global_target[start:stop].to(device)
    loss_fn = LLaVEDistributedContrastiveLoss(
        temperature=temperature,
        alpha=alpha,
    )
    loss_fn.expected_target_per_query = 1
    cache = GradCache(
        models=[torch.nn.Identity(), torch.nn.Identity()],
        chunk_sizes=[4, 4],
        loss_fn=loss_fn,
        dynamic_limit=1,
        embedding_gradient_strategy="llave",
    )
    cached_gradients, distributed_loss = cache.build_cache(local_query, local_target)

    torch.testing.assert_close(
        distributed_loss / world_size,
        reference_loss.detach().to(device),
        rtol=2e-5,
        atol=2e-6,
    )
    torch.testing.assert_close(
        cached_gradients[0].float() / world_size,
        reference_q.grad[start:stop].to(device),
        rtol=3e-5,
        atol=3e-6,
    )
    torch.testing.assert_close(
        cached_gradients[1].float() / world_size,
        reference_t.grad[start:stop].to(device),
        rtol=3e-5,
        atol=3e-6,
    )
    dist.barrier()
    if rank == 0:
        print(
            "[LLAVE_DISTRIBUTED_PREFLIGHT_OK] world_size=2 "
            "global_labels_rank_ordered=True global_loss=True "
            "query_and_target_gradients=True",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
