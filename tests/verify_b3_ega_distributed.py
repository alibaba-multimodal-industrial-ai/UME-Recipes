"""Two-process check of rectangular B3+EGA gather order and DDP scaling.

Run with:
  torchrun --standalone --nproc_per_node=2 tests/verify_b3_ega_distributed.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.training.grad_cache.grad_cache import GradCache
from src.training.ega import b3_ega_gradients_and_loss
from src.training.loss import DistributedContrastiveLoss


def main():
    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError(
            f"This preflight requires exactly two ranks, got {world_size}"
        )
    if use_cuda:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    local_queries = 4
    target_per_query = 6
    hidden_size = 11
    generator = torch.Generator(device="cpu").manual_seed(20260904)
    global_query = F.normalize(
        torch.randn(
            world_size * local_queries, hidden_size, generator=generator
        ),
        dim=1,
    )
    global_target = F.normalize(
        torch.randn(
            world_size * local_queries * target_per_query,
            hidden_size,
            generator=generator,
        ),
        dim=1,
    )
    query_start = rank * local_queries
    query_stop = query_start + local_queries
    target_start = rank * local_queries * target_per_query
    target_stop = target_start + local_queries * target_per_query
    local_query = global_query[query_start:query_stop].to(device)
    local_target = global_target[target_start:target_stop].to(device)

    loss_fn = DistributedContrastiveLoss(temperature=0.02)
    loss_fn.expected_target_per_query = target_per_query
    common = dict(
        models=[torch.nn.Identity(), torch.nn.Identity()],
        chunk_sizes=[4, 4],
        loss_fn=loss_fn,
        dynamic_limit=1,
    )
    standard = GradCache(**common)
    b3_ega_alpha_zero = GradCache(
        **common,
        embedding_gradient_strategy="b3_ega",
        ega_alpha=0.0,
    )
    standard_cache, standard_loss = standard.build_cache(
        local_query, local_target
    )
    b3_ega_cache, b3_ega_loss = b3_ega_alpha_zero.build_cache(
        local_query, local_target
    )

    torch.testing.assert_close(
        b3_ega_loss, standard_loss, rtol=2e-5, atol=2e-6
    )
    torch.testing.assert_close(
        b3_ega_cache[0].float(),
        standard_cache[0].float(),
        rtol=3e-5,
        atol=3e-6,
    )
    torch.testing.assert_close(
        b3_ega_cache[1].float(),
        standard_cache[1].float(),
        rtol=3e-5,
        atol=3e-6,
    )

    # Exercise the actual alpha=20 path as well. Comparing its distributed
    # rank-local caches to one centralized global computation verifies gather
    # order and the asymmetric B-versus-6B cache slicing.
    b3_ega_alpha_twenty = GradCache(
        **common,
        embedding_gradient_strategy="b3_ega",
        ega_alpha=20.0,
    )
    amplified_cache, amplified_reported_loss = (
        b3_ega_alpha_twenty.build_cache(local_query, local_target)
    )
    expected_query, expected_target, expected_mean_loss, _ = (
        b3_ega_gradients_and_loss(
            global_query.to(device),
            global_target.to(device),
            target_per_query=target_per_query,
            temperature=0.02,
            alpha=20.0,
        )
    )
    expected_query = (
        expected_query[query_start:query_stop] / local_queries
    ).to(local_query.dtype)
    expected_target = (
        expected_target[target_start:target_stop] / local_queries
    ).to(local_target.dtype)
    torch.testing.assert_close(
        amplified_cache[0].float(),
        expected_query.float(),
        rtol=3e-5,
        atol=3e-6,
    )
    torch.testing.assert_close(
        amplified_cache[1].float(),
        expected_target.float(),
        rtol=3e-5,
        atol=3e-6,
    )
    torch.testing.assert_close(
        amplified_reported_loss,
        expected_mean_loss * world_size,
        rtol=2e-5,
        atol=2e-6,
    )
    dist.barrier()
    if rank == 0:
        print(
            "[B3_EGA_DISTRIBUTED_PREFLIGHT_OK] world_size=2 "
            "target_per_query=6 "
            "alpha_zero_matches_distributed_b3_info_nce=True "
            "alpha_twenty_global_gather_and_slicing=True",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
