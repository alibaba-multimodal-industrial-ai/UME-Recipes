"""Two-process check of EGA gather order and GradCache DDP scaling.

Run with:
  torchrun --standalone --nproc_per_node=2 tests/verify_ega_distributed.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.training.grad_cache.grad_cache import GradCache
from src.training.loss import DistributedContrastiveLoss


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
    hidden_size = 9
    generator = torch.Generator(device="cpu").manual_seed(20250814)
    global_query = F.normalize(
        torch.randn(world_size * local_batch, hidden_size, generator=generator), dim=1
    )
    global_target = F.normalize(
        torch.randn(world_size * local_batch, hidden_size, generator=generator), dim=1
    )
    start = rank * local_batch
    stop = start + local_batch
    local_query = global_query[start:stop].to(device)
    local_target = global_target[start:stop].to(device)

    loss_fn = DistributedContrastiveLoss(temperature=0.02)
    loss_fn.expected_target_per_query = 1
    common = dict(
        models=[torch.nn.Identity(), torch.nn.Identity()],
        chunk_sizes=[4, 4],
        loss_fn=loss_fn,
        dynamic_limit=1,
    )
    standard = GradCache(**common)
    ega = GradCache(
        **common,
        embedding_gradient_strategy="ega",
        ega_alpha=0.0,
    )
    standard_cache, standard_loss = standard.build_cache(local_query, local_target)
    ega_cache, ega_loss = ega.build_cache(local_query, local_target)

    torch.testing.assert_close(ega_loss, standard_loss, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(
        ega_cache[0].float(), standard_cache[0].float(), rtol=3e-5, atol=3e-6
    )
    torch.testing.assert_close(
        ega_cache[1].float(), standard_cache[1].float(), rtol=3e-5, atol=3e-6
    )
    dist.barrier()
    if rank == 0:
        print(
            "[EGA_DISTRIBUTED_PREFLIGHT_OK] world_size=2 "
            "alpha_zero_matches_distributed_info_nce=True",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
