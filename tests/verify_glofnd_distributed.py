"""Run: PYTHONPATH=. torchrun --standalone --nproc_per_node=2 tests/verify_glofnd_distributed.py"""

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.training.glofnd import GloFNDDistributedLoss
from src.training.glofnd_core import (
    GloFNDHyperparameters,
    PerAnchorGloFNDState,
    glofnd_filtered_info_nce,
)


def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 2:
        raise RuntimeError(f"This verifier requires exactly two ranks, got {world_size}")

    generator = torch.Generator().manual_seed(173)
    query_all = F.normalize(torch.randn(6, 7, generator=generator), dim=1)
    target_all = F.normalize(torch.randn(6, 7, generator=generator), dim=1)
    local_start = rank * 3
    local_stop = local_start + 3
    query_local = query_all[local_start:local_stop].clone().requires_grad_()
    target_local = target_all[local_start:local_stop].clone().requires_grad_()
    local_ids = torch.arange(local_start, local_stop)

    hyperparameters = GloFNDHyperparameters(
        alpha=0.2,
        threshold_lr=0.05,
        beta1=0.9,
        beta2=0.98,
        eps=1e-8,
    )
    distributed_state = PerAnchorGloFNDState(6, hyperparameters)
    distributed_loss = GloFNDDistributedLoss(
        temperature=0.07,
        state=distributed_state,
        log_path=f"/tmp/glofnd_distributed_verify_{os.getpid()}.jsonl",
        debug_interval=1,
    )
    distributed_loss.set_batch_context(1, local_ids)
    scaled_loss = distributed_loss(query_local, target_local)
    scaled_loss.backward()

    reference_query = query_all.clone().requires_grad_()
    reference_target = target_all.clone().requires_grad_()
    reference_state = PerAnchorGloFNDState(6, hyperparameters)
    reference_loss, _ = glofnd_filtered_info_nce(
        reference_query,
        reference_target,
        torch.arange(6),
        reference_state,
        0.07,
    )
    (reference_loss * world_size).backward()

    torch.testing.assert_close(scaled_loss / world_size, reference_loss)
    torch.testing.assert_close(
        query_local.grad, reference_query.grad[local_start:local_stop]
    )
    torch.testing.assert_close(
        target_local.grad, reference_target.grad[local_start:local_stop]
    )
    for name in (
        "threshold",
        "first_moment",
        "second_moment",
        "visit_count",
        "initialized",
    ):
        torch.testing.assert_close(
            getattr(distributed_state, name), getattr(reference_state, name)
        )

    dist.barrier()
    if rank == 0:
        print(
            "[GLOFND_DISTRIBUTED_VERIFY_OK] "
            "global_id_order=True state_sync=True loss_scaling=True gradients=True",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
