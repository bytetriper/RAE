
import os
import platform
import torch
import torch.distributed as dist
from typing import Tuple


def default_backend() -> str:
    """NCCL is Linux-only; Windows builds of PyTorch ship gloo instead."""
    return "gloo" if platform.system() == "Windows" else "nccl"


def setup_distributed() -> Tuple[int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(backend=os.environ.get("DIST_BACKEND", default_backend()))
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world_size, device


class SingleProcessWrapper(torch.nn.Module):
    """Transparent stand-in for DDP, exposing the `.module` attribute that
    callers rely on. Used when only one process is taking part, where DDP adds
    gradient synchronisation that has nothing to synchronise with.

    On Windows this is not just an optimisation: gloo is the only backend
    available there, and it cannot run collectives on CUDA tensors, so DDP's
    gradient all-reduce crashes the process.
    """

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()