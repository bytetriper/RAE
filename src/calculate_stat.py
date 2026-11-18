#!/usr/bin/env python3
"""
Distributed latent mean/var estimation for a pre-trained stage-1 RAE.

- Pure PyTorch DDP (nccl), NO torchxla.
- Uses your custom LayerNormwStatistics as default for 3D/4D latents:
    (B,C)           -> BatchNorm1d(C)
    (B,C,L)         -> LayerNormwStatistics((C,L))
    (B,C,H,W)       -> LayerNormwStatistics((C,H,W))
    (B,C,D,H,W)     -> BatchNorm3d(C)

Output format matches your RAE __init__:
  torch.save({"mean": mean_tensor_cpu, "var": var_tensor_cpu}, normalization_stat_path)
"""

import argparse
import math
import os
import sys
from typing import List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm import tqdm

from stage1 import RAE
from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs
from utils.dist_utils import default_backend


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


class IndexedImageFolder(ImageFolder):
    """ImageFolder that also returns the dataset index."""
    def __getitem__(self, index):
        image, _ = super().__getitem__(index)
        return image, index


def sanitize_component(component: str) -> str:
    return component.replace(os.sep, "-")


class LayerNormwStatistics(nn.Module):
    """
    BN-like running-statistics accumulator over batch dim only.

    Input: x of shape (B, *input_shape) where input_shape = (C, ...) typically.
    - train: updates running_mean/running_var using batch stats over dim=0
    - eval : uses running stats
    """
    def __init__(self, input_shape, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.input_shape = tuple(input_shape)

        # kept for compatibility; not required for stats-only
        self.gamma = nn.Parameter(torch.ones((1, *self.input_shape)))
        self.beta = nn.Parameter(torch.zeros((1, *self.input_shape)))

        self.register_buffer("running_mean", torch.zeros(*self.input_shape))
        self.register_buffer("running_var", torch.ones(*self.input_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 1 + len(self.input_shape):
            raise ValueError(f"Expected x shape (B, {self.input_shape}), got {tuple(x.shape)}")

        if self.training:
            mean = x.mean(dim=0, keepdim=False)
            var = x.var(dim=0, keepdim=False, unbiased=False)
            m = self.momentum
            self.running_mean.mul_(1 - m).add_(mean, alpha=m)
            self.running_var.mul_(1 - m).add_(var, alpha=m)
        else:
            mean = self.running_mean
            var = self.running_var

        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_normalized + self.beta


def _extract_latents(rae: RAE, images: torch.Tensor) -> torch.Tensor:
    """
    Supports both:
      - rae.encode(images) -> Tensor
      - rae.encode(images) -> object with .zs
    """
    z = rae.encode(images)
    if torch.is_tensor(z):
        return z
    if hasattr(z, "zs"):
        return z.zs
    raise TypeError(f"Unknown encode() return type: {type(z)} (expected Tensor or object with .zs)")


def _make_bn_for_latents(latents: torch.Tensor, eps: float = 1e-5, momentum: float = 0.1) -> nn.Module:
    nd = latents.ndim
    if nd == 2:
        return nn.BatchNorm1d(latents.shape[1], affine=False, track_running_stats=True, eps=eps, momentum=momentum)
    if nd == 3:
        return LayerNormwStatistics(latents.shape[1:], eps=eps, momentum=momentum)
    if nd == 4:
        return LayerNormwStatistics(latents.shape[1:], eps=eps, momentum=momentum)
    if nd == 5:
        return nn.BatchNorm3d(latents.shape[1], affine=False, track_running_stats=True, eps=eps, momentum=momentum)
    raise ValueError(f"Unsupported latent shape {tuple(latents.shape)} (ndim={nd}); expected 2/3/4/5D.")


@torch.no_grad()
def _sync_mean_var_across_ranks(
    mean_r: torch.Tensor,
    var_r: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Standard BatchNorm-style synchronization.

    Given per-rank running stats:
        mean_r = E_r[x]
        var_r  = Var_r[x]   (population variance, unbiased=False)

    Compute global stats:
        mean = E_r[mean_r]
        var  = E_r[var_r] + Var_r(mean_r)

    Uses only all_reduce (no all_gather).
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        # With a single rank the reduction below is the identity.
        return mean_r, var_r

    world_size = dist.get_world_size()

    # work in float32 for numerical stability
    mean = mean_r.float()
    var = var_r.float()

    # sums
    mean_sum = mean.clone()
    mean_sq_sum = mean * mean
    var_sum = var.clone()

    dist.all_reduce(mean_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(mean_sq_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(var_sum, op=dist.ReduceOp.SUM)

    mean_global = mean_sum / world_size
    mean_sq_global = mean_sq_sum / world_size
    var_of_mean = mean_sq_global - mean_global * mean_global

    var_global = var_sum / world_size + var_of_mean
    var_global = torch.clamp(var_global, min=0.0)

    return mean_global, var_global



def _get_running_stats(module: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
    if hasattr(module, "running_mean") and hasattr(module, "running_var"):
        return module.running_mean.detach(), module.running_var.detach()
    raise AttributeError(f"Module {type(module).__name__} does not have running_mean/running_var.")


def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This script assumes CUDA + nccl DDP.")

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.set_grad_enabled(False)

    dist.init_process_group(default_backend())
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    device_idx = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_idx)
    device = torch.device("cuda", device_idx)

    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    use_bf16 = args.precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested bf16 precision, but this CUDA device does not support bfloat16.")
    autocast_kwargs = dict(dtype=torch.bfloat16, enabled=use_bf16)

    rae_config, *_ = parse_configs(args.config)
    if rae_config is None:
        raise ValueError("Config must provide a stage_1 section.")
    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval()

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
    ])
    #### Data init
    first_crop_size = 384 if args.image_size == 256 else int(args.image_size * 1.5)
    stage1_transform = transforms.Compose(
        [
            transforms.Resize(first_crop_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(args.image_size),
            transforms.ToTensor(),
        ]
    )    
    stage2_transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    dataset = IndexedImageFolder(args.data_path, transform=stage2_transform)
    total_available = len(dataset)
    if total_available == 0:
        raise ValueError(f"No images found at {args.data_path}.")

    requested = total_available if args.num_samples is None else min(args.num_samples, total_available)
    if requested <= 0:
        raise ValueError("Number of samples to process must be positive.")
    base_ds = dataset if requested == total_available else Subset(dataset, list(range(requested)))

    sampler = DistributedSampler(
        base_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True, # very important
        drop_last=False,
    )

    loader = DataLoader(
        base_ds,
        batch_size=args.per_proc_batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
    )
    # selected_indices = list(range(requested))
    # rank_indices = selected_indices[rank::world_size]
    # subset = Subset(dataset, rank_indices)

    if rank == 0:
        os.makedirs(args.sample_dir, exist_ok=True)

    # folder naming (same spirit as your recon ddp script) + SAVE_FOLDER override
    model_target = rae_config.get("target", "stage1")
    ckpt_path = rae_config.get("ckpt")
    ckpt_name = "pretrained" if not ckpt_path else os.path.splitext(os.path.basename(str(ckpt_path)))[0]
    folder_components: List[str] = [
        sanitize_component(str(model_target).split(".")[-1]),
        sanitize_component(ckpt_name),
        f"bs{args.per_proc_batch_size}",
        args.precision,
    ]
    base_folder = "-".join(folder_components)
    folder_name = os.environ.get("SAVE_FOLDER", base_folder)
    out_dir = os.path.join(args.sample_dir, folder_name)

    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
        print(f"[init] world_size={world_size}  requested_samples={requested}")
        print(f"[path] saving stats under: {out_dir}")
    dist.barrier()

    # loader = DataLoader(
    #     subset,
    #     batch_size=args.per_proc_batch_size,
    #     shuffle=False,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    #     persistent_workers=True,
    #     prefetch_factor=2,   # start small; increase only if stable
    # )

    # probe latent shape (rank-local) to build stats module
    first_batch = next(iter(loader), None)
    if first_batch is None:
        raise RuntimeError("Empty loader on this rank (shouldn't happen if requested>0).")
    images0, _ = first_batch
    images0 = images0.to(device, non_blocking=True)
    with autocast(**autocast_kwargs):
        z0 = _extract_latents(rae, images0)

    stats_mod = _make_bn_for_latents(z0, eps=args.eps, momentum=args.momentum).to(device)
    stats_mod.train()

    if rank == 0:
        print(f"[stats_mod] module={type(stats_mod).__name__}  latent_shape={tuple(z0.shape)}")
    dist.barrier()

    # accumulate running stats
    local_total = len(loader)
    for _ in range(1):
        iterator = tqdm(
        loader,
        desc="Latent stats",
        total=len(loader),
        disable=(rank != 0),
        )
        for images, _indices in iterator:
            images = images.to(device, non_blocking=False)
            with autocast(**autocast_kwargs):
                z = _extract_latents(rae, images)
            _ = stats_mod(z)  # updates running stats
            torch.cuda.synchronize()
    # sync across ranks
    stats_mod.eval()
    mean_r, var_r = _get_running_stats(stats_mod)
    mean, var = _sync_mean_var_across_ranks(mean_r, var_r)
    #mean, var = mean_r, var_r
    # save exactly what RAE reads: mean & var only
    dist.barrier()
    if rank == 0:
        out_path = os.path.join(out_dir, "normalization_stats.pt")
        payload = {
            "mean": mean.cpu(),
            "var": var.cpu(),
        }
        torch.save(payload, out_path)
        print(f"[done] wrote: {out_path}")
        print(f"[mean] shape={tuple(payload['mean'].shape)} dtype={payload['mean'].dtype}")
        print(f"[var ] shape={tuple(payload['var'].shape)} dtype={payload['var'].dtype}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to an ImageFolder directory with input images.")
    parser.add_argument("--sample-dir", type=str, default="stats/", help="Base directory to store stats output.")
    parser.add_argument("--per-proc-batch-size", type=int, default=256, help="Images processed per GPU step.")
    parser.add_argument("--num-samples", type=int, default=None, help="How many images to use (default: all).")
    parser.add_argument("--image-size", type=int, default=256, help="Center crop size before feeding the model.")
    parser.add_argument("--num-workers", type=int, default=64, help="Dataloader workers per process.")
    parser.add_argument("--global-seed", type=int, default=0, help="Base seed for RNG (adjusted per rank).")
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="fp32", help="Autocast precision.")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True, help="Enable TF32 matmuls.")
    parser.add_argument("--eps", type=float, default=1e-5, help="Stats eps.")
    parser.add_argument("--momentum", type=float, default=0.1, help="Running stats momentum.")
    args = parser.parse_args()
    main(args)
