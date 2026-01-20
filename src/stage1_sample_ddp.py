# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Runs distributed reconstructions with a pre-trained stage-1 model.
Inputs are loaded from an ImageFolder dataset, processed with center crops,
and the reconstructed images are saved as .png files alongside a packed .npz.
"""
import argparse
import math
import os
import sys
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
from PIL import Image
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import numpy as np

from sample_ddp import create_npz_from_sample_folder
from stage1 import RAE
from utils.model_utils import instantiate_from_config
from utils.train_utils import parse_configs


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class IndexedImageFolder(ImageFolder):
    """ImageFolder that also returns the dataset index."""

    def __getitem__(self, index):
        image, _ = super().__getitem__(index)
        return image, index


def sanitize_component(component: str) -> str:
    """Replace OS separators to keep path components valid."""
    return component.replace(os.sep, "-")


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Sampling with DDP requires at least one GPU.")

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.set_grad_enabled(False)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device_idx = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_idx)
    device = torch.device("cuda", device_idx)

    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if rank == 0:
        print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    use_bf16 = args.precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested bf16 precision, but the current CUDA device does not support bfloat16.")
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
    dataset = IndexedImageFolder(args.data_path, transform=transform)
    total_available = len(dataset)
    if total_available == 0:
        raise ValueError(f"No images found at {args.data_path}.")

    requested = total_available if args.num_samples is None else min(args.num_samples, total_available)
    if requested <= 0:
        raise ValueError("Number of samples to process must be positive.")

    selected_indices = list(range(requested))
    rank_indices = selected_indices[rank::world_size]
    subset = Subset(dataset, rank_indices)

    if rank == 0:
        os.makedirs(args.sample_dir, exist_ok=True)

    model_target = rae_config.get("target", "stage1")
    ckpt_path = rae_config.get("ckpt")
    ckpt_name = "pretrained" if not ckpt_path else os.path.splitext(os.path.basename(str(ckpt_path)))[0]
    folder_components: List[str] = [
        sanitize_component(str(model_target).split(".")[-1]),
        sanitize_component(ckpt_name),
        f"bs{args.per_proc_batch_size}",
        args.precision,
    ]
    folder_name = "-".join(folder_components)
    possible_folder_name = os.environ.get('SAVE_FOLDER', None)
    if possible_folder_name:
        folder_name = possible_folder_name
    sample_folder_dir = os.path.join(args.sample_dir, folder_name)
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving reconstructed samples at {sample_folder_dir}")
    dist.barrier()
    loader = DataLoader(
        subset,
        batch_size=args.per_proc_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    local_total = len(rank_indices)
    iterator = tqdm(loader, desc="Stage1 recon", total=math.ceil(local_total / args.per_proc_batch_size)) if rank == 0 else loader

    with torch.inference_mode():
        for images, indices in iterator:
            if images.numel() == 0:
                continue
            images = images.to(device, non_blocking=True)
            with autocast(**autocast_kwargs):
                latents = rae.encode(images)
                recon = rae.decode(latents)
            recon = recon.clamp(0, 1)
            recon_np = recon.mul(255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            indices_list = indices.tolist() if hasattr(indices, "tolist") else list(indices)
            for sample, idx in zip(recon_np, indices_list):
                Image.fromarray(sample).save(f"{sample_folder_dir}/{idx:06d}.png")

    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, requested)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to an ImageFolder directory with input images.")
    parser.add_argument("--sample-dir", type=str, default="samples", help="Directory to store reconstructed samples.")
    parser.add_argument("--per-proc-batch-size", type=int, default=4, help="Number of images processed per GPU step.")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to reconstruct (defaults to full dataset).")
    parser.add_argument("--image-size", type=int, default=256, help="Target crop size before feeding images to the model.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers per process.")
    parser.add_argument("--global-seed", type=int, default=0, help="Base seed for RNG (adjusted per rank).")
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="fp32", help="Autocast precision mode.")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable TF32 matmuls (Ampere+). Disable if deterministic results are required.")
    args = parser.parse_args()
    main(args)
