# Copyright (c) Meta Platforms.
# Licensed under the MIT license.
"""
Stage-1 RAE training script with reconstruction, LPIPS, and GAN losses.

This script adapts the training logic from the Kakao Brain VQGAN trainer while
targeting the RAE autoencoder architecture used in this repository.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from glob import glob
from torchvision.utils import make_grid
from omegaconf import OmegaConf
from eval import evaluate_reconstruction_distributed
from disc import (
    DiffAug,
    LPIPS,
    build_discriminator,
    hinge_d_loss,
    vanilla_d_loss,
    vanilla_g_loss,
)
from stage1 import RAE

##### general utils
from utils import wandb_utils
from utils.model_utils import instantiate_from_config
from utils.train_utils import *
from utils.optim_utils import *
from utils.resume_utils import *
from utils.wandb_utils import *
from utils.dist_utils import *
from PIL import Image
import numpy as np

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-1 RAE with GAN and LPIPS losses.")
    parser.add_argument("--config", type=str, required=True, help="YAML config containing a stage_1 section.")
    parser.add_argument("--data-path", type=Path, required=True, help="Directory with ImageFolder structure.")
    parser.add_argument("--results-dir", type=str, default="ckpts", help="Directory to store training outputs.")
    parser.add_argument("--image-size", type=int, default=256, help="Image resolution (assumes square images).")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--global-seed", type=int, default=None, help="Override training.global_seed from the config.")    
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases for logging if set.')
    parser.add_argument("--compile", action="store_true", help="Use torch compile (for rae.encode, rae.forward).")
    return parser.parse_args()


def calculate_adaptive_weight(
    recon_loss: torch.Tensor,
    gan_loss: torch.Tensor,
    layer: torch.nn.Parameter,
    max_d_weight: float = 1e4,
) -> torch.Tensor:
    recon_grads = torch.autograd.grad(recon_loss, layer, retain_graph=True)[0]
    gan_grads = torch.autograd.grad(gan_loss, layer, retain_graph=True)[0]
    d_weight = torch.norm(recon_grads) / (torch.norm(gan_grads) + 1e-6)
    d_weight = torch.clamp(d_weight, 0.0, max_d_weight)
    return d_weight.detach()




def select_gan_losses(disc_kind: str, gen_kind: str):
    if disc_kind == "hinge":
        disc_loss_fn = hinge_d_loss
    elif disc_kind == "vanilla":
        disc_loss_fn = vanilla_d_loss
    else:
        raise ValueError(f"Unsupported discriminator loss '{disc_kind}'")

    if gen_kind == "vanilla":
        gen_loss_fn = vanilla_g_loss
    else:
        raise ValueError(f"Unsupported generator loss '{gen_kind}'")
    return disc_loss_fn, gen_loss_fn


def save_checkpoint(
    path: str,
    step: int,
    epoch: int,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    disc: torch.nn.Module,
    disc_optimizer: torch.optim.Optimizer,
    disc_scheduler: Optional[LambdaLR],
) -> None:
    state = {
        "step": step,
        "epoch": epoch,
        "model": model.module.state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "disc": disc.state_dict(),
        "disc_optimizer": disc_optimizer.state_dict(),
        "disc_scheduler": disc_scheduler.state_dict() if disc_scheduler is not None else None,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: DDP,
    ema_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    disc: torch.nn.Module,
    disc_optimizer: torch.optim.Optimizer,
    disc_scheduler: Optional[LambdaLR],
) -> Tuple[int, int]:
    checkpoint = torch.load(path, map_location="cpu")
    model.module.load_state_dict(checkpoint["model"])
    ema_model.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    disc.load_state_dict(checkpoint["disc"])
    disc_optimizer.load_state_dict(checkpoint["disc_optimizer"])
    if disc_scheduler is not None and checkpoint.get("disc_scheduler") is not None:
        disc_scheduler.load_state_dict(checkpoint["disc_scheduler"])
    return checkpoint.get("epoch", 0), checkpoint.get("step", 0)

def main():
    args = parse_args()
    #### Dist Init
    rank, world_size, device = setup_distributed()
    
    #### Config init
    full_cfg = OmegaConf.load(args.config)
    (rae_config, *_) = parse_configs(full_cfg)
    training_section = full_cfg.get("training", None)
    training_cfg = OmegaConf.to_container(training_section, resolve=True) if training_section is not None else {}
    training_cfg = dict(training_cfg) if isinstance(training_cfg, dict) else {}

    gan_section = full_cfg.get("gan", None)
    gan_cfg = OmegaConf.to_container(gan_section, resolve=True) if gan_section is not None else {}
    if not gan_cfg:
        raise ValueError("Config must define a top-level 'gan' section for stage-1 training.")
    disc_cfg = gan_cfg.get("disc", {})
    if not disc_cfg:
        raise ValueError("gan.disc configuration is required for stage-1 training.")
    loss_cfg = gan_cfg.get("loss", {})
    perceptual_weight = float(loss_cfg.get("perceptual_weight", 0.0))
    disc_weight = float(loss_cfg.get("disc_weight", 0.0))
    gan_start_epoch = int(loss_cfg.get("disc_start", 0))
    disc_update_epoch = int(loss_cfg.get("disc_upd_start", gan_start_epoch))
    lpips_start_epoch = int(loss_cfg.get("lpips_start", 0))
    
    disc_updates = int(loss_cfg.get("disc_updates", 1))
    max_d_weight = float(loss_cfg.get("max_d_weight", 1e4))
    disc_loss_type = loss_cfg.get("disc_loss", "hinge")
    gen_loss_type = loss_cfg.get("gen_loss", "vanilla")
    batch_size = int(training_cfg.get("batch_size", 16))
    global_batch_size = training_cfg.get("global_batch_size", None) # optional global batch size for override
    if global_batch_size is not None:
        global_batch_size = int(global_batch_size)
        assert global_batch_size % world_size == 0, "global_batch_size must be divisible by world_size"
        batch_size = global_batch_size // world_size
    else:
        global_batch_size = batch_size * world_size
    num_workers = int(training_cfg.get("num_workers", 4))
    clip_grad_val = training_cfg.get("clip_grad", 1.0)
    clip_grad = float(clip_grad_val) if clip_grad_val is not None else None
    if clip_grad is not None and clip_grad <= 0:
        clip_grad = None
    log_interval = int(training_cfg.get("log_interval", 100))
    sample_every = int(training_cfg.get("sample_every", 1250)) 
    checkpoint_interval = int(training_cfg.get("checkpoint_interval", 4)) # ckpt interval is epoch based
    ema_decay = float(training_cfg.get("ema_decay", 0.9999))
    num_epochs = int(training_cfg.get("epochs", 200))
    default_seed = int(training_cfg.get("global_seed", 0))
    eval_section = full_cfg.get("eval", None)
    
    if eval_section:
        do_eval = True
        eval_interval = int(eval_section.get("eval_interval", 5000))
        eval_model = eval_section.get("eval_model", False) # by default eval ema. This decides whether to **additionally** eval the non-ema model.
        eval_metrics = eval_section.get("metrics", ("rfid", "psnr", "ssim")) # by default eval all
        eval_data = eval_section.get("data_path", None)
        reference_npz_path = eval_section.get("reference_npz_path", None)
        assert eval_data, "eval.data_path must be specified to enable evaluation."
        assert reference_npz_path, "eval.reference_npz_path must be specified to enable evaluation."
        assert len(eval_metrics) > 0, "eval.metrics must contain at least one metric to compute."
    else:
        do_eval = False
    global_seed = args.global_seed if args.global_seed is not None else default_seed
    seed = global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    experiment_dir, checkpoint_dir, logger = configure_experiment_dirs(args, rank)
    # update args as a dict to full_cfg
    full_cfg.cmd_args = vars(args)
    full_cfg.experiment_dir = experiment_dir
    full_cfg.checkpoint_dir = checkpoint_dir
    
    
    #### Model init
    rae: RAE = instantiate_from_config(rae_config).to(device)
    if args.compile:
        rae.encode = torch.compile(rae.encode)
        rae.forward = torch.compile(rae.forward)
    rae.encoder.eval()
    rae.decoder.train()
    ema_model = deepcopy(rae).to(device).eval()
    ema_model.requires_grad_(False)
    # only train decoder
    rae.encoder.requires_grad_(False)
    rae.decoder.requires_grad_(True)
    if world_size > 1:
        ddp_model = DDP(rae, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False)  # type: ignore[arg-type]
    else:
        ddp_model = SingleProcessWrapper(rae)
    rae = ddp_model.module
    decoder = ddp_model.module.decoder
    discriminator, disc_aug = build_discriminator(disc_cfg, device)
    if world_size > 1:
        ddp_disc = DDP(discriminator, device_ids=[device.index], broadcast_buffers=False, find_unused_parameters=False)  # type: ignore[arg-type]
    else:
        ddp_disc = SingleProcessWrapper(discriminator)
    discriminator = ddp_disc.module
    disc_scheduler: LambdaLR | None = None
    disc_sched_msg: Optional[str] = None

    discriminator.train()
    disc_loss_fn, gen_loss_fn = select_gan_losses(disc_loss_type, gen_loss_type)
    
    lpips = LPIPS().to(device)
    lpips.eval()
    
    #### Opt, Schedl init
    optimizer, optim_msg = build_optimizer(decoder.parameters(), training_cfg)
    disc_params = [p for p in discriminator.parameters() if p.requires_grad]
    disc_optimizer, disc_optim_msg = build_optimizer(disc_params, disc_cfg)
    
    #### AMP init
    scaler, autocast_kwargs = get_autocast_scaler(args)
    
    
    #### Data init
    first_crop_size = 384 if args.image_size == 256 else int(args.image_size * 1.5)
    stage1_transform = transforms.Compose(
        [
            transforms.Resize(first_crop_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(args.image_size),
            transforms.ToTensor(),
        ]
    )    
    loader, sampler = prepare_dataloader(
        args.data_path, batch_size, num_workers, rank, world_size, transform=stage1_transform
    )
    if do_eval:
        eval_dataset = ImageFolder(
            str(eval_data),
            transform=transforms.Compose([
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
                transforms.ToTensor(),
            ])
        )
        logger.info(f"Evaluation dataset loaded from {eval_data}, containing {len(eval_dataset)} images.")
    
    steps_per_epoch = len(loader)
    if steps_per_epoch == 0:
        raise RuntimeError("Dataloader returned zero batches. Check dataset and batch size settings.")
    
    # Schedl init after knowing dataset length
    scheduler: LambdaLR | None = None
    sched_msg: Optional[str] = None
    if training_cfg.get("scheduler"):
        scheduler, sched_msg = build_scheduler(optimizer, steps_per_epoch, training_cfg)
    if disc_cfg.get("scheduler"):
        disc_scheduler, disc_sched_msg = build_scheduler(disc_optimizer, steps_per_epoch, disc_cfg)
    
    ### Resuming and checkpointing
    start_epoch = 0
    global_step = 0
    maybe_resume_ckpt_path = find_resume_checkpoint(experiment_dir)
    if maybe_resume_ckpt_path is not None:
        logger.info(f"Experiment resume checkpoint found at {maybe_resume_ckpt_path}, automatically resuming...")
        ckpt_path = Path(maybe_resume_ckpt_path)
        if ckpt_path.is_file():
            start_epoch, global_step = load_checkpoint(
                ckpt_path,
                ddp_model,
                ema_model,
                optimizer,
                scheduler,
                discriminator,
                disc_optimizer,
                disc_scheduler,
            )
            logger.info(f"[Rank {rank}] Resumed from {ckpt_path} (epoch={start_epoch}, step={global_step}).")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    else:
        # starting from fresh, save worktree and configs
        if rank == 0:
            save_worktree(experiment_dir, full_cfg)
            logger.info(f"Saved training worktree and config to {experiment_dir}.")
    
    ### Logging experiment details
    if rank == 0:
        num_params = sum(p.numel() for p in ddp_model.parameters() if p.requires_grad)
        logger.info(f"Stage-1 RAE trainable parameters: {num_params/1e6:.2f}M")
        logger.info(f"Discriminator architecture:\n{discriminator}")
        num_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
        logger.info(f"Discriminator trainable parameters: {num_params/1e6:.2f}M")
        logger.info(f"Using {disc_loss_type} discriminator loss and {gen_loss_type} generator loss.")
        logger.info(f"Perceptual (LPIPS) weight: {perceptual_weight:.6f}, GAN weight: {disc_weight:.6f}")
        logger.info(f"GAN training starts at epoch {gan_start_epoch}, discriminator updates start at epoch {disc_update_epoch}, LPIPS loss starts at epoch {lpips_start_epoch}.")
        if disc_aug is not None:
            logger.info(f"Using DiffAug with policies: {disc_aug}")
        else:
            logger.info("Not using DiffAug.")
        if clip_grad is not None:
            logger.info(f"Clipping gradients to max norm {clip_grad}.")
        else:
            logger.info("Not clipping gradients.")
        # print optim and schel
        logger.info(optim_msg)
        print(sched_msg if sched_msg else "No LR scheduler for generator.")
        logger.info(disc_optim_msg)
        print(disc_sched_msg if disc_sched_msg else "No LR scheduler for discriminator.")
        logger.info(f"Training for {num_epochs} epochs, batch size {batch_size} per GPU.")
        logger.info(f"Dataset contains {len(loader.dataset)} samples, {steps_per_epoch} steps per epoch.")
        logger.info(f"Running with world size {world_size}, starting from epoch {start_epoch} to {num_epochs}.")


    last_layer = decoder.decoder_pred.weight
    gan_start_step = gan_start_epoch * steps_per_epoch
    disc_update_step = disc_update_epoch * steps_per_epoch
    lpips_start_step = lpips_start_epoch * steps_per_epoch
    dist.barrier()
    for epoch in range(start_epoch, num_epochs):
        ddp_model.train()
        sampler.set_epoch(epoch)
        epoch_metrics: Dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(1, device=device))
        num_batches = 0
        if checkpoint_interval > 0 and epoch % checkpoint_interval == 0  and rank == 0:
            logger.info(f"Saving checkpoint at epoch {epoch}...")
            ckpt_path = f"{checkpoint_dir}/ep-{epoch:07d}.pt" 
            save_checkpoint(
                ckpt_path,
                global_step,
                epoch,
                ddp_model,
                ema_model,
                optimizer,
                scheduler,
                discriminator,
                disc_optimizer,
                disc_scheduler,
            )
        for step, (images, _) in enumerate(loader):
            use_gan = global_step >= gan_start_step and disc_weight > 0.0
            train_disc = global_step >= disc_update_step and disc_weight > 0.0
            use_lpips = global_step >= lpips_start_step and perceptual_weight > 0.0
            images = images.to(device, non_blocking=True)
            real_normed = images * 2.0 - 1.0
            optimizer.zero_grad(set_to_none=True)
            discriminator.eval()
            with autocast(**autocast_kwargs):
                recon = ddp_model(images) # keep gradient synced
                recon_normed = recon * 2.0 - 1.0
                rec_loss = (recon - images).abs().mean() # L1
                if use_lpips:
                    lpips_loss = lpips(real_normed, recon_normed)
                else:
                    lpips_loss = rec_loss.new_zeros(())
                recon_total = rec_loss + perceptual_weight * lpips_loss
                if use_gan:
                    fake_aug = disc_aug.aug(recon_normed)
                    logits_fake, _ = ddp_disc(fake_aug, None)
                    gan_loss = gen_loss_fn(logits_fake)
                else:
                    gan_loss = torch.zeros_like(recon_total)
            # Calculate adaptive weight outside autocast (autograd operation, not forward pass)
            if use_gan:
                adaptive_weight = calculate_adaptive_weight(
                    recon_total, gan_loss, last_layer, max_d_weight
                )
                total_loss = recon_total + disc_weight * adaptive_weight * gan_loss
            else:
                adaptive_weight = torch.zeros_like(recon_total)
                total_loss = recon_total
            total_loss.float()
            if scaler:
                scaler.scale(total_loss).backward()
                if clip_grad is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), clip_grad)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            update_ema(ema_model, ddp_model.module, ema_decay)

            disc_metrics: Dict[str, torch.Tensor] = {}
            if train_disc:
                # Set model to eval mode and get fresh reconstruction with updated weights
                ddp_model.eval()
                ddp_disc.train()
                for _ in range(disc_updates):
                    disc_optimizer.zero_grad(set_to_none=True)
                    with autocast(**autocast_kwargs):
                        # Fresh forward pass with updated model weights (no gradient)
                        with torch.no_grad():
                            recon_disc = ddp_model(images)
                            recon_disc_normed = recon_disc * 2.0 - 1.0
                        # discretize
                        fake_detached = recon_disc_normed.clamp(-1.0, 1.0)
                        fake_detached = torch.round((fake_detached + 1.0) * 127.5) / 127.5 - 1.0
                        fake_input = disc_aug.aug(fake_detached)
                        real_input = disc_aug.aug(real_normed)
                        logits_fake, logits_real = discriminator(fake_input, real_input)
                        d_loss = disc_loss_fn(logits_real, logits_fake)
                        accuracy = (logits_real > logits_fake).float().mean() # accuracy of the discriminator
                    d_loss.float()
                    if scaler:
                        scaler.scale(d_loss).backward()
                        scaler.step(disc_optimizer)
                        scaler.update()
                    else:
                        d_loss.backward()
                        disc_optimizer.step()
                    disc_metrics = {
                        "disc_loss": d_loss.detach(),
                        "logits_real": logits_real.detach().mean(),
                        "logits_fake": logits_fake.detach().mean(),
                        "disc_accuracy": accuracy.detach(),
                    }
                    epoch_metrics["disc_loss"] += d_loss.detach()
                    epoch_metrics["disc_accuracy"] += accuracy.detach()
                    if disc_scheduler is not None:
                        disc_scheduler.step()
                ddp_disc.eval()
                # Set model back to train mode
                ddp_model.train()

            epoch_metrics["recon"] += rec_loss.detach()
            epoch_metrics["lpips"] += lpips_loss.detach()
            epoch_metrics["gan"] += gan_loss.detach()
            epoch_metrics["total"] += total_loss.detach()
            num_batches += 1

            if log_interval > 0 and global_step % log_interval == 0 and rank == 0:
                stats = {
                    "loss/total": total_loss.detach().item(),
                    "loss/recon": rec_loss.detach().item(),
                    "loss/lpips": lpips_loss.detach().item(),
                    "loss/gan": gan_loss.detach().item(),
                    "lr/generator": optimizer.param_groups[0]["lr"],
                }
                if disc_metrics:
                    stats.update(
                        {
                            "loss/disc": disc_metrics["disc_loss"].item(),
                            "disc/logits_real": disc_metrics["logits_real"].item(),
                            "disc/logits_fake": disc_metrics["logits_fake"].item(),
                            "lr/discriminator": disc_optimizer.param_groups[0]["lr"],
                            "disc/accuracy": disc_metrics["disc_accuracy"].item(),
                            "disc/weight": adaptive_weight.item(),
                        }
                    )
                logger.info(
                    f"[Epoch {epoch} | Step {global_step}] "
                    + ", ".join(f"{k}: {v:.4f}" for k, v in stats.items())
                )
                if args.wandb:
                    wandb_utils.log(stats, step=global_step)
            if global_step % sample_every == 0:
                logger.info("Generating EMA samples...")
                with torch.no_grad():
                    # only keep first 4 sample
                    sample_images = images[:4]
                    samples = ema_model.decode(ema_model.encode(sample_images))
                    # also concat input and reconstruction
                    comparison = torch.cat([sample_images, samples], dim=0).cpu().float()
                    # reshape to grid (sample at first row, recon at second row)
                    n = sample_images.size(0)
                    grid = make_grid(comparison, nrow=n)
                    if args.wandb:
                        wandb_utils.log_image(grid, step=global_step)
                logger.info("Generating EMA samples done.")
            if do_eval and (eval_interval > 0 and global_step % eval_interval == 0):
                logger.info("Starting evaluation...")
                eval_models = [(ema_model, "ema")]
                if eval_model:
                    eval_models.append((ddp_model.module, "model"))
                for eval_mod, mod_name in eval_models:
                    eval_stats = evaluate_reconstruction_distributed(
                        eval_mod,
                        eval_dataset,
                        len(eval_dataset),
                        rank = rank,
                        world_size = world_size,
                        device = device,
                        batch_size = batch_size,
                        metrics_to_compute = eval_metrics,
                        experiment_dir = experiment_dir,
                        global_step = global_step,
                        autocast_kwargs = autocast_kwargs,
                        reference_npz_path = reference_npz_path
                    )
                    # log with prefix
                    eval_stats = {f"eval_{mod_name}/{k}": v for k, v in eval_stats.items()} if eval_stats is not None else {}
                    if args.wandb:
                        wandb_utils.log(eval_stats, step=global_step)
                logger.info("Evaluation done.")
            global_step += 1
        if rank == 0 and num_batches > 0:
            avg_recon = (epoch_metrics["recon"] / num_batches).item()
            avg_lpips = (epoch_metrics["lpips"] / num_batches).item()
            avg_gan = (epoch_metrics["gan"] / num_batches).item()
            avg_total = (epoch_metrics["total"] / num_batches).item()
            epoch_stats = {
                "epoch/loss_total": avg_total,
                "epoch/loss_recon": avg_recon,
                "epoch/loss_lpips": avg_lpips,
                "epoch/loss_gan": avg_gan,
            }
            if disc_metrics:
                epoch_stats.update(
                    {
                        "epoch/loss_disc": (epoch_metrics["disc_loss"] / num_batches).item(),
                        "epoch/disc_logits_real": disc_metrics["logits_real"].item(),
                        "epoch/disc_logits_fake": disc_metrics["logits_fake"].item(),
                        "epoch/disc_accuracy": (epoch_metrics["disc_accuracy"] / num_batches).item(),
                    }
                )
            logger.info(
                f"[Epoch {epoch}] "
                + ", ".join(f"{k}: {v:.4f}" for k, v in epoch_stats.items())
            )
            if args.wandb:
                wandb_utils.log(epoch_stats, step=global_step)
    # save the final ckpt
    if rank == 0:
        logger.info(f"Saving final checkpoint at epoch {num_epochs}...")
        ckpt_path = f"{checkpoint_dir}/ep-last.pt" 
        save_checkpoint(
            ckpt_path,
            global_step,
            num_epochs,
            ddp_model,
            ema_model,
            optimizer,
            scheduler,
            discriminator,
            disc_optimizer,
            disc_scheduler,
        )
    dist.barrier()
    logger.info("Done!")
    cleanup_distributed()


if __name__ == "__main__":
    main()