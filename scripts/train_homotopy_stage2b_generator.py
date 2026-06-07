#!/usr/bin/env python
"""Homotopy RAE Stage 2b — Fixed Homotopy Generator Training.

Trains a DiT on latent streams (z_sem, z_det) from Stage 2a, comparing
five variants of the clean target:

    H0  z_sem only (semantic baseline)
    H1  Norm(z_sem + 0.5 * z_det)                       static α=0.5
    H2  Norm(z_sem + 0.5 * (1-t)^1 * z_det)             homotopy γ=1
    H3  Norm(z_sem + 0.5 * (1-t)^2 * z_det)             homotopy γ=2
    H4  Norm(z_sem + 1.0 * z_det)                        static α=1.0 (param-matched)

ICPlan convention: t=0 = clean data, t=1 = pure noise.
Velocity prediction: model output = x_0 - x_1 (i.e. noise - clean).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler

# --------------------------------------------------------------------------
# Path setup
# --------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from homotopy.stage2b_data import (
    VALID_VARIANTS,
    StreamLatentDataset,
    build_streams,
    compute_homotopy_target,
    write_manifest,
)
from stage2.models.DDT import DiTwDDTHead
from utils.model_utils import instantiate_from_config


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Homotopy Stage 2b generator training")

    # Data
    p.add_argument("--cache-dir", required=True, type=Path)
    p.add_argument("--stream-params-dir", required=True, type=Path)
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument("--detail-layer", type=int, default=4)
    p.add_argument("--max-train-images", type=int, default=None)
    p.add_argument("--max-val-images", type=int, default=1000)

    # Variant
    p.add_argument("--variant", required=True, choices=VALID_VARIANTS)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--gamma", type=int, default=1, choices=[1, 2])

    # Model
    p.add_argument("--hidden-size-enc", type=int, default=384)
    p.add_argument("--hidden-size-dec", type=int, default=2048)
    p.add_argument("--depth-enc", type=int, default=12)
    p.add_argument("--depth-dec", type=int, default=2)
    p.add_argument("--num-heads-enc", type=int, default=6)
    p.add_argument("--num-heads-dec", type=int, default=16)
    p.add_argument("--init-from-ckpt", type=Path, default=None,
                   help="Optional pre-trained DiT checkpoint to warm-start from")

    # Training
    p.add_argument("--n-steps", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")

    # Eval / sampling
    p.add_argument("--val-every", type=int, default=2000)
    p.add_argument("--sample-every", type=int, default=4000)
    p.add_argument("--ckpt-every", type=int, default=5000)
    p.add_argument("--num-val-steps", type=int, default=20,
                   help="Number of fixed t values to eval val loss at")
    p.add_argument("--n-sample-classes", type=int, default=8)
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--decoder-config", type=str,
                   default="configs/decoder/ViTXL",
                   help="Path to decoder config for RAE (for sample decoding)")
    p.add_argument("--decoder-ckpt", type=str,
                   default="models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt",
                   help="Pre-trained decoder checkpoint")
    p.add_argument("--decoder-stat", type=str,
                   default="models/stats/dinov2/wReg_base/imagenet1k/stat.pt",
                   help="RAE normalization stats (used when decoding generated latents)")
    p.add_argument("--skip-decode", action="store_true",
                   help="Skip image decoding in sampling (useful when decoder not available)")

    # W&B
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="homotopy-rae")
    p.add_argument("--wandb-group", type=str, default="stage2b-fixed-homotopy")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--run-name", type=str, default=None)

    # Smoke test
    p.add_argument("--smoke", action="store_true",
                   help="Tiny smoke-test: 2 steps, 16 images, no real training")

    return p.parse_args()


# --------------------------------------------------------------------------
# Time distribution (uniform ICPlan)
# --------------------------------------------------------------------------
T_EPS = 1e-3  # avoid exact 0 and 1

def sample_t(batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample diffusion timestep uniformly in (eps, 1-eps)."""
    return torch.rand(batch_size, device=device) * (1.0 - 2 * T_EPS) + T_EPS


def val_timesteps(n: int) -> list[float]:
    """Fixed grid of timesteps for validation loss computation."""
    return [T_EPS + (1 - 2 * T_EPS) * i / (n - 1) for i in range(n)]


BUCKET_THRESHOLDS = (0.33, 0.67)  # low=[0,0.33], mid=[0.33,0.67], high=[0.67,1]


def timestep_bucket(t: float) -> str:
    if t < BUCKET_THRESHOLDS[0]:
        return "low"
    if t < BUCKET_THRESHOLDS[1]:
        return "mid"
    return "high"


# --------------------------------------------------------------------------
# ICPlan transport
# --------------------------------------------------------------------------

def icplan_loss(
    model: nn.Module,
    x1: torch.Tensor,      # [N, D, H, W]  clean target
    t: torch.Tensor,       # [N]
    y: torch.Tensor,       # [N] class labels
    training: bool = True,
) -> torch.Tensor:
    """Compute ICPlan velocity prediction loss.

    x_t = (1-t) * x1 + t * x0,   x0 ~ N(0, I)
    velocity target: ut = x0 - x1
    loss = ||model(x_t, t, y) - ut||^2  per element, averaged over N
    """
    N = x1.shape[0]
    x0 = torch.randn_like(x1)
    t_expanded = t.view(N, *([1] * (x1.ndim - 1)))
    xt = (1.0 - t_expanded) * x1 + t_expanded * x0
    ut = x0 - x1   # velocity target (noise - clean)
    pred = model(xt, t, y)
    return ((pred - ut) ** 2).mean()


# --------------------------------------------------------------------------
# EMA update
# --------------------------------------------------------------------------

@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    for (_, p_ema), (_, p) in zip(ema.named_parameters(), model.named_parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


# --------------------------------------------------------------------------
# LR schedule (warmup + cosine)
# --------------------------------------------------------------------------

def build_lr_schedule(optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


# --------------------------------------------------------------------------
# Sampling (ODE Euler)
# --------------------------------------------------------------------------

@torch.no_grad()
def euler_sample(
    model: nn.Module,
    z_shape: tuple,
    y: torch.Tensor,
    n_steps: int,
    device: torch.device,
    dtype: torch.dtype,
    cfg_scale: float = 1.0,
    null_label: int = 1000,
) -> torch.Tensor:
    """Simple Euler ODE sampler for ICPlan (velocity prediction).

    Integrates x_t from t=1 (noise) to t=0 (clean data).
    dx/dt = -v(x_t, t)  (negative because we go from t=1 to t=0)
    """
    N = y.shape[0]
    x = torch.randn(N, *z_shape, device=device, dtype=dtype)
    dt = -1.0 / n_steps
    ts = torch.linspace(1.0, T_EPS, n_steps + 1, device=device)

    model.eval()
    for i in range(n_steps):
        t_batch = ts[i].expand(N)
        if cfg_scale > 1.0:
            y_null = torch.full_like(y, null_label)
            x_cat = torch.cat([x, x], dim=0)
            t_cat = torch.cat([t_batch, t_batch], dim=0)
            y_cat = torch.cat([y, y_null], dim=0)
            with autocast(dtype=dtype, enabled=(dtype != torch.float32)):
                v_cat = model(x_cat, t_cat, y_cat)
            v_cond, v_uncond = v_cat.chunk(2, dim=0)
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            with autocast(dtype=dtype, enabled=(dtype != torch.float32)):
                v = model(x, t_batch, y)
        x = x + dt * v
    return x


# --------------------------------------------------------------------------
# Decoder for visualization
# --------------------------------------------------------------------------

def load_rae_decoder(
    decoder_config: str,
    decoder_ckpt: str,
    decoder_stat: str,
    device: torch.device,
):
    """Load RAE decoder for decoding generated latents."""
    from stage1.decoders import GeneralDecoder
    from transformers import AutoConfig
    import json as _json

    config_path = Path(decoder_config)
    if config_path.is_dir():
        cfg_file = config_path / "config.json"
        with open(cfg_file) as f:
            cfg_dict = _json.load(f)
        if cfg_dict.get("patch_size") == "SHOULD BE RELOADED":
            cfg_dict["patch_size"] = 14
        model_type = cfg_dict.pop("model_type")
        dec_cfg = AutoConfig.for_model(model_type, **cfg_dict)
    else:
        dec_cfg = AutoConfig.from_pretrained(decoder_config)

    dec_cfg.hidden_size = 768
    dec_cfg.patch_size = 14
    dec_cfg.image_size = int(14 * 16)  # 224
    decoder = GeneralDecoder(dec_cfg, num_patches=256)

    ckpt = torch.load(decoder_ckpt, map_location="cpu")
    decoder.load_state_dict(ckpt, strict=False)
    decoder = decoder.to(device).eval()

    stats = torch.load(decoder_stat, map_location="cpu")
    latent_mean = stats.get("mean", None)
    latent_var = stats.get("var", None)
    return decoder, latent_mean, latent_var


def decode_latents(
    z: torch.Tensor,       # [N, D, H, W]
    decoder,
    latent_mean,
    latent_var,
    device: torch.device,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Decode latents to images [N, 3, H_img, W_img]."""
    if latent_mean is not None:
        z = z * torch.sqrt(latent_var.to(device) + eps) + latent_mean.to(device)
    b, c, h, w = z.shape
    z_tokens = z.view(b, c, h * w).permute(0, 2, 1)  # [N, P, D]
    with torch.no_grad():
        logits = decoder(z_tokens, drop_cls_token=False).logits
        x_rec = decoder.unpatchify(logits)
    # Re-scale from DINOv2 stats
    dinov2_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    dinov2_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    x_rec = x_rec * dinov2_std + dinov2_mean
    return x_rec.clamp(0.0, 1.0)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.smoke:
        args.n_steps = 4
        args.val_every = 2
        args.sample_every = 4
        args.ckpt_every = 9999
        args.max_train_images = 32
        args.max_val_images = 16
        args.batch_size = 8
        log.info("=== SMOKE TEST MODE ===")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    use_amp = args.precision == "bf16" and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    log.info(f"Precision: {args.precision} (amp={use_amp})")

    # ---- Results directory ----
    run_name = args.run_name or (
        f"{args.variant.lower()}_L{args.detail_layer}_a{args.alpha:.2f}"
        + (f"_g{args.gamma}" if args.variant in ("H2", "H3") else "")
        + f"_s{args.seed}"
    )
    results_dir = Path(args.results_dir) / run_name
    ckpt_dir = results_dir / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Redirect stdout to log file too
    file_log = open(results_dir / "train.log", "w")
    original_stdout = sys.stdout

    log.info(f"Run: {run_name}  →  {results_dir}")

    # ---- W&B ----
    run = None
    if args.wandb:
        import wandb
        commit = subprocess.getoutput("git rev-parse --short HEAD 2>/dev/null || echo unknown")
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group,
            name=run_name,
            tags=["homotopy-rae", "stage2b", "fixed-homotopy",
                  args.variant, f"L{args.detail_layer}", f"alpha-{args.alpha}",
                  f"seed-{args.seed}", "runai"],
            config={
                "variant": args.variant,
                "alpha": args.alpha,
                "gamma": args.gamma,
                "detail_layer": args.detail_layer,
                "n_steps": args.n_steps,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "warmup_steps": args.warmup_steps,
                "ema_decay": args.ema_decay,
                "seed": args.seed,
                "precision": args.precision,
                "git_commit": commit,
            },
        )
        wandb.define_metric("step")
        wandb.define_metric("*", step_metric="step")

    # ---- Load streams ----
    log.info("Loading training streams ...")
    z_sem_tr, z_det_tr, labels_tr = build_streams(
        args.cache_dir, args.stream_params_dir,
        split="train",
        detail_layer=args.detail_layer,
        max_images=args.max_train_images,
    )
    log.info("Loading validation streams ...")
    z_sem_va, z_det_va, labels_va = build_streams(
        args.cache_dir, args.stream_params_dir,
        split="val",
        detail_layer=args.detail_layer,
        max_images=args.max_val_images,
    )
    N_tr, D, H, W = z_sem_tr.shape
    N_va = z_sem_va.shape[0]
    log.info(f"Train: {N_tr} images  |  Val: {N_va} images  |  Latent: [{D},{H},{W}]")

    # ---- Manifest ----
    commit = subprocess.getoutput("git rev-parse --short HEAD 2>/dev/null || echo unknown")
    write_manifest(
        results_dir, args.variant, args.alpha, args.gamma,
        args.detail_layer, N_tr, N_va,
        args.stream_params_dir, args.cache_dir, git_commit=commit,
    )

    # ---- Dataset / Dataloader ----
    train_ds = StreamLatentDataset(z_sem_tr, z_det_tr, labels_tr)
    val_ds = StreamLatentDataset(z_sem_va, z_det_va, labels_va)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=RandomSampler(train_ds, replacement=True, num_samples=args.n_steps * args.batch_size),
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=min(args.batch_size, N_va),
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    # ---- Model ----
    num_classes = 1000
    null_label = num_classes  # for classifier-free guidance
    model = DiTwDDTHead(
        input_size=H,                                          # 16
        patch_size=[1, 1],
        in_channels=D,                                         # 768
        hidden_size=[args.hidden_size_enc, args.hidden_size_dec],
        depth=[args.depth_enc, args.depth_dec],
        num_heads=[args.num_heads_enc, args.num_heads_dec],
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=num_classes + 1,  # +1 for null label
        use_qknorm=False,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
        wo_shift=False,
        use_pos_embed=True,
    ).to(device)

    if args.init_from_ckpt is not None:
        log.info(f"Loading checkpoint from {args.init_from_ckpt}")
        ckpt = torch.load(args.init_from_ckpt, map_location="cpu")
        state = ckpt.get("model", ckpt.get("ema", ckpt))
        missing, unexpected = model.load_state_dict(state, strict=False)
        log.info(f"  missing: {len(missing)}, unexpected: {len(unexpected)}")

    ema_model = deepcopy(model).to(device)
    ema_model.requires_grad_(False)
    ema_model.eval()
    model.requires_grad_(True)

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {n_params/1e6:.1f}M")

    # ---- Optimizer / scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = build_lr_schedule(optimizer, args.warmup_steps, args.n_steps)
    scaler = GradScaler(enabled=(args.precision == "fp16"))

    # ---- Load decoder for visualization ----
    decoder = None
    lat_mean = None
    lat_var = None
    if not args.skip_decode:
        dec_cfg_path = str(REPO_ROOT / args.decoder_config)
        dec_ckpt_path = str(REPO_ROOT / args.decoder_ckpt)
        dec_stat_path = str(REPO_ROOT / args.decoder_stat)
        if Path(dec_ckpt_path).exists() and Path(dec_stat_path).exists():
            try:
                decoder, lat_mean, lat_var = load_rae_decoder(
                    dec_cfg_path, dec_ckpt_path, dec_stat_path, device
                )
                log.info("RAE decoder loaded for visualization.")
            except Exception as e:
                log.warning(f"Could not load decoder: {e}. Skipping image generation.")
        else:
            log.warning("Decoder files not found. Skipping image generation.")

    # ---- Fixed val setup ----
    t_grid = val_timesteps(args.num_val_steps)  # fixed t values for per-timestep loss

    # Fixed sample labels (same 8 classes every time for visual consistency)
    sample_labels = torch.arange(args.n_sample_classes, device=device) * (num_classes // args.n_sample_classes)

    # ---- Metrics tracker ----
    results_log: list[dict] = []

    # --------------------------------------------------------------------------
    # Validation function
    # --------------------------------------------------------------------------
    def run_validation(step: int) -> dict:
        """Compute validation loss at fixed timesteps."""
        model.eval()
        bucket_losses: dict[str, list[float]] = {"low": [], "mid": [], "high": []}
        per_t_losses: dict[float, float] = {}

        with torch.no_grad():
            for t_val in t_grid:
                t_tensor_val = torch.full((1,), t_val, device=device)
                batch_losses = []
                for z_sem_b, z_det_b, labels_b in val_loader:
                    z_sem_b = z_sem_b.to(device)
                    z_det_b = z_det_b.to(device)
                    labels_b = labels_b.to(device)
                    N_b = z_sem_b.shape[0]
                    t_b = t_tensor_val.expand(N_b)
                    x1 = compute_homotopy_target(
                        z_sem_b, z_det_b, t_b,
                        args.variant, args.alpha, args.gamma,
                    )
                    with autocast(dtype=amp_dtype, enabled=use_amp):
                        loss_b = icplan_loss(model, x1, t_b, labels_b, training=False)
                    batch_losses.append(loss_b.item())
                t_loss = float(np.mean(batch_losses))
                per_t_losses[t_val] = t_loss
                bucket_losses[timestep_bucket(t_val)].append(t_loss)

        metrics = {
            "val/loss_low_noise": float(np.mean(bucket_losses["low"])),
            "val/loss_mid_noise": float(np.mean(bucket_losses["mid"])),
            "val/loss_high_noise": float(np.mean(bucket_losses["high"])),
            "val/loss_all": float(np.mean(list(per_t_losses.values()))),
        }
        model.train()
        return metrics, per_t_losses

    # --------------------------------------------------------------------------
    # Sampling function
    # --------------------------------------------------------------------------
    def run_sampling(step: int) -> None:
        """Generate samples using EMA model and log to W&B."""
        ema_model.eval()
        with torch.no_grad():
            z_gen = euler_sample(
                ema_model, (D, H, W), sample_labels,
                n_steps=args.sample_steps,
                device=device, dtype=amp_dtype,
            )

        if decoder is not None:
            try:
                imgs = decode_latents(z_gen.float(), decoder, lat_mean, lat_var, device)
                # Log to W&B
                if run is not None:
                    from torchvision.utils import make_grid
                    grid = make_grid(imgs.cpu(), nrow=4, normalize=False)
                    import wandb
                    run.log({
                        "samples/grid": wandb.Image(grid.permute(1, 2, 0).numpy()),
                        "step": step,
                    })
                    log.info(f"  [step {step}] Sample grid logged to W&B.")
                # Save to disk
                grid_np = (grid.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)
                from PIL import Image
                img_path = results_dir / f"samples_step{step:07d}.png"
                Image.fromarray(grid_np).save(img_path)
            except Exception as e:
                log.warning(f"  Sampling decode failed: {e}")
        else:
            # Log latent norms as proxy
            z_norm = z_gen.norm(dim=1).mean().item()
            log.info(f"  [step {step}] z_gen norm={z_norm:.3f} (no decoder)")
            if run is not None:
                run.log({"samples/z_gen_norm": z_norm, "step": step})

    # --------------------------------------------------------------------------
    # Training loop
    # --------------------------------------------------------------------------
    log.info(f"Starting training: {args.n_steps} steps, variant={args.variant}, "
             f"alpha={args.alpha}, gamma={args.gamma}")

    step = 0
    running_loss = 0.0
    log_window = 100
    t0 = time.time()
    model.train()

    for z_sem_b, z_det_b, labels_b in train_loader:
        if step >= args.n_steps:
            break

        z_sem_b = z_sem_b.to(device)
        z_det_b = z_det_b.to(device)
        labels_b = labels_b.to(device)
        N_b = z_sem_b.shape[0]

        # Sample timestep
        t_b = sample_t(N_b, device)

        # Compute homotopy target (depends on t for H2/H3)
        x1 = compute_homotopy_target(
            z_sem_b, z_det_b, t_b,
            args.variant, args.alpha, args.gamma,
        )

        # Forward + loss
        optimizer.zero_grad(set_to_none=True)
        with autocast(dtype=amp_dtype, enabled=use_amp):
            loss = icplan_loss(model, x1, t_b, labels_b, training=True)

        if args.precision == "fp16":
            scaler.scale(loss).backward()
            if args.clip_grad > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

        scheduler.step()
        update_ema(ema_model, model, args.ema_decay)

        # --- Logging ---
        running_loss += loss.item()
        step += 1

        if step % log_window == 0:
            avg = running_loss / log_window
            elapsed = time.time() - t0
            sps = log_window / elapsed
            lr = optimizer.param_groups[0]["lr"]
            log.info(f"step={step:6d}  loss={avg:.4f}  lr={lr:.2e}  {sps:.1f} step/s")

            # Homotopy-specific: log mean w_det for H2/H3
            extra = {}
            if args.variant in ("H2", "H3"):
                w_mean = float(((1 - t_b) ** args.gamma).mean().item())
                extra["train/w_det_mean"] = w_mean

            if run is not None:
                run.log({
                    "train/loss": avg,
                    "train/lr": lr,
                    "step": step,
                    **extra,
                })
            running_loss = 0.0
            t0 = time.time()

        # --- Validation ---
        if step % args.val_every == 0:
            log.info(f"[step {step}] Running validation ...")
            val_metrics, per_t = run_validation(step)
            log.info(
                f"  val_low={val_metrics['val/loss_low_noise']:.4f}  "
                f"val_mid={val_metrics['val/loss_mid_noise']:.4f}  "
                f"val_high={val_metrics['val/loss_high_noise']:.4f}  "
                f"val_all={val_metrics['val/loss_all']:.4f}"
            )
            row = {
                "step": step,
                "variant": args.variant,
                "seed": args.seed,
                **val_metrics,
            }
            results_log.append(row)
            if run is not None:
                run.log({**val_metrics, "step": step})
            # Save per-timestep loss to disk
            pt_path = results_dir / f"per_t_val_step{step:07d}.json"
            pt_path.write_text(json.dumps({"step": step, "per_t": per_t}, indent=2))

        # --- Sampling ---
        if step % args.sample_every == 0:
            log.info(f"[step {step}] Generating samples ...")
            run_sampling(step)

        # --- Checkpoint ---
        if step % args.ckpt_every == 0:
            ckpt_path = ckpt_dir / f"step{step:07d}.pt"
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "ema": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "args": vars(args),
            }, ckpt_path)
            log.info(f"  Checkpoint saved: {ckpt_path}")

    # ---- Final validation ----
    log.info("=== Final validation ===")
    val_metrics_final, per_t_final = run_validation(step)
    log.info(
        f"Final val_low={val_metrics_final['val/loss_low_noise']:.4f}  "
        f"val_mid={val_metrics_final['val/loss_mid_noise']:.4f}  "
        f"val_high={val_metrics_final['val/loss_high_noise']:.4f}  "
        f"val_all={val_metrics_final['val/loss_all']:.4f}"
    )
    if run is not None:
        run.log({**val_metrics_final, "step": step})

    # ---- Final sampling ----
    log.info("=== Final sampling ===")
    run_sampling(step)

    # ---- Save final checkpoint ----
    final_ckpt = ckpt_dir / "final.pt"
    torch.save({
        "step": step,
        "model": model.state_dict(),
        "ema": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "args": vars(args),
    }, final_ckpt)
    log.info(f"Final checkpoint: {final_ckpt}")

    # ---- Save results summary ----
    summary = {
        "variant": args.variant,
        "seed": args.seed,
        "detail_layer": args.detail_layer,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "n_steps": step,
        "n_train": N_tr,
        "n_val": N_va,
        "n_params": n_params,
        "final_val": val_metrics_final,
        "final_per_t": per_t_final,
        "run_name": run_name,
        "wandb_run_id": run.id if run is not None else None,
        "wandb_run_url": run.url if run is not None else None,
    }
    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log.info(f"Summary saved: {summary_path}")

    if run is not None:
        run.log({
            "summary/val_loss_low": val_metrics_final["val/loss_low_noise"],
            "summary/val_loss_mid": val_metrics_final["val/loss_mid_noise"],
            "summary/val_loss_high": val_metrics_final["val/loss_high_noise"],
            "summary/val_loss_all": val_metrics_final["val/loss_all"],
            "step": step,
        })
        run.finish()

    log.info("=== Training complete ===")
    log.info(f"Results: {results_dir}")
    file_log.close()


if __name__ == "__main__":
    main()
