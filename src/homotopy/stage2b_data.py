"""Dataset and target computation for Homotopy RAE Stage 2b generator.

Loads pre-computed z_sem and z_det from the Stage 2a feature cache and
stream params, constructs homotopy targets for all five variants.

Latent shape conventions:
    Cache format:  [N, P, D] = [N, 256, 768]  (tokens × dim)
    DiT input:     [N, D, H, W] = [N, 768, 16, 16]  (channels × height × width)

Homotopy weight schedule (ICPlan convention: t=0 clean, t=1 pure noise):
    w_det(t; gamma) = (1 - t)^gamma
    → t=0 (clean):  w=1 → full detail injection
    → t=1 (noise):  w=0 → pure semantic
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .feature_cache import FeatureCache
from .stream_construction import load_stream_params

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------

def tokens_to_chw(z: np.ndarray) -> np.ndarray:
    """[N, P, D] float32 → [N, D, H, W] float32 where P = H×W."""
    N, P, D = z.shape
    H = W = int(P ** 0.5)
    assert H * W == P, f"P={P} is not a perfect square"
    return z.transpose(0, 2, 1).reshape(N, D, H, W)


def chw_to_tokens(z: np.ndarray) -> np.ndarray:
    """[N, D, H, W] float32 → [N, P, D] float32."""
    N, D, H, W = z.shape
    return z.reshape(N, D, H * W).transpose(0, 2, 1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StreamLatentDataset(Dataset):
    """Pair of (z_sem, z_det) streams for Stage 2b diffusion training.

    Args:
        z_sem:  [N, D, H, W] float32 — semantic stream
        z_det:  [N, D, H, W] float32 — residual detail stream
        labels: [N] int64 — class labels
    """

    def __init__(
        self,
        z_sem: np.ndarray,
        z_det: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        assert z_sem.shape == z_det.shape, (
            f"z_sem {z_sem.shape} and z_det {z_det.shape} must match"
        )
        self.z_sem = torch.from_numpy(np.ascontiguousarray(z_sem)).float()
        self.z_det = torch.from_numpy(np.ascontiguousarray(z_det)).float()
        self.labels = torch.from_numpy(np.ascontiguousarray(labels)).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.z_sem[idx], self.z_det[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Normalizers
# ---------------------------------------------------------------------------

def channel_norm_np(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-channel normalization [N, D, H, W] in numpy."""
    mean = x.mean(axis=(0, 2, 3), keepdims=True)
    std = x.std(axis=(0, 2, 3), keepdims=True).clip(eps)
    return (x - mean) / std


def channel_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-channel normalization [N, D, H, W] in PyTorch."""
    mean = x.mean(dim=(0, 2, 3), keepdim=True)
    std = x.std(dim=(0, 2, 3), keepdim=True).clamp(eps)
    return (x - mean) / std


# ---------------------------------------------------------------------------
# Homotopy target computation
# ---------------------------------------------------------------------------

VALID_VARIANTS = ("H0", "H1", "H2", "H3", "H4")


def compute_homotopy_target(
    z_sem: torch.Tensor,   # [N, D, H, W]
    z_det: torch.Tensor,   # [N, D, H, W]
    t: torch.Tensor,       # [N]  ICPlan timestep (0=clean, 1=noise)
    variant: str,
    alpha: float = 0.5,
    gamma: int = 1,
) -> torch.Tensor:
    """Compute clean data target x1 for the given variant and timestep.

    Variants:
        H0:  z_sem only (no detail stream)
        H1:  Norm(z_sem + alpha * z_det) — static α=0.5 fusion
        H2:  Norm(z_sem + alpha * (1-t)^1 * z_det) — homotopy γ=1
        H3:  Norm(z_sem + alpha * (1-t)^2 * z_det) — homotopy γ=2
        H4:  Norm(z_sem + 1.0 * z_det) — static α=1.0 (param-matched control)

    Args:
        z_sem: Semantic stream [N, D, H, W].
        z_det: Residual detail stream [N, D, H, W].
        t:     ICPlan timestep [N]. Used only for H2/H3.
        variant: One of VALID_VARIANTS.
        alpha:   Locked at 0.5 per Stage 2a results; H4 uses 1.0.
        gamma:   Exponent in w_det(t) = (1-t)^gamma.

    Returns:
        x1: Clean target latent [N, D, H, W].
    """
    if variant not in VALID_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {VALID_VARIANTS}")

    if variant == "H0":
        return z_sem.clone()

    if variant == "H1":
        return channel_norm(z_sem + alpha * z_det)

    if variant == "H4":
        # Parameter-matched static control: same model, α=1.0
        # Purpose: test whether injecting MORE detail statically beats homotopy.
        return channel_norm(z_sem + 1.0 * z_det)

    # H2 or H3: timestep-aware homotopy
    g = gamma  # gamma parameter from config (1 for H2, 2 for H3)
    N = z_sem.shape[0]
    w = (1.0 - t.clamp(0.0, 1.0)) ** g        # [N]
    w = w.view(N, 1, 1, 1).to(z_sem.dtype)
    return channel_norm(z_sem + alpha * w * z_det)


# ---------------------------------------------------------------------------
# Stream pre-computation
# ---------------------------------------------------------------------------

def build_streams(
    cache_dir: str | Path,
    stream_params_dir: str | Path,
    split: str,
    detail_layer: int = 4,
    max_images: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load feature cache + stream params, return (z_sem, z_det, labels).

    Returns arrays in CHW format [N, D, H, W].
    """
    cache = FeatureCache(cache_dir, split)
    sem_builder, det_builders = load_stream_params(stream_params_dir)

    if detail_layer not in det_builders:
        raise ValueError(
            f"Detail layer {detail_layer} not found in stream params. "
            f"Available: {list(det_builders.keys())}"
        )
    det_builder = det_builders[detail_layer]

    n = cache.spec.n_images if max_images is None else min(max_images, cache.spec.n_images)
    log.info(f"Loading {n} {split} images from {cache_dir}")

    layers_needed = sem_builder.k_layer_indices + [detail_layer]
    layer_feats: dict[int, np.ndarray] = {}
    for idx in layers_needed:
        log.info(f"  loading layer {idx} ...")
        layer_feats[idx] = cache.load_layer(idx, max_images=n)

    log.info("Building semantic stream ...")
    z_sem_np, _ = sem_builder.transform(layer_feats)           # [N, P, D]

    log.info("Building detail stream ...")
    z_det_np = det_builder.transform(z_sem_np, layer_feats[detail_layer])  # [N, P, D]

    labels = np.asarray(cache.labels[:n], dtype=np.int64)

    z_sem_chw = tokens_to_chw(z_sem_np.astype(np.float32))    # [N, D, H, W]
    z_det_chw = tokens_to_chw(z_det_np.astype(np.float32))    # [N, D, H, W]

    log.info(
        f"Streams built: z_sem {z_sem_chw.shape}, z_det {z_det_chw.shape}, "
        f"norm(z_sem)={np.linalg.norm(z_sem_chw, axis=1).mean():.3f} "
        f"norm(z_det)={np.linalg.norm(z_det_chw, axis=1).mean():.3f}"
    )
    return z_sem_chw, z_det_chw, labels


# ---------------------------------------------------------------------------
# Stage 2b manifest
# ---------------------------------------------------------------------------

def write_manifest(
    output_dir: str | Path,
    variant: str,
    alpha: float,
    gamma: int,
    detail_layer: int,
    n_train: int,
    n_val: int,
    stream_params_dir: str | Path,
    cache_dir: str | Path,
    git_commit: str = "unknown",
) -> None:
    """Write a manifest.json summarising this run's data provenance."""
    manifest = {
        "stage": "2b",
        "variant": variant,
        "alpha": alpha,
        "gamma": gamma,
        "detail_layer": detail_layer,
        "semantic_layers": [9, 10, 11, 12],
        "n_train": n_train,
        "n_val": n_val,
        "stream_params_dir": str(stream_params_dir),
        "cache_dir": str(cache_dir),
        "git_commit": git_commit,
        "latent_shape": "N x 768 x 16 x 16",
        "timestep_convention": "ICPlan: t=0 clean, t=1 pure noise",
        "homotopy_weight": "w_det(t; gamma) = (1-t)^gamma",
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    (Path(output_dir) / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
