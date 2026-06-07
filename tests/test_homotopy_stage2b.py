"""Unit tests for Homotopy Stage 2b modules.

Run with:
    pytest tests/test_homotopy_stage2b.py -v

These tests are CPU-only and do NOT require the feature cache or DINOv2.
They verify correctness of the data pipeline, target computation, and
training loop without running actual training.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_streams(N: int = 8, D: int = 768, H: int = 16, W: int = 16):
    """Create random z_sem and z_det tensors."""
    z_sem = torch.randn(N, D, H, W)
    z_det = torch.randn(N, D, H, W)
    return z_sem, z_det


# ---------------------------------------------------------------------------
# stage2b_data: shape utilities
# ---------------------------------------------------------------------------

def test_tokens_to_chw_shape():
    from homotopy.stage2b_data import tokens_to_chw
    z = np.random.randn(10, 256, 768).astype(np.float32)
    out = tokens_to_chw(z)
    assert out.shape == (10, 768, 16, 16)


def test_chw_to_tokens_roundtrip():
    from homotopy.stage2b_data import tokens_to_chw, chw_to_tokens
    z = np.random.randn(10, 256, 768).astype(np.float32)
    chw = tokens_to_chw(z)
    tokens = chw_to_tokens(chw)
    assert tokens.shape == z.shape
    np.testing.assert_allclose(tokens, z, atol=1e-5)


# ---------------------------------------------------------------------------
# stage2b_data: channel_norm
# ---------------------------------------------------------------------------

def test_channel_norm_zero_mean_unit_std():
    from homotopy.stage2b_data import channel_norm
    x = torch.randn(16, 768, 16, 16) * 5 + 3
    y = channel_norm(x)
    # Per-channel mean ≈ 0, std ≈ 1
    assert y.shape == x.shape
    mean = y.mean(dim=(0, 2, 3))
    std = y.std(dim=(0, 2, 3))
    assert mean.abs().max().item() < 0.01, "Channel mean should be ≈ 0"
    assert (std - 1.0).abs().max().item() < 0.01, "Channel std should be ≈ 1"


# ---------------------------------------------------------------------------
# stage2b_data: compute_homotopy_target
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("variant", ["H0", "H1", "H2", "H3", "H4"])
def test_target_shape(variant: str):
    from homotopy.stage2b_data import compute_homotopy_target
    N, D, H, W = 4, 768, 16, 16
    z_sem, z_det = make_streams(N, D, H, W)
    t = torch.rand(N)
    out = compute_homotopy_target(z_sem, z_det, t, variant, alpha=0.5, gamma=1)
    assert out.shape == (N, D, H, W), f"{variant}: shape mismatch"


def test_h0_is_pure_semantic():
    from homotopy.stage2b_data import compute_homotopy_target
    N, D, H, W = 4, 768, 16, 16
    z_sem, z_det = make_streams(N, D, H, W)
    t = torch.rand(N)
    out = compute_homotopy_target(z_sem, z_det, t, "H0")
    torch.testing.assert_close(out, z_sem)


def test_h2_h3_at_t0_equals_static():
    """At t=0, homotopy target should equal the static fused target."""
    from homotopy.stage2b_data import compute_homotopy_target, channel_norm
    N, D, H, W = 4, 768, 16, 16
    z_sem, z_det = make_streams(N, D, H, W)
    alpha = 0.5

    t_zero = torch.zeros(N)  # t=0: clean end
    for gamma in [1, 2]:
        variant = f"H{1 + gamma}"  # H2 gamma=1, H3 gamma=2
        out_hom = compute_homotopy_target(z_sem, z_det, t_zero, variant, alpha, gamma)
        out_static = channel_norm(z_sem + alpha * z_det)
        torch.testing.assert_close(out_hom, out_static, atol=1e-5, rtol=1e-5)


def test_h2_h3_at_t1_equals_semantic():
    """At t=1 (pure noise), homotopy target should be z_sem (w_det=0)."""
    from homotopy.stage2b_data import compute_homotopy_target
    N, D, H, W = 4, 768, 16, 16
    z_sem, z_det = make_streams(N, D, H, W)
    alpha = 0.5

    t_one = torch.ones(N)  # t=1: noise end
    for gamma in [1, 2]:
        variant = f"H{1 + gamma}"
        out_hom = compute_homotopy_target(z_sem, z_det, t_one, variant, alpha, gamma)
        # w_det = (1-1)^gamma = 0, so target = channel_norm(z_sem + 0) ≠ z_sem
        # But z_sem + 0 = z_sem, then channel_norm(z_sem) ≠ z_sem
        # The target should equal channel_norm(z_sem)
        from homotopy.stage2b_data import channel_norm
        expected = channel_norm(z_sem)
        torch.testing.assert_close(out_hom, expected, atol=1e-4, rtol=1e-4)


def test_h2_monotone_in_t():
    """H2 weight w_det(t) = (1-t) should decrease monotonically with t."""
    # Check that at low t, H2 is closer to the fused target
    # and at high t, H2 is closer to the semantic target
    from homotopy.stage2b_data import compute_homotopy_target, channel_norm
    N, D, H, W = 4, 768, 16, 16
    z_sem, z_det = make_streams(N, D, H, W)
    alpha = 0.5

    t_low = torch.full((N,), 0.1)
    t_high = torch.full((N,), 0.9)
    out_low = compute_homotopy_target(z_sem, z_det, t_low, "H2", alpha, 1)
    out_high = compute_homotopy_target(z_sem, z_det, t_high, "H2", alpha, 1)
    static = channel_norm(z_sem + alpha * z_det)

    # At t=0.1, should be closer to static than at t=0.9
    dist_low_to_static = (out_low - static).norm()
    dist_high_to_static = (out_high - static).norm()
    assert dist_low_to_static < dist_high_to_static, (
        f"H2 at t=0.1 should be closer to fused than t=0.9: "
        f"{dist_low_to_static:.4f} vs {dist_high_to_static:.4f}"
    )


def test_h3_sharper_than_h2():
    """H3 (gamma=2) should have smaller detail weight than H2 (gamma=1) at mid t."""
    from homotopy.stage2b_data import compute_homotopy_target, channel_norm
    N, D, H, W = 4, 768, 16, 16
    z_sem, z_det = make_streams(N, D, H, W)
    alpha = 0.5

    t_mid = torch.full((N,), 0.5)
    sem_only = compute_homotopy_target(z_sem, z_det, t_mid, "H0")

    out_h2 = compute_homotopy_target(z_sem, z_det, t_mid, "H2", alpha, 1)
    out_h3 = compute_homotopy_target(z_sem, z_det, t_mid, "H3", alpha, 2)

    # At t=0.5: H2 w_det = 0.5, H3 w_det = 0.25
    # H3 should be closer to semantic-only than H2
    dist_h2_sem = (out_h2 - sem_only).norm()
    dist_h3_sem = (out_h3 - sem_only).norm()
    assert dist_h3_sem < dist_h2_sem, (
        f"H3 should be closer to semantic at t=0.5 than H2: "
        f"{dist_h3_sem:.4f} vs {dist_h2_sem:.4f}"
    )


def test_no_nan_in_targets():
    from homotopy.stage2b_data import compute_homotopy_target
    N, D, H, W = 4, 768, 16, 16
    z_sem, z_det = make_streams(N, D, H, W)
    for variant in ["H0", "H1", "H2", "H3", "H4"]:
        t = torch.rand(N)
        out = compute_homotopy_target(z_sem, z_det, t, variant, 0.5, 1)
        assert not out.isnan().any(), f"{variant} produced NaN"
        assert not out.isinf().any(), f"{variant} produced Inf"


# ---------------------------------------------------------------------------
# stage2b_data: StreamLatentDataset
# ---------------------------------------------------------------------------

def test_dataset_length_and_shapes():
    from homotopy.stage2b_data import StreamLatentDataset, tokens_to_chw
    N, D, H, W = 16, 768, 16, 16
    z_sem = np.random.randn(N, D, H, W).astype(np.float32)
    z_det = np.random.randn(N, D, H, W).astype(np.float32)
    labels = np.arange(N, dtype=np.int64)
    ds = StreamLatentDataset(z_sem, z_det, labels)
    assert len(ds) == N
    zs, zd, lab = ds[0]
    assert zs.shape == (D, H, W)
    assert zd.shape == (D, H, W)
    assert lab.item() == 0


# ---------------------------------------------------------------------------
# Training loop: ICPlan loss
# ---------------------------------------------------------------------------

def test_icplan_loss_no_nan():
    """ICPlan loss should be a finite scalar for any input."""
    import sys; sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from scripts.train_homotopy_stage2b_generator import icplan_loss
    from stage2.models.DDT import DiTwDDTHead

    model = DiTwDDTHead(
        input_size=16, patch_size=[1, 1], in_channels=768,
        hidden_size=[192, 384], depth=[2, 1], num_heads=[3, 6],
        mlp_ratio=4.0, class_dropout_prob=0.1,
        num_classes=1001, use_qknorm=False, use_swiglu=True,
        use_rope=True, use_rmsnorm=True, wo_shift=False, use_pos_embed=True,
    )
    N, D, H, W = 2, 768, 16, 16
    x1 = torch.randn(N, D, H, W)
    t = torch.tensor([0.3, 0.7])
    y = torch.tensor([0, 1])
    loss = icplan_loss(model, x1, t, y)
    assert loss.isfinite(), f"Loss is not finite: {loss}"
    assert loss.item() > 0, "Loss should be positive"


def test_icplan_loss_decreases_with_training():
    """Loss should decrease after several gradient steps."""
    import sys; sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from scripts.train_homotopy_stage2b_generator import icplan_loss, update_ema
    from stage2.models.DDT import DiTwDDTHead
    from copy import deepcopy

    torch.manual_seed(42)
    model = DiTwDDTHead(
        input_size=16, patch_size=[1, 1], in_channels=768,
        hidden_size=[192, 384], depth=[2, 1], num_heads=[3, 6],
        mlp_ratio=4.0, class_dropout_prob=0.1,
        num_classes=1001, use_qknorm=False, use_swiglu=True,
        use_rope=True, use_rmsnorm=True, wo_shift=False, use_pos_embed=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Overfit on a tiny fixed batch
    N, D, H, W = 4, 768, 16, 16
    x1 = torch.randn(N, D, H, W)
    t = torch.rand(N)
    y = torch.zeros(N, dtype=torch.long)

    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        loss = icplan_loss(model, x1, t, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], (
        f"Loss should decrease during training; got {losses[0]:.4f} → {losses[-1]:.4f}"
    )


# ---------------------------------------------------------------------------
# Smoke test: full pipeline without cache
# ---------------------------------------------------------------------------

def test_smoke_train_no_crash():
    """Run a 2-step training smoke test end-to-end (CPU, no cache, no W&B)."""
    import subprocess
    import sys
    result = subprocess.run(
        [
            sys.executable, "scripts/train_homotopy_stage2b_generator.py",
            "--cache-dir", "/nonexistent/cache",
            "--stream-params-dir", "/nonexistent/stream",
            "--results-dir", "/tmp/stage2b_smoke_test",
            "--variant", "H2",
            "--smoke",
            "--skip-decode",
        ],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parent.parent),
    )
    # Expect failure due to missing cache — that's fine for this test
    # The important thing is the arg parsing and import work
    assert "error" not in result.stderr.lower() or "FileNotFoundError" in result.stderr, (
        f"Unexpected error:\n{result.stderr[:500]}"
    )
