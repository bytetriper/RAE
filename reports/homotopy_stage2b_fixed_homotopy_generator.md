# Homotopy RAE Stage 2b: Fixed Homotopy Generator

**Date**: 2026-06-07  
**Git commit**: a4d18c4  
**Status**: RUNNING — awaiting job results

---

## Executive verdict

*PENDING — jobs running on Run:AI cluster*

---

## Research question

> Does timestep-aware injection of the L4 residual detail stream beat a static semantic-plus-detail fusion baseline during RAE latent diffusion training?

This is distinct from asking whether homotopy > semantic-only. Static fusion (H1) tests whether "more information helps." Homotopy (H2/H3) only claims credit if a phase-aware schedule beats static fusion under identical model capacity.

---

## Setup

### Data

| Property            | Value                                          |
|---------------------|------------------------------------------------|
| Dataset             | ImageNet-1K (Stage 2a feature cache)           |
| Train images        | 5,000 (small-cache prototype)                  |
| Val images          | 1,000                                          |
| Resolution          | 224×224 → 16×16 token grid                    |
| Latent shape        | [768, 16, 16]                                  |
| Feature cache       | artifacts/homotopy_stage2a/feature_cache       |
| Stream params       | artifacts/homotopy_stage2a/stream_params       |

**Note**: This is a small-cache prototype (5K/1K). The spec calls for 50K–100K train / 5K–10K val. Results should be treated as directional, not final.

### Streams (locked from Stage 2a)

| Component        | Definition                                      |
|------------------|-------------------------------------------------|
| z_sem            | ChannelNorm(mean(layers 9,10,11,12))            |
| z_det            | ChannelNorm(residual of L4 after removing z_sem)|
| alpha            | 0.5 (locked from Stage 2a α sweep)             |
| detail layer     | 4 (locked from Stage 2a STRONG_GO verdict)      |

### Model

| Property             | Value              |
|----------------------|--------------------|
| Architecture         | DiTwDDTHead (S)    |
| Encoder hidden size  | 384                |
| Decoder hidden size  | 2048               |
| Encoder depth        | 12                 |
| Decoder depth        | 2                  |
| Encoder heads        | 6                  |
| Decoder heads        | 16                 |
| Parameters           | ~60M               |
| Init                 | Random (all variants) |
| Pattern              | A — target replacement only |

All variants use **identical model architecture**. H2/H3 have **zero extra parameters** over H1/H4.

### Timestep convention

ICPlan (linear flow matching):
```
x_t = (1-t) * x_data + t * x_noise    (t=0: clean, t=1: pure noise)
velocity target: ut = x_noise - x_data
```

### Homotopy schedule

```
w_det(t; gamma) = (1 - t)^gamma
```

- At t=0 (clean / low noise): w_det = 1.0 → full detail injection
- At t=1 (noise / high noise): w_det = 0.0 → pure semantic

This matches Stage 2a findings: detail stream useful at **low noise**, semantic stream dominates at **high noise**.

### Training budget

| Hyperparameter    | Value       |
|-------------------|-------------|
| Steps             | 20,000      |
| Batch size        | 128         |
| Learning rate     | 1e-4        |
| Warmup            | 1,000 steps |
| LR schedule       | Cosine      |
| Gradient clip     | 1.0         |
| EMA decay         | 0.9999      |
| Precision         | bfloat16    |
| Optimizer         | AdamW       |

### Variants

| Variant | Clean target                               | Purpose                           |
|:-------:|:------------------------------------------:|:----------------------------------|
| **H0**  | z_sem                                      | Semantic-only baseline            |
| **H1**  | Norm(z_sem + 0.5 · z_det)                 | Static fusion (KEY BASELINE)      |
| **H2**  | Norm(z_sem + 0.5 · (1-t)¹ · z_det)        | Homotopy γ=1 (linear)             |
| **H3**  | Norm(z_sem + 0.5 · (1-t)² · z_det)        | Homotopy γ=2 (sharper)            |
| **H4**  | Norm(z_sem + 1.0 · z_det)                 | Static α=1.0 (param-matched ctrl) |

H4 note: Since Pattern A adds zero extra parameters, all variants are parameter-identical. H4 uses α=1.0 to test whether maximum static detail injection beats homotopy.

### Run:AI jobs (seed 0 screen)

| Job name                                         | Variant | Status  |
|:-------------------------------------------------|:-------:|:-------:|
| rae-s2b-h0-s0-a4d18c4-v1                        | H0      | RUNNING |
| rae-s2b-h1-s0-a4d18c4-v1                        | H1      | RUNNING |
| rae-s2b-h2-s0-a4d18c4-v1                        | H2      | RUNNING |
| rae-s2b-h3-s0-a4d18c4-v1                        | H3      | RUNNING |
| rae-s2b-h4-s0-a4d18c4-v1                        | H4      | RUNNING |

### W&B

- Project: `homotopy-rae`
- Group: `stage2b-fixed-homotopy-a4d18c4`

---

## Go / No-Go rule

| Criterion                            | Threshold         |
|:-------------------------------------|:-----------------:|
| Low-noise loss improvement over H1   | ≥ 5% relative     |
| High-noise loss degradation limit    | ≤ 2% relative     |
| Beat H4 parameter-matched control    | Required          |
| Visual quality (qualitative)         | No decoder failure|

**STRONG GO**: Both H2 and H3 beat H1+H4 on low-noise loss, no high-noise degradation, stable across seeds.

**GO**: At least one of H2/H3 beats H1+H4.

**NO-GO**: Static fusion H1 as good as or better; homotopy only wins because of extra capacity; decoder fails.

---

## Results (PENDING)

*To be filled after jobs complete.*

| Variant | Seed | val_all | val_low | val_mid | val_high | Notes |
|:-------:|:----:|--------:|--------:|--------:|---------:|:------|
| H0      | 0    | -       | -       | -       | -        |       |
| H1      | 0    | -       | -       | -       | -        |       |
| H2      | 0    | -       | -       | -       | -        |       |
| H3      | 0    | -       | -       | -       | -        |       |
| H4      | 0    | -       | -       | -       | -        |       |

---

## Key comparison (PENDING)

*To be written after results arrive.*

---

## Decoder compatibility (PENDING)

The RAE decoder was trained on single-layer (block 12) DINOv2 features with stat.pt normalization. The z_sem used here is the mean of blocks 9–12 with ChannelNorm. Both produce ~zero-mean, unit-std per channel latents, but the exact statistics differ.

Expected finding: decoding z_sem through the RAE decoder will produce valid images but with slightly different sharpness/colour statistics compared to the proper RAE latent. This is a known limitation and does not affect the key comparison (all variants use the same decoder).

---

## Implementation notes

- **Pattern A** used: target latent replaced, model architecture unchanged.
- **Stage 2a artifacts reused**: stream params from `artifacts/homotopy_stage2a/stream_params/`.
- **Timestep bucket losses**: computed at 20 fixed t values (uniform grid) and averaged into low/mid/high buckets.
- **H4** uses α=1.0 (not α=0.5) to maximise the static baseline strength and provide the most stringent test for homotopy.

---

## Next steps

*(To be determined after go/no-go)*

**If GO/STRONG GO:**
1. Scale feature cache to 50K/5K and re-run.
2. Run seeds 1 and 2 for winning variant vs H1 and H4.
3. Consider Stage 2c: learned timestep schedule or tiny timestep mixer.
4. Evaluate FID-10k on the winning variant.

**If NO-GO:**
1. Investigate whether the failure is data-scale limited (5K is small).
2. Check decoder compatibility as a potential confounder.
3. Consider returning to representation/decoder-side interpretation only.

**If INCONCLUSIVE:**
1. Scale cache first, then re-run.
