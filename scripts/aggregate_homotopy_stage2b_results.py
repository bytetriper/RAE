#!/usr/bin/env python
"""Aggregate Stage 2b results across variants and seeds.

Reads summary.json files from each variant directory, computes:
  - Final val loss (low/mid/high/all) per variant per seed
  - Mean and std across seeds (if multiple)
  - Go/no-go decision

Usage:
    python scripts/aggregate_homotopy_stage2b_results.py \
        --results-dir /storage/nada/RAE/artifacts/homotopy_stage2b/probe_results \
        --output-dir results/homotopy_stage2b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


VARIANTS = ["H0", "H1", "H2", "H3", "H4"]

# Go/no-go thresholds (relative improvement over H1)
LOW_NOISE_IMPROVEMENT_THRESHOLD = 0.05   # 5% relative
HIGH_NOISE_DEGRADATION_LIMIT = 0.02      # 2% relative max degradation


def load_summary(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def go_no_go(
    h1_low: float,
    h1_high: float,
    hx_low: float,
    hx_high: float,
    h4_low: float,
) -> str:
    """Evaluate go/no-go for a homotopy variant (H2 or H3) vs H1 and H4."""
    if h1_low <= 0:
        return "INCONCLUSIVE"

    low_improvement = (h1_low - hx_low) / h1_low
    high_degradation = max(0, (hx_high - h1_high) / max(abs(h1_high), 1e-9))

    beats_h1_low = low_improvement >= LOW_NOISE_IMPROVEMENT_THRESHOLD
    safe_high = high_degradation <= HIGH_NOISE_DEGRADATION_LIMIT
    beats_h4 = hx_low <= h4_low

    if not safe_high:
        return "NO-GO (high-noise degraded)"
    if not beats_h1_low:
        return "NO-GO (no low-noise improvement)"
    if not beats_h4:
        return "NO-GO (does not beat param-matched static)"

    return "GO"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Collect summaries
    rows = []
    by_variant: dict[str, list[dict]] = {v: [] for v in VARIANTS}

    for variant in VARIANTS:
        for seed in args.seeds:
            pattern = f"*{variant.lower()}*s{seed}*/summary.json"
            matches = list(args.results_dir.rglob(pattern))
            if not matches:
                # Also try direct path
                direct = args.results_dir / f"{variant.lower()}_L4_a0.50_s{seed}" / "summary.json"
                matches = [direct] if direct.exists() else []
            if not matches:
                print(f"WARNING: no summary for {variant} seed {seed}")
                continue
            summary = load_summary(matches[0])
            if summary is None:
                continue
            fv = summary.get("final_val", {})
            row = {
                "variant": variant,
                "seed": seed,
                "val_all":   fv.get("val/loss_all", float("nan")),
                "val_low":   fv.get("val/loss_low_noise", float("nan")),
                "val_mid":   fv.get("val/loss_mid_noise", float("nan")),
                "val_high":  fv.get("val/loss_high_noise", float("nan")),
                "n_steps":   summary.get("n_steps", 0),
                "n_params":  summary.get("n_params", 0),
                "wandb_url": summary.get("wandb_run_url", ""),
            }
            rows.append(row)
            by_variant[variant].append(row)

    if not rows:
        print("No results found.")
        return

    # Print results table
    header = f"{'Variant':<8} {'Seed':>4} {'val_all':>8} {'val_low':>8} {'val_mid':>8} {'val_high':>8}"
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['variant']:<8} {r['seed']:>4} "
            f"{r['val_all']:>8.4f} {r['val_low']:>8.4f} "
            f"{r['val_mid']:>8.4f} {r['val_high']:>8.4f}"
        )

    # Multi-seed aggregation
    agg: dict[str, dict] = {}
    for variant, vrows in by_variant.items():
        if not vrows:
            continue
        vals = {k: [r[k] for r in vrows] for k in ("val_all", "val_low", "val_mid", "val_high")}
        agg[variant] = {
            "mean": {k: float(np.mean(vs)) for k, vs in vals.items()},
            "std": {k: float(np.std(vs)) for k, vs in vals.items()},
            "n_seeds": len(vrows),
        }

    print("\n=== Aggregated (mean ± std across seeds) ===")
    print(f"{'Variant':<8} {'val_all':>14} {'val_low':>14} {'val_mid':>14} {'val_high':>14}")
    for variant in VARIANTS:
        if variant not in agg:
            continue
        m = agg[variant]["mean"]
        s = agg[variant]["std"]
        def fmt(k): return f"{m[k]:.4f}±{s[k]:.4f}"
        print(f"{variant:<8} {fmt('val_all'):>14} {fmt('val_low'):>14} "
              f"{fmt('val_mid'):>14} {fmt('val_high'):>14}")

    # Go/no-go
    print("\n=== Go / No-Go ===")
    verdicts = {}
    h1_agg = agg.get("H1", {}).get("mean", {})
    h4_agg = agg.get("H4", {}).get("mean", {})

    for variant in ("H2", "H3"):
        if variant not in agg:
            verdicts[variant] = "NO DATA"
            continue
        hx_agg = agg[variant]["mean"]
        v = go_no_go(
            h1_low=h1_agg.get("val_low", float("nan")),
            h1_high=h1_agg.get("val_high", float("nan")),
            hx_low=hx_agg.get("val_low", float("nan")),
            hx_high=hx_agg.get("val_high", float("nan")),
            h4_low=h4_agg.get("val_low", float("nan")),
        )
        verdicts[variant] = v
        print(f"  {variant}: {v}")

    # Save full output
    output = {
        "rows": rows,
        "aggregated": agg,
        "verdicts": verdicts,
        "thresholds": {
            "low_noise_improvement": LOW_NOISE_IMPROVEMENT_THRESHOLD,
            "high_noise_degradation_limit": HIGH_NOISE_DEGRADATION_LIMIT,
        },
    }
    out_path = args.output_dir / "aggregated_results.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
