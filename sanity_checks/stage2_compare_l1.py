"""Compare Stage 2 sampling outputs using L1 distances.

Loads two ``.npz`` files produced by stage2 sanity scripts (containing
``latent`` and ``image`` tensors) and computes the L1 difference between the
latents and decoded images. The script reports aggregate metrics and optional
per-channel summaries.
"""
import argparse
import os
import sys
from typing import Tuple

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def load_npz(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load latent and image arrays from ``path``."""
    with np.load(path) as data:
        if "latent" not in data or "image" not in data:
            raise KeyError(f"{path} is missing 'latent' or 'image' entries.")
        latent = data["latent"]
        image = data["image"]
    return latent, image


def l1_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def channel_wise_l1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(a - b), axis=(-2, -1))


def compare_npzs(path_a: str, path_b: str, verbose: bool) -> None:
    latent_a, image_a = load_npz(path_a)
    latent_b, image_b = load_npz(path_b)

    if latent_a.shape != latent_b.shape:
        raise ValueError(f"Latent shapes differ: {latent_a.shape} vs {latent_b.shape}")
    if image_a.shape != image_b.shape:
        raise ValueError(f"Image shapes differ: {image_a.shape} vs {image_b.shape}")

    latent_l1 = l1_distance(latent_a, latent_b)
    image_l1 = l1_distance(image_a, image_b)

    print(f"Latent L1 distance: {latent_l1:.6f}")
    print(f"Image  L1 distance: {image_l1:.6f}")

    if verbose:
        latent_channels = channel_wise_l1(latent_a, latent_b)
        image_channels = channel_wise_l1(image_a, image_b)
        for idx, value in enumerate(latent_channels.flatten()):
            print(f"  latent channel {idx}: L1={value:.6f}")
        for idx, value in enumerate(image_channels.flatten()):
            print(f"  image  channel {idx}: L1={value:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz-a", required=True, help="Path to first npz file")
    parser.add_argument("--npz-b", required=True, help="Path to second npz file")
    parser.add_argument("--verbose", action="store_true", help="Print per-channel L1 distances")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compare_npzs(args.npz_a, args.npz_b, args.verbose)
