"""Manual stage 2 diffusion sampling using the deterministic loop from stage2_inference.

The script reuses a stored latent noise tensor (creating it if necessary),
iterates the hand-written timestep schedule, and decodes the final latent with
the stage 1 decoder. Outputs include both the latent and decoded image in an
``.npz`` archive plus a PNG snapshot.
"""
import argparse
import math
import os
import sys
from time import time

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from download import find_model  # noqa: E402
from stage1 import RAE  # noqa: E402
from stage2.model import STAGE2_ARCHS  # noqa: E402


def load_or_create_noise(noise_path: str, shape, device: torch.device) -> torch.Tensor:
    """Load noise from ``noise_path`` or create it with the requested ``shape``."""
    noise_dir = os.path.dirname(noise_path)
    if noise_dir:
        os.makedirs(noise_dir, exist_ok=True)

    if os.path.exists(noise_path):
        with np.load(noise_path) as data:
            key = "noise" if "noise" in data.files else data.files[0]
            noise = data[key]
    else:
        noise = torch.randn(*shape, device=device).cpu().numpy()
        np.savez(noise_path, noise=noise)
        print(f"Created new noise file at {noise_path}.")

    if noise.shape != tuple(shape):
        raise ValueError(
            f"Noise at {noise_path} has shape {noise.shape}, expected {tuple(shape)}."
        )

    return torch.from_numpy(noise).to(device=device, dtype=torch.float32)


def save_sample(
    image_path: str,
    png_path: str,
    latent: torch.Tensor,
    sample: torch.Tensor,
    class_label: int,
) -> None:
    """Persist latent and decoded tensors to npz, plus a PNG render."""
    image_dir = os.path.dirname(image_path)
    if image_dir:
        os.makedirs(image_dir, exist_ok=True)

    png_dir = os.path.dirname(png_path)
    if png_dir:
        os.makedirs(png_dir, exist_ok=True)

    latent_np = latent.detach().cpu().numpy().astype(np.float32)
    sample_np = sample.detach().cpu().numpy().astype(np.float32)
    np.savez(
        image_path,
        latent=latent_np,
        image=sample_np,
        class_label=np.array([class_label]),
        mode=np.array(["manual"]),
    )
    print(f"Saved decoded sample to {image_path} (shape={sample_np.shape}).")

    png_image = sample_np.clip(0.0, 1.0)
    png_image = (png_image * 255.0).astype(np.uint8)
    png_image = np.transpose(png_image, (1, 2, 0))
    Image.fromarray(png_image).save(png_path)
    print(f"Saved PNG render to {png_path}.")


def build_models(args, device: torch.device):
    """Instantiate the stage 2 diffusion model and the stage 1 decoder."""
    latent_size = args.image_size // 16
    model = STAGE2_ARCHS[args.model](
        token_dim=768,
        input_size=latent_size,
    ).to(device)

    ckpt_path = args.ckpt or f"SiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()

    rae = RAE(
        encoder_cls="Dinov2withNorm",
        encoder_config_path="models/encoders/dinov2/wReg_base",
        encoder_input_size=224,
        encoder_params={"dinov2_path": "models/encoders/dinov2/wReg_base", "normalize": True},
        decoder_config_path="configs/decoder/ViTXL",
        pretrained_decoder_path="models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt",
        noise_tau=0.0,
        reshape_to_2d=True,
        normalization_stat_path="models/stats/dinov2/wReg_base/imagenet1k/stat.pt",
    ).to(device)
    rae.eval()
    return model, rae, latent_size


def make_timesteps(num_steps: int, t_min: float, t_max: float, shift: float) -> torch.Tensor:
    schedule = torch.linspace(t_min, t_max, steps=num_steps, dtype=torch.float32)
    schedule = shift * schedule / (1 + (shift - 1) * schedule)
    return schedule


def manual_sample(stage2_model, z_init, y, schedule, device: torch.device) -> torch.Tensor:
    zt = z_init
    with torch.no_grad():
        iterator = range(len(schedule) - 1, 0, -1)
        for i in tqdm(iterator, desc="Sampling", total=len(schedule) - 1):
            t_cur = torch.full((zt.shape[0],), schedule[i], device=device, dtype=zt.dtype)
            t_prev = torch.full((zt.shape[0],), schedule[i - 1], device=device, dtype=zt.dtype)
            v_pred = stage2_model(zt, t_cur, y)
            delta = (t_prev - t_cur).view(-1, 1, 1, 1)
            zt = zt + delta * v_pred
    return zt


def run(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(args.cuda_device)
        print(f"Using GPU device {torch.cuda.current_device()}.")
    else:
        print("Using CPU.")

    stage2_model, rae, latent_size = build_models(args, device)

    noise_shape = (1, 768, latent_size, latent_size)
    z0 = load_or_create_noise(args.noise_path, noise_shape, device)

    y = torch.tensor([args.class_label], device=device, dtype=torch.long)

    rae_dim = 768 * latent_size * latent_size
    time_dist_shift = math.sqrt(rae_dim / args.time_dist_shift_base)
    schedule = make_timesteps(args.num_steps, args.t_min, args.t_max, time_dist_shift)

    start = time()
    latent_samples = manual_sample(stage2_model, z0, y, schedule, device)
    elapsed = time() - start
    print(f"Sampling finished in {elapsed:.2f}s.")

    decoded = rae.decode(latent_samples).clamp_(0.0, 1.0)
    save_sample(args.image_path, args.png_path, latent_samples[0], decoded[0], args.class_label)


def parse_args() -> argparse.Namespace:
    default_data_dir = os.path.join(os.path.dirname(__file__), "data")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, default="DDTXL")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--class-label", type=int, default=207)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--t-min", type=float, default=1.0 / 1000.0)
    parser.add_argument("--t-max", type=float, default=1.0)
    parser.add_argument("--time-dist-shift-base", type=float, default=4096.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument(
        "--noise-path",
        type=str,
        default=os.path.join(default_data_dir, "stage2_initial_noise.npz"),
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default=os.path.join(default_data_dir, "stage2_manual_sample.npz"),
    )
    parser.add_argument(
        "--png-path",
        type=str,
        default=None,
    )

    args = parser.parse_args()
    if args.png_path is None:
        stem, _ = os.path.splitext(args.image_path)
        args.png_path = stem + ".png"
    return args


if __name__ == "__main__":
    run(parse_args())
