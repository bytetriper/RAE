"""Sanity script to run stage 2 diffusion and store a single decoded sample.

The script loads an initial latent noise tensor from ``--noise-path``. If the
file does not exist yet, new Gaussian noise is generated, saved, and reused for
future runs. The sampled latent and decoded RGB output are written to
``--image-path`` as a NumPy ``.npz`` archive (with metadata) and also exported
as a PNG file.
"""
import argparse
import math
import os
import sys
from time import time

import numpy as np
import torch
from PIL import Image

# Allow importing project modules when running from repo root.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from download import find_model  # noqa: E402
from stage1 import RAE  # noqa: E402
from stage2.model import STAGE2_ARCHS  # noqa: E402
from stage2.transport import Sampler, create_transport  # noqa: E402
from stage2.train_utils import (  # noqa: E402
    parse_ode_args,
    parse_sde_args,
    parse_transport_args,
)


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
    mode: str,
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
        mode=np.array([mode]),
    )
    print(f"Saved decoded sample to {image_path} (shape={sample_np.shape}).")

    png_image = sample_np.clip(0.0, 1.0)
    png_image = (png_image * 255.0).astype(np.uint8)
    png_image = np.transpose(png_image, (1, 2, 0))  # (H, W, C)
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


def run_sampling(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    force_cpu = os.environ.get("RAE_FORCE_CPU", "0") == "1"
    device = torch.device("cpu" if force_cpu else "cuda" if torch.cuda.is_available() else "cpu")

    model, rae, latent_size = build_models(args, device)

    rae_dim = 768 * latent_size * latent_size
    args.time_dist_shift = math.sqrt(rae_dim / args.time_dist_shift_base)
    print(f"Using time_dist_shift={args.time_dist_shift:.4f} for latent size {latent_size}.")

    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps,
        args.time_dist_type,
        args.time_dist_shift,
    )
    sampler = Sampler(transport)

    if args.mode == "ODE":
        sample_fn = sampler.sample_ode(
            sampling_method=args.sampling_method,
            num_steps=args.num_sampling_steps,
            atol=args.atol,
            rtol=args.rtol,
            reverse=args.reverse,
        )
    else:
        sample_fn = sampler.sample_sde(
            sampling_method=args.sampling_method,
            diffusion_form=args.diffusion_form,
            diffusion_norm=args.diffusion_norm,
            last_step=args.last_step,
            last_step_size=args.last_step_size,
            num_steps=args.num_sampling_steps,
        )

    noise_shape = (1, 768, latent_size, latent_size)
    z = load_or_create_noise(args.noise_path, noise_shape, device)

    y = torch.tensor([args.class_label], device=device)
    if args.cfg_scale > 1.0:
        z = torch.cat([z, z], dim=0)
        y_null = torch.full_like(y, fill_value=args.null_class)
        y = torch.cat([y, y_null], dim=0)
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
        model_forward = model.forward_with_cfg
    else:
        model_kwargs = dict(y=y)
        model_forward = model.forward

    start_time = time()
    latent_samples = sample_fn(z, model_forward, **model_kwargs)[-1]
    if args.cfg_scale > 1.0:
        latent_samples, _ = latent_samples.chunk(2, dim=0)
    elapsed = time() - start_time
    print(f"Sampling finished in {elapsed:.2f}s.")
    
    decoded = rae.decode(latent_samples).clamp_(0.0, 1.0)
    save_sample(
        args.image_path,
        args.png_path,
        latent_samples[0],
        decoded[0],
        args.class_label,
        args.mode,
    )


def parse_args() -> argparse.Namespace:
    default_data_dir = os.path.join(os.path.dirname(__file__), "data")
    parser = argparse.ArgumentParser(description=__doc__)
    if len(sys.argv) < 2:
        print("Usage: program.py <mode> [options]")
        sys.exit(1)
    
    mode = sys.argv[1]
    assert mode[:2] != "--", "Usage: program.py <mode> [options]"
    assert mode in ["ODE", "SDE"], "Invalid mode. Please choose 'ODE' or 'SDE'"
    parser.add_argument("--model", type=str, default="DDTXL")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--class-label", type=int, default=207)
    parser.add_argument("--null-class", type=int, default=1000)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument(
        "--noise-path",
        type=str,
        default=os.path.join(default_data_dir, "stage2_initial_noise.npz"),
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default=os.path.join(default_data_dir, "stage2_decoded_sample.npz"),
    )
    parser.add_argument(
        "--png-path",
        type=str,
        default=None,
    )

    parse_transport_args(parser)
    if mode == "ODE":
        parse_ode_args(parser)
        # Further processing for ODE
    elif mode == "SDE":
        parse_sde_args(parser)
        # Further processing for SDE
    args = parser.parse_known_args()[0]
    if args.png_path is None:
        stem, _ = os.path.splitext(args.image_path)
        args.png_path = stem + ".png"
    args.mode = mode
    return args


if __name__ == "__main__":
    args = parse_args()
    run_sampling(args)
