# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained SiT.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from download import find_model
from stage1 import RAE
from stage2.model import STAGE2_ARCHS, DiTwDDTHead
from train_utils import parse_ode_args, parse_sde_args, parse_transport_args
from transport import create_transport, Sampler
import argparse
from time import time
import math
def main(mode, args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DDTXL", "Only DDTXL models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000
        # assert args.image_size == 256, "512x512 models are not yet available for auto-download." # remove this line when 512x512 models are available
        learn_sigma = args.image_size == 256
    else:
        learn_sigma = False

    # Load model:
    latent_size = args.image_size // 16
    model = STAGE2_ARCHS[args.model](
        token_dim=768,  # Assuming the latent token dimension from stage 1
        input_size=16,  # Assuming the latent size from stage 1 is 32x32 for 256x256 input
    ).to(device)
    # Auto-download a pre-trained model or load a custom SiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"SiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path, ema_only=True)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    rae = RAE(
        encoder_cls='Dinov2withNorm',
        encoder_config_path='models/encoders/dinov2/wReg_base',
        encoder_input_size=224,
        encoder_params={'dinov2_path': 'models/encoders/dinov2/wReg_base', 'normalize': True},
        decoder_config_path='configs/decoder/ViTXL',
        pretrained_decoder_path='models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt',
        noise_tau=0.,
        reshape_to_2d=True,   
        normalization_stat_path='models/stats/dinov2/wReg_base/imagenet1k/stat.pt',
    ).to(device)
    rae.eval()
    rae_dim = 768 * 16 * 16
    args.time_dist_shift = math.sqrt(rae_dim/args.time_dist_shift_base)  # default base=4096, for 256x256 images with latent size 16x16
    print(f"Using time_dist_shift={args.time_dist_shift:.4f} based on latent dimension {rae_dim}.")
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
    if mode == "ODE":
        sample_fn = sampler.sample_ode(
            sampling_method=args.sampling_method,
            num_steps=args.num_sampling_steps,
            atol=args.atol,
            rtol=args.rtol,
            reverse=args.reverse
        )
            
    elif mode == "SDE":
        sample_fn = sampler.sample_sde(
            sampling_method=args.sampling_method,
            diffusion_form=args.diffusion_form,
            diffusion_norm=args.diffusion_norm,
            last_step=args.last_step,
            last_step_size=args.last_step_size,
            num_steps=args.num_sampling_steps,
        )
    
    
    # Labels to condition the model with (feel free to change):
    class_labels = [207, 360]
    
    # Create sampling noise:
    n = len(class_labels)
    z = torch.randn(n, 768, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    if args.cfg_scale > 1.0:
        model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
        model_fwd = model.forward_with_cfg
    else:
        model_kwargs = dict(y=y)
        model_fwd = model.forward
    # Sample images:
    start_time = time()
    samples = sample_fn(z, model_fwd, **model_kwargs)[-1]
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    # samples = vae.decode(samples / 0.18215).sample
    samples = rae.decode(samples)
    print(f"Sampling took {time() - start_time:.2f} seconds.")

    # Save and display images:
    save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(0, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a SiT checkpoint (default: auto-download a pre-trained SiT-XL/2 model).")


    parse_transport_args(parser)
    if mode == "ODE":
        parse_ode_args(parser)
        # Further processing for ODE
    elif mode == "SDE":
        parse_sde_args(parser)
        # Further processing for SDE
    
    args = parser.parse_known_args()[0]
    main(mode, args)
