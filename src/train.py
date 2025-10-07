# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
"""
import os
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging

import math
from torch.cuda.amp import autocast
from omegaconf import OmegaConf
from stage1 import RAE
from stage2.models import Stage2ModelProtocol
from stage2.transport import create_transport, Sampler
from train_utils import parse_configs
from model_utils import instantiate_from_config
from utils import wandb_utils


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################


def main(args):
    """Trains a new SiT model using config-driven hyperparameters."""
    if not torch.cuda.is_available():
        raise RuntimeError("Training currently requires at least one GPU.")

    (
        rae_config,
        model_config,
        transport_config,
        sampler_config,
        guidance_config,
        misc_config,
        training_config,
    ) = parse_configs(args.config)

    if rae_config is None or model_config is None:
        raise ValueError("Config must provide both stage_1 and stage_2 sections.")

    def to_dict(cfg_section):
        if cfg_section is None:
            return {}
        return OmegaConf.to_container(cfg_section, resolve=True)

    misc = to_dict(misc_config)
    transport_cfg = to_dict(transport_config)
    sampler_cfg = to_dict(sampler_config)
    guidance_cfg = to_dict(guidance_config)
    training_cfg = to_dict(training_config)

    num_classes = int(misc.get("num_classes", 1000))
    null_label = int(misc.get("null_label", num_classes))
    latent_size = tuple(int(dim) for dim in misc.get("latent_size", (768, 16, 16)))
    shift_dim = misc.get("time_dist_shift_dim", math.prod(latent_size))
    shift_base = misc.get("time_dist_shift_base", 4096)
    time_dist_shift = math.sqrt(shift_dim / shift_base)

    grad_accum_steps = int(training_cfg.get("grad_accum_steps", 1))
    ema_decay = float(training_cfg.get("ema_decay", 0.9995))
    epochs = int(training_cfg.get("epochs", 1400))
    global_batch_size = int(training_cfg.get("global_batch_size", 1024))
    num_workers = int(training_cfg.get("num_workers", 4))
    log_every = int(training_cfg.get("log_every", 100))
    ckpt_every = int(training_cfg.get("ckpt_every", 5_000))
    sample_every = int(training_cfg.get("sample_every", 10_000))
    base_lr = float(training_cfg.get("base_lr", 2e-4))
    final_lr = float(training_cfg.get("final_lr", 2e-5))
    decay_start_epoch = int(training_cfg.get("decay_start_epoch", 40))
    decay_end_epoch = int(training_cfg.get("decay_end_epoch", 800))
    cfg_scale_override = training_cfg.get("cfg_scale", None)
    default_seed = int(training_cfg.get("global_seed", 0))
    global_seed = args.global_seed if args.global_seed is not None else default_seed

    if grad_accum_steps < 1:
        raise ValueError("Gradient accumulation steps must be >= 1.")
    if args.image_size % 16 != 0:
        raise ValueError("Image size must be divisible by 16 for the RAE encoder.")

    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    if global_batch_size % (world_size * grad_accum_steps) != 0:
        raise ValueError("Global batch size must be divisible by world_size * grad_accum_steps.")
    rank = dist.get_rank()
    device_idx = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_idx)
    device = torch.device("cuda", device_idx)

    seed = global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if rank == 0:
        print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    micro_batch_size = global_batch_size // (world_size * grad_accum_steps)
    use_bf16 = args.precision == "bf16"
    if use_bf16 and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested bf16 precision, but the current CUDA device does not support bfloat16.")
    autocast_kwargs = dict(dtype=torch.bfloat16, enabled=use_bf16)
    latent_dtype = autocast_kwargs["dtype"] if use_bf16 else torch.float32

    transport_params = dict(transport_cfg.get("params", {}))
    path_type = transport_params.get("path_type", "Linear")
    prediction = transport_params.get("prediction", "velocity")
    loss_weight = transport_params.get("loss_weight")
    transport_params.pop("time_dist_shift", None)

    sampler_mode = sampler_cfg.get("mode", "ODE").upper()
    sampler_params = dict(sampler_cfg.get("params", {}))

    guidance_scale = float(guidance_cfg.get("scale", 1.0))
    if cfg_scale_override is not None:
        guidance_scale = float(cfg_scale_override)
    guidance_method = guidance_cfg.get("method", "cfg")

    def guidance_value(key: str, default: float) -> float:
        if key in guidance_cfg:
            return guidance_cfg[key]
        dashed_key = key.replace("_", "-")
        return guidance_cfg.get(dashed_key, default)

    t_min = float(guidance_value("t_min", 0.0))
    t_max = float(guidance_value("t_max", 1.0))

    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*")) - 1
        model_target = str(model_config.get("target", "stage2"))
        model_string_name = model_target.split(".")[-1]
        precision_suffix = f"-{args.precision}" if args.precision == "bf16" else ""
        loss_weight_str = loss_weight if loss_weight is not None else "none"
        experiment_name = (
            f"{experiment_index:03d}-{model_string_name}-"
            f"{path_type}-{prediction}-{loss_weight_str}{precision_suffix}-acc{grad_accum_steps}"
        )
        experiment_dir = os.path.join(args.results_dir, experiment_name)
        checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        if args.wandb:
            entity = os.environ["ENTITY"]
            project = os.environ["PROJECT"]
            wandb_utils.initialize(args, entity, experiment_name, project)
    else:
        experiment_dir = None
        checkpoint_dir = None
        logger = create_logger(None)

    rae: RAE = instantiate_from_config(rae_config).to(device)
    rae.eval()

    model: Stage2ModelProtocol = instantiate_from_config(model_config).to(device)
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)

    opt_state = None
    sched_state = None
    train_steps = 0

    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt, map_location="cpu")
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        if "ema" in checkpoint:
            ema.load_state_dict(checkpoint["ema"])
        opt_state = checkpoint.get("opt")
        sched_state = checkpoint.get("scheduler")
        train_steps = int(checkpoint.get("train_steps", 0))

    model_param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"Model Parameters: {model_param_count/1e6:.2f}M")

    model = DDP(model, device_ids=[device_idx], gradient_as_bucket_view=False)

    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0)
    if opt_state is not None:
        opt.load_state_dict(opt_state)

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=global_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    logger.info(
        f"Gradient accumulation: steps={grad_accum_steps}, micro batch={micro_batch_size}, "
        f"per-GPU batch={micro_batch_size * grad_accum_steps}, global batch={global_batch_size}"
    )
    logger.info(f"Precision mode: {args.precision}")

    loader_batches = len(loader)
    if loader_batches % grad_accum_steps != 0:
        raise ValueError("Number of loader batches must be divisible by grad_accum_steps when drop_last=True.")
    steps_per_epoch = loader_batches // grad_accum_steps
    if steps_per_epoch <= 0:
        raise ValueError("Gradient accumulation configuration results in zero optimizer steps per epoch.")
    decay_start_step = decay_start_epoch * steps_per_epoch
    decay_end_step = decay_end_epoch * steps_per_epoch
    final_lr_factor = final_lr / base_lr
    total_decay_steps = max(decay_end_step - decay_start_step, 1)

    def linear_lr_lambda(step: int) -> float:
        if step < decay_start_step:
            return 1.0
        if step >= decay_end_step:
            return final_lr_factor
        progress = (step - decay_start_step) / total_decay_steps
        return 1.0 - (1.0 - final_lr_factor) * progress

    schedl = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=linear_lr_lambda)
    if sched_state is not None:
        schedl.load_state_dict(sched_state)

    transport = create_transport(
        **transport_params,
        time_dist_shift=time_dist_shift,
    )
    transport_sampler = Sampler(transport)

    if sampler_mode == "ODE":
        eval_sampler = transport_sampler.sample_ode(**sampler_params)
    elif sampler_mode == "SDE":
        eval_sampler = transport_sampler.sample_sde(**sampler_params)
    else:
        raise NotImplementedError(f"Invalid sampling mode {sampler_mode}.")

    guid_model_forward = None
    if guidance_scale > 1.0 and guidance_method == "autoguidance":
        guidance_model_cfg = guidance_cfg.get("guidance_model")
        if guidance_model_cfg is None:
            raise ValueError("Please provide a guidance model config when using autoguidance.")
        guid_model: Stage2ModelProtocol = instantiate_from_config(guidance_model_cfg).to(device)
        guid_model.eval()
        guid_model_forward = guid_model.forward

    update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    log_steps = 0
    running_loss = 0.0
    start_time = time()

    ys = torch.randint(num_classes, size=(micro_batch_size,), device=device)
    using_cfg = guidance_scale > 1.0
    n = ys.size(0)
    zs = torch.randn(n, *latent_size, device=device, dtype=latent_dtype)

    if using_cfg:
        zs = torch.cat([zs, zs], dim=0)
        y_null = torch.full((n,), null_label, device=device)
        ys = torch.cat([ys, y_null], dim=0)
        sample_model_kwargs = dict(
            y=ys,
            cfg_scale=guidance_scale,
            cfg_interval=(t_min, t_max),
        )
        if guidance_method == "autoguidance":
            if guid_model_forward is None:
                raise RuntimeError("Guidance model forward is not initialized.")
            sample_model_kwargs["additional_model_forward"] = guid_model_forward
            model_fn = ema.forward_with_autoguidance
        else:
            model_fn = ema.forward_with_cfg
    else:
        sample_model_kwargs = dict(y=ys)
        model_fn = ema.forward

    logger.info(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        opt.zero_grad()
        accum_counter = 0
        step_loss_accum = 0.0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                x = rae.encode(x)
            if use_bf16:
                x = x.to(latent_dtype)
            model_kwargs = dict(y=y)
            with autocast(**autocast_kwargs):
                loss_tensor = transport.training_losses(model, x, model_kwargs)["loss"].mean()
            step_loss_accum += loss_tensor.item()
            (loss_tensor / grad_accum_steps).backward()
            accum_counter += 1

            if accum_counter < grad_accum_steps:
                continue

            opt.step()
            schedl.step()
            update_ema(ema, model.module, decay=ema_decay)
            opt.zero_grad()

            running_loss += step_loss_accum / grad_accum_steps
            log_steps += 1
            train_steps += 1
            accum_counter = 0
            step_loss_accum = 0.0

            if train_steps % log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / world_size
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                if args.wandb:
                    wandb_utils.log(
                        {"train loss": avg_loss, "train steps/sec": steps_per_sec},
                        step=train_steps,
                    )
                running_loss = 0.0
                log_steps = 0
                start_time = time()

            if train_steps % ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "scheduler": schedl.state_dict(),
                        "train_steps": train_steps,
                        "config_path": args.config,
                        "training_cfg": training_cfg,
                        "cli_overrides": {
                            "data_path": args.data_path,
                            "results_dir": args.results_dir,
                            "image_size": args.image_size,
                            "precision": args.precision,
                            "global_seed": global_seed,
                        },
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

            if train_steps % sample_every == 0 or train_steps == 1:
                logger.info("Generating EMA samples...")
                with torch.no_grad():
                    with autocast(**autocast_kwargs):
                        samples = eval_sampler(zs, model_fn, **sample_model_kwargs)[-1]
                    dist.barrier()

                    if using_cfg:
                        samples, _ = samples.chunk(2, dim=0)
                    samples = rae.decode(samples.to(torch.float32))
                    out_samples = torch.zeros(
                        (global_batch_size // grad_accum_steps, 3, args.image_size, args.image_size),
                        device=device,
                    )
                    dist.all_gather_into_tensor(out_samples, samples)
                    if args.wandb:
                        wandb_utils.log_image(out_samples, train_steps)
                logger.info("Generating EMA samples done.")

        if accum_counter != 0:
            raise RuntimeError("Gradient accumulation counter not zero at epoch end.")

    model.eval()
    logger.info("Done!")
    cleanup()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the training dataset root.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to store training outputs.")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256, help="Input image resolution.")
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="fp32", help="Compute precision for training.")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional checkpoint path to resume training.")
    parser.add_argument("--global-seed", type=int, default=None, help="Override training.global_seed from the config.")
    args = parser.parse_args()
    main(args)
