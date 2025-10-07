# RAE Stage 2 Quickstart

This repository contains the Stage 2 latent diffusion training and sampling
utilities used alongside the Stage 1 RAE encoder/decoder. The commands below
highlight how to launch distributed training, run single-GPU sampling, and
produce large evaluation batches with DDP sampling.

## Environment

### Standard Setup (A100, CUDA 11.7+)

1. Install the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate rae
   ```
2. Download the Stage 1 encoder/decoder checkpoints referenced under
   `models/` (follow the instructions in `environment.yml` or the upstream
   paper repository).
3. Ensure image data is arranged in an ImageNet-style folder tree for
   `torchvision.datasets.ImageFolder`.

### H100 Setup (CUDA 12.1+)

If you encounter `iJIT_NotifyEvent` or NumPy 2.x errors, use this clean install:

1. Create environment and install via `uv`:
   ```bash
   conda create -n rae python=3.10 -y
   conda activate rae
   pip install uv
   
   # Install PyTorch 2.2.0 with CUDA 12.1
   uv pip install torch==2.2.0 torchvision==0.17.0 torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # Install other dependencies
   uv pip install timm==0.9.16 accelerate==0.23.0 torchdiffeq==0.2.5 wandb
   uv pip install "numpy<2" transformers einops
   ```

2. (Optional) Load CUDA 12.4 if needed:
   ```bash
   module load cuda/12.4
   ```

3. Download Stage 1 checkpoints and prepare data as in standard setup.

## Data & Model Preparation

### Download Pre-trained Models

The Stage 1 encoder uses `facebook/dinov2-with-registers-base` (downloaded automatically from HuggingFace), but you need to manually download the decoder and normalization statistics:

```bash
cd RAE

# Install huggingface_hub if not already installed
pip install huggingface_hub

# Download decoder model and normalization stats
huggingface-cli download nyu-visionx/RAE-models \
  decoders/dinov2/wReg_base/ViTXL_n08/model.pt \
  --local-dir models --local-dir-use-symlinks False

huggingface-cli download nyu-visionx/RAE-models \
  stats/dinov2/wReg_base/imagenet1k/stat.pt \
  --local-dir models --local-dir-use-symlinks False
```

### Prepare Dataset

Download standard ImageNet-1k and put it to somewhere you like.

Then it should be directly runnable.


## Distributed Training

Stage 2 training now lives in `src/train.py` and is configured via a YAML file
(see `configs/training`). Launch it with `torchrun` (PyTorch DDP). Key CLI
switches:

- `--config`: path to a config describing Stage 1/2 models, transport, sampler,
  guidance, and training hyperparameters.
- `--data-path`: root of the training images (ImageFolder format).
- `--results-dir`: directory where experiment logs/checkpoints are written.
- `--image-size`: input resolution; must match the Stage 1 encoder.
- `--precision {fp32,bf16}`: enable bfloat16 autocast on supported GPUs.
- `--wandb`: opt-in to Weights & Biases logging (expects `ENTITY`/`PROJECT`).
- `--ckpt`: optional checkpoint to resume from.
- `--global-seed`: override `training.global_seed` from the config file.

Global batch size, gradient accumulation, LR schedule, EMA decay, etc. are
pulled from the config's `training` block, so edit those YAML files to change
defaults (examples live under `configs/training/ImageNet256/`).

Example: single node, 4 GPUs, bf16 enabled:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  src/train.py --config configs/training/ImageNet256/DiTDH-S_DINOv2-B.yaml \
  --data-path /data/boyang/imagenet/train --precision bf16
```

Training checkpoints are written to `results/<run-name>/checkpoints/`. The run
name automatically encodes precision (`-bf16`) and other run metadata.

## Single-GPU Sampling

`src/sample.py` generates images on a single GPU/CPU using the same config
schema as training. Provide a config that defines Stage 1/2, transport, sampler
mode, and optional guidance. Useful flags:

- `--config`: path to the sampling config (see `configs/models` or training configs).
- `--seed`: random seed for latent sampling (default 0).

The script saves the decoded grid to `sample.png` in the working directory.
Example:

```bash
python src/sample.py --config configs/models/DiTDH-XL_DINOv2-B_decXL.yaml --seed 42
```

## Distributed Sampling for Evaluation

`src/sample_ddp.py` parallelises sampling across multiple GPUs and writes
PNG files plus an `.npz` payload suitable for FID evaluation. Launch it with
`torchrun`, reusing the same config structure as `sample.py`.

Key CLI switches:

- `--config`: path to a sampling config.
- `--sample-dir`: base directory for sample PNGs and the resulting `.npz`.
- `--per-proc-batch-size`: batch size per GPU.
- `--num-fid-samples`: number of images to generate (default 50k).
- `--precision {fp32,bf16}` and `--tf32/--no-tf32`: numerical controls.
- `--global-seed`: per-rank seeding (defaults to config value or 0).

Example command to draw 50k samples on 4 GPUs:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  src/sample_ddp.py --config configs/models/DiTDH-XL_DINOv2-B_decXL.yaml \
  --sample-dir samples --per-proc-batch-size 8 --num-fid-samples 50000
```

Once sampling finishes, rank 0 aggregates the PNGs into `samples/<run>.npz`.
You can pass that file to downstream FID tooling (e.g., the ADM evaluation
scripts).

Autoguidance is also supported. To enable autoguidance, simply pass in:

- `--cfg-scale`: the guidance scale
- `--guid-model`: model size used for guiding generation
- `--guid-model-ckpt`: the path to the checkpoint of the guidance model

## Tips

- Stage 2 assumes Stage 1 assets are available locally; verify the paths in
  `src/train.py` and `src/sample*.py` before launching jobs.
- If your dataset folder is large, bump `training.num_workers` in the config to
  fully utilise CPU prefetch.
- To resume training from a checkpoint, pass `--ckpt <path>` to `train.py`.
