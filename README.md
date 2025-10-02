# RAE Stage 2 Quickstart

This repository contains the Stage 2 latent diffusion training and sampling
utilities used alongside the Stage 1 RAE encoder/decoder. The commands below
highlight how to launch distributed training, run single-GPU sampling, and
produce large evaluation batches with DDP sampling.

## Environment

1. Install the conda environment (CUDA 11.7+ recommended):
   ```bash
   conda env create -f environment.yml
   conda activate rae
   ```
2. Download the Stage 1 encoder/decoder checkpoints referenced under
   `models/` (follow the instructions in `environment.yml` or the upstream
   paper repository).
3. Ensure image data is arranged in an ImageNet-style folder tree for
   `torchvision.datasets.ImageFolder`.

## Data & Model Preparation

Download `/data/boyang/models/` on 74 and `ln -s` to `RAE/models/` 

If you don't have access to 74, you can also download the folder from : `gs://boyang-data-east/gpu_models/models/` using `gsutil -m cp -r gs://boyang-data-east/gpu_models/models/ RAE/`

Download standard ImageNet-1k and put it to somewhere you like.

Then it should be directly runnable.


## Distributed Training

Stage 2 training is implemented in `stage2/train.py` and expects to be launched
with `torchrun` (PyTorch DDP). Key options:

- `--data-path`: root of the training images (ImageFolder format).
- `--results-dir`: directory where experiment logs/checkpoints are written.
- `--global-batch-size`: total batch across all GPUs and accumulation steps.
- `--grad-accum-steps`: gradient accumulation factor (default 1).
- `--precision {fp32,bf16}`: enable bfloat16 autocast on supported GPUs.

Example: 8 GPUs, global batch 1024, gradient accumulation 1, bf16 enabled:

```bash
torchrun --nproc_per_node=8 stage2/train.py \
  --data-path <path-to-imagenet-train-split> \
  --results-dir results \
  --model DDTS \
  --global-batch-size 1024 \
  --grad-accum-steps 1 \
  --precision bf16 \
  --wandb
```

Training checkpoints are written to `results/<run-name>/checkpoints/`. The run
name will automatically include `-bf16` when mixed-precision is enabled.

Current selections for `--model` include `DDTS, DDTXL, DiTXL`

## Single-GPU Sampling

`stage2/sample.py` generates images on a single GPU/CPU. You must choose an
ODE or SDE sampler as the first positional argument. Useful flags:

- `--ckpt`: path to a Stage 2 checkpoint (defaults to auto-downloading a
  pretrained SiT-XL/2 model).
- `--cfg-scale`: classifier-free guidance scale (default 4.0).
- `--num-sampling-steps`: sampler steps.
- Transport/solver flags come from `train_utils.parse_transport_args` and the
  respective ODE/SDE parsers.

Example ODE sampling run that saves `sample.png`:

```bash
python stage2/sample.py ODE --cfg-scale 1.0 --ckpt models/DiTs/Dinov2/wReg_base/DDTXL/stage2_model.pt --num-sampling-steps 50 --sampling-method euler
```

## Distributed Sampling for Evaluation

`stage2/sample_ddp.py` parallelises sampling across multiple GPUs and writes
PNG files plus an `.npz` payload suitable for FID evaluation. Launch it with
`torchrun` the same way as training. Important arguments:

- `--sample-dir`: base directory for sample PNGs and the resulting `.npz`.
- `--per-proc-batch-size`: batch size per GPU.
- `--num-fid-samples`: number of images to generate (default 50k).
- `--tf32/--no-tf32`: toggle TensorFloat-32 matmuls.

Example command to draw 50k ODE samples on 8 GPUs:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  stage2/sample_ddp.py ODE --cfg-scale 1.0 --ckpt models/DiTs/Dinov2/wReg_base/DDTXL/stage2_model.pt --num-sampling-steps 50 --sampling-method euler --per-proc-batch-size 8
```

Once sampling finishes, rank 0 aggregates the PNGs into `samples/<run>.npz`.
You can pass that file to downstream FID tooling (e.g., the ADM evaluation
scripts).

## Tips

- Stage 2 assumes Stage 1 assets are available locally; verify the paths in
  `stage2/train.py` and `stage2/sample*.py` before launching jobs.
- If your dataset folder is large, set `--num-workers` to fully utilise CPU
  prefetch.
- To resume training from a checkpoint, pass `--ckpt <path>` to `train.py`.
