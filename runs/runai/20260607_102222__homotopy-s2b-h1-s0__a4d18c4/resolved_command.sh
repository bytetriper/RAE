#!/usr/bin/env bash
# Homotopy Stage 2b — H1: Static fused L4 baseline (seed 0)
set -euo pipefail

REPO_DIR="${REPO_DIR:-/storage/nada/RAE}"
SHA="a4d18c4"
RUN_DIR="${RUN_DIR:-$REPO_DIR/runs/runai/20260607_102222__homotopy-s2b-h1-s0__${SHA}}"
CACHE_DIR="$REPO_DIR/artifacts/homotopy_stage2a/feature_cache"
STREAM_DIR="$REPO_DIR/artifacts/homotopy_stage2a/stream_params"
PROBE_DIR="$REPO_DIR/artifacts/homotopy_stage2b/results"

mkdir -p "$RUN_DIR" "$PROBE_DIR"
cd "$REPO_DIR"

apt-get update -qq 2>/dev/null; apt-get install -y -qq git 2>/dev/null || true
{
  echo "DATE=$(date -Iseconds)"
  echo "HOSTNAME=$(hostname)"
  echo "GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  python --version 2>&1
  python -c "import torch; print('TORCH', torch.__version__); print('CUDA', torch.cuda.is_available())" 2>&1 || true
  nvidia-smi 2>/dev/null || echo "no GPU"
} | tee "$RUN_DIR/env_snapshot.txt"

pip install -q --no-warn-script-location --break-system-packages \
  "timm==0.9.16" "transformers==4.56.2" "einops" "omegaconf==2.3.0" \
  "numpy==1.26.0" "torchvision" "wandb" "scipy" "Pillow" "regex"
export HF_HOME=/storage/nada/runai_hf_cache

export WANDB_ANONYMOUS="${WANDB_ANONYMOUS:-allow}"
if [ -f "$REPO_DIR/.secrets/wandb.env" ]; then set -a; source "$REPO_DIR/.secrets/wandb.env"; set +a; fi
export WANDB_ENTITY="${WANDB_ENTITY:-nadavo11}"
export WANDB_PROJECT="${WANDB_PROJECT:-homotopy-rae}"

SHA_RUN=$(git rev-parse --short HEAD 2>/dev/null || echo "$SHA")

python -u scripts/train_homotopy_stage2b_generator.py \
  --cache-dir "$CACHE_DIR" \
  --stream-params-dir "$STREAM_DIR" \
  --results-dir "$PROBE_DIR" \
  --detail-layer 4 \
  --variant H1 \
  --alpha 0.5 \
  --gamma 1 \
  --n-steps 20000 \
  --batch-size 128 \
  --lr 1e-4 \
  --weight-decay 0.0 \
  --warmup-steps 1000 \
  --clip-grad 1.0 \
  --ema-decay 0.9999 \
  --seed 0 \
  --precision bf16 \
  --val-every 2000 \
  --sample-every 4000 \
  --ckpt-every 5000 \
  --num-val-steps 20 \
  --sample-steps 50 \
  --wandb \
  --wandb-group "stage2b-fixed-homotopy-$SHA_RUN" \
  2>&1 | tee "$RUN_DIR/train.log"

echo "==> H1 training done."
ls -lh "$PROBE_DIR"
