#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

python3 -m scripts.benchmark_constraint_scenarios \
  --cfg configs/default.yaml \
  --device cuda:0 \
  --scenarios hard_stress \
  --variants hybrid \
  --ablations full wo_meta wo_lagrangian hard_clip \
  --baselines heuristic_safe sac_lagrangian \
  --meta-iters 100 \
  --seeds 101 202 303 404 505 606 707 808 909 1001 \
  --eval-tasks 10 \
  --eval-eps 3 \
  --env-tasks 8 \
  --env-eps 1 \
  --strict-meta \
  --no-shared-init \
  --out-dir logs/hard_stress_full_ablation_baseline_v2
