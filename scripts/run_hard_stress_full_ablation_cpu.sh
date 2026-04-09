#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1

python3 -m scripts.benchmark_constraint_scenarios \
  --cfg configs/default.yaml \
  --device cpu \
  --scenarios hard_stress \
  --variants hybrid \
  --ablations full wo_meta wo_lagrangian hard_clip \
  --baselines heuristic_safe sac_lagrangian \
  --meta-iters 80 \
  --seeds 101 202 303 404 505 \
  --eval-tasks 8 \
  --eval-eps 2 \
  --env-tasks 6 \
  --env-eps 1 \
  --no-shared-init \
  --out-dir logs/hard_stress_full_ablation_baseline_v2
