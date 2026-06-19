#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-auto}"
CFG="${CFG:-configs/default.yaml}"
STAMP="$(date +%Y%m%d_%H%M%S)"
ROOT="${ROOT:-logs/formal_fairness_evidence_pack_${STAMP}}"
SEEDS="${SEEDS:-101 202 303 404 505 606 707 808 909 1001}"
META_ITERS="${META_ITERS:-45}"
EVAL_TASKS="${EVAL_TASKS:-8}"
EVAL_EPS="${EVAL_EPS:-2}"
ENV_TASKS="${ENV_TASKS:-8}"
ENV_EPS="${ENV_EPS:-1}"
SHIN_BASELINES="${SHIN_BASELINES:-shin2024_adapted_codebook}"

mkdir -p "$ROOT"
echo "$ROOT" > "$ROOT/ROOT.txt"

echo "[1/3] Meta/context causal diagnostic"
"$PYTHON_BIN" -m scripts.run_formal_meta_context_diagnostics \
  --cfg "$CFG" \
  --out-dir "$ROOT/meta_context" \
  --scenarios moderate_practical hard_stress \
  --seeds $SEEDS \
  --train-iters "$META_ITERS" \
  --tasks "$EVAL_TASKS" \
  --pre-query-episodes 3 \
  --support-episodes 6 \
  --query-episodes 3 \
  --episode-len 80 \
  --device "$DEVICE"

echo "[2/3] Fair hard-clip safety baseline under old main-paper online protocol"
"$PYTHON_BIN" -m scripts.benchmark_constraint_scenarios \
  --cfg "$CFG" \
  --out-dir "$ROOT/fair_hard_clip" \
  --scenarios hard_stress \
  --seeds $SEEDS \
  --meta-iters "$META_ITERS" \
  --online-meta \
  --device "$DEVICE" \
  --variants hybrid \
  --ablations full hard_clip qos_aware_hard_clip \
  --baselines sac_dalal_safe \
  --eval-tasks "$EVAL_TASKS" \
  --eval-eps "$EVAL_EPS" \
  --env-tasks "$ENV_TASKS" \
  --env-eps "$ENV_EPS" \
  --no-shared-init

echo "[3/3] Shin adapted baseline(s) under old main-paper online protocol: $SHIN_BASELINES"
"$PYTHON_BIN" -m scripts.benchmark_constraint_scenarios \
  --cfg "$CFG" \
  --out-dir "$ROOT/shin_fairness" \
  --scenarios moderate_practical hard_stress \
  --seeds $SEEDS \
  --meta-iters "$META_ITERS" \
  --online-meta \
  --device "$DEVICE" \
  --variants hybrid \
  --ablations full \
  --baselines $SHIN_BASELINES \
  --eval-tasks "$EVAL_TASKS" \
  --eval-eps "$EVAL_EPS" \
  --env-tasks "$ENV_TASKS" \
  --env-eps "$ENV_EPS" \
  --no-shared-init

echo "Formal fairness evidence pack complete: $ROOT"
