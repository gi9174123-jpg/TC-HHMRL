#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-auto}"
CFG="${CFG:-configs/default.yaml}"
STAMP="$(date +%Y%m%d_%H%M%S)"
ROOT="${ROOT:-logs/formal_final_evidence_pack_${STAMP}}"
SEEDS="${SEEDS:-101 202 303 404 505 606 707 808 909 1001}"
META_ITERS="${META_ITERS:-100}"
EVAL_TASKS="${EVAL_TASKS:-10}"
EVAL_EPS="${EVAL_EPS:-3}"
ENV_TASKS="${ENV_TASKS:-8}"
ENV_EPS="${ENV_EPS:-1}"
SHIN_BASELINES="${SHIN_BASELINES:-shin2024_adapted_codebook}"
RUN_SHIN_FORMAL="${RUN_SHIN_FORMAL:-1}"
RUN_META_DIAGNOSTIC="${RUN_META_DIAGNOSTIC:-1}"
RUN_LONG_HORIZON="${RUN_LONG_HORIZON:-1}"
LONG_HORIZON_LEN="${LONG_HORIZON_LEN:-160}"

mkdir -p "$ROOT"
echo "$ROOT" > "$ROOT/ROOT.txt"

echo "[0/6] Contract tests"
"$PYTHON_BIN" -m pytest -q

echo "[1/6] Gated structural benchmark: Hybrid / Single-LED / Single-LD"
"$PYTHON_BIN" -m scripts.benchmark_constraint_scenarios \
  --cfg "$CFG" \
  --out-dir "$ROOT/structural_gated" \
  --scenarios moderate_practical hard_stress \
  --seeds $SEEDS \
  --meta-iters "$META_ITERS" \
  --strict-meta \
  --device "$DEVICE" \
  --variants hybrid single_led single_ld \
  --ablations full \
  --eval-tasks "$EVAL_TASKS" \
  --eval-eps "$EVAL_EPS" \
  --env-tasks "$ENV_TASKS" \
  --env-eps "$ENV_EPS" \
  --no-shared-init

echo "[2/6] Main benchmark ablations: Gated Full / Ungated Meta / w-o Meta / w-o Lagrangian"
"$PYTHON_BIN" -m scripts.benchmark_constraint_scenarios \
  --cfg "$CFG" \
  --out-dir "$ROOT/meta_and_lagrangian_gated" \
  --scenarios moderate_practical hard_stress \
  --seeds $SEEDS \
  --meta-iters "$META_ITERS" \
  --strict-meta \
  --device "$DEVICE" \
  --variants hybrid \
  --ablations full meta_ungated wo_meta wo_lagrangian \
  --eval-tasks "$EVAL_TASKS" \
  --eval-eps "$EVAL_EPS" \
  --env-tasks "$ENV_TASKS" \
  --env-eps "$ENV_EPS" \
  --no-shared-init

echo "[3/6] Safety-layer fairness: Gated Full / Naive Clip / QoS-aware projection / SAC+Dalal"
"$PYTHON_BIN" -m scripts.benchmark_constraint_scenarios \
  --cfg "$CFG" \
  --out-dir "$ROOT/safety_hard_stress_gated" \
  --scenarios hard_stress \
  --seeds $SEEDS \
  --meta-iters "$META_ITERS" \
  --strict-meta \
  --device "$DEVICE" \
  --variants hybrid \
  --ablations full hard_clip qos_aware_hard_clip \
  --baselines sac_dalal_safe \
  --eval-tasks "$EVAL_TASKS" \
  --eval-eps "$EVAL_EPS" \
  --env-tasks "$ENV_TASKS" \
  --env-eps "$ENV_EPS" \
  --no-shared-init

if [[ "$RUN_SHIN_FORMAL" == "1" ]]; then
  echo "[4/6] Shin-inspired adapted baseline(s): $SHIN_BASELINES"
  "$PYTHON_BIN" -m scripts.benchmark_constraint_scenarios \
    --cfg "$CFG" \
    --out-dir "$ROOT/shin_fairness_gated_protocol" \
    --scenarios moderate_practical hard_stress \
    --seeds $SEEDS \
    --meta-iters "$META_ITERS" \
    --strict-meta \
    --device "$DEVICE" \
    --variants hybrid \
    --ablations full \
    --baselines $SHIN_BASELINES \
    --eval-tasks "$EVAL_TASKS" \
    --eval-eps "$EVAL_EPS" \
    --env-tasks "$ENV_TASKS" \
    --env-eps "$ENV_EPS" \
    --no-shared-init
else
  echo "[4/6] Skipping Shin formal because RUN_SHIN_FORMAL=$RUN_SHIN_FORMAL"
fi

if [[ "$RUN_META_DIAGNOSTIC" == "1" ]]; then
  echo "[5/6] Strict support-query meta diagnostic table: Gated / Ungated / w-o Meta"
  "$PYTHON_BIN" -m scripts.run_formal_meta_context_diagnostics \
    --cfg "$CFG" \
    --out-dir "$ROOT/meta_context_strict_diagnostic" \
    --scenarios moderate_practical hard_stress \
    --seeds $SEEDS \
    --train-iters "$META_ITERS" \
    --tasks "$EVAL_TASKS" \
    --pre-query-episodes 3 \
    --support-episodes 5 \
    --query-episodes 2 \
    --episode-len 80 \
    --device "$DEVICE"
else
  echo "[5/6] Skipping meta diagnostic because RUN_META_DIAGNOSTIC=$RUN_META_DIAGNOSTIC"
fi

if [[ "$RUN_LONG_HORIZON" == "1" ]]; then
  echo "[6/6] Long-horizon Lagrangian diagnostic, episode_len=$LONG_HORIZON_LEN"
  "$PYTHON_BIN" -m scripts.benchmark_constraint_scenarios \
    --cfg "$CFG" \
    --out-dir "$ROOT/lagrangian_long_horizon" \
    --scenarios hard_stress \
    --seeds $SEEDS \
    --meta-iters "$META_ITERS" \
    --strict-meta \
    --episode-len "$LONG_HORIZON_LEN" \
    --device "$DEVICE" \
    --variants hybrid \
    --ablations full wo_lagrangian \
    --eval-tasks "$EVAL_TASKS" \
    --eval-eps "$EVAL_EPS" \
    --env-tasks "$ENV_TASKS" \
    --env-eps "$ENV_EPS" \
    --no-shared-init
else
  echo "[6/6] Skipping long-horizon diagnostic because RUN_LONG_HORIZON=$RUN_LONG_HORIZON"
fi

echo "Formal final evidence pack complete: $ROOT"
