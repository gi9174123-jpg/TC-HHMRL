#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
CFG_PATH="${CFG_PATH:-configs/default.yaml}"
DEVICE="${DEVICE:-auto}"
META_ITERS="${META_ITERS:-45}"
EVAL_TASKS="${EVAL_TASKS:-8}"
EVAL_EPS="${EVAL_EPS:-2}"
ENV_TASKS="${ENV_TASKS:-6}"
ENV_EPS="${ENV_EPS:-1}"
SEEDS="${SEEDS:-101 202 303 404 505}"
OUT_DIR="${OUT_DIR:-logs/two_main_scenarios_live_$(date +%Y%m%d_%H%M%S)}"
INSTALL_TORCH_CUDA="${INSTALL_TORCH_CUDA:-0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

export MPLBACKEND="${MPLBACKEND:-Agg}"

"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if [[ "$INSTALL_TORCH_CUDA" == "1" ]]; then
  python -m pip install --upgrade --index-url "$TORCH_INDEX_URL" torch
fi

python - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda_device_count={torch.cuda.device_count()}")
    print(f"cuda_device_name={torch.cuda.get_device_name(0)}")
else:
    print("cuda_device_name=n/a")
PY

read -r -a seed_args <<< "$SEEDS"

cmd=(
  python -m scripts.benchmark_constraint_scenarios
  --cfg "$CFG_PATH"
  --device "$DEVICE"
  --out-dir "$OUT_DIR"
  --scenarios moderate_practical practical_hard
  --meta-iters "$META_ITERS"
  --eval-tasks "$EVAL_TASKS"
  --eval-eps "$EVAL_EPS"
  --env-tasks "$ENV_TASKS"
  --env-eps "$ENV_EPS"
  --seeds
)
cmd+=("${seed_args[@]}")

printf 'Starting benchmark with output dir: %s\n' "$OUT_DIR"
"${cmd[@]}"

printf '\nFinished.\n'
printf 'Report: %s/report.json\n' "$OUT_DIR"
printf 'Requested metrics: %s/requested_metrics.csv\n' "$OUT_DIR"
