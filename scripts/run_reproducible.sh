#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "No python interpreter found in PATH." >&2
    exit 1
  fi
fi

CFG_PATH="${CFG_PATH:-configs/default.yaml}"
META_ITERS="${META_ITERS:-5}"
DEVICE="${DEVICE:-auto}"
CSV_PATH="${CSV_PATH:-logs/meta_run/metrics.csv}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

"${PYTHON_BIN}" -m pip install -r requirements.txt
"${PYTHON_BIN}" -m scripts.train_meta --cfg "${CFG_PATH}" --meta-iters "${META_ITERS}" --device "${DEVICE}"
"${PYTHON_BIN}" -m scripts.eval --cfg "${CFG_PATH}" --n-tasks 2 --episodes-per-task 1 --device "${DEVICE}"
"${PYTHON_BIN}" -m scripts.plot_results --csv "${CSV_PATH}"

echo "Repro run complete."
