#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-auto}"
BASE_CFG="${BASE_CFG:-configs/default.yaml}"
STAMP="$(date +%Y%m%d_%H%M%S)"
ROOT="${ROOT:-logs/validation_shin_tuned_${STAMP}}"
SEEDS="${SEEDS:-101 202 303}"
META_ITERS="${META_ITERS:-45}"

mkdir -p "$ROOT/configs"
echo "$ROOT" > "$ROOT/ROOT.txt"

make_cfg() {
  local set_name="$1"
  local lr_scale="$2"
  local out_cfg="$ROOT/configs/shin_${set_name}_lr${lr_scale}.yaml"
  "$PYTHON_BIN" - "$BASE_CFG" "$out_cfg" "$set_name" "$lr_scale" <<'PY'
import sys
from pathlib import Path
import yaml

base_path, out_path, set_name, lr_scale = sys.argv[1:5]
with open(base_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

families = {
    # Source-aware adaptations of the requested low/mid/high validation levels.
    "A": {
        "names": ["low_A", "mid_A", "high_A"],
        "codewords": [[0.25, 0.18, 0.18], [0.50, 0.42, 0.42], [0.75, 0.68, 0.68]],
    },
    "B": {
        "names": ["low_B", "mid_B", "high_B"],
        "codewords": [[0.40, 0.25, 0.25], [0.55, 0.45, 0.45], [0.70, 0.65, 0.65]],
    },
    "C": {
        "names": ["low_C", "mid_C", "high_C"],
        "codewords": [[0.45, 0.35, 0.35], [0.60, 0.52, 0.52], [0.75, 0.70, 0.70]],
    },
}
if set_name not in families:
    raise SystemExit(f"unknown set {set_name}")

cfg.setdefault("baselines", {})["shin2024_adapted_codebook_tuned"] = {
    "current_template_codeword_names": families[set_name]["names"],
    "current_template_codewords": families[set_name]["codewords"],
    "ddpg_lr_scale": float(lr_scale),
    "validation_set": set_name,
    "validation_scope": "source_aware_codebook_and_ddpg_lr_scale_only",
}
Path(out_path).parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(out_path)
PY
}

for set_name in A B C; do
  for lr_scale in 0.5 1.0 2.0; do
    cfg_path="$(make_cfg "$set_name" "$lr_scale")"
    out_dir="$ROOT/${set_name}_lr${lr_scale}"
    echo "[shin validation] set=$set_name lr_scale=$lr_scale out=$out_dir"
    "$PYTHON_BIN" -m scripts.benchmark_constraint_scenarios \
      --cfg "$cfg_path" \
      --out-dir "$out_dir" \
      --scenarios moderate_practical hard_stress \
      --seeds $SEEDS \
      --meta-iters "$META_ITERS" \
      --online-meta \
      --device "$DEVICE" \
      --variants hybrid \
      --ablations full \
      --baselines shin2024_adapted_codebook_tuned \
      --eval-tasks 8 \
      --eval-eps 2 \
      --env-tasks 4 \
      --env-eps 1 \
      --no-shared-init
  done
done

echo "Shin validation sweep complete: $ROOT"
echo "Select one config from $ROOT/configs, then run formal with:"
echo "  CFG=<selected.yaml> SHIN_BASELINES='shin2024_adapted_codebook shin2024_adapted_codebook_tuned' bash scripts/run_formal_fairness_evidence_pack.sh"
