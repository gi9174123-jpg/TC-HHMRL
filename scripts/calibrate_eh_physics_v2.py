from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scripts.benchmark_constraint_scenarios import apply_scenario, sample_fixed_tasks
from tchhmrl.envs.physics_v2 import calibrate_logistic_eh_from_samples, eh_calibration_hash
from tchhmrl.envs.uw_slipt_env import MultiTxUwSliptEnv
from tchhmrl.safety.safety_layer import SafetyLayer
from tchhmrl.utils.config import load_cfg


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _collect_samples(
    cfg: dict[str, Any],
    *,
    scenarios: list[str],
    seeds: list[int],
    tasks_per_scenario: int,
    episodes_per_task: int,
) -> np.ndarray:
    samples: list[float] = []
    rng = np.random.default_rng(12345)
    for scenario in scenarios:
        scenario_cfg = yaml.safe_load(yaml.safe_dump(cfg))
        apply_scenario(scenario_cfg, scenario)
        safety = SafetyLayer(scenario_cfg)
        for seed in seeds:
            tasks = sample_fixed_tasks(scenario_cfg, seed, tasks_per_scenario, seed_offset=41_000)
            for task_id, task in enumerate(tasks):
                env = MultiTxUwSliptEnv(scenario_cfg, overrides=task.to_env_overrides())
                for ep in range(episodes_per_task):
                    obs, _ = env.reset(seed=seed + task_id * 100 + ep)
                    del obs
                    mem = {"current_boost": 0, "dwell_count": safety.min_dwell_steps}
                    done = False
                    while not done:
                        upper_raw = int(rng.integers(0, 12))
                        lower_raw = rng.uniform(-1.0, 1.0, size=5).astype(np.float32)
                        safe, mem = safety.project_np(
                            upper_raw,
                            lower_raw,
                            temps=env.temps.copy(),
                            amb_temp=env.amb_temp,
                            gamma=env.gamma,
                            delta=env.delta,
                            mem=mem,
                        )
                        _, _, terminated, truncated, info = env.step(safe)
                        samples.append(float(info.get("eh_input_eff", 0.0)))
                        done = bool(terminated or truncated)
    return np.asarray(samples, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/default.yaml")
    parser.add_argument("--out", type=str, default="configs/eh_calibration_manifest.json")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["moderate_practical", "hard_stress", "channel_harsh"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303])
    parser.add_argument("--tasks-per-scenario", type=int, default=4)
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--r-sat", type=float, default=0.85)
    args = parser.parse_args()

    cfg = load_cfg(args.cfg)
    samples = _collect_samples(
        cfg,
        scenarios=list(args.scenarios),
        seeds=list(args.seeds),
        tasks_per_scenario=int(args.tasks_per_scenario),
        episodes_per_task=int(args.episodes_per_task),
    )
    params = calibrate_logistic_eh_from_samples(samples, r_sat=float(args.r_sat))
    manifest = {
        "physics_version": str(cfg.get("physics", {}).get("physics_version", "physics_v2")),
        "eh_model": "logistic",
        "calibration_scenarios": list(args.scenarios),
        "seeds": list(args.seeds),
        "sample_count": int(params["eh_calibration_sample_count"]),
        "Q25": float(params["eh_calibration_q25"]),
        "Q90": float(params["eh_calibration_q90"]),
        "Q95": float(params["eh_calibration_q95"]),
        "r_sat": float(params["eh_calibration_r_sat"]),
        "M_EH": float(params["eh_nl_M"]),
        "a_EH": float(params["eh_nl_a"]),
        "b_EH": float(params["eh_nl_b"]),
        "commit_hash": _git_commit(),
    }
    manifest["eh_calibration_hash"] = eh_calibration_hash(manifest)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
