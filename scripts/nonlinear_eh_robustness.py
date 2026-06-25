from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from scripts.benchmark_constraint_scenarios import (
    fixed_task_bank_hash,
    ordered_fixed_task_bank_hash,
    sample_fixed_tasks,
)
from scripts.profile_online_latency import _choose_checkpoint, _load_yaml
from tchhmrl.agents.hierarchical_agent import HierarchicalAgent
from tchhmrl.envs.task_contract import build_context_task_summary_v2
from tchhmrl.envs.uw_slipt_env import MultiTxUwSliptEnv
from tchhmrl.envs.physics_v2 import INDEPENDENT_SAFETY_PROJECTION_VERSION, normalize_safety_projection_version
from tchhmrl.utils.config import resolve_device


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_DIR = ROOT / "paper_official_data" / "fig4_hard_stress_thermal_cap_10seeds_gpu"
DEFAULT_OUT_DIR = ROOT / "logs" / "nonlinear_eh_robustness"


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _configure_eh(cfg: Dict[str, Any], *, eh_model: str, eh_scale: float | None = None) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)
    out.setdefault("env", {})
    out["env"]["eh_model"] = str(eh_model)
    out["env"].setdefault("eh_nonlinear", {})
    if eh_scale is not None:
        out["env"]["eh_nonlinear"]["scale"] = float(eh_scale)
    out.setdefault("physics", {})
    out["physics"]["thermal_model"] = "independent"
    out["physics"]["safety_projection_version"] = normalize_safety_projection_version(
        "independent",
        out["physics"].get("safety_projection_version", INDEPENDENT_SAFETY_PROJECTION_VERSION),
    )
    if str(eh_model).strip().lower() == "nonlinear":
        out["physics"]["eh_model"] = "linear"
    return out


def _load_agent(cfg: Dict[str, Any], ckpt_path: Path, device_name: str | None) -> Tuple[HierarchicalAgent, torch.device]:
    requested = str(device_name or cfg.get("experiment", {}).get("device", "cpu"))
    device = resolve_device(requested)
    cfg.setdefault("experiment", {})
    cfg["experiment"]["device_requested"] = requested
    cfg["experiment"]["device_resolved"] = str(device)
    agent = HierarchicalAgent(cfg, device)
    agent.load(ckpt_path)
    agent.clear_learning_buffers()
    return agent, device


def _run_eval(
    *,
    cfg: Dict[str, Any],
    ckpt_path: Path,
    device_name: str | None,
    seed: int,
    eval_tasks: int,
    eval_eps: int,
    scenario: str,
    variant: str,
) -> Tuple[pd.DataFrame, Dict[str, float], str, str]:
    agent, _device = _load_agent(cfg, ckpt_path, device_name)
    tasks = sample_fixed_tasks(cfg, seed, eval_tasks, seed_offset=21_000)
    task_hash = fixed_task_bank_hash(tasks)
    ordered_task_hash = ordered_fixed_task_bank_hash(tasks)
    rows: List[Dict[str, Any]] = []

    with torch.no_grad():
        for task_id, task in enumerate(tasks):
            env = MultiTxUwSliptEnv(cfg, overrides=task.to_env_overrides())
            for ep in range(eval_eps):
                obs, _ = env.reset(seed=int(seed) + task_id * 100 + ep)
                agent.reset_rollout_state(clear_context=True)
                done = False
                step = 0
                while not done:
                    z = agent.infer_z()
                    temps_before = env.temps.copy().astype(np.float32)
                    action, aux = agent.act(
                        obs=obs,
                        temps=temps_before,
                        amb_temp=env.amb_temp,
                        gamma=env.gamma,
                        delta=env.delta,
                        z=z,
                        eval_mode=True,
                    )
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = bool(terminated or truncated)
                    cost_vec = np.asarray(info.get("cost_vec", [info.get("cost", 0.0)]), dtype=np.float32).reshape(-1)
                    row = {
                        "scenario": scenario,
                        "variant": variant,
                        "eh_model": str(info.get("eh_model", cfg.get("env", {}).get("eh_model", "linear"))),
                        "seed": int(seed),
                        "task_id": int(task_id),
                        "episode": int(ep),
                        "step": int(step),
                        "reward": float(reward),
                        "se": float(info.get("se", 0.0)),
                        "eh": float(info.get("eh", 0.0)),
                        "qos_rate": float(info.get("qos_rate", 0.0)),
                        "eh_metric": float(info.get("eh_metric", 0.0)),
                        "cost": float(info.get("cost", 0.0)),
                        "violation": float(np.any(cost_vec > 0.0)),
                        "info_share": float(info.get("info_share", 0.0)),
                        "eh_share": float(info.get("eh_share", 0.0)),
                        "eh_input_eff": float(info.get("eh_input_eff", info.get("eh_metric", 0.0))),
                        "eh_metric_linear_proxy": float(
                            info.get("eh_metric_linear_proxy", info.get("eh_metric", 0.0))
                        ),
                        "eh_metric_raw_nonlinear": float(info.get("eh_metric_raw_nonlinear", 0.0)),
                        "eh_saturation_fraction": float(info.get("eh_saturation_fraction", 0.0)),
                        "eh_near_zero_fraction": float(info.get("eh_near_zero_fraction", 0.0)),
                        "eh_scale": float(info.get("eh_scale", 1.0)),
                        "mode_exec": int(info.get("mode_exec", action.get("mode_exec", 0))),
                        "upper_idx_exec": int(info.get("upper_idx_exec", action.get("upper_idx_exec", 0))),
                        "current_total": float(info.get("current_total", 0.0)),
                        "temp_max_after": float(np.max(info.get("temps", env.temps))),
                        "site_id": int(info.get("site_id", getattr(env, "site_id", -1))),
                        "distance_tx0": float(info.get("distance_tx0", env.distances[0])),
                        "distance_tx1": float(info.get("distance_tx1", env.distances[1])),
                        "distance_tx2": float(info.get("distance_tx2", env.distances[2])),
                        "eval_task_batch_hash": task_hash,
                        "ordered_eval_task_batch_hash": ordered_task_hash,
                    }
                    for tx_idx, current_val in enumerate(np.asarray(info.get("currents_exec", []), dtype=np.float32)):
                        row[f"current_tx{tx_idx}"] = float(current_val)
                    rows.append(row)

                    agent.episode.add(
                        {
                            "obs": obs.astype(np.float32),
                            "upper_idx_exec": float(info.get("upper_idx_exec", action.get("upper_idx_exec", 0))),
                            "boost_combo_exec": float(info.get("boost_combo_exec", action.get("boost_combo_exec", 0))),
                            "mode_exec": float(info.get("mode_exec", action.get("mode_exec", 0))),
                            "act_exec": aux["act_exec"].astype(np.float32),
                            "reward": float(reward),
                            "reward_raw": float(reward),
                            "cost": float(info.get("cost", 0.0)),
                            "cost_vec": cost_vec.astype(np.float32),
                            "task_params": np.asarray(
                                build_context_task_summary_v2(
                                    {
                                        "attenuation_c_env": float(env.attenuation_c),
                                        "misalign_std_env": float(env.misalign_std),
                                        "amb_temp_env": float(env.amb_temp),
                                        "gamma_env": float(env.gamma),
                                        "delta_env": float(env.delta),
                                        "qos_min_rate_env": float(env.qos_min_rate),
                                        "distances_env": np.asarray(env.distances, dtype=np.float32),
                                    }
                                ),
                                dtype=np.float32,
                            ),
                        }
                    )
                    obs = next_obs
                    step += 1

    df = pd.DataFrame(rows)
    metrics = {
        "reward": float(df["reward"].mean()),
        "se": float(df["se"].mean()),
        "eh": float(df["eh"].mean()),
        "cost": float(df["cost"].mean()),
        "violation": float(df["violation"].mean()),
        "eh_input_eff": float(df["eh_input_eff"].mean()),
        "eh_metric_raw_nonlinear": float(df["eh_metric_raw_nonlinear"].mean()),
        "eh_saturation_fraction": float(df["eh_saturation_fraction"].mean()),
        "eh_near_zero_fraction": float(df["eh_near_zero_fraction"].mean()),
        "eh_scale": float(df["eh_scale"].mean()),
    }
    return df, metrics, task_hash, ordered_task_hash


def calibrate_nonlinear_scale(
    eh_input_eff: np.ndarray,
    raw_nonlinear: np.ndarray,
    *,
    min_samples: int = 10,
) -> Dict[str, float]:
    x = np.asarray(eh_input_eff, dtype=np.float64).reshape(-1)
    raw = np.asarray(raw_nonlinear, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(raw)
    x = x[finite]
    raw = raw[finite]
    if x.size < int(min_samples):
        raise ValueError(f"Degenerate EH calibration: only {x.size} finite samples")
    raw_mean = float(np.mean(raw))
    linear_mean = float(np.mean(x))
    if raw_mean <= 1.0e-12 or linear_mean <= 1.0e-12:
        raise ValueError(
            f"Degenerate EH calibration: linear_mean={linear_mean:.6g}, raw_nonlinear_mean={raw_mean:.6g}"
        )
    return {
        "sample_count": int(x.size),
        "eh_input_eff_mean": linear_mean,
        "eh_input_eff_q25": float(np.quantile(x, 0.25)),
        "eh_input_eff_q50": float(np.quantile(x, 0.50)),
        "eh_input_eff_q90": float(np.quantile(x, 0.90)),
        "raw_nonlinear_mean": raw_mean,
        "linear_eh_mean": linear_mean,
        "scale": float(linear_mean / raw_mean),
    }


def run_stage1(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir) / "stage1_eval_only"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.out_dir) / "eh_scale_manifest.json"

    ckpt_path, cfg_path, ckpt_meta = _choose_checkpoint(
        experiment_dir=Path(args.experiment_dir),
        scenario=args.scenario,
        variant=args.variant,
        seed=args.seed,
        explicit_ckpt=Path(args.ckpt) if args.ckpt else None,
    )
    base_cfg = _load_yaml(cfg_path)
    seed = int(ckpt_meta.get("seed", args.seed if args.seed is not None else 101))

    linear_cfg = _configure_eh(base_cfg, eh_model="linear")
    linear_df, linear_metrics, task_hash, ordered_hash = _run_eval(
        cfg=linear_cfg,
        ckpt_path=ckpt_path,
        device_name=args.device,
        seed=seed,
        eval_tasks=args.eval_tasks,
        eval_eps=args.eval_eps,
        scenario=args.scenario,
        variant=args.variant,
    )
    calibration = calibrate_nonlinear_scale(
        linear_df["eh_input_eff"].to_numpy(),
        linear_df["eh_metric_raw_nonlinear"].to_numpy(),
        min_samples=max(10, args.eval_tasks * args.eval_eps),
    )
    nl_cfg = _configure_eh(base_cfg, eh_model="nonlinear", eh_scale=float(calibration["scale"]))
    nonlinear_df, nonlinear_metrics, task_hash_nl, ordered_hash_nl = _run_eval(
        cfg=nl_cfg,
        ckpt_path=ckpt_path,
        device_name=args.device,
        seed=seed,
        eval_tasks=args.eval_tasks,
        eval_eps=args.eval_eps,
        scenario=args.scenario,
        variant=args.variant,
    )
    if task_hash != task_hash_nl or ordered_hash != ordered_hash_nl:
        raise RuntimeError("Linear and nonlinear Stage-1 eval used different held-out task batches")

    linear_df.to_csv(out_dir / "linear_eval.csv", index=False)
    nonlinear_df.to_csv(out_dir / "nonlinear_eval.csv", index=False)
    compare = pd.DataFrame(
        [
            {"eh_model": "linear", **linear_metrics},
            {"eh_model": "nonlinear", **nonlinear_metrics},
        ]
    )
    compare.to_csv(out_dir / "eval_compare.csv", index=False)

    eh_cfg = dict(base_cfg.get("env", {}).get("eh_nonlinear", {}) or {})
    manifest = {
        "source_experiment": str(args.experiment_dir),
        "source_checkpoint": str(ckpt_path),
        "source_resolved_config": str(cfg_path),
        "checkpoint_meta": ckpt_meta,
        "scenario": args.scenario,
        "variant": args.variant,
        "seed": seed,
        "eval_tasks": int(args.eval_tasks),
        "eval_eps": int(args.eval_eps),
        "task_hash": task_hash,
        "ordered_task_hash": ordered_hash,
        "logistic_params": {
            "type": str(eh_cfg.get("type", "logistic_normalized")),
            "e_max": _as_float(eh_cfg.get("e_max", 1.0), 1.0),
            "a": _as_float(eh_cfg.get("a", 12.0), 12.0),
            "b": _as_float(eh_cfg.get("b", 0.10), 0.10),
        },
        **calibration,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    report = {
        "stage": "stage1_eval_only",
        "manifest": str(manifest_path),
        "linear_eval_csv": str(out_dir / "linear_eval.csv"),
        "nonlinear_eval_csv": str(out_dir / "nonlinear_eval.csv"),
        "eval_compare_csv": str(out_dir / "eval_compare.csv"),
        "task_hash_match": bool(task_hash == task_hash_nl),
        "ordered_task_hash_match": bool(ordered_hash == ordered_hash_nl),
        "acceptance_probe": {
            "nonlinear_eh_not_all_zero": bool(nonlinear_metrics["eh_metric_raw_nonlinear"] > 1.0e-12),
            "nonlinear_not_fully_saturated": bool(nonlinear_metrics["eh_saturation_fraction"] < 0.95),
            "nonlinear_not_all_near_zero": bool(nonlinear_metrics["eh_near_zero_fraction"] < 0.95),
        },
        "metrics": {
            "linear": linear_metrics,
            "nonlinear": nonlinear_metrics,
        },
    }
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Stage-1 nonlinear EH robustness check complete: {out_dir / 'report.json'}")
    return out_dir / "report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supplementary nonlinear-EH robustness checks.")
    parser.add_argument("--stage", choices=["stage1"], default="stage1")
    parser.add_argument("--experiment-dir", type=str, default=str(DEFAULT_EXPERIMENT_DIR))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--scenario", type=str, default="hard_stress")
    parser.add_argument("--variant", type=str, default="hybrid")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--eval-tasks", type=int, default=10)
    parser.add_argument("--eval-eps", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "stage1":
        run_stage1(args)
    else:
        raise ValueError(f"Unsupported stage: {args.stage}")


if __name__ == "__main__":
    main()
