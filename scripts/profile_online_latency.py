from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import yaml

from scripts.benchmark_constraint_scenarios import (
    fixed_task_bank_hash,
    ordered_fixed_task_bank_hash,
    sample_fixed_tasks,
)
from tchhmrl.agents.hierarchical_agent import HierarchicalAgent
from tchhmrl.agents.sac_lower import LowerSAC
from tchhmrl.envs.uw_slipt_env import MultiTxUwSliptEnv
from tchhmrl.envs.task_contract import build_context_task_summary_v2
from tchhmrl.utils.config import apply_cli_overrides, load_cfg, resolve_device


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_DIR = ROOT / "paper_official_data" / "fig4_hard_stress_thermal_cap_10seeds_gpu"
DEFAULT_OUT_DIR = ROOT / "logs" / "online_latency_profile"


@dataclass
class ProfileSample:
    obs_np: np.ndarray
    obs_t: torch.Tensor
    temps_np: np.ndarray
    amb_temp: float
    gamma: float
    delta: float
    context_seq: torch.Tensor | None
    z_t: torch.Tensor
    z_np: np.ndarray
    exec_map_np: np.ndarray
    exec_map_t: torch.Tensor
    macro_new: bool
    held_upper_idx_raw: int
    safety_mem: dict[str, int]
    boost_exec: int
    mode_exec: int
    upper_ctx_t: torch.Tensor
    obs_aug_t: torch.Tensor
    physical_features_np: np.ndarray
    lower_raw_np: np.ndarray


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _checkpoint_iter_from_path(path: Path) -> int:
    try:
        return int(path.stem.split("_")[-1])
    except Exception:
        return -1


def _find_run_summary(experiment_dir: Path, scenario: str) -> Path:
    candidates = [
        experiment_dir / scenario / "run_summary.csv",
        experiment_dir / "run_summary.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No run_summary.csv found under {experiment_dir}")


def _choose_checkpoint(
    *,
    experiment_dir: Path,
    scenario: str,
    variant: str,
    seed: int | None,
    explicit_ckpt: Path | None,
) -> tuple[Path, Path, dict[str, Any]]:
    if explicit_ckpt is not None:
        if not explicit_ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {explicit_ckpt}")
        run_dir = explicit_ckpt.parents[1] if explicit_ckpt.parent.name == "checkpoints" else explicit_ckpt.parent
        cfg_path = run_dir / "resolved_config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"Explicit checkpoint was provided, but resolved_config.yaml was not found at {cfg_path}. "
                "Pass a checkpoint inside a benchmark run directory."
            )
        return explicit_ckpt, cfg_path, {
            "selection_source": "explicit_ckpt",
            "run_dir": str(run_dir),
            "checkpoint_iter": _checkpoint_iter_from_path(explicit_ckpt),
        }

    run_summary = _find_run_summary(experiment_dir, scenario)
    df = pd.read_csv(run_summary)
    rows = df[(df["scenario"] == scenario) & (df["variant"] == variant)].copy()
    if seed is not None:
        rows = rows[rows["seed"].astype(int) == int(seed)]
    if rows.empty:
        raise RuntimeError(f"No run_summary row for scenario={scenario}, variant={variant}, seed={seed}")

    if "formally_comparable" in rows.columns:
        rows = rows[rows["formally_comparable"].map(_as_bool)]
    if rows.empty:
        raise RuntimeError("Candidate rows exist, but none are formally comparable")

    score_col = "checkpoint_score" if "checkpoint_score" in rows.columns else "eval_reward"
    rows[score_col] = pd.to_numeric(rows[score_col], errors="coerce")
    if rows[score_col].notna().any():
        row = rows.sort_values(score_col, ascending=False).iloc[0]
    else:
        row = rows.iloc[0]

    run_name = str(row["run_name"])
    run_dir = experiment_dir / scenario / run_name
    if not run_dir.exists():
        # Some archived CSVs keep original logs/... paths; prefer the local archive structure.
        resolved = Path(str(row.get("resolved_config", "")))
        if resolved.exists():
            run_dir = resolved.parent
        else:
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

    cfg_path = run_dir / "resolved_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"resolved_config.yaml not found: {cfg_path}")

    ckpt_iter = int(row.get("checkpoint_iter", -1))
    ckpt_path = run_dir / "checkpoints" / f"iter_{ckpt_iter}.pt"
    if ckpt_iter < 0 or not ckpt_path.exists():
        ckpts = sorted((run_dir / "checkpoints").glob("iter_*.pt"), key=_checkpoint_iter_from_path)
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found in {run_dir / 'checkpoints'}")
        ckpt_path = ckpts[-1]
        ckpt_iter = _checkpoint_iter_from_path(ckpt_path)

    meta = {
        "selection_source": "run_summary",
        "run_summary": str(run_summary),
        "run_name": run_name,
        "run_dir": str(run_dir),
        "checkpoint_iter": ckpt_iter,
        "checkpoint_score": None if pd.isna(row.get("checkpoint_score", np.nan)) else float(row.get("checkpoint_score")),
        "seed": int(row["seed"]),
        "eval_task_batch_hash": str(row.get("eval_task_batch_hash", "")),
        "ordered_eval_task_batch_hash": str(row.get("ordered_eval_task_batch_hash", "")),
    }
    return ckpt_path, cfg_path, meta


def _context_seq_from_episode(agent: HierarchicalAgent) -> torch.Tensor | None:
    if (not agent.context_enabled) or len(agent.episode) < 2:
        return None
    rows = []
    for tr in agent.episode.as_list():
        rows.append(
            np.concatenate(
                [
                    tr["obs"],
                    agent._context_upper_exec(tr),
                    tr["act_exec"],
                    agent._context_feedback(tr),
                ]
            ).astype(np.float32)
        )
    return torch.tensor(np.stack(rows), dtype=torch.float32, device=agent.device).unsqueeze(0)


def _upper_ctx_tensor(upper_idx_exec: int, device: torch.device) -> torch.Tensor:
    return torch.tensor(LowerSAC._upper_ctx_np(upper_idx_exec), dtype=torch.float32, device=device).unsqueeze(0)


def _collect_real_eval_samples(
    *,
    agent: HierarchicalAgent,
    cfg: dict[str, Any],
    seed: int,
    eval_tasks: int,
    eval_eps: int,
    max_samples: int,
) -> tuple[list[ProfileSample], dict[str, Any]]:
    tasks = sample_fixed_tasks(cfg, seed, eval_tasks, seed_offset=21_000)
    samples: list[ProfileSample] = []
    macro_count = 0
    context_count = 0

    agent.reset_rollout_state(clear_context=True)
    with torch.no_grad():
        for task_id, task in enumerate(tasks):
            env = MultiTxUwSliptEnv(cfg, overrides=task.to_env_overrides())
            for ep in range(eval_eps):
                obs, _ = env.reset(seed=int(seed) + task_id * 100 + ep)
                agent.reset_rollout_state(clear_context=(ep == 0))
                done = False
                while not done:
                    context_seq = _context_seq_from_episode(agent)
                    if context_seq is None:
                        z_t = torch.zeros((1, agent.z_dim), dtype=torch.float32, device=agent.device)
                        z_np = np.zeros(agent.z_dim, dtype=np.float32)
                    else:
                        z_infer, _, _, _ = agent.context_encoder.infer(context_seq)
                        z_t = z_infer.detach()
                        z_np = z_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
                        context_count += 1

                    macro_new = bool(agent.upper_mem["hold_left"] <= 0)
                    exec_map_np = agent.safety.raw_to_exec_map(agent.safety_mem).astype(np.int64)
                    if macro_new:
                        upper_idx_raw = agent.upper.select_action(
                            obs.astype(np.float32),
                            z_np,
                            t=agent.global_step,
                            eval_mode=True,
                            exec_map=exec_map_np,
                        )
                    else:
                        upper_idx_raw = int(agent.upper_mem["upper_idx"])
                    boost_exec, mode_exec = agent.safety.preview_exec(upper_idx_raw, agent.safety_mem)
                    upper_idx_exec = agent.safety.encode_exec(boost_exec, mode_exec)
                    upper_ctx_t = _upper_ctx_tensor(upper_idx_exec, agent.device)
                    obs_t = torch.tensor(obs.astype(np.float32), dtype=torch.float32, device=agent.device).unsqueeze(0)
                    physical_features_np = agent.current_physical_features(temps=env.temps)
                    obs_aug_t = agent.lower._augment_np(
                        obs.astype(np.float32),
                        upper_idx=upper_idx_exec,
                        physical_features=physical_features_np,
                        encoder=agent.lower.actor_phys,
                    )

                    lower_raw_t = agent.lower.actor.deterministic(obs_aug_t, z_t)
                    lower_raw_np = lower_raw_t.squeeze(0).detach().cpu().numpy().astype(np.float32)

                    samples.append(
                        ProfileSample(
                            obs_np=obs.astype(np.float32).copy(),
                            obs_t=obs_t.detach(),
                            temps_np=env.temps.astype(np.float32).copy(),
                            amb_temp=float(env.amb_temp),
                            gamma=float(env.gamma),
                            delta=float(env.delta),
                            context_seq=context_seq.detach() if context_seq is not None else None,
                            z_t=z_t.detach(),
                            z_np=z_np.copy(),
                            exec_map_np=exec_map_np.copy(),
                            exec_map_t=torch.as_tensor(exec_map_np, dtype=torch.long, device=agent.device).view(1, -1),
                            macro_new=macro_new,
                            held_upper_idx_raw=int(upper_idx_raw),
                            safety_mem=dict(agent.safety_mem),
                            boost_exec=int(boost_exec),
                            mode_exec=int(mode_exec),
                            upper_ctx_t=upper_ctx_t.detach(),
                            obs_aug_t=obs_aug_t.detach(),
                            physical_features_np=physical_features_np.astype(np.float32).copy(),
                            lower_raw_np=lower_raw_np.copy(),
                        )
                    )
                    macro_count += int(macro_new)
                    if len(samples) >= max_samples:
                        return samples, {
                            "task_batch_hash": fixed_task_bank_hash(tasks),
                            "ordered_task_batch_hash": ordered_fixed_task_bank_hash(tasks),
                            "sample_count": len(samples),
                            "macro_decision_samples": macro_count,
                            "context_history_samples": context_count,
                        }

                    # Advance the real evaluation rollout so later samples have real context histories.
                    action, aux = agent.act(
                        obs=obs.astype(np.float32),
                        temps=env.temps.astype(np.float32).copy(),
                        amb_temp=float(env.amb_temp),
                        gamma=float(env.gamma),
                        delta=float(env.delta),
                        z=z_np,
                        eval_mode=True,
                    )
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = bool(terminated or truncated)
                    cost_vec = np.asarray(info.get("cost_vec", [info.get("cost", 0.0)]), dtype=np.float32).reshape(-1)
                    transition = {
                        "obs": obs.astype(np.float32),
                        "upper_idx_exec": float(aux["upper_idx_exec"]),
                        "boost_combo_exec": float(aux["boost_combo_exec"]),
                        "mode_exec": float(aux["mode_exec"]),
                        "act_exec": aux["act_exec"].astype(np.float32),
                        "reward": float(reward),
                        "reward_raw": float(reward),
                        "reward_task": float(info.get("reward_task", reward)),
                        "reward_benchmark": float(info.get("reward_benchmark", reward)),
                        "reward_dual_penalized": float(info.get("reward_task", reward)),
                        "cost": float(info.get("cost", 0.0)),
                        "cost_vec": cost_vec,
                        "attenuation_c_env": float(env.attenuation_c),
                        "misalign_std_env": float(env.misalign_std),
                        "amb_temp_env": float(env.amb_temp),
                        "gamma_env": float(env.gamma),
                        "delta_env": float(env.delta),
                        "qos_min_rate_env": float(env.qos_min_rate),
                        "distances_env": np.asarray(env.distances, dtype=np.float32).copy(),
                    }
                    transition["task_params"] = np.asarray(
                        build_context_task_summary_v2(transition),
                        dtype=np.float32,
                    )
                    agent.episode.add(transition)
                    obs = next_obs

    return samples, {
        "task_batch_hash": fixed_task_bank_hash(tasks),
        "ordered_task_batch_hash": ordered_fixed_task_bank_hash(tasks),
        "sample_count": len(samples),
        "macro_decision_samples": macro_count,
        "context_history_samples": context_count,
    }


def _profile_function(
    *,
    name: str,
    fn: Callable[[ProfileSample], Any],
    samples: list[ProfileSample],
    warmup: int,
    runs: int,
    device: torch.device,
) -> dict[str, Any]:
    if not samples:
        return {
            "component": name,
            "runs": 0,
            "mean_ms": math.nan,
            "std_ms": math.nan,
            "median_ms": math.nan,
            "p95_ms": math.nan,
            "p99_ms": math.nan,
        }

    with torch.no_grad():
        for i in range(warmup):
            fn(samples[i % len(samples)])
        _sync_if_needed(device)

        timings = np.empty(runs, dtype=np.float64)
        for i in range(runs):
            sample = samples[i % len(samples)]
            _sync_if_needed(device)
            t0 = time.perf_counter_ns()
            fn(sample)
            _sync_if_needed(device)
            t1 = time.perf_counter_ns()
            timings[i] = (t1 - t0) / 1.0e6

    return {
        "component": name,
        "runs": int(runs),
        "mean_ms": float(np.mean(timings)),
        "std_ms": float(np.std(timings, ddof=0)),
        "median_ms": float(np.median(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
        "p99_ms": float(np.percentile(timings, 99)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["component", "frequency", "runs", "mean_ms", "std_ms", "median_ms", "p95_ms", "p99_ms"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _write_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    def fmt(x: Any) -> str:
        try:
            if x is None or (isinstance(x, float) and math.isnan(x)):
                return "--"
            return f"{float(x):.4f}"
        except Exception:
            return str(x)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Online inference latency profile of the full Hybrid controller under hard-stress evaluation.}",
        r"\label{tab:online_latency_profile}",
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Component & Frequency & Mean latency (ms) & 95th percentile (ms) \\",
        r"\hline",
    ]
    for row in rows:
        lines.append(
            f"{row['component']} & {row.get('frequency', '')} & {fmt(row.get('mean_ms'))} & {fmt(row.get('p95_ms'))} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile deployment-time online inference latency for the full Hybrid controller.")
    parser.add_argument("--cfg", default=str(ROOT / "configs" / "default.yaml"), help="Fallback config path if resolved_config is unavailable.")
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT_DIR), help="Hard-stress targeted experiment directory.")
    parser.add_argument("--ckpt", default=None, help="Optional explicit full Hybrid checkpoint path.")
    parser.add_argument("--scenario", default="hard_stress")
    parser.add_argument("--variant", default="hybrid")
    parser.add_argument("--seed", type=int, default=None, help="Seed to profile. Default: best checkpoint_score among full Hybrid runs.")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, mps")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=10000)
    parser.add_argument("--eval-tasks", type=int, default=10)
    parser.add_argument("--eval-eps", type=int, default=3)
    parser.add_argument("--max-samples", type=int, default=2048)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    experiment_dir = Path(args.experiment_dir)
    ckpt_path, resolved_cfg_path, ckpt_meta = _choose_checkpoint(
        experiment_dir=experiment_dir,
        scenario=args.scenario,
        variant=args.variant,
        seed=args.seed,
        explicit_ckpt=Path(args.ckpt) if args.ckpt else None,
    )

    cfg = _load_yaml(resolved_cfg_path) if resolved_cfg_path.exists() else load_cfg(args.cfg)
    cfg = apply_cli_overrides(cfg, device=str(device))
    seed = int(ckpt_meta.get("seed", args.seed if args.seed is not None else cfg.get("experiment", {}).get("seed", 101)))

    agent = HierarchicalAgent(cfg, device)
    agent.load(ckpt_path)
    agent.upper.q.eval()
    agent.lower.actor.eval()
    agent.context_encoder.eval()
    agent.context_predictor.eval()

    samples, sample_meta = _collect_real_eval_samples(
        agent=agent,
        cfg=cfg,
        seed=seed,
        eval_tasks=args.eval_tasks,
        eval_eps=args.eval_eps,
        max_samples=args.max_samples,
    )
    if not samples:
        raise RuntimeError("No real evaluation samples collected; refusing to profile random tensors.")

    context_samples = [s for s in samples if s.context_seq is not None]
    macro_samples = [s for s in samples if s.macro_new]
    if not context_samples:
        raise RuntimeError("No real context histories were collected for GRU profiling.")
    if not macro_samples:
        raise RuntimeError("No macro-decision samples were collected for Upper DQN profiling.")

    safety = agent.safety

    def context_forward(s: ProfileSample):
        return agent.context_encoder.infer(s.context_seq)  # type: ignore[arg-type]

    def upper_forward(s: ProfileSample):
        q = agent.upper.q(s.obs_t, s.z_t)
        q_raw = torch.gather(q, 1, s.exec_map_t)
        return torch.argmax(q_raw, dim=1)

    def lower_actor_forward(s: ProfileSample):
        return agent.lower.actor.deterministic(s.obs_aug_t, s.z_t)

    def safety_projection(s: ProfileSample):
        return safety.project_np(
            s.held_upper_idx_raw,
            s.lower_raw_np,
            temps=s.temps_np,
            amb_temp=s.amb_temp,
            gamma=s.gamma,
            delta=s.delta,
            mem=s.safety_mem,
        )

    def end_to_end_step(s: ProfileSample):
        if s.context_seq is None:
            z_t = torch.zeros((1, agent.z_dim), dtype=torch.float32, device=device)
            z_np = np.zeros(agent.z_dim, dtype=np.float32)
        else:
            z_t, _, _, _ = agent.context_encoder.infer(s.context_seq)
            z_np = z_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if s.macro_new:
            q = agent.upper.q(s.obs_t, z_t)
            q_raw = torch.gather(q, 1, s.exec_map_t)
            upper_idx_raw = int(torch.argmax(q_raw, dim=1).item())
        else:
            upper_idx_raw = int(s.held_upper_idx_raw)
        boost_exec, mode_exec = safety.preview_exec(upper_idx_raw, s.safety_mem)
        upper_idx_exec = safety.encode_exec(boost_exec, mode_exec)
        obs_aug = agent.lower._augment_np(
            s.obs_np,
            upper_idx=upper_idx_exec,
            physical_features=s.physical_features_np,
            encoder=agent.lower.actor_phys,
        )
        raw_t = agent.lower.actor.deterministic(obs_aug, z_t)
        raw_np = raw_t.squeeze(0).detach().cpu().numpy().astype(np.float32)
        return safety.project_np(
            upper_idx_raw,
            raw_np,
            temps=s.temps_np,
            amb_temp=s.amb_temp,
            gamma=s.gamma,
            delta=s.delta,
            mem=s.safety_mem,
        )

    component_specs = [
        ("GRU context encoder", "Every step after context warm-up", context_forward, context_samples),
        ("Upper DQN", "Macro-decision steps only", upper_forward, macro_samples),
        ("Lower SAC actor", "Every online step", lower_actor_forward, samples),
        ("Thermal-cap safety projection", "Every online step", safety_projection, samples),
        ("End-to-end online decision", "Every online step", end_to_end_step, samples),
    ]

    rows: list[dict[str, Any]] = []
    for component, frequency, fn, sample_subset in component_specs:
        stats = _profile_function(
            name=component,
            fn=fn,
            samples=sample_subset,
            warmup=int(args.warmup),
            runs=int(args.runs),
            device=device,
        )
        stats["frequency"] = frequency
        rows.append(stats)

    report = {
        "profile_goal": "deployment_time_online_inference_latency_not_training_time",
        "timing_protocol": {
            "warmup_iterations": int(args.warmup),
            "timed_runs": int(args.runs),
            "batch_size": 1,
            "torch_no_grad": True,
            "timer": "time.perf_counter_ns with torch.cuda.synchronize around timed regions when CUDA is used",
            "excluded": [
                "critic updates",
                "replay-buffer sampling",
                "SAC/DQN training updates",
                "Reptile inner/outer meta-updates",
                "backpropagation",
            ],
        },
        "device": str(device),
        "checkpoint": str(ckpt_path),
        "resolved_config": str(resolved_cfg_path),
        "checkpoint_meta": ckpt_meta,
        "sample_meta": sample_meta,
        "results": rows,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_csv(out_dir / "online_latency_profile.csv", rows)
    _write_latex(out_dir / "online_latency_profile.tex", rows)

    print(f"Saved latency report: {out_dir / 'report.json'}")
    print(f"Saved CSV: {out_dir / 'online_latency_profile.csv'}")
    print(f"Saved LaTeX table: {out_dir / 'online_latency_profile.tex'}")
    for row in rows:
        print(f"{row['component']}: mean={row['mean_ms']:.4f} ms, p95={row['p95_ms']:.4f} ms, p99={row['p99_ms']:.4f} ms")


if __name__ == "__main__":
    main()
