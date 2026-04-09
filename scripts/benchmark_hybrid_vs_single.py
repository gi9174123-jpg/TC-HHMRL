from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tchhmrl.envs.uw_slipt_env import MultiTxUwSliptEnv
from tchhmrl.meta.meta_trainer import MetaTrainer
from tchhmrl.utils.config import apply_cli_overrides, load_cfg


def make_variant_cfg(
    base_cfg: Dict,
    variant: str,
    seed: int,
    out_dir: Path,
    meta_iters: int,
    fast_mode: bool,
) -> Dict:
    cfg = copy.deepcopy(base_cfg)
    cfg["experiment"]["seed"] = int(seed)
    cfg["experiment"]["log_dir"] = str(out_dir)
    cfg["experiment"]["run_name"] = f"{variant}_seed{seed}"

    cfg["meta"]["meta_iters"] = int(meta_iters)

    if fast_mode:
        # Keep benchmark runtime practical while preserving behavior.
        cfg["meta"]["n_tasks_per_iter"] = int(min(3, cfg["meta"]["n_tasks_per_iter"]))
        cfg["meta"]["support_episodes"] = int(min(1, cfg["meta"]["support_episodes"]))
        cfg["meta"]["query_episodes"] = int(min(1, cfg["meta"]["query_episodes"]))
        cfg["env"]["episode_len"] = int(min(60, cfg["env"]["episode_len"]))
        cfg["agent"]["warmup_steps"] = int(min(120, cfg["agent"]["warmup_steps"]))

    if variant == "single_ld":
        # Baseline B: homogeneous narrow-beam LD links.
        cfg["env"]["hybrid"]["tx_device"] = ["LD"] * int(cfg["env"]["n_tx"])
    elif variant == "single_led":
        # Baseline A: homogeneous wide-beam LED links.
        cfg["env"]["hybrid"]["tx_device"] = ["LED"] * int(cfg["env"]["n_tx"])
    elif variant == "hybrid":
        # Baseline C: fixed hetero Anchor(LED)+Boost(LD,LD).
        cfg["env"]["hybrid"]["tx_device"] = ["LED", "LD", "LD"]
    else:
        raise ValueError(f"Unsupported variant: {variant}")

    return cfg


def convergence_stats(run_df: pd.DataFrame) -> Dict[str, float]:
    run_df = run_df.sort_values("iter")
    k = max(3, len(run_df) // 5)
    first = float(run_df["query_reward"].iloc[:k].mean())
    last = float(run_df["query_reward"].iloc[-k:].mean())
    last_seg = run_df["query_reward"].iloc[-k:].to_numpy(dtype=np.float64)

    slope = 0.0
    if len(last_seg) >= 2:
        x = np.arange(len(last_seg), dtype=np.float64)
        slope = float(np.polyfit(x, last_seg, 1)[0])

    return {
        "first_query_reward": first,
        "last_query_reward": last,
        "reward_gain": last - first,
        "last_segment_std": float(np.std(last_seg)),
        "last_segment_slope": slope,
    }


def collect_env_data(
    trainer: MetaTrainer,
    cfg: Dict,
    variant: str,
    seed: int,
    n_tasks: int,
    episodes_per_task: int,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    tasks = trainer.task_sampler.sample(n_tasks)

    for task_id, task in enumerate(tasks):
        env = MultiTxUwSliptEnv(cfg, overrides=task.to_env_overrides())

        for ep in range(episodes_per_task):
            obs, _ = env.reset(seed=seed + task_id * 100 + ep)
            trainer.agent.reset_episode_state()
            step_idx = 0
            done = False

            while not done:
                z = trainer.agent.infer_z()
                temps_before = env.temps.copy().astype(np.float32)
                ch_led_before = env.channel_led.copy().astype(np.float32)
                ch_ld_before = env.channel_ld.copy().astype(np.float32)

                action, aux = trainer.agent.act(
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

                trainer.agent.episode.add(
                    {
                        "obs": obs.astype(np.float32),
                        "act_exec": aux["act_exec"].astype(np.float32),
                        "reward": float(reward),
                        "cost": float(info["cost"]),
                    }
                )

                rows.append(
                    {
                        "variant": variant,
                        "seed": float(seed),
                        "task_id": float(task_id),
                        "episode": float(ep),
                        "step": float(step_idx),
                        "attenuation_c": float(env.attenuation_c),
                        "misalign_std": float(env.misalign_std),
                        "amb_temp": float(env.amb_temp),
                        "gamma": float(env.gamma),
                        "delta": float(env.delta),
                        "channel_led_mean": float(np.mean(ch_led_before)),
                        "channel_ld_mean": float(np.mean(ch_ld_before)),
                        "channel_led_max": float(np.max(ch_led_before)),
                        "channel_ld_max": float(np.max(ch_ld_before)),
                        "signal_ld_share": float(info["signal_ld_share"]),
                        "led_tx_fraction": float(info["led_tx_fraction"]),
                        "signal_led": float(info["signal_led"]),
                        "signal_ld": float(info["signal_ld"]),
                        "signal_total": float(info["signal"]),
                        "snr": float(info["snr"]),
                        "se": float(info["se"]),
                        "eh": float(info["eh"]),
                        "reward_se_term": float(info.get("reward_se_term", info["se"])),
                        "reward_eh_term": float(info.get("reward_eh_term", info["eh"])),
                        "penalty_cost_term": float(info.get("penalty_cost_term", 0.0)),
                        "penalty_power_term": float(info.get("penalty_power_term", 0.0)),
                        "cost": float(info["cost"]),
                        "thermal_violation": float(info["thermal_violation"]),
                        "temp_mean_before": float(np.mean(temps_before)),
                        "temp_mean_after": float(np.mean(info["temps"])),
                        "temp_max_after": float(np.max(info["temps"])),
                        "total_current": float(np.sum(action["currents_exec"])),
                        "rho": float(np.asarray(action["rho_exec"]).item()),
                        "tau": float(np.asarray(action["tau_exec"]).item()),
                    }
                )

                obs = next_obs
                step_idx += 1

    return pd.DataFrame(rows)


def summarize_eval(eval_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    metrics = ["reward", "se", "eh", "cost", "violation_rate", "len"]

    for variant, g in eval_df.groupby("variant"):
        out[variant] = {}
        for m in metrics:
            out[variant][f"{m}_mean"] = float(g[m].mean())
            out[variant][f"{m}_std"] = float(g[m].std(ddof=0))

    return out


def plot_convergence(train_df: pd.DataFrame, out_path: Path) -> None:
    agg = (
        train_df.groupby(["variant", "iter"], as_index=False)
        .agg(
            query_reward_mean=("query_reward", "mean"),
            query_reward_std=("query_reward", "std"),
            query_cost_mean=("query_cost", "mean"),
            query_cost_std=("query_cost", "std"),
        )
        .fillna(0.0)
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    for variant, g in agg.groupby("variant"):
        x = g["iter"].to_numpy()
        y = g["query_reward_mean"].to_numpy()
        s = g["query_reward_std"].to_numpy()
        axes[0].plot(x, y, label=variant)
        axes[0].fill_between(x, y - s, y + s, alpha=0.2)

        y2 = g["query_cost_mean"].to_numpy()
        s2 = g["query_cost_std"].to_numpy()
        axes[1].plot(x, y2, label=variant)
        axes[1].fill_between(x, y2 - s2, y2 + s2, alpha=0.2)

    axes[0].set_title("Convergence: Query Reward")
    axes[1].set_title("Constraint Trend: Query Cost")
    axes[0].set_xlabel("Meta Iter")
    axes[1].set_xlabel("Meta Iter")
    axes[0].set_ylabel("Reward")
    axes[1].set_ylabel("Cost")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_final_metrics(eval_df: pd.DataFrame, out_path: Path) -> None:
    metric_pairs = [
        ("reward", "Reward"),
        ("se", "SE"),
        ("eh", "EH"),
        ("cost", "Cost"),
        ("violation_rate", "Violation Rate"),
    ]

    variants = sorted(eval_df["variant"].unique().tolist())
    x = np.arange(len(metric_pairs), dtype=np.float64)
    w = max(0.22, 0.82 / max(1, len(variants)))

    fig, ax = plt.subplots(figsize=(10, 4.2))

    for i, v in enumerate(variants):
        g = eval_df[eval_df["variant"] == v]
        means = [float(g[m].mean()) for m, _ in metric_pairs]
        stds = [float(g[m].std(ddof=0)) for m, _ in metric_pairs]
        offset = (i - (len(variants) - 1) / 2.0) * w
        ax.bar(x + offset, means, width=w, yerr=stds, capsize=3, label=v)

    ax.set_xticks(x)
    ax.set_xticklabels([name for _, name in metric_pairs])
    ax.set_title("Final Evaluation Metrics")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_env_realism(env_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    variants = sorted(env_df["variant"].unique().tolist())

    for v in variants:
        g = env_df[env_df["variant"] == v]
        axes[0, 0].hist(g["snr"], bins=40, alpha=0.45, label=v)
    axes[0, 0].set_title("SNR Distribution")
    axes[0, 0].set_xlabel("SNR")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    for v in variants:
        g = env_df[env_df["variant"] == v]
        axes[0, 1].hist(g["temp_max_after"], bins=40, alpha=0.45, label=v)
    axes[0, 1].set_title("Peak Temperature Distribution")
    axes[0, 1].set_xlabel("Temperature")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    for v in variants:
        g = env_df[env_df["variant"] == v]
        axes[1, 0].hist(g["signal_ld_share"], bins=20, alpha=0.45, label=v)
    axes[1, 0].set_title("LD Signal Share Distribution")
    axes[1, 0].set_xlabel("LD Signal Share")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    hybrid = env_df[env_df["variant"] == "hybrid"]
    if not hybrid.empty:
        sample = hybrid.sample(n=min(1200, len(hybrid)), random_state=0)
        axes[1, 1].scatter(sample["signal_led"], sample["signal_ld"], s=8, alpha=0.35)
    axes[1, 1].set_title("Hybrid Signal Split (LED vs LD)")
    axes[1, 1].set_xlabel("signal_led")
    axes[1, 1].set_ylabel("signal_ld")
    axes[1, 1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_benchmark(
    cfg_path: str,
    out_dir: str,
    meta_iters: int,
    fast_mode: bool,
    seeds: List[int],
    eval_tasks: int,
    eval_eps: int,
    env_tasks: int,
    env_eps: int,
    device: str | None = None,
) -> Tuple[Path, Path, Path, Path]:
    base_cfg = apply_cli_overrides(load_cfg(cfg_path), device=device)
    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    variants = ["hybrid", "single_led", "single_ld"]

    train_rows: List[pd.DataFrame] = []
    eval_rows: List[Dict[str, float]] = []
    env_rows: List[pd.DataFrame] = []
    conv_rows: List[Dict[str, float]] = []

    for variant in variants:
        for seed in seeds:
            cfg = make_variant_cfg(
                base_cfg,
                variant,
                seed,
                out_base,
                meta_iters,
                fast_mode=fast_mode,
            )
            trainer = MetaTrainer(cfg)
            train_csv = trainer.train(meta_iters=meta_iters)

            run_df = pd.read_csv(train_csv)
            run_df["variant"] = variant
            run_df["seed"] = float(seed)
            train_rows.append(run_df)

            conv = convergence_stats(run_df)
            conv["variant"] = variant
            conv["seed"] = float(seed)
            conv_rows.append(conv)

            ev = trainer.evaluate(n_tasks=eval_tasks, episodes_per_task=eval_eps)
            ev["variant"] = variant
            ev["seed"] = float(seed)
            eval_rows.append(ev)

            env_df = collect_env_data(
                trainer=trainer,
                cfg=cfg,
                variant=variant,
                seed=seed,
                n_tasks=env_tasks,
                episodes_per_task=env_eps,
            )
            env_rows.append(env_df)

    train_df = pd.concat(train_rows, axis=0, ignore_index=True)
    eval_df = pd.DataFrame(eval_rows)
    env_df = pd.concat(env_rows, axis=0, ignore_index=True)
    conv_df = pd.DataFrame(conv_rows)

    train_csv = out_base / "benchmark_training.csv"
    eval_csv = out_base / "benchmark_eval.csv"
    env_csv = out_base / "benchmark_env.csv"
    conv_csv = out_base / "benchmark_convergence.csv"

    train_df.to_csv(train_csv, index=False)
    eval_df.to_csv(eval_csv, index=False)
    env_df.to_csv(env_csv, index=False)
    conv_df.to_csv(conv_csv, index=False)

    convergence_png = out_base / "compare_convergence.png"
    final_metrics_png = out_base / "compare_final_metrics.png"
    env_png = out_base / "env_realism.png"

    plot_convergence(train_df, convergence_png)
    plot_final_metrics(eval_df, final_metrics_png)
    plot_env_realism(env_df, env_png)

    eval_summary = summarize_eval(eval_df)
    reward_components = (
        env_df.groupby("variant", as_index=False)[
            ["reward_se_term", "reward_eh_term", "penalty_cost_term", "penalty_power_term"]
        ]
        .mean()
        .set_index("variant")
        .to_dict(orient="index")
    )
    hybrid_reward = eval_summary["hybrid"]["reward_mean"]
    single_ld_reward = eval_summary["single_ld"]["reward_mean"]
    single_led_reward = eval_summary["single_led"]["reward_mean"]
    gain_vs_ld = 100.0 * (hybrid_reward - single_ld_reward) / (abs(single_ld_reward) + 1e-8)
    gain_vs_led = 100.0 * (hybrid_reward - single_led_reward) / (abs(single_led_reward) + 1e-8)

    summary = {
        "meta_iters": meta_iters,
        "fast_mode": bool(fast_mode),
        "seeds": seeds,
        "eval_summary": eval_summary,
        "reward_component_means": reward_components,
        "convergence_by_run": conv_rows,
        "hybrid_vs_single_ld_reward_gain_pct": gain_vs_ld,
        "hybrid_vs_single_led_reward_gain_pct": gain_vs_led,
        "artifacts": {
            "train_csv": str(train_csv),
            "eval_csv": str(eval_csv),
            "env_csv": str(env_csv),
            "conv_csv": str(conv_csv),
            "convergence_png": str(convergence_png),
            "final_metrics_png": str(final_metrics_png),
            "env_png": str(env_png),
        },
    }

    summary_json = out_base / "benchmark_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Benchmark complete")
    print(f"Summary JSON: {summary_json}")
    print(f"Reward gain (hybrid vs single_ld): {gain_vs_ld:.2f}%")
    print(f"Reward gain (hybrid vs single_led): {gain_vs_led:.2f}%")

    return train_csv, eval_csv, env_csv, summary_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default=None, help="Override config device: auto/cpu/cuda/mps")
    parser.add_argument("--out-dir", type=str, default="logs/benchmark_hybrid_vs_single")
    parser.add_argument("--meta-iters", type=int, default=20)
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Use reduced workload for quick smoke benchmarks.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--eval-tasks", type=int, default=8)
    parser.add_argument("--eval-eps", type=int, default=2)
    parser.add_argument("--env-tasks", type=int, default=6)
    parser.add_argument("--env-eps", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark(
        cfg_path=args.cfg,
        out_dir=args.out_dir,
        meta_iters=args.meta_iters,
        fast_mode=args.fast_mode,
        seeds=args.seeds,
        eval_tasks=args.eval_tasks,
        eval_eps=args.eval_eps,
        env_tasks=args.env_tasks,
        env_eps=args.env_eps,
        device=args.device,
    )


if __name__ == "__main__":
    main()
