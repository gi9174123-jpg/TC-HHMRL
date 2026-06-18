from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from scripts.meta_adaptation_diagnostics import run_meta_adaptation_diagnostics


DEFAULT_SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1001]
DEFAULT_SCENARIOS = ["moderate_practical", "hard_stress"]


def _flatten_adaptation_summary(summary: dict) -> list[dict]:
    rows: list[dict] = []
    scenario = str(summary["scenario"])
    seed = int(summary["seed"])
    comparison = summary.get("comparison", {})
    for variant, metrics in summary.get("adaptation_summary", {}).items():
        row = {
            "scenario": scenario,
            "seed": seed,
            "variant": variant,
            "train_iters": int(summary["train_iters"]),
            "n_tasks": int(summary["n_tasks"]),
            "pre_query_episodes": int(summary["pre_query_episodes"]),
            "support_episodes": int(summary["support_episodes"]),
            "query_episodes": int(summary["query_episodes"]),
            "fixed_task_batch_hash": summary.get("fixed_task_batch_hash", ""),
            "ordered_fixed_task_batch_hash": summary.get("ordered_fixed_task_batch_hash", ""),
        }
        row.update({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
        for key, val in comparison.items():
            if isinstance(val, (int, float)):
                row[f"comparison__{key}"] = float(val)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal 10-seed meta/context causal diagnostics.")
    parser.add_argument("--cfg", default="configs/default.yaml")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--scenarios", nargs="+", default=DEFAULT_SCENARIOS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--train-iters", type=int, default=45)
    parser.add_argument("--tasks", type=int, default=8)
    parser.add_argument("--pre-query-episodes", type=int, default=2)
    parser.add_argument("--support-episodes", type=int, default=5)
    parser.add_argument("--query-episodes", type=int, default=2)
    parser.add_argument("--episode-len", type=int, default=80)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fast-mode", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)

    seed_rows: list[dict] = []
    episode_frames: list[pd.DataFrame] = []
    summaries: list[dict] = []
    for scenario in args.scenarios:
        for seed in args.seeds:
            run_dir = root / scenario / f"seed{seed}"
            summary = run_meta_adaptation_diagnostics(
                cfg_path=args.cfg,
                out_dir=run_dir,
                scenario=scenario,
                seed=seed,
                train_iters=args.train_iters,
                n_tasks=args.tasks,
                support_episodes=args.support_episodes,
                query_episodes=args.query_episodes,
                pre_query_episodes=args.pre_query_episodes,
                episode_len=args.episode_len,
                fast_mode=bool(args.fast_mode),
                device=args.device,
                make_plots=not bool(args.no_plots),
            )
            summaries.append(summary)
            seed_rows.extend(_flatten_adaptation_summary(summary))
            csv_path = Path(str(summary["csv_path"]))
            if csv_path.exists():
                frame = pd.read_csv(csv_path)
                frame["scenario"] = scenario
                frame["seed"] = seed
                episode_frames.append(frame)

    seed_df = pd.DataFrame(seed_rows)
    seed_csv = root / "meta_context_seed_summary.csv"
    seed_df.to_csv(seed_csv, index=False)

    stats_rows: list[dict] = []
    if not seed_df.empty:
        metric_cols = [
            c
            for c in seed_df.columns
            if c
            not in {
                "scenario",
                "seed",
                "variant",
                "fixed_task_batch_hash",
                "ordered_fixed_task_batch_hash",
            }
            and pd.api.types.is_numeric_dtype(seed_df[c])
        ]
        grouped = seed_df.groupby(["scenario", "variant"], as_index=False)
        for (scenario, variant), group in grouped:
            row = {"scenario": scenario, "variant": variant, "n": int(len(group))}
            for col in metric_cols:
                row[f"{col}_mean"] = float(group[col].mean())
                row[f"{col}_std"] = float(group[col].std(ddof=0))
            stats_rows.append(row)
    stats_df = pd.DataFrame(stats_rows)
    stats_csv = root / "meta_context_variant_stats.csv"
    stats_df.to_csv(stats_csv, index=False)

    episode_csv = root / "meta_context_episode_metrics.csv"
    if episode_frames:
        pd.concat(episode_frames, ignore_index=True).to_csv(episode_csv, index=False)
    else:
        pd.DataFrame().to_csv(episode_csv, index=False)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cfg": str(args.cfg),
        "out_dir": str(root),
        "scenarios": list(args.scenarios),
        "seeds": list(args.seeds),
        "train_iters": int(args.train_iters),
        "tasks": int(args.tasks),
        "pre_query_episodes": int(args.pre_query_episodes),
        "support_episodes": int(args.support_episodes),
        "query_episodes": int(args.query_episodes),
        "episode_len": int(args.episode_len),
        "device": str(args.device),
        "variants": [
            "hybrid_meta",
            "hybrid_meta_no_support_adapt",
            "hybrid_context_only",
            "hybrid_wo_meta",
        ],
        "purpose": "meta_context_causal_diagnostic",
        "artifacts": {
            "seed_summary_csv": str(seed_csv),
            "variant_stats_csv": str(stats_csv),
            "episode_metrics_csv": str(episode_csv),
        },
        "summaries": summaries,
    }
    report_path = root / "report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report["artifacts"], indent=2))
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
