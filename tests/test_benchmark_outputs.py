from __future__ import annotations

import json
from pathlib import Path

import yaml

from scripts.benchmark_constraint_scenarios import run_benchmark


def test_benchmark_writes_resolved_config_and_run_summary(tmp_path: Path):
    out_dir = tmp_path / "bench"
    report_path = run_benchmark(
        cfg_path="configs/default.yaml",
        out_dir=str(out_dir),
        scenarios=["easy_baseline"],
        meta_iters=1,
        fast_mode=True,
        seeds=[101],
        eval_tasks=1,
        eval_eps=1,
        env_tasks=1,
        env_eps=1,
        use_curriculum=False,
    )
    assert report_path.exists()

    scenario_dir = out_dir / "easy_baseline"
    run_summary_json = scenario_dir / "run_summary.json"
    precheck_json = scenario_dir / "precheck.json"
    requested_metrics_csv = scenario_dir / "requested_metrics.csv"
    assert run_summary_json.exists()
    assert precheck_json.exists()
    assert requested_metrics_csv.exists()

    rows = json.loads(run_summary_json.read_text(encoding="utf-8"))
    # 3 variants x 1 seed
    assert len(rows) == 3
    for row in rows:
        assert "scenario" in row and row["scenario"] == "easy_baseline"
        assert "variant" in row
        assert "seed" in row
        assert "sampler_ranges" in row
        assert "shared_init" in row and bool(row["shared_init"]) is True
        assert "shared_init_pretrain_iters" in row
        assert int(row["shared_init_pretrain_iters"]) >= 0
        assert "shared_init_ckpt" in row
        shared_ckpt = Path(row["shared_init_ckpt"])
        assert shared_ckpt.exists()
        resolved = Path(row["resolved_config"])
        assert resolved.exists()

        resolved_cfg = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        assert "env" in resolved_cfg and "safety" in resolved_cfg
        assert "hybrid" in resolved_cfg["env"]

    precheck = json.loads(precheck_json.read_text(encoding="utf-8"))
    assert precheck["all_passed"] is True


def test_benchmark_supports_ablations_and_learning_baseline(tmp_path: Path):
    out_dir = tmp_path / "bench_ablation"
    report_path = run_benchmark(
        cfg_path="configs/default.yaml",
        out_dir=str(out_dir),
        scenarios=["hard_stress"],
        meta_iters=1,
        fast_mode=True,
        seeds=[101],
        eval_tasks=1,
        eval_eps=1,
        env_tasks=1,
        env_eps=1,
        use_curriculum=False,
        shared_init=False,
        variants=["hybrid"],
        ablations=["full", "wo_meta", "wo_lagrangian"],
        baselines=["heuristic_safe", "sac_lagrangian"],
    )
    assert report_path.exists()

    scenario_dir = out_dir / "hard_stress"
    run_summary_json = scenario_dir / "run_summary.json"
    train_csv = scenario_dir / "training.csv"
    eval_csv = scenario_dir / "eval.csv"
    stability_csv = scenario_dir / "stability.csv"
    assert run_summary_json.exists()
    assert train_csv.exists()
    assert eval_csv.exists()
    assert stability_csv.exists()

    rows = json.loads(run_summary_json.read_text(encoding="utf-8"))
    labels = {row["variant"] for row in rows}
    assert labels == {
        "hybrid",
        "hybrid_wo_meta",
        "hybrid_wo_lagrangian",
        "heuristic_safe",
        "sac_lagrangian",
    }

    saclag_rows = [row for row in rows if row["variant"] == "sac_lagrangian"]
    assert len(saclag_rows) == 1
    assert saclag_rows[0]["runner"] == "sac_lagrangian"
    saclag_cfg = yaml.safe_load(Path(saclag_rows[0]["resolved_config"]).read_text(encoding="utf-8"))
    assert int(saclag_cfg["agent"]["z_dim"]) == 0
    assert bool(saclag_cfg["context"]["enabled"]) is False
    assert bool(saclag_cfg["meta"]["explicit_inner_outer"]) is False
