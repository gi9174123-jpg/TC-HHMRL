from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from scripts.benchmark_constraint_scenarios import build_statistics_artifact, run_benchmark


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
    assert len({row["selection_task_batch_hash"] for row in rows}) == 1
    assert len({row["eval_task_batch_hash"] for row in rows}) == 1
    assert len({row["env_task_batch_hash"] for row in rows}) == 1
    assert len({row["ordered_selection_task_batch_hash"] for row in rows}) == 1
    assert len({row["ordered_eval_task_batch_hash"] for row in rows}) == 1
    assert len({row["ordered_env_task_batch_hash"] for row in rows}) == 1
    for row in rows:
        assert "scenario" in row and row["scenario"] == "easy_baseline"
        assert "variant" in row
        assert "seed" in row
        assert "sampler_ranges" in row
        assert row["alignment_version"] == "teacher_model_v1"
        assert row["task_summary_version"] == "site_v2"
        assert row["pre_alignment"] is False
        assert row["task_source"] == "site_bank"
        assert row["formally_comparable"] is True
        assert row["selection_task_batch_hash"]
        assert row["eval_task_batch_hash"]
        assert row["env_task_batch_hash"]
        assert row["ordered_selection_task_batch_hash"]
        assert row["ordered_eval_task_batch_hash"]
        assert row["ordered_env_task_batch_hash"]
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
        assert "alignment" in resolved_cfg
        assert resolved_cfg["alignment"]["alignment_version"] == "teacher_model_v1"
        assert resolved_cfg["alignment"]["task_summary_version"] == "site_v2"
        assert "site_bank" in resolved_cfg["sampler"]

    precheck = json.loads(precheck_json.read_text(encoding="utf-8"))
    assert precheck["all_passed"] is True
    assert precheck["alignment_version"] == "teacher_model_v1"
    assert precheck["task_summary_version"] == "site_v2"
    assert precheck["task_source"] == "site_bank"
    assert precheck["task_distribution"]["task_source"] == "site_bank"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["task_distribution_scope"] == "base_config_snapshot"
    stats = report.get("stats_artifacts", {})
    assert stats == {}


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
    assert len({row["selection_task_batch_hash"] for row in rows}) == 1
    assert len({row["eval_task_batch_hash"] for row in rows}) == 1
    assert len({row["env_task_batch_hash"] for row in rows}) == 1
    assert len({row["ordered_eval_task_batch_hash"] for row in rows}) == 1
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
    assert saclag_rows[0]["formally_comparable"] is True
    saclag_cfg = yaml.safe_load(Path(saclag_rows[0]["resolved_config"]).read_text(encoding="utf-8"))
    assert int(saclag_cfg["agent"]["z_dim"]) == 0
    assert bool(saclag_cfg["context"]["enabled"]) is False
    assert bool(saclag_cfg["meta"]["explicit_inner_outer"]) is False

    report = json.loads(report_path.read_text(encoding="utf-8"))
    stats = report.get("stats_artifacts", {})
    assert "stats_hard_stress_targeted" in stats
    assert Path(stats["stats_hard_stress_targeted"]["json"]).exists()
    assert Path(stats["stats_hard_stress_targeted"]["csv"]).exists()
    stats_payload = json.loads(Path(stats["stats_hard_stress_targeted"]["json"]).read_text(encoding="utf-8"))
    assert stats_payload["pairing_key_fields"] == [
        "scenario",
        "seed",
        "eval_task_batch_hash",
        "ordered_eval_task_batch_hash",
    ]
    pairs = stats_payload["pairwise"]["hard_stress"]
    reward_pair = next(
        row
        for row in pairs
        if row["metric"] == "reward"
        and row["left_variant"] == "hybrid"
        and row["right_variant"] == "sac_lagrangian"
    )
    assert reward_pair["pairing_key_fields"] == stats_payload["pairing_key_fields"]
    assert reward_pair["n_pairs"] == 1
    assert reward_pair["insufficient_pairs"] is True
    assert reward_pair["p_value_trusted"] is False
    assert reward_pair["p_value"] is None
    assert reward_pair["pair_keys"][0][0] == "hard_stress"


def test_benchmark_supports_shin2024_and_dalal_baselines(tmp_path: Path):
    out_dir = tmp_path / "bench_phase2"
    report_path = run_benchmark(
        cfg_path="configs/default.yaml",
        out_dir=str(out_dir),
        scenarios=["thermal_rebalanced"],
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
        ablations=["full"],
        baselines=["shin2024_matched", "dalal2018_safe"],
    )
    assert report_path.exists()

    rows = json.loads((out_dir / "thermal_rebalanced" / "run_summary.json").read_text(encoding="utf-8"))
    labels = {row["variant"] for row in rows}
    assert labels == {"hybrid", "shin2024_matched", "dalal2018_safe"}

    shin_rows = [row for row in rows if row["variant"] == "shin2024_matched"]
    dalal_rows = [row for row in rows if row["variant"] == "dalal2018_safe"]
    assert len(shin_rows) == 1
    assert len(dalal_rows) == 1
    assert shin_rows[0]["runner"] == "shin2024_matched"
    assert dalal_rows[0]["runner"] == "trainer"
    assert shin_rows[0]["baseline_family"] == "shin2024_matched"
    assert shin_rows[0]["exact_reproduction"] is False
    assert shin_rows[0]["safety_protocol"] == "common_smooth_projection"
    assert shin_rows[0]["lower_learned_action_dim"] == 2
    assert shin_rows[0]["fixed_mode_exec"] == 2
    assert shin_rows[0]["fixed_current_template"] == "sigmoid_logit_fraction"

    shin_cfg = yaml.safe_load(Path(shin_rows[0]["resolved_config"]).read_text(encoding="utf-8"))
    dalal_cfg = yaml.safe_load(Path(dalal_rows[0]["resolved_config"]).read_text(encoding="utf-8"))
    assert int(shin_cfg["agent"]["z_dim"]) == 0
    assert bool(shin_cfg["context"]["enabled"]) is False
    assert shin_cfg["safety"]["projection_mode"] == "smooth"
    assert shin_cfg["lower_ddpg"]["action_contract"] == "rho_tau_fixed_current"
    assert int(shin_cfg["lower_ddpg"]["learned_action_dim"]) == 2
    assert dalal_cfg["safety"]["projection_mode"] == "dalal_safe"

    env_df = pd.read_csv(out_dir / "thermal_rebalanced" / "env.csv")
    shin_env = env_df[env_df["variant"] == "shin2024_matched"]
    assert not shin_env.empty
    assert set(shin_env["mode_exec"].astype(int).unique()) == {2}
    assert all((shin_env["upper_idx_exec"].astype(int) % 3) == 2)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    stats = report.get("stats_artifacts", {})
    assert "stats_thermal_rebalanced" in stats
    assert Path(stats["stats_thermal_rebalanced"]["json"]).exists()
    assert Path(stats["stats_thermal_rebalanced"]["csv"]).exists()
    stats_payload = json.loads(Path(stats["stats_thermal_rebalanced"]["json"]).read_text(encoding="utf-8"))
    assert stats_payload["pairing_key_fields"] == [
        "scenario",
        "seed",
        "eval_task_batch_hash",
        "ordered_eval_task_batch_hash",
    ]


def test_statistics_contract_trusts_pvalue_only_with_two_or_more_pairs():
    rows = []
    for seed, h_reward, s_reward in [(101, 10.0, 8.0), (202, 11.0, 7.0)]:
        for variant, reward in [("hybrid", h_reward), ("sac_lagrangian", s_reward)]:
            rows.append(
                {
                    "scenario": "hard_stress",
                    "variant": variant,
                    "seed": seed,
                    "eval_task_batch_hash": "same-task-set",
                    "ordered_eval_task_batch_hash": "same-task-order",
                    "eval_reward": reward,
                    "eval_se": reward,
                    "eval_eh": reward,
                    "eval_cost": 0.1,
                    "eval_violation_rate": 0.0,
                }
            )

    artifact = build_statistics_artifact(
        rows,
        artifact_name="stats_contract",
        scenarios=["hard_stress"],
        variant_order=("hybrid", "sac_lagrangian"),
        metrics=["reward"],
    )
    assert artifact is not None
    pair = artifact["pairwise"]["hard_stress"][0]
    assert pair["pairing_key_fields"] == [
        "scenario",
        "seed",
        "eval_task_batch_hash",
        "ordered_eval_task_batch_hash",
    ]
    assert pair["n_pairs"] == 2
    assert pair["insufficient_pairs"] is False
    assert pair["p_value_trusted"] is True
    assert pair["p_value"] is not None
    assert pair["pair_keys"] == [
        ["hard_stress", 101, "same-task-set", "same-task-order"],
        ["hard_stress", 202, "same-task-set", "same-task-order"],
    ]
