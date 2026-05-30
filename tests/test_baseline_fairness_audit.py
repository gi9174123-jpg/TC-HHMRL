from __future__ import annotations

import json

import pandas as pd

from scripts.audit_baseline_fairness import audit_run_dir


def _summary_row(scenario: str, baseline: str, **extra):
    meta = {
        "baseline_family": baseline,
        "paper_inspired": True,
        "exact_reproduction": False,
        "uses_same_safety_projection": True,
        "meta_learning": False,
        "shared_lagrangian": False,
        "action_contract": f"{baseline}_contract",
        "selected_action_contract": f"{baseline}_contract",
        "paper_core_mechanism": f"{baseline} paper core",
        "adapted_mapping_to_tc_hhmrl": f"{baseline} adapted mapping",
        "domain_match": "test_domain",
        "environment_dependency": "depends_on_scenario_thresholds_and_channel",
        "not_exact_reproduction_reason": "adapted_to_project_environment",
    }
    row = {
        "scenario": scenario,
        "variant": baseline,
        "baseline_family": baseline,
        "baseline_metadata": meta,
        **meta,
    }
    row.update(extra)
    return row


def test_baseline_fairness_audit_outputs_contract_rows(tmp_path):
    run_dir = tmp_path / "smoke"
    scenario_dir = run_dir / "moderate_practical"
    scenario_dir.mkdir(parents=True)

    rows = [
        _summary_row("moderate_practical", "uysal_policy_optimizer"),
        _summary_row("moderate_practical", "mpc_grid"),
    ]
    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f)

    env_rows = [
        {
            "variant": "uysal_policy_optimizer",
            "selected_env_rho": 0.4,
            "selected_paper_rho": 0.6,
            "predicted_qos_rate": 0.8,
            "predicted_eh_metric": 0.0025,
            "selected_uysal_subpolicy": "uysal_tsps",
            "ads_balanced_predicted_qos_rate": 0.7,
            "ads_balanced_predicted_eh_metric": 0.0022,
            "ads_qos_threshold": 0.1,
            "ads_eh_threshold": 0.002,
            "ads_qos_deficit": 0.0,
            "ads_eh_deficit": 0.0,
            "ads_decision_reason": "qos_and_eh_satisfied_select_tsps",
        },
        {
            "variant": "mpc_grid",
            "selected_env_rho": 0.5,
            "selected_paper_rho": 0.5,
            "predicted_qos_rate": 0.9,
            "predicted_eh_metric": 0.001,
            "candidate_count": 700,
            "online_latency_ms": 101.0,
        },
    ]
    pd.DataFrame(env_rows).to_csv(scenario_dir / "env.csv", index=False)

    audit_df, csv_path, json_path = audit_run_dir(
        run_dir,
        baselines=["uysal_policy_optimizer", "mpc_grid"],
    )

    assert csv_path.exists()
    assert json_path.exists()
    assert len(audit_df) == 2
    assert audit_df["contract_ok"].all()

    uysal = audit_df[audit_df["baseline_family"] == "uysal_policy_optimizer"].iloc[0]
    assert uysal["uysal_tsps_fraction"] == 1.0
    assert uysal["mean_ads_eh_threshold"] == 0.002
    assert uysal["ads_decision_reason_mode"] == "qos_and_eh_satisfied_select_tsps"

    mpc = audit_df[audit_df["baseline_family"] == "mpc_grid"].iloc[0]
    assert mpc["mean_candidate_count"] == 700.0
    assert mpc["mean_online_latency_ms"] == 101.0
