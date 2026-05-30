from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PAPER_BASELINES = [
    "shin2024_adapted_codebook",
    "uysal_policy_optimizer",
    "mpc_grid",
    "javadi_ppo_dimming",
    "deeprat_assignment_power",
    "pdqn_hybrid_action",
]

SUMMARY_REQUIRED_FIELDS = [
    "baseline_metadata",
    "paper_inspired",
    "exact_reproduction",
    "action_contract",
    "selected_action_contract",
    "uses_same_safety_projection",
    "meta_learning",
    "shared_lagrangian",
    "paper_core_mechanism",
    "adapted_mapping_to_tc_hhmrl",
    "domain_match",
    "environment_dependency",
    "not_exact_reproduction_reason",
]

ENV_REQUIRED_FIELDS = [
    "selected_env_rho",
    "selected_paper_rho",
    "predicted_qos_rate",
    "predicted_eh_metric",
]

UYSAL_ADS_FIELDS = [
    "ads_balanced_predicted_qos_rate",
    "ads_balanced_predicted_eh_metric",
    "ads_qos_threshold",
    "ads_eh_threshold",
    "ads_qos_deficit",
    "ads_eh_deficit",
    "ads_decision_reason",
]

MPC_FIELDS = ["candidate_count", "online_latency_ms"]


def _load_run_summary(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "run_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"missing root run_summary.json: {path}")
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"run_summary.json must contain a list of rows: {path}")
    return [dict(row) for row in rows]


def _scenario_env(run_dir: Path, scenario: str) -> pd.DataFrame:
    path = run_dir / str(scenario) / "env.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, dict) and len(value) == 0:
        return True
    return False


def _mean_if_present(df: pd.DataFrame, column: str) -> float | None:
    if df.empty or column not in df:
        return None
    vals = pd.to_numeric(df[column], errors="coerce").dropna()
    if vals.empty:
        return None
    return float(vals.mean())


def _fraction_if_present(df: pd.DataFrame, column: str, value: str) -> float | None:
    if df.empty or column not in df:
        return None
    return float((df[column].astype(str) == value).mean())


def _mode_if_present(df: pd.DataFrame, column: str) -> str:
    if df.empty or column not in df:
        return ""
    vals = df[column].dropna().astype(str)
    if vals.empty:
        return ""
    return str(vals.mode().iloc[0])


def _env_rows_for_baseline(env_df: pd.DataFrame, baseline: str) -> pd.DataFrame:
    if env_df.empty:
        return env_df
    for col in ("baseline_family", "variant", "runner"):
        if col in env_df:
            matched = env_df[env_df[col].astype(str) == baseline]
            if not matched.empty:
                return matched
    return pd.DataFrame()


def _summary_group_key(row: dict[str, Any]) -> tuple[str, str]:
    scenario = str(row.get("scenario", ""))
    baseline = str(row.get("baseline_family") or row.get("variant") or "")
    return scenario, baseline


def _audit_one(scenario: str, baseline: str, summary_rows: list[dict[str, Any]], env_df: pd.DataFrame) -> dict[str, Any]:
    first = summary_rows[0] if summary_rows else {}
    meta = first.get("baseline_metadata", {})
    if not isinstance(meta, dict):
        meta = {}

    summary_missing = [key for key in SUMMARY_REQUIRED_FIELDS if _is_missing(first.get(key, meta.get(key)))]
    env_missing = [key for key in ENV_REQUIRED_FIELDS if env_df.empty or key not in env_df]
    if baseline == "uysal_policy_optimizer":
        env_missing.extend([key for key in UYSAL_ADS_FIELDS if env_df.empty or key not in env_df])
    if baseline == "mpc_grid":
        env_missing.extend([key for key in MPC_FIELDS if env_df.empty or key not in env_df])

    paper_inspired = first.get("paper_inspired", meta.get("paper_inspired"))
    exact_reproduction = first.get("exact_reproduction", meta.get("exact_reproduction"))
    uses_same_safety = first.get("uses_same_safety_projection", meta.get("uses_same_safety_projection"))

    contract_ok = (
        bool(paper_inspired) is True
        and bool(exact_reproduction) is False
        and bool(uses_same_safety) is True
        and not summary_missing
        and not env_missing
    )

    row = {
        "scenario": scenario,
        "baseline_family": baseline,
        "n_summary_rows": len(summary_rows),
        "n_env_rows": int(len(env_df)),
        "paper_inspired": paper_inspired,
        "exact_reproduction": exact_reproduction,
        "uses_same_safety_projection": uses_same_safety,
        "domain_match": first.get("domain_match", meta.get("domain_match", "")),
        "action_contract": first.get("action_contract", meta.get("action_contract", "")),
        "selected_action_contract": first.get("selected_action_contract", meta.get("selected_action_contract", "")),
        "paper_core_mechanism_present": not _is_missing(first.get("paper_core_mechanism", meta.get("paper_core_mechanism"))),
        "adapted_mapping_present": not _is_missing(
            first.get("adapted_mapping_to_tc_hhmrl", meta.get("adapted_mapping_to_tc_hhmrl"))
        ),
        "summary_missing_fields": ";".join(summary_missing),
        "env_missing_fields": ";".join(env_missing),
        "field_ok": not summary_missing and not env_missing,
        "contract_ok": contract_ok,
        "mean_predicted_qos_rate": _mean_if_present(env_df, "predicted_qos_rate"),
        "mean_predicted_eh_metric": _mean_if_present(env_df, "predicted_eh_metric"),
        "mean_online_latency_ms": _mean_if_present(env_df, "online_latency_ms"),
        "mean_candidate_count": _mean_if_present(env_df, "candidate_count"),
        "uysal_ts_fraction": _fraction_if_present(env_df, "selected_uysal_subpolicy", "uysal_ts"),
        "uysal_ps_fraction": _fraction_if_present(env_df, "selected_uysal_subpolicy", "uysal_ps"),
        "uysal_tsps_fraction": _fraction_if_present(env_df, "selected_uysal_subpolicy", "uysal_tsps"),
        "mean_ads_balanced_predicted_qos_rate": _mean_if_present(env_df, "ads_balanced_predicted_qos_rate"),
        "mean_ads_balanced_predicted_eh_metric": _mean_if_present(env_df, "ads_balanced_predicted_eh_metric"),
        "mean_ads_qos_threshold": _mean_if_present(env_df, "ads_qos_threshold"),
        "mean_ads_eh_threshold": _mean_if_present(env_df, "ads_eh_threshold"),
        "mean_ads_qos_deficit": _mean_if_present(env_df, "ads_qos_deficit"),
        "mean_ads_eh_deficit": _mean_if_present(env_df, "ads_eh_deficit"),
        "ads_decision_reason_mode": _mode_if_present(env_df, "ads_decision_reason"),
    }
    return row


def audit_run_dir(run_dir: str | Path, baselines: list[str] | None = None) -> tuple[pd.DataFrame, Path, Path]:
    run_dir = Path(run_dir)
    baselines = list(baselines or PAPER_BASELINES)
    summary_rows = _load_run_summary(run_dir)

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in summary_rows:
        key = _summary_group_key(row)
        if key[1] in baselines:
            grouped.setdefault(key, []).append(row)

    scenarios = sorted({key[0] for key in grouped})
    env_by_scenario = {scenario: _scenario_env(run_dir, scenario) for scenario in scenarios}

    audit_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        env_df = env_by_scenario[scenario]
        for baseline in baselines:
            rows = grouped.get((scenario, baseline), [])
            if not rows:
                continue
            audit_rows.append(_audit_one(scenario, baseline, rows, _env_rows_for_baseline(env_df, baseline)))

    audit_df = pd.DataFrame(audit_rows)
    csv_path = run_dir / "baseline_fairness_audit.csv"
    json_path = run_dir / "baseline_fairness_audit.json"
    audit_df.to_csv(csv_path, index=False)

    payload = {
        "run_dir": str(run_dir),
        "baselines": baselines,
        "n_rows": int(len(audit_df)),
        "all_contract_ok": bool(len(audit_df) > 0 and audit_df["contract_ok"].fillna(False).all()),
        "rows": audit_rows,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return audit_df, csv_path, json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit paper-inspired baseline fairness and environment-fit fields.")
    parser.add_argument("--run-dir", required=True, help="Smoke/formal benchmark output directory containing run_summary.json.")
    parser.add_argument("--baselines", nargs="*", default=PAPER_BASELINES, help="Baseline families to audit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_df, csv_path, json_path = audit_run_dir(args.run_dir, baselines=list(args.baselines))
    print(f"Baseline fairness audit rows: {len(audit_df)}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    if len(audit_df) > 0:
        failed = audit_df[~audit_df["contract_ok"].fillna(False)]
        if not failed.empty:
            print("Rows needing attention:")
            print(failed[["scenario", "baseline_family", "summary_missing_fields", "env_missing_fields"]].to_string(index=False))


if __name__ == "__main__":
    main()
