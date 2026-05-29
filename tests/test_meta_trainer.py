from __future__ import annotations

import csv

from scripts.meta_adaptation_diagnostics import run_meta_adaptation_diagnostics
from tchhmrl.meta.meta_trainer import MetaTrainer
from tchhmrl.utils.config import load_cfg


def test_meta_trainer_one_iter_explicit_inner_outer_smoke(tmp_path):
    cfg = load_cfg("configs/default.yaml")
    cfg["experiment"]["log_dir"] = str(tmp_path)
    cfg["experiment"]["run_name"] = "meta_smoke"
    cfg["experiment"]["seed"] = 7
    cfg["env"]["episode_len"] = 6
    cfg["agent"]["hidden_dim"] = 32
    cfg["agent"]["batch_size"] = 4
    cfg["agent"]["warmup_steps"] = 4
    cfg["buffer"]["replay_size"] = 128
    cfg["buffer"]["context_max_len"] = 32
    cfg["meta"]["meta_iters"] = 1
    cfg["meta"]["n_tasks_per_iter"] = 1
    cfg["meta"]["support_episodes"] = 1
    cfg["meta"]["query_episodes"] = 1
    cfg["meta"]["inner_warmup_steps"] = 4
    cfg["meta"]["inner_upper_warmup_steps"] = 2
    cfg["upper_dqn"]["epsilon_decay_steps"] = 10
    cfg["context"]["gru_hidden"] = 16

    trainer = MetaTrainer(cfg)
    csv_path = trainer.train(meta_iters=1)

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows
    assert "query_reward" in rows[0]
    assert "lambda_qos" in rows[0]
    assert "query_cost_temp_anchor" in rows[0]


def test_meta_trainer_no_dual_and_no_context_smoke(tmp_path):
    cfg = load_cfg("configs/default.yaml")
    cfg["experiment"]["log_dir"] = str(tmp_path)
    cfg["experiment"]["run_name"] = "ablation_smoke"
    cfg["experiment"]["seed"] = 11
    cfg["env"]["episode_len"] = 6
    cfg["agent"]["hidden_dim"] = 32
    cfg["agent"]["batch_size"] = 4
    cfg["agent"]["warmup_steps"] = 4
    cfg["buffer"]["replay_size"] = 128
    cfg["buffer"]["context_max_len"] = 32
    cfg["meta"]["meta_iters"] = 1
    cfg["meta"]["n_tasks_per_iter"] = 1
    cfg["meta"]["support_episodes"] = 2
    cfg["meta"]["query_episodes"] = 0
    cfg["meta"]["explicit_inner_outer"] = False
    cfg["meta"]["query_updates_enabled"] = False
    cfg["meta"]["dual_enabled"] = False
    cfg["meta"]["dual_lr"] = 0.0
    cfg["meta"]["dual_lrs"] = [0.0, 0.0, 0.0, 0.0]
    cfg["context"]["enabled"] = False
    cfg["upper_dqn"]["epsilon_decay_steps"] = 10
    cfg["context"]["gru_hidden"] = 16

    trainer = MetaTrainer(cfg)
    csv_path = trainer.train(meta_iters=1)

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows
    assert float(rows[0]["lambda_qos"]) == 0.0


def test_meta_adaptation_diagnostics_outputs_meta_vs_wo_meta_rows(tmp_path):
    summary = run_meta_adaptation_diagnostics(
        out_dir=tmp_path,
        scenario="moderate_practical",
        seed=13,
        train_iters=0,
        n_tasks=1,
        support_episodes=1,
        query_episodes=1,
        episode_len=4,
        fast_mode=True,
        device="cpu",
        make_plots=False,
    )

    assert summary["diagnostic_contract"]["hybrid_meta"]["support_train_adapts"] is True
    assert summary["diagnostic_contract"]["hybrid_wo_meta"]["support_train_adapts"] is False
    assert "query_reward_delta_meta_minus_wo_meta" in summary["comparison"]
    assert summary["fixed_task_batch_hash"]
    assert summary["ordered_fixed_task_batch_hash"]
    assert "meta_query_reward_after_minus_before_support" in summary["comparison"]
    assert "few_shot_reward_gain_meta_minus_wo_meta" in summary["comparison"]
    assert "support_reward_mean" in summary["adaptation_summary"]["hybrid_meta"]
    assert "query_reward_before_support" in summary["adaptation_summary"]["hybrid_meta"]
    assert "query_reward_after_support" in summary["adaptation_summary"]["hybrid_meta"]
    assert summary["adaptation_summary"]["hybrid_meta"]["query_has_support_context_fraction"] == 1.0
    assert summary["adaptation_summary"]["hybrid_wo_meta"]["query_has_support_context_fraction"] == 0.0

    with open(summary["csv_path"], newline="") as f:
        rows = list(csv.DictReader(f))

    assert {row["variant"] for row in rows} == {"hybrid_meta", "hybrid_wo_meta"}
    assert {row["phase"] for row in rows} == {"pre_query", "support", "query"}
    assert all("violation_rate" in row for row in rows)
    pre_query = [row for row in rows if row["phase"] == "pre_query"]
    assert pre_query
    assert pre_query[0]["pre_query_eval_before_support"] == "True"
    for variant in {"hybrid_meta", "hybrid_wo_meta"}:
        before = [row for row in rows if row["variant"] == variant and row["phase"] == "pre_query"][0]
        after = [row for row in rows if row["variant"] == variant and row["phase"] == "query"][0]
        assert before["episode_seed"] == after["episode_seed"]
    meta_support = [row for row in rows if row["variant"] == "hybrid_meta" and row["phase"] == "support"]
    wo_support = [row for row in rows if row["variant"] == "hybrid_wo_meta" and row["phase"] == "support"]
    assert meta_support[0]["support_train_adapts"] == "True"
    assert wo_support[0]["support_train_adapts"] == "False"
    assert int(meta_support[0]["context_history_len_before_query"]) > 0
    assert int(wo_support[0]["context_history_len_before_query"]) == 0
    assert "support_parameter_delta_norm" in rows[0]
