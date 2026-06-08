from __future__ import annotations

import copy
import csv

import torch

from scripts.meta_adaptation_diagnostics import run_meta_adaptation_diagnostics
from tchhmrl.meta.meta_trainer import MetaTrainer
from tchhmrl.utils.config import load_cfg


def test_default_meta_protocol_uses_heldout_query_and_stable_checkpoint_selection():
    for cfg_path in ("configs/default.yaml", "configs/moderate.yaml"):
        cfg = load_cfg(cfg_path)
        assert int(cfg["meta"]["meta_iters"]) == 80
        assert int(cfg["meta"]["support_episodes"]) == 5
        assert int(cfg["meta"]["query_episodes"]) == 2
        assert bool(cfg["meta"]["query_updates_enabled"]) is False
        assert bool(cfg["meta"]["query_context_updates_enabled"]) is True
        assert str(cfg["meta"]["protocol_name"]) == "strict_support_query"
        assert int(cfg["buffer"]["context_max_len"]) >= int(cfg["env"]["episode_len"]) * int(cfg["meta"]["support_episodes"])
        assert int(cfg["upper_dqn"]["batch_size"]) == 64

        selection = cfg["meta"]["checkpoint_selection"]
        assert bool(selection["enabled"]) is True
        assert str(selection["mode"]) == "heldout_eval"
        assert int(selection["eval_tasks"]) == 8
        assert int(selection["eval_eps"]) == 3


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
    assert "query_updates_enabled" in rows[0]
    assert "query_context_updates_enabled" in rows[0]
    assert "heldout_query_evaluation" in rows[0]
    assert "support_parameter_delta_norm" in rows[0]
    assert "support_target_parameter_delta_norm" in rows[0]
    assert "upper_batch_size" in rows[0]
    assert "iter_upper_update_step_delta" in rows[0]


def test_support_delta_norm_ignores_optimizer_state():
    base = {
        "upper": {
            "q": {"w": torch.zeros(2)},
            "q_tgt": {"w": torch.zeros(2)},
            "optim": {"state": {"m": torch.zeros(2)}},
        },
        "lower": {
            "actor": {"w": torch.zeros(2)},
            "q1": {"w": torch.zeros(2)},
            "q2": {"w": torch.zeros(2)},
            "q1_tgt": {"w": torch.zeros(2)},
            "q2_tgt": {"w": torch.zeros(2)},
            "actor_optim": {"state": {"m": torch.zeros(2)}},
        },
        "context_encoder": {"w": torch.zeros(2)},
        "context_predictor": {"w": torch.zeros(2)},
        "context_optim": {"state": {"m": torch.zeros(2)}},
    }
    optimizer_only = copy.deepcopy(base)
    optimizer_only["upper"]["optim"]["state"]["m"] = torch.ones(2)
    optimizer_only["context_optim"]["state"]["m"] = torch.ones(2)

    assert MetaTrainer._trainable_parameter_delta_norm(base, optimizer_only) == 0.0

    param_changed = copy.deepcopy(base)
    param_changed["lower"]["actor"]["w"] = torch.ones(2)
    assert MetaTrainer._trainable_parameter_delta_norm(base, param_changed) > 0.0
    assert MetaTrainer._target_parameter_delta_norm(base, param_changed) == 0.0


def test_meta_trainer_upper_batch_size_allows_short_support_upper_update(tmp_path):
    cfg = load_cfg("configs/default.yaml")
    cfg["experiment"]["log_dir"] = str(tmp_path)
    cfg["experiment"]["run_name"] = "upper_short_support"
    cfg["experiment"]["seed"] = 17
    cfg["env"]["episode_len"] = 8
    cfg["agent"]["hidden_dim"] = 32
    cfg["agent"]["batch_size"] = 8
    cfg["upper_dqn"]["batch_size"] = 2
    cfg["agent"]["warmup_steps"] = 4
    cfg["agent"]["upper_update_every"] = 1
    cfg["buffer"]["replay_size"] = 128
    cfg["buffer"]["context_max_len"] = 32
    cfg["meta"]["meta_iters"] = 1
    cfg["meta"]["n_tasks_per_iter"] = 1
    cfg["meta"]["support_episodes"] = 1
    cfg["meta"]["query_episodes"] = 0
    cfg["meta"]["inner_warmup_steps"] = 4
    cfg["meta"]["inner_upper_warmup_steps"] = 2
    cfg["upper_dqn"]["epsilon_decay_steps"] = 10
    cfg["context"]["gru_hidden"] = 16

    trainer = MetaTrainer(cfg)
    trainer.train(meta_iters=1)

    assert trainer.agent.upper.update_steps > 0


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
    assert summary["diagnostic_contract"]["hybrid_meta"]["query_train_updates"] is False
    assert summary["diagnostic_contract"]["hybrid_meta_no_support_adapt"]["support_train_adapts"] is False
    assert summary["diagnostic_contract"]["hybrid_meta_no_support_adapt"]["same_checkpoint_as"] == "hybrid_meta"
    assert summary["diagnostic_contract"]["hybrid_wo_meta"]["support_train_adapts"] is False
    assert "query_reward_delta_meta_minus_wo_meta" in summary["comparison"]
    assert "query_reward_delta_meta_minus_no_support_adapt" in summary["comparison"]
    assert "few_shot_reward_gain_meta_minus_no_support_adapt" in summary["comparison"]
    assert summary["fixed_task_batch_hash"]
    assert summary["ordered_fixed_task_batch_hash"]
    assert "meta_query_reward_after_minus_before_support" in summary["comparison"]
    assert "few_shot_reward_gain_meta_minus_wo_meta" in summary["comparison"]
    assert "support_reward_mean" in summary["adaptation_summary"]["hybrid_meta"]
    assert "query_reward_before_support" in summary["adaptation_summary"]["hybrid_meta"]
    assert "query_reward_after_support" in summary["adaptation_summary"]["hybrid_meta"]
    assert summary["adaptation_summary"]["hybrid_meta"]["query_has_support_context_fraction"] == 1.0
    assert summary["adaptation_summary"]["hybrid_meta_no_support_adapt"]["query_has_support_context_fraction"] == 1.0
    assert summary["adaptation_summary"]["hybrid_wo_meta"]["query_has_support_context_fraction"] == 0.0
    assert "support_upper_delta_norm" in summary["adaptation_summary"]["hybrid_meta"]
    assert "support_lower_actor_delta_norm" in summary["adaptation_summary"]["hybrid_meta"]
    assert "support_context_encoder_delta_norm" in summary["adaptation_summary"]["hybrid_meta"]
    assert summary["adaptation_summary"]["hybrid_meta"]["query_global_step_delta"] == 0.0
    assert summary["adaptation_summary"]["hybrid_meta"]["query_lower_update_delta"] == 0.0
    assert summary["adaptation_summary"]["hybrid_meta"]["query_upper_update_delta"] == 0.0
    assert summary["adaptation_summary"]["hybrid_meta"]["query_parameter_delta_norm"] == 0.0

    with open(summary["csv_path"], newline="") as f:
        rows = list(csv.DictReader(f))

    assert {row["variant"] for row in rows} == {
        "hybrid_meta",
        "hybrid_meta_no_support_adapt",
        "hybrid_wo_meta",
    }
    assert {row["phase"] for row in rows} == {"pre_query", "support", "query"}
    assert all("violation_rate" in row for row in rows)
    pre_query = [row for row in rows if row["phase"] == "pre_query"]
    assert pre_query
    assert pre_query[0]["pre_query_eval_before_support"] == "True"
    for variant in {"hybrid_meta", "hybrid_meta_no_support_adapt", "hybrid_wo_meta"}:
        before = [row for row in rows if row["variant"] == variant and row["phase"] == "pre_query"][0]
        after = [row for row in rows if row["variant"] == variant and row["phase"] == "query"][0]
        assert before["episode_seed"] == after["episode_seed"]
    meta_support = [row for row in rows if row["variant"] == "hybrid_meta" and row["phase"] == "support"]
    no_adapt_support = [
        row for row in rows if row["variant"] == "hybrid_meta_no_support_adapt" and row["phase"] == "support"
    ]
    wo_support = [row for row in rows if row["variant"] == "hybrid_wo_meta" and row["phase"] == "support"]
    assert meta_support[0]["support_train_adapts"] == "True"
    assert no_adapt_support[0]["support_train_adapts"] == "False"
    assert wo_support[0]["support_train_adapts"] == "False"
    assert int(meta_support[0]["context_history_len_before_query"]) > 0
    assert int(no_adapt_support[0]["context_history_len_before_query"]) > 0
    assert int(wo_support[0]["context_history_len_before_query"]) == 0
    assert "support_parameter_delta_norm" in rows[0]
    assert "support_upper_delta_norm" in rows[0]
    assert "support_lower_replay_len_after_support" in rows[0]
    assert "support_upper_replay_len_after_support" in rows[0]
    assert "query_parameter_delta_norm" in rows[0]
    meta_query = [row for row in rows if row["variant"] == "hybrid_meta" and row["phase"] == "query"]
    assert meta_query
    assert float(meta_query[0]["query_parameter_delta_norm"]) == 0.0
