from __future__ import annotations

import copy
import csv

import numpy as np
import torch

from scripts.benchmark_constraint_scenarios import apply_ablation
from scripts.meta_adaptation_diagnostics import run_meta_adaptation_diagnostics
from tchhmrl.agents.hierarchical_agent import HierarchicalAgent
from tchhmrl.envs.uw_slipt_env import MultiTxUwSliptEnv
from tchhmrl.meta.support_gate import SupportGateStats, evaluate_support_gate
from tchhmrl.meta.meta_trainer import MetaTrainer
from tchhmrl.utils.config import load_cfg


def test_default_meta_protocol_uses_heldout_query_and_stable_checkpoint_selection():
    for cfg_path in ("configs/default.yaml", "configs/moderate.yaml"):
        cfg = load_cfg(cfg_path)
        assert int(cfg["meta"]["meta_iters"]) == 100
        assert int(cfg["meta"]["n_tasks_per_iter"]) == 8
        assert int(cfg["meta"]["support_episodes"]) == 5
        assert int(cfg["meta"]["support_adaptation_episodes"]) == 3
        assert int(cfg["meta"]["support_gate_validation_episodes"]) == 0
        assert int(cfg["meta"]["query_episodes"]) == 2
        assert bool(cfg["meta"]["query_updates_enabled"]) is False
        assert bool(cfg["meta"]["query_context_updates_enabled"]) is False
        assert str(cfg["meta"]["protocol_name"]) == "strict_support_query"
        assert bool(cfg["meta"]["support_gate"]["enabled"]) is False
        assert str(cfg["meta"]["support_gate"]["role"]) == "rollback_guard"
        assert str(cfg["meta"]["support_gate"]["rule"]) == "safety_first"
        assert cfg["meta"]["support_gate"]["query_leakage"] is False
        assert cfg["meta"]["support_gate"]["budget_mode"] == "support_gate_disabled"
        assert int(cfg["meta"]["support_gate"]["extra_support_rollouts"]) == 0
        assert float(cfg["meta"]["support_gate"]["max_violation_increase"]) == 1.0e-4
        assert float(cfg["meta"]["support_gate"]["max_cost_increase"]) == 1.0e-5
        assert bool(cfg["meta"]["reset_optimizer_after_outer_update"]) is True
        assert bool(cfg["context"]["policy_deterministic"]) is True
        assert int(cfg["context"]["updates_per_env_step"]) == 1
        assert int(cfg["context"]["train_window_len"]) == 128
        assert cfg["context"]["target_mask"] == [0, 0, 1, 0, 0, 1, 1, 1, 1]
        assert int(cfg["buffer"]["context_max_len"]) >= int(cfg["env"]["episode_len"]) * (
            int(cfg["meta"]["support_episodes"]) + int(cfg["meta"]["query_episodes"])
        )
        assert int(cfg["agent"]["upper_update_every"]) == 1
        assert int(cfg["agent"]["lower_updates_per_step"]) == 2
        assert int(cfg["agent"]["upper_batch_size"]) == 64
        assert int(cfg["upper_dqn"]["batch_size"]) == 64

        selection = cfg["meta"]["checkpoint_selection"]
        assert bool(selection["enabled"]) is True
        assert str(selection["mode"]) == "heldout_eval"
        assert int(selection["min_iter"]) == int(cfg["residual_planner"]["thermal_horizon_start_meta_iter"])
        assert int(selection["eval_tasks"]) == 10
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
    assert "reset_optimizer_after_outer_update" in rows[0]
    assert rows[0]["reset_optimizer_after_outer_update"] == "True"
    assert "support_parameter_delta_norm" in rows[0]
    assert "support_target_parameter_delta_norm" in rows[0]
    assert "upper_batch_size" in rows[0]
    assert "upper_update_every" in rows[0]
    assert "lower_updates_per_step" in rows[0]
    assert "upper_warmup_steps" in rows[0]
    assert "iter_upper_update_step_delta" in rows[0]
    assert "support_gate_enabled" in rows[0]
    assert rows[0]["support_gate_enabled"] == "False"
    assert rows[0]["support_update_acceptance"] == "unconditional"
    assert rows[0]["query_leakage"] == "False"


def test_upper_macro_reward_accumulates_full_hold_horizon(tmp_path):
    cfg = load_cfg("configs/default.yaml")
    cfg["experiment"]["log_dir"] = str(tmp_path)
    cfg["experiment"]["run_name"] = "macro_reward_horizon"
    cfg["experiment"]["seed"] = 11
    cfg["env"]["episode_len"] = 2
    cfg["agent"]["hidden_dim"] = 32
    cfg["agent"]["batch_size"] = 4
    cfg["agent"]["warmup_steps"] = 999
    cfg["agent"]["upper_batch_size"] = 4
    cfg["agent"]["upper_hold_steps"] = 2
    cfg["agent"]["upper_update_every"] = 1
    cfg["upper_dqn"]["batch_size"] = 4
    cfg["upper_dqn"]["epsilon_start"] = 0.0
    cfg["upper_dqn"]["epsilon_final"] = 0.0
    cfg["buffer"]["replay_size"] = 128
    cfg["buffer"]["context_max_len"] = 64
    cfg["meta"]["support_gate"]["enabled"] = False
    cfg["context"]["gru_hidden"] = 16

    trainer = MetaTrainer(cfg)
    env = MultiTxUwSliptEnv(cfg)
    stats = trainer._run_episode(env, train=True, clear_context=True, reset_seed=123)

    lower_items = trainer.agent.replay.state_dict()["items"]
    upper_items = trainer.agent.upper_replay.state_dict()["items"]
    assert stats.length == 2
    assert len(lower_items) == 2
    assert len(upper_items) == 1
    expected = float(lower_items[0]["reward"]) + float(trainer.agent.upper.gamma) * float(lower_items[1]["reward"])
    assert np.isclose(float(upper_items[0]["reward"]), expected, atol=1e-6)
    assert float(upper_items[0]["horizon"]) == 2.0


def test_lower_transition_records_constraint_replay_boundary_fields(tmp_path):
    cfg = load_cfg("configs/default.yaml")
    cfg["experiment"]["log_dir"] = str(tmp_path)
    cfg["experiment"]["run_name"] = "constraint_fields"
    cfg["experiment"]["seed"] = 12
    cfg["env"]["episode_len"] = 3
    cfg["agent"]["hidden_dim"] = 32
    cfg["agent"]["batch_size"] = 4
    cfg["agent"]["warmup_steps"] = 999
    cfg["buffer"]["replay_size"] = 128
    cfg["buffer"]["context_max_len"] = 64
    cfg["meta"]["support_gate"]["enabled"] = False
    cfg["context"]["gru_hidden"] = 16

    trainer = MetaTrainer(cfg)
    trainer._run_episode(MultiTxUwSliptEnv(cfg), train=True, clear_context=True, reset_seed=123)
    item = trainer.agent.replay.state_dict()["items"][0]

    for key in ("bus_utilization", "projected_current_total", "qos_margin", "burst_active"):
        assert key in item
        assert np.isfinite(float(item[key]))


def test_context_policy_inference_uses_deterministic_posterior_mean():
    cfg = load_cfg("configs/default.yaml")
    cfg["agent"]["hidden_dim"] = 32
    cfg["context"]["gru_hidden"] = 16
    agent = HierarchicalAgent(cfg, torch.device("cpu"))
    obs = np.zeros(int(cfg["agent"]["obs_dim"]), dtype=np.float32)
    act = np.zeros(int(cfg["agent"]["act_lower_dim"]), dtype=np.float32)
    for idx in range(3):
        agent.episode.add(
            {
                "obs": obs + float(idx),
                "upper_idx_exec": float(idx),
                "boost_combo_exec": float(idx % 4),
                "mode_exec": float(idx % 3),
                "act_exec": act + 0.01 * float(idx),
                "reward_task": float(idx),
                "cost_vec": np.zeros(4, dtype=np.float32),
            }
        )

    z1 = agent.infer_z()
    z2 = agent.infer_z()

    assert np.allclose(z1, z2)


def test_support_gate_accept_reject_and_query_independence():
    cfg = {"score_threshold": 0.0, "reward_weight": 1.0, "cost_weight": 1.0, "violation_weight": 1.0}
    accepted = evaluate_support_gate(
        SupportGateStats(reward=1.0, cost=0.1, violation_rate=0.0),
        SupportGateStats(reward=1.3, cost=0.1, violation_rate=0.0),
        parameter_delta=0.5,
        config={**cfg, "query_reward": -9999.0},
    )
    accepted_changed_query = evaluate_support_gate(
        SupportGateStats(reward=1.0, cost=0.1, violation_rate=0.0),
        SupportGateStats(reward=1.3, cost=0.1, violation_rate=0.0),
        parameter_delta=0.5,
        config={**cfg, "query_reward": 9999.0, "query_violation": 1.0},
    )
    rejected = evaluate_support_gate(
        SupportGateStats(reward=1.0, cost=0.1, violation_rate=0.0),
        SupportGateStats(reward=0.2, cost=0.5, violation_rate=0.2),
        parameter_delta=0.5,
        config=cfg,
    )

    assert accepted.accepted is True
    assert accepted_changed_query == accepted
    assert accepted.query_leakage is False
    assert accepted.extra_support_rollouts == 0
    assert accepted.extra_gradient_updates == 0
    assert accepted.extra_query_evaluations == 0
    assert rejected.accepted is False
    assert rejected.reason == "reject_support_score_degradation"


def test_safety_first_support_gate_rejects_safety_degradation_despite_reward_gain():
    cfg = {
        "rule": "safety_first",
        "normalized_reward_threshold": 0.0,
        "reward_normalization_eps": 1.0,
        "max_cost_increase": 1.0e-5,
        "max_violation_increase": 1.0e-4,
    }
    violation_rejected = evaluate_support_gate(
        SupportGateStats(reward=1.0, cost=0.1, violation_rate=0.0),
        SupportGateStats(reward=2.0, cost=0.1, violation_rate=2.0e-4),
        parameter_delta=0.5,
        config={**cfg, "query_reward": 999.0},
    )
    cost_rejected = evaluate_support_gate(
        SupportGateStats(reward=1.0, cost=0.1, violation_rate=0.0),
        SupportGateStats(reward=2.0, cost=0.10002, violation_rate=0.0),
        parameter_delta=0.5,
        config=cfg,
    )
    accepted = evaluate_support_gate(
        SupportGateStats(reward=1.0, cost=0.1, violation_rate=0.0),
        SupportGateStats(reward=1.1, cost=0.1, violation_rate=0.0),
        parameter_delta=0.5,
        config={**cfg, "query_reward": -999.0},
    )

    assert violation_rejected.accepted is False
    assert violation_rejected.reason == "reject_support_violation_increase"
    assert violation_rejected.query_leakage is False
    assert cost_rejected.accepted is False
    assert cost_rejected.reason == "reject_support_cost_increase"
    assert accepted.accepted is True
    assert accepted.reason == "accept_safety_first_support_gate"


def test_support_gate_reject_rolls_back_full_training_state(tmp_path):
    cfg = load_cfg("configs/default.yaml")
    cfg["experiment"]["log_dir"] = str(tmp_path)
    cfg["experiment"]["run_name"] = "gate_reject_smoke"
    cfg["experiment"]["seed"] = 19
    cfg["env"]["episode_len"] = 8
    cfg["agent"]["hidden_dim"] = 32
    cfg["agent"]["batch_size"] = 4
    cfg["agent"]["warmup_steps"] = 4
    cfg["agent"]["upper_batch_size"] = 2
    cfg["upper_dqn"]["batch_size"] = 2
    cfg["agent"]["upper_update_every"] = 1
    cfg["buffer"]["replay_size"] = 128
    cfg["buffer"]["context_max_len"] = 64
    cfg["meta"]["meta_iters"] = 1
    cfg["meta"]["n_tasks_per_iter"] = 1
    cfg["meta"]["support_episodes"] = 1
    cfg["meta"]["query_episodes"] = 0
    cfg["meta"]["inner_warmup_steps"] = 4
    cfg["meta"]["inner_upper_warmup_steps"] = 2
    cfg["meta"]["support_gate"]["enabled"] = True
    cfg["meta"]["support_gate"]["normalized_reward_threshold"] = 1.0e9
    cfg["upper_dqn"]["epsilon_decay_steps"] = 10
    cfg["context"]["gru_hidden"] = 16

    trainer = MetaTrainer(cfg)
    csv_path = trainer.train(meta_iters=1)
    with open(csv_path, newline="") as f:
        row = list(csv.DictReader(f))[0]

    assert row["support_gate_enabled"] == "True"
    assert float(row["support_gate_reject_rate"]) == 1.0
    assert row["rollback_performed"] == "True"
    assert float(row["rollback_parameter_residual"]) == 0.0
    assert float(row["rollback_dual_residual"]) == 0.0
    assert float(row["rollback_optimizer_residual"]) == 0.0
    assert float(row["rollback_target_parameter_residual"]) == 0.0
    assert float(row["rollback_context_residual"]) == 0.0
    assert row["optimizer_state_restored"] == "True"
    assert row["target_network_state_restored"] == "True"
    assert row["context_state_restored"] == "True"
    assert row["dual_state_restored"] == "True"
    assert row["thermal_estimator_state_restored"] == "True"
    assert float(row["rollback_safety_estimator_residual"]) == 0.0
    assert int(float(row["extra_support_rollouts"])) == 0
    assert int(float(row["extra_gradient_updates"])) == 0
    assert int(float(row["extra_query_evaluations"])) == 0


def test_agent_mutable_snapshot_restores_safety_runtime_state():
    cfg = copy.deepcopy(load_cfg("configs/default.yaml"))
    cfg["experiment"]["device"] = "cpu"
    agent = HierarchicalAgent(cfg, torch.device("cpu"))

    snap = agent.snapshot_mutable_state()
    agent.safety.update_thermal_estimator(
        currents=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        temps_before=np.array([30.0, 30.0, 30.0], dtype=np.float32),
        temps_after=np.array([35.0, 36.0, 37.0], dtype=np.float32),
        thermal_base=np.array([30.0, 30.0, 30.0], dtype=np.float32),
    )
    agent.safety.upper_boost_allowed_mask(
        temps=np.array([30.0, float(agent.safety.thermal_safe) - 0.5, 30.0], dtype=np.float32)
    )
    assert np.any(agent.safety.thermal_diagnostics()["thermal_gain_valid_count"] > 0.0)
    assert bool(agent.safety.upper_shield_hot_locked[1])

    agent.restore_mutable_state(snap)
    restored = agent.safety.thermal_diagnostics()
    assert np.allclose(restored["thermal_gain_valid_count"], 0.0)
    assert np.allclose(
        restored["thermal_gain_mean"],
        np.asarray(cfg["safety"]["effective_gain_initial"], dtype=np.float32),
    )
    assert not bool(agent.safety.upper_shield_hot_locked[1])


def test_support_gate_accept_keeps_adapted_parameters_without_extra_budget(tmp_path):
    cfg = load_cfg("configs/default.yaml")
    cfg["experiment"]["log_dir"] = str(tmp_path)
    cfg["experiment"]["run_name"] = "gate_accept_smoke"
    cfg["experiment"]["seed"] = 23
    cfg["env"]["episode_len"] = 8
    cfg["agent"]["hidden_dim"] = 32
    cfg["agent"]["batch_size"] = 4
    cfg["agent"]["warmup_steps"] = 4
    cfg["agent"]["upper_batch_size"] = 2
    cfg["upper_dqn"]["batch_size"] = 2
    cfg["agent"]["upper_update_every"] = 1
    cfg["buffer"]["replay_size"] = 128
    cfg["buffer"]["context_max_len"] = 64
    cfg["meta"]["meta_iters"] = 1
    cfg["meta"]["n_tasks_per_iter"] = 1
    cfg["meta"]["support_episodes"] = 1
    cfg["meta"]["query_episodes"] = 0
    cfg["meta"]["inner_warmup_steps"] = 4
    cfg["meta"]["inner_upper_warmup_steps"] = 2
    cfg["meta"]["support_gate"]["enabled"] = True
    cfg["meta"]["support_gate"]["score_threshold"] = -1.0e9
    cfg["upper_dqn"]["epsilon_decay_steps"] = 10
    cfg["context"]["gru_hidden"] = 16

    trainer = MetaTrainer(cfg)
    csv_path = trainer.train(meta_iters=1)
    with open(csv_path, newline="") as f:
        row = list(csv.DictReader(f))[0]

    assert row["support_gate_enabled"] == "True"
    assert float(row["support_gate_accept_rate"]) == 1.0
    assert row["rollback_performed"] == "False"
    assert int(float(row["extra_support_rollouts"])) == 0
    assert int(float(row["extra_gradient_updates"])) == 0
    assert int(float(row["extra_query_evaluations"])) == 0


def test_support_gate_uses_paired_pre_post_validation_seeds(tmp_path):
    cfg = load_cfg("configs/default.yaml")
    cfg["experiment"]["log_dir"] = str(tmp_path)
    cfg["experiment"]["run_name"] = "gate_paired_validation_smoke"
    cfg["experiment"]["seed"] = 29
    cfg["env"]["episode_len"] = 4
    cfg["agent"]["hidden_dim"] = 32
    cfg["agent"]["batch_size"] = 4
    cfg["agent"]["warmup_steps"] = 4
    cfg["agent"]["upper_batch_size"] = 2
    cfg["upper_dqn"]["batch_size"] = 2
    cfg["agent"]["upper_update_every"] = 1
    cfg["buffer"]["replay_size"] = 128
    cfg["buffer"]["context_max_len"] = 64
    cfg["meta"]["meta_iters"] = 1
    cfg["meta"]["n_tasks_per_iter"] = 1
    cfg["meta"]["support_episodes"] = 2
    cfg["meta"]["support_adaptation_episodes"] = 1
    cfg["meta"]["support_gate_validation_episodes"] = 1
    cfg["meta"]["query_episodes"] = 0
    cfg["meta"]["inner_warmup_steps"] = 4
    cfg["meta"]["inner_upper_warmup_steps"] = 2
    cfg["meta"]["support_gate"]["enabled"] = True
    cfg["meta"]["support_gate"]["paired_validation"] = True
    cfg["meta"]["support_gate"]["budget_mode"] = "paired_support_validation"
    cfg["meta"]["support_gate"]["extra_support_rollouts"] = 1
    cfg["meta"]["support_gate"]["score_threshold"] = -1.0e9
    cfg["upper_dqn"]["epsilon_decay_steps"] = 10
    cfg["context"]["gru_hidden"] = 16

    trainer = MetaTrainer(cfg)
    csv_path = trainer.train(meta_iters=1)
    with open(csv_path, newline="") as f:
        row = list(csv.DictReader(f))[0]

    assert row["support_gate_enabled"] == "True"
    assert row["support_gate_paired_validation"] == "True"
    assert row["support_gate_budget_mode"] == "paired_support_validation"
    assert row["support_gate_same_validation_seeds"] == "True"
    assert int(float(row["support_gate_pre_validation_episodes"])) == 1
    assert int(float(row["support_gate_post_validation_episodes"])) == 1
    assert int(float(row["support_gate_validation_seed_pairs"])) == 1
    assert int(float(row["support_gate_extra_rollouts"])) == 1
    assert int(float(row["extra_gradient_updates"])) == 0
    assert int(float(row["extra_query_evaluations"])) == 0
    assert row["query_leakage"] == "False"


def test_support_gate_validation_restores_candidate_context_and_rng(tmp_path):
    cfg = load_cfg("configs/default.yaml")
    cfg["experiment"]["log_dir"] = str(tmp_path)
    cfg["experiment"]["run_name"] = "gate_validation_restore"
    cfg["experiment"]["seed"] = 31
    cfg["env"]["episode_len"] = 3
    cfg["agent"]["hidden_dim"] = 32
    cfg["agent"]["batch_size"] = 4
    cfg["agent"]["warmup_steps"] = 4
    cfg["agent"]["upper_batch_size"] = 2
    cfg["upper_dqn"]["batch_size"] = 2
    cfg["agent"]["upper_update_every"] = 1
    cfg["buffer"]["replay_size"] = 128
    cfg["buffer"]["context_max_len"] = 64
    cfg["meta"]["meta_iters"] = 1
    cfg["meta"]["n_tasks_per_iter"] = 1
    cfg["meta"]["support_gate_validation_episodes"] = 1
    cfg["meta"]["support_gate"]["enabled"] = True
    cfg["meta"]["support_gate"]["paired_validation"] = True
    cfg["upper_dqn"]["epsilon_decay_steps"] = 10
    cfg["context"]["gru_hidden"] = 16

    trainer = MetaTrainer(cfg)
    task = trainer.task_sampler.sample(1)[0]
    pre_state = trainer.agent.snapshot_mutable_state()

    env = MultiTxUwSliptEnv(cfg, overrides=task.to_env_overrides())
    trainer._run_episode(env, train=True, clear_context=True, reset_seed=123)
    candidate_state = trainer.agent.snapshot_mutable_state()
    candidate_dual_state = copy.deepcopy(trainer.dual.state_dict())
    candidate_context_len = len(trainer.agent.episode)

    trainer._run_gate_validation_pair(
        task=task,
        iteration=1,
        task_idx=0,
        candidate_state=candidate_state,
        candidate_dual_state=candidate_dual_state,
        pre_state=pre_state,
        pre_dual_state=copy.deepcopy(candidate_dual_state),
    )
    restored_state = trainer.agent.snapshot_mutable_state()

    assert len(trainer.agent.episode) == candidate_context_len
    assert MetaTrainer._trainable_parameter_delta_norm(candidate_state, restored_state) == 0.0
    assert MetaTrainer._optimizer_state_delta_norm(candidate_state, restored_state) == 0.0
    assert MetaTrainer._target_parameter_delta_norm(candidate_state, restored_state) == 0.0
    assert MetaTrainer._numeric_state_delta_norm(candidate_state["rng"], restored_state["rng"]) == 0.0


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
    assert MetaTrainer._optimizer_state_delta_norm(base, optimizer_only) > 0.0

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
    cfg["agent"]["upper_batch_size"] = 2
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


def test_thermal_cap_margin_schedule_uses_control_steps_and_eval_margin(tmp_path):
    cfg = load_cfg("configs/default.yaml")
    apply_ablation(cfg, "oldbase_cap048_to050_s3200")
    cfg["experiment"]["log_dir"] = str(tmp_path)
    cfg["experiment"]["run_name"] = "margin_schedule"
    cfg["experiment"]["seed"] = 17
    cfg["experiment"]["device"] = "cpu"
    cfg["env"]["episode_len"] = 4
    cfg["agent"]["hidden_dim"] = 32
    cfg["agent"]["batch_size"] = 4
    cfg["agent"]["warmup_steps"] = 4
    cfg["agent"]["upper_warmup_steps"] = 2
    cfg["buffer"]["replay_size"] = 128
    cfg["buffer"]["context_max_len"] = 32
    cfg["meta"]["meta_iters"] = 2
    cfg["meta"]["n_tasks_per_iter"] = 1
    cfg["meta"]["support_episodes"] = 1
    cfg["meta"]["support_adaptation_episodes"] = 1
    cfg["meta"]["query_episodes"] = 0
    cfg["meta"]["checkpoint_selection"]["enabled"] = False
    cfg["safety"]["thermal_cap_margin_schedule"]["warmup_control_steps"] = 5
    cfg["context"]["gru_hidden"] = 16
    cfg["upper_dqn"]["epsilon_decay_steps"] = 10

    trainer = MetaTrainer(cfg)
    csv_path = trainer.train(meta_iters=2)

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert float(rows[0]["thermal_cap_margin_c"]) == 0.48
    assert rows[0]["thermal_cap_margin_schedule_phase"] == "warmup"
    assert rows[0]["thermal_cap_margin_schedule_active"] == "True"
    assert int(float(rows[0]["global_adaptation_control_steps"])) == 4
    assert int(float(rows[0]["warmup_control_steps"])) == 5
    assert float(rows[1]["thermal_cap_margin_c"]) == 0.50
    assert rows[1]["thermal_cap_margin_schedule_phase"] == "nominal"
    assert rows[1]["thermal_cap_margin_schedule_active"] == "False"
    assert int(float(rows[1]["global_adaptation_control_steps"])) == 8

    ev = trainer.evaluate(n_tasks=1, episodes_per_task=1)
    assert float(ev["eval_thermal_cap_margin_c"]) == 0.50


def test_lower_sac_alpha_schedule_uses_control_steps(tmp_path):
    cfg = load_cfg("configs/default.yaml")
    apply_ablation(cfg, "oldbase_alpha010_to008_s24000")
    cfg["experiment"]["log_dir"] = str(tmp_path)
    cfg["experiment"]["run_name"] = "alpha_schedule"
    cfg["experiment"]["seed"] = 18
    cfg["experiment"]["device"] = "cpu"
    cfg["env"]["episode_len"] = 4
    cfg["agent"]["hidden_dim"] = 32
    cfg["agent"]["batch_size"] = 4
    cfg["agent"]["warmup_steps"] = 4
    cfg["agent"]["upper_warmup_steps"] = 2
    cfg["buffer"]["replay_size"] = 128
    cfg["buffer"]["context_max_len"] = 32
    cfg["meta"]["meta_iters"] = 2
    cfg["meta"]["n_tasks_per_iter"] = 1
    cfg["meta"]["support_episodes"] = 1
    cfg["meta"]["support_adaptation_episodes"] = 1
    cfg["meta"]["query_episodes"] = 0
    cfg["meta"]["checkpoint_selection"]["enabled"] = False
    cfg["lower_sac"]["alpha_schedule"]["warmup_control_steps"] = 5
    cfg["context"]["gru_hidden"] = 16
    cfg["upper_dqn"]["epsilon_decay_steps"] = 10

    assert float(cfg["safety"]["thermal_cap_margin_c"]) == 0.50
    assert not cfg["safety"].get("thermal_cap_margin_schedule", {}).get("enabled", False)

    trainer = MetaTrainer(cfg)
    csv_path = trainer.train(meta_iters=2)

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert float(rows[0]["thermal_cap_margin_c"]) == 0.50
    assert rows[0]["thermal_cap_margin_schedule_phase"] == "disabled"
    assert float(rows[0]["lower_sac_alpha"]) == 0.10
    assert rows[0]["alpha_schedule_phase"] == "warmup"
    assert rows[0]["alpha_schedule_active"] == "True"
    assert int(float(rows[0]["global_adaptation_control_steps"])) == 4
    assert int(float(rows[0]["alpha_warmup_control_steps"])) == 5
    assert float(rows[1]["lower_sac_alpha"]) == 0.08
    assert rows[1]["alpha_schedule_phase"] == "nominal"
    assert rows[1]["alpha_schedule_active"] == "False"
    assert int(float(rows[1]["global_adaptation_control_steps"])) == 8


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
    assert summary["diagnostic_contract"]["hybrid_meta_support_gated"]["support_gate_enabled"] is True
    assert summary["diagnostic_contract"]["hybrid_meta_support_gated"]["query_used_by_gate"] is False
    assert summary["diagnostic_contract"]["hybrid_meta_support_gated"]["same_checkpoint_as"] == "hybrid_meta"
    assert summary["diagnostic_contract"]["hybrid_context_only"]["context_enabled"] is True
    assert summary["diagnostic_contract"]["hybrid_context_only"]["explicit_inner_outer"] is False
    assert summary["diagnostic_contract"]["hybrid_context_only"]["support_train_adapts"] is False
    assert summary["diagnostic_contract"]["hybrid_context_only"]["query_train_updates"] is False
    assert summary["diagnostic_contract"]["hybrid_wo_meta"]["support_train_adapts"] is False
    assert "query_reward_delta_meta_minus_wo_meta" in summary["comparison"]
    assert "query_reward_delta_meta_minus_no_support_adapt" in summary["comparison"]
    assert "query_reward_delta_gated_minus_meta" in summary["comparison"]
    assert "query_reward_delta_gated_minus_context_only" in summary["comparison"]
    assert "query_reward_delta_meta_minus_context_only" in summary["comparison"]
    assert "query_reward_delta_context_only_minus_wo_meta" in summary["comparison"]
    assert "few_shot_reward_gain_meta_minus_no_support_adapt" in summary["comparison"]
    assert "few_shot_reward_gain_gated_minus_meta" in summary["comparison"]
    assert "few_shot_reward_gain_meta_minus_context_only" in summary["comparison"]
    assert summary["fixed_task_batch_hash"]
    assert summary["ordered_fixed_task_batch_hash"]
    assert "meta_query_reward_after_minus_before_support" in summary["comparison"]
    assert "few_shot_reward_gain_meta_minus_wo_meta" in summary["comparison"]
    assert "support_reward_mean" in summary["adaptation_summary"]["hybrid_meta"]
    assert "query_reward_before_support" in summary["adaptation_summary"]["hybrid_meta"]
    assert "query_reward_after_support" in summary["adaptation_summary"]["hybrid_meta"]
    assert summary["adaptation_summary"]["hybrid_meta"]["query_has_support_context_fraction"] == 1.0
    assert summary["adaptation_summary"]["hybrid_meta_no_support_adapt"]["query_has_support_context_fraction"] == 0.0
    assert 0.0 <= summary["adaptation_summary"]["hybrid_meta_support_gated"]["query_has_support_context_fraction"] <= 1.0
    assert "support_gate_accept_rate" in summary["adaptation_summary"]["hybrid_meta_support_gated"]
    assert "support_gate_no_support_rate" in summary["adaptation_summary"]["hybrid_meta_support_gated"]
    assert "support_gate_rollback_rate" in summary["adaptation_summary"]["hybrid_meta_support_gated"]
    assert "support_gate_context_only_rate" in summary["adaptation_summary"]["hybrid_meta_support_gated"]
    assert summary["adaptation_summary"]["hybrid_context_only"]["query_has_support_context_fraction"] == 1.0
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
        "hybrid_meta_support_gated",
        "hybrid_context_only",
        "hybrid_wo_meta",
    }
    assert {row["phase"] for row in rows} == {"pre_query", "support", "query"}
    assert all("violation_rate" in row for row in rows)
    pre_query = [row for row in rows if row["phase"] == "pre_query"]
    assert pre_query
    assert pre_query[0]["pre_query_eval_before_support"] == "True"
    for variant in {
        "hybrid_meta",
        "hybrid_meta_no_support_adapt",
        "hybrid_meta_support_gated",
        "hybrid_context_only",
        "hybrid_wo_meta",
    }:
        before = [row for row in rows if row["variant"] == variant and row["phase"] == "pre_query"][0]
        after = [row for row in rows if row["variant"] == variant and row["phase"] == "query"][0]
        assert before["episode_seed"] == after["episode_seed"]
    meta_support = [row for row in rows if row["variant"] == "hybrid_meta" and row["phase"] == "support"]
    no_adapt_support = [
        row for row in rows if row["variant"] == "hybrid_meta_no_support_adapt" and row["phase"] == "support"
    ]
    gated_support = [
        row for row in rows if row["variant"] == "hybrid_meta_support_gated" and row["phase"] == "support"
    ]
    context_only_support = [
        row for row in rows if row["variant"] == "hybrid_context_only" and row["phase"] == "support"
    ]
    wo_support = [row for row in rows if row["variant"] == "hybrid_wo_meta" and row["phase"] == "support"]
    assert meta_support[0]["support_train_adapts"] == "True"
    assert no_adapt_support[0]["support_train_adapts"] == "False"
    assert gated_support
    assert context_only_support[0]["support_train_adapts"] == "False"
    assert wo_support[0]["support_train_adapts"] == "False"
    assert int(meta_support[0]["context_history_len_before_query"]) > 0
    assert int(no_adapt_support[0]["context_history_len_before_query"]) == 0
    assert int(gated_support[0]["context_history_len_before_query"]) >= 0
    assert int(context_only_support[0]["context_history_len_before_query"]) > 0
    assert int(wo_support[0]["context_history_len_before_query"]) == 0
    assert "support_parameter_delta_norm" in rows[0]
    assert "support_gate_enabled" in rows[0]
    assert "support_gate_accepted" in rows[0]
    assert "support_gate_selected" in rows[0]
    assert "support_gate_score_no_context" in rows[0]
    assert "support_gate_score_delta" in rows[0]
    assert "support_upper_delta_norm" in rows[0]
    assert "support_lower_replay_len_after_support" in rows[0]
    assert "support_upper_replay_len_after_support" in rows[0]


def test_support_gated_meta_uses_support_validation_not_query(tmp_path):
    summary = run_meta_adaptation_diagnostics(
        out_dir=tmp_path,
        scenario="moderate_practical",
        seed=17,
        train_iters=0,
        n_tasks=1,
        support_episodes=2,
        query_episodes=1,
        episode_len=3,
        fast_mode=True,
        device="cpu",
        make_plots=False,
    )

    assert summary["diagnostic_contract"]["hybrid_meta_support_gated"]["support_gate_enabled"] is True
    assert summary["diagnostic_contract"]["hybrid_meta_support_gated"]["query_used_by_gate"] is False
    gated = summary["adaptation_summary"]["hybrid_meta_support_gated"]
    assert gated["support_gate_enabled_fraction"] == 1.0
    assert 0.0 <= gated["support_gate_accept_rate"] <= 1.0

    with open(summary["csv_path"], newline="") as f:
        rows = list(csv.DictReader(f))

    gated_rows = [row for row in rows if row["variant"] == "hybrid_meta_support_gated"]
    assert gated_rows
    validation_rows = [row for row in gated_rows if row["support_gate_validation"] == "True"]
    assert validation_rows
    assert all(row["support_train_adapts"] == "False" for row in validation_rows)
    support_rows = [row for row in gated_rows if row["phase"] == "support"]
    assert support_rows
    adaptation_rows = [row for row in support_rows if row["support_gate_validation"] != "True"]
    assert adaptation_rows
    assert all(row["support_train_adapts"] == "True" for row in adaptation_rows)
    assert all(row["support_gate_selected"] in {"adapted", "rollback"} for row in support_rows)
    query_rows = [row for row in gated_rows if row["phase"] == "query"]
    assert query_rows
    assert all(row["query_parameter_delta_norm"] == "0.0" for row in query_rows)
    assert "query_parameter_delta_norm" in rows[0]
    meta_query = [row for row in rows if row["variant"] == "hybrid_meta" and row["phase"] == "query"]
    assert meta_query
    assert float(meta_query[0]["query_parameter_delta_norm"]) == 0.0
