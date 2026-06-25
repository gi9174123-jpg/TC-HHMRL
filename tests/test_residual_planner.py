from __future__ import annotations

import copy
import random

import numpy as np
import torch

from tchhmrl.agents.hierarchical_agent import HierarchicalAgent
from tchhmrl.planning.residual_basis import residual_basis
from tchhmrl.utils.config import load_cfg


def _assert_same_nested(a, b):
    if torch.is_tensor(a):
        if torch.is_floating_point(a):
            assert torch.allclose(a.cpu(), b.cpu(), equal_nan=True)
        else:
            assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b, equal_nan=True)
    elif isinstance(a, dict):
        assert set(a.keys()) == set(b.keys())
        for key in a:
            _assert_same_nested(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for av, bv in zip(a, b):
            _assert_same_nested(av, bv)
    else:
        assert a == b


def _cfg() -> dict:
    cfg = copy.deepcopy(load_cfg("configs/default.yaml"))
    cfg["agent"]["hidden_dim"] = 64
    cfg["agent"]["batch_size"] = 4
    cfg["agent"]["upper_batch_size"] = 4
    cfg["residual_planner"]["start_meta_iter"] = 0
    cfg["residual_planner"]["thermal_horizon_start_meta_iter"] = 0
    cfg["residual_planner"]["candidate_count"] = 24
    return cfg


def test_residual_basis_returns_policy_centered_structured_candidates():
    basis = residual_basis(
        candidate_count=24,
        current_step=0.05,
        ratio_step=0.05,
        thermal_headroom=np.asarray([10.0, 1.0, 5.0], dtype=np.float32),
    )

    assert basis.shape == (24, 5)
    assert np.allclose(basis[0], np.zeros(5, dtype=np.float32))
    assert np.any(np.abs(basis[:, :3]).sum(axis=1) > 0.0)
    assert np.any(np.abs(basis[:, 3:]).sum(axis=1) > 0.0)


def test_agent_act_uses_residual_planner_after_start_iter():
    cfg = _cfg()
    agent = HierarchicalAgent(cfg, torch.device("cpu"))
    agent.set_meta_iter(1)

    obs = np.zeros(int(cfg["agent"]["obs_dim"]), dtype=np.float32)
    z = np.zeros(int(cfg["agent"]["z_dim"]), dtype=np.float32)
    temps = np.asarray([28.0, 28.0, 28.0], dtype=np.float32)

    action, aux = agent.act(
        obs=obs,
        temps=temps,
        amb_temp=float(cfg["env"]["amb_temp"]),
        gamma=float(cfg["env"]["gamma"]),
        delta=float(cfg["env"]["delta"]),
        z=z,
        eval_mode=True,
    )

    assert aux["residual_planner_enabled"] is True
    assert int(aux["residual_planner_candidate_count"]) == 24
    assert int(aux["residual_planner_selected_idx"]) >= 0
    assert int(aux["residual_planner_effective_thermal_horizon"]) == 2
    assert float(aux["residual_planner_latency_ms"]) >= 0.0
    assert "residual_planner_thermal_risk" in aux
    assert "residual_planner_score_improvement" in aux
    assert np.asarray(aux["act_policy_raw"], dtype=np.float32).shape == (5,)
    assert np.asarray(aux["policy_action_raw"], dtype=np.float32).shape == (5,)
    assert np.asarray(aux["planner_action_raw"], dtype=np.float32).shape == (5,)
    assert np.asarray(aux["executed_action"], dtype=np.float32).shape == (5,)
    assert np.allclose(np.asarray(aux["act_refined_raw"], dtype=np.float32), np.asarray(aux["planner_action_raw"], dtype=np.float32))
    assert np.allclose(np.asarray(aux["act_exec"], dtype=np.float32), np.asarray(aux["executed_action"], dtype=np.float32))
    assert isinstance(aux["planner_selected"], bool)
    assert np.asarray(action["currents_exec"], dtype=np.float32).shape == (3,)


def test_residual_planner_scores_with_target_critics_not_online_critics():
    cfg = _cfg()
    agent = HierarchicalAgent(cfg, torch.device("cpu"))
    agent.set_meta_iter(1)

    class RaisingCritic(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError("online critic should not be used by residual planner scoring")

    agent.lower.q1 = RaisingCritic()
    agent.lower.q2 = RaisingCritic()

    obs = np.zeros(int(cfg["agent"]["obs_dim"]), dtype=np.float32)
    z = np.zeros(int(cfg["agent"]["z_dim"]), dtype=np.float32)
    temps = np.asarray([28.0, 28.0, 28.0], dtype=np.float32)

    _, aux = agent.act(
        obs=obs,
        temps=temps,
        amb_temp=float(cfg["env"]["amb_temp"]),
        gamma=float(cfg["env"]["gamma"]),
        delta=float(cfg["env"]["delta"]),
        z=z,
        eval_mode=True,
    )

    assert aux["residual_planner_enabled"] is True
    assert "residual_planner_best_score_improvement" in aux


def test_residual_planner_trust_region_rejects_non_policy_candidates():
    cfg = _cfg()
    cfg["residual_planner"]["trust_region_enabled"] = True
    cfg["residual_planner"]["trust_region_raw_l2"] = 0.0
    cfg["residual_planner"]["trust_region_exec_l2"] = 0.0
    agent = HierarchicalAgent(cfg, torch.device("cpu"))
    agent.set_meta_iter(1)

    obs = np.zeros(int(cfg["agent"]["obs_dim"]), dtype=np.float32)
    z = np.zeros(int(cfg["agent"]["z_dim"]), dtype=np.float32)
    temps = np.asarray([28.0, 28.0, 28.0], dtype=np.float32)

    _, aux = agent.act(
        obs=obs,
        temps=temps,
        amb_temp=float(cfg["env"]["amb_temp"]),
        gamma=float(cfg["env"]["gamma"]),
        delta=float(cfg["env"]["delta"]),
        z=z,
        eval_mode=True,
    )

    assert int(aux["residual_planner_selected_idx"]) == 0
    assert int(aux["residual_planner_trust_region_rejected_count"]) > 0
    assert int(aux["residual_planner_valid_candidate_count"]) == 1


def test_residual_planner_positive_margin_can_force_policy_fallback():
    cfg = _cfg()
    cfg["residual_planner"]["replacement_margin_mode"] = "absolute"
    cfg["residual_planner"]["replacement_margin"] = 1.0e6
    agent = HierarchicalAgent(cfg, torch.device("cpu"))
    agent.set_meta_iter(1)

    obs = np.zeros(int(cfg["agent"]["obs_dim"]), dtype=np.float32)
    z = np.zeros(int(cfg["agent"]["z_dim"]), dtype=np.float32)
    temps = np.asarray([28.0, 28.0, 28.0], dtype=np.float32)

    _, aux = agent.act(
        obs=obs,
        temps=temps,
        amb_temp=float(cfg["env"]["amb_temp"]),
        gamma=float(cfg["env"]["gamma"]),
        delta=float(cfg["env"]["delta"]),
        z=z,
        eval_mode=True,
    )

    assert int(aux["residual_planner_selected_idx"]) == 0
    assert float(aux["residual_planner_replacement_margin"]) > 0.0
    assert float(aux["residual_planner_fallback_rate"]) == 1.0


def test_residual_planner_h2_veto_is_reported_near_thermal_boundary():
    cfg = _cfg()
    cfg["safety"]["thermal_safe"] = 29.0
    cfg["residual_planner"]["h2_veto_enabled"] = True
    agent = HierarchicalAgent(cfg, torch.device("cpu"))
    agent.set_meta_iter(1)

    obs = np.zeros(int(cfg["agent"]["obs_dim"]), dtype=np.float32)
    z = np.zeros(int(cfg["agent"]["z_dim"]), dtype=np.float32)
    temps = np.asarray([28.8, 28.8, 28.8], dtype=np.float32)

    _, aux = agent.act(
        obs=obs,
        temps=temps,
        amb_temp=28.0,
        gamma=float(cfg["env"]["gamma"]),
        delta=float(cfg["env"]["delta"]),
        z=z,
        eval_mode=True,
    )

    assert "residual_planner_h2_veto_rate" in aux
    assert float(aux["residual_planner_h2_max_temperature"]) >= 0.0
    assert int(aux["residual_planner_selected_idx"]) >= 0


def test_residual_planner_preview_has_no_mutable_side_effects():
    cfg = _cfg()
    agent = HierarchicalAgent(cfg, torch.device("cpu"))
    agent.set_meta_iter(1)
    obs = np.zeros(int(cfg["agent"]["obs_dim"]), dtype=np.float32)
    z = np.zeros(int(cfg["agent"]["z_dim"]), dtype=np.float32)
    temps = np.asarray([28.0, 28.0, 28.0], dtype=np.float32)
    physical = agent.current_physical_features(temps=temps)
    policy_raw = np.zeros(5, dtype=np.float32)
    before = copy.deepcopy(agent.snapshot_mutable_state())

    _candidate, _diag = agent.residual_planner.plan(
        lower=agent.lower,
        safety=agent.safety,
        obs=obs,
        z=z,
        upper_idx_exec=11,
        boost_combo=3,
        mode=2,
        policy_raw=policy_raw,
        physical_features=physical,
        thermal_headroom=(float(agent.safety.thermal_safe) - temps),
        temps=temps,
        amb_temp=float(cfg["env"]["amb_temp"]),
        meta_iter=1,
    )

    after = agent.snapshot_mutable_state()
    _assert_same_nested(before, after)


def test_agent_checkpoint_preserves_planner_stage_and_safety_state(tmp_path):
    cfg = _cfg()
    agent = HierarchicalAgent(cfg, torch.device("cpu"))
    agent.set_meta_iter(73)
    agent.safety.thermal_estimator.gain_mean[:] = np.asarray([1.1, 1.2, 1.3], dtype=np.float32)
    path = tmp_path / "agent.pt"
    agent.save(path)

    restored = HierarchicalAgent(cfg, torch.device("cpu"))
    restored.load(path)

    assert restored.current_meta_iter == 73
    assert restored.residual_planner.active(meta_iter=restored.current_meta_iter) is True
    assert np.allclose(restored.safety.thermal_estimator.gain_mean, np.asarray([1.1, 1.2, 1.3], dtype=np.float32))


def test_mutable_snapshot_restores_global_rng_states():
    cfg = _cfg()
    agent = HierarchicalAgent(cfg, torch.device("cpu"))
    random.seed(11)
    np.random.seed(22)
    torch.manual_seed(33)
    snapshot = agent.snapshot_mutable_state()

    expected_py = random.random()
    expected_np = float(np.random.rand())
    expected_torch = float(torch.rand(1).item())

    random.seed(101)
    np.random.seed(202)
    torch.manual_seed(303)
    agent.restore_mutable_state(snapshot)

    assert random.random() == expected_py
    assert float(np.random.rand()) == expected_np
    assert float(torch.rand(1).item()) == expected_torch
