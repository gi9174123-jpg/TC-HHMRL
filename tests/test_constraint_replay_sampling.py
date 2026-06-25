from __future__ import annotations

import copy

import numpy as np
import torch
import pytest

from tchhmrl.agents.hierarchical_agent import HierarchicalAgent
from tchhmrl.agents.sac_lower import LowerSAC
from tchhmrl.buffers.replay_buffer import ReplayBuffer
from tchhmrl.safety.safety_layer import SafetyLayer
from tchhmrl.utils.config import load_cfg


def _small_cfg() -> dict:
    cfg = copy.deepcopy(load_cfg("configs/default.yaml"))
    cfg["agent"]["hidden_dim"] = 64
    cfg["agent"]["batch_size"] = 4
    return cfg


def _transition(cfg: dict, *, cost: float = 0.0, residual: float = 0.0, headroom: float = 10.0) -> dict:
    obs_dim = int(cfg["agent"]["obs_dim"])
    z_dim = int(cfg["agent"]["z_dim"])
    n_tx = int(cfg["env"]["n_tx"])
    physical_dim = int(cfg["physical_context"]["input_dim"])
    return {
        "obs": np.random.randn(obs_dim).astype(np.float32),
        "z": np.random.randn(z_dim).astype(np.float32),
        "act_exec": np.random.randn(5).astype(np.float32),
        "reward": np.float32(np.random.randn()),
        "reward_raw": np.float32(np.random.randn()),
        "reward_task": np.float32(np.random.randn()),
        "reward_benchmark": np.float32(np.random.randn()),
        "reward_dual_penalized": np.float32(np.random.randn()),
        "next_obs": np.random.randn(obs_dim).astype(np.float32),
        "z_next": np.random.randn(z_dim).astype(np.float32),
        "done": np.float32(0.0),
        "cost": np.float32(cost),
        "cost_vec": np.asarray([cost, 0.0, 0.0, 0.0], dtype=np.float32),
        "boost_combo_exec": np.float32(3.0),
        "mode_exec": np.float32(2.0),
        "boost_combo_exec_next": np.float32(3.0),
        "mode_exec_next": np.float32(2.0),
        "physical_features": np.random.randn(physical_dim).astype(np.float32),
        "physical_features_next": np.random.randn(physical_dim).astype(np.float32),
        "temps": np.full(n_tx, 30.0, dtype=np.float32),
        "next_temps": np.full(n_tx, 30.0, dtype=np.float32),
        "amb_temp": np.float32(29.0),
        "thermal_headroom": np.full(n_tx, headroom, dtype=np.float32),
        "projection_residual": np.full(5, residual, dtype=np.float32),
    }


def test_replay_buffer_stratified_constraint_sampling_tracks_buckets():
    cfg = _small_cfg()
    replay = ReplayBuffer(capacity=64)
    for _ in range(20):
        replay.add(_transition(cfg, cost=0.0, residual=0.0, headroom=8.0))
    for _ in range(10):
        replay.add(_transition(cfg, cost=0.0, residual=1.0, headroom=0.5))
    for _ in range(10):
        replay.add(_transition(cfg, cost=0.02, residual=0.0, headroom=8.0))

    batch, stats = replay.sample_stratified_constraint(
        12,
        uniform_fraction=0.50,
        boundary_fraction=0.30,
        violation_fraction=0.20,
        thresholds={
            "thermal_headroom_threshold": 1.0,
            "projection_residual_threshold": 0.5,
            "constraint_cost_threshold": 1.0e-8,
        },
        importance_weighting=True,
    )

    assert batch["obs"].shape[0] == 12
    assert batch["constraint_replay_importance_weight"].shape == (12,)
    assert batch["constraint_replay_bucket_id"].shape == (12,)
    assert not np.allclose(batch["constraint_replay_importance_weight"], 1.0)
    assert stats["constraint_batch_violation_count"] > 0
    assert stats["constraint_batch_boundary_count"] > 0
    assert stats["constraint_bucket_boundary_pool_count"] > 0
    assert stats["constraint_bucket_violation_pool_count"] > 0
    assert stats["constraint_replay_importance_weight_max"] > stats["constraint_replay_importance_weight_min"]
    assert (
        stats["constraint_batch_uniform_count"]
        + stats["constraint_batch_boundary_count"]
        + stats["constraint_batch_violation_count"]
        == 12
    )
    assert stats["constraint_bucket_total_count"] == 40
    assert np.isclose(stats["constraint_replay_uniform_pool_fraction"], 20.0 / 40.0)
    assert np.isclose(stats["constraint_replay_boundary_pool_fraction"], 10.0 / 40.0)
    assert np.isclose(stats["constraint_replay_violation_pool_fraction"], 10.0 / 40.0)
    assert np.isclose(stats["constraint_replay_qos_only_violation_fraction"], 10.0 / 40.0)
    assert np.isclose(stats["constraint_replay_thermal_violation_fraction"], 0.0)
    assert np.isclose(stats["constraint_replay_burst_only_fraction"], 0.0)


def test_constraint_replay_reports_violation_source_fractions():
    cfg = _small_cfg()
    replay = ReplayBuffer(capacity=16)
    replay.add(_transition(cfg, cost=0.0, residual=0.0, headroom=8.0))

    qos = _transition(cfg, cost=0.02, residual=0.0, headroom=8.0)
    replay.add(qos)

    thermal = _transition(cfg, cost=0.02, residual=0.0, headroom=8.0)
    thermal["cost_vec"] = np.asarray([0.0, 0.02, 0.0, 0.0], dtype=np.float32)
    replay.add(thermal)

    burst = _transition(cfg, cost=0.0, residual=0.0, headroom=8.0)
    burst["burst_event"] = np.float32(1.0)
    replay.add(burst)

    _batch, stats = replay.sample_stratified_constraint(
        4,
        uniform_fraction=0.25,
        boundary_fraction=0.25,
        violation_fraction=0.50,
        thresholds={
            "thermal_headroom_threshold": 1.0,
            "projection_residual_threshold": 0.5,
            "constraint_cost_threshold": 1.0e-8,
        },
        importance_weighting=False,
    )

    assert np.isclose(stats["constraint_replay_qos_only_violation_fraction"], 1.0 / 4.0)
    assert np.isclose(stats["constraint_replay_thermal_violation_fraction"], 1.0 / 4.0)
    assert np.isclose(stats["constraint_replay_burst_only_fraction"], 1.0 / 4.0)


def test_replay_buffer_stratified_constraint_shortage_falls_back_to_uniform():
    cfg = _small_cfg()
    replay = ReplayBuffer(capacity=16)
    for _ in range(5):
        replay.add(_transition(cfg, cost=0.0, residual=0.0, headroom=8.0))

    batch, stats = replay.sample_stratified_constraint(
        8,
        uniform_fraction=0.20,
        boundary_fraction=0.40,
        violation_fraction=0.40,
        thresholds={
            "thermal_headroom_threshold": 1.0,
            "projection_residual_threshold": 0.5,
            "constraint_cost_threshold": 1.0e-8,
        },
        importance_weighting=True,
    )

    assert batch["obs"].shape[0] == 8
    assert batch["constraint_replay_bucket_id"].shape == (8,)
    assert stats["constraint_batch_uniform_count"] == 8
    assert stats["constraint_bucket_total_count"] == 5


def test_bus_current_fallback_is_normalized_for_boundary_detection():
    cfg = _small_cfg()
    tr = _transition(cfg, cost=0.0, residual=0.0, headroom=8.0)
    tr["projected_current_total"] = np.float32(5.6)

    violation, boundary = ReplayBuffer._constraint_flags(
        tr,
        {
            "bus_utilization_threshold": 0.85,
            "bus_current_max": 6.4,
            "constraint_cost_threshold": 1.0e-8,
        },
    )

    assert violation is False
    assert boundary is True


def test_lower_sac_accepts_separate_stratified_constraint_batch():
    cfg = _small_cfg()
    safety = SafetyLayer(cfg)
    sac = LowerSAC(cfg, safety, torch.device("cpu"))
    replay = ReplayBuffer(capacity=32)
    for _ in range(16):
        replay.add(_transition(cfg, cost=0.0, residual=0.0, headroom=8.0))
    for _ in range(16):
        replay.add(_transition(cfg, cost=0.02, residual=1.0, headroom=0.5))

    reward_batch = replay.sample(4)
    constraint_batch, constraint_stats = replay.sample_stratified_constraint(
        4,
        uniform_fraction=0.50,
        boundary_fraction=0.25,
        violation_fraction=0.25,
        thresholds={
            "thermal_headroom_threshold": 1.0,
            "projection_residual_threshold": 0.5,
            "constraint_cost_threshold": 1.0e-8,
        },
        importance_weighting=True,
    )
    stats = sac.update(reward_batch, constraint_batch=constraint_batch)
    stats.update(constraint_stats)

    assert stats["constraint_replay_batch_size"] == 4
    assert stats["constraint_replay_weight_mean"] == 1.0
    assert stats["constraint_batch_violation_count"] >= 1
    assert "constraint_critic_loss_uniform" in stats
    assert "constraint_critic_loss_boundary" in stats
    assert "constraint_critic_loss_violation" in stats
    for j in range(int(cfg["constraint_critics"]["out_dim"])):
        assert f"constraint_target_mean_{j}" in stats
        assert f"constraint_target_std_{j}" in stats
        assert f"constraint_mae_{j}" in stats
        assert f"constraint_max_{j}" in stats
        assert f"constraint_positive_fraction_{j}" in stats
    assert np.isfinite(stats["constraint_critic_loss"])


def test_constraint_replay_requires_at_least_one_boundary_threshold():
    cfg = _small_cfg()
    for key in [
        "thermal_headroom_threshold",
        "qos_margin_threshold",
        "bus_utilization_threshold",
        "projection_residual_threshold",
        "temperature_slope_threshold",
    ]:
        cfg["constraint_replay"][key] = None

    with pytest.raises(ValueError, match="boundary threshold"):
        HierarchicalAgent(cfg, torch.device("cpu"))
