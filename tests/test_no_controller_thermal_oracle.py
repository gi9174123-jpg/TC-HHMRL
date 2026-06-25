from __future__ import annotations

import copy

import numpy as np
import torch

from tchhmrl.agents.hierarchical_agent import HierarchicalAgent
from tchhmrl.agents.sac_lower import LowerSAC
from tchhmrl.safety.safety_layer import SafetyLayer
from tchhmrl.utils.config import load_cfg


def _cfg() -> dict:
    cfg = copy.deepcopy(load_cfg("configs/default.yaml"))
    cfg["agent"]["hidden_dim"] = 64
    cfg["agent"]["batch_size"] = 4
    cfg["agent"]["upper_batch_size"] = 4
    cfg["residual_planner"]["start_meta_iter"] = 0
    cfg["residual_planner"]["thermal_horizon_start_meta_iter"] = 0
    cfg["residual_planner"]["candidate_count"] = 8
    return cfg


def test_safety_projection_ignores_task_level_gamma_delta_inputs():
    cfg = _cfg()
    safety = SafetyLayer(cfg)
    kwargs = {
        "upper_raw": 11,
        "lower_raw": np.asarray([0.4, 0.2, 0.1, 0.0, 0.0], dtype=np.float32),
        "temps": np.asarray([42.0, 41.0, 40.0], dtype=np.float32),
        "amb_temp": 32.0,
        "mem": {"current_boost": 3, "dwell_count": 3},
    }

    out_a, _ = safety.project_np(**kwargs, gamma=0.01, delta=1.0)
    out_b, _ = safety.project_np(**kwargs, gamma=0.20, delta=20.0)

    assert np.allclose(out_a["currents_exec"], out_b["currents_exec"])
    assert np.allclose(out_a["t_pred"], out_b["t_pred"])
    assert np.isclose(out_a["gamma_nominal"], float(cfg["safety"]["gamma_nominal"]))
    assert out_a["thermal_parameter_source"] == "nominal_plus_online_effective_gain"


def test_lower_sac_update_does_not_require_gamma_delta_env_fields():
    cfg = _cfg()
    safety = SafetyLayer(cfg)
    sac = LowerSAC(cfg, safety, torch.device("cpu"))
    batch_size = 4
    obs_dim = int(cfg["agent"]["obs_dim"])
    z_dim = int(cfg["agent"]["z_dim"])
    n_tx = int(cfg["env"]["n_tx"])
    physical_dim = int(cfg["physical_context"]["input_dim"])
    batch = {
        "obs": np.random.randn(batch_size, obs_dim).astype(np.float32),
        "z": np.random.randn(batch_size, z_dim).astype(np.float32),
        "act_exec": np.random.randn(batch_size, 5).astype(np.float32),
        "reward": np.random.randn(batch_size).astype(np.float32),
        "reward_raw": np.random.randn(batch_size).astype(np.float32),
        "reward_task": np.random.randn(batch_size).astype(np.float32),
        "reward_benchmark": np.random.randn(batch_size).astype(np.float32),
        "reward_dual_penalized": np.random.randn(batch_size).astype(np.float32),
        "next_obs": np.random.randn(batch_size, obs_dim).astype(np.float32),
        "z_next": np.random.randn(batch_size, z_dim).astype(np.float32),
        "done": np.zeros(batch_size, dtype=np.float32),
        "cost_vec": np.abs(np.random.randn(batch_size, 4).astype(np.float32)) * 0.01,
        "boost_combo_exec": np.zeros(batch_size, dtype=np.float32),
        "mode_exec": np.zeros(batch_size, dtype=np.float32),
        "boost_combo_exec_next": np.full(batch_size, 3, dtype=np.float32),
        "mode_exec_next": np.full(batch_size, 2, dtype=np.float32),
        "physical_features": np.random.randn(batch_size, physical_dim).astype(np.float32),
        "physical_features_next": np.random.randn(batch_size, physical_dim).astype(np.float32),
        "temps": np.full((batch_size, n_tx), 25.0, dtype=np.float32),
        "next_temps": np.full((batch_size, n_tx), 25.0, dtype=np.float32),
        "amb_temp": np.full(batch_size, 25.0, dtype=np.float32),
    }

    stats = sac.update(batch)

    assert stats["reward_target_is_task"] == 1.0
    assert np.isfinite(stats["critic_loss"])


def test_agent_action_ignores_task_level_gamma_delta_inputs():
    cfg = _cfg()
    agent = HierarchicalAgent(cfg, torch.device("cpu"))
    agent.set_meta_iter(1)
    obs = np.zeros(int(cfg["agent"]["obs_dim"]), dtype=np.float32)
    z = np.zeros(int(cfg["agent"]["z_dim"]), dtype=np.float32)
    temps = np.asarray([42.0, 41.0, 40.0], dtype=np.float32)

    torch.manual_seed(123)
    np.random.seed(123)
    action_a, aux_a = agent.act(
        obs=obs,
        temps=temps,
        amb_temp=32.0,
        gamma=0.01,
        delta=1.0,
        z=z,
        eval_mode=True,
    )
    agent.reset_rollout_state(clear_context=True)
    agent.set_meta_iter(1)
    torch.manual_seed(123)
    np.random.seed(123)
    action_b, aux_b = agent.act(
        obs=obs,
        temps=temps,
        amb_temp=32.0,
        gamma=0.20,
        delta=20.0,
        z=z,
        eval_mode=True,
    )

    assert np.allclose(action_a["currents_exec"], action_b["currents_exec"])
    assert np.allclose(aux_a["act_exec"], aux_b["act_exec"])
    assert np.allclose(aux_a["t_pred"], aux_b["t_pred"])
