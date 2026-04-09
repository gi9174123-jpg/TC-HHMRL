from __future__ import annotations

import numpy as np

from tchhmrl.envs.uw_slipt_env import MultiTxUwSliptEnv
from tchhmrl.utils.config import load_cfg


def test_env_reset_and_step_shape():
    cfg = load_cfg("configs/default.yaml")
    env = MultiTxUwSliptEnv(cfg)

    obs, info = env.reset(seed=7)
    assert isinstance(info, dict)
    assert obs.shape == (cfg["agent"]["obs_dim"],)

    action = {
        "upper_idx": 3,
        "boost_combo_exec": 1,
        "mode_exec": 0,
        "currents_exec": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "rho_exec": np.array([0.4], dtype=np.float32),
        "tau_exec": np.array([0.7], dtype=np.float32),
    }

    next_obs, reward, terminated, truncated, step_info = env.step(action)
    assert next_obs.shape == (cfg["agent"]["obs_dim"],)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "cost" in step_info
    assert "signal_led" in step_info
    assert "signal_ld" in step_info
    assert "signal_ld_share" in step_info
    assert "led_tx_fraction" in step_info
    assert "cost_vec" in step_info
    assert "qos_rate" in step_info


def test_env_action_smooth_penalty_increases_on_large_action_jump():
    cfg = load_cfg("configs/default.yaml")
    cfg["env"]["action_smooth_weight"] = 0.1
    env = MultiTxUwSliptEnv(cfg)
    env.reset(seed=11)

    action_a = {
        "upper_idx": 0,
        "boost_combo_exec": 0,
        "mode_exec": 0,
        "currents_exec": np.array([0.2, 0.0, 0.0], dtype=np.float32),
        "rho_exec": np.array([0.5], dtype=np.float32),
        "tau_exec": np.array([0.5], dtype=np.float32),
    }
    _, _, _, _, info_a = env.step(action_a)

    action_b = {
        "upper_idx": 11,
        "boost_combo_exec": 3,
        "mode_exec": 2,
        "currents_exec": np.array([2.8, 2.8, 2.8], dtype=np.float32),
        "rho_exec": np.array([1.0], dtype=np.float32),
        "tau_exec": np.array([0.0], dtype=np.float32),
    }
    _, _, _, _, info_b = env.step(action_b)

    assert float(info_a.get("penalty_smooth_term", 0.0)) >= 0.0
    assert float(info_b.get("penalty_smooth_term", 0.0)) > float(info_a.get("penalty_smooth_term", 0.0))


def test_env_observation_hold_tracks_boost_combo_not_mode():
    cfg = load_cfg("configs/default.yaml")
    env = MultiTxUwSliptEnv(cfg)
    env.reset(seed=3)

    action_a = {
        "upper_idx": 0,
        "upper_idx_exec": 0,
        "boost_combo_exec": 0,
        "mode_exec": 0,
        "currents_exec": np.array([0.3, 0.0, 0.0], dtype=np.float32),
        "rho_exec": np.array([0.4], dtype=np.float32),
        "tau_exec": np.array([1.0], dtype=np.float32),
    }
    env.step(action_a)
    hold_after_a = env.boost_hold_steps

    action_b = dict(action_a)
    action_b["mode_exec"] = 2
    action_b["upper_idx_exec"] = 2
    env.step(action_b)

    assert hold_after_a == 1
    assert env.boost_hold_steps == 2
