from __future__ import annotations

import copy
import types

import numpy as np
import torch

from tchhmrl.agents.sac_lower import LowerSAC
from tchhmrl.safety.safety_layer import SafetyLayer
from tchhmrl.utils.config import load_cfg


def _small_cfg() -> dict:
    cfg = load_cfg("configs/default.yaml")
    cfg = copy.deepcopy(cfg)
    cfg["agent"]["hidden_dim"] = 64
    cfg["agent"]["batch_size"] = 4
    return cfg


def test_lower_upper_condition_dim_is_consistent():
    cfg = _small_cfg()
    assert int(cfg["agent"]["lower_upper_ctx_dim"]) == 7

    ctx = LowerSAC._upper_ctx_np(11)
    assert ctx.shape == (7,)
    # [boost_onehot(4), mode_onehot(3)]
    assert float(np.sum(ctx[:4])) == 1.0
    assert float(np.sum(ctx[4:])) == 1.0


def test_lower_sac_uses_physical_encoder_when_enabled():
    cfg = _small_cfg()
    safety = SafetyLayer(cfg)
    sac = LowerSAC(cfg, safety, torch.device("cpu"))

    assert sac.physical_enabled is True
    assert sac.physical_dim == 18
    assert sac.physical_embedding_dim == 32
    assert sac.actor_phys is not None
    assert sac.q1_phys is not None

    obs = np.zeros(int(cfg["agent"]["obs_dim"]), dtype=np.float32)
    z = np.zeros(int(cfg["agent"]["z_dim"]), dtype=np.float32)
    physical = np.ones(18, dtype=np.float32)
    raw = sac.select_action(obs, z, upper_idx=11, physical_features=physical, eval_mode=True)

    assert raw.shape == (5,)
    assert sac.obs_aug_dim == int(cfg["agent"]["obs_dim"]) + int(cfg["agent"]["lower_upper_ctx_dim"]) + 32


def _dummy_lower_batch(cfg: dict, batch_size: int = 4) -> dict:
    obs_dim = int(cfg["agent"]["obs_dim"])
    z_dim = int(cfg["agent"]["z_dim"])
    n_tx = int(cfg["env"]["n_tx"])
    physical_dim = int(cfg["physical_context"]["input_dim"])
    return {
        "obs": np.random.randn(batch_size, obs_dim).astype(np.float32),
        "z": np.random.randn(batch_size, z_dim).astype(np.float32),
        "act_exec": np.random.randn(batch_size, 5).astype(np.float32),
        "reward": np.random.randn(batch_size).astype(np.float32),
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
        "gamma_env": np.full(batch_size, cfg["env"]["gamma"], dtype=np.float32),
        "delta_env": np.full(batch_size, cfg["env"]["delta"], dtype=np.float32),
    }


def test_lower_sac_trains_independent_constraint_critics():
    cfg = _small_cfg()
    safety = SafetyLayer(cfg)
    sac = LowerSAC(cfg, safety, torch.device("cpu"))

    assert sac.constraint_critics_enabled is True
    assert sac.constraint_q is not None
    assert sac.constraint_q_tgt is not None
    assert sac.constraint_dim == 4
    assert sac.constraint_actor_weights.shape == (1, 4)

    stats = sac.update(_dummy_lower_batch(cfg))

    assert "constraint_critic_loss" in stats
    assert "constraint_actor_penalty" in stats
    assert np.isfinite(stats["constraint_critic_loss"])
    assert np.isfinite(stats["constraint_actor_penalty"])
    assert stats["constraint_actor_penalty"] >= 0.0


def test_lower_reward_critic_uses_raw_reward_when_constraint_critics_are_enabled():
    cfg = _small_cfg()
    safety = SafetyLayer(cfg)
    sac = LowerSAC(cfg, safety, torch.device("cpu"))
    batch = _dummy_lower_batch(cfg)
    batch["reward"] = np.full(batch["reward"].shape, -10.0, dtype=np.float32)
    batch["reward_raw"] = np.full(batch["reward"].shape, 2.0, dtype=np.float32)

    stats = sac.update(batch)

    assert sac.reward_target_mode == "raw_reward"
    assert stats["reward_target_is_raw"] == 1.0
    assert np.isclose(float(stats["reward_target_mean"]), 2.0)


def test_lower_update_uses_next_macro_for_target_projection():
    cfg = _small_cfg()
    safety = SafetyLayer(cfg)
    calls: list[torch.Tensor] = []
    original = safety.project_torch

    def wrapped(self, lower_raw, boost_combo, mode, temps, amb_temp, gamma=None, delta=None):
        calls.append(boost_combo.detach().cpu().clone())
        return original(lower_raw, boost_combo, mode, temps, amb_temp, gamma, delta)

    safety.project_torch = types.MethodType(wrapped, safety)
    sac = LowerSAC(cfg, safety, torch.device("cpu"))

    b = 4
    obs_dim = int(cfg["agent"]["obs_dim"])
    z_dim = int(cfg["agent"]["z_dim"])
    n_tx = int(cfg["env"]["n_tx"])

    batch = {
        "obs": np.random.randn(b, obs_dim).astype(np.float32),
        "z": np.random.randn(b, z_dim).astype(np.float32),
        "act_exec": np.random.randn(b, 5).astype(np.float32),
        "reward": np.random.randn(b).astype(np.float32),
        "next_obs": np.random.randn(b, obs_dim).astype(np.float32),
        "z_next": np.random.randn(b, z_dim).astype(np.float32),
        "done": np.zeros(b, dtype=np.float32),
        "boost_combo_exec": np.zeros(b, dtype=np.float32),
        "mode_exec": np.zeros(b, dtype=np.float32),
        "boost_combo_exec_next": np.full(b, 3, dtype=np.float32),
        "mode_exec_next": np.full(b, 2, dtype=np.float32),
        "temps": np.full((b, n_tx), 25.0, dtype=np.float32),
        "next_temps": np.full((b, n_tx), 25.0, dtype=np.float32),
        "amb_temp": np.full(b, 25.0, dtype=np.float32),
        "gamma_env": np.full(b, cfg["env"]["gamma"], dtype=np.float32),
        "delta_env": np.full(b, cfg["env"]["delta"], dtype=np.float32),
    }

    sac.update(batch)

    assert len(calls) >= 2
    # First call is target projection, which should use next macro condition.
    assert torch.all(calls[0] == 3)
    # Second call is policy projection on current macro condition.
    assert torch.all(calls[1] == 0)
