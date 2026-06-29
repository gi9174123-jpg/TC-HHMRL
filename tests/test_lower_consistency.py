from __future__ import annotations

import copy
import types

import numpy as np
import pytest
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


def test_lower_entropy_mask_matches_boost_and_mode_contract():
    cfg = _small_cfg()
    safety = SafetyLayer(cfg)
    sac = LowerSAC(cfg, safety, torch.device("cpu"))
    boost = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    mode = torch.tensor([0, 1, 2, 2], dtype=torch.long)

    mask = sac._entropy_mask(boost, mode).cpu().numpy()

    assert np.allclose(mask[0], [1, 0, 0, 1, 0])  # anchor-only PS
    assert np.allclose(mask[1], [1, 1, 0, 0, 1])  # anchor+LD1 TS
    assert np.allclose(mask[2], [1, 0, 1, 1, 1])  # anchor+LD2 HY
    assert np.allclose(mask[3], [1, 1, 1, 1, 1])  # full HY


def test_lower_auto_alpha_target_entropy_uses_effective_action_dimensions():
    cfg = _small_cfg()
    cfg["lower_sac"]["auto_alpha"] = True
    cfg["lower_sac"]["target_entropy"] = -5.0
    safety = SafetyLayer(cfg)
    sac = LowerSAC(cfg, safety, torch.device("cpu"))
    batch = _dummy_lower_batch(cfg)
    # Anchor-only PS leaves only total-current and rho latent dimensions active.
    batch["boost_combo_exec"] = np.zeros_like(batch["boost_combo_exec"], dtype=np.float32)
    batch["mode_exec"] = np.zeros_like(batch["mode_exec"], dtype=np.float32)
    batch["boost_combo_exec_next"] = np.zeros_like(batch["boost_combo_exec_next"], dtype=np.float32)
    batch["mode_exec_next"] = np.zeros_like(batch["mode_exec_next"], dtype=np.float32)

    stats = sac.update(batch)

    assert np.isclose(float(stats["entropy_mask_active_dim_mean"]), 2.0)
    assert np.isclose(float(stats["target_entropy_effective_mean"]), -2.0)


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
    assert sac.constraint_phys is not None
    assert sac.constraint_tgt_phys is not None
    assert sac.constraint_phys is not sac.q1_phys
    assert sac.constraint_tgt_phys is not sac.q1_tgt_phys
    assert sac.constraint_optim is not None
    assert sac.constraint_dim == 4
    assert sac.constraint_actor_weights.shape == (1, 4)
    critic_param_ids = {id(p) for group in sac.critic_optim.param_groups for p in group["params"]}
    constraint_param_ids = {id(p) for group in sac.constraint_optim.param_groups for p in group["params"]}
    assert critic_param_ids.isdisjoint(constraint_param_ids)
    assert {id(p) for p in sac.constraint_q.parameters()}.issubset(constraint_param_ids)
    assert {id(p) for p in sac.constraint_phys.parameters()}.issubset(constraint_param_ids)

    stats = sac.update(_dummy_lower_batch(cfg))

    assert "constraint_critic_loss" in stats
    assert "constraint_actor_penalty" in stats
    assert np.isfinite(stats["constraint_critic_loss"])
    assert np.isfinite(stats["constraint_actor_penalty"])
    assert stats["constraint_actor_penalty"] >= 0.0


def test_lower_sac_state_dict_restores_constraint_physical_encoder_and_optimizer():
    cfg = _small_cfg()
    safety = SafetyLayer(cfg)
    sac = LowerSAC(cfg, safety, torch.device("cpu"))
    batch = _dummy_lower_batch(cfg)
    sac.update(batch, constraint_batch=batch)
    state = copy.deepcopy(sac.state_dict())

    assert state["constraint_phys"] is not None
    assert state["constraint_tgt_phys"] is not None
    assert state["constraint_optim"] is not None

    with torch.no_grad():
        for p in sac.constraint_phys.parameters():
            p.add_(1.0)
    sac.load_state_dict(state)
    restored = sac.state_dict()
    for key, value in state["constraint_phys"].items():
        assert torch.allclose(restored["constraint_phys"][key], value)


def test_lower_reward_critic_uses_task_reward_when_constraint_critics_are_enabled():
    cfg = _small_cfg()
    safety = SafetyLayer(cfg)
    sac = LowerSAC(cfg, safety, torch.device("cpu"))
    batch = _dummy_lower_batch(cfg)
    batch["reward"] = np.full(batch["reward"].shape, -10.0, dtype=np.float32)
    batch["reward_raw"] = np.full(batch["reward"].shape, 2.0, dtype=np.float32)
    batch["reward_task"] = np.full(batch["reward"].shape, 3.0, dtype=np.float32)

    stats = sac.update(batch)

    assert sac.reward_target_mode == "reward_task"
    assert stats["reward_target_is_task"] == 1.0
    assert np.isclose(float(stats["reward_target_mean"]), 3.0)


def test_lower_sac_strict_schema_rejects_missing_raw_reward_and_cost_vec():
    cfg = _small_cfg()
    safety = SafetyLayer(cfg)
    sac = LowerSAC(cfg, safety, torch.device("cpu"))
    batch = _dummy_lower_batch(cfg)

    missing_task = dict(batch)
    missing_task.pop("reward_task")
    with pytest.raises(KeyError, match="reward_task"):
        sac.update(missing_task)

    missing_cost = dict(batch)
    missing_cost.pop("cost_vec")
    with pytest.raises(KeyError, match="cost_vec"):
        sac.update(missing_cost)


def test_lower_update_uses_next_macro_for_target_projection():
    cfg = _small_cfg()
    safety = SafetyLayer(cfg)
    calls: list[torch.Tensor] = []
    gamma_calls: list[torch.Tensor | None] = []
    delta_calls: list[torch.Tensor | None] = []
    original = safety.project_torch

    def wrapped(self, lower_raw, boost_combo, mode, temps, amb_temp, gamma=None, delta=None):
        calls.append(boost_combo.detach().cpu().clone())
        gamma_calls.append(None if gamma is None else gamma.detach().cpu().clone())
        delta_calls.append(None if delta is None else delta.detach().cpu().clone())
        return original(lower_raw, boost_combo, mode, temps, amb_temp, gamma, delta)

    safety.project_torch = types.MethodType(wrapped, safety)
    sac = LowerSAC(cfg, safety, torch.device("cpu"))

    b = 4
    obs_dim = int(cfg["agent"]["obs_dim"])
    z_dim = int(cfg["agent"]["z_dim"])
    n_tx = int(cfg["env"]["n_tx"])
    physical_dim = int(cfg["physical_context"]["input_dim"])

    batch = {
        "obs": np.random.randn(b, obs_dim).astype(np.float32),
        "z": np.random.randn(b, z_dim).astype(np.float32),
        "act_exec": np.random.randn(b, 5).astype(np.float32),
        "reward": np.random.randn(b).astype(np.float32),
        "reward_raw": np.random.randn(b).astype(np.float32),
        "reward_task": np.random.randn(b).astype(np.float32),
        "reward_benchmark": np.random.randn(b).astype(np.float32),
        "reward_dual_penalized": np.random.randn(b).astype(np.float32),
        "next_obs": np.random.randn(b, obs_dim).astype(np.float32),
        "z_next": np.random.randn(b, z_dim).astype(np.float32),
        "done": np.zeros(b, dtype=np.float32),
        "boost_combo_exec": np.zeros(b, dtype=np.float32),
        "mode_exec": np.zeros(b, dtype=np.float32),
        "boost_combo_exec_next": np.full(b, 3, dtype=np.float32),
        "mode_exec_next": np.full(b, 2, dtype=np.float32),
        "cost_vec": np.abs(np.random.randn(b, 4).astype(np.float32)) * 0.01,
        "physical_features": np.random.randn(b, physical_dim).astype(np.float32),
        "physical_features_next": np.random.randn(b, physical_dim).astype(np.float32),
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
    assert all(g is not None for g in gamma_calls[:2])
    assert all(d is not None for d in delta_calls[:2])
    assert torch.allclose(gamma_calls[0], torch.full((b,), float(cfg["env"]["gamma"])))
    assert torch.allclose(delta_calls[0], torch.full((b,), float(cfg["env"]["delta"])))
