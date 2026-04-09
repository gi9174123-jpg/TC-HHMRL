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


def test_lower_update_uses_next_macro_for_target_projection():
    cfg = _small_cfg()
    safety = SafetyLayer(cfg)
    calls: list[torch.Tensor] = []
    original = safety.project_torch

    def wrapped(self, lower_raw, boost_combo, mode, temps, amb_temp, gamma, delta):
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
