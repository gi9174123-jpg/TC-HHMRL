from __future__ import annotations

import types

import numpy as np
import torch

from tchhmrl.agents.dqn_upper import UpperDQN
from tchhmrl.models.networks import DuelingDiscreteQNetwork
from tchhmrl.utils.config import load_cfg


def test_upper_dqn_select_action_uses_exec_map():
    cfg = load_cfg("configs/default.yaml")
    upper = UpperDQN(cfg, torch.device("cpu"))

    q_vals = torch.arange(12, dtype=torch.float32).view(1, -1)

    def fake_forward(self, obs, z):
        return q_vals.expand(obs.shape[0], -1)

    upper.q.forward = types.MethodType(fake_forward, upper.q)

    obs = np.zeros(cfg["agent"]["obs_dim"], dtype=np.float32)
    z = np.zeros(cfg["agent"]["z_dim"], dtype=np.float32)
    exec_map = np.zeros(12, dtype=np.int64)
    exec_map[0] = 11

    picked = upper.select_action(obs, z, t=10_000, eval_mode=True, exec_map=exec_map)

    assert picked == 0


def test_upper_dqn_random_selection_respects_action_mask():
    cfg = load_cfg("configs/default.yaml")
    upper = UpperDQN(cfg, torch.device("cpu"))
    upper.epsilon_start = 1.0
    upper.epsilon_final = 1.0

    obs = np.zeros(cfg["agent"]["obs_dim"], dtype=np.float32)
    z = np.zeros(cfg["agent"]["z_dim"], dtype=np.float32)
    action_mask = np.zeros(12, dtype=bool)
    action_mask[[0, 1, 2]] = True

    for _ in range(20):
        picked = upper.select_action(obs, z, t=0, eval_mode=False, action_mask=action_mask)
        assert picked in {0, 1, 2}


def test_upper_dqn_greedy_selection_respects_action_mask():
    cfg = load_cfg("configs/default.yaml")
    upper = UpperDQN(cfg, torch.device("cpu"))
    q_vals = torch.arange(12, dtype=torch.float32).view(1, -1)

    def fake_forward(self, obs, z):
        return q_vals.expand(obs.shape[0], -1)

    upper.q.forward = types.MethodType(fake_forward, upper.q)

    obs = np.zeros(cfg["agent"]["obs_dim"], dtype=np.float32)
    z = np.zeros(cfg["agent"]["z_dim"], dtype=np.float32)
    action_mask = np.zeros(12, dtype=bool)
    action_mask[[0, 1, 2]] = True

    picked = upper.select_action(obs, z, t=10_000, eval_mode=True, action_mask=action_mask)
    assert picked == 2


def test_upper_dqn_uses_dueling_double_dqn_and_updates_with_exec_map():
    cfg = load_cfg("configs/default.yaml")
    cfg["agent"]["hidden_dim"] = 64
    cfg["upper_dqn"]["batch_size"] = 4
    upper = UpperDQN(cfg, torch.device("cpu"))

    assert upper.double_dqn is True
    assert upper.dueling is True
    assert isinstance(upper.q, DuelingDiscreteQNetwork)
    assert upper.physical_enabled is True
    assert upper.q_phys is not None

    b = 4
    batch = {
        "obs": np.random.randn(b, int(cfg["agent"]["obs_dim"])).astype(np.float32),
        "z": np.random.randn(b, int(cfg["agent"]["z_dim"])).astype(np.float32),
        "upper_idx_train": np.asarray([0, 1, 2, 3], dtype=np.float32),
        "reward": np.random.randn(b).astype(np.float32),
        "next_obs": np.random.randn(b, int(cfg["agent"]["obs_dim"])).astype(np.float32),
        "z_next": np.random.randn(b, int(cfg["agent"]["z_dim"])).astype(np.float32),
        "done": np.zeros(b, dtype=np.float32),
        "horizon": np.ones(b, dtype=np.float32),
        "physical_features": np.random.randn(b, int(cfg["physical_context"]["input_dim"])).astype(np.float32),
        "physical_features_next": np.random.randn(b, int(cfg["physical_context"]["input_dim"])).astype(np.float32),
        "next_exec_map": np.tile(np.arange(int(cfg["agent"]["n_upper_actions"]), dtype=np.float32), (b, 1)),
        "next_action_mask": np.tile(
            np.asarray([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=np.float32),
            (b, 1),
        ),
    }

    stats = upper.update(batch)

    assert "q_loss" in stats
    assert np.isfinite(stats["q_loss"])
