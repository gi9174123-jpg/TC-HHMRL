from __future__ import annotations

import types

import numpy as np
import torch

from tchhmrl.agents.dqn_upper import UpperDQN
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
