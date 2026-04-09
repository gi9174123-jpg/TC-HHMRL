from __future__ import annotations

import numpy as np
import torch

from tchhmrl.agents.hierarchical_agent import HierarchicalAgent
from tchhmrl.envs.uw_slipt_env import MultiTxUwSliptEnv
from tchhmrl.utils.config import load_cfg


def test_hierarchical_agent_one_step_interaction():
    cfg = load_cfg("configs/default.yaml")
    agent = HierarchicalAgent(cfg, torch.device("cpu"))
    env = MultiTxUwSliptEnv(cfg)

    obs, _ = env.reset(seed=0)
    z = np.zeros(cfg["agent"]["z_dim"], dtype=np.float32)

    action, aux = agent.act(
        obs=obs,
        temps=env.temps,
        amb_temp=env.amb_temp,
        gamma=env.gamma,
        delta=env.delta,
        z=z,
        eval_mode=False,
    )
    next_obs, reward, terminated, truncated, info = env.step(action)

    transition = {
        "obs": obs.astype(np.float32),
        "next_obs": next_obs.astype(np.float32),
        "upper_idx_raw": float(aux["upper_idx_raw"]),
        "upper_idx_exec": float(aux["upper_idx_exec"]),
        "reward": float(reward),
        "reward_raw": float(reward),
        "done": float(terminated or truncated),
        "z": z,
        "z_next": z,
        "act_exec": aux["act_exec"],
        "boost_combo_exec": float(aux["boost_combo_exec"]),
        "mode_exec": float(aux["mode_exec"]),
        "temps": env.temps.astype(np.float32),
        "next_temps": info["temps"].astype(np.float32),
        "amb_temp": float(info["amb_temp"]),
        "gamma_env": float(info["gamma"]),
        "delta_env": float(info["delta"]),
        "cost": float(info["cost"]),
        "cost_vec": info["cost_vec"].astype(np.float32),
    }

    agent.observe(transition)
    metrics = agent.learn()

    assert isinstance(metrics, dict)
