from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from tchhmrl.models.networks import DiscreteQNetwork


class UpperDQN:
    def __init__(self, cfg: Dict, device: torch.device):
        agent_cfg = cfg["agent"]
        dqn_cfg = cfg["upper_dqn"]

        self.device = device
        self.obs_dim = int(agent_cfg["obs_dim"])
        self.z_dim = int(agent_cfg["z_dim"])
        self.n_actions = int(agent_cfg["n_upper_actions"])
        hidden = int(agent_cfg["hidden_dim"])

        self.gamma = float(agent_cfg["gamma"])
        self.grad_clip = float(dqn_cfg["grad_clip"])

        self.q = DiscreteQNetwork(self.obs_dim, self.z_dim, self.n_actions, hidden).to(device)
        self.q_tgt = DiscreteQNetwork(self.obs_dim, self.z_dim, self.n_actions, hidden).to(device)
        self.q_tgt.load_state_dict(self.q.state_dict())

        self.optim = torch.optim.Adam(self.q.parameters(), lr=float(dqn_cfg["lr"]))

        self.epsilon_start = float(dqn_cfg["epsilon_start"])
        self.epsilon_final = float(dqn_cfg["epsilon_final"])
        self.epsilon_decay_steps = int(dqn_cfg["epsilon_decay_steps"])
        self.target_update_every = int(dqn_cfg["target_update_every"])

        self.update_steps = 0

    def epsilon(self, t: int) -> float:
        frac = min(float(t) / max(1, self.epsilon_decay_steps), 1.0)
        return self.epsilon_start + frac * (self.epsilon_final - self.epsilon_start)

    def select_action(
        self,
        obs: np.ndarray,
        z: np.ndarray,
        t: int,
        eval_mode: bool = False,
        exec_map: np.ndarray | None = None,
    ) -> int:
        if (not eval_mode) and np.random.rand() < self.epsilon(t):
            return int(np.random.randint(0, self.n_actions))

        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            z_t = torch.tensor(z, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q(obs_t, z_t)
            if exec_map is None:
                return int(torch.argmax(q, dim=1).item())
            exec_idx = torch.as_tensor(np.asarray(exec_map, dtype=np.int64), device=self.device).view(1, -1)
            exec_idx = torch.clamp(exec_idx, 0, self.n_actions - 1)
            q_raw = torch.gather(q, 1, exec_idx)
            return int(torch.argmax(q_raw, dim=1).item())

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        obs = torch.tensor(batch["obs"], dtype=torch.float32, device=self.device)
        z = torch.tensor(batch["z"], dtype=torch.float32, device=self.device)
        act = torch.tensor(
            batch.get("upper_idx_exec", batch.get("upper_idx", np.zeros_like(batch["reward"]))),
            dtype=torch.long,
            device=self.device,
        )
        rew = torch.tensor(batch["reward"], dtype=torch.float32, device=self.device).view(-1, 1)
        next_obs = torch.tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        z_next = torch.tensor(batch["z_next"], dtype=torch.float32, device=self.device)
        done = torch.tensor(batch["done"], dtype=torch.float32, device=self.device).view(-1, 1)
        horizon = torch.tensor(
            batch.get("horizon", np.ones_like(batch["reward"], dtype=np.float32)),
            dtype=torch.float32,
            device=self.device,
        ).view(-1, 1)

        q_all = self.q(obs, z)
        q_a = torch.gather(q_all, 1, act.view(-1, 1))

        with torch.no_grad():
            if "next_exec_map" in batch:
                next_exec_map = torch.tensor(batch["next_exec_map"], dtype=torch.long, device=self.device)
                next_exec_map = torch.clamp(next_exec_map, 0, self.n_actions - 1)
                q_next_all = self.q_tgt(next_obs, z_next)
                q_next_raw = torch.gather(q_next_all, 1, next_exec_map)
                q_next = q_next_raw.max(dim=1, keepdim=True)[0]
            else:
                q_next = self.q_tgt(next_obs, z_next).max(dim=1, keepdim=True)[0]
            discount = torch.pow(torch.full_like(horizon, self.gamma), horizon)
            td_target = rew + discount * (1.0 - done) * q_next

        loss = F.smooth_l1_loss(q_a, td_target)

        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
        self.optim.step()

        self.update_steps += 1
        if self.update_steps % self.target_update_every == 0:
            self.q_tgt.load_state_dict(self.q.state_dict())

        return {"q_loss": float(loss.item())}

    def state_dict(self):
        return {
            "q": self.q.state_dict(),
            "q_tgt": self.q_tgt.state_dict(),
            "optim": self.optim.state_dict(),
            "update_steps": self.update_steps,
        }

    def load_state_dict(self, state):
        self.q.load_state_dict(state["q"])
        self.q_tgt.load_state_dict(state["q_tgt"])
        self.optim.load_state_dict(state["optim"])
        self.update_steps = state.get("update_steps", 0)
