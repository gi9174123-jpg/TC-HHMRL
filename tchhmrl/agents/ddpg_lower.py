from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from tchhmrl.models.networks import ContinuousQNetwork, DeterministicTanhPolicy
from tchhmrl.safety.safety_layer import SafetyLayer


class LowerDDPG:
    def __init__(self, cfg: Dict, safety: SafetyLayer, device: torch.device):
        agent_cfg = cfg["agent"]
        ddpg_cfg = cfg["lower_ddpg"]

        self.device = device
        self.safety = safety

        self.obs_dim = int(agent_cfg["obs_dim"])
        self.z_dim = int(agent_cfg["z_dim"])
        self.act_dim = int(agent_cfg["act_lower_dim"])
        self.upper_ctx_dim = int(agent_cfg.get("lower_upper_ctx_dim", 7))
        self.obs_aug_dim = self.obs_dim + self.upper_ctx_dim
        hidden = int(agent_cfg["hidden_dim"])

        self.gamma_rl = float(ddpg_cfg.get("gamma", agent_cfg["gamma"]))
        self.target_tau = float(ddpg_cfg.get("target_tau", ddpg_cfg.get("tau", agent_cfg["tau"])))
        self.grad_clip = float(ddpg_cfg.get("grad_clip", 5.0))
        self.noise_std = float(ddpg_cfg.get("noise_std", 0.10))

        self.actor = DeterministicTanhPolicy(self.obs_aug_dim, self.z_dim, self.act_dim, hidden).to(device)
        self.actor_tgt = DeterministicTanhPolicy(self.obs_aug_dim, self.z_dim, self.act_dim, hidden).to(device)
        self.critic = ContinuousQNetwork(self.obs_aug_dim, self.z_dim, self.act_dim, hidden).to(device)
        self.critic_tgt = ContinuousQNetwork(self.obs_aug_dim, self.z_dim, self.act_dim, hidden).to(device)

        self.actor_tgt.load_state_dict(self.actor.state_dict())
        self.critic_tgt.load_state_dict(self.critic.state_dict())

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=float(ddpg_cfg["actor_lr"]))
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=float(ddpg_cfg["critic_lr"]))
        self.update_steps = 0

    @staticmethod
    def _upper_ctx_np(upper_idx: float | int) -> np.ndarray:
        idx = int(np.clip(int(upper_idx), 0, 11))
        boost = idx // 3
        mode = idx % 3
        boost_oh = np.eye(4, dtype=np.float32)[boost]
        mode_oh = np.eye(3, dtype=np.float32)[mode]
        return np.concatenate([boost_oh, mode_oh]).astype(np.float32)

    @staticmethod
    def _upper_ctx_torch(boost: torch.Tensor, mode: torch.Tensor) -> torch.Tensor:
        boost = torch.clamp(boost.long().view(-1), 0, 3)
        mode = torch.clamp(mode.long().view(-1), 0, 2)
        boost_oh = F.one_hot(boost, num_classes=4).float()
        mode_oh = F.one_hot(mode, num_classes=3).float()
        return torch.cat([boost_oh, mode_oh], dim=1)

    def select_action(
        self,
        obs: np.ndarray,
        z: np.ndarray,
        upper_idx: float | int = 0,
        eval_mode: bool = False,
    ) -> np.ndarray:
        with torch.no_grad():
            upper_ctx = self._upper_ctx_np(upper_idx)
            obs_aug = np.concatenate([obs.astype(np.float32), upper_ctx]).astype(np.float32)
            obs_t = torch.tensor(obs_aug, dtype=torch.float32, device=self.device).unsqueeze(0)
            z_t = torch.tensor(z, dtype=torch.float32, device=self.device).unsqueeze(0)
            raw = self.actor(obs_t, z_t)
            if not eval_mode and self.noise_std > 0.0:
                raw = raw + self.noise_std * torch.randn_like(raw)
            raw = torch.clamp(raw, -1.0, 1.0)
        return raw.squeeze(0).cpu().numpy().astype(np.float32)

    def _soft_update(self, src: torch.nn.Module, dst: torch.nn.Module) -> None:
        for p, p_tgt in zip(src.parameters(), dst.parameters()):
            p_tgt.data.mul_(1.0 - self.target_tau)
            p_tgt.data.add_(self.target_tau * p.data)

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        obs = torch.tensor(batch["obs"], dtype=torch.float32, device=self.device)
        z = torch.tensor(batch["z"], dtype=torch.float32, device=self.device)
        act_exec = torch.tensor(batch["act_exec"], dtype=torch.float32, device=self.device)
        rew = torch.tensor(batch["reward"], dtype=torch.float32, device=self.device).view(-1, 1)
        next_obs = torch.tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        z_next = torch.tensor(batch["z_next"], dtype=torch.float32, device=self.device)
        done = torch.tensor(batch["done"], dtype=torch.float32, device=self.device).view(-1, 1)

        boost = torch.tensor(batch["boost_combo_exec"], dtype=torch.long, device=self.device)
        mode = torch.tensor(
            batch.get(
                "mode_exec",
                np.mod(batch.get("upper_idx_exec", batch.get("upper_idx", np.zeros_like(batch["reward"]))), 3),
            ),
            dtype=torch.long,
            device=self.device,
        )
        boost_next = torch.tensor(
            batch.get("boost_combo_exec_next", batch["boost_combo_exec"]),
            dtype=torch.long,
            device=self.device,
        )
        mode_next = torch.tensor(
            batch.get(
                "mode_exec_next",
                batch.get(
                    "mode_exec",
                    np.mod(batch.get("upper_idx_exec", batch.get("upper_idx", np.zeros_like(batch["reward"]))), 3),
                ),
            ),
            dtype=torch.long,
            device=self.device,
        )

        obs_aug = torch.cat([obs, self._upper_ctx_torch(boost, mode)], dim=1)
        next_obs_aug = torch.cat([next_obs, self._upper_ctx_torch(boost_next, mode_next)], dim=1)

        temps = torch.tensor(batch["temps"], dtype=torch.float32, device=self.device)
        next_temps = torch.tensor(batch["next_temps"], dtype=torch.float32, device=self.device)
        amb = torch.tensor(batch["amb_temp"], dtype=torch.float32, device=self.device)
        gamma_env = torch.tensor(batch["gamma_env"], dtype=torch.float32, device=self.device)
        delta_env = torch.tensor(batch["delta_env"], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            raw_next = self.actor_tgt(next_obs_aug, z_next)
            safe_next = self.safety.project_torch(raw_next, boost_next, mode_next, next_temps, amb, gamma_env, delta_env)
            a_next = torch.cat(
                [safe_next["currents_exec"], safe_next["rho_exec"], safe_next["tau_exec"]],
                dim=1,
            )
            td_target = rew + self.gamma_rl * (1.0 - done) * self.critic_tgt(next_obs_aug, z_next, a_next)

        q = self.critic(obs_aug, z, act_exec)
        critic_loss = F.mse_loss(q, td_target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optim.step()

        raw_pi = self.actor(obs_aug, z)
        safe_pi = self.safety.project_torch(raw_pi, boost, mode, temps, amb, gamma_env, delta_env)
        a_pi = torch.cat([safe_pi["currents_exec"], safe_pi["rho_exec"], safe_pi["tau_exec"]], dim=1)
        actor_loss = -self.critic(obs_aug, z, a_pi).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optim.step()

        self._soft_update(self.actor, self.actor_tgt)
        self._soft_update(self.critic, self.critic_tgt)
        self.update_steps += 1

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "q_mean": float(q.mean().item()),
        }

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "actor_tgt": self.actor_tgt.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_tgt": self.critic_tgt.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "update_steps": self.update_steps,
        }

    def load_state_dict(self, state):
        self.actor.load_state_dict(state["actor"])
        self.actor_tgt.load_state_dict(state["actor_tgt"])
        self.critic.load_state_dict(state["critic"])
        self.critic_tgt.load_state_dict(state["critic_tgt"])
        self.actor_optim.load_state_dict(state["actor_optim"])
        self.critic_optim.load_state_dict(state["critic_optim"])
        self.update_steps = int(state.get("update_steps", 0))
