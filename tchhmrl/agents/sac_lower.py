from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from tchhmrl.models.networks import ContinuousQNetwork, GaussianTanhPolicy
from tchhmrl.safety.safety_layer import SafetyLayer


class LowerSAC:
    def __init__(self, cfg: Dict, safety: SafetyLayer, device: torch.device):
        agent_cfg = cfg["agent"]
        sac_cfg = cfg["lower_sac"]

        self.device = device
        self.safety = safety

        self.obs_dim = int(agent_cfg["obs_dim"])
        self.z_dim = int(agent_cfg["z_dim"])
        self.act_dim = int(agent_cfg["act_lower_dim"])
        # Explicitly condition lower policy/value on upper structural choice.
        # [boost_onehot(4), mode_onehot(3)].
        self.upper_ctx_dim = int(agent_cfg.get("lower_upper_ctx_dim", 7))
        self.obs_aug_dim = self.obs_dim + self.upper_ctx_dim
        hidden = int(agent_cfg["hidden_dim"])

        self.gamma_rl = float(agent_cfg["gamma"])
        self.tau = float(agent_cfg["tau"])
        alpha_default = float(sac_cfg.get("alpha", 0.10))
        self.alpha_start = float(sac_cfg.get("alpha_start", alpha_default))
        self.alpha_end = float(sac_cfg.get("alpha_end", self.alpha_start))
        self.alpha_decay_steps = int(sac_cfg.get("alpha_decay_steps", 0))
        self.auto_alpha = bool(sac_cfg.get("auto_alpha", False))
        self.target_entropy = float(sac_cfg.get("target_entropy", -float(self.act_dim)))
        self.grad_clip = float(sac_cfg["grad_clip"])
        self.update_steps = 0

        self.actor = GaussianTanhPolicy(self.obs_aug_dim, self.z_dim, self.act_dim, hidden).to(device)
        self.q1 = ContinuousQNetwork(self.obs_aug_dim, self.z_dim, self.act_dim, hidden).to(device)
        self.q2 = ContinuousQNetwork(self.obs_aug_dim, self.z_dim, self.act_dim, hidden).to(device)
        self.q1_tgt = ContinuousQNetwork(self.obs_aug_dim, self.z_dim, self.act_dim, hidden).to(device)
        self.q2_tgt = ContinuousQNetwork(self.obs_aug_dim, self.z_dim, self.act_dim, hidden).to(device)

        self.q1_tgt.load_state_dict(self.q1.state_dict())
        self.q2_tgt.load_state_dict(self.q2.state_dict())

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=float(sac_cfg["actor_lr"]))
        self.critic_optim = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=float(sac_cfg["critic_lr"]),
        )
        if self.auto_alpha:
            init_alpha = max(self.alpha_start, 1.0e-6)
            self.log_alpha = torch.tensor(np.log(init_alpha), dtype=torch.float32, device=self.device, requires_grad=True)
            self.alpha_optim = torch.optim.Adam(
                [self.log_alpha],
                lr=float(sac_cfg.get("alpha_lr", sac_cfg["actor_lr"])),
            )
        else:
            self.log_alpha = None
            self.alpha_optim = None

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
            if eval_mode:
                raw = self.actor.deterministic(obs_t, z_t)
            else:
                raw, _ = self.actor.sample(obs_t, z_t)
        return raw.squeeze(0).cpu().numpy().astype(np.float32)

    def _soft_update(self, src: torch.nn.Module, dst: torch.nn.Module) -> None:
        for p, p_tgt in zip(src.parameters(), dst.parameters()):
            p_tgt.data.mul_(1.0 - self.tau)
            p_tgt.data.add_(self.tau * p.data)

    def _alpha_tensor(self, dtype: torch.dtype) -> torch.Tensor:
        if self.auto_alpha:
            return self.log_alpha.exp()
        if self.alpha_decay_steps <= 0:
            alpha_val = self.alpha_start
        else:
            frac = min(float(self.update_steps) / float(max(1, self.alpha_decay_steps)), 1.0)
            alpha_val = self.alpha_start + frac * (self.alpha_end - self.alpha_start)
        return torch.tensor(alpha_val, dtype=dtype, device=self.device)

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
        alpha_t = self._alpha_tensor(dtype=obs.dtype)

        with torch.no_grad():
            raw_next, logp_next = self.actor.sample(next_obs_aug, z_next)
            safe_next = self.safety.project_torch(raw_next, boost_next, mode_next, next_temps, amb, gamma_env, delta_env)
            a_next = torch.cat(
                [safe_next["currents_exec"], safe_next["rho_exec"], safe_next["tau_exec"]], dim=1
            )
            q_next = torch.min(
                self.q1_tgt(next_obs_aug, z_next, a_next),
                self.q2_tgt(next_obs_aug, z_next, a_next),
            ) - alpha_t * logp_next
            td_target = rew + self.gamma_rl * (1.0 - done) * q_next

        q1 = self.q1(obs_aug, z, act_exec)
        q2 = self.q2(obs_aug, z, act_exec)
        loss_q = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        self.critic_optim.zero_grad()
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), self.grad_clip)
        self.critic_optim.step()

        raw_pi, logp = self.actor.sample(obs_aug, z)
        safe_pi = self.safety.project_torch(raw_pi, boost, mode, temps, amb, gamma_env, delta_env)
        a_pi = torch.cat([safe_pi["currents_exec"], safe_pi["rho_exec"], safe_pi["tau_exec"]], dim=1)
        q_pi = torch.min(self.q1(obs_aug, z, a_pi), self.q2(obs_aug, z, a_pi))
        loss_pi = (alpha_t * logp - q_pi).mean()

        self.actor_optim.zero_grad()
        loss_pi.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optim.step()

        alpha_loss_val = 0.0
        if self.auto_alpha and self.alpha_optim is not None and self.log_alpha is not None:
            alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            alpha_loss_val = float(alpha_loss.item())

        self._soft_update(self.q1, self.q1_tgt)
        self._soft_update(self.q2, self.q2_tgt)
        self.update_steps += 1

        return {
            "q1_mean": float(q1.mean().item()),
            "q2_mean": float(q2.mean().item()),
            "critic_loss": float(loss_q.item()),
            "actor_loss": float(loss_pi.item()),
            "entropy": float((-logp).mean().item()),
            "alpha": float(alpha_t.detach().item()),
            "alpha_loss": float(alpha_loss_val),
        }

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_tgt": self.q1_tgt.state_dict(),
            "q2_tgt": self.q2_tgt.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "update_steps": self.update_steps,
            "auto_alpha": self.auto_alpha,
            "alpha_start": self.alpha_start,
            "alpha_end": self.alpha_end,
            "alpha_decay_steps": self.alpha_decay_steps,
            "target_entropy": self.target_entropy,
            "log_alpha": self.log_alpha.detach().cpu().item() if self.log_alpha is not None else None,
            "alpha_optim": self.alpha_optim.state_dict() if self.alpha_optim is not None else None,
        }

    def load_state_dict(self, state):
        self.actor.load_state_dict(state["actor"])
        self.q1.load_state_dict(state["q1"])
        self.q2.load_state_dict(state["q2"])
        self.q1_tgt.load_state_dict(state["q1_tgt"])
        self.q2_tgt.load_state_dict(state["q2_tgt"])
        self.actor_optim.load_state_dict(state["actor_optim"])
        self.critic_optim.load_state_dict(state["critic_optim"])
        self.update_steps = int(state.get("update_steps", 0))
        if self.auto_alpha and self.log_alpha is not None and state.get("log_alpha") is not None:
            with torch.no_grad():
                self.log_alpha.copy_(torch.tensor(float(state["log_alpha"]), device=self.device))
            if self.alpha_optim is not None and state.get("alpha_optim") is not None:
                self.alpha_optim.load_state_dict(state["alpha_optim"])
