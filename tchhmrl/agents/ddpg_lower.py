from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from tchhmrl.models.networks import ContinuousQNetwork, DeterministicTanhPolicy
from tchhmrl.safety.safety_layer import SafetyLayer, raw_from_frac01


class LowerDDPG:
    _RHO_TAU_CONTRACTS = {"rho_tau_fixed_current", "rho_tau_codebook_current"}

    def __init__(self, cfg: Dict, safety: SafetyLayer, device: torch.device):
        agent_cfg = cfg["agent"]
        ddpg_cfg = cfg["lower_ddpg"]

        self.device = device
        self.safety = safety

        self.obs_dim = int(agent_cfg["obs_dim"])
        self.z_dim = int(agent_cfg["z_dim"])
        self.act_dim = int(agent_cfg["act_lower_dim"])
        self.action_contract = str(ddpg_cfg.get("action_contract", "full_lower_action"))
        if self.action_contract in self._RHO_TAU_CONTRACTS:
            self.learned_act_dim = 2
        else:
            self.learned_act_dim = int(ddpg_cfg.get("learned_action_dim", self.act_dim))
        self.fixed_current_fraction = float(ddpg_cfg.get("fixed_current_fraction", 0.5))
        self.fixed_current_fraction = float(np.clip(self.fixed_current_fraction, 1.0e-4, 1.0 - 1.0e-4))
        current_template_levels = np.asarray(
            ddpg_cfg.get("current_template_levels", [0.35, 0.50, 0.65]),
            dtype=np.float32,
        ).reshape(-1)
        if current_template_levels.size != 3:
            if self.action_contract == "rho_tau_codebook_current":
                raise ValueError("lower_ddpg.current_template_levels must contain exactly 3 levels")
            current_template_levels = np.asarray([0.35, 0.50, 0.65], dtype=np.float32)
        self.current_template_levels = current_template_levels
        self.current_template_levels = np.clip(self.current_template_levels, 1.0e-4, 1.0 - 1.0e-4)
        self.upper_contract = str(ddpg_cfg.get("upper_contract", "boost_mode")).lower()
        if self.upper_contract not in {"boost_mode", "boost_current_template"}:
            raise ValueError(f"unsupported lower_ddpg.upper_contract={self.upper_contract}")
        if self.action_contract == "rho_tau_codebook_current" and self.upper_contract != "boost_current_template":
            raise ValueError("rho_tau_codebook_current requires lower_ddpg.upper_contract=boost_current_template")
        if self.upper_contract == "boost_current_template" and self.action_contract != "rho_tau_codebook_current":
            raise ValueError("boost_current_template requires lower_ddpg.action_contract=rho_tau_codebook_current")
        self.action_decode_mode = str(cfg.get("safety", {}).get("action_decode_mode", "tanh_affine")).lower()
        self._fixed_current_raw = float(
            raw_from_frac01(self.fixed_current_fraction, self.action_decode_mode).reshape(-1)[0]
        )
        self._current_template_raw = raw_from_frac01(self.current_template_levels, self.action_decode_mode).astype(
            np.float32
        )
        self.upper_ctx_dim = int(agent_cfg.get("lower_upper_ctx_dim", 7))
        self.obs_aug_dim = self.obs_dim + self.upper_ctx_dim
        hidden = int(agent_cfg["hidden_dim"])

        self.gamma_rl = float(ddpg_cfg.get("gamma", agent_cfg["gamma"]))
        self.target_tau = float(ddpg_cfg.get("target_tau", ddpg_cfg.get("tau", agent_cfg["tau"])))
        self.grad_clip = float(ddpg_cfg.get("grad_clip", 5.0))
        self.noise_std = float(ddpg_cfg.get("noise_std", 0.10))

        self.actor = DeterministicTanhPolicy(self.obs_aug_dim, self.z_dim, self.learned_act_dim, hidden).to(device)
        self.actor_tgt = DeterministicTanhPolicy(self.obs_aug_dim, self.z_dim, self.learned_act_dim, hidden).to(device)
        self.critic = ContinuousQNetwork(self.obs_aug_dim, self.z_dim, self.act_dim, hidden).to(device)
        self.critic_tgt = ContinuousQNetwork(self.obs_aug_dim, self.z_dim, self.act_dim, hidden).to(device)

        self.actor_tgt.load_state_dict(self.actor.state_dict())
        self.critic_tgt.load_state_dict(self.critic.state_dict())

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=float(ddpg_cfg["actor_lr"]))
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=float(ddpg_cfg["critic_lr"]))
        self.update_steps = 0

    def _raw_from_template_level_torch(self, current_template_level: torch.Tensor) -> torch.Tensor:
        levels = torch.as_tensor(self.current_template_levels, dtype=torch.float32, device=current_template_level.device)
        level_idx = torch.clamp(current_template_level.long().view(-1), 0, levels.numel() - 1)
        frac = levels[level_idx]
        if self.action_decode_mode == "sigmoid_logit":
            frac = torch.clamp(frac, min=1.0e-4, max=1.0 - 1.0e-4)
            return torch.log(frac / (1.0 - frac))
        return 2.0 * frac - 1.0

    def _expand_learned_raw_torch(
        self,
        learned_raw: torch.Tensor,
        current_template_level: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.action_contract not in self._RHO_TAU_CONTRACTS:
            return learned_raw
        if learned_raw.shape[1] != 2:
            raise ValueError(f"{self.action_contract} expects 2 learned action dims, got {learned_raw.shape[1]}")
        if self.action_contract == "rho_tau_codebook_current":
            if current_template_level is None:
                raise ValueError("rho_tau_codebook_current requires current_template_level")
            current_raw = self._raw_from_template_level_torch(current_template_level).to(dtype=learned_raw.dtype)
            fixed = current_raw.view(-1, 1).expand(-1, 3)
        else:
            fixed = torch.full(
                (learned_raw.shape[0], 3),
                self._fixed_current_raw,
                dtype=learned_raw.dtype,
                device=learned_raw.device,
            )
        return torch.cat([fixed, learned_raw], dim=1)

    def _expand_learned_raw_np(
        self,
        learned_raw: np.ndarray,
        upper_idx: float | int | None = None,
        current_template_level: float | int | None = None,
    ) -> np.ndarray:
        learned_raw = np.asarray(learned_raw, dtype=np.float32).reshape(-1)
        if self.action_contract not in self._RHO_TAU_CONTRACTS:
            return learned_raw.astype(np.float32)
        if learned_raw.size != 2:
            raise ValueError(f"{self.action_contract} expects 2 learned action dims, got {learned_raw.size}")
        if self.action_contract == "rho_tau_codebook_current":
            if current_template_level is None:
                if upper_idx is None:
                    raise ValueError("rho_tau_codebook_current requires upper_idx or current_template_level")
                current_template_level = int(np.clip(int(upper_idx), 0, 11)) % 3
            level = int(np.clip(int(current_template_level), 0, self._current_template_raw.size - 1))
            fixed = np.full((3,), float(self._current_template_raw[level]), dtype=np.float32)
        else:
            fixed = np.full((3,), self._fixed_current_raw, dtype=np.float32)
        return np.concatenate([fixed, learned_raw.astype(np.float32)], axis=0)

    def _upper_ctx_np(self, upper_idx: float | int) -> np.ndarray:
        idx = int(np.clip(int(upper_idx), 0, 11))
        boost = idx // 3
        selector = idx % 3
        boost_oh = np.eye(4, dtype=np.float32)[boost]
        selector_oh = np.eye(3, dtype=np.float32)[selector]
        return np.concatenate([boost_oh, selector_oh]).astype(np.float32)

    @staticmethod
    def _upper_ctx_torch(boost: torch.Tensor, selector: torch.Tensor) -> torch.Tensor:
        boost = torch.clamp(boost.long().view(-1), 0, 3)
        selector = torch.clamp(selector.long().view(-1), 0, 2)
        boost_oh = F.one_hot(boost, num_classes=4).float()
        selector_oh = F.one_hot(selector, num_classes=3).float()
        return torch.cat([boost_oh, selector_oh], dim=1)

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
        return self._expand_learned_raw_np(raw.squeeze(0).cpu().numpy(), upper_idx=upper_idx)

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
        current_template_level = torch.tensor(
            batch.get(
                "current_template_level_exec",
                np.mod(batch.get("upper_idx_train", batch.get("upper_idx_exec", batch.get("upper_idx", np.zeros_like(batch["reward"])))), 3),
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
        current_template_level_next = torch.tensor(
            batch.get(
                "current_template_level_exec_next",
                batch.get(
                    "current_template_level_exec",
                    np.mod(
                        batch.get(
                            "upper_idx_train_next",
                            batch.get(
                                "upper_idx_exec_next",
                                batch.get("upper_idx_exec", batch.get("upper_idx", np.zeros_like(batch["reward"]))),
                            ),
                        ),
                        3,
                    ),
                ),
            ),
            dtype=torch.long,
            device=self.device,
        )

        selector = current_template_level if self.upper_contract == "boost_current_template" else mode
        selector_next = current_template_level_next if self.upper_contract == "boost_current_template" else mode_next

        obs_aug = torch.cat([obs, self._upper_ctx_torch(boost, selector)], dim=1)
        next_obs_aug = torch.cat([next_obs, self._upper_ctx_torch(boost_next, selector_next)], dim=1)

        temps = torch.tensor(batch["temps"], dtype=torch.float32, device=self.device)
        next_temps = torch.tensor(batch["next_temps"], dtype=torch.float32, device=self.device)
        amb = torch.tensor(batch["amb_temp"], dtype=torch.float32, device=self.device)
        gamma_env = torch.tensor(batch["gamma_env"], dtype=torch.float32, device=self.device)
        delta_env = torch.tensor(batch["delta_env"], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            raw_next = self._expand_learned_raw_torch(
                self.actor_tgt(next_obs_aug, z_next),
                current_template_level=current_template_level_next,
            )
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

        raw_pi = self._expand_learned_raw_torch(self.actor(obs_aug, z), current_template_level=current_template_level)
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
