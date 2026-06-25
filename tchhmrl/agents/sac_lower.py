from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tchhmrl.models.networks import ContinuousQNetwork, GaussianTanhPolicy
from tchhmrl.models.constraint_critics import ConstraintQNetwork
from tchhmrl.models.physical_encoder import PhysicalEncoder
from tchhmrl.safety.safety_layer import SafetyLayer
from tchhmrl.agents.transition_schema import require_transition_keys, validate_executed_action_shape


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
        self.obs_upper_dim = self.obs_dim + self.upper_ctx_dim
        phys_cfg = cfg.get("physical_context", {}) or {}
        self.physical_enabled = bool(phys_cfg.get("enabled", False))
        self.physical_dim = int(phys_cfg.get("input_dim", 18))
        self.physical_embedding_dim = int(phys_cfg.get("embedding_dim", 32)) if self.physical_enabled else 0
        self.obs_aug_dim = self.obs_upper_dim + self.physical_embedding_dim
        hidden = int(agent_cfg["hidden_dim"])
        physical_hidden = int(phys_cfg.get("hidden_dim", 64))

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
        constraint_cfg = cfg.get("constraint_critics", {}) or {}
        self.constraint_critics_enabled = bool(constraint_cfg.get("enabled", False))
        self.constraint_dim = int(constraint_cfg.get("out_dim", len(cfg.get("meta", {}).get("dual_names", [])) or 4))
        default_reward_target = "raw_reward" if self.constraint_critics_enabled else "penalized_reward"
        self.reward_target_mode = str(constraint_cfg.get("reward_target", default_reward_target)).strip().lower()
        if self.reward_target_mode not in {"raw_reward", "penalized_reward"}:
            raise ValueError(
                "constraint_critics.reward_target must be 'raw_reward' or 'penalized_reward', "
                f"got {self.reward_target_mode!r}"
            )
        self.constraint_target_nonnegative = bool(constraint_cfg.get("target_nonnegative", True))
        self.constraint_actor_penalty_nonnegative = bool(constraint_cfg.get("actor_penalty_nonnegative", True))
        raw_weights = constraint_cfg.get("actor_weights", [0.0] * self.constraint_dim)
        weights = np.asarray(raw_weights, dtype=np.float32).reshape(-1)
        if weights.size < self.constraint_dim:
            weights = np.pad(weights, (0, self.constraint_dim - weights.size), constant_values=0.0)
        self.constraint_actor_weights = torch.tensor(
            weights[: self.constraint_dim],
            dtype=torch.float32,
            device=self.device,
        ).view(1, -1)

        self.actor_phys = (
            PhysicalEncoder(self.physical_dim, physical_hidden, self.physical_embedding_dim).to(device)
            if self.physical_enabled
            else None
        )
        self.q1_phys = (
            PhysicalEncoder(self.physical_dim, physical_hidden, self.physical_embedding_dim).to(device)
            if self.physical_enabled
            else None
        )
        self.q2_phys = (
            PhysicalEncoder(self.physical_dim, physical_hidden, self.physical_embedding_dim).to(device)
            if self.physical_enabled
            else None
        )
        self.q1_tgt_phys = (
            PhysicalEncoder(self.physical_dim, physical_hidden, self.physical_embedding_dim).to(device)
            if self.physical_enabled
            else None
        )
        self.q2_tgt_phys = (
            PhysicalEncoder(self.physical_dim, physical_hidden, self.physical_embedding_dim).to(device)
            if self.physical_enabled
            else None
        )

        self.actor = GaussianTanhPolicy(self.obs_aug_dim, self.z_dim, self.act_dim, hidden).to(device)
        self.q1 = ContinuousQNetwork(self.obs_aug_dim, self.z_dim, self.act_dim, hidden).to(device)
        self.q2 = ContinuousQNetwork(self.obs_aug_dim, self.z_dim, self.act_dim, hidden).to(device)
        self.q1_tgt = ContinuousQNetwork(self.obs_aug_dim, self.z_dim, self.act_dim, hidden).to(device)
        self.q2_tgt = ContinuousQNetwork(self.obs_aug_dim, self.z_dim, self.act_dim, hidden).to(device)
        self.constraint_q = (
            ConstraintQNetwork(self.obs_aug_dim, self.z_dim, self.act_dim, self.constraint_dim, hidden).to(device)
            if self.constraint_critics_enabled
            else None
        )
        self.constraint_q_tgt = (
            ConstraintQNetwork(self.obs_aug_dim, self.z_dim, self.act_dim, self.constraint_dim, hidden).to(device)
            if self.constraint_critics_enabled
            else None
        )

        self.q1_tgt.load_state_dict(self.q1.state_dict())
        self.q2_tgt.load_state_dict(self.q2.state_dict())
        if self.constraint_q is not None and self.constraint_q_tgt is not None:
            self.constraint_q_tgt.load_state_dict(self.constraint_q.state_dict())
        if self.physical_enabled:
            assert self.q1_phys is not None and self.q2_phys is not None
            assert self.q1_tgt_phys is not None and self.q2_tgt_phys is not None
            self.q1_tgt_phys.load_state_dict(self.q1_phys.state_dict())
            self.q2_tgt_phys.load_state_dict(self.q2_phys.state_dict())

        actor_params = list(self.actor.parameters())
        if self.actor_phys is not None:
            actor_params += list(self.actor_phys.parameters())
        critic_params = list(self.q1.parameters()) + list(self.q2.parameters())
        if self.q1_phys is not None and self.q2_phys is not None:
            critic_params += list(self.q1_phys.parameters()) + list(self.q2_phys.parameters())
        if self.constraint_q is not None:
            critic_params += list(self.constraint_q.parameters())

        self.actor_optim = torch.optim.Adam(actor_params, lr=float(sac_cfg["actor_lr"]))
        self.critic_optim = torch.optim.Adam(
            critic_params,
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

    def _zero_physical_np(self) -> np.ndarray:
        return np.zeros((self.physical_dim,), dtype=np.float32)

    def _physical_torch(self, batch: Dict[str, np.ndarray], key: str, batch_size: int) -> torch.Tensor:
        val = batch.get(key)
        if val is None:
            return torch.zeros((batch_size, self.physical_dim), dtype=torch.float32, device=self.device)
        arr = np.asarray(val, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self.physical_dim:
            out = np.zeros((arr.shape[0], self.physical_dim), dtype=np.float32)
            n = min(arr.shape[1], self.physical_dim)
            out[:, :n] = arr[:, :n]
            arr = out
        return torch.tensor(arr, dtype=torch.float32, device=self.device)

    def _augment_np(
        self,
        obs: np.ndarray,
        *,
        upper_idx: float | int,
        physical_features: np.ndarray | None,
        encoder: nn.Module | None,
    ) -> torch.Tensor:
        upper_ctx = self._upper_ctx_np(upper_idx)
        obs_upper = np.concatenate([obs.astype(np.float32), upper_ctx]).astype(np.float32)
        obs_t = torch.tensor(obs_upper, dtype=torch.float32, device=self.device).unsqueeze(0)
        if not self.physical_enabled:
            return obs_t
        if physical_features is None:
            physical_features = self._zero_physical_np()
        phys = np.asarray(physical_features, dtype=np.float32).reshape(-1)
        if phys.size != self.physical_dim:
            padded = self._zero_physical_np()
            padded[: min(phys.size, self.physical_dim)] = phys[: self.physical_dim]
            phys = padded
        phys_t = torch.tensor(phys, dtype=torch.float32, device=self.device).unsqueeze(0)
        assert encoder is not None
        return torch.cat([obs_t, encoder(phys_t)], dim=1)

    def _augment_torch(
        self,
        obs: torch.Tensor,
        boost: torch.Tensor,
        mode: torch.Tensor,
        physical_features: torch.Tensor,
        encoder: nn.Module | None,
    ) -> torch.Tensor:
        obs_upper = torch.cat([obs, self._upper_ctx_torch(boost, mode)], dim=1)
        if not self.physical_enabled:
            return obs_upper
        assert encoder is not None
        return torch.cat([obs_upper, encoder(physical_features)], dim=1)

    def select_action(
        self,
        obs: np.ndarray,
        z: np.ndarray,
        upper_idx: float | int = 0,
        physical_features: np.ndarray | None = None,
        eval_mode: bool = False,
    ) -> np.ndarray:
        with torch.no_grad():
            obs_t = self._augment_np(
                obs,
                upper_idx=upper_idx,
                physical_features=physical_features,
                encoder=self.actor_phys,
            )
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

    def update(self, batch: Dict[str, np.ndarray], constraint_batch: Dict[str, np.ndarray] | None = None) -> Dict[str, float]:
        if self.constraint_critics_enabled:
            require_transition_keys(batch)
            if constraint_batch is not None:
                require_transition_keys(constraint_batch)
        obs = torch.tensor(batch["obs"], dtype=torch.float32, device=self.device)
        batch_size = int(obs.shape[0])
        z = torch.tensor(batch["z"], dtype=torch.float32, device=self.device)
        validate_executed_action_shape(batch, batch_size=batch_size)
        act_exec = torch.tensor(batch["act_exec"], dtype=torch.float32, device=self.device)
        reward_key = "reward_raw" if self.constraint_critics_enabled and self.reward_target_mode == "raw_reward" else "reward"
        if reward_key not in batch:
            raise KeyError(f"Missing {reward_key} required by lower reward critic schema")
        rew = torch.tensor(batch[reward_key], dtype=torch.float32, device=self.device).view(-1, 1)
        next_obs = torch.tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        z_next = torch.tensor(batch["z_next"], dtype=torch.float32, device=self.device)
        done = torch.tensor(batch["done"], dtype=torch.float32, device=self.device).view(-1, 1)
        cost_vec = torch.tensor(batch["cost_vec"], dtype=torch.float32, device=self.device)
        if cost_vec.dim() == 1:
            cost_vec = cost_vec.view(-1, 1)
        if cost_vec.shape[1] != self.constraint_dim:
            padded = torch.zeros((batch_size, self.constraint_dim), dtype=torch.float32, device=self.device)
            n = min(int(cost_vec.shape[1]), self.constraint_dim)
            padded[:, :n] = cost_vec[:, :n]
            cost_vec = padded
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
        physical = self._physical_torch(batch, "physical_features", batch_size)
        physical_next = self._physical_torch(batch, "physical_features_next", batch_size)
        obs_aug_actor = self._augment_torch(obs, boost, mode, physical, self.actor_phys)
        next_obs_aug_actor = self._augment_torch(next_obs, boost_next, mode_next, physical_next, self.actor_phys)
        obs_aug_q1 = self._augment_torch(obs, boost, mode, physical, self.q1_phys)
        obs_aug_q2 = self._augment_torch(obs, boost, mode, physical, self.q2_phys)
        next_obs_aug_q1_tgt = self._augment_torch(next_obs, boost_next, mode_next, physical_next, self.q1_tgt_phys)
        next_obs_aug_q2_tgt = self._augment_torch(next_obs, boost_next, mode_next, physical_next, self.q2_tgt_phys)

        temps = torch.tensor(batch["temps"], dtype=torch.float32, device=self.device)
        next_temps = torch.tensor(batch["next_temps"], dtype=torch.float32, device=self.device)

        amb = torch.tensor(batch["amb_temp"], dtype=torch.float32, device=self.device)
        alpha_t = self._alpha_tensor(dtype=obs.dtype)

        with torch.no_grad():
            raw_next, logp_next = self.actor.sample(next_obs_aug_actor, z_next)
            safe_next = self.safety.project_torch(raw_next, boost_next, mode_next, next_temps, amb)
            a_next = torch.cat(
                [safe_next["currents_exec"], safe_next["rho_exec"], safe_next["tau_exec"]], dim=1
            )
            q_next = torch.min(
                self.q1_tgt(next_obs_aug_q1_tgt, z_next, a_next),
                self.q2_tgt(next_obs_aug_q2_tgt, z_next, a_next),
            ) - alpha_t * logp_next
            td_target = rew + self.gamma_rl * (1.0 - done) * q_next
            if self.constraint_q_tgt is not None and constraint_batch is None:
                constraint_next = self.constraint_q_tgt(next_obs_aug_q1_tgt, z_next, a_next)
                if self.constraint_target_nonnegative:
                    constraint_next = torch.relu(constraint_next)
                constraint_target = cost_vec + self.gamma_rl * (1.0 - done) * constraint_next
            else:
                constraint_target = None

        q1 = self.q1(obs_aug_q1, z, act_exec)
        q2 = self.q2(obs_aug_q2, z, act_exec)
        loss_q = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)
        constraint_loss = torch.tensor(0.0, dtype=obs.dtype, device=self.device)
        constraint_batch_stats: Dict[str, float] = {}
        if self.constraint_q is not None and constraint_batch is not None:
            cb = constraint_batch
            c_obs = torch.tensor(cb["obs"], dtype=torch.float32, device=self.device)
            c_batch_size = int(c_obs.shape[0])
            c_z = torch.tensor(cb["z"], dtype=torch.float32, device=self.device)
            validate_executed_action_shape(cb, batch_size=c_batch_size)
            c_act_exec = torch.tensor(cb["act_exec"], dtype=torch.float32, device=self.device)
            c_next_obs = torch.tensor(cb["next_obs"], dtype=torch.float32, device=self.device)
            c_z_next = torch.tensor(cb["z_next"], dtype=torch.float32, device=self.device)
            c_done = torch.tensor(cb["done"], dtype=torch.float32, device=self.device).view(-1, 1)
            c_cost_vec = torch.tensor(cb["cost_vec"], dtype=torch.float32, device=self.device)
            if c_cost_vec.dim() == 1:
                c_cost_vec = c_cost_vec.view(-1, 1)
            if c_cost_vec.shape[1] != self.constraint_dim:
                padded = torch.zeros((c_batch_size, self.constraint_dim), dtype=torch.float32, device=self.device)
                n = min(int(c_cost_vec.shape[1]), self.constraint_dim)
                padded[:, :n] = c_cost_vec[:, :n]
                c_cost_vec = padded
            c_boost = torch.tensor(cb["boost_combo_exec"], dtype=torch.long, device=self.device)
            c_mode = torch.tensor(
                cb.get(
                    "mode_exec",
                    np.mod(cb.get("upper_idx_exec", cb.get("upper_idx", np.zeros(c_batch_size))), 3),
                ),
                dtype=torch.long,
                device=self.device,
            )
            c_boost_next = torch.tensor(cb.get("boost_combo_exec_next", cb["boost_combo_exec"]), dtype=torch.long, device=self.device)
            c_mode_next = torch.tensor(
                cb.get("mode_exec_next", cb.get("mode_exec", np.mod(cb.get("upper_idx_exec", np.zeros(c_batch_size)), 3))),
                dtype=torch.long,
                device=self.device,
            )
            c_physical = self._physical_torch(cb, "physical_features", c_batch_size)
            c_physical_next = self._physical_torch(cb, "physical_features_next", c_batch_size)
            c_obs_aug_q1 = self._augment_torch(c_obs, c_boost, c_mode, c_physical, self.q1_phys)
            c_next_obs_aug_actor = self._augment_torch(c_next_obs, c_boost_next, c_mode_next, c_physical_next, self.actor_phys)
            c_next_obs_aug_q1_tgt = self._augment_torch(c_next_obs, c_boost_next, c_mode_next, c_physical_next, self.q1_tgt_phys)
            c_next_temps = torch.tensor(cb["next_temps"], dtype=torch.float32, device=self.device)
            c_amb = torch.tensor(cb["amb_temp"], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                c_raw_next, _ = self.actor.sample(c_next_obs_aug_actor, c_z_next)
                c_safe_next = self.safety.project_torch(c_raw_next, c_boost_next, c_mode_next, c_next_temps, c_amb)
                c_a_next = torch.cat(
                    [c_safe_next["currents_exec"], c_safe_next["rho_exec"], c_safe_next["tau_exec"]],
                    dim=1,
                )
                c_constraint_next = self.constraint_q_tgt(c_next_obs_aug_q1_tgt, c_z_next, c_a_next)
                if self.constraint_target_nonnegative:
                    c_constraint_next = torch.relu(c_constraint_next)
                constraint_target = c_cost_vec + self.gamma_rl * (1.0 - c_done) * c_constraint_next
            constraint_pred = self.constraint_q(c_obs_aug_q1, c_z, c_act_exec)
            per_item = F.mse_loss(constraint_pred, constraint_target, reduction="none").mean(dim=1)
            weights = torch.tensor(
                cb.get("constraint_replay_importance_weight", np.ones(c_batch_size, dtype=np.float32)),
                dtype=torch.float32,
                device=self.device,
            ).view(-1)
            constraint_loss = (per_item * weights).sum() / torch.clamp(weights.sum(), min=1.0)
            bucket_id = torch.tensor(
                cb.get("constraint_replay_bucket_id", np.zeros(c_batch_size, dtype=np.float32)),
                dtype=torch.long,
                device=self.device,
            ).view(-1)

            def _bucket_loss(bucket: int) -> float:
                mask = bucket_id == int(bucket)
                if not bool(mask.any().item()):
                    return 0.0
                return float(per_item[mask].detach().mean().item())

            constraint_batch_stats = {
                "constraint_replay_weight_mean": float(weights.detach().mean().item()),
                "constraint_replay_batch_size": float(c_batch_size),
                "constraint_critic_loss_uniform": _bucket_loss(0),
                "constraint_critic_loss_boundary": _bucket_loss(1),
                "constraint_critic_loss_violation": _bucket_loss(2),
            }
        elif self.constraint_q is not None and constraint_target is not None:
            constraint_pred = self.constraint_q(obs_aug_q1, z, act_exec)
            constraint_loss = F.mse_loss(constraint_pred, constraint_target)
            loss_q = loss_q + constraint_loss
        if self.constraint_q is not None and constraint_batch is not None:
            loss_q = loss_q + constraint_loss

        self.critic_optim.zero_grad()
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for group in self.critic_optim.param_groups for p in group["params"]],
            self.grad_clip,
        )
        self.critic_optim.step()

        raw_pi, logp = self.actor.sample(obs_aug_actor, z)
        safe_pi = self.safety.project_torch(raw_pi, boost, mode, temps, amb)
        a_pi = torch.cat([safe_pi["currents_exec"], safe_pi["rho_exec"], safe_pi["tau_exec"]], dim=1)
        obs_aug_q1_pi = self._augment_torch(obs, boost, mode, physical, self.q1_phys)
        obs_aug_q2_pi = self._augment_torch(obs, boost, mode, physical, self.q2_phys)
        q_pi = torch.min(self.q1(obs_aug_q1_pi, z, a_pi), self.q2(obs_aug_q2_pi, z, a_pi))
        constraint_pi_weighted = torch.zeros_like(q_pi)
        constraint_pi_mean = 0.0
        if self.constraint_q is not None:
            constraint_pi = self.constraint_q(obs_aug_q1_pi, z, a_pi)
            if self.constraint_actor_penalty_nonnegative:
                constraint_pi = torch.relu(constraint_pi)
            constraint_pi_weighted = (constraint_pi * self.constraint_actor_weights).sum(dim=1, keepdim=True)
            constraint_pi_mean = float(constraint_pi_weighted.detach().mean().item())
        loss_pi = (alpha_t * logp - q_pi + constraint_pi_weighted).mean()

        self.actor_optim.zero_grad()
        loss_pi.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for group in self.actor_optim.param_groups for p in group["params"]],
            self.grad_clip,
        )
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
        if self.q1_phys is not None and self.q1_tgt_phys is not None:
            self._soft_update(self.q1_phys, self.q1_tgt_phys)
        if self.q2_phys is not None and self.q2_tgt_phys is not None:
            self._soft_update(self.q2_phys, self.q2_tgt_phys)
        if self.constraint_q is not None and self.constraint_q_tgt is not None:
            self._soft_update(self.constraint_q, self.constraint_q_tgt)
        self.update_steps += 1

        return {
            "q1_mean": float(q1.mean().item()),
            "q2_mean": float(q2.mean().item()),
            "reward_target_is_raw": float(reward_key == "reward_raw"),
            "reward_target_mean": float(rew.detach().mean().item()),
            "critic_loss": float(loss_q.item()),
            "constraint_critic_loss": float(constraint_loss.detach().item()),
            "constraint_actor_penalty": float(constraint_pi_mean),
            "actor_loss": float(loss_pi.item()),
            "entropy": float((-logp).mean().item()),
            "alpha": float(alpha_t.detach().item()),
            "alpha_loss": float(alpha_loss_val),
            **constraint_batch_stats,
        }

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_tgt": self.q1_tgt.state_dict(),
            "q2_tgt": self.q2_tgt.state_dict(),
            "constraint_q": self.constraint_q.state_dict() if self.constraint_q is not None else None,
            "constraint_q_tgt": self.constraint_q_tgt.state_dict() if self.constraint_q_tgt is not None else None,
            "actor_phys": self.actor_phys.state_dict() if self.actor_phys is not None else None,
            "q1_phys": self.q1_phys.state_dict() if self.q1_phys is not None else None,
            "q2_phys": self.q2_phys.state_dict() if self.q2_phys is not None else None,
            "q1_tgt_phys": self.q1_tgt_phys.state_dict() if self.q1_tgt_phys is not None else None,
            "q2_tgt_phys": self.q2_tgt_phys.state_dict() if self.q2_tgt_phys is not None else None,
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
        if self.constraint_q is not None and state.get("constraint_q") is not None:
            self.constraint_q.load_state_dict(state["constraint_q"])
        if self.constraint_q_tgt is not None and state.get("constraint_q_tgt") is not None:
            self.constraint_q_tgt.load_state_dict(state["constraint_q_tgt"])
        if self.actor_phys is not None and state.get("actor_phys") is not None:
            self.actor_phys.load_state_dict(state["actor_phys"])
        if self.q1_phys is not None and state.get("q1_phys") is not None:
            self.q1_phys.load_state_dict(state["q1_phys"])
        if self.q2_phys is not None and state.get("q2_phys") is not None:
            self.q2_phys.load_state_dict(state["q2_phys"])
        if self.q1_tgt_phys is not None and state.get("q1_tgt_phys") is not None:
            self.q1_tgt_phys.load_state_dict(state["q1_tgt_phys"])
        if self.q2_tgt_phys is not None and state.get("q2_tgt_phys") is not None:
            self.q2_tgt_phys.load_state_dict(state["q2_tgt_phys"])
        self.actor_optim.load_state_dict(state["actor_optim"])
        self.critic_optim.load_state_dict(state["critic_optim"])
        self.update_steps = int(state.get("update_steps", 0))
        if self.auto_alpha and self.log_alpha is not None and state.get("log_alpha") is not None:
            with torch.no_grad():
                self.log_alpha.copy_(torch.tensor(float(state["log_alpha"]), device=self.device))
            if self.alpha_optim is not None and state.get("alpha_optim") is not None:
                self.alpha_optim.load_state_dict(state["alpha_optim"])
