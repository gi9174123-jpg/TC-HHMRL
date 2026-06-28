from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F

from tchhmrl.models.networks import DiscreteQNetwork, DuelingDiscreteQNetwork
from tchhmrl.models.physical_encoder import PhysicalEncoder


class UpperDQN:
    def __init__(self, cfg: Dict, device: torch.device):
        agent_cfg = cfg["agent"]
        dqn_cfg = cfg["upper_dqn"]

        self.device = device
        self.obs_dim = int(agent_cfg["obs_dim"])
        self.z_dim = int(agent_cfg["z_dim"])
        self.n_actions = int(agent_cfg["n_upper_actions"])
        hidden = int(agent_cfg["hidden_dim"])
        phys_cfg = cfg.get("physical_context", {}) or {}
        self.physical_enabled = bool(phys_cfg.get("enabled", False))
        self.physical_dim = int(phys_cfg.get("input_dim", 18))
        self.physical_embedding_dim = int(phys_cfg.get("embedding_dim", 32)) if self.physical_enabled else 0
        self.obs_aug_dim = self.obs_dim + self.physical_embedding_dim
        physical_hidden = int(phys_cfg.get("hidden_dim", 64))

        self.gamma = float(agent_cfg["gamma"])
        self.grad_clip = float(dqn_cfg["grad_clip"])
        self.double_dqn = bool(dqn_cfg.get("double_dqn", False))
        self.dueling = bool(dqn_cfg.get("dueling", False))

        q_cls = DuelingDiscreteQNetwork if self.dueling else DiscreteQNetwork
        self.q_phys = (
            PhysicalEncoder(self.physical_dim, physical_hidden, self.physical_embedding_dim).to(device)
            if self.physical_enabled
            else None
        )
        self.q_tgt_phys = (
            PhysicalEncoder(self.physical_dim, physical_hidden, self.physical_embedding_dim).to(device)
            if self.physical_enabled
            else None
        )
        self.q = q_cls(self.obs_aug_dim, self.z_dim, self.n_actions, hidden).to(device)
        self.q_tgt = q_cls(self.obs_aug_dim, self.z_dim, self.n_actions, hidden).to(device)
        self.q_tgt.load_state_dict(self.q.state_dict())
        if self.q_phys is not None and self.q_tgt_phys is not None:
            self.q_tgt_phys.load_state_dict(self.q_phys.state_dict())

        params = list(self.q.parameters())
        if self.q_phys is not None:
            params += list(self.q_phys.parameters())
        self.optim = torch.optim.Adam(params, lr=float(dqn_cfg["lr"]))

        self.epsilon_start = float(dqn_cfg["epsilon_start"])
        self.epsilon_final = float(dqn_cfg["epsilon_final"])
        self.epsilon_decay_steps = int(dqn_cfg["epsilon_decay_steps"])
        self.target_update_every = int(dqn_cfg["target_update_every"])

        self.update_steps = 0

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

    def _augment_obs(self, obs: torch.Tensor, physical_features: torch.Tensor, encoder) -> torch.Tensor:
        if not self.physical_enabled:
            return obs
        assert encoder is not None
        return torch.cat([obs, encoder(physical_features)], dim=1)

    def _augment_np(self, obs: np.ndarray, physical_features: np.ndarray | None, encoder) -> torch.Tensor:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if not self.physical_enabled:
            return obs_t
        if physical_features is None:
            physical_features = np.zeros((self.physical_dim,), dtype=np.float32)
        phys = np.asarray(physical_features, dtype=np.float32).reshape(-1)
        if phys.size != self.physical_dim:
            padded = np.zeros((self.physical_dim,), dtype=np.float32)
            padded[: min(phys.size, self.physical_dim)] = phys[: self.physical_dim]
            phys = padded
        phys_t = torch.tensor(phys, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self._augment_obs(obs_t, phys_t, encoder)

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
        action_mask: np.ndarray | None = None,
        physical_features: np.ndarray | None = None,
    ) -> int:
        allowed = None
        if action_mask is not None:
            allowed = np.asarray(action_mask, dtype=bool).reshape(-1)
            if allowed.size != self.n_actions:
                padded = np.zeros(self.n_actions, dtype=bool)
                padded[: min(self.n_actions, allowed.size)] = allowed[: self.n_actions]
                allowed = padded
            if not np.any(allowed):
                allowed = np.ones(self.n_actions, dtype=bool)
        if (not eval_mode) and np.random.rand() < self.epsilon(t):
            if allowed is not None:
                choices = np.flatnonzero(allowed)
                return int(np.random.choice(choices))
            return int(np.random.randint(0, self.n_actions))

        with torch.no_grad():
            obs_t = self._augment_np(obs, physical_features, self.q_phys)
            z_t = torch.tensor(z, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q(obs_t, z_t)
            if exec_map is None:
                if allowed is not None:
                    mask_t = torch.as_tensor(allowed, dtype=torch.bool, device=self.device).view(1, -1)
                    q = torch.where(mask_t, q, torch.full_like(q, -torch.inf))
                return int(torch.argmax(q, dim=1).item())
            exec_idx = torch.as_tensor(np.asarray(exec_map, dtype=np.int64), device=self.device).view(1, -1)
            exec_idx = torch.clamp(exec_idx, 0, self.n_actions - 1)
            q_raw = torch.gather(q, 1, exec_idx)
            if allowed is not None:
                mask_t = torch.as_tensor(allowed, dtype=torch.bool, device=self.device).view(1, -1)
                q_raw = torch.where(mask_t, q_raw, torch.full_like(q_raw, -torch.inf))
            return int(torch.argmax(q_raw, dim=1).item())

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        obs = torch.tensor(batch["obs"], dtype=torch.float32, device=self.device)
        batch_size = int(obs.shape[0])
        z = torch.tensor(batch["z"], dtype=torch.float32, device=self.device)
        act = torch.tensor(
            batch.get(
                "upper_idx_train",
                batch.get("upper_idx_exec", batch.get("upper_idx", np.zeros_like(batch["reward"]))),
            ),
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

        physical = self._physical_torch(batch, "physical_features", batch_size)
        physical_next = self._physical_torch(batch, "physical_features_next", batch_size)
        obs_aug = self._augment_obs(obs, physical, self.q_phys)
        next_obs_aug_online = self._augment_obs(next_obs, physical_next, self.q_phys)
        next_obs_aug_tgt = self._augment_obs(next_obs, physical_next, self.q_tgt_phys)

        q_all = self.q(obs_aug, z)
        q_a = torch.gather(q_all, 1, act.view(-1, 1))

        with torch.no_grad():
            if "next_exec_map" in batch:
                next_exec_map = torch.tensor(batch["next_exec_map"], dtype=torch.long, device=self.device)
                next_exec_map = torch.clamp(next_exec_map, 0, self.n_actions - 1)
                next_mask = None
                if "next_action_mask" in batch:
                    next_mask = torch.tensor(batch["next_action_mask"], dtype=torch.bool, device=self.device)
                    if next_mask.dim() == 1:
                        next_mask = next_mask.view(1, -1).expand(next_exec_map.shape[0], -1)
                    if next_mask.shape != next_exec_map.shape:
                        fixed = torch.ones_like(next_exec_map, dtype=torch.bool)
                        n = min(fixed.shape[1], next_mask.shape[1])
                        fixed[:, :n] = next_mask[:, :n]
                        next_mask = fixed
                    empty_rows = ~next_mask.any(dim=1, keepdim=True)
                    if bool(empty_rows.any().item()):
                        next_mask = torch.where(empty_rows.expand_as(next_mask), torch.ones_like(next_mask), next_mask)
                if self.double_dqn:
                    q_next_online = torch.gather(self.q(next_obs_aug_online, z_next), 1, next_exec_map)
                    if next_mask is not None:
                        q_next_online = torch.where(next_mask, q_next_online, torch.full_like(q_next_online, -torch.inf))
                    next_raw_action = torch.argmax(q_next_online, dim=1, keepdim=True)
                    q_next_target_raw = torch.gather(self.q_tgt(next_obs_aug_tgt, z_next), 1, next_exec_map)
                    if next_mask is not None:
                        q_next_target_raw = torch.where(
                            next_mask,
                            q_next_target_raw,
                            torch.full_like(q_next_target_raw, -torch.inf),
                        )
                    q_next = torch.gather(q_next_target_raw, 1, next_raw_action)
                else:
                    q_next_all = self.q_tgt(next_obs_aug_tgt, z_next)
                    q_next_raw = torch.gather(q_next_all, 1, next_exec_map)
                    if next_mask is not None:
                        q_next_raw = torch.where(next_mask, q_next_raw, torch.full_like(q_next_raw, -torch.inf))
                    q_next = q_next_raw.max(dim=1, keepdim=True)[0]
            else:
                if self.double_dqn:
                    next_action = torch.argmax(self.q(next_obs_aug_online, z_next), dim=1, keepdim=True)
                    q_next = torch.gather(self.q_tgt(next_obs_aug_tgt, z_next), 1, next_action)
                else:
                    q_next = self.q_tgt(next_obs_aug_tgt, z_next).max(dim=1, keepdim=True)[0]
            discount = torch.pow(torch.full_like(horizon, self.gamma), horizon)
            td_target = rew + discount * (1.0 - done) * q_next

        loss = F.smooth_l1_loss(q_a, td_target)

        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for group in self.optim.param_groups for p in group["params"]], self.grad_clip)
        self.optim.step()

        self.update_steps += 1
        if self.update_steps % self.target_update_every == 0:
            self.q_tgt.load_state_dict(self.q.state_dict())
            if self.q_phys is not None and self.q_tgt_phys is not None:
                self.q_tgt_phys.load_state_dict(self.q_phys.state_dict())

        return {"q_loss": float(loss.item())}

    def state_dict(self):
        return {
            "q": self.q.state_dict(),
            "q_tgt": self.q_tgt.state_dict(),
            "q_phys": self.q_phys.state_dict() if self.q_phys is not None else None,
            "q_tgt_phys": self.q_tgt_phys.state_dict() if self.q_tgt_phys is not None else None,
            "optim": self.optim.state_dict(),
            "update_steps": self.update_steps,
            "double_dqn": self.double_dqn,
            "dueling": self.dueling,
        }

    def load_state_dict(self, state):
        self.q.load_state_dict(state["q"])
        self.q_tgt.load_state_dict(state["q_tgt"])
        if self.q_phys is not None and state.get("q_phys") is not None:
            self.q_phys.load_state_dict(state["q_phys"])
        if self.q_tgt_phys is not None and state.get("q_tgt_phys") is not None:
            self.q_tgt_phys.load_state_dict(state["q_tgt_phys"])
        self.optim.load_state_dict(state["optim"])
        self.update_steps = state.get("update_steps", 0)
