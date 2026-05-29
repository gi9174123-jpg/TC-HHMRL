from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tchhmrl.baselines.common import BasePaperBaseline, expected_step_metrics
from tchhmrl.buffers.replay_buffer import ReplayBuffer
from tchhmrl.envs.uw_slipt_env import MultiTxUwSliptEnv
from tchhmrl.models.networks import MLP


class PDQNParameterNetwork(nn.Module):
    def __init__(self, obs_dim: int, discrete_action_dim: int, continuous_parameter_dim: int, hidden_dim: int):
        super().__init__()
        self.discrete_action_dim = int(discrete_action_dim)
        self.continuous_parameter_dim = int(continuous_parameter_dim)
        self.net = MLP(obs_dim, [hidden_dim, hidden_dim], self.discrete_action_dim * self.continuous_parameter_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        out = torch.tanh(self.net(obs))
        return out.view(-1, self.discrete_action_dim, self.continuous_parameter_dim)


class PDQNQNetwork(nn.Module):
    def __init__(self, obs_dim: int, discrete_action_dim: int, continuous_parameter_dim: int, hidden_dim: int):
        super().__init__()
        self.discrete_action_dim = int(discrete_action_dim)
        self.q = MLP(obs_dim + discrete_action_dim + continuous_parameter_dim, [hidden_dim, hidden_dim], 1)

    def forward(self, obs: torch.Tensor, action_idx: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(action_idx.long().view(-1), num_classes=self.discrete_action_dim).float()
        x = torch.cat([obs, one_hot.to(obs.device), params], dim=-1)
        return self.q(x)

    def all_q(self, obs: torch.Tensor, params_all: torch.Tensor) -> torch.Tensor:
        q_vals = []
        for k in range(self.discrete_action_dim):
            idx = torch.full((obs.shape[0],), k, dtype=torch.long, device=obs.device)
            q_vals.append(self.forward(obs, idx, params_all[:, k, :]))
        return torch.cat(q_vals, dim=1)


class PDQNHybridActionBaseline(BasePaperBaseline):
    baseline_family = "pdqn_hybrid_action"

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        opts = cfg.get("baselines", {}).get("pdqn_hybrid_action", {})
        obs_dim = int(cfg["agent"]["obs_dim"])
        hidden_dim = int(opts.get("hidden_dim", cfg["agent"].get("hidden_dim", 128)))
        self.discrete_action_dim = int(opts.get("discrete_action_dim", 12))
        self.continuous_parameter_dim = int(opts.get("continuous_parameter_dim", 5))
        self.param_net = PDQNParameterNetwork(obs_dim, self.discrete_action_dim, self.continuous_parameter_dim, hidden_dim).to(self.device)
        self.param_tgt = PDQNParameterNetwork(obs_dim, self.discrete_action_dim, self.continuous_parameter_dim, hidden_dim).to(self.device)
        self.q_net = PDQNQNetwork(obs_dim, self.discrete_action_dim, self.continuous_parameter_dim, hidden_dim).to(self.device)
        self.q_tgt = PDQNQNetwork(obs_dim, self.discrete_action_dim, self.continuous_parameter_dim, hidden_dim).to(self.device)
        self.param_tgt.load_state_dict(self.param_net.state_dict())
        self.q_tgt.load_state_dict(self.q_net.state_dict())
        self.param_optim = torch.optim.Adam(self.param_net.parameters(), lr=float(opts.get("param_lr", 1.0e-4)))
        self.q_optim = torch.optim.Adam(self.q_net.parameters(), lr=float(opts.get("q_lr", 3.0e-4)))
        self.replay = ReplayBuffer(int(opts.get("replay_size", 100000)))
        self.batch_size = int(opts.get("batch_size", 64))
        self.warmup_steps = int(opts.get("warmup_steps", cfg["agent"].get("warmup_steps", 100)))
        self.gamma = float(opts.get("gamma", cfg["agent"].get("gamma", 0.99)))
        self.target_tau = float(opts.get("target_tau", 0.01))
        self.noise_std = float(opts.get("noise_std", 0.10))
        self.epsilon = float(opts.get("epsilon", 0.05))
        self.action_contract = "parameterized_discrete_continuous_q_learning"

    def select_parameterized_action(self, obs_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        params_all = self.param_net(obs_batch)
        q_all = self.q_net.all_q(obs_batch, params_all)
        action_idx = torch.argmax(q_all, dim=1)
        selected = params_all[torch.arange(obs_batch.shape[0], device=obs_batch.device), action_idx, :]
        return action_idx, selected, q_all

    def act(self, obs: np.ndarray, env: MultiTxUwSliptEnv, eval_mode: bool = False) -> tuple[Dict, Dict]:
        obs_t = torch.as_tensor(obs.astype(np.float32), dtype=torch.float32, device=self.device).view(1, -1)
        with torch.no_grad():
            action_idx, params, q_all = self.select_parameterized_action(obs_t)
            if (not eval_mode) and np.random.rand() < self.epsilon:
                action_idx = torch.as_tensor([np.random.randint(0, self.discrete_action_dim)], dtype=torch.long, device=self.device)
                params_all = self.param_net(obs_t)
                params = params_all[:, int(action_idx.item()), :]
            if not eval_mode and self.noise_std > 0.0:
                params = torch.clamp(params + self.noise_std * torch.randn_like(params), -1.0, 1.0)
        upper_raw = int(action_idx.item())
        lower_raw = params.squeeze(0).cpu().numpy().astype(np.float32)
        safe, _ = self._project_raw_action(env, upper_raw, lower_raw, commit=True)
        predicted = expected_step_metrics(env, safe)
        action, aux = self._action_from_safe(
            upper_raw,
            lower_raw,
            safe,
            aux_extra={
                "pdqn_selected_k": int(upper_raw),
                "pdqn_argmax_q": float(torch.max(q_all).item()),
                "predicted_qos_rate": float(predicted["qos_rate"]),
                "predicted_eh_metric": float(predicted["eh_metric"]),
                "predicted_snr": float(predicted["snr"]),
                "predicted_bus_utilization": float(predicted["bus_utilization"]),
                "selected_action_contract": self.action_contract,
            },
        )
        return action, aux

    def record_transition(self, obs, aux, reward, done, next_obs, info) -> None:
        del info
        self.replay.add(
            {
                "obs": obs.astype(np.float32),
                "next_obs": next_obs.astype(np.float32),
                "action_idx": float(aux["upper_idx_raw"]),
                "param": np.asarray(aux["act_raw"], dtype=np.float32),
                "reward": float(reward),
                "done": float(done),
            }
        )

    def _soft_update(self, src: nn.Module, dst: nn.Module) -> None:
        for p, p_tgt in zip(src.parameters(), dst.parameters()):
            p_tgt.data.mul_(1.0 - self.target_tau)
            p_tgt.data.add_(self.target_tau * p.data)

    def learn(self) -> Dict[str, float]:
        if len(self.replay) < max(self.batch_size, self.warmup_steps):
            return {}
        batch = self.replay.sample(self.batch_size)
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
        action_idx = torch.as_tensor(batch["action_idx"], dtype=torch.long, device=self.device)
        param = torch.as_tensor(batch["param"], dtype=torch.float32, device=self.device)
        rew = torch.as_tensor(batch["reward"], dtype=torch.float32, device=self.device).view(-1, 1)
        done = torch.as_tensor(batch["done"], dtype=torch.float32, device=self.device).view(-1, 1)

        with torch.no_grad():
            next_params = self.param_tgt(next_obs)
            next_q = self.q_tgt.all_q(next_obs, next_params).max(dim=1, keepdim=True)[0]
            target = rew + self.gamma * (1.0 - done) * next_q
        q = self.q_net(obs, action_idx, param)
        q_loss = F.smooth_l1_loss(q, target)
        self.q_optim.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 5.0)
        self.q_optim.step()

        for p in self.q_net.parameters():
            p.requires_grad_(False)
        try:
            params_all = self.param_net(obs)
            q_all = self.q_net.all_q(obs, params_all)
            param_loss = -q_all.max(dim=1)[0].mean()
            self.param_optim.zero_grad()
            param_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.param_net.parameters(), 5.0)
            self.param_optim.step()
        finally:
            for p in self.q_net.parameters():
                p.requires_grad_(True)
        self._soft_update(self.param_net, self.param_tgt)
        self._soft_update(self.q_net, self.q_tgt)
        return {"pdqn_q_loss": float(q_loss.item()), "pdqn_param_loss": float(param_loss.item())}

    def save(self, ckpt_path):
        torch.save(
            {
                "baseline_metadata": self.metadata,
                "param_net": self.param_net.state_dict(),
                "q_net": self.q_net.state_dict(),
            },
            ckpt_path,
        )

    def load(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.param_net.load_state_dict(ckpt["param_net"])
        self.param_tgt.load_state_dict(ckpt["param_net"])
        self.q_net.load_state_dict(ckpt["q_net"])
        self.q_tgt.load_state_dict(ckpt["q_net"])
