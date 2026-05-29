from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from tchhmrl.baselines.common import BasePaperBaseline
from tchhmrl.envs.uw_slipt_env import MultiTxUwSliptEnv
from tchhmrl.models.networks import MLP


class JavadiActorCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, continuous_dim: int = 3, n_sources: int = 4):
        super().__init__()
        self.backbone = MLP(obs_dim, [hidden_dim, hidden_dim], hidden_dim)
        self.source_logits = nn.Linear(hidden_dim, n_sources)
        self.mu = nn.Linear(hidden_dim, continuous_dim)
        self.log_std = nn.Parameter(torch.full((continuous_dim,), -0.6))
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor):
        h = F.relu(self.backbone(obs))
        logits = self.source_logits(h)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std, -5.0, 2.0).expand_as(mu)
        value = self.value(h)
        return logits, mu, log_std, value

    def sample(self, obs: torch.Tensor, deterministic: bool = False):
        logits, mu, log_std, value = self.forward(obs)
        source_dist = Categorical(logits=logits)
        if deterministic:
            source = torch.argmax(logits, dim=-1)
            pre_tanh = mu
        else:
            source = source_dist.sample()
            pre_tanh = Normal(mu, log_std.exp()).rsample()
        cont = torch.tanh(pre_tanh)
        normal = Normal(mu, log_std.exp())
        cont_logp = normal.log_prob(pre_tanh) - torch.log(1.0 - cont.pow(2) + 1.0e-6)
        logp = source_dist.log_prob(source).view(-1, 1) + cont_logp.sum(dim=-1, keepdim=True)
        return source, cont, logp, value

    def evaluate_actions(self, obs: torch.Tensor, source: torch.Tensor, cont: torch.Tensor):
        logits, mu, log_std, value = self.forward(obs)
        source_dist = Categorical(logits=logits)
        clipped = torch.clamp(cont, -0.999, 0.999)
        pre_tanh = 0.5 * torch.log((1.0 + clipped) / (1.0 - clipped))
        normal = Normal(mu, log_std.exp())
        cont_logp = normal.log_prob(pre_tanh) - torch.log(1.0 - clipped.pow(2) + 1.0e-6)
        logp = source_dist.log_prob(source.long()).view(-1, 1) + cont_logp.sum(dim=-1, keepdim=True)
        entropy = source_dist.entropy().mean() + normal.entropy().sum(dim=-1).mean()
        return logp, entropy, value


class JavadiPPODimmingBaseline(BasePaperBaseline):
    """OWC-SLIPT active-source and joint-dimming PPO adapted baseline."""

    baseline_family = "javadi_ppo_dimming"

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        opts = cfg.get("baselines", {}).get("javadi_ppo_dimming", {})
        obs_dim = int(cfg["agent"]["obs_dim"])
        hidden_dim = int(opts.get("hidden_dim", cfg["agent"].get("hidden_dim", 128)))
        self.dimming_type = str(opts.get("dimming_type", "common_dimming_scale"))
        self.policy = JavadiActorCritic(obs_dim, hidden_dim, continuous_dim=3).to(self.device)
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=float(opts.get("lr", 3.0e-4)))
        self.clip_ratio = float(opts.get("clip_ratio", 0.2))
        self.ppo_epochs = int(opts.get("ppo_epochs", 2))
        self.entropy_coef = float(opts.get("entropy_coef", 0.005))
        self.value_coef = float(opts.get("value_coef", 0.5))
        self.gamma = float(opts.get("gamma", cfg["agent"].get("gamma", 0.99)))
        self.buffer: List[Dict[str, object]] = []
        self.action_contract = "ppo_active_source_common_dimming_hy"

    def act(self, obs: np.ndarray, env: MultiTxUwSliptEnv, eval_mode: bool = False) -> tuple[Dict, Dict]:
        obs_t = torch.as_tensor(obs.astype(np.float32), dtype=torch.float32, device=self.device).view(1, -1)
        with torch.no_grad():
            source, cont, logp, value = self.policy.sample(obs_t, deterministic=eval_mode)
        boost_combo = int(source.item())
        upper_raw = boost_combo * 3 + 2
        policy_raw = cont.squeeze(0).cpu().numpy().astype(np.float32)
        dimming_raw = float(policy_raw[0])
        lower_raw = np.asarray(
            [dimming_raw, dimming_raw, dimming_raw, float(policy_raw[1]), float(policy_raw[2])],
            dtype=np.float32,
        )
        safe, _ = self._project_raw_action(env, upper_raw, lower_raw, commit=True)
        dimming_scale = float((dimming_raw + 1.0) * 0.5)
        action, aux = self._action_from_safe(
            upper_raw,
            lower_raw,
            safe,
            aux_extra={
                "source_subset_id": int(boost_combo),
                "active_source_number": int(np.sum(env._boost_mask(boost_combo) > 0.5)),
                "joint_dimming_scale": dimming_scale,
                "joint_dimming_scale_tx0": dimming_scale,
                "joint_dimming_scale_tx1": dimming_scale,
                "joint_dimming_scale_tx2": dimming_scale,
                "dimming_type": self.dimming_type,
                "ppo_cont_raw": policy_raw,
                "ppo_log_prob": float(logp.item()),
                "ppo_value": float(value.item()),
                "selected_action_contract": self.action_contract,
            },
        )
        return action, aux

    def record_transition(
        self,
        obs: np.ndarray,
        aux: Dict[str, object],
        reward: float,
        done: bool,
        next_obs: np.ndarray,
        info: Dict[str, object],
    ) -> None:
        del next_obs, info
        self.buffer.append(
            {
                "obs": obs.astype(np.float32),
                "source": int(aux["upper_idx_raw"]) // 3,
                "cont": np.asarray(aux["ppo_cont_raw"], dtype=np.float32),
                "logp": float(aux.get("ppo_log_prob", 0.0)),
                "value": float(aux.get("ppo_value", 0.0)),
                "reward": float(reward),
                "done": float(done),
            }
        )

    def after_training_iteration(self) -> Dict[str, float]:
        if not self.buffer:
            return {}
        rewards = np.asarray([b["reward"] for b in self.buffer], dtype=np.float32)
        dones = np.asarray([b["done"] for b in self.buffer], dtype=np.float32)
        returns = np.zeros_like(rewards)
        running = 0.0
        for i in reversed(range(len(rewards))):
            running = float(rewards[i]) + self.gamma * running * (1.0 - float(dones[i]))
            returns[i] = running
        values = np.asarray([b["value"] for b in self.buffer], dtype=np.float32)
        adv = returns - values
        adv = (adv - adv.mean()) / (adv.std() + 1.0e-6)

        obs = torch.as_tensor(np.stack([b["obs"] for b in self.buffer]), dtype=torch.float32, device=self.device)
        source = torch.as_tensor([b["source"] for b in self.buffer], dtype=torch.long, device=self.device)
        cont = torch.as_tensor(np.stack([b["cont"] for b in self.buffer]), dtype=torch.float32, device=self.device)
        old_logp = torch.as_tensor([b["logp"] for b in self.buffer], dtype=torch.float32, device=self.device).view(-1, 1)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device).view(-1, 1)
        adv_t = torch.as_tensor(adv, dtype=torch.float32, device=self.device).view(-1, 1)

        last_loss = 0.0
        for _ in range(max(1, self.ppo_epochs)):
            logp, entropy, value = self.policy.evaluate_actions(obs, source, cont)
            ratio = torch.exp(logp - old_logp)
            pg1 = ratio * adv_t
            pg2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_t
            policy_loss = -torch.min(pg1, pg2).mean()
            value_loss = F.mse_loss(value, ret_t)
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            self.optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
            self.optim.step()
            last_loss = float(loss.item())
        self.buffer.clear()
        return {"ppo_loss": last_loss}

    def save(self, ckpt_path):
        torch.save(
            {
                "baseline_metadata": self.metadata,
                "policy": self.policy.state_dict(),
                "optim": self.optim.state_dict(),
            },
            ckpt_path,
        )

    def load(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        if "optim" in ckpt:
            self.optim.load_state_dict(ckpt["optim"])
