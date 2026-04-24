from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: Sequence[int], out_dim: int):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiscreteQNetwork(nn.Module):
    def __init__(self, obs_dim: int, z_dim: int, n_actions: int, hidden_dim: int = 128):
        super().__init__()
        self.q = MLP(obs_dim + z_dim, [hidden_dim, hidden_dim], n_actions)

    def forward(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, z], dim=-1)
        return self.q(x)


class ContinuousQNetwork(nn.Module):
    def __init__(self, obs_dim: int, z_dim: int, act_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.q = MLP(obs_dim + z_dim + act_dim, [hidden_dim, hidden_dim], 1)

    def forward(self, obs: torch.Tensor, z: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, z, act], dim=-1)
        return self.q(x)


class GaussianTanhPolicy(nn.Module):
    def __init__(self, obs_dim: int, z_dim: int, act_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.backbone = MLP(obs_dim + z_dim, [hidden_dim, hidden_dim], hidden_dim)
        self.mu = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs: torch.Tensor, z: torch.Tensor):
        x = torch.cat([obs, z], dim=-1)
        h = F.relu(self.backbone(x))
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), min=-5.0, max=2.0)
        return mu, log_std

    def sample(self, obs: torch.Tensor, z: torch.Tensor):
        mu, log_std = self.forward(obs, z)
        std = log_std.exp()
        dist = Normal(mu, std)
        pre_tanh = dist.rsample()
        action = torch.tanh(pre_tanh)

        log_prob = dist.log_prob(pre_tanh) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic(self, obs: torch.Tensor, z: torch.Tensor):
        mu, _ = self.forward(obs, z)
        return torch.tanh(mu)


class DeterministicTanhPolicy(nn.Module):
    def __init__(self, obs_dim: int, z_dim: int, act_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.policy = MLP(obs_dim + z_dim, [hidden_dim, hidden_dim], act_dim)

    def forward(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, z], dim=-1)
        return torch.tanh(self.policy(x))
