from __future__ import annotations

import torch
import torch.nn as nn

from tchhmrl.models.networks import MLP


class ConstraintQNetwork(nn.Module):
    """Vector critic for discounted constraint-cost values."""

    def __init__(self, obs_dim: int, z_dim: int, act_dim: int, out_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.out_dim = int(out_dim)
        self.q = MLP(obs_dim + z_dim + act_dim, [hidden_dim, hidden_dim], self.out_dim)

    def forward(self, obs: torch.Tensor, z: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, z, act], dim=-1)
        return self.q(x)
