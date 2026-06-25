from __future__ import annotations

import torch
import torch.nn as nn

from tchhmrl.models.networks import MLP


class PhysicalEncoder(nn.Module):
    """Small encoder for online physical safety/thermal features."""

    def __init__(self, input_dim: int = 18, hidden_dim: int = 64, embedding_dim: int = 32):
        super().__init__()
        self.input_dim = int(input_dim)
        self.embedding_dim = int(embedding_dim)
        self.net = MLP(self.input_dim, [int(hidden_dim)], self.embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x)
