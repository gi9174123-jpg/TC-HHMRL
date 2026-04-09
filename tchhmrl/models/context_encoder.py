from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence


class ContextEncoder(nn.Module):
    """PEARL-style context encoder using a GRU and Gaussian latent variable."""

    def __init__(self, input_dim: int, hidden_dim: int, z_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.logvar = nn.Linear(hidden_dim, z_dim)

    def forward(self, seq: torch.Tensor):
        _, h = self.gru(seq)
        h = h[-1]
        mu = self.mu(h)
        logvar = torch.clamp(self.logvar(h), min=-8.0, max=8.0)
        return mu, logvar

    def infer(self, seq: torch.Tensor):
        mu, logvar = self.forward(seq)
        std = torch.exp(0.5 * logvar)
        q = Normal(mu, std)
        p = Normal(torch.zeros_like(mu), torch.ones_like(std))
        z = q.rsample()
        kl = kl_divergence(q, p).sum(dim=1).mean()
        return z, kl, mu, logvar
