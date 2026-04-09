from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


def _as_array(x: float | Sequence[float], n: int, default: float) -> np.ndarray:
    if isinstance(x, (list, tuple, np.ndarray)):
        arr = np.asarray(x, dtype=np.float32).reshape(-1)
        if arr.size == n:
            return arr.astype(np.float32)
        if arr.size == 1:
            return np.full(n, float(arr.item()), dtype=np.float32)
        raise ValueError(f"expected scalar or length-{n} sequence, got shape {arr.shape}")
    return np.full(n, float(x if x is not None else default), dtype=np.float32)


@dataclass
class DualLayer:
    """Vector Lagrangian dual updater for long-term QoS/thermal constraints."""

    names: tuple[str, ...]
    lrs: np.ndarray
    target_costs: np.ndarray
    max_lambdas: np.ndarray
    values: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=np.float32))

    @classmethod
    def from_meta_cfg(cls, meta_cfg: dict, n_tx: int) -> "DualLayer":
        names = tuple(meta_cfg.get("dual_names", ["qos"] + [f"temp_tx{i}" for i in range(n_tx)]))
        n = len(names)
        target_default = float(meta_cfg.get("dual_target_cost", 0.0))
        max_default = float(meta_cfg.get("dual_max_lambda", 100.0))
        lr_default = float(meta_cfg.get("dual_lr", 1.0e-3))
        target_costs = _as_array(meta_cfg.get("dual_target_costs", None), n, target_default)
        if "dual_target_costs" not in meta_cfg and n >= 2:
            target_costs = np.asarray([0.0] + [target_default] * (n - 1), dtype=np.float32)
        lrs = _as_array(meta_cfg.get("dual_lrs", None), n, lr_default)
        max_lambdas = _as_array(meta_cfg.get("dual_max_lambdas", None), n, max_default)
        values = np.zeros(n, dtype=np.float32)
        return cls(names=names, lrs=lrs, target_costs=target_costs, max_lambdas=max_lambdas, values=values)

    @property
    def n_constraints(self) -> int:
        return int(self.values.shape[0])

    def _coerce_cost(self, cost: float | Iterable[float]) -> np.ndarray:
        if np.isscalar(cost):
            arr = np.asarray([float(cost)], dtype=np.float32)
        else:
            arr = np.asarray(list(cost), dtype=np.float32).reshape(-1)
        if arr.size == self.n_constraints:
            return arr
        if arr.size == 1 and self.n_constraints > 1:
            out = np.zeros(self.n_constraints, dtype=np.float32)
            out[0] = float(arr.item())
            return out
        raise ValueError(f"expected scalar or length-{self.n_constraints} cost vector, got shape {arr.shape}")

    def update(self, mean_cost: float | Iterable[float]) -> float:
        cost_vec = self._coerce_cost(mean_cost)
        self.values = self.values + self.lrs * (cost_vec - self.target_costs)
        self.values = np.clip(self.values, 0.0, self.max_lambdas).astype(np.float32)
        return float(np.mean(self.values))

    def penalty(self, cost: float | Iterable[float]) -> float:
        cost_vec = self._coerce_cost(cost)
        return float(np.dot(self.values, cost_vec))

    def as_dict(self) -> dict[str, float]:
        return {f"lambda_{name}": float(val) for name, val in zip(self.names, self.values)}

    def target_dict(self) -> dict[str, float]:
        return {f"target_cost_{name}": float(val) for name, val in zip(self.names, self.target_costs)}

    def state_dict(self) -> dict:
        return {
            "names": list(self.names),
            "lrs": self.lrs.copy(),
            "target_costs": self.target_costs.copy(),
            "max_lambdas": self.max_lambdas.copy(),
            "values": self.values.copy(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.names = tuple(state["names"])
        self.lrs = np.asarray(state["lrs"], dtype=np.float32)
        self.target_costs = np.asarray(state["target_costs"], dtype=np.float32)
        self.max_lambdas = np.asarray(state["max_lambdas"], dtype=np.float32)
        self.values = np.asarray(state["values"], dtype=np.float32)
