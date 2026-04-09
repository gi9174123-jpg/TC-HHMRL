from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np


@dataclass
class TaskParams:
    attenuation_c: float
    misalign_std: float
    amb_temp: float
    gamma: float
    delta: float

    def to_env_overrides(self) -> Dict[str, float]:
        return asdict(self)


class TaskSampler:
    """Sample environment parameters to represent different water conditions."""

    def __init__(self, sampler_cfg: Dict, seed: int = 0):
        self.cfg = sampler_cfg
        self.rng = np.random.default_rng(seed)

    def _uniform(self, key: str) -> float:
        lo, hi = self.cfg[f"{key}_range"]
        return float(self.rng.uniform(lo, hi))

    def _bucket_uniform(self, key: str, bucket: int, n_buckets: int = 3) -> float:
        lo, hi = self.cfg[f"{key}_range"]
        lo = float(lo)
        hi = float(hi)
        if hi <= lo:
            return lo
        n_buckets = max(1, int(n_buckets))
        bucket = int(np.clip(bucket, 0, n_buckets - 1))
        width = (hi - lo) / float(n_buckets)
        b_lo = lo + bucket * width
        b_hi = lo + (bucket + 1) * width
        return float(self.rng.uniform(b_lo, b_hi))

    def sample_task(self, bucket: int | None = None) -> TaskParams:
        sample_fn = self._uniform if bucket is None else (lambda key: self._bucket_uniform(key, bucket))
        return TaskParams(
            attenuation_c=sample_fn("attenuation_c"),
            misalign_std=sample_fn("misalign_std"),
            amb_temp=sample_fn("amb_temp"),
            gamma=sample_fn("gamma"),
            delta=sample_fn("delta"),
        )

    def sample(self, n_tasks: int) -> List[TaskParams]:
        if bool(self.cfg.get("balanced_sampling", False)):
            tasks = [self.sample_task(bucket=(i % 3)) for i in range(n_tasks)]
            self.rng.shuffle(tasks)
            return tasks
        return [self.sample_task() for _ in range(n_tasks)]
