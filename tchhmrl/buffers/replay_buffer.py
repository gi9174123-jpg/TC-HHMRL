from __future__ import annotations

import random
from collections import deque
from typing import Callable, Deque, Dict, List

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self._buf: Deque[Dict] = deque(maxlen=self.capacity)

    def add(self, transition: Dict) -> None:
        self._buf.append(transition)

    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)

    def sample( 
        self,
        batch_size: int,
        hard_fraction: float = 0.0,
        scorer: Callable[[Dict], float] | None = None,
    ) -> Dict[str, np.ndarray]:
        if scorer is None or hard_fraction <= 0.0 or len(self._buf) < batch_size + 1:
            batch = random.sample(self._buf, batch_size)
        else:
            hard_fraction = float(np.clip(hard_fraction, 0.0, 1.0))
            n_hard = int(round(batch_size * hard_fraction))
            n_hard = int(np.clip(n_hard, 0, batch_size))

            buf_list = list(self._buf)
            scores = np.asarray([float(scorer(tr)) for tr in buf_list], dtype=np.float32)

            # Use top-30% as hard candidate pool for lightweight hard mining.
            top_k = max(batch_size, int(0.30 * len(buf_list)))
            top_idx = np.argsort(scores)[-top_k:]
            hard_pool = [buf_list[i] for i in top_idx]

            hard_batch = random.sample(hard_pool, min(n_hard, len(hard_pool)))
            n_rest = batch_size - len(hard_batch)

            used = {id(x) for x in hard_batch}
            rest_pool = [x for x in buf_list if id(x) not in used]
            rest_batch = random.sample(rest_pool, n_rest) if n_rest > 0 else []
            batch = hard_batch + rest_batch
            random.shuffle(batch)

        keys = batch[0].keys()
        out: Dict[str, np.ndarray] = {}
        for k in keys:
            vals = [b[k] for b in batch]
            first = vals[0]
            if np.isscalar(first):
                out[k] = np.asarray(vals, dtype=np.float32)
            else:
                out[k] = np.asarray(vals, dtype=np.float32)
        return out


class EpisodeBuffer:
    def __init__(self, max_len: int):
        self.max_len = int(max_len)
        self.data: List[Dict] = []

    def add(self, item: Dict) -> None:
        self.data.append(item)
        if len(self.data) > self.max_len:
            self.data.pop(0)

    def clear(self) -> None:
        self.data.clear()

    def __len__(self) -> int:
        return len(self.data)

    def as_list(self) -> List[Dict]:
        return list(self.data)
