from __future__ import annotations

import copy
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

    def state_dict(self) -> Dict:
        return {
            "capacity": int(self.capacity),
            "items": copy.deepcopy(list(self._buf)),
        }

    def load_state_dict(self, state: Dict) -> None:
        capacity = int(state.get("capacity", self.capacity))
        self.capacity = capacity
        self._buf = deque(copy.deepcopy(list(state.get("items", []))), maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self._buf)

    @staticmethod
    def _as_batch(batch: List[Dict]) -> Dict[str, np.ndarray]:
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

    @staticmethod
    def _constraint_flags(tr: Dict, thresholds: Dict[str, float | None]) -> tuple[bool, bool]:
        cost_vec = np.asarray(tr.get("cost_vec", [tr.get("cost", 0.0)]), dtype=np.float32).reshape(-1)
        cost = float(tr.get("cost", float(np.sum(np.maximum(cost_vec, 0.0)))))
        violation = bool(np.any(cost_vec > 0.0) or cost > float(thresholds.get("constraint_cost_threshold") or 1.0e-8))
        violation = violation or bool(float(tr.get("burst_event", tr.get("burst_active", 0.0))) > 0.0)

        boundary = False
        thermal_thr = thresholds.get("thermal_headroom_threshold")
        if thermal_thr is not None and "thermal_headroom" in tr:
            headroom = np.asarray(tr["thermal_headroom"], dtype=np.float32).reshape(-1)
            if headroom.size:
                boundary = boundary or bool(np.nanmin(headroom) < float(thermal_thr))
        residual_thr = thresholds.get("projection_residual_threshold")
        if residual_thr is not None and "projection_residual" in tr:
            residual = np.asarray(tr["projection_residual"], dtype=np.float32).reshape(-1)
            boundary = boundary or bool(np.linalg.norm(residual) > float(residual_thr))
        bus_thr = thresholds.get("bus_utilization_threshold")
        if bus_thr is not None:
            bus_util = tr.get("bus_utilization", None)
            if bus_util is None and "projected_current_total" in tr:
                bus_current_max = float(thresholds.get("bus_current_max") or 1.0)
                bus_util = float(tr["projected_current_total"]) / max(bus_current_max, 1.0e-6)
            if bus_util is not None:
                boundary = boundary or bool(float(bus_util) > float(bus_thr))
        qos_thr = thresholds.get("qos_margin_threshold")
        if qos_thr is not None and "qos_margin" in tr:
            boundary = boundary or bool(float(tr["qos_margin"]) < float(qos_thr))
        slope_thr = thresholds.get("temperature_slope_threshold")
        if slope_thr is not None and "temperature_slope" in tr:
            slope = np.asarray(tr["temperature_slope"], dtype=np.float32).reshape(-1)
            if slope.size:
                boundary = boundary or bool(np.nanmax(np.abs(slope)) > float(slope_thr))
        return violation, boundary

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

        return self._as_batch(batch)

    def sample_stratified_constraint(
        self,
        batch_size: int,
        *,
        uniform_fraction: float,
        boundary_fraction: float,
        violation_fraction: float,
        thresholds: Dict[str, float | None],
        importance_weighting: bool = True,
        importance_weight_clip: tuple[float, float] = (0.25, 4.0),
    ) -> tuple[Dict[str, np.ndarray], Dict[str, float]]:
        if len(self._buf) < batch_size:
            if len(self._buf) == 0:
                raise ValueError("cannot sample stratified constraint batch from an empty replay buffer")
            batch = random.choices(list(self._buf), k=batch_size)
            out = self._as_batch(batch)
            out["constraint_replay_importance_weight"] = np.ones((batch_size,), dtype=np.float32)
            out["constraint_replay_bucket_id"] = np.zeros((batch_size,), dtype=np.float32)
            return out, {
                "constraint_batch_uniform_count": float(batch_size),
                "constraint_batch_boundary_count": 0.0,
                "constraint_batch_violation_count": 0.0,
                "constraint_bucket_uniform_pool_count": float(len(self._buf)),
                "constraint_bucket_boundary_pool_count": 0.0,
                "constraint_bucket_violation_pool_count": 0.0,
                "constraint_bucket_total_count": float(len(self._buf)),
                "constraint_bucket_shortage_count": 0.0,
                "constraint_replay_importance_weight_min": 1.0,
                "constraint_replay_importance_weight_max": 1.0,
            }

        uniform_fraction = float(np.clip(uniform_fraction, 0.0, 1.0))
        boundary_fraction = float(np.clip(boundary_fraction, 0.0, 1.0))
        violation_fraction = float(np.clip(violation_fraction, 0.0, 1.0))
        frac_sum = max(uniform_fraction + boundary_fraction + violation_fraction, 1.0e-6)
        uniform_fraction /= frac_sum
        boundary_fraction /= frac_sum
        violation_fraction /= frac_sum

        n_violation = int(round(batch_size * violation_fraction))
        n_boundary = int(round(batch_size * boundary_fraction))
        n_uniform = max(0, batch_size - n_violation - n_boundary)

        buf_list = list(self._buf)
        violation_pool: list[Dict] = []
        boundary_pool: list[Dict] = []
        uniform_pool: list[Dict] = []
        for tr in buf_list:
            is_violation, is_boundary = self._constraint_flags(tr, thresholds)
            if is_violation:
                violation_pool.append(tr)
            elif is_boundary:
                boundary_pool.append(tr)
            else:
                uniform_pool.append(tr)
        pool_total_counts = {
            "uniform": float(len(uniform_pool)),
            "boundary": float(len(boundary_pool)),
            "violation": float(len(violation_pool)),
        }

        selected_pairs: list[tuple[Dict, str]] = []
        shortage = 0

        def draw(pool: list[Dict], n: int, label: str) -> None:
            nonlocal shortage
            take = min(max(0, n), len(pool))
            if take > 0:
                items = random.sample(pool, take)
                selected_pairs.extend((item, label) for item in items)
            shortage += max(0, n - take)

        draw(violation_pool, n_violation, "violation")
        used_ids = {id(x) for x, _ in selected_pairs}
        boundary_pool = [x for x in boundary_pool if id(x) not in used_ids]
        uniform_pool = [x for x in uniform_pool if id(x) not in used_ids]
        draw(boundary_pool, n_boundary, "boundary")
        used_ids = {id(x) for x, _ in selected_pairs}
        fallback_pool = [x for x in buf_list if id(x) not in used_ids]
        uniform_draw_pool = [x for x in uniform_pool if id(x) not in used_ids] or fallback_pool
        draw(uniform_draw_pool, n_uniform, "uniform")

        used_ids = {id(x) for x, _ in selected_pairs}
        while len(selected_pairs) < batch_size:
            fallback_pool = [x for x in buf_list if id(x) not in used_ids]
            if not fallback_pool:
                break
            item = random.choice(fallback_pool)
            selected_pairs.append((item, "uniform"))
            used_ids.add(id(item))
            shortage += 1

        selected_pairs = selected_pairs[:batch_size]
        random.shuffle(selected_pairs)
        selected = [item for item, _label in selected_pairs]
        selected_labels = [label for _item, label in selected_pairs]
        out = self._as_batch(selected)
        bucket_map = {"uniform": 0.0, "boundary": 1.0, "violation": 2.0}
        out["constraint_replay_bucket_id"] = np.asarray(
            [bucket_map.get(label, 0.0) for label in selected_labels],
            dtype=np.float32,
        )
        if importance_weighting:
            batch_counts = {
                "uniform": float(sum(1 for x in selected_labels if x == "uniform")),
                "boundary": float(sum(1 for x in selected_labels if x == "boundary")),
                "violation": float(sum(1 for x in selected_labels if x == "violation")),
            }
            total_count = float(len(buf_list))
            batch_count = float(max(len(selected_labels), 1))
            low_clip, high_clip = float(importance_weight_clip[0]), float(importance_weight_clip[1])
            weights = []
            for label in selected_labels:
                replay_prob = pool_total_counts.get(label, 0.0) / max(total_count, 1.0)
                sample_prob = batch_counts.get(label, 0.0) / batch_count
                weight = replay_prob / max(sample_prob, 1.0e-8)
                weights.append(float(np.clip(weight, low_clip, high_clip)))
            weights_arr = np.asarray(weights, dtype=np.float32)
            weights_arr = weights_arr / max(float(weights_arr.mean()), 1.0e-8)
            out["constraint_replay_importance_weight"] = weights_arr.astype(np.float32)
        else:
            out["constraint_replay_importance_weight"] = np.ones((len(selected),), dtype=np.float32)
        weights_out = np.asarray(out["constraint_replay_importance_weight"], dtype=np.float32)
        return out, {
            "constraint_batch_uniform_count": float(sum(1 for x in selected_labels if x == "uniform")),
            "constraint_batch_boundary_count": float(sum(1 for x in selected_labels if x == "boundary")),
            "constraint_batch_violation_count": float(sum(1 for x in selected_labels if x == "violation")),
            "constraint_bucket_uniform_pool_count": pool_total_counts["uniform"],
            "constraint_bucket_boundary_pool_count": pool_total_counts["boundary"],
            "constraint_bucket_violation_pool_count": pool_total_counts["violation"],
            "constraint_bucket_total_count": float(len(buf_list)),
            "constraint_bucket_shortage_count": float(shortage),
            "constraint_replay_importance_weight_min": float(weights_out.min()) if weights_out.size else 1.0,
            "constraint_replay_importance_weight_max": float(weights_out.max()) if weights_out.size else 1.0,
        }


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

    def state_dict(self) -> Dict:
        return {
            "max_len": int(self.max_len),
            "items": copy.deepcopy(self.data),
        }

    def load_state_dict(self, state: Dict) -> None:
        self.max_len = int(state.get("max_len", self.max_len))
        self.data = copy.deepcopy(list(state.get("items", [])))
        if len(self.data) > self.max_len:
            self.data = self.data[-self.max_len :]

    def __len__(self) -> int:
        return len(self.data)

    def as_list(self) -> List[Dict]:
        return list(self.data)
