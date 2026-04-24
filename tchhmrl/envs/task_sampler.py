from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from tchhmrl.envs.task_contract import (
    DEFAULT_ALIGNMENT_VERSION,
    DEFAULT_TASK_SUMMARY_VERSION,
    TaskParams,
    TaskSpec,
)


TASK_RANGE_KEYS = [
    "attenuation_c_range",
    "misalign_std_range",
    "amb_temp_range",
    "gamma_range",
    "delta_range",
]


def validate_site_bank(site_bank: Sequence[Dict] | None) -> List[str]:
    issues: List[str] = []
    if not site_bank:
        return issues

    seen_ids: set[int] = set()
    fingerprints: set[tuple] = set()
    for idx, site in enumerate(site_bank):
        site_id = site.get("site_id", None)
        if site_id is None:
            issues.append(f"site_bank[{idx}] missing site_id")
        else:
            site_id_int = int(site_id)
            if site_id_int in seen_ids:
                issues.append(f"site_bank has duplicate site_id={site_id_int}")
            seen_ids.add(site_id_int)

        distances = site.get("distances", None)
        if not isinstance(distances, (list, tuple)) or len(distances) != 3:
            issues.append(f"site_bank[{idx}] distances must have length 3")
        else:
            try:
                dist_fp = tuple(float(x) for x in distances)
            except (TypeError, ValueError):
                issues.append(f"site_bank[{idx}] distances must be numeric")
                dist_fp = ("nan",)

        site_fp = []
        for key in TASK_RANGE_KEYS:
            rng = site.get(key, None)
            if not isinstance(rng, (list, tuple)) or len(rng) != 2:
                issues.append(f"site_bank[{idx}] {key} must be [lo, hi]")
                site_fp.append((None, None))
                continue
            lo = float(rng[0])
            hi = float(rng[1])
            if hi < lo:
                issues.append(f"site_bank[{idx}] {key} has hi < lo")
            site_fp.append((lo, hi))
        if isinstance(distances, (list, tuple)) and len(distances) == 3:
            site_fp.append(tuple(float(x) for x in distances))
        fingerprints.add(tuple(site_fp))

    if len(site_bank) > 1 and len(fingerprints) == 1:
        issues.append("site_bank sites are all identical across ranges and distances")
    return issues


class TaskSampler:
    """Sample site-specific episode tasks with site-aware fallback behavior."""

    def __init__(self, sampler_cfg: Dict, seed: int = 0, task_defaults: Dict[str, Any] | None = None):
        self.cfg = sampler_cfg
        self.rng = np.random.default_rng(seed)
        self.task_defaults = dict(task_defaults or {})
        self.strict_site_bank = bool(self.cfg.get("strict_site_bank", False))
        site_bank = list(self.cfg.get("site_bank", []) or [])
        issues = validate_site_bank(site_bank)
        if self.strict_site_bank:
            if not site_bank:
                raise ValueError("strict_site_bank=true requires sampler.site_bank")
            if issues:
                raise ValueError(f"Invalid site_bank: {issues}")

    def _make_task(
        self,
        *,
        site_id: int,
        task_source: str,
        distances: Sequence[float],
        attenuation_c: float,
        misalign_std: float,
        amb_temp: float,
        gamma: float,
        delta: float,
    ) -> TaskSpec:
        return TaskSpec(
            site_id=int(site_id),
            task_source=str(task_source),
            distances=tuple(float(x) for x in distances[:3]),
            attenuation_c=float(attenuation_c),
            misalign_std=float(misalign_std),
            amb_temp=float(amb_temp),
            gamma=float(gamma),
            delta=float(delta),
            qos_min_rate=float(self.task_defaults.get("qos_min_rate", 0.0)),
            alignment_version=str(self.task_defaults.get("alignment_version", DEFAULT_ALIGNMENT_VERSION)),
            task_summary_version=str(
                self.task_defaults.get("task_summary_version", DEFAULT_TASK_SUMMARY_VERSION)
            ),
        )

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

    def _range_from_source(self, source: Dict, fallback: Dict, key: str) -> Tuple[float, float]:
        rng = source.get(f"{key}_range", fallback.get(f"{key}_range"))
        if rng is None:
            raise KeyError(f"Missing range for {key}")
        return float(rng[0]), float(rng[1])

    def _sample_site_cfg(self, bucket: int | None = None) -> Dict:
        site_bank = list(self.cfg.get("site_bank", []) or [])
        if not site_bank:
            if self.strict_site_bank:
                raise ValueError("strict_site_bank=true does not allow global_fallback task sampling")
            return {
                "site_id": -1,
                "distances": tuple(float(x) for x in self.cfg.get("default_distances", [5.0, 6.0, 6.5])),
                "task_source": "global_fallback",
            }
        n_sites = len(site_bank)
        if bucket is None:
            idx = int(self.rng.integers(0, n_sites))
        else:
            idx = int(np.clip(bucket, 0, n_sites - 1))
        site = dict(site_bank[idx])
        site["site_id"] = int(site.get("site_id", idx))
        site["distances"] = tuple(float(x) for x in site.get("distances", self.cfg.get("default_distances", [5.0, 6.0, 6.5])))
        site["task_source"] = "site_bank"
        return site

    def sample_task(self, bucket: int | None = None) -> TaskParams:
        site = self._sample_site_cfg(bucket)
        if site["task_source"] == "global_fallback":
            sample_fn = self._uniform if bucket is None else (lambda key: self._bucket_uniform(key, bucket))
            return self._make_task(
                site_id=int(site["site_id"]),
                task_source=str(site["task_source"]),
                distances=tuple(float(x) for x in site["distances"]),
                attenuation_c=sample_fn("attenuation_c"),
                misalign_std=sample_fn("misalign_std"),
                amb_temp=sample_fn("amb_temp"),
                gamma=sample_fn("gamma"),
                delta=sample_fn("delta"),
            )

        def sample_from_site(key: str) -> float:
            lo, hi = self._range_from_source(site, self.cfg, key)
            return float(self.rng.uniform(lo, hi))

        return self._make_task(
            site_id=int(site["site_id"]),
            task_source=str(site["task_source"]),
            distances=tuple(float(x) for x in site["distances"]),
            attenuation_c=sample_from_site("attenuation_c"),
            misalign_std=sample_from_site("misalign_std"),
            amb_temp=sample_from_site("amb_temp"),
            gamma=sample_from_site("gamma"),
            delta=sample_from_site("delta"),
        )

    def sample(self, n_tasks: int) -> List[TaskParams]:
        n_tasks = int(max(1, n_tasks))
        site_bank = list(self.cfg.get("site_bank", []) or [])
        if site_bank and bool(self.cfg.get("balanced_sampling", False)):
            tasks = [self.sample_task(bucket=(i % len(site_bank))) for i in range(n_tasks)]
            self.rng.shuffle(tasks)
            return tasks
        if bool(self.cfg.get("balanced_sampling", False)):
            tasks = [self.sample_task(bucket=(i % 3)) for i in range(n_tasks)]
            self.rng.shuffle(tasks)
            return tasks
        return [self.sample_task() for _ in range(n_tasks)]
