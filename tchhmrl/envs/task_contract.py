from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence, Tuple

import numpy as np


DEFAULT_ALIGNMENT_VERSION = "teacher_model_v1"
DEFAULT_TASK_SUMMARY_VERSION = "site_v2"

TASK_SPEC_FIELDS = (
    "site_id",
    "task_source",
    "distances",
    "attenuation_c",
    "misalign_std",
    "amb_temp",
    "gamma",
    "delta",
    "qos_min_rate",
    "alignment_version",
    "task_summary_version",
)

TASK_SUMMARY_V2_FIELDS = (
    "attenuation_c",
    "misalign_std",
    "amb_temp",
    "gamma",
    "delta",
    "qos_min_rate",
    "distance_tx0",
    "distance_tx1",
    "distance_tx2",
)


@dataclass(frozen=True)
class TaskSpec:
    site_id: int
    task_source: str
    distances: Tuple[float, float, float]
    attenuation_c: float
    misalign_std: float
    amb_temp: float
    gamma: float
    delta: float
    qos_min_rate: float = 0.0
    alignment_version: str = DEFAULT_ALIGNMENT_VERSION
    task_summary_version: str = DEFAULT_TASK_SUMMARY_VERSION

    def to_env_overrides(self) -> dict[str, Any]:
        out = asdict(self)
        out["distances"] = [float(x) for x in self.distances]
        return out

    def to_task_dict(self, *, pre_alignment: bool = False) -> dict[str, Any]:
        return {
            "site_id": int(self.site_id),
            "task_source": str(self.task_source),
            "attenuation_c": float(self.attenuation_c),
            "misalign_std": float(self.misalign_std),
            "amb_temp": float(self.amb_temp),
            "amb_temp_env": float(self.amb_temp),
            "gamma": float(self.gamma),
            "delta": float(self.delta),
            "qos_min_rate": float(self.qos_min_rate),
            "distance_tx0": float(self.distances[0]),
            "distance_tx1": float(self.distances[1]),
            "distance_tx2": float(self.distances[2]),
            "alignment_version": str(self.alignment_version),
            "task_summary_version": str(self.task_summary_version),
            "pre_alignment": bool(pre_alignment),
        }

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        *,
        fallback_task_source: str = "global_fallback",
        fallback_alignment_version: str = DEFAULT_ALIGNMENT_VERSION,
        fallback_task_summary_version: str = DEFAULT_TASK_SUMMARY_VERSION,
    ) -> "TaskSpec":
        distances_raw = mapping.get(
            "distances",
            mapping.get(
                "distances_env",
                [
                    mapping.get("distance_tx0", mapping.get("distance_tx0_env", 0.0)),
                    mapping.get("distance_tx1", mapping.get("distance_tx1_env", 0.0)),
                    mapping.get("distance_tx2", mapping.get("distance_tx2_env", 0.0)),
                ],
            ),
        )
        distances = np.asarray(distances_raw, dtype=np.float32).reshape(-1)
        if distances.size < 3:
            distances = np.pad(distances, (0, 3 - distances.size))
        return cls(
            site_id=int(mapping.get("site_id", mapping.get("site_id_env", -1))),
            task_source=str(mapping.get("task_source", fallback_task_source)),
            distances=(float(distances[0]), float(distances[1]), float(distances[2])),
            attenuation_c=float(mapping.get("attenuation_c", mapping.get("attenuation_c_env", 0.0))),
            misalign_std=float(mapping.get("misalign_std", mapping.get("misalign_std_env", 0.0))),
            amb_temp=float(mapping.get("amb_temp_env", mapping.get("amb_temp", 0.0))),
            gamma=float(mapping.get("gamma", mapping.get("gamma_env", 0.0))),
            delta=float(mapping.get("delta", mapping.get("delta_env", 0.0))),
            qos_min_rate=float(mapping.get("qos_min_rate", mapping.get("qos_min_rate_env", 0.0))),
            alignment_version=str(mapping.get("alignment_version", fallback_alignment_version)),
            task_summary_version=str(mapping.get("task_summary_version", fallback_task_summary_version)),
        )


TaskParams = TaskSpec


def task_defaults_from_cfg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    env_cfg = cfg.get("env", {})
    alignment_cfg = cfg.get("alignment", {})
    return {
        "qos_min_rate": float(env_cfg.get("qos_min_rate", 0.0)),
        "alignment_version": str(alignment_cfg.get("alignment_version", DEFAULT_ALIGNMENT_VERSION)),
        "task_summary_version": str(alignment_cfg.get("task_summary_version", DEFAULT_TASK_SUMMARY_VERSION)),
    }


def build_task_summary_v2(task_like: TaskSpec | Mapping[str, Any]) -> np.ndarray:
    spec = task_like if isinstance(task_like, TaskSpec) else TaskSpec.from_mapping(task_like)
    return np.asarray(
        [
            float(spec.attenuation_c),
            float(spec.misalign_std),
            float(spec.amb_temp),
            float(spec.gamma),
            float(spec.delta),
            float(spec.qos_min_rate),
            float(spec.distances[0]),
            float(spec.distances[1]),
            float(spec.distances[2]),
        ],
        dtype=np.float32,
    )


def stable_task_payload(task_like: TaskSpec | Mapping[str, Any]) -> dict[str, Any]:
    spec = task_like if isinstance(task_like, TaskSpec) else TaskSpec.from_mapping(task_like)
    return {
        "site_id": int(spec.site_id),
        "task_source": str(spec.task_source),
        "distances": [float(spec.distances[0]), float(spec.distances[1]), float(spec.distances[2])],
        "attenuation_c": float(spec.attenuation_c),
        "misalign_std": float(spec.misalign_std),
        "amb_temp": float(spec.amb_temp),
        "gamma": float(spec.gamma),
        "delta": float(spec.delta),
        "qos_min_rate": float(spec.qos_min_rate),
        "alignment_version": str(spec.alignment_version),
        "task_summary_version": str(spec.task_summary_version),
    }


def task_batch_hash(tasks: Sequence[TaskSpec | Mapping[str, Any]]) -> str:
    payload = [stable_task_payload(task) for task in tasks]
    payload_sorted = sorted(
        payload,
        key=lambda row: (
            int(row["site_id"]),
            json.dumps(row["distances"], separators=(",", ":"), ensure_ascii=False),
            float(row["attenuation_c"]),
            float(row["misalign_std"]),
            float(row["amb_temp"]),
            float(row["gamma"]),
            float(row["delta"]),
            float(row["qos_min_rate"]),
        ),
    )
    blob = json.dumps(payload_sorted, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def is_formally_comparable_record(record: Mapping[str, Any]) -> bool:
    return (
        record.get("alignment_version") == DEFAULT_ALIGNMENT_VERSION
        and record.get("task_summary_version") == DEFAULT_TASK_SUMMARY_VERSION
        and bool(record.get("pre_alignment", True)) is False
    )


def filter_formally_comparable_records(
    records: Iterable[Mapping[str, Any]],
    *,
    strict: bool = False,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    bad = 0
    for record in records:
        if is_formally_comparable_record(record):
            kept.append(dict(record))
        else:
            bad += 1
    if strict and bad:
        raise ValueError(f"Found {bad} non-comparable pre-alignment records")
    return kept


def task_distribution_summary(cfg: Mapping[str, Any]) -> dict[str, Any]:
    sampler_cfg = cfg.get("sampler", {})
    site_bank = list(sampler_cfg.get("site_bank", []) or [])
    out: dict[str, Any] = {
        "task_source": "site_bank" if site_bank else "global_fallback",
        "strict_site_bank": bool(sampler_cfg.get("strict_site_bank", False)),
        "balanced_sampling": bool(sampler_cfg.get("balanced_sampling", False)),
        "site_count": len(site_bank),
    }
    if site_bank:
        out["site_ids"] = [int(site["site_id"]) for site in site_bank]
        out["sites"] = [
            {
                "site_id": int(site["site_id"]),
                "distances": [float(x) for x in site["distances"]],
                "attenuation_c_range": [float(x) for x in site["attenuation_c_range"]],
                "misalign_std_range": [float(x) for x in site["misalign_std_range"]],
                "amb_temp_range": [float(x) for x in site["amb_temp_range"]],
                "gamma_range": [float(x) for x in site["gamma_range"]],
                "delta_range": [float(x) for x in site["delta_range"]],
            }
            for site in site_bank
        ]
    return out
