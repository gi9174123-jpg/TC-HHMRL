from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


TRANSITION_SCHEMA_VERSION = "transition_schema_v2_fail_fast"
REWARD_SCHEMA_VERSION = "reward_task_plus_benchmark_and_constraint_cost_vec_v2"
ACTION_CONTRACT_VERSION = "policy_planner_executed_action_v1"

STRICT_LOWER_REQUIRED_KEYS: tuple[str, ...] = (
    "reward_raw",
    "reward_task",
    "cost_vec",
    "act_exec",
    "physical_features",
    "physical_features_next",
)


def require_transition_keys(batch: Mapping[str, object], keys: Sequence[str] = STRICT_LOWER_REQUIRED_KEYS) -> None:
    missing = [str(k) for k in keys if k not in batch]
    if missing:
        raise KeyError(
            "Missing strict lower-transition field(s): "
            + ", ".join(missing)
            + f" required by {TRANSITION_SCHEMA_VERSION}. "
            "Do not silently fall back from reward_task/reward_raw to reward or from cost_vec to zeros."
        )


def validate_executed_action_shape(batch: Mapping[str, object], *, batch_size: int | None = None) -> None:
    if "act_exec" not in batch:
        raise KeyError("Missing act_exec required by executed-action critic contract")
    arr = np.asarray(batch["act_exec"], dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 5:
        raise ValueError(f"act_exec must have shape [B, 5], got {arr.shape}")
    if batch_size is not None and int(arr.shape[0]) != int(batch_size):
        raise ValueError(f"act_exec batch dimension {arr.shape[0]} does not match expected {batch_size}")
