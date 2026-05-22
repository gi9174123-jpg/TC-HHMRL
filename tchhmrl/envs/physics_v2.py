from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np


DEFAULT_PHYSICS_VERSION = "physics_v2"
DEFAULT_EH_MODEL = "logistic"
DEFAULT_THERMAL_MODEL = "coupled"
DEFAULT_SAFETY_PROJECTION_VERSION = "coupled_thermal_cap_v1"

DEFAULT_THERMAL_COUPLING_MATRIX = np.asarray(
    [
        [0.0, 0.015, 0.0075],
        [0.015, 0.0, 0.015],
        [0.0075, 0.015, 0.0],
    ],
    dtype=np.float32,
)


def _stable_hash(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def validate_coupling_matrix(matrix: Any, *, n_tx: int = 3, max_row_sum: float = 0.03) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float32)
    if mat.shape != (n_tx, n_tx):
        raise ValueError(f"thermal_coupling_matrix must have shape ({n_tx}, {n_tx}), got {mat.shape}")
    if not np.all(np.isfinite(mat)):
        raise ValueError("thermal_coupling_matrix contains non-finite values")
    if np.any(mat < -1.0e-8):
        raise ValueError("thermal_coupling_matrix must be non-negative")
    if np.max(np.abs(np.diag(mat))) > 1.0e-8:
        raise ValueError("thermal_coupling_matrix diagonal must be zero")
    row_sums = np.sum(mat, axis=1)
    if np.any(row_sums > max_row_sum + 1.0e-8):
        raise ValueError(
            f"thermal_coupling_matrix row sums must be <= {max_row_sum}, got {row_sums.tolist()}"
        )
    return mat.astype(np.float32)


def coupling_matrix_hash(matrix: Any) -> str:
    mat = np.asarray(matrix, dtype=np.float32)
    payload = {"thermal_coupling_matrix": [[float(x) for x in row] for row in mat.tolist()]}
    return _stable_hash(payload)


def eh_calibration_hash(params: Mapping[str, Any]) -> str:
    payload = {
        "eh_model": str(params.get("eh_model", DEFAULT_EH_MODEL)),
        "M_EH": float(params.get("M_EH", params.get("eh_nl_M", 0.0))),
        "a_EH": float(params.get("a_EH", params.get("eh_nl_a", 0.0))),
        "b_EH": float(params.get("b_EH", params.get("eh_nl_b", 0.0))),
        "r_sat": float(params.get("r_sat", params.get("eh_calibration_r_sat", 0.85))),
    }
    optional_keys = [
        "physics_version",
        "calibration_scenarios",
        "seeds",
        "sample_count",
        "Q25",
        "Q90",
        "Q95",
        "commit_hash",
    ]
    for key in optional_keys:
        if key in params:
            payload[key] = params[key]
    return _stable_hash(payload)


def logistic_eh_metric(x: float | np.ndarray, *, M: float, a: float, b: float) -> float | np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    M = float(M)
    a = float(a)
    b = float(b)
    if M <= 0.0:
        out = np.zeros_like(x_arr, dtype=np.float64)
    else:
        omega = 1.0 / (1.0 + math.exp(a * b))
        sigmoid = 1.0 / (1.0 + np.exp(-a * (x_arr - b)))
        denom = max(1.0 - omega, 1.0e-12)
        out = M / denom * (sigmoid - omega)
        out = np.clip(out, 0.0, M)
    if np.asarray(x).shape == ():
        return float(out.reshape(-1)[0])
    return out.astype(np.float32)


def calibrate_logistic_eh_from_samples(
    samples: Any,
    *,
    r_sat: float = 0.85,
    eps: float = 1.0e-9,
) -> dict[str, float]:
    x = np.asarray(samples, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    x = x[x > eps]
    if x.size < 32:
        raise ValueError(f"EH calibration requires at least 32 positive finite samples, got {x.size}")
    q25 = float(np.quantile(x, 0.25))
    q90 = float(np.quantile(x, 0.90))
    q95 = float(np.quantile(x, 0.95))
    if q90 <= q25 + eps or q95 <= eps:
        raise ValueError(
            f"Degenerate EH calibration distribution: Q25={q25}, Q90={q90}, Q95={q95}"
        )
    b = q25
    M = q95
    # Fixed-point form of:
    #   Omega(a) = 1 / (1 + exp(a*b))
    #   s90(a) = Omega(a) + r_sat * (1 - Omega(a))
    #   a = logit(s90(a)) / (x90 - b)
    # This makes the calibration deterministic and avoids hand tuning.
    a = 1.0 / max(q90 - b, eps)
    for _ in range(64):
        omega = 1.0 / (1.0 + math.exp(min(max(a * b, -60.0), 60.0)))
        s90 = omega + float(r_sat) * (1.0 - omega)
        s90 = min(max(s90, eps), 1.0 - eps)
        a_next = math.log(s90 / (1.0 - s90)) / (q90 - b + eps)
        if abs(a_next - a) <= 1.0e-10 * max(1.0, abs(a)):
            a = a_next
            break
        a = a_next
    if not math.isfinite(a) or a <= 0.0:
        raise ValueError(f"Invalid calibrated EH slope a={a}")
    return {
        "eh_nl_M": float(M),
        "eh_nl_a": float(a),
        "eh_nl_b": float(b),
        "eh_calibration_q25": float(q25),
        "eh_calibration_q90": float(q90),
        "eh_calibration_q95": float(q95),
        "eh_calibration_r_sat": float(r_sat),
        "eh_calibration_sample_count": int(x.size),
    }


def thermal_coupling_term_np(
    temps: np.ndarray,
    matrix: np.ndarray,
    *,
    thermal_model: str,
) -> np.ndarray:
    temps = np.asarray(temps, dtype=np.float32)
    if str(thermal_model).lower() != "coupled":
        return np.zeros_like(temps, dtype=np.float32)
    mat = np.asarray(matrix, dtype=np.float32)
    diff = temps.reshape(1, -1) - temps.reshape(-1, 1)
    return np.sum(mat * diff, axis=1).astype(np.float32)
