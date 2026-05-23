from __future__ import annotations

import copy

import numpy as np

from scripts.nonlinear_eh_robustness import calibrate_nonlinear_scale
from tchhmrl.envs.uw_slipt_env import MultiTxUwSliptEnv
from tchhmrl.utils.config import load_cfg


def test_linear_eh_model_matches_existing_mapping():
    cfg = load_cfg("configs/default.yaml")
    cfg["env"]["eh_model"] = "linear"
    env = MultiTxUwSliptEnv(cfg)

    diag = env._compute_eh_metric(0.123)

    assert diag["eh_model"] == "linear"
    assert np.isclose(diag["eh_metric"], 0.123, atol=1.0e-12)
    assert np.isclose(diag["eh_metric_linear_proxy"], 0.123, atol=1.0e-12)


def test_nonlinear_eh_is_near_zero_monotonic_and_bounded():
    cfg = copy.deepcopy(load_cfg("configs/default.yaml"))
    cfg["env"]["eh_model"] = "nonlinear"
    cfg["env"]["eh_nonlinear"] = {
        "type": "logistic_normalized",
        "e_max": 1.0,
        "a": 12.0,
        "b": 0.10,
        "scale_match": True,
        "scale": 2.0,
    }
    env = MultiTxUwSliptEnv(cfg)

    xs = np.asarray([0.0, 0.02, 0.10, 0.20, 0.50], dtype=np.float32)
    metrics = np.asarray([env._compute_eh_metric(float(x))["eh_metric"] for x in xs], dtype=np.float64)
    raw = np.asarray([env._compute_eh_metric(float(x))["eh_metric_raw_nonlinear"] for x in xs], dtype=np.float64)

    assert metrics[0] <= 1.0e-10
    assert np.all(np.diff(metrics) >= -1.0e-12)
    assert np.all(raw <= 1.0 + 1.0e-12)
    assert np.all(metrics <= 2.0 + 1.0e-12)


def test_nonlinear_eh_calibration_rejects_degenerate_raw_values():
    x = np.ones(20, dtype=np.float64) * 0.1
    raw = np.zeros(20, dtype=np.float64)

    try:
        calibrate_nonlinear_scale(x, raw)
    except ValueError as exc:
        assert "Degenerate EH calibration" in str(exc)
    else:
        raise AssertionError("Expected degenerate nonlinear EH calibration to fail")


def test_nonlinear_eh_calibration_scale_matches_linear_mean():
    x = np.linspace(0.01, 0.20, 50, dtype=np.float64)
    raw = np.linspace(0.02, 0.40, 50, dtype=np.float64)

    manifest = calibrate_nonlinear_scale(x, raw)

    assert manifest["sample_count"] == 50
    assert np.isclose(manifest["scale"], np.mean(x) / np.mean(raw), atol=1.0e-12)
