from __future__ import annotations

import copy

import numpy as np

from tchhmrl.safety.safety_layer import SafetyLayer
from tchhmrl.utils.config import load_cfg


def test_safety_projection_bounds():
    cfg = load_cfg("configs/default.yaml")
    safety = SafetyLayer(cfg)

    lower_raw = np.array([5.0, 5.0, 5.0, 0.0, 0.0], dtype=np.float32)
    temps = np.array([20.0, 20.0, 20.0], dtype=np.float32)
    out, _ = safety.project_np(
        upper_raw=11,
        lower_raw=lower_raw,
        temps=temps,
        amb_temp=26.0,
        gamma=0.04,
        delta=2.4,
        mem={"current_boost": 3, "dwell_count": 3},
    )

    currents = out["currents_exec"]
    assert np.all(currents >= 0.0)
    assert float(np.sum(currents)) <= cfg["safety"]["bus_current_max"] + 1e-5
    assert 0.0 <= out["rho_exec"] <= 1.0
    assert 0.0 <= out["tau_exec"] <= 1.0


def test_safety_high_temp_soft_suppression():
    cfg = load_cfg("configs/default.yaml")
    safety = SafetyLayer(cfg)

    lower_raw = np.array([5.0, 5.0, 5.0, 0.0, 0.0], dtype=np.float32)
    cool_temps = np.array([20.0, 20.0, 20.0], dtype=np.float32)
    hot_temps = np.array([200.0, 200.0, 200.0], dtype=np.float32)

    out_cool, _ = safety.project_np(
        upper_raw=10,
        lower_raw=lower_raw,
        temps=cool_temps,
        amb_temp=26.0,
        gamma=0.04,
        delta=2.4,
        mem={"current_boost": 3, "dwell_count": 3},
    )
    out_hot, _ = safety.project_np(
        upper_raw=10,
        lower_raw=lower_raw,
        temps=hot_temps,
        amb_temp=26.0,
        gamma=0.04,
        delta=2.4,
        mem={"current_boost": 3, "dwell_count": 3},
    )
    assert float(np.sum(out_hot["currents_exec"])) < float(np.sum(out_cool["currents_exec"]))
    assert np.all(out_hot["currents_exec"] >= 0.0)


def test_safety_ld_thermal_prediction_higher_than_led():
    cfg = load_cfg("configs/default.yaml")
    safety = SafetyLayer(cfg)

    lower_raw = np.array([4.0, 4.0, 4.0, 0.0, 0.0], dtype=np.float32)
    temps = np.array([20.0, 20.0, 20.0], dtype=np.float32)
    out, _ = safety.project_np(
        upper_raw=11,
        lower_raw=lower_raw,
        temps=temps,
        amb_temp=26.0,
        gamma=0.04,
        delta=2.4,
        mem={"current_boost": 3, "dwell_count": 3},
    )
    t_pred = out["t_pred"]
    # Default mapping: tx0=LED, tx1/tx2=LD, with thermal_ld_coeff > thermal_led_coeff.
    assert float(t_pred[1]) > float(t_pred[0])


def test_safety_respects_tx_enabled_hardware_mask():
    cfg = load_cfg("configs/default.yaml")
    cfg = copy.deepcopy(cfg)
    cfg["env"]["hybrid"]["tx_enabled"] = [1.0, 0.0, 0.0]
    safety = SafetyLayer(cfg)

    lower_raw = np.array([5.0, 5.0, 5.0, 0.0, 0.0], dtype=np.float32)
    temps = np.array([20.0, 20.0, 20.0], dtype=np.float32)
    out, _ = safety.project_np(
        upper_raw=11,
        lower_raw=lower_raw,
        temps=temps,
        amb_temp=26.0,
        gamma=0.04,
        delta=2.4,
        mem={"current_boost": 3, "dwell_count": 3},
    )
    assert float(out["currents_exec"][1]) == 0.0
    assert float(out["currents_exec"][2]) == 0.0


def test_safety_mode_aware_projection_for_rho_tau():
    cfg = load_cfg("configs/default.yaml")
    safety = SafetyLayer(cfg)
    lower_raw = np.array([0.0, 0.0, 0.0, 4.0, -4.0], dtype=np.float32)
    temps = np.array([20.0, 20.0, 20.0], dtype=np.float32)
    mem = {"current_boost": 0, "dwell_count": 3}

    out_ps, _ = safety.project_np(upper_raw=0, lower_raw=lower_raw, temps=temps, amb_temp=26.0, gamma=0.04, delta=2.4, mem=mem)
    out_ts, _ = safety.project_np(upper_raw=1, lower_raw=lower_raw, temps=temps, amb_temp=26.0, gamma=0.04, delta=2.4, mem=mem)
    out_hy, _ = safety.project_np(upper_raw=2, lower_raw=lower_raw, temps=temps, amb_temp=26.0, gamma=0.04, delta=2.4, mem=mem)

    assert float(out_ps["tau_exec"]) == 1.0
    assert float(out_ts["rho_exec"]) == 0.0
    assert 0.0 < float(out_hy["rho_exec"]) < 1.0
    assert 0.0 < float(out_hy["tau_exec"]) < 1.0


def test_safety_hard_clip_zeroes_currents_above_thermal_safe():
    cfg = load_cfg("configs/default.yaml")
    cfg = copy.deepcopy(cfg)
    cfg["safety"]["projection_mode"] = "hard_clip"
    safety = SafetyLayer(cfg)
    lower_raw = np.array([5.0, 5.0, 5.0, 0.0, 0.0], dtype=np.float32)
    hot_temps = np.array([200.0, 200.0, 200.0], dtype=np.float32)

    out_hot, _ = safety.project_np(
        upper_raw=11,
        lower_raw=lower_raw,
        temps=hot_temps,
        amb_temp=26.0,
        gamma=0.04,
        delta=2.4,
        mem={"current_boost": 3, "dwell_count": 3},
    )
    assert float(np.sum(out_hot["currents_exec"])) == 0.0


def test_safety_raw_to_exec_map_matches_preview():
    cfg = load_cfg("configs/default.yaml")
    safety = SafetyLayer(cfg)
    mem = {"current_boost": 0, "dwell_count": 0}

    exec_map = safety.raw_to_exec_map(mem)

    assert exec_map.shape == (12,)
    for raw_idx in range(12):
        boost_exec, mode_exec = safety.preview_exec(raw_idx, mem=mem)
        assert int(exec_map[raw_idx]) == safety.encode_exec(boost_exec, mode_exec)
