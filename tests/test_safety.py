from __future__ import annotations

import copy

import numpy as np
import torch

from tchhmrl.safety.safety_layer import SafetyLayer, raw_from_frac01
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


def test_action_decode_tanh_affine_maps_tanh_outputs_to_fractions():
    cfg = copy.deepcopy(load_cfg("configs/default.yaml"))
    cfg["safety"]["action_decode_mode"] = "tanh_affine"
    safety = SafetyLayer(cfg)

    raw = np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)
    decoded_np = safety._decode_frac_np(raw)
    decoded_torch = safety._decode_frac_torch(torch.tensor(raw)).detach().cpu().numpy()

    assert np.allclose(decoded_np, [0.0, 0.5, 1.0], atol=1e-7)
    assert np.allclose(decoded_torch, decoded_np, atol=1e-7)


def test_action_decode_sigmoid_logit_preserves_legacy_sigmoid_behavior():
    cfg = copy.deepcopy(load_cfg("configs/default.yaml"))
    cfg["safety"]["action_decode_mode"] = "sigmoid_logit"
    safety = SafetyLayer(cfg)

    raw = np.asarray([-1.0, 0.0, 1.0], dtype=np.float32)
    decoded = safety._decode_frac_np(raw)
    expected = 1.0 / (1.0 + np.exp(-raw))

    assert np.allclose(decoded, expected, atol=1e-7)


def test_raw_from_frac01_matches_decode_modes():
    frac = np.asarray([0.0, 0.5, 1.0], dtype=np.float32)

    tanh_raw = raw_from_frac01(frac, "tanh_affine")
    sigmoid_raw = raw_from_frac01(frac, "sigmoid_logit")

    assert np.allclose(tanh_raw, [-1.0, 0.0, 1.0], atol=1e-7)
    assert np.allclose(sigmoid_raw[1], 0.0, atol=1e-7)
    assert sigmoid_raw[0] < -9.0
    assert sigmoid_raw[2] > 9.0


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
    lower_raw = np.array([0.0, 0.0, 0.0, 0.4, -0.4], dtype=np.float32)
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


def test_safety_dalal_correction_respects_bus_and_thermal_safe():
    cfg = load_cfg("configs/default.yaml")
    cfg = copy.deepcopy(cfg)
    cfg["safety"]["projection_mode"] = "dalal_safe"
    safety = SafetyLayer(cfg)
    lower_raw = np.array([6.0, 6.0, 6.0, 0.0, 0.0], dtype=np.float32)
    temps = np.array([44.0, 44.0, 44.0], dtype=np.float32)

    out, _ = safety.project_np(
        upper_raw=11,
        lower_raw=lower_raw,
        temps=temps,
        amb_temp=26.0,
        gamma=0.04,
        delta=2.4,
        mem={"current_boost": 3, "dwell_count": 3},
    )
    assert float(np.sum(out["currents_exec"])) <= cfg["safety"]["bus_current_max"] + 1e-4
    assert float(np.max(out["t_pred"])) <= cfg["safety"]["thermal_safe"] + 1e-3


def test_safety_smooth_relaxed_keeps_smooth_unchanged_and_relaxes_cool_margin():
    cfg_smooth = copy.deepcopy(load_cfg("configs/default.yaml"))
    cfg_relaxed = copy.deepcopy(load_cfg("configs/default.yaml"))
    cfg_smooth["safety"]["projection_mode"] = "smooth"
    cfg_relaxed["safety"]["projection_mode"] = "smooth_relaxed"
    cfg_relaxed["safety"]["smooth_relaxed_margin_c"] = 1.0

    lower_raw = np.array([0.4, 0.4, 0.4, 0.0, 0.0], dtype=np.float32)
    temps = np.array([28.0, 28.0, 28.0], dtype=np.float32)
    kwargs = {
        "upper_raw": 11,
        "lower_raw": lower_raw,
        "temps": temps,
        "amb_temp": 30.0,
        "gamma": 0.04,
        "delta": 1.2,
        "mem": {"current_boost": 3, "dwell_count": 3},
    }

    smooth_out, _ = SafetyLayer(cfg_smooth).project_np(**kwargs)
    relaxed_out, _ = SafetyLayer(cfg_relaxed).project_np(**kwargs)

    assert float(np.min(relaxed_out["thermal_margin"])) >= 1.0
    assert np.allclose(relaxed_out["thermal_soft_scale"], 1.0, atol=1e-6)
    assert float(np.sum(relaxed_out["currents_exec"])) > float(np.sum(smooth_out["currents_exec"]))
    assert np.all(relaxed_out["thermal_scale"] >= smooth_out["thermal_scale"] - 1e-6)


def test_safety_smooth_relaxed_matches_smooth_above_safe_boundary():
    cfg_smooth = copy.deepcopy(load_cfg("configs/default.yaml"))
    cfg_relaxed = copy.deepcopy(load_cfg("configs/default.yaml"))
    cfg_smooth["safety"]["projection_mode"] = "smooth"
    cfg_relaxed["safety"]["projection_mode"] = "smooth_relaxed"
    cfg_relaxed["safety"]["smooth_relaxed_margin_c"] = 1.0

    lower_raw = np.array([2.0, 2.0, 2.0, 0.0, 0.0], dtype=np.float32)
    temps = np.array([60.0, 60.0, 60.0], dtype=np.float32)
    kwargs = {
        "upper_raw": 11,
        "lower_raw": lower_raw,
        "temps": temps,
        "amb_temp": 30.0,
        "gamma": 0.04,
        "delta": 1.2,
        "mem": {"current_boost": 3, "dwell_count": 3},
    }

    smooth_out, _ = SafetyLayer(cfg_smooth).project_np(**kwargs)
    relaxed_out, _ = SafetyLayer(cfg_relaxed).project_np(**kwargs)

    assert float(np.max(relaxed_out["thermal_margin"])) <= 0.0
    assert np.allclose(relaxed_out["thermal_soft_scale"], smooth_out["thermal_soft_scale"], atol=1e-6)
    assert np.allclose(relaxed_out["thermal_scale"], smooth_out["thermal_scale"], atol=1e-6)


def test_safety_smooth_relaxed_numpy_torch_consistency():
    cfg = copy.deepcopy(load_cfg("configs/default.yaml"))
    cfg["safety"]["projection_mode"] = "smooth_relaxed"
    cfg["safety"]["smooth_relaxed_margin_c"] = 1.0
    safety = SafetyLayer(cfg)

    lower_raw_np = np.array([0.8, 0.4, -0.1, 0.2, -0.2], dtype=np.float32)
    temps_np = np.array([35.0, 36.0, 34.0], dtype=np.float32)
    np_out, _ = safety.project_np(
        upper_raw=11,
        lower_raw=lower_raw_np,
        temps=temps_np,
        amb_temp=32.0,
        gamma=0.06,
        delta=2.0,
        mem={"current_boost": 3, "dwell_count": 3},
    )
    torch_out = safety.project_torch(
        lower_raw=torch.tensor(lower_raw_np).unsqueeze(0),
        boost_combo=torch.tensor([3]),
        mode=torch.tensor([2]),
        temps=torch.tensor(temps_np).unsqueeze(0),
        amb_temp=torch.tensor([32.0]),
        gamma=torch.tensor([0.06]),
        delta=torch.tensor([2.0]),
    )

    assert np.allclose(
        np_out["currents_exec"],
        torch_out["currents_exec"].detach().cpu().numpy().reshape(-1),
        atol=1e-5,
    )
    assert np.allclose(
        np_out["thermal_scale"],
        torch_out["thermal_scale"].detach().cpu().numpy().reshape(-1),
        atol=1e-5,
    )


def test_safety_raw_to_exec_map_matches_preview():
    cfg = load_cfg("configs/default.yaml")
    safety = SafetyLayer(cfg)
    mem = {"current_boost": 0, "dwell_count": 0}

    exec_map = safety.raw_to_exec_map(mem)

    assert exec_map.shape == (12,)
    for raw_idx in range(12):
        boost_exec, mode_exec = safety.preview_exec(raw_idx, mem=mem)
        assert int(exec_map[raw_idx]) == safety.encode_exec(boost_exec, mode_exec)
