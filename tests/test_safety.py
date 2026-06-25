from __future__ import annotations

import copy

import numpy as np
import torch

from tchhmrl.envs.physics_v2 import calibrate_logistic_eh_from_samples, logistic_eh_metric, validate_coupling_matrix
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

    lower_raw = np.array([-0.8, -0.8, -0.8, 0.0, 0.0], dtype=np.float32)
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


def test_safety_thermal_cap_limits_final_predicted_temperature_when_feasible():
    cfg = copy.deepcopy(load_cfg("configs/default.yaml"))
    cfg["safety"]["projection_mode"] = "thermal_cap"
    cfg["safety"]["thermal_cap_margin_c"] = 0.5
    safety = SafetyLayer(cfg)

    lower_raw = np.array([1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    temps = np.array([42.0, 42.0, 42.0], dtype=np.float32)
    out, _ = safety.project_np(
        upper_raw=11,
        lower_raw=lower_raw,
        temps=temps,
        amb_temp=35.0,
        gamma=0.06,
        delta=4.0,
        mem={"current_boost": 3, "dwell_count": 3},
    )

    target = float(cfg["safety"]["thermal_safe"]) - float(cfg["safety"]["thermal_cap_margin_c"])
    assert float(np.max(out["t_pred"])) <= target + 2e-5
    assert np.all(out["thermal_cap_scale"] <= 1.0 + 1e-6)
    assert np.any(out["thermal_cap_scale"] < 1.0 - 1e-5)


def test_safety_thermal_cap_does_not_reduce_current_when_below_cap():
    cfg = copy.deepcopy(load_cfg("configs/default.yaml"))
    cfg["safety"]["projection_mode"] = "thermal_cap"
    cfg["safety"]["thermal_cap_margin_c"] = 0.5
    safety = SafetyLayer(cfg)

    lower_raw = np.array([-0.8, -0.8, -0.8, 0.0, 0.0], dtype=np.float32)
    temps = np.array([25.0, 25.0, 25.0], dtype=np.float32)
    out, _ = safety.project_np(
        upper_raw=11,
        lower_raw=lower_raw,
        temps=temps,
        amb_temp=26.0,
        gamma=0.04,
        delta=1.0,
        mem={"current_boost": 3, "dwell_count": 3},
    )

    assert np.allclose(out["thermal_cap_scale"], 1.0, atol=1e-6)
    assert np.isclose(
        float(out["projected_current_total"]),
        float(out["bus_projected_current_total"]),
        atol=1e-6,
    )


def test_safety_thermal_cap_zeroes_current_when_thermal_base_exceeds_target():
    cfg = copy.deepcopy(load_cfg("configs/default.yaml"))
    cfg["safety"]["projection_mode"] = "thermal_cap"
    cfg["safety"]["thermal_cap_margin_c"] = 0.5
    safety = SafetyLayer(cfg)

    lower_raw = np.array([1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    hot_temps = np.array([100.0, 100.0, 100.0], dtype=np.float32)
    out, _ = safety.project_np(
        upper_raw=11,
        lower_raw=lower_raw,
        temps=hot_temps,
        amb_temp=35.0,
        gamma=0.04,
        delta=4.0,
        mem={"current_boost": 3, "dwell_count": 3},
    )

    assert np.allclose(out["currents_exec"], 0.0, atol=1e-7)
    assert np.allclose(out["thermal_cap_current"], 0.0, atol=1e-7)


def test_safety_thermal_cap_numpy_torch_consistency():
    cfg = copy.deepcopy(load_cfg("configs/default.yaml"))
    cfg["safety"]["projection_mode"] = "thermal_cap"
    cfg["safety"]["thermal_cap_margin_c"] = 0.5
    safety = SafetyLayer(cfg)

    lower_raw_np = np.array([0.9, 0.6, 0.3, 0.2, -0.2], dtype=np.float32)
    temps_np = np.array([43.0, 42.0, 41.0], dtype=np.float32)
    np_out, _ = safety.project_np(
        upper_raw=11,
        lower_raw=lower_raw_np,
        temps=temps_np,
        amb_temp=35.0,
        gamma=0.06,
        delta=4.0,
        mem={"current_boost": 3, "dwell_count": 3},
    )
    torch_out = safety.project_torch(
        lower_raw=torch.tensor(lower_raw_np).unsqueeze(0),
        boost_combo=torch.tensor([3]),
        mode=torch.tensor([2]),
        temps=torch.tensor(temps_np).unsqueeze(0),
        amb_temp=torch.tensor([35.0]),
        gamma=torch.tensor([0.06]),
        delta=torch.tensor([4.0]),
    )

    for key in ["currents_exec", "thermal_cap_scale", "t_pred"]:
        assert np.allclose(
            np_out[key],
            torch_out[key].detach().cpu().numpy().reshape(-1),
            atol=1e-5,
        )


def test_adaptive_thermal_estimator_updates_only_with_valid_excitation():
    cfg = copy.deepcopy(load_cfg("configs/default.yaml"))
    cfg["adaptive_thermal"]["enabled"] = True
    cfg["adaptive_thermal"]["initial_std"] = 0.0
    cfg["safety"]["effective_gain_initial"] = [4.0, 4.0, 4.0]
    cfg["adaptive_thermal"]["initial_effective_gain"] = [4.0, 4.0, 4.0]
    safety = SafetyLayer(cfg)

    before = safety.thermal_diagnostics()
    assert np.allclose(before["effective_gain_mean"], 4.0)
    assert np.allclose(before["thermal_gain_valid_count"], 0.0)

    temps_before = np.array([30.0, 30.0, 30.0], dtype=np.float32)
    currents = np.array([1.0, 0.01, 1.0], dtype=np.float32)
    thermal_base = np.array([30.0, 30.0, 30.0], dtype=np.float32)
    coeff = safety.tx_thermal_coeff.astype(np.float32)
    temps_after = thermal_base + 4.0 * coeff * np.array([1.2, 1.0, 0.8], dtype=np.float32) * currents**2

    diag = safety.update_thermal_estimator(
        currents=currents,
        temps_before=temps_before,
        temps_after=temps_after,
        thermal_base=thermal_base,
    )

    assert diag["adaptive_thermal_enabled"] is True
    assert float(diag["thermal_gain_valid_count"][0]) == 1.0
    assert float(diag["thermal_gain_valid_count"][1]) == 0.0
    assert float(diag["thermal_gain_valid_count"][2]) == 1.0
    assert float(diag["effective_gain_mean"][0]) > 4.0
    assert float(diag["effective_gain_mean"][2]) > 4.0


def test_default_adaptive_thermal_has_nonzero_initial_uncertainty():
    cfg = copy.deepcopy(load_cfg("configs/default.yaml"))
    safety = SafetyLayer(cfg)
    diag = safety.thermal_diagnostics()

    assert np.all(np.asarray(diag["effective_gain_uncertainty"], dtype=np.float32) > 0.0)
    assert np.all(np.asarray(diag["effective_gain_safe"], dtype=np.float32) > np.asarray(diag["effective_gain_mean"], dtype=np.float32))


def test_uncertainty_aware_thermal_cap_tightens_current_limit():
    cfg_nominal = copy.deepcopy(load_cfg("configs/default.yaml"))
    cfg_nominal["adaptive_thermal"]["enabled"] = False
    cfg_adaptive = copy.deepcopy(load_cfg("configs/default.yaml"))
    cfg_adaptive["adaptive_thermal"]["enabled"] = True

    nominal = SafetyLayer(cfg_nominal)
    adaptive = SafetyLayer(cfg_adaptive)
    adaptive.load_state_dict(
        {
            "thermal_estimator": {
                "gain_mean": np.array([9.0, 12.0, 12.0], dtype=np.float32),
                "gain_var": np.zeros(3, dtype=np.float32),
                "valid_count": np.array([10, 10, 10], dtype=np.int64),
                "temperature_slope": np.zeros(3, dtype=np.float32),
                "last_headroom": np.zeros(3, dtype=np.float32),
            }
        }
    )

    kwargs = {
        "upper_raw": 11,
        "lower_raw": np.array([1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32),
        "temps": np.array([42.0, 42.0, 42.0], dtype=np.float32),
        "amb_temp": 35.0,
        "gamma": 0.06,
        "delta": 4.0,
        "mem": {"current_boost": 3, "dwell_count": 3},
    }
    out_nominal, _ = nominal.project_np(**kwargs)
    out_adaptive, _ = adaptive.project_np(**kwargs)

    assert np.all(out_adaptive["thermal_cap_current"] <= out_nominal["thermal_cap_current"] + 1e-7)
    assert float(np.sum(out_adaptive["currents_exec"])) < float(np.sum(out_nominal["currents_exec"]))
    assert np.all(out_adaptive["effective_gain_safe"] > np.array([9.0, 12.0, 12.0], dtype=np.float32))


def test_safety_state_dict_restores_adaptive_thermal_estimator():
    cfg = copy.deepcopy(load_cfg("configs/default.yaml"))
    cfg["adaptive_thermal"]["enabled"] = True
    safety = SafetyLayer(cfg)

    state = safety.state_dict()
    safety.update_thermal_estimator(
        currents=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        temps_before=np.array([30.0, 30.0, 30.0], dtype=np.float32),
        temps_after=np.array([35.0, 36.0, 37.0], dtype=np.float32),
        thermal_base=np.array([30.0, 30.0, 30.0], dtype=np.float32),
    )
    assert np.any(safety.thermal_diagnostics()["thermal_gain_valid_count"] > 0.0)

    safety.load_state_dict(state)
    restored = safety.thermal_diagnostics()
    assert np.allclose(restored["thermal_gain_valid_count"], 0.0)
    assert np.allclose(restored["thermal_gain_mean"], np.asarray(cfg["safety"]["effective_gain_initial"], dtype=np.float32))


def test_thermal_diagnostics_has_no_side_effects():
    cfg = copy.deepcopy(load_cfg("configs/default.yaml"))
    safety = SafetyLayer(cfg)
    safety.observe_temperature(np.asarray([41.0, 42.0, 43.0], dtype=np.float32))
    before = copy.deepcopy(safety.state_dict())

    _ = safety.thermal_diagnostics()
    _ = safety.thermal_diagnostics()
    after = safety.state_dict()

    for key, value in before["thermal_estimator"].items():
        if isinstance(value, np.ndarray):
            assert np.array_equal(value, after["thermal_estimator"][key])
        else:
            assert value == after["thermal_estimator"][key]


def test_safety_raw_to_exec_map_matches_preview():
    cfg = load_cfg("configs/default.yaml")
    safety = SafetyLayer(cfg)
    mem = {"current_boost": 0, "dwell_count": 0}

    exec_map = safety.raw_to_exec_map(mem)

    assert exec_map.shape == (12,)
    for raw_idx in range(12):
        boost_exec, mode_exec = safety.preview_exec(raw_idx, mem=mem)
        assert int(exec_map[raw_idx]) == safety.encode_exec(boost_exec, mode_exec)


def test_logistic_eh_zero_monotonic_saturation():
    vals = np.asarray(
        [logistic_eh_metric(x, M=0.2, a=12.0, b=0.1) for x in [0.0, 0.05, 0.1, 0.2, 1.0]],
        dtype=np.float64,
    )
    assert vals[0] <= 1.0e-10
    assert np.all(np.diff(vals) >= -1.0e-12)
    assert vals[-1] <= 0.2 + 1.0e-12
    assert vals[-1] > vals[1]


def test_eh_calibration_manifest_reproducible_and_rejects_degenerate_distribution():
    samples = np.linspace(0.001, 0.02, 256)
    first = calibrate_logistic_eh_from_samples(samples)
    second = calibrate_logistic_eh_from_samples(samples)
    assert first == second
    assert first["eh_nl_M"] > 0.0
    assert first["eh_nl_a"] > 0.0
    try:
        calibrate_logistic_eh_from_samples(np.ones(64) * 0.001)
    except ValueError as exc:
        assert "Degenerate" in str(exc)
    else:
        raise AssertionError("degenerate EH calibration samples should fail")


def test_coupling_matrix_shape_direction_and_row_sum():
    mat = validate_coupling_matrix(
        [
            [0.0, 0.015, 0.0075],
            [0.015, 0.0, 0.015],
            [0.0075, 0.015, 0.0],
        ],
        n_tx=3,
    )
    assert mat.shape == (3, 3)
    assert np.allclose(np.diag(mat), 0.0)
    assert np.all(np.sum(mat, axis=1) <= 0.03 + 1.0e-8)


def test_thermal_coupling_zero_equals_independent_and_direction():
    cfg_ind = copy.deepcopy(load_cfg("configs/default.yaml"))
    cfg_coup = copy.deepcopy(load_cfg("configs/default.yaml"))
    cfg_ind["physics"]["thermal_model"] = "independent"
    cfg_coup["physics"]["thermal_model"] = "coupled"
    cfg_coup["physics"]["thermal_coupling_matrix"] = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    temps = np.asarray([42.0, 40.0, 39.0], dtype=np.float32)
    safety_ind = SafetyLayer(cfg_ind)
    safety_zero = SafetyLayer(cfg_coup)
    base_ind, coupling_ind = safety_ind._thermal_base_np(temps, 35.0, 0.06)
    base_zero, coupling_zero = safety_zero._thermal_base_np(temps, 35.0, 0.06)
    assert np.allclose(base_ind, base_zero, atol=1.0e-7)
    assert np.allclose(coupling_ind, 0.0, atol=1.0e-7)
    assert np.allclose(coupling_zero, 0.0, atol=1.0e-7)

    cfg_dir = copy.deepcopy(load_cfg("configs/default.yaml"))
    cfg_dir["physics"]["thermal_model"] = "coupled"
    cfg_dir["physics"]["safety_projection_version"] = "coupled_thermal_cap_v1"
    cfg_dir["physics"]["thermal_coupling_matrix"] = [
        [0.0, 0.015, 0.0075],
        [0.015, 0.0, 0.015],
        [0.0075, 0.015, 0.0],
    ]
    safety_dir = SafetyLayer(cfg_dir)
    _, coupling = safety_dir._thermal_base_np(np.asarray([50.0, 40.0, 40.0], dtype=np.float32), 35.0, 0.06)
    assert coupling[1] > 0.0
    assert coupling[0] < 0.0


def test_thermal_pred_temp_and_margin_logged():
    cfg = copy.deepcopy(load_cfg("configs/default.yaml"))
    safety = SafetyLayer(cfg)
    out, _ = safety.project_np(
        upper_raw=11,
        lower_raw=np.asarray([1.0, 0.8, 0.6, 0.0, 0.0], dtype=np.float32),
        temps=np.asarray([44.0, 42.0, 41.0], dtype=np.float32),
        amb_temp=35.0,
        gamma=0.06,
        delta=4.0,
        mem={"current_boost": 3, "dwell_count": 3},
    )
    for key in ["thermal_source_term", "thermal_base", "thermal_pred_temp", "thermal_pred_margin"]:
        assert key in out
        assert np.asarray(out[key]).shape == (3,)
    assert "thermal_coupling_term" not in out
    assert "thermal_base_coupled" not in out
    assert np.allclose(out["thermal_source_term"], 0.0, atol=1.0e-7)
    assert np.allclose(out["thermal_pred_temp"], out["t_pred"], atol=1.0e-7)
    assert np.allclose(out["thermal_pred_margin"], out["thermal_margin"], atol=1.0e-7)
