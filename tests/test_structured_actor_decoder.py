from __future__ import annotations

import copy

import numpy as np
import torch

from scripts.benchmark_constraint_scenarios import apply_baseline_overrides, formal_metadata_snapshot
from tchhmrl.safety.safety_layer import SafetyLayer
from tchhmrl.utils.config import load_cfg


def _cfg() -> dict:
    cfg = copy.deepcopy(load_cfg("configs/default.yaml"))
    cfg["safety"]["projection_mode"] = "thermal_cap"
    cfg["safety"]["current_decoder"] = "structured_total_allocation"
    cfg["safety"]["inactive_source_mask_mode"] = "hard_zero"
    return cfg


def test_structured_current_decoder_masks_inactive_sources_to_zero():
    cfg = _cfg()
    safety = SafetyLayer(cfg)
    out, _ = safety.project_np(
        upper_raw=0,  # anchor only, PS mode
        lower_raw=np.asarray([1.0, 10.0, 10.0, 0.2, -0.2], dtype=np.float32),
        temps=np.asarray([25.0, 25.0, 25.0], dtype=np.float32),
        amb_temp=25.0,
        mem={"current_boost": 0, "dwell_count": 3},
    )

    assert out["current_decoder"] == "structured_total_allocation"
    assert out["inactive_source_mask_mode"] == "hard_zero"
    assert float(out["currents_exec"][1]) == 0.0
    assert float(out["currents_exec"][2]) == 0.0
    assert float(out["actor_inactive_allocation_sum"]) == 0.0
    assert float(out["actor_total_current_requested"]) <= float(cfg["safety"]["current_max"][0]) + 1.0e-6
    assert float(out["actor_active_current_capacity"]) == float(cfg["safety"]["current_max"][0])
    assert float(np.sum(out["currents_exec"])) <= float(cfg["safety"]["bus_current_max"]) + 1.0e-6


def test_structured_current_decoder_respects_single_source_limits_and_logs_clip():
    cfg = _cfg()
    safety = SafetyLayer(cfg)
    out, _ = safety.project_np(
        upper_raw=11,  # all sources, HY
        lower_raw=np.asarray([1.0, -20.0, -20.0, 0.0, 0.0], dtype=np.float32),
        temps=np.asarray([25.0, 25.0, 25.0], dtype=np.float32),
        amb_temp=25.0,
        mem={"current_boost": 3, "dwell_count": 3},
    )

    assert np.all(np.asarray(out["currents_exec"]) <= np.asarray(cfg["safety"]["current_max"]) + 1.0e-6)
    assert float(out["actor_per_source_clip_count"]) >= 1.0
    assert float(out["structured_actor_per_source_clip_rate"]) > 0.0
    assert float(np.sum(out["current_requested"])) > float(cfg["safety"]["current_max"][0])


def test_structured_decoder_preserves_mode_specific_receiver_semantics():
    cfg = _cfg()
    safety = SafetyLayer(cfg)
    lower_raw = np.asarray([0.0, 0.0, 0.0, 0.4, -0.4], dtype=np.float32)
    temps = np.asarray([25.0, 25.0, 25.0], dtype=np.float32)
    mem = {"current_boost": 0, "dwell_count": 3}

    ps, _ = safety.project_np(upper_raw=0, lower_raw=lower_raw, temps=temps, amb_temp=25.0, mem=mem)
    ts, _ = safety.project_np(upper_raw=1, lower_raw=lower_raw, temps=temps, amb_temp=25.0, mem=mem)
    hy, _ = safety.project_np(upper_raw=2, lower_raw=lower_raw, temps=temps, amb_temp=25.0, mem=mem)

    assert float(ps["tau_exec"]) == 1.0
    assert float(ts["rho_exec"]) == 0.0
    assert 0.0 < float(hy["rho_exec"]) < 1.0
    assert 0.0 < float(hy["tau_exec"]) < 1.0
    assert float(ps["mode_effective_latent_dim"]) == 4.0
    assert float(ts["mode_effective_latent_dim"]) == 4.0
    assert float(hy["mode_effective_latent_dim"]) == 5.0


def test_structured_decoder_torch_contract_shapes():
    cfg = _cfg()
    safety = SafetyLayer(cfg)
    raw = torch.tensor([[0.2, 0.5, -0.5, 0.0, 0.0], [0.4, -0.5, 0.5, 0.0, 0.0]], dtype=torch.float32)
    out = safety.project_torch(
        raw,
        torch.tensor([3, 1], dtype=torch.long),
        torch.tensor([2, 2], dtype=torch.long),
        torch.full((2, 3), 25.0),
        torch.full((2,), 25.0),
    )

    assert out["actor_allocation"].shape == (2, 3)
    assert out["current_requested"].shape == (2, 3)
    assert out["mode_effective_latent_dim"].shape == (2,)
    assert torch.all(out["currents_exec"][:, :3] <= torch.tensor(cfg["safety"]["current_max"]) + 1.0e-6)


def test_baseline_override_keeps_prior_per_source_current_contract():
    cfg = _cfg()
    apply_baseline_overrides(cfg, "shin2024_adapted_codebook")

    assert cfg["safety"]["current_decoder"] == "per_source"
    assert cfg["safety"]["inactive_source_mask_mode"] == "hard_zero"


def test_formal_metadata_reports_structured_latent_entropy_contract():
    cfg = _cfg()
    meta = formal_metadata_snapshot(cfg)

    assert meta["current_decoder"] == "structured_total_allocation"
    assert meta["policy_distribution_space"] == "latent_structured_action"
    assert meta["critic_action_space"] == "executed_physical_action"
    assert meta["entropy_space"] == "mode_boost_masked_latent_action"
