from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from tchhmrl.agents.dqn_upper import UpperDQN
from tchhmrl.agents.ddpg_lower import LowerDDPG
from tchhmrl.agents.sac_lower import LowerSAC
from tchhmrl.buffers.replay_buffer import ReplayBuffer
from tchhmrl.constraints.dual_layer import DualLayer
from tchhmrl.envs.task_contract import (
    build_task_summary_v2,
    filter_formally_comparable_records,
    is_formally_comparable_record,
    ordered_task_batch_hash,
    task_batch_hash,
    task_defaults_from_cfg,
    task_distribution_summary,
)
from tchhmrl.envs.task_sampler import TaskSampler, validate_site_bank
from tchhmrl.envs.uw_slipt_env import MultiTxUwSliptEnv
from tchhmrl.meta.meta_trainer import MetaTrainer
from tchhmrl.safety.safety_layer import SafetyLayer
from tchhmrl.utils.config import apply_cli_overrides, load_cfg, resolve_device
from tchhmrl.utils.logger import Logger
from tchhmrl.utils.seed import set_seed


def apply_common_settings(
    cfg: Dict,
    meta_iters: int,
    out_dir: Path,
    run_name: str,
    seed: int,
    fast_mode: bool,
    use_curriculum: bool = False,
) -> Dict:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("alignment", {})
    cfg["alignment"].setdefault("alignment_version", "system_model_v1")
    cfg["alignment"].setdefault("task_summary_version", "site_v2")
    cfg["alignment"].setdefault("pre_alignment", False)
    cfg["experiment"]["seed"] = int(seed)
    cfg["experiment"]["log_dir"] = str(out_dir)
    cfg["experiment"]["run_name"] = run_name

    cfg["meta"]["meta_iters"] = int(meta_iters)
    if use_curriculum:
        cfg.setdefault("meta", {}).setdefault("curriculum", {})
        cfg["meta"]["curriculum"]["enabled"] = True
    if fast_mode:
        cfg["meta"]["n_tasks_per_iter"] = int(min(3, cfg["meta"]["n_tasks_per_iter"]))
        cfg["meta"]["support_episodes"] = int(min(1, cfg["meta"]["support_episodes"]))
        cfg["meta"]["query_episodes"] = int(min(1, cfg["meta"]["query_episodes"]))

        cfg["agent"]["warmup_steps"] = int(min(100, cfg["agent"]["warmup_steps"]))
        cfg["agent"]["batch_size"] = int(min(48, cfg["agent"]["batch_size"]))
        cfg["env"]["episode_len"] = int(min(60, cfg["env"]["episode_len"]))
    return cfg


def infer_task_source(cfg: Dict) -> str:
    site_bank = cfg.get("sampler", {}).get("site_bank", [])
    return "site_bank" if site_bank else "global_fallback"


def alignment_snapshot(cfg: Dict, *, pre_alignment: bool | None = None, task_source: str | None = None) -> Dict[str, object]:
    alignment_cfg = cfg.get("alignment", {})
    return {
        "alignment_version": str(alignment_cfg.get("alignment_version", "system_model_v1")),
        "task_summary_version": str(alignment_cfg.get("task_summary_version", "site_v2")),
        "pre_alignment": bool(alignment_cfg.get("pre_alignment", False) if pre_alignment is None else pre_alignment),
        "task_source": str(task_source or infer_task_source(cfg)),
    }


def sync_site_bank_with_cfg(cfg: Dict) -> None:
    sampler_cfg = cfg.setdefault("sampler", {})
    env_cfg = cfg.setdefault("env", {})
    base_distances = np.asarray(env_cfg.get("distances", sampler_cfg.get("default_distances", [5.0, 6.0, 6.5])), dtype=float)
    if base_distances.shape != (3,):
        raise ValueError(f"env.distances must have shape (3,), got {base_distances.shape}")
    sampler_cfg["default_distances"] = [float(x) for x in base_distances]
    offsets = np.asarray(
        [
            [-0.35, -0.25, -0.20],
            [0.00, 0.00, 0.00],
            [0.35, 0.25, 0.20],
        ],
        dtype=float,
    )
    site_bank = []
    for site_id, delta in enumerate(offsets.tolist()):
        site_distances = np.maximum(base_distances + np.asarray(delta, dtype=float), 0.5)
        site_bank.append(
            {
                "site_id": int(site_id),
                "distances": [float(x) for x in site_distances],
                "attenuation_c_range": [float(x) for x in sampler_cfg["attenuation_c_range"]],
                "misalign_std_range": [float(x) for x in sampler_cfg["misalign_std_range"]],
                "amb_temp_range": [float(x) for x in sampler_cfg["amb_temp_range"]],
                "gamma_range": [float(x) for x in sampler_cfg["gamma_range"]],
                "delta_range": [float(x) for x in sampler_cfg["delta_range"]],
            }
        )
    sampler_cfg["site_bank"] = site_bank


def fixed_task_bank_hash(tasks: List[object]) -> str:
    return task_batch_hash(tasks)


def ordered_fixed_task_bank_hash(tasks: List[object]) -> str:
    return ordered_task_batch_hash(tasks)


def inject_default_curriculum(cfg: Dict) -> None:
    """Inject a lightweight easy->moderate->target sampler curriculum."""
    range_keys = ["attenuation_c_range", "misalign_std_range", "amb_temp_range", "gamma_range", "delta_range"]
    target = {k: copy.deepcopy(cfg["sampler"][k]) for k in range_keys if k in cfg["sampler"]}
    easy = {
        "attenuation_c_range": [0.10, 0.20],
        "misalign_std_range": [0.03, 0.08],
        "amb_temp_range": [22.0, 28.0],
        "gamma_range": [0.02, 0.05],
        "delta_range": [1.4, 3.0],
    }
    moderate = {}
    for k, target_rng in target.items():
        if k not in easy:
            moderate[k] = target_rng
            continue
        lo_e, hi_e = easy[k]
        lo_t, hi_t = target_rng
        moderate[k] = [0.5 * (lo_e + lo_t), 0.5 * (hi_e + hi_t)]

    cfg.setdefault("meta", {}).setdefault("curriculum", {})
    cfg["meta"]["curriculum"]["enabled"] = True
    cfg["meta"]["curriculum"]["phases"] = [
        {"name": "easy", "until_frac": 0.33, "sampler": easy},
        {"name": "moderate", "until_frac": 0.70, "sampler": moderate},
        {"name": "target", "until_frac": 1.00, "sampler": target},
    ]


def dump_resolved_config(cfg: Dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def apply_variant(cfg: Dict, variant: str) -> None:
    if variant == "hybrid":
        cfg["env"]["hybrid"]["tx_device"] = ["LED", "LD", "LD"]
        cfg["env"]["hybrid"]["tx_enabled"] = [1.0, 1.0, 1.0]
        sync_site_bank_with_cfg(cfg)
        return
    if variant == "single_led":
        # True single-transmitter LED baseline: only one LED emitter is physically active.
        n_tx = int(cfg["env"]["n_tx"])
        cfg["env"]["hybrid"]["tx_device"] = ["LED"] * n_tx
        cfg["env"]["hybrid"]["tx_enabled"] = [1.0] + [0.0] * max(0, n_tx - 1)
        sync_site_bank_with_cfg(cfg)
        return
    if variant == "single_ld":
        # True single-transmitter LD baseline: only one LD emitter is physically active.
        n_tx = int(cfg["env"]["n_tx"])
        cfg["env"]["hybrid"]["tx_device"] = ["LD"] * n_tx
        cfg["env"]["hybrid"]["tx_enabled"] = [1.0] + [0.0] * max(0, n_tx - 1)
        sync_site_bank_with_cfg(cfg)
        return
    raise ValueError(f"Unknown variant: {variant}")


def apply_ablation(cfg: Dict, ablation: str) -> None:
    ablation = str(ablation)
    if ablation == "full":
        sync_site_bank_with_cfg(cfg)
        return
    if ablation == "wo_meta":
        cfg.setdefault("context", {})["enabled"] = False
        cfg.setdefault("meta", {})["explicit_inner_outer"] = False
        cfg["meta"]["query_updates_enabled"] = False
        support_eps = int(cfg["meta"].get("support_episodes", 0))
        query_eps = int(cfg["meta"].get("query_episodes", 0))
        cfg["meta"]["support_episodes"] = support_eps + query_eps
        cfg["meta"]["query_episodes"] = 0
        sync_site_bank_with_cfg(cfg)
        return
    if ablation == "wo_lagrangian":
        cfg.setdefault("meta", {})["dual_enabled"] = False
        n_duals = len(cfg["meta"].get("dual_names", ["qos"] + [f"temp_tx{i}" for i in range(int(cfg["env"]["n_tx"]))]))
        cfg["meta"]["dual_lr"] = 0.0
        cfg["meta"]["dual_lrs"] = [0.0] * n_duals
        sync_site_bank_with_cfg(cfg)
        return
    if ablation == "hard_clip":
        cfg.setdefault("safety", {})["projection_mode"] = "hard_clip"
        return
    raise ValueError(f"Unknown ablation: {ablation}")


def apply_baseline_overrides(cfg: Dict, baseline: str) -> None:
    baseline = str(baseline)
    if baseline == "sac_lagrangian":
        cfg.setdefault("context", {})["enabled"] = False
        cfg.setdefault("agent", {})["z_dim"] = 0
        cfg.setdefault("meta", {})["explicit_inner_outer"] = False
        cfg["meta"]["query_updates_enabled"] = False
        return
    if baseline == "sac_dalal_safe":
        cfg.setdefault("context", {})["enabled"] = False
        cfg.setdefault("agent", {})["z_dim"] = 0
        cfg.setdefault("meta", {})["explicit_inner_outer"] = False
        cfg["meta"]["query_updates_enabled"] = False
        cfg["meta"]["dual_enabled"] = False
        cfg["meta"]["dual_lr"] = 0.0
        n_duals = len(cfg["meta"].get("dual_names", ["qos"] + [f"temp_tx{i}" for i in range(int(cfg["env"]["n_tx"]))]))
        cfg["meta"]["dual_lrs"] = [0.0] * n_duals
        cfg.setdefault("safety", {})["projection_mode"] = "dalal_safe"
        cfg["baseline_metadata"] = {
            "baseline_family": "sac_dalal_safe",
            "exact_reproduction": False,
            "external_baseline": True,
            "safety_protocol": "dalal_style_projection",
            "comparison_role": "external_safety_layer_baseline",
        }
        return
    if baseline == "shin2024_matched":
        cfg.setdefault("context", {})["enabled"] = False
        cfg.setdefault("agent", {})["z_dim"] = 0
        cfg.setdefault("meta", {})["explicit_inner_outer"] = False
        cfg["meta"]["query_updates_enabled"] = False
        cfg["meta"]["dual_enabled"] = False
        cfg["meta"]["dual_lr"] = 0.0
        cfg["meta"]["dual_lrs"] = [0.0] * len(cfg["meta"].get("dual_names", []))
        cfg.setdefault("upper_dqn", {})
        cfg["upper_dqn"]["lr"] = 1.0e-4
        cfg["upper_dqn"]["epsilon_start"] = 0.01
        cfg["upper_dqn"]["epsilon_final"] = 0.01
        cfg["upper_dqn"]["replay_size"] = 2000
        cfg.setdefault("lower_ddpg", {})
        cfg["lower_ddpg"].update(
            {
                "action_contract": "rho_tau_fixed_current",
                "learned_action_dim": 2,
                "fixed_current_fraction": 0.5,
                "replay_size": int(1.0e6),
                "batch_size": 64,
                "actor_lr": 1.0e-4,
                "critic_lr": 3.0e-4,
                "gamma": 0.99,
                "target_tau": 0.2,
                "noise_std": 0.10,
                "grad_clip": 5.0,
            }
        )
        cfg["baseline_metadata"] = {
            "baseline_family": "shin2024_matched",
            "exact_reproduction": False,
            "safety_protocol": "common_smooth_projection",
            "lower_learned_action_dim": 2,
            "fixed_mode_exec": 2,
            "fixed_mode_name": "HY",
            "fixed_current_template": "sigmoid_logit_fraction",
            "fixed_current_fraction": 0.5,
        }
        cfg["agent"]["batch_size"] = 64
        return
    if baseline == "dalal2018_safe":
        cfg.setdefault("safety", {})["projection_mode"] = "dalal_safe"
        return
    if baseline == "heuristic_safe":
        return
    raise ValueError(f"Unknown baseline override target: {baseline}")


def apply_scenario(cfg: Dict, scenario: str) -> None:
    def finish() -> None:
        sync_site_bank_with_cfg(cfg)

    if scenario in {"easy_baseline", "baseline_easy"}:
        # Easy / baseline: mild channel, low thermal pressure, smooth dynamics.
        cfg["env"]["attenuation_c"] = 0.14
        cfg["env"]["misalign_std"] = 0.05
        cfg["env"]["amb_temp"] = 24.0
        cfg["env"]["gamma"] = 0.03
        cfg["env"]["delta"] = 2.0
        cfg["env"]["thermal_safe"] = 64.0
        cfg["env"]["thermal_cutoff"] = 76.0
        cfg["env"]["se_weight"] = 1.0
        cfg["env"]["eh_weight"] = 0.25
        cfg["env"]["power_weight"] = 0.010
        cfg["env"]["cost_weight"] = 1.4

        cfg["env"]["temporal_misalign_rho"] = 0.15
        cfg["env"]["attenuation_drift_rho"] = 0.40
        cfg["env"]["attenuation_drift_std"] = 0.004
        cfg["env"]["burst_prob"] = 0.0
        cfg["env"]["burst_strength_range"] = [0.0, 0.0]
        cfg["env"]["burst_decay"] = 0.9
        cfg["env"]["obs_bias_rho"] = 0.0
        cfg["env"]["obs_bias_step_std"] = 0.0

        cfg["env"]["hybrid"]["misalign_led_scale"] = 1.25
        cfg["env"]["hybrid"]["misalign_ld_scale"] = 0.72
        cfg["env"]["hybrid"]["noise_led_coeff"] = 0.015
        cfg["env"]["hybrid"]["noise_ld_coeff"] = 0.024
        cfg["env"]["hybrid"]["burst_led_factor"] = 0.9
        cfg["env"]["hybrid"]["burst_ld_factor"] = 1.1

        cfg["sampler"]["attenuation_c_range"] = [0.10, 0.20]
        cfg["sampler"]["misalign_std_range"] = [0.03, 0.08]
        cfg["sampler"]["amb_temp_range"] = [22.0, 28.0]
        cfg["sampler"]["gamma_range"] = [0.02, 0.05]
        cfg["sampler"]["delta_range"] = [1.4, 3.0]

        cfg["safety"]["thermal_safe"] = 64.0
        cfg["safety"]["thermal_cutoff"] = 76.0
        cfg["safety"]["soft_alpha"] = 0.45
        cfg["safety"]["cutoff_alpha"] = 0.75
        cfg["safety"]["current_max"] = [2.2, 2.2, 2.2]
        cfg["safety"]["bus_current_max"] = 5.0
        cfg["safety"]["mask_floor"] = 0.04
        return finish()

    if scenario in {"moderate_practical", "thermal_moderate"}:
        # Moderate condition: still constrained, but meaningfully easier than the main practical-hard setup.
        cfg["env"]["attenuation_c"] = 0.19
        cfg["env"]["misalign_std"] = 0.08
        cfg["env"]["thermal_safe"] = 56.0
        cfg["env"]["thermal_cutoff"] = 66.0
        cfg["env"]["amb_temp"] = 29.0
        cfg["env"]["gamma"] = 0.055
        cfg["env"]["delta"] = 4.2
        cfg["env"]["se_weight"] = 1.05
        cfg["env"]["eh_weight"] = 0.23
        cfg["env"]["power_weight"] = 0.0050
        cfg["env"]["cost_weight"] = 2.00
        cfg["env"]["mode_se_gain"] = [1.0, 1.24, 0.78]
        cfg["env"]["mode_eh_gain"] = [1.0, 0.78, 1.24]
        cfg["env"]["temporal_misalign_rho"] = 0.55
        cfg["env"]["attenuation_drift_rho"] = 0.70
        cfg["env"]["attenuation_drift_std"] = 0.010
        cfg["env"]["burst_prob"] = 0.02
        cfg["env"]["burst_strength_range"] = [0.06, 0.16]
        cfg["env"]["burst_decay"] = 0.88
        cfg["env"]["obs_bias_rho"] = 0.60
        cfg["env"]["obs_bias_step_std"] = 0.0015

        cfg["env"]["hybrid"]["eta_led"] = 0.90
        cfg["env"]["hybrid"]["eta_ld"] = 1.20
        cfg["env"]["hybrid"]["attenuation_led_factor"] = 1.10
        cfg["env"]["hybrid"]["attenuation_ld_factor"] = 0.90
        cfg["env"]["hybrid"]["misalign_led_scale"] = 1.28
        cfg["env"]["hybrid"]["misalign_ld_scale"] = 0.62
        cfg["env"]["hybrid"]["noise_led_coeff"] = 0.016
        cfg["env"]["hybrid"]["noise_ld_coeff"] = 0.029
        cfg["env"]["hybrid"]["se_led_weight"] = 0.97
        cfg["env"]["hybrid"]["se_ld_weight"] = 1.06
        cfg["env"]["hybrid"]["eh_led_weight"] = 1.18
        cfg["env"]["hybrid"]["eh_ld_weight"] = 0.85
        cfg["env"]["hybrid"]["thermal_led_coeff"] = 1.05
        cfg["env"]["hybrid"]["thermal_ld_coeff"] = 1.42
        cfg["env"]["hybrid"]["burst_led_factor"] = 0.92
        cfg["env"]["hybrid"]["burst_ld_factor"] = 1.25

        cfg["sampler"]["attenuation_c_range"] = [0.14, 0.24]
        cfg["sampler"]["misalign_std_range"] = [0.05, 0.11]
        cfg["sampler"]["amb_temp_range"] = [27.0, 32.0]
        cfg["sampler"]["gamma_range"] = [0.04, 0.07]
        cfg["sampler"]["delta_range"] = [3.4, 5.2]

        cfg["safety"]["min_dwell_steps"] = 2
        cfg["safety"]["thermal_safe"] = 55.0
        cfg["safety"]["thermal_cutoff"] = 65.0
        cfg["safety"]["soft_alpha"] = 0.28
        cfg["safety"]["cutoff_alpha"] = 0.36
        cfg["safety"]["current_max"] = [2.8, 2.8, 2.8]
        cfg["safety"]["bus_current_max"] = 5.8
        cfg["safety"]["mask_floor"] = 0.02
        return finish()

    if scenario in {"practical_hard"}:
        # Main practical-hard (recommended primary setting):
        # high pressure but still balanced enough for fair LED/LD/Hybrid comparison.
        cfg["env"]["attenuation_c"] = 0.24
        cfg["env"]["misalign_std"] = 0.10
        cfg["env"]["thermal_safe"] = 50.0
        cfg["env"]["thermal_cutoff"] = 60.0
        cfg["env"]["amb_temp"] = 32.0
        cfg["env"]["gamma"] = 0.070
        cfg["env"]["delta"] = 5.2
        cfg["env"]["se_weight"] = 1.08
        cfg["env"]["eh_weight"] = 0.22
        cfg["env"]["power_weight"] = 0.0032
        cfg["env"]["cost_weight"] = 2.40
        cfg["env"]["mode_se_gain"] = [1.0, 1.30, 0.72]
        cfg["env"]["mode_eh_gain"] = [1.0, 0.72, 1.30]
        cfg["env"]["temporal_misalign_rho"] = 0.75
        cfg["env"]["attenuation_drift_rho"] = 0.86
        cfg["env"]["attenuation_drift_std"] = 0.016
        cfg["env"]["burst_prob"] = 0.05
        cfg["env"]["burst_strength_range"] = [0.10, 0.22]
        cfg["env"]["burst_decay"] = 0.85
        cfg["env"]["obs_bias_rho"] = 0.90
        cfg["env"]["obs_bias_step_std"] = 0.003

        cfg["env"]["hybrid"]["eta_led"] = 0.90
        cfg["env"]["hybrid"]["eta_ld"] = 1.25
        cfg["env"]["hybrid"]["attenuation_led_factor"] = 1.12
        cfg["env"]["hybrid"]["attenuation_ld_factor"] = 0.88
        cfg["env"]["hybrid"]["misalign_led_scale"] = 1.30
        cfg["env"]["hybrid"]["misalign_ld_scale"] = 0.55
        cfg["env"]["hybrid"]["noise_led_coeff"] = 0.017
        cfg["env"]["hybrid"]["noise_ld_coeff"] = 0.034
        cfg["env"]["hybrid"]["se_led_weight"] = 0.95
        cfg["env"]["hybrid"]["se_ld_weight"] = 1.10
        cfg["env"]["hybrid"]["eh_led_weight"] = 1.15
        cfg["env"]["hybrid"]["eh_ld_weight"] = 0.82
        cfg["env"]["hybrid"]["thermal_led_coeff"] = 1.08
        cfg["env"]["hybrid"]["thermal_ld_coeff"] = 1.60
        cfg["env"]["hybrid"]["burst_led_factor"] = 0.90
        cfg["env"]["hybrid"]["burst_ld_factor"] = 1.40

        cfg["sampler"]["attenuation_c_range"] = [0.18, 0.30]
        cfg["sampler"]["misalign_std_range"] = [0.07, 0.14]
        cfg["sampler"]["amb_temp_range"] = [30.0, 35.0]
        cfg["sampler"]["gamma_range"] = [0.05, 0.09]
        cfg["sampler"]["delta_range"] = [4.4, 6.4]

        cfg["safety"]["min_dwell_steps"] = 2
        cfg["safety"]["thermal_safe"] = 49.0
        cfg["safety"]["thermal_cutoff"] = 59.0
        cfg["safety"]["soft_alpha"] = 0.22
        cfg["safety"]["cutoff_alpha"] = 0.30
        cfg["safety"]["current_max"] = [3.0, 3.0, 3.0]
        cfg["safety"]["bus_current_max"] = 6.4
        cfg["safety"]["mask_floor"] = 0.02
        return finish()

    if scenario in {"hard_balanced", "balanced_hard"}:
        # Hard-balanced: constraints should activate moderately without saturation.
        cfg["env"]["attenuation_c"] = 0.27
        cfg["env"]["misalign_std"] = 0.12
        cfg["env"]["thermal_safe"] = 49.0
        cfg["env"]["thermal_cutoff"] = 59.0
        cfg["env"]["amb_temp"] = 33.0
        cfg["env"]["gamma"] = 0.078
        cfg["env"]["delta"] = 5.8
        cfg["env"]["se_weight"] = 1.10
        cfg["env"]["eh_weight"] = 0.24
        cfg["env"]["power_weight"] = 0.0028
        cfg["env"]["cost_weight"] = 2.60
        cfg["env"]["mode_se_gain"] = [1.0, 1.30, 0.72]
        cfg["env"]["mode_eh_gain"] = [1.0, 0.72, 1.30]
        cfg["env"]["temporal_misalign_rho"] = 0.82
        cfg["env"]["attenuation_drift_rho"] = 0.89
        cfg["env"]["attenuation_drift_std"] = 0.022
        cfg["env"]["burst_prob"] = 0.06
        cfg["env"]["burst_strength_range"] = [0.12, 0.28]
        cfg["env"]["burst_decay"] = 0.86
        cfg["env"]["obs_bias_rho"] = 0.92
        cfg["env"]["obs_bias_step_std"] = 0.004

        cfg["env"]["hybrid"]["eta_led"] = 0.90
        cfg["env"]["hybrid"]["eta_ld"] = 1.25
        cfg["env"]["hybrid"]["attenuation_led_factor"] = 1.14
        cfg["env"]["hybrid"]["attenuation_ld_factor"] = 0.87
        cfg["env"]["hybrid"]["misalign_led_scale"] = 1.35
        cfg["env"]["hybrid"]["misalign_ld_scale"] = 0.52
        cfg["env"]["hybrid"]["noise_led_coeff"] = 0.018
        cfg["env"]["hybrid"]["noise_ld_coeff"] = 0.036
        cfg["env"]["hybrid"]["se_led_weight"] = 0.95
        cfg["env"]["hybrid"]["se_ld_weight"] = 1.10
        cfg["env"]["hybrid"]["eh_led_weight"] = 1.15
        cfg["env"]["hybrid"]["eh_ld_weight"] = 0.82
        cfg["env"]["hybrid"]["thermal_led_coeff"] = 1.10
        cfg["env"]["hybrid"]["thermal_ld_coeff"] = 1.65
        cfg["env"]["hybrid"]["burst_led_factor"] = 0.90
        cfg["env"]["hybrid"]["burst_ld_factor"] = 1.50

        cfg["sampler"]["attenuation_c_range"] = [0.21, 0.34]
        cfg["sampler"]["misalign_std_range"] = [0.09, 0.17]
        cfg["sampler"]["amb_temp_range"] = [31.0, 36.0]
        cfg["sampler"]["gamma_range"] = [0.06, 0.10]
        cfg["sampler"]["delta_range"] = [4.8, 6.8]

        cfg["safety"]["min_dwell_steps"] = 2
        cfg["safety"]["thermal_safe"] = 48.0
        cfg["safety"]["thermal_cutoff"] = 58.0
        cfg["safety"]["soft_alpha"] = 0.21
        cfg["safety"]["cutoff_alpha"] = 0.29
        cfg["safety"]["current_max"] = [3.1, 3.1, 3.1]
        cfg["safety"]["bus_current_max"] = 6.6
        cfg["safety"]["mask_floor"] = 0.02
        return finish()

    if scenario in {"hard_stress", "thermal_tight", "ld_adverse_hard"}:
        # LD-adverse hard (recommended for appendix/extreme stress):
        # intentionally harsher on LD sensitivity and burst vulnerability.
        cfg["env"]["attenuation_c"] = 0.31
        cfg["env"]["misalign_std"] = 0.16
        cfg["env"]["thermal_safe"] = 47.0
        cfg["env"]["thermal_cutoff"] = 57.0
        cfg["env"]["amb_temp"] = 35.0
        cfg["env"]["gamma"] = 0.09
        cfg["env"]["delta"] = 6.2
        cfg["env"]["se_weight"] = 1.12
        cfg["env"]["eh_weight"] = 0.30
        cfg["env"]["power_weight"] = 0.0025
        cfg["env"]["cost_weight"] = 2.8
        cfg["env"]["temporal_misalign_rho"] = 0.88
        cfg["env"]["attenuation_drift_rho"] = 0.92
        cfg["env"]["attenuation_drift_std"] = 0.03
        cfg["env"]["burst_prob"] = 0.08
        cfg["env"]["burst_strength_range"] = [0.16, 0.36]
        cfg["env"]["burst_decay"] = 0.88
        cfg["env"]["obs_bias_rho"] = 0.94
        cfg["env"]["obs_bias_step_std"] = 0.006

        cfg["env"]["hybrid"]["thermal_led_coeff"] = 1.10
        cfg["env"]["hybrid"]["thermal_ld_coeff"] = 1.75
        cfg["env"]["hybrid"]["misalign_led_scale"] = 1.40
        cfg["env"]["hybrid"]["misalign_ld_scale"] = 0.48
        cfg["env"]["hybrid"]["noise_led_coeff"] = 0.020
        cfg["env"]["hybrid"]["noise_ld_coeff"] = 0.040
        cfg["env"]["hybrid"]["burst_led_factor"] = 0.90
        cfg["env"]["hybrid"]["burst_ld_factor"] = 1.65

        cfg["sampler"]["attenuation_c_range"] = [0.24, 0.40]
        cfg["sampler"]["misalign_std_range"] = [0.12, 0.22]
        cfg["sampler"]["amb_temp_range"] = [33.0, 38.0]
        cfg["sampler"]["gamma_range"] = [0.075, 0.12]
        cfg["sampler"]["delta_range"] = [5.4, 7.2]

        cfg["safety"]["thermal_safe"] = 46.0
        cfg["safety"]["thermal_cutoff"] = 56.0
        cfg["safety"]["soft_alpha"] = 0.20
        cfg["safety"]["cutoff_alpha"] = 0.28
        cfg["safety"]["current_max"] = [3.2, 3.2, 3.2]
        cfg["safety"]["bus_current_max"] = 6.8
        # Keep LD boost softly available instead of collapsing too close to zero.
        cfg["safety"]["mask_floor"] = 0.06
        return finish()

    if scenario == "channel_harsh":
        cfg["env"]["attenuation_c"] = 0.32
        cfg["env"]["misalign_std"] = 0.17
        cfg["env"]["distances"] = [6.5, 7.5, 8.2]

        cfg["sampler"]["attenuation_c_range"] = [0.24, 0.46]
        cfg["sampler"]["misalign_std_range"] = [0.12, 0.25]

        # Make LED anchor robust, LD high gain but sensitive.
        cfg["env"]["hybrid"]["attenuation_led_factor"] = 1.16
        cfg["env"]["hybrid"]["attenuation_ld_factor"] = 0.92
        cfg["env"]["hybrid"]["misalign_led_scale"] = 1.35
        cfg["env"]["hybrid"]["misalign_ld_scale"] = 0.40
        cfg["env"]["hybrid"]["eta_led"] = 0.95
        cfg["env"]["hybrid"]["eta_ld"] = 1.10

        cfg["env"]["hybrid"]["noise_led_coeff"] = 0.018
        cfg["env"]["hybrid"]["noise_ld_coeff"] = 0.042
        cfg["env"]["hybrid"]["se_led_weight"] = 0.92
        cfg["env"]["hybrid"]["se_ld_weight"] = 1.08
        cfg["env"]["hybrid"]["eh_led_weight"] = 1.25
        cfg["env"]["hybrid"]["eh_ld_weight"] = 0.72

        cfg["env"]["thermal_safe"] = 57.0
        cfg["env"]["thermal_cutoff"] = 66.0
        cfg["env"]["amb_temp"] = 29.0
        cfg["env"]["gamma"] = 0.060
        cfg["env"]["delta"] = 3.6

        cfg["safety"]["thermal_safe"] = 56.0
        cfg["safety"]["thermal_cutoff"] = 65.0
        cfg["safety"]["current_max"] = [2.8, 2.8, 2.8]
        cfg["safety"]["bus_current_max"] = 6.2
        return finish()

    if scenario == "thermal_rebalanced":
        cfg["env"]["attenuation_c"] = 0.22
        cfg["env"]["misalign_std"] = 0.08
        cfg["env"]["distances"] = [4.6, 5.2, 5.8]
        cfg["env"]["thermal_safe"] = 44.0
        cfg["env"]["thermal_cutoff"] = 51.0
        cfg["env"]["amb_temp"] = 41.5
        cfg["env"]["gamma"] = 0.115
        cfg["env"]["delta"] = 7.9
        cfg["env"]["qos_min_rate"] = 0.008
        cfg["env"]["se_weight"] = 1.08
        cfg["env"]["eh_weight"] = 0.30
        cfg["env"]["power_weight"] = 0.0012
        cfg["env"]["cost_weight"] = 3.0
        cfg["env"]["burst_prob"] = 0.01
        cfg["env"]["burst_strength_range"] = [0.03, 0.08]
        cfg["env"]["burst_decay"] = 0.90
        cfg["env"]["obs_bias_rho"] = 0.40
        cfg["env"]["obs_bias_step_std"] = 0.001
        cfg["env"]["temporal_misalign_rho"] = 0.50
        cfg["env"]["attenuation_drift_rho"] = 0.70
        cfg["env"]["attenuation_drift_std"] = 0.010
        cfg["env"]["hybrid"]["thermal_led_coeff"] = 1.25
        cfg["env"]["hybrid"]["thermal_ld_coeff"] = 2.15

        cfg["sampler"]["attenuation_c_range"] = [0.20, 0.26]
        cfg["sampler"]["misalign_std_range"] = [0.06, 0.12]
        cfg["sampler"]["amb_temp_range"] = [40.5, 42.5]
        cfg["sampler"]["gamma_range"] = [0.10, 0.125]
        cfg["sampler"]["delta_range"] = [7.2, 8.4]

        cfg["safety"]["thermal_safe"] = 43.0
        cfg["safety"]["thermal_cutoff"] = 50.0
        cfg["safety"]["current_max"] = [3.5, 3.5, 3.5]
        cfg["safety"]["bus_current_max"] = 7.6
        cfg["safety"]["mask_floor"] = 0.06
        return finish()

    raise ValueError(f"Unknown scenario: {scenario}")


def convergence_stats(run_df: pd.DataFrame) -> Dict[str, float]:
    run_df = run_df.sort_values("iter")
    k = max(3, len(run_df) // 5)
    first = float(run_df["query_reward"].iloc[:k].mean())
    last = float(run_df["query_reward"].iloc[-k:].mean())

    tail = run_df.iloc[-k:]
    tail_cost = float(tail["query_cost"].mean())
    tail_violation = float(tail["query_violation_rate"].mean())

    return {
        "first_query_reward": first,
        "last_query_reward": last,
        "reward_gain": last - first,
        "tail_query_cost": tail_cost,
        "tail_query_violation": tail_violation,
    }


def sampler_snapshot(cfg: Dict) -> Dict[str, List[float]]:
    sampler = cfg.get("sampler", {})
    out = {}
    for k in ["attenuation_c_range", "misalign_std_range", "amb_temp_range", "gamma_range", "delta_range"]:
        if k in sampler:
            rng = sampler[k]
            out[k] = [float(rng[0]), float(rng[1])]
    out["task_source"] = infer_task_source(cfg)
    site_bank = sampler.get("site_bank", [])
    if site_bank:
        out["site_ids"] = [int(site["site_id"]) for site in site_bank]
    out["strict_site_bank"] = bool(sampler.get("strict_site_bank", False))
    out["balanced_sampling"] = bool(sampler.get("balanced_sampling", False))
    return out


def validate_training_config(cfg: Dict, scenario: str, *, strict_thermal: bool = True) -> Dict[str, object]:
    meta_cfg = cfg.get("meta", {})
    safety_cfg = cfg.get("safety", {})
    env_cfg = cfg.get("env", {})
    sampler_cfg = cfg.get("sampler", {})
    alignment_cfg = alignment_snapshot(cfg)
    dual_enabled = bool(meta_cfg.get("dual_enabled", True))

    expected_dual_lrs = np.asarray([0.02, 0.05, 0.05, 0.05], dtype=np.float32)
    actual_dual_lrs = np.asarray(meta_cfg.get("dual_lrs", []), dtype=np.float32).reshape(-1)
    actual_dual_lr = float(meta_cfg.get("dual_lr", float("nan")))
    env_safe = float(env_cfg.get("thermal_safe", float("nan")))
    env_cutoff = float(env_cfg.get("thermal_cutoff", float("nan")))
    safety_safe = float(safety_cfg.get("thermal_safe", float("nan")))
    safety_cutoff = float(safety_cfg.get("thermal_cutoff", float("nan")))
    site_bank_issues = validate_site_bank(sampler_cfg.get("site_bank", []))

    checks = {
        "scenario": scenario,
        "strict_thermal": bool(strict_thermal),
        "dual_lrs_expected": expected_dual_lrs.tolist(),
        "dual_lrs_actual": actual_dual_lrs.tolist(),
        "dual_lr_expected": 0.05,
        "dual_lr_actual": actual_dual_lr,
        "env_thermal_safe": env_safe,
        "env_thermal_cutoff": env_cutoff,
        "safety_thermal_safe": safety_safe,
        "safety_thermal_cutoff": safety_cutoff,
        "dual_enabled": bool(dual_enabled),
        "dual_lrs_match": (not dual_enabled)
        or bool(actual_dual_lrs.shape == expected_dual_lrs.shape and np.allclose(actual_dual_lrs, expected_dual_lrs)),
        "dual_lr_match": (not dual_enabled) or bool(np.isclose(actual_dual_lr, 0.05)),
        "safety_safe_earlier": bool(safety_safe < env_safe) if strict_thermal else bool(safety_safe <= env_safe),
        "safety_cutoff_earlier": bool(safety_cutoff < env_cutoff) if strict_thermal else bool(safety_cutoff <= env_cutoff),
        "task_source": str(alignment_cfg["task_source"]),
        "alignment_version": str(alignment_cfg["alignment_version"]),
        "task_summary_version": str(alignment_cfg["task_summary_version"]),
        "pre_alignment": bool(alignment_cfg["pre_alignment"]),
        "site_bank_valid": len(site_bank_issues) == 0,
        "site_bank_issues": site_bank_issues,
    }
    checks["all_passed"] = bool(
        checks["dual_lrs_match"]
        and checks["dual_lr_match"]
        and checks["safety_safe_earlier"]
        and checks["safety_cutoff_earlier"]
        and checks["site_bank_valid"]
        and checks["task_source"] == "site_bank"
        and checks["alignment_version"] == "system_model_v1"
        and checks["task_summary_version"] == "site_v2"
        and (checks["pre_alignment"] is False)
    )

    if not checks["all_passed"]:
        raise ValueError(f"Precheck failed for scenario={scenario}: {json.dumps(checks, ensure_ascii=False)}")
    return checks


def build_requested_metrics_table(stability_df: pd.DataFrame, scenario: str) -> pd.DataFrame:
    if stability_df.empty:
        return pd.DataFrame(
            columns=[
                "scenario",
                "variant",
                "reward_mean",
                "violation_rate_mean",
                "temp_max_mean",
                "temp_max_q90",
                "current_total_mean",
                "bus_utilization_mean",
                "bus_utilization_q90",
                "all3_active_fraction",
                "boost_any_active_fraction",
                "signal_ld_share_mean",
                "signal_ld_share_q90",
            ]
        )

    cols = [
        "variant",
        "reward_mean",
        "violation_rate_mean",
        "temp_max_mean",
        "temp_max_q90",
        "current_total_mean",
        "bus_utilization_mean",
        "bus_utilization_q90",
        "all3_active_fraction",
        "boost_any_active_fraction",
        "signal_ld_share_mean",
        "signal_ld_share_q90",
    ]
    out = stability_df[cols].copy()
    out.insert(0, "scenario", scenario)
    return out


def checkpoint_score_from_metrics(metrics: Dict[str, float], score_cfg: Dict | None = None, prefix: str = "") -> float:
    score_cfg = score_cfg or {}
    reward_w = float(score_cfg.get("reward_w", 1.0))
    se_w = float(score_cfg.get("se_w", 0.0))
    eh_w = float(score_cfg.get("eh_w", 0.0))
    cost_w = float(score_cfg.get("cost_w", 1.0))
    violation_w = float(score_cfg.get("violation_w", 1.0))
    return float(
        reward_w * float(metrics.get(f"{prefix}reward", 0.0))
        + se_w * float(metrics.get(f"{prefix}se", 0.0))
        + eh_w * float(metrics.get(f"{prefix}eh", 0.0))
        - cost_w * float(metrics.get(f"{prefix}cost", 0.0))
        - violation_w * float(metrics.get(f"{prefix}violation_rate", 0.0))
    )


def sample_fixed_tasks(cfg: Dict, seed: int, n_tasks: int, seed_offset: int) -> list:
    sampler = TaskSampler(
        copy.deepcopy(cfg["sampler"]),
        seed=int(seed) + int(seed_offset),
        task_defaults=task_defaults_from_cfg(cfg),
    )
    return sampler.sample(int(max(1, n_tasks)))


def select_checkpoint(
    run_df: pd.DataFrame,
    ckpt_dir: Path,
    score_cfg: Dict | None = None,
    evaluator=None,
) -> Dict[str, float | int | str | list | dict]:
    score_cfg = score_cfg or {}
    enabled = bool(score_cfg.get("enabled", False))
    mode = str(score_cfg.get("mode", "training_curve"))

    ckpt_paths = sorted(ckpt_dir.glob("iter_*.pt"))
    if not ckpt_paths:
        return {
            "strategy": "none",
            "selected_iter": -1,
            "selected_score": float("nan"),
            "selected_path": "",
            "selection_rows": [],
            "selected_metrics": {},
        }

    iter_to_path = {}
    for p in ckpt_paths:
        try:
            it = int(p.stem.split("_")[-1])
            iter_to_path[it] = p
        except ValueError:
            continue
    if not iter_to_path:
        return {
            "strategy": "none",
            "selected_iter": -1,
            "selected_score": float("nan"),
            "selected_path": "",
            "selection_rows": [],
            "selected_metrics": {},
        }

    if not enabled:
        best_iter = max(iter_to_path.keys())
        return {
            "strategy": "last_checkpoint",
            "selected_iter": int(best_iter),
            "selected_score": float("nan"),
            "selected_path": str(iter_to_path[best_iter]),
            "selection_rows": [],
            "selected_metrics": {},
        }

    if mode == "heldout_eval" and evaluator is not None:
        rows = []
        for it in sorted(iter_to_path.keys()):
            metrics = evaluator(iter_to_path[it])
            score = checkpoint_score_from_metrics(metrics, score_cfg)
            rows.append(
                {
                    "iter": int(it),
                    "reward": float(metrics.get("reward", 0.0)),
                    "se": float(metrics.get("se", 0.0)),
                    "eh": float(metrics.get("eh", 0.0)),
                    "cost": float(metrics.get("cost", 0.0)),
                    "violation_rate": float(metrics.get("violation_rate", 0.0)),
                    "score": float(score),
                }
            )
        if rows:
            best = max(rows, key=lambda r: float(r["score"]))
            best_iter = int(best["iter"])
            return {
                "strategy": "heldout_eval_score",
                "selected_iter": best_iter,
                "selected_score": float(best["score"]),
                "selected_path": str(iter_to_path[best_iter]),
                "selection_rows": rows,
                "selected_metrics": dict(best),
            }

    cand = run_df.copy()
    cand["iter_int"] = cand["iter"].round().astype(int)
    cand = cand[cand["iter_int"].isin(iter_to_path.keys())].copy()
    if cand.empty:
        best_iter = max(iter_to_path.keys())
        return {
            "strategy": "last_checkpoint_fallback",
            "selected_iter": int(best_iter),
            "selected_score": float("nan"),
            "selected_path": str(iter_to_path[best_iter]),
            "selection_rows": [],
            "selected_metrics": {},
        }

    cand["score"] = cand.apply(
        lambda row: checkpoint_score_from_metrics(
            {
                "query_reward": float(row["query_reward"]),
                "query_se": float(row["query_se"]),
                "query_eh": float(row["query_eh"]),
                "query_cost": float(row["query_cost"]),
                "query_violation_rate": float(row["query_violation_rate"]),
            },
            score_cfg,
            prefix="query_",
        ),
        axis=1,
    )
    best_row = cand.iloc[int(np.argmax(cand["score"].values))]
    best_iter = int(best_row["iter_int"])
    rows = [
        {
            "iter": int(r.iter_int),
            "reward": float(r.query_reward),
            "se": float(r.query_se),
            "eh": float(r.query_eh),
            "cost": float(r.query_cost),
            "violation_rate": float(r.query_violation_rate),
            "score": float(r.score),
        }
        for r in cand.itertuples()
    ]
    return {
        "strategy": "training_curve_score",
        "selected_iter": int(best_iter),
        "selected_score": float(best_row["score"]),
        "selected_path": str(iter_to_path[best_iter]),
        "selection_rows": rows,
        "selected_metrics": {
            "iter": int(best_iter),
            "reward": float(best_row["query_reward"]),
            "se": float(best_row["query_se"]),
            "eh": float(best_row["query_eh"]),
            "cost": float(best_row["query_cost"]),
            "violation_rate": float(best_row["query_violation_rate"]),
            "score": float(best_row["score"]),
        },
    }


def collect_env_data(
    trainer: MetaTrainer,
    cfg: Dict,
    scenario: str,
    variant: str,
    seed: int,
    tasks,
    episodes_per_task: int,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []

    for task_id, task in enumerate(tasks):
        env = MultiTxUwSliptEnv(cfg, overrides=task.to_env_overrides())

        for ep in range(episodes_per_task):
            obs, _ = env.reset(seed=seed + task_id * 100 + ep)
            trainer.agent.reset_episode_state()
            done = False
            step = 0

            while not done:
                z = trainer.agent.infer_z()
                temps_before = env.temps.copy().astype(np.float32)

                action, aux = trainer.agent.act(
                    obs=obs,
                    temps=temps_before,
                    amb_temp=env.amb_temp,
                    gamma=env.gamma,
                    delta=env.delta,
                    z=z,
                    eval_mode=True,
                )

                next_obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                currents_exec = np.asarray(info.get("currents_exec", action["currents_exec"]), dtype=np.float32)
                current_total = float(info.get("current_total", float(np.sum(currents_exec))))
                bus_current_max = float(info.get("bus_current_max", env.bus_current_max))
                row = {
                    "scenario": scenario,
                    "variant": variant,
                    "seed": float(seed),
                    "task_id": float(task_id),
                    "episode": float(ep),
                    "step": float(step),
                    "attenuation_c": float(env.attenuation_c),
                    "misalign_std": float(env.misalign_std),
                    "amb_temp": float(env.amb_temp),
                    "gamma": float(env.gamma),
                    "delta": float(env.delta),
                    "site_id": int(getattr(env, "site_id", -1)),
                    "task_source": str(getattr(env, "task_source", "global_fallback")),
                    "alignment_version": str(getattr(env, "alignment_version", "system_model_v1")),
                    "task_summary_version": str(getattr(env, "task_summary_version", "site_v2")),
                    "pre_alignment": bool(getattr(env, "pre_alignment", False)),
                    "distance_tx0": float(env.distances[0]),
                    "distance_tx1": float(env.distances[1]),
                    "distance_tx2": float(env.distances[2]),
                    "thermal_safe": float(env.thermal_safe),
                    "thermal_cutoff": float(env.thermal_cutoff),
                    "signal_ld_share": float(info["signal_ld_share"]),
                    "led_tx_fraction": float(info["led_tx_fraction"]),
                    "tx_enabled_fraction": float(info.get("tx_enabled_fraction", 1.0)),
                    "signal_led": float(info["signal_led"]),
                    "signal_ld": float(info["signal_ld"]),
                    "snr": float(info["snr"]),
                    "qos_rate": float(info["qos_rate"]),
                    "eh_metric": float(info["eh_metric"]),
                    "info_share": float(info["info_share"]),
                    "eh_share": float(info["eh_share"]),
                    "se": float(info["se"]),
                    "eh": float(info["eh"]),
                    "reward_id_term": float(info.get("reward_id_term", info.get("reward_se_term", info["se"]))),
                    "reward_se_term": float(info.get("reward_se_term", info["se"])),
                    "reward_eh_term": float(info.get("reward_eh_term", info["eh"])),
                    "reward_margin_term": float(info.get("reward_margin_term", 0.0)),
                    "penalty_cost_term": float(info.get("penalty_cost_term", 0.0)),
                    "penalty_power_term": float(info.get("penalty_power_term", 0.0)),
                    "penalty_smooth_term": float(info.get("penalty_smooth_term", 0.0)),
                    "penalty_switch_term": float(info.get("penalty_switch_term", 0.0)),
                    "mode_switch": float(info.get("mode_switch", 0.0)),
                    "boost_switch": float(info.get("boost_switch", 0.0)),
                    "mode_exec": float(info.get("mode_exec", action.get("mode_exec", 0))),
                    "boost_combo_exec": float(info.get("boost_combo_exec", action.get("boost_combo_exec", 0))),
                    "upper_idx_exec": float(info.get("upper_idx_exec", action.get("upper_idx_exec", 0))),
                    "cost": float(info["cost"]),
                    "cost_qos": float(info.get("cost_qos", 0.0)),
                    "cost_temp_anchor": float(info.get("cost_temp_anchor", 0.0)),
                    "cost_temp_boost1": float(info.get("cost_temp_boost1", 0.0)),
                    "cost_temp_boost2": float(info.get("cost_temp_boost2", 0.0)),
                    "thermal_violation": float(info["thermal_violation"]),
                    "temp_mean_before": float(np.mean(temps_before)),
                    "temp_mean_after": float(np.mean(info["temps"])),
                    "temp_max_after": float(np.max(info["temps"])),
                    "current_total": current_total,
                    "bus_current_max": bus_current_max,
                    "bus_utilization": float(current_total / max(bus_current_max, 1.0e-6)),
                }
                for tx_idx, current_val in enumerate(currents_exec.tolist()):
                    row[f"current_tx{tx_idx}"] = float(current_val)

                trainer.agent.episode.add(
                    {
                        "obs": obs.astype(np.float32),
                        "upper_idx_exec": float(info.get("upper_idx_exec", action.get("upper_idx_exec", 0))),
                        "boost_combo_exec": float(info.get("boost_combo_exec", action.get("boost_combo_exec", 0))),
                        "mode_exec": float(info.get("mode_exec", action.get("mode_exec", 0))),
                        "act_exec": aux["act_exec"].astype(np.float32),
                        "reward": float(reward),
                        "reward_raw": float(reward),
                        "cost": float(info["cost"]),
                        "cost_vec": np.asarray(info.get("cost_vec", [float(info["cost"])]), dtype=np.float32),
                        "task_params": build_task_summary_v2(
                            {
                                "attenuation_c": env.attenuation_c,
                                "misalign_std": env.misalign_std,
                                "amb_temp_env": env.amb_temp,
                                "gamma": env.gamma,
                                "delta": env.delta,
                                "qos_min_rate": env.qos_min_rate,
                                "distances": env.distances,
                            }
                        ),
                    }
                )

                rows.append(row)

                obs = next_obs
                step += 1

    return pd.DataFrame(rows)


def evaluate_on_tasks(
    trainer: MetaTrainer,
    cfg: Dict,
    tasks,
    episodes_per_task: int,
) -> Dict[str, float]:
    stats = []
    for task in tasks:
        env = MultiTxUwSliptEnv(cfg, overrides=task.to_env_overrides())
        for _ in range(episodes_per_task):
            stats.append(trainer._run_episode(env, train=False))

    return {
        "reward": float(np.mean([s.reward for s in stats])),
        "se": float(np.mean([s.se for s in stats])),
        "eh": float(np.mean([s.eh for s in stats])),
        "cost": float(np.mean([s.cost for s in stats])),
        "violation_rate": float(np.mean([s.violations for s in stats])),
        "len": float(np.mean([s.length for s in stats])),
    }


@dataclass
class SacLagEpisodeStats:
    reward: float
    se: float
    eh: float
    cost: float
    cost_vec: np.ndarray
    violations: float
    length: int
    se_term: float
    eh_term: float
    cost_term: float
    power_term: float
    smooth_term: float


def _logit01(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float32), 1.0e-4, 1.0 - 1.0e-4)
    return np.log(x / (1.0 - x)).astype(np.float32)


def _heuristic_macro_selection(env: MultiTxUwSliptEnv) -> tuple[int, int, np.ndarray, np.ndarray]:
    temps = env.temps.copy().astype(np.float32)
    channel = env.channel.copy().astype(np.float32)
    temp_margin = np.clip(
        (env.thermal_safe - temps) / max(env.thermal_safe - env.amb_temp, 1.0e-6),
        0.0,
        1.0,
    ).astype(np.float32)

    boost_scores = channel[1:] * temp_margin[1:]
    best_boost = int(np.argmax(boost_scores)) if boost_scores.size else 0
    best_score = float(boost_scores[best_boost]) if boost_scores.size else 0.0
    both_good = bool(boost_scores.size >= 2 and np.min(boost_scores) > 0.012 and np.mean(boost_scores) > 0.02)

    if both_good and float(np.mean(temp_margin)) > 0.35:
        boost_combo = 3
    elif best_score > 0.010:
        boost_combo = 1 + best_boost
    else:
        boost_combo = 0

    qos_gap = float(max(env.qos_min_rate - env.prev_qos_rate, 0.0))
    mean_channel = float(np.mean(channel))
    mean_margin = float(np.mean(temp_margin))
    if qos_gap > 0.03 or mean_channel < 0.030:
        mode = 1  # TS: prioritize information delivery
        rho_des, tau_des = 0.10, 0.92
    elif mean_margin > 0.45 and mean_channel > 0.055:
        mode = 2  # HY: balanced mode
        rho_des, tau_des = 0.22, 0.78
    else:
        mode = 0  # PS: conservative split
    return int(boost_combo), int(mode), channel, temp_margin


def heuristic_safe_action(env: MultiTxUwSliptEnv, trainer: MetaTrainer) -> tuple[Dict, Dict]:
    safety = trainer.agent.safety
    temps = env.temps.copy().astype(np.float32)
    boost_combo, mode, channel, temp_margin = _heuristic_macro_selection(env)
    mean_channel = float(np.mean(channel))
    mean_margin = float(np.mean(temp_margin))
    if mode == 1:
        rho_des, tau_des = 0.10, 0.92
    elif mode == 2:
        rho_des, tau_des = 0.22, 0.78
    else:
        rho_des, tau_des = 0.32, 1.0
    upper_idx_raw = int(boost_combo * 3 + mode)

    mask_nom = safety._boost_mask(boost_combo).astype(np.float32)
    weights = np.maximum(channel * (0.25 + 0.75 * temp_margin), 1.0e-4)
    weights[0] += 0.30  # Keep LED anchor as the primary stabilizing link.
    weights *= (0.15 + 0.85 * mask_nom)
    weights = np.maximum(weights, 1.0e-4)
    weights /= float(np.sum(weights))

    desired_total = float(
        env.bus_current_max
        * np.clip(0.18 + 0.30 * (mean_channel / (mean_channel + 0.05)) + 0.12 * mean_margin, 0.12, 0.48)
    )
    currents_des = desired_total * weights
    currents_des = np.minimum(currents_des, env.current_max.astype(np.float32))
    current_ratio = currents_des / np.maximum(env.current_max.astype(np.float32), 1.0e-6)
    lower_raw = np.concatenate(
        [
            _logit01(current_ratio),
            _logit01(np.asarray([rho_des], dtype=np.float32)),
            _logit01(np.asarray([min(max(tau_des, 0.0), 1.0)], dtype=np.float32)),
        ]
    ).astype(np.float32)

    safe, trainer.agent.safety_mem = safety.project_np(
        upper_idx_raw,
        lower_raw,
        temps=temps,
        amb_temp=env.amb_temp,
        gamma=env.gamma,
        delta=env.delta,
        mem=trainer.agent.safety_mem,
    )
    action = {
        "upper_idx": int(upper_idx_raw),
        "upper_idx_exec": int(safe["upper_idx_exec"]),
        "boost_combo_exec": int(safe["boost_combo_exec"]),
        "mode_exec": int(safe["mode_exec"]),
        "currents_exec": safe["currents_exec"],
        "rho_exec": np.asarray([safe["rho_exec"]], dtype=np.float32),
        "tau_exec": np.asarray([safe["tau_exec"]], dtype=np.float32),
    }
    aux = {
        "act_exec": np.concatenate(
            [safe["currents_exec"], np.asarray([safe["rho_exec"], safe["tau_exec"]], dtype=np.float32)]
        ).astype(np.float32)
    }
    return action, aux


class SacLagrangianBaseline:
    """Plain learning baseline: same hybrid structure, learned macro DQN + lower SAC-Lagrangian, without context/meta."""

    def __init__(self, cfg: Dict):
        self.cfg = copy.deepcopy(cfg)
        self.cfg.setdefault("context", {})["enabled"] = False
        self.cfg.setdefault("agent", {})["z_dim"] = 0
        self.cfg.setdefault("meta", {})["explicit_inner_outer"] = False
        self.cfg["meta"]["query_updates_enabled"] = False
        seed = int(self.cfg["experiment"]["seed"])
        set_seed(seed)

        requested_device = str(self.cfg["experiment"].get("device", "auto"))
        self.device = resolve_device(requested_device)
        self.cfg.setdefault("experiment", {})
        self.cfg["experiment"]["device_requested"] = requested_device
        self.cfg["experiment"]["device_resolved"] = str(self.device)

        self.safety = SafetyLayer(self.cfg)
        self.upper = UpperDQN(self.cfg, self.device)
        self.lower = LowerSAC(self.cfg, self.safety, self.device)
        self.replay = ReplayBuffer(int(self.cfg["buffer"]["replay_size"]))
        self.upper_replay = ReplayBuffer(int(self.cfg["buffer"]["replay_size"]))
        self.task_sampler = TaskSampler(
            copy.deepcopy(self.cfg["sampler"]),
            seed=seed,
            task_defaults=task_defaults_from_cfg(cfg),
        )
        self.dual = DualLayer.from_meta_cfg(self.cfg.get("meta", {}), n_tx=int(self.cfg["env"]["n_tx"]))
        self.dual_enabled = bool(self.cfg.get("meta", {}).get("dual_enabled", True))
        alignment_cfg = self.cfg.get("alignment", {})
        self.alignment_version = str(alignment_cfg.get("alignment_version", "system_model_v1"))
        self.task_summary_version = str(alignment_cfg.get("task_summary_version", "site_v2"))
        self.pre_alignment = bool(alignment_cfg.get("pre_alignment", False))
        self.loaded_alignment_meta = self._alignment_meta()

        self.batch_size = int(self.cfg["agent"]["batch_size"])
        self.warmup_steps = int(self.cfg["agent"]["warmup_steps"])
        self.lower_updates_per_step = int(self.cfg["agent"].get("lower_updates_per_step", 1))
        self.upper_update_every = int(self.cfg["agent"].get("upper_update_every", 1))
        self.upper_warmup_steps = int(
            self.cfg["agent"].get(
                "upper_warmup_steps",
                max(self.batch_size, self.warmup_steps // max(1, int(self.cfg["agent"].get("upper_hold_steps", 1)))),
            )
        )
        self.upper_hold_steps = int(self.cfg["agent"].get("upper_hold_steps", 1))
        self.z_dim = int(self.cfg["agent"]["z_dim"])

        log_dir = self.cfg["experiment"]["log_dir"]
        run_name = self.cfg["experiment"]["run_name"]
        self.logger = Logger(log_dir=log_dir, run_name=run_name)
        self.ckpt_dir = Path(log_dir) / run_name / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.safety_mem = {"current_boost": 0, "dwell_count": self.cfg["safety"]["min_dwell_steps"]}
        self.upper_mem = {"upper_idx": 0, "hold_left": 0}
        self.upper_plan = None

    def _alignment_meta(self, *, pre_alignment: bool | None = None) -> Dict[str, object]:
        return {
            "alignment_version": self.alignment_version,
            "task_summary_version": self.task_summary_version,
            "pre_alignment": bool(self.pre_alignment if pre_alignment is None else pre_alignment),
        }

    def is_formally_comparable(self) -> bool:
        return is_formally_comparable_record(self.loaded_alignment_meta)

    def reset_episode_state(self) -> None:
        self.safety_mem = {
            "current_boost": 0,
            "dwell_count": self.cfg["safety"]["min_dwell_steps"],
        }
        self.upper_mem = {"upper_idx": 0, "hold_left": 0}
        self.upper_plan = None

    def _empty_latent(self) -> np.ndarray:
        return np.zeros((self.z_dim,), dtype=np.float32)

    @staticmethod
    def _mean_metrics(metrics_list: list[Dict[str, float]]) -> Dict[str, float]:
        if not metrics_list:
            return {}
        keys = metrics_list[0].keys()
        return {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}

    def act(self, obs: np.ndarray, env: MultiTxUwSliptEnv, eval_mode: bool = False) -> tuple[Dict, Dict]:
        z = self._empty_latent()
        macro_new = self.upper_mem["hold_left"] <= 0
        exec_map = self.safety.raw_to_exec_map(self.safety_mem)
        if macro_new:
            if self.upper_plan is not None:
                upper_idx_raw = int(self.upper_plan)
                self.upper_plan = None
            else:
                upper_idx_raw = self.upper.select_action(
                    obs.astype(np.float32),
                    z,
                    t=self.global_step,
                    eval_mode=eval_mode,
                    exec_map=exec_map,
                )
            self.upper_mem["upper_idx"] = int(upper_idx_raw)
            self.upper_mem["hold_left"] = max(1, self.upper_hold_steps)
        else:
            upper_idx_raw = int(self.upper_mem["upper_idx"])
        self.upper_mem["hold_left"] = max(0, int(self.upper_mem["hold_left"]) - 1)

        temps_before = env.temps.copy().astype(np.float32)
        boost_preview, mode_preview = self.safety.preview_exec(upper_idx_raw, self.safety_mem)
        upper_idx_exec = self.safety.encode_exec(boost_preview, mode_preview)
        lower_raw = self.lower.select_action(obs, z, upper_idx=upper_idx_exec, eval_mode=eval_mode)

        safe, self.safety_mem = self.safety.project_np(
            upper_idx_raw,
            lower_raw,
            temps=temps_before,
            amb_temp=env.amb_temp,
            gamma=env.gamma,
            delta=env.delta,
            mem=self.safety_mem,
        )
        action = {
            "upper_idx": int(upper_idx_raw),
            "upper_idx_exec": int(safe["upper_idx_exec"]),
            "boost_combo_exec": int(safe["boost_combo_exec"]),
            "mode_exec": int(safe["mode_exec"]),
            "currents_exec": safe["currents_exec"],
            "rho_exec": np.asarray([safe["rho_exec"]], dtype=np.float32),
            "tau_exec": np.asarray([safe["tau_exec"]], dtype=np.float32),
        }
        aux = {
            "z": z,
            "upper_idx_raw": int(upper_idx_raw),
            "upper_idx_exec": int(safe["upper_idx_exec"]),
            "boost_combo_exec": int(safe["boost_combo_exec"]),
            "mode_exec": int(safe["mode_exec"]),
            "act_raw": lower_raw.astype(np.float32),
            "act_exec": np.concatenate(
                [safe["currents_exec"], np.asarray([safe["rho_exec"], safe["tau_exec"]], dtype=np.float32)]
            ).astype(np.float32),
            "macro_new": bool(macro_new),
            "hold_left": int(self.upper_mem["hold_left"]),
        }
        return action, aux

    def preview_next_macro(self, next_obs: np.ndarray, eval_mode: bool = False, commit_plan: bool = False) -> Dict[str, float]:
        next_exec_map = self.safety.raw_to_exec_map(self.safety_mem)
        macro_new_next = self.upper_mem["hold_left"] <= 0
        if macro_new_next:
            upper_idx_next_raw = int(
                self.upper.select_action(
                    next_obs.astype(np.float32),
                    self._empty_latent(),
                    t=self.global_step + 1,
                    eval_mode=eval_mode,
                    exec_map=next_exec_map,
                )
            )
            if commit_plan:
                self.upper_plan = upper_idx_next_raw
        else:
            upper_idx_next_raw = int(self.upper_mem["upper_idx"])
        boost_next, mode_next = self.safety.preview_exec(upper_idx_next_raw, self.safety_mem)
        return {
            "upper_idx_raw_next": float(upper_idx_next_raw),
            "upper_idx_exec_next": float(self.safety.encode_exec(boost_next, mode_next)),
            "boost_combo_exec_next": float(boost_next),
            "mode_exec_next": float(mode_next),
            "next_exec_map": next_exec_map.astype(np.float32),
        }

    def learn(self) -> Dict[str, float]:
        self.global_step += 1
        merged: Dict[str, float] = {}
        if len(self.replay) >= max(self.batch_size, self.warmup_steps):
            metrics = []
            for _ in range(max(1, self.lower_updates_per_step)):
                batch = self.replay.sample(self.batch_size)
                metrics.append(self.lower.update(batch))
            merged.update(self._mean_metrics(metrics))
        if (
            len(self.upper_replay) >= max(self.batch_size, self.upper_warmup_steps)
            and self.global_step % max(1, self.upper_update_every) == 0
        ):
            upper_batch = self.upper_replay.sample(self.batch_size)
            merged.update(self.upper.update(upper_batch))
        return merged

    def _run_episode(self, env: MultiTxUwSliptEnv, train: bool) -> SacLagEpisodeStats:
        obs, _ = env.reset()
        self.reset_episode_state()

        ep_reward = 0.0
        ep_se = 0.0
        ep_eh = 0.0
        ep_cost = 0.0
        ep_cost_vec = np.zeros(self.dual.n_constraints, dtype=np.float32)
        ep_viol = 0.0
        ep_len = 0
        ep_se_term = 0.0
        ep_eh_term = 0.0
        ep_cost_term = 0.0
        ep_power_term = 0.0
        ep_smooth_term = 0.0

        macro_start_obs = None
        macro_start_z = None
        macro_upper_idx_raw = 0.0
        macro_upper_idx_exec = 0.0
        macro_reward = 0.0
        macro_steps = 0

        done = False
        while not done:
            temps_before = env.temps.copy().astype(np.float32)
            action, aux = self.act(obs, env, eval_mode=not train)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_len += 1

            cost = float(info["cost"])
            cost_vec = np.asarray(info.get("cost_vec", [cost]), dtype=np.float32).reshape(-1)
            dual_penalty = self.dual.penalty(cost_vec) if self.dual_enabled else 0.0
            penalized_reward = float(reward - dual_penalty)

            lower_transition = {
                "obs": obs.astype(np.float32),
                "next_obs": next_obs.astype(np.float32),
                "upper_idx_raw": float(aux["upper_idx_raw"]),
                "upper_idx_exec": float(aux["upper_idx_exec"]),
                "reward": penalized_reward,
                "reward_raw": float(reward),
                "done": float(done),
                "z": self._empty_latent(),
                "act_exec": aux["act_exec"].astype(np.float32),
                "act_raw": aux["act_raw"].astype(np.float32),
                "boost_combo_exec": float(aux["boost_combo_exec"]),
                "mode_exec": float(aux["mode_exec"]),
                "temps": temps_before.astype(np.float32),
                "next_temps": info["temps"].astype(np.float32),
                "amb_temp": float(info["amb_temp"]),
                "amb_temp_env": float(info["amb_temp"]),
                "gamma_env": float(info["gamma"]),
                "delta_env": float(info["delta"]),
                "attenuation_c_env": float(env.attenuation_c),
                "misalign_std_env": float(env.misalign_std),
                "qos_min_rate_env": float(env.qos_min_rate),
                "site_id_env": int(info.get("site_id", getattr(env, "site_id", -1))),
                "distances_env": np.asarray(env.distances, dtype=np.float32).copy(),
                "cost": cost,
                "cost_vec": cost_vec.astype(np.float32),
            }

            if bool(aux.get("macro_new", False)) or macro_start_obs is None:
                macro_start_obs = obs.astype(np.float32)
                macro_start_z = self._empty_latent()
                macro_upper_idx_raw = float(aux["upper_idx_raw"])
                macro_upper_idx_exec = float(aux["upper_idx_exec"])
                macro_reward = 0.0
                macro_steps = 0

            macro_reward += penalized_reward
            macro_steps += 1
            macro_done = bool(done)
            macro_end = macro_done or (int(aux.get("hold_left", 0)) <= 0)

            if train:
                if done:
                    next_macro_info = {
                        "upper_idx_raw_next": float(aux["upper_idx_raw"]),
                        "upper_idx_exec_next": float(aux["upper_idx_exec"]),
                        "boost_combo_exec_next": float(aux["boost_combo_exec"]),
                        "mode_exec_next": float(aux["mode_exec"]),
                        "next_exec_map": np.arange(int(self.cfg["agent"]["n_upper_actions"]), dtype=np.float32),
                    }
                else:
                    next_macro_info = self.preview_next_macro(next_obs=next_obs, eval_mode=False, commit_plan=True)
                lower_transition["z_next"] = self._empty_latent()
                lower_transition.update(next_macro_info)
                self.replay.add(lower_transition)
                if macro_end and macro_start_obs is not None and macro_start_z is not None:
                    upper_transition = {
                        "obs": macro_start_obs,
                        "next_obs": next_obs.astype(np.float32),
                        "upper_idx_raw": macro_upper_idx_raw,
                        "upper_idx_exec": macro_upper_idx_exec,
                        "reward": float(macro_reward),
                        "done": float(macro_done),
                        "z": macro_start_z.astype(np.float32),
                        "z_next": self._empty_latent(),
                        "horizon": float(macro_steps),
                        "next_exec_map": np.asarray(next_macro_info["next_exec_map"], dtype=np.float32),
                    }
                    self.upper_replay.add(upper_transition)
                self.learn()

            ep_reward += penalized_reward
            ep_se += float(info["se"])
            ep_eh += float(info["eh"])
            ep_cost += cost
            ep_cost_vec += cost_vec
            ep_viol += float(np.any(cost_vec > 0.0))
            ep_se_term += float(info.get("reward_se_term", info["se"]))
            ep_eh_term += float(info.get("reward_eh_term", info["eh"]))
            ep_cost_term += float(info.get("penalty_cost_term", 0.0))
            ep_power_term += float(info.get("penalty_power_term", 0.0))
            ep_smooth_term += float(info.get("penalty_smooth_term", 0.0))
            obs = next_obs

        return SacLagEpisodeStats(
            reward=ep_reward,
            se=ep_se / max(ep_len, 1),
            eh=ep_eh / max(ep_len, 1),
            cost=ep_cost / max(ep_len, 1),
            cost_vec=ep_cost_vec / max(ep_len, 1),
            violations=ep_viol / max(ep_len, 1),
            length=ep_len,
            se_term=ep_se_term / max(ep_len, 1),
            eh_term=ep_eh_term / max(ep_len, 1),
            cost_term=ep_cost_term / max(ep_len, 1),
            power_term=ep_power_term / max(ep_len, 1),
            smooth_term=ep_smooth_term / max(ep_len, 1),
        )

    def train(self, meta_iters: int | None = None) -> Path:
        meta_cfg = self.cfg["meta"]
        meta_iters = int(meta_iters or meta_cfg["meta_iters"])
        episodes_per_task = max(1, int(meta_cfg.get("support_episodes", 0)) + int(meta_cfg.get("query_episodes", 0)))

        for it in range(1, meta_iters + 1):
            tasks = self.task_sampler.sample(int(meta_cfg["n_tasks_per_iter"]))
            train_stats: List[SacLagEpisodeStats] = []

            for task in tasks:
                env = MultiTxUwSliptEnv(self.cfg, overrides=task.to_env_overrides())
                task_stats: List[SacLagEpisodeStats] = []
                for _ in range(episodes_per_task):
                    ep_stats = self._run_episode(env, train=True)
                    train_stats.append(ep_stats)
                    task_stats.append(ep_stats)
                if self.dual_enabled and task_stats:
                    task_mean_cost_vec = np.mean(np.stack([s.cost_vec for s in task_stats], axis=0), axis=0)
                    self.dual.update(task_mean_cost_vec)

            if train_stats:
                mean_cost_vec = np.mean(np.stack([s.cost_vec for s in train_stats], axis=0), axis=0)
            else:
                mean_cost_vec = np.zeros(self.dual.n_constraints, dtype=np.float32)

            row = {
                "iter": float(it),
                "support_reward": float(np.mean([s.reward for s in train_stats])) if train_stats else 0.0,
                "support_se": float(np.mean([s.se for s in train_stats])) if train_stats else 0.0,
                "support_eh": float(np.mean([s.eh for s in train_stats])) if train_stats else 0.0,
                "support_cost": float(np.mean([s.cost for s in train_stats])) if train_stats else 0.0,
                "support_violation_rate": float(np.mean([s.violations for s in train_stats])) if train_stats else 0.0,
                "support_se_term": float(np.mean([s.se_term for s in train_stats])) if train_stats else 0.0,
                "support_eh_term": float(np.mean([s.eh_term for s in train_stats])) if train_stats else 0.0,
                "support_cost_term": float(np.mean([s.cost_term for s in train_stats])) if train_stats else 0.0,
                "support_power_term": float(np.mean([s.power_term for s in train_stats])) if train_stats else 0.0,
                "support_smooth_term": float(np.mean([s.smooth_term for s in train_stats])) if train_stats else 0.0,
                "query_reward": float(np.mean([s.reward for s in train_stats])) if train_stats else 0.0,
                "query_se": float(np.mean([s.se for s in train_stats])) if train_stats else 0.0,
                "query_eh": float(np.mean([s.eh for s in train_stats])) if train_stats else 0.0,
                "query_cost": float(np.mean([s.cost for s in train_stats])) if train_stats else 0.0,
                "query_violation_rate": float(np.mean([s.violations for s in train_stats])) if train_stats else 0.0,
                "query_se_term": float(np.mean([s.se_term for s in train_stats])) if train_stats else 0.0,
                "query_eh_term": float(np.mean([s.eh_term for s in train_stats])) if train_stats else 0.0,
                "query_cost_term": float(np.mean([s.cost_term for s in train_stats])) if train_stats else 0.0,
                "query_power_term": float(np.mean([s.power_term for s in train_stats])) if train_stats else 0.0,
                "query_smooth_term": float(np.mean([s.smooth_term for s in train_stats])) if train_stats else 0.0,
                "lambda": float(np.mean(self.dual.values)) if self.dual_enabled else 0.0,
                "curriculum_stage": "base",
                "outer_step_size": 0.0,
            }
            for idx, name in enumerate(self.dual.names):
                row[f"support_cost_{name}"] = float(mean_cost_vec[idx])
                row[f"query_cost_{name}"] = float(mean_cost_vec[idx])
            if self.dual_enabled:
                row.update(self.dual.as_dict())
            else:
                row.update({f"lambda_{name}": 0.0 for name in self.dual.names})
            self.logger.log(row)

            if it % 10 == 0 or it == meta_iters:
                self.save(self.ckpt_dir / f"iter_{it}.pt")

        return self.logger.csv_path

    def save(self, ckpt_path: str | Path) -> None:
        dual_state = self.dual.state_dict()
        dual_state_safe = {
            **dual_state,
            "names": list(dual_state["names"]),
            "target_costs": np.asarray(dual_state["target_costs"], dtype=np.float32).tolist(),
            "lrs": np.asarray(dual_state["lrs"], dtype=np.float32).tolist(),
            "max_lambdas": np.asarray(dual_state["max_lambdas"], dtype=np.float32).tolist(),
            "values": np.asarray(dual_state["values"], dtype=np.float32).tolist(),
        }
        ckpt = {
            "upper": self.upper.state_dict(),
            "lower": self.lower.state_dict(),
            "dual": dual_state_safe,
            "global_step": int(self.global_step),
            "alignment_meta": self._alignment_meta(pre_alignment=False),
        }
        torch.save(ckpt, ckpt_path)

    def load(self, ckpt_path: str | Path) -> None:
        ckpt = torch.load(ckpt_path, map_location=self.device)
        if "upper" in ckpt:
            self.upper.load_state_dict(ckpt["upper"])
        self.lower.load_state_dict(ckpt["lower"])
        if "dual" in ckpt:
            self.dual.load_state_dict(ckpt["dual"])
        self.loaded_alignment_meta = dict(ckpt.get("alignment_meta", self._alignment_meta(pre_alignment=True)))
        if "pre_alignment" not in self.loaded_alignment_meta:
            self.loaded_alignment_meta["pre_alignment"] = True
        self.global_step = int(ckpt.get("global_step", 0))


class Shin2024MatchedBaseline(SacLagrangianBaseline):
    """Matched hierarchical DQN-DDPG baseline under the current benchmark semantics."""

    def __init__(self, cfg: Dict):
        self.cfg = copy.deepcopy(cfg)
        self.cfg.setdefault("context", {})["enabled"] = False
        self.cfg.setdefault("agent", {})["z_dim"] = 0
        self.cfg.setdefault("meta", {})["explicit_inner_outer"] = False
        self.cfg["meta"]["query_updates_enabled"] = False
        self.cfg["meta"]["dual_enabled"] = False
        seed = int(self.cfg["experiment"]["seed"])
        set_seed(seed)

        requested_device = str(self.cfg["experiment"].get("device", "auto"))
        self.device = resolve_device(requested_device)
        self.cfg.setdefault("experiment", {})
        self.cfg["experiment"]["device_requested"] = requested_device
        self.cfg["experiment"]["device_resolved"] = str(self.device)

        self.safety = SafetyLayer(self.cfg)
        self.upper = UpperDQN(self.cfg, self.device)
        self.lower = LowerDDPG(self.cfg, self.safety, self.device)
        upper_replay_size = int(self.cfg.get("upper_dqn", {}).get("replay_size", 2000))
        lower_replay_size = int(self.cfg.get("lower_ddpg", {}).get("replay_size", 1.0e6))
        self.replay = ReplayBuffer(lower_replay_size)
        self.upper_replay = ReplayBuffer(upper_replay_size)
        self.task_sampler = TaskSampler(
            copy.deepcopy(self.cfg["sampler"]),
            seed=seed,
            task_defaults=task_defaults_from_cfg(cfg),
        )
        self.dual = DualLayer.from_meta_cfg(self.cfg.get("meta", {}), n_tx=int(self.cfg["env"]["n_tx"]))
        self.dual_enabled = False
        alignment_cfg = self.cfg.get("alignment", {})
        self.alignment_version = str(alignment_cfg.get("alignment_version", "system_model_v1"))
        self.task_summary_version = str(alignment_cfg.get("task_summary_version", "site_v2"))
        self.pre_alignment = bool(alignment_cfg.get("pre_alignment", False))
        self.loaded_alignment_meta = self._alignment_meta()

        lower_batch = int(self.cfg.get("lower_ddpg", {}).get("batch_size", self.cfg["agent"]["batch_size"]))
        self.batch_size = lower_batch
        self.warmup_steps = int(self.cfg["agent"]["warmup_steps"])
        self.lower_updates_per_step = int(self.cfg["agent"].get("lower_updates_per_step", 1))
        self.upper_update_every = int(self.cfg["agent"].get("upper_update_every", 1))
        self.upper_warmup_steps = int(
            self.cfg["agent"].get(
                "upper_warmup_steps",
                max(lower_batch, self.warmup_steps // max(1, int(self.cfg["agent"].get("upper_hold_steps", 1)))),
            )
        )
        self.upper_hold_steps = int(self.cfg["agent"].get("upper_hold_steps", 1))
        self.z_dim = int(self.cfg["agent"]["z_dim"])

        log_dir = self.cfg["experiment"]["log_dir"]
        run_name = self.cfg["experiment"]["run_name"]
        self.logger = Logger(log_dir=log_dir, run_name=run_name)
        self.ckpt_dir = Path(log_dir) / run_name / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.safety_mem = {"current_boost": 0, "dwell_count": self.cfg["safety"]["min_dwell_steps"]}
        self.upper_mem = {"upper_idx": 0, "hold_left": 0}
        self.upper_plan = None

    @staticmethod
    def _force_hy_index(raw_idx: int) -> int:
        raw_idx = int(np.clip(raw_idx, 0, 11))
        return int((raw_idx // 3) * 3 + 2)

    def _hy_exec_map(self) -> np.ndarray:
        exec_map = np.zeros(int(self.cfg["agent"]["n_upper_actions"]), dtype=np.int64)
        for raw_idx in range(exec_map.shape[0]):
            hy_idx = self._force_hy_index(raw_idx)
            boost_exec, _ = self.safety.preview_exec(hy_idx, mem=self.safety_mem)
            exec_map[raw_idx] = self.safety.encode_exec(boost_exec, 2)
        return exec_map

    def act(self, obs: np.ndarray, env: MultiTxUwSliptEnv, eval_mode: bool = False) -> tuple[Dict, Dict]:
        z = self._empty_latent()
        macro_new = self.upper_mem["hold_left"] <= 0
        exec_map = self._hy_exec_map()
        if macro_new:
            if self.upper_plan is not None:
                upper_idx_raw = int(self.upper_plan)
                self.upper_plan = None
            else:
                upper_idx_raw = self.upper.select_action(
                    obs.astype(np.float32),
                    z,
                    t=self.global_step,
                    eval_mode=eval_mode,
                    exec_map=exec_map,
                )
            upper_idx_raw = self._force_hy_index(upper_idx_raw)
            self.upper_mem["upper_idx"] = int(upper_idx_raw)
            self.upper_mem["hold_left"] = max(1, self.upper_hold_steps)
        else:
            upper_idx_raw = self._force_hy_index(int(self.upper_mem["upper_idx"]))
        self.upper_mem["hold_left"] = max(0, int(self.upper_mem["hold_left"]) - 1)

        temps_before = env.temps.copy().astype(np.float32)
        boost_preview, _ = self.safety.preview_exec(upper_idx_raw, self.safety_mem)
        upper_idx_exec = self.safety.encode_exec(boost_preview, 2)
        lower_raw = self.lower.select_action(obs, z, upper_idx=upper_idx_exec, eval_mode=eval_mode)

        safe, self.safety_mem = self.safety.project_np(
            upper_idx_raw,
            lower_raw,
            temps=temps_before,
            amb_temp=env.amb_temp,
            gamma=env.gamma,
            delta=env.delta,
            mem=self.safety_mem,
        )
        if int(safe["mode_exec"]) != 2:
            raise RuntimeError("shin2024_matched contract violated: executed mode must be HY")
        action = {
            "upper_idx": int(upper_idx_raw),
            "upper_idx_exec": int(safe["upper_idx_exec"]),
            "boost_combo_exec": int(safe["boost_combo_exec"]),
            "mode_exec": int(safe["mode_exec"]),
            "currents_exec": safe["currents_exec"],
            "rho_exec": np.asarray([safe["rho_exec"]], dtype=np.float32),
            "tau_exec": np.asarray([safe["tau_exec"]], dtype=np.float32),
        }
        aux = {
            "z": z,
            "upper_idx_raw": int(upper_idx_raw),
            "upper_idx_exec": int(safe["upper_idx_exec"]),
            "boost_combo_exec": int(safe["boost_combo_exec"]),
            "mode_exec": int(safe["mode_exec"]),
            "act_raw": lower_raw.astype(np.float32),
            "act_exec": np.concatenate(
                [safe["currents_exec"], np.asarray([safe["rho_exec"], safe["tau_exec"]], dtype=np.float32)]
            ).astype(np.float32),
            "macro_new": bool(macro_new),
            "hold_left": int(self.upper_mem["hold_left"]),
        }
        return action, aux

    def preview_next_macro(self, next_obs: np.ndarray, eval_mode: bool = False, commit_plan: bool = False) -> Dict[str, float]:
        next_exec_map = self._hy_exec_map()
        macro_new_next = self.upper_mem["hold_left"] <= 0
        if macro_new_next:
            upper_idx_next_raw = int(
                self.upper.select_action(
                    next_obs.astype(np.float32),
                    self._empty_latent(),
                    t=self.global_step + 1,
                    eval_mode=eval_mode,
                    exec_map=next_exec_map,
                )
            )
            upper_idx_next_raw = self._force_hy_index(upper_idx_next_raw)
            if commit_plan:
                self.upper_plan = upper_idx_next_raw
        else:
            upper_idx_next_raw = self._force_hy_index(int(self.upper_mem["upper_idx"]))
        boost_next, _ = self.safety.preview_exec(upper_idx_next_raw, self.safety_mem)
        return {
            "upper_idx_raw_next": float(upper_idx_next_raw),
            "upper_idx_exec_next": float(self.safety.encode_exec(boost_next, 2)),
            "boost_combo_exec_next": float(boost_next),
            "mode_exec_next": 2.0,
            "next_exec_map": next_exec_map.astype(np.float32),
        }

def _run_heuristic_episode(trainer: MetaTrainer, env: MultiTxUwSliptEnv) -> Dict[str, float | np.ndarray]:
    obs, _ = env.reset()
    trainer.agent.reset_episode_state()
    done = False
    ep_reward = ep_se = ep_eh = ep_cost = ep_viol = 0.0
    ep_len = 0

    while not done:
        action, aux = heuristic_safe_action(env, trainer)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        trainer.agent.episode.add(
            {
                "obs": obs.astype(np.float32),
                "act_exec": aux["act_exec"].astype(np.float32),
                "reward": float(reward),
                "reward_raw": float(reward),
                "cost": float(info["cost"]),
                "cost_vec": np.asarray(info.get("cost_vec", [float(info["cost"])]), dtype=np.float32),
                "task_params": build_task_summary_v2(
                    {
                        "attenuation_c": env.attenuation_c,
                        "misalign_std": env.misalign_std,
                        "amb_temp_env": env.amb_temp,
                        "gamma": env.gamma,
                        "delta": env.delta,
                        "qos_min_rate": env.qos_min_rate,
                        "distances": env.distances,
                    }
                ),
            }
        )
        obs = next_obs
        ep_reward += float(reward)
        ep_se += float(info["se"])
        ep_eh += float(info["eh"])
        ep_cost += float(info["cost"])
        ep_viol += float(np.any(np.asarray(info.get("cost_vec", [info["cost"]]), dtype=np.float32) > 0.0))
        ep_len += 1

    return {
        "reward": ep_reward / max(ep_len, 1),
        "se": ep_se / max(ep_len, 1),
        "eh": ep_eh / max(ep_len, 1),
        "cost": ep_cost / max(ep_len, 1),
        "violation_rate": ep_viol / max(ep_len, 1),
        "len": float(ep_len),
    }


def evaluate_heuristic_on_tasks(
    trainer: MetaTrainer,
    cfg: Dict,
    tasks,
    episodes_per_task: int,
) -> Dict[str, float]:
    stats = []
    for task in tasks:
        env = MultiTxUwSliptEnv(cfg, overrides=task.to_env_overrides())
        for _ in range(episodes_per_task):
            stats.append(_run_heuristic_episode(trainer, env))
    return {
        "reward": float(np.mean([s["reward"] for s in stats])),
        "se": float(np.mean([s["se"] for s in stats])),
        "eh": float(np.mean([s["eh"] for s in stats])),
        "cost": float(np.mean([s["cost"] for s in stats])),
        "violation_rate": float(np.mean([s["violation_rate"] for s in stats])),
        "len": float(np.mean([s["len"] for s in stats])),
    }


def evaluate_plain_hierarchical_baseline_on_tasks(
    trainer: SacLagrangianBaseline,
    cfg: Dict,
    tasks,
    episodes_per_task: int,
) -> Dict[str, float]:
    stats: List[SacLagEpisodeStats] = []
    for task in tasks:
        env = MultiTxUwSliptEnv(cfg, overrides=task.to_env_overrides())
        for _ in range(episodes_per_task):
            stats.append(trainer._run_episode(env, train=False))
    return {
        "reward": float(np.mean([s.reward for s in stats])),
        "se": float(np.mean([s.se for s in stats])),
        "eh": float(np.mean([s.eh for s in stats])),
        "cost": float(np.mean([s.cost for s in stats])),
        "violation_rate": float(np.mean([s.violations for s in stats])),
        "len": float(np.mean([s.length for s in stats])),
    }


def evaluate_sac_lagrangian_on_tasks(
    trainer: SacLagrangianBaseline,
    cfg: Dict,
    tasks,
    episodes_per_task: int,
) -> Dict[str, float]:
    return evaluate_plain_hierarchical_baseline_on_tasks(trainer, cfg, tasks, episodes_per_task)


def collect_env_data_heuristic(
    trainer: MetaTrainer,
    cfg: Dict,
    scenario: str,
    variant: str,
    seed: int,
    tasks,
    episodes_per_task: int,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []

    for task_id, task in enumerate(tasks):
        env = MultiTxUwSliptEnv(cfg, overrides=task.to_env_overrides())
        for ep in range(episodes_per_task):
            obs, _ = env.reset(seed=seed + task_id * 100 + ep)
            trainer.agent.reset_episode_state()
            done = False
            step = 0
            while not done:
                temps_before = env.temps.copy().astype(np.float32)
                action, aux = heuristic_safe_action(env, trainer)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                currents_exec = np.asarray(info.get("currents_exec", action["currents_exec"]), dtype=np.float32)
                current_total = float(info.get("current_total", float(np.sum(currents_exec))))
                bus_current_max = float(info.get("bus_current_max", env.bus_current_max))
                row = {
                    "scenario": scenario,
                    "variant": variant,
                    "seed": float(seed),
                    "task_id": float(task_id),
                    "episode": float(ep),
                    "step": float(step),
                    "attenuation_c": float(env.attenuation_c),
                    "misalign_std": float(env.misalign_std),
                    "amb_temp": float(env.amb_temp),
                    "gamma": float(env.gamma),
                    "delta": float(env.delta),
                    "site_id": int(getattr(env, "site_id", -1)),
                    "task_source": str(getattr(env, "task_source", "global_fallback")),
                    "alignment_version": str(getattr(env, "alignment_version", "system_model_v1")),
                    "task_summary_version": str(getattr(env, "task_summary_version", "site_v2")),
                    "pre_alignment": bool(getattr(env, "pre_alignment", False)),
                    "distance_tx0": float(env.distances[0]),
                    "distance_tx1": float(env.distances[1]),
                    "distance_tx2": float(env.distances[2]),
                    "thermal_safe": float(env.thermal_safe),
                    "thermal_cutoff": float(env.thermal_cutoff),
                    "signal_ld_share": float(info["signal_ld_share"]),
                    "led_tx_fraction": float(info["led_tx_fraction"]),
                    "tx_enabled_fraction": float(info.get("tx_enabled_fraction", 1.0)),
                    "signal_led": float(info["signal_led"]),
                    "signal_ld": float(info["signal_ld"]),
                    "snr": float(info["snr"]),
                    "qos_rate": float(info["qos_rate"]),
                    "eh_metric": float(info["eh_metric"]),
                    "info_share": float(info["info_share"]),
                    "eh_share": float(info["eh_share"]),
                    "se": float(info["se"]),
                    "eh": float(info["eh"]),
                    "reward_id_term": float(info.get("reward_id_term", info.get("reward_se_term", info["se"]))),
                    "reward_se_term": float(info.get("reward_se_term", info["se"])),
                    "reward_eh_term": float(info.get("reward_eh_term", info["eh"])),
                    "reward_margin_term": float(info.get("reward_margin_term", 0.0)),
                    "penalty_cost_term": float(info.get("penalty_cost_term", 0.0)),
                    "penalty_power_term": float(info.get("penalty_power_term", 0.0)),
                    "penalty_smooth_term": float(info.get("penalty_smooth_term", 0.0)),
                    "penalty_switch_term": float(info.get("penalty_switch_term", 0.0)),
                    "mode_switch": float(info.get("mode_switch", 0.0)),
                    "boost_switch": float(info.get("boost_switch", 0.0)),
                    "mode_exec": float(info.get("mode_exec", action.get("mode_exec", 0))),
                    "boost_combo_exec": float(info.get("boost_combo_exec", action.get("boost_combo_exec", 0))),
                    "upper_idx_exec": float(info.get("upper_idx_exec", action.get("upper_idx_exec", 0))),
                    "cost": float(info["cost"]),
                    "cost_qos": float(info.get("cost_qos", 0.0)),
                    "cost_temp_anchor": float(info.get("cost_temp_anchor", 0.0)),
                    "cost_temp_boost1": float(info.get("cost_temp_boost1", 0.0)),
                    "cost_temp_boost2": float(info.get("cost_temp_boost2", 0.0)),
                    "thermal_violation": float(info["thermal_violation"]),
                    "temp_mean_before": float(np.mean(temps_before)),
                    "temp_mean_after": float(np.mean(info["temps"])),
                    "temp_max_after": float(np.max(info["temps"])),
                    "current_total": current_total,
                    "bus_current_max": bus_current_max,
                    "bus_utilization": float(current_total / max(bus_current_max, 1.0e-6)),
                }
                for tx_idx, current_val in enumerate(currents_exec.tolist()):
                    row[f"current_tx{tx_idx}"] = float(current_val)
                rows.append(row)
                obs = next_obs
                step += 1
    return pd.DataFrame(rows)


def collect_env_data_plain_hierarchical_baseline(
    trainer: SacLagrangianBaseline,
    cfg: Dict,
    scenario: str,
    variant: str,
    seed: int,
    tasks,
    episodes_per_task: int,
) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []

    for task_id, task in enumerate(tasks):
        env = MultiTxUwSliptEnv(cfg, overrides=task.to_env_overrides())
        for ep in range(episodes_per_task):
            obs, _ = env.reset(seed=seed + task_id * 100 + ep)
            trainer.reset_episode_state()
            done = False
            step = 0
            while not done:
                temps_before = env.temps.copy().astype(np.float32)
                action, aux = trainer.act(obs, env, eval_mode=True)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                currents_exec = np.asarray(info.get("currents_exec", action["currents_exec"]), dtype=np.float32)
                current_total = float(info.get("current_total", float(np.sum(currents_exec))))
                bus_current_max = float(info.get("bus_current_max", env.bus_current_max))
                row = {
                    "scenario": scenario,
                    "variant": variant,
                    "seed": float(seed),
                    "task_id": float(task_id),
                    "episode": float(ep),
                    "step": float(step),
                    "attenuation_c": float(env.attenuation_c),
                    "misalign_std": float(env.misalign_std),
                    "amb_temp": float(env.amb_temp),
                    "gamma": float(env.gamma),
                    "delta": float(env.delta),
                    "site_id": int(getattr(env, "site_id", -1)),
                    "task_source": str(getattr(env, "task_source", "global_fallback")),
                    "alignment_version": str(getattr(env, "alignment_version", "system_model_v1")),
                    "task_summary_version": str(getattr(env, "task_summary_version", "site_v2")),
                    "pre_alignment": bool(getattr(env, "pre_alignment", False)),
                    "distance_tx0": float(env.distances[0]),
                    "distance_tx1": float(env.distances[1]),
                    "distance_tx2": float(env.distances[2]),
                    "thermal_safe": float(env.thermal_safe),
                    "thermal_cutoff": float(env.thermal_cutoff),
                    "signal_ld_share": float(info["signal_ld_share"]),
                    "led_tx_fraction": float(info["led_tx_fraction"]),
                    "tx_enabled_fraction": float(info.get("tx_enabled_fraction", 1.0)),
                    "signal_led": float(info["signal_led"]),
                    "signal_ld": float(info["signal_ld"]),
                    "snr": float(info["snr"]),
                    "qos_rate": float(info["qos_rate"]),
                    "eh_metric": float(info["eh_metric"]),
                    "info_share": float(info["info_share"]),
                    "eh_share": float(info["eh_share"]),
                    "se": float(info["se"]),
                    "eh": float(info["eh"]),
                    "reward_id_term": float(info.get("reward_id_term", info.get("reward_se_term", info["se"]))),
                    "reward_se_term": float(info.get("reward_se_term", info["se"])),
                    "reward_eh_term": float(info.get("reward_eh_term", info["eh"])),
                    "reward_margin_term": float(info.get("reward_margin_term", 0.0)),
                    "penalty_cost_term": float(info.get("penalty_cost_term", 0.0)),
                    "penalty_power_term": float(info.get("penalty_power_term", 0.0)),
                    "penalty_smooth_term": float(info.get("penalty_smooth_term", 0.0)),
                    "penalty_switch_term": float(info.get("penalty_switch_term", 0.0)),
                    "mode_switch": float(info.get("mode_switch", 0.0)),
                    "boost_switch": float(info.get("boost_switch", 0.0)),
                    "mode_exec": float(info.get("mode_exec", action.get("mode_exec", 0))),
                    "boost_combo_exec": float(info.get("boost_combo_exec", action.get("boost_combo_exec", 0))),
                    "upper_idx_exec": float(info.get("upper_idx_exec", action.get("upper_idx_exec", 0))),
                    "cost": float(info["cost"]),
                    "cost_qos": float(info.get("cost_qos", 0.0)),
                    "cost_temp_anchor": float(info.get("cost_temp_anchor", 0.0)),
                    "cost_temp_boost1": float(info.get("cost_temp_boost1", 0.0)),
                    "cost_temp_boost2": float(info.get("cost_temp_boost2", 0.0)),
                    "thermal_violation": float(info["thermal_violation"]),
                    "temp_mean_before": float(np.mean(temps_before)),
                    "temp_mean_after": float(np.mean(info["temps"])),
                    "temp_max_after": float(np.max(info["temps"])),
                    "current_total": current_total,
                    "bus_current_max": bus_current_max,
                    "bus_utilization": float(current_total / max(bus_current_max, 1.0e-6)),
                }
                for tx_idx, current_val in enumerate(currents_exec.tolist()):
                    row[f"current_tx{tx_idx}"] = float(current_val)
                rows.append(row)
                obs = next_obs
                step += 1
    return pd.DataFrame(rows)


def collect_env_data_sac_lagrangian(
    trainer: SacLagrangianBaseline,
    cfg: Dict,
    scenario: str,
    variant: str,
    seed: int,
    tasks,
    episodes_per_task: int,
) -> pd.DataFrame:
    return collect_env_data_plain_hierarchical_baseline(
        trainer=trainer,
        cfg=cfg,
        scenario=scenario,
        variant=variant,
        seed=seed,
        tasks=tasks,
        episodes_per_task=episodes_per_task,
    )


def plot_scenario_convergence(train_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    metric_specs = [
        ("query_reward", "Query Reward"),
        ("query_cost", "Query Cost"),
        ("query_violation_rate", "Query Violation Rate"),
    ]

    for variant, g in train_df.groupby("variant"):
        for ax, (metric, title) in zip(axes, metric_specs):
            agg = (
                g.groupby("iter", as_index=False)[metric]
                .agg(["mean", "std"])
                .reset_index()
                .rename(columns={"mean": "metric_mean", "std": "metric_std"})
            )
            agg["metric_std"] = agg["metric_std"].fillna(0.0)
            x = agg["iter"].to_numpy(dtype=float)
            y = agg["metric_mean"].to_numpy(dtype=float)
            s = agg["metric_std"].to_numpy(dtype=float)
            ax.plot(x, y, label=variant)
            ax.fill_between(x, y - s, y + s, alpha=0.18)
            ax.set_title(title)

    for ax in axes:
        ax.set_xlabel("Meta Iter")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_eval_group(
    eval_df: pd.DataFrame,
    metric_pairs: List[tuple[str, str]],
    out_path: Path,
    title: str,
) -> None:
    variants = sorted(eval_df["variant"].unique())
    x = np.arange(len(metric_pairs), dtype=float)
    w = 0.22

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, v in enumerate(variants):
        g = eval_df[eval_df["variant"] == v]
        means = [float(g[m].mean()) for m, _ in metric_pairs]
        stds = [float(g[m].std(ddof=0)) for m, _ in metric_pairs]
        ax.bar(x + (i - 1) * w, means, width=w, yerr=stds, capsize=3, label=v)

    ax.set_xticks(x)
    ax.set_xticklabels([name for _, name in metric_pairs])
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_single_eval_metric(
    eval_df: pd.DataFrame,
    metric: str,
    label: str,
    out_path: Path,
    title: str,
    scale: float = 1.0,
) -> None:
    variants = sorted(eval_df["variant"].unique())
    x = np.arange(len(variants), dtype=float)

    means = []
    stds = []
    for variant in variants:
        g = eval_df[eval_df["variant"] == variant]
        means.append(float(g[metric].mean()) * scale)
        stds.append(float(g[metric].std(ddof=0)) * scale)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    bars = ax.bar(x, means, yerr=stds, capsize=4, width=0.58)
    for bar, variant in zip(bars, variants):
        bar.set_label(variant)

    ax.set_xticks(x)
    ax.set_xticklabels(variants)
    ax.set_ylabel(label)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_scenario_eval(eval_df: pd.DataFrame, out_path: Path) -> None:
    metric_pairs = [
        ("reward", "Reward"),
        ("cost", "Cost"),
        ("violation_rate", "Violation"),
    ]
    _plot_eval_group(eval_df, metric_pairs, out_path, "Final Eval Metrics: Reward / Cost / Violation")


def plot_scenario_se_eh(eval_df: pd.DataFrame, out_path: Path) -> None:
    metric_pairs = [
        ("se", "SE"),
        ("eh", "EH"),
    ]
    _plot_eval_group(eval_df, metric_pairs, out_path, "Final Eval Metrics: SE / EH")


def plot_scenario_env(env_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for v, g in env_df.groupby("variant"):
        axes[0, 0].hist(g["snr"], bins=40, alpha=0.4, label=v)
        axes[0, 1].hist(g["temp_max_after"], bins=40, alpha=0.4, label=v)
        axes[1, 0].hist(g["cost"], bins=40, alpha=0.4, label=v)
        axes[1, 1].hist(g["signal_ld_share"], bins=30, alpha=0.4, label=v)

    axes[0, 0].set_title("SNR Distribution")
    axes[0, 1].set_title("Peak Temperature Distribution")
    axes[1, 0].set_title("Cost Distribution")
    axes[1, 1].set_title("LD Signal Share Distribution")

    for ax in axes.ravel():
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_stepwise_stability(env_df: pd.DataFrame, out_path: Path) -> None:
    metric_specs = [
        ("snr", "SNR"),
        ("temp_max_after", "Peak Temperature"),
        ("current_total", "Total Current"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))

    for variant, g in env_df.groupby("variant"):
        for ax, (metric, title) in zip(axes, metric_specs):
            agg = (
                g.groupby("step", as_index=False)[metric]
                .agg(["mean", "std"])
                .reset_index()
                .rename(columns={"mean": "metric_mean", "std": "metric_std"})
            )
            agg["metric_std"] = agg["metric_std"].fillna(0.0)
            x = agg["step"].to_numpy(dtype=float)
            y = agg["metric_mean"].to_numpy(dtype=float)
            s = agg["metric_std"].to_numpy(dtype=float)
            ax.plot(x, y, label=variant)
            ax.fill_between(x, y - s, y + s, alpha=0.18)
            ax.set_title(title)
            ax.set_xlabel("Env Step")
            ax.grid(alpha=0.3)

    axes[0].set_ylabel("Mean +/- Std")
    for ax in axes:
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_utilization_tradeoff(env_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    variants = sorted(env_df["variant"].unique())
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {variant: colors[idx % len(colors)] for idx, variant in enumerate(variants)}

    reward_proxy = (
        env_df["reward_se_term"]
        + env_df["reward_eh_term"]
        + env_df["reward_margin_term"]
        - env_df["penalty_cost_term"]
        - env_df["penalty_power_term"]
        - env_df["penalty_smooth_term"]
        - env_df["penalty_switch_term"]
    )

    for variant in variants:
        g = env_df[env_df["variant"] == variant]
        color = color_map[variant]
        axes[0].scatter(g["bus_utilization"], g["snr"], s=10, alpha=0.15, label=variant, color=color)
        axes[1].scatter(g["bus_utilization"], g["temp_max_after"], s=10, alpha=0.15, label=variant, color=color)
        axes[2].scatter(g["bus_utilization"], reward_proxy[g.index], s=10, alpha=0.15, label=variant, color=color)

    axes[0].set_title("Bus Utilization vs SNR")
    axes[1].set_title("Bus Utilization vs Peak Temp")
    axes[2].set_title("Bus Utilization vs Reward")

    axes[0].set_ylabel("SNR")
    axes[1].set_ylabel("Peak Temperature")
    axes[2].set_ylabel("Reward Proxy")
    for ax in axes:
        ax.set_xlabel("Bus Utilization")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def evaluate_summary(eval_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for v, g in eval_df.groupby("variant"):
        out[v] = {
            "reward_mean": float(g["reward"].mean()),
            "se_mean": float(g["se"].mean()),
            "eh_mean": float(g["eh"].mean()),
            "cost_mean": float(g["cost"].mean()),
            "violation_rate_mean": float(g["violation_rate"].mean()),
        }
    return out


def _current_columns(df: pd.DataFrame) -> List[str]:
    return sorted(
        [col for col in df.columns if col.startswith("current_tx")],
        key=lambda name: int(name.replace("current_tx", "")),
    )


def build_stability_table(eval_df: pd.DataFrame, env_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    current_cols = _current_columns(env_df)

    for variant, env_g in env_df.groupby("variant"):
        eval_g = eval_df[eval_df["variant"] == variant]
        row: Dict[str, float | str] = {
            "variant": variant,
            "reward_mean": float(eval_g["reward"].mean()) if not eval_g.empty else 0.0,
            "reward_std": float(eval_g["reward"].std(ddof=0)) if not eval_g.empty else 0.0,
            "violation_rate_mean": float(eval_g["violation_rate"].mean()) if not eval_g.empty else 0.0,
            "cost_mean": float(eval_g["cost"].mean()) if not eval_g.empty else 0.0,
            "temp_mean_mean": float(env_g["temp_mean_after"].mean()),
            "temp_max_mean": float(env_g["temp_max_after"].mean()),
            "temp_max_q90": float(env_g["temp_max_after"].quantile(0.9)),
            "current_total_mean": float(env_g["current_total"].mean()),
            "current_total_q90": float(env_g["current_total"].quantile(0.9)),
            "bus_current_max": float(env_g["bus_current_max"].mean()),
            "bus_utilization_mean": float(env_g["bus_utilization"].mean()),
            "bus_utilization_q90": float(env_g["bus_utilization"].quantile(0.9)),
            "signal_ld_share_mean": float(env_g["signal_ld_share"].mean()),
            "signal_ld_share_q90": float(env_g["signal_ld_share"].quantile(0.9)),
            "thermal_step_violation_fraction": float((env_g["thermal_violation"] > 0).mean()),
        }
        for col in current_cols:
            row[f"{col}_mean"] = float(env_g[col].mean())
        # Physical activation statistics from executed currents.
        if {"current_tx0", "current_tx1", "current_tx2"}.issubset(set(env_g.columns)):
            tx0_on = env_g["current_tx0"] > 1.0e-3
            tx1_on = env_g["current_tx1"] > 1.0e-3
            tx2_on = env_g["current_tx2"] > 1.0e-3
            row["anchor_only_active_fraction"] = float((tx0_on & (~tx1_on) & (~tx2_on)).mean())
            row["anchor_boost1_active_fraction"] = float((tx0_on & tx1_on & (~tx2_on)).mean())
            row["anchor_boost2_active_fraction"] = float((tx0_on & (~tx1_on) & tx2_on).mean())
            row["all3_active_fraction"] = float((tx0_on & tx1_on & tx2_on).mean())
            row["boost_any_active_fraction"] = float((tx1_on | tx2_on).mean())
        if "boost_combo_exec" in env_g.columns:
            row["anchor_only_fraction"] = float((env_g["boost_combo_exec"] == 0).mean())
            row["anchor_boost1_fraction"] = float((env_g["boost_combo_exec"] == 1).mean())
            row["anchor_boost2_fraction"] = float((env_g["boost_combo_exec"] == 2).mean())
            row["all3_fraction"] = float((env_g["boost_combo_exec"] == 3).mean())
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["variant"])
    return pd.DataFrame(rows).sort_values("variant").reset_index(drop=True)


def build_current_trace_table(env_df: pd.DataFrame) -> pd.DataFrame:
    current_cols = _current_columns(env_df)
    metric_cols = current_cols + ["current_total", "bus_utilization", "bus_current_max"]
    if env_df.empty:
        return pd.DataFrame(columns=["variant", "step"] + metric_cols)
    return (
        env_df.groupby(["variant", "step"], as_index=False)[metric_cols]
        .mean()
        .sort_values(["variant", "step"])
        .reset_index(drop=True)
    )



STAT_METRIC_FIELDS = {
    "reward": "eval_reward",
    "se": "eval_se",
    "eh": "eval_eh",
    "cost": "eval_cost",
    "violation_rate": "eval_violation_rate",
    "temp_q90": "env_temp_max_q90",
    "thermal_step_violation_fraction": "env_step_violation_fraction",
}

MAIN_BENCHMARK_SCENARIOS = ("moderate_practical", "hard_stress", "channel_harsh")
STRUCTURAL_VARIANT_ORDER = ("hybrid", "single_led", "single_ld", "shin2024_matched")
HARD_TARGETED_VARIANT_ORDER = (
    "hybrid",
    "hybrid_wo_meta",
    "hybrid_wo_lagrangian",
    "hybrid_hard_clip",
    "heuristic_safe",
    "sac_lagrangian",
    "shin2024_matched",
    "sac_dalal_safe",
    "dalal2018_safe",
)
THERMAL_VARIANT_ORDER = (
    "hybrid",
    "hybrid_wo_lagrangian",
    "hybrid_hard_clip",
    "sac_lagrangian",
    "shin2024_matched",
    "sac_dalal_safe",
    "dalal2018_safe",
)

PAIRING_KEY_FIELDS = ("scenario", "seed", "eval_task_batch_hash", "ordered_eval_task_batch_hash")


def _stats_pairing_key(row: Dict[str, object], scenario: str) -> tuple[str, int, str, str]:
    eval_hash = str(row.get("eval_task_batch_hash", ""))
    ordered_eval_hash = str(row.get("ordered_eval_task_batch_hash", eval_hash))
    return (str(row.get("scenario", scenario)), int(row["seed"]), eval_hash, ordered_eval_hash)


def _paired_signflip_pvalue(diffs: np.ndarray) -> float | None:
    diffs = np.asarray(diffs, dtype=np.float64).reshape(-1)
    n = int(diffs.size)
    if n <= 1:
        return None
    observed = abs(float(np.mean(diffs)))
    if n <= 16:
        total = 1 << n
        ge = 0
        for signs in itertools.product((-1.0, 1.0), repeat=n):
            signed = diffs * np.asarray(signs, dtype=np.float64)
            if abs(float(np.mean(signed))) + 1.0e-12 >= observed:
                ge += 1
        return float(ge / total)

    rng = np.random.default_rng(0)
    draws = 20000
    sign_matrix = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float64), size=(draws, n))
    perm_means = np.abs((sign_matrix * diffs[None, :]).mean(axis=1))
    return float((np.count_nonzero(perm_means + 1.0e-12 >= observed) + 1) / (draws + 1))


def _paired_diff_stats(diffs: np.ndarray) -> Dict[str, float | int | None]:
    diffs = np.asarray(diffs, dtype=np.float64).reshape(-1)
    n_pairs = int(diffs.size)
    if n_pairs == 0:
        return {
            "n_pairs": 0,
            "insufficient_pairs": True,
            "p_value_trusted": False,
            "mean_diff": float("nan"),
            "std_diff": float("nan"),
            "t_stat": float("nan"),
            "effect_size_dz": float("nan"),
            "p_value": None,
        }
    mean_diff = float(np.mean(diffs))
    if n_pairs == 1:
        return {
            "n_pairs": 1,
            "insufficient_pairs": True,
            "p_value_trusted": False,
            "mean_diff": mean_diff,
            "std_diff": float("nan"),
            "t_stat": float("nan"),
            "effect_size_dz": float("nan"),
            "p_value": None,
        }
    std_diff = float(np.std(diffs, ddof=1))
    if std_diff <= 1.0e-12:
        t_stat = float("inf") if abs(mean_diff) > 1.0e-12 else 0.0
        effect_size = float("inf") if abs(mean_diff) > 1.0e-12 else 0.0
    else:
        t_stat = float(mean_diff / (std_diff / math.sqrt(n_pairs)))
        effect_size = float(mean_diff / std_diff)
    p_value = _paired_signflip_pvalue(diffs)
    return {
        "n_pairs": n_pairs,
        "insufficient_pairs": False,
        "p_value_trusted": p_value is not None,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "t_stat": t_stat,
        "effect_size_dz": effect_size,
        "p_value": p_value,
    }


def _apply_holm_correction(rows: List[Dict[str, object]]) -> None:
    indexed = [(idx, float(row["p_value"])) for idx, row in enumerate(rows) if row.get("p_value") is not None]
    if not indexed:
        return
    indexed.sort(key=lambda item: item[1])
    m = len(indexed)
    running = 0.0
    for rank, (idx, pval) in enumerate(indexed):
        adjusted = min(1.0, (m - rank) * pval)
        running = max(running, adjusted)
        rows[idx]["p_value_holm"] = float(running)
    for row in rows:
        row.setdefault("p_value_holm", None)


def build_statistics_artifact(
    run_rows: List[Dict],
    *,
    artifact_name: str,
    scenarios: List[str],
    variant_order: tuple[str, ...],
    metrics: List[str],
) -> Dict[str, object] | None:
    filtered = [
        dict(row)
        for row in run_rows
        if str(row.get("scenario")) in scenarios and str(row.get("variant")) in variant_order
    ]
    if not filtered:
        return None

    grouped_payload: Dict[str, Dict[str, Dict[str, float]]] = {}
    pairwise_payload: Dict[str, List[Dict[str, object]]] = {}
    csv_rows: List[Dict[str, object]] = []

    for scenario in scenarios:
        scenario_rows = [row for row in filtered if str(row.get("scenario")) == scenario]
        if not scenario_rows:
            continue
        present_variants = [variant for variant in variant_order if any(str(row.get("variant")) == variant for row in scenario_rows)]
        grouped_payload[scenario] = {}
        pairwise_rows: List[Dict[str, object]] = []

        for variant in present_variants:
            variant_rows = [row for row in scenario_rows if str(row.get("variant")) == variant]
            grouped_payload[scenario][variant] = {}
            for metric in metrics:
                field = STAT_METRIC_FIELDS[metric]
                values = np.asarray([float(row[field]) for row in variant_rows if field in row], dtype=np.float64)
                if values.size == 0:
                    continue
                grouped_payload[scenario][variant][metric] = {
                    "n": int(values.size),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                }
                csv_rows.append(
                    {
                        "artifact": artifact_name,
                        "row_type": "group",
                        "scenario": scenario,
                        "metric": metric,
                        "variant": variant,
                        "n": int(values.size),
                        "mean": float(values.mean()),
                        "std": float(values.std(ddof=0)),
                    }
                )

        for metric in metrics:
            field = STAT_METRIC_FIELDS[metric]
            for left_idx in range(len(present_variants)):
                for right_idx in range(left_idx + 1, len(present_variants)):
                    left_variant = present_variants[left_idx]
                    right_variant = present_variants[right_idx]
                    left_map = {
                        _stats_pairing_key(row, scenario): float(row[field])
                        for row in scenario_rows
                        if str(row.get("variant")) == left_variant and field in row
                    }
                    right_map = {
                        _stats_pairing_key(row, scenario): float(row[field])
                        for row in scenario_rows
                        if str(row.get("variant")) == right_variant and field in row
                    }
                    common_keys = sorted(set(left_map.keys()) & set(right_map.keys()))
                    diffs = np.asarray([left_map[key] - right_map[key] for key in common_keys], dtype=np.float64)
                    stat = _paired_diff_stats(diffs)
                    pairwise_rows.append(
                        {
                            "artifact": artifact_name,
                            "scenario": scenario,
                            "metric": metric,
                            "left_variant": left_variant,
                            "right_variant": right_variant,
                            "pairing_key_fields": list(PAIRING_KEY_FIELDS),
                            "pair_keys": [list(key) for key in common_keys],
                            **stat,
                        }
                    )

        _apply_holm_correction(pairwise_rows)
        pairwise_payload[scenario] = pairwise_rows
        for row in pairwise_rows:
            csv_rows.append(
                {
                    "artifact": artifact_name,
                    "row_type": "pairwise",
                    "scenario": row["scenario"],
                    "metric": row["metric"],
                    "left_variant": row["left_variant"],
                    "right_variant": row["right_variant"],
                    "n_pairs": row["n_pairs"],
                    "insufficient_pairs": row["insufficient_pairs"],
                    "p_value_trusted": row["p_value_trusted"],
                    "mean_diff": row["mean_diff"],
                    "std_diff": row["std_diff"],
                    "t_stat": row["t_stat"],
                    "effect_size_dz": row["effect_size_dz"],
                    "p_value": row["p_value"],
                    "p_value_holm": row.get("p_value_holm"),
                }
            )

    if not grouped_payload and not pairwise_payload:
        return None
    return {
        "artifact": artifact_name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "test": "paired_t_with_signflip_pvalue",
        "pairing_key_fields": list(PAIRING_KEY_FIELDS),
        "metrics": metrics,
        "scenarios": grouped_payload,
        "pairwise": pairwise_payload,
        "csv_rows": csv_rows,
    }


def write_statistics_artifact(out_root: Path, artifact: Dict[str, object]) -> Dict[str, str]:
    stem = str(artifact["artifact"])
    json_path = out_root / f"{stem}.json"
    csv_path = out_root / f"{stem}.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump({k: v for k, v in artifact.items() if k != "csv_rows"}, f, ensure_ascii=False, indent=2)

    csv_rows = list(artifact.get("csv_rows", []))
    if csv_rows:
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    else:
        pd.DataFrame().to_csv(csv_path, index=False)
    return {"json": str(json_path), "csv": str(csv_path)}


def plot_current_allocation(current_df: pd.DataFrame, out_path: Path) -> None:
    variants = list(current_df["variant"].unique())
    current_cols = _current_columns(current_df)
    if not variants or not current_cols:
        return

    fig, axes = plt.subplots(1, len(variants), figsize=(5.2 * len(variants), 4.2), sharey=False)
    axes = np.atleast_1d(axes)

    color_cycle = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for ax, variant in zip(axes, variants):
        g = current_df[current_df["variant"] == variant].sort_values("step")
        for idx, col in enumerate(current_cols):
            ax.plot(g["step"], g[col], label=f"Tx{idx}", linewidth=2.0, color=color_cycle[idx % len(color_cycle)])
        ax.plot(g["step"], g["current_total"], label="Total", linewidth=2.2, linestyle="--", color="black")
        bus_current_max = float(g["bus_current_max"].mean())
        ax.axhline(bus_current_max, color="crimson", linestyle=":", linewidth=1.8, label="Bus Max")
        ax.set_title(variant)
        ax.set_xlabel("Env Step")
        ax.set_ylabel("Executed Current")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle("Per-Transmitter Current Allocation", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_one_scenario(
    base_cfg: Dict,
    out_root: Path,
    scenario: str,
    meta_iters: int,
    fast_mode: bool,
    seeds: List[int],
    eval_tasks: int,
    eval_eps: int,
    env_tasks: int,
    env_eps: int,
    use_curriculum: bool = False,
    shared_init: bool = True,
    shared_init_pretrain_iters: int = 4,
    variants: List[str] | None = None,
    ablations: List[str] | None = None,
    baselines: List[str] | None = None,
    include_variants: bool = True,
) -> Dict:
    scenario_dir = out_root / scenario
    scenario_dir.mkdir(parents=True, exist_ok=True)

    variants = list(variants or ["hybrid", "single_led", "single_ld"])
    ablations = list(ablations or ["full"])
    baselines = list(baselines or [])
    exp_specs: List[Dict[str, str]] = []
    if include_variants:
        for variant in variants:
            for ablation in ablations:
                label = variant if ablation == "full" else f"{variant}_{ablation}"
                exp_specs.append(
                    {
                        "runner": "trainer",
                        "variant": variant,
                        "ablation": ablation,
                        "label": label,
                        "baseline_override": "",
                    }
                )
    elif not baselines:
        raise ValueError("include_variants=False requires at least one baseline.")
    for baseline in baselines:
        if baseline not in {
            "heuristic_safe",
            "sac_lagrangian",
            "sac_dalal_safe",
            "shin2024_matched",
            "dalal2018_safe",
        }:
            raise ValueError(f"Unknown baseline: {baseline}")
        if baseline == "heuristic_safe":
            runner = "heuristic"
            baseline_override = baseline
        elif baseline in {"sac_lagrangian", "sac_dalal_safe"}:
            runner = "sac_lagrangian"
            baseline_override = baseline
        elif baseline == "shin2024_matched":
            runner = "shin2024_matched"
            baseline_override = baseline
        else:
            runner = "trainer"
            baseline_override = baseline
        exp_specs.append(
            {
                "runner": runner,
                "variant": "hybrid",
                "ablation": "full",
                "label": baseline,
                "baseline_override": baseline_override,
            }
        )

    effective_shared_init = bool(shared_init and all(spec["runner"] == "trainer" and spec["ablation"] == "full" for spec in exp_specs))
    if shared_init and not effective_shared_init:
        print(f"[benchmark] shared_init disabled for scenario={scenario} because ablations/baselines are requested.")
    precheck_cfg = apply_common_settings(
        base_cfg,
        meta_iters,
        scenario_dir,
        f"{scenario}_precheck",
        seeds[0] if seeds else 0,
        fast_mode=fast_mode,
        use_curriculum=use_curriculum,
    )
    apply_scenario(precheck_cfg, scenario)
    strict_thermal = scenario in {"moderate_practical", "thermal_moderate", "practical_hard"}
    precheck_result = validate_training_config(precheck_cfg, scenario, strict_thermal=strict_thermal)
    precheck_result.update(alignment_snapshot(precheck_cfg))
    precheck_result["task_distribution"] = task_distribution_summary(precheck_cfg)
    precheck_path = scenario_dir / "precheck.json"
    with precheck_path.open("w", encoding="utf-8") as f:
        json.dump(precheck_result, f, ensure_ascii=False, indent=2)
    print(
        f"[{scenario}] task_distribution="
        f"{json.dumps(precheck_result['task_distribution'], ensure_ascii=False, sort_keys=True)}"
    )

    train_all: List[pd.DataFrame] = []
    eval_rows: List[Dict[str, float]] = []
    env_all: List[pd.DataFrame] = []
    conv_rows: List[Dict[str, float]] = []
    run_summaries: List[Dict] = []
    shared_init_paths: Dict[int, Path] = {}

    if effective_shared_init:
        pre_iters = int(max(0, shared_init_pretrain_iters))
        if fast_mode:
            pre_iters = min(pre_iters, 1)
        shared_root = scenario_dir / "_shared_init"
        shared_root.mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            init_run_name = f"seed{seed}_shared_init"
            init_cfg = apply_common_settings(
                base_cfg,
                pre_iters if pre_iters > 0 else 1,
                shared_root,
                init_run_name,
                seed,
                fast_mode=fast_mode,
                use_curriculum=use_curriculum,
            )
            apply_scenario(init_cfg, scenario)
            apply_variant(init_cfg, "hybrid")
            if use_curriculum:
                inject_default_curriculum(init_cfg)
            init_trainer = MetaTrainer(init_cfg)
            if pre_iters > 0:
                train_csv = init_trainer.train(meta_iters=pre_iters)
                init_df = pd.read_csv(train_csv)
                pick = select_checkpoint(
                    run_df=init_df,
                    ckpt_dir=init_trainer.ckpt_dir,
                    score_cfg=init_cfg.get("meta", {}).get("checkpoint_selection", {}),
                )
                if pick.get("selected_path"):
                    shared_init_paths[int(seed)] = Path(str(pick["selected_path"]))
                else:
                    init_ckpt = shared_root / f"seed{seed}.pt"
                    init_trainer.agent.save(init_ckpt)
                    shared_init_paths[int(seed)] = init_ckpt
            else:
                init_ckpt = shared_root / f"seed{seed}.pt"
                init_trainer.agent.save(init_ckpt)
                shared_init_paths[int(seed)] = init_ckpt

    for spec in exp_specs:
        variant = str(spec["variant"])
        ablation = str(spec["ablation"])
        label = str(spec["label"])
        runner = str(spec["runner"])
        baseline_override = str(spec.get("baseline_override", ""))
        for seed in seeds:
            run_name = f"{scenario}_{label}_seed{seed}"
            cfg = apply_common_settings(
                base_cfg,
                meta_iters,
                scenario_dir,
                run_name,
                seed,
                fast_mode=fast_mode,
                use_curriculum=use_curriculum,
            )
            apply_scenario(cfg, scenario)
            apply_variant(cfg, variant)
            apply_ablation(cfg, ablation)
            if baseline_override:
                apply_baseline_overrides(cfg, baseline_override)
            if use_curriculum:
                inject_default_curriculum(cfg)

            run_dir = scenario_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            resolved_cfg_path = run_dir / "resolved_config.yaml"
            dump_resolved_config(cfg, resolved_cfg_path)

            if runner == "sac_lagrangian":
                trainer = SacLagrangianBaseline(cfg)
            elif runner == "shin2024_matched":
                trainer = Shin2024MatchedBaseline(cfg)
            else:
                trainer = MetaTrainer(cfg)
                if effective_shared_init:
                    trainer.agent.load(shared_init_paths[int(seed)])
            ckpt_pick = {"strategy": "none", "selected_iter": -1, "selected_score": float("nan"), "selected_path": ""}

            score_cfg = copy.deepcopy(cfg.get("meta", {}).get("checkpoint_selection", {}))
            selection_tasks = sample_fixed_tasks(
                cfg,
                seed,
                int(max(1, score_cfg.get("eval_tasks", eval_tasks))),
                seed_offset=11_000,
            )
            selection_task_hash = fixed_task_bank_hash(selection_tasks)
            ordered_selection_task_hash = ordered_fixed_task_bank_hash(selection_tasks)
            selection_eps = int(max(1, score_cfg.get("eval_eps", 1)))

            if runner == "trainer":
                train_csv = trainer.train(meta_iters=meta_iters)

                run_df = pd.read_csv(train_csv)
                run_df["scenario"] = scenario
                run_df["variant"] = label
                run_df["seed"] = float(seed)
                train_all.append(run_df)

                conv = convergence_stats(run_df)
                conv.update({"scenario": scenario, "variant": label, "seed": float(seed)})
                conv_rows.append(conv)

                ckpt_pick = select_checkpoint(
                    run_df=run_df,
                    ckpt_dir=trainer.ckpt_dir,
                    score_cfg=score_cfg,
                    evaluator=lambda path: (trainer.agent.load(path), evaluate_on_tasks(trainer=trainer, cfg=cfg, tasks=selection_tasks, episodes_per_task=selection_eps))[1],
                )
                if ckpt_pick.get("selected_path"):
                    trainer.agent.load(ckpt_pick["selected_path"])
            elif runner in {"sac_lagrangian", "shin2024_matched"}:
                train_csv = trainer.train(meta_iters=meta_iters)

                run_df = pd.read_csv(train_csv)
                run_df["scenario"] = scenario
                run_df["variant"] = label
                run_df["seed"] = float(seed)
                train_all.append(run_df)

                conv = convergence_stats(run_df)
                conv.update({"scenario": scenario, "variant": label, "seed": float(seed)})
                conv_rows.append(conv)

                ckpt_pick = select_checkpoint(
                    run_df=run_df,
                    ckpt_dir=trainer.ckpt_dir,
                    score_cfg=score_cfg,
                    evaluator=lambda path: (
                        trainer.load(path),
                        evaluate_plain_hierarchical_baseline_on_tasks(
                            trainer=trainer,
                            cfg=cfg,
                            tasks=selection_tasks,
                            episodes_per_task=selection_eps,
                        ),
                    )[1],
                )
                if ckpt_pick.get("selected_path"):
                    trainer.load(ckpt_pick["selected_path"])

            if ckpt_pick.get("selection_rows"):
                pd.DataFrame(ckpt_pick["selection_rows"]).to_csv(run_dir / "checkpoint_selection.csv", index=False)

            eval_task_subset = sample_fixed_tasks(cfg, seed, eval_tasks, seed_offset=21_000)
            env_task_subset = sample_fixed_tasks(cfg, seed, env_tasks, seed_offset=31_000)
            eval_task_hash = fixed_task_bank_hash(eval_task_subset)
            env_task_hash = fixed_task_bank_hash(env_task_subset)
            ordered_eval_task_hash = ordered_fixed_task_bank_hash(eval_task_subset)
            ordered_env_task_hash = ordered_fixed_task_bank_hash(env_task_subset)

            if runner == "trainer":
                ev = evaluate_on_tasks(
                    trainer=trainer,
                    cfg=cfg,
                    tasks=eval_task_subset,
                    episodes_per_task=eval_eps,
                )
                env_df = collect_env_data(
                    trainer=trainer,
                    cfg=cfg,
                    scenario=scenario,
                    variant=label,
                    seed=seed,
                    tasks=env_task_subset,
                    episodes_per_task=env_eps,
                )
            elif runner in {"sac_lagrangian", "shin2024_matched"}:
                ev = evaluate_plain_hierarchical_baseline_on_tasks(
                    trainer=trainer,
                    cfg=cfg,
                    tasks=eval_task_subset,
                    episodes_per_task=eval_eps,
                )
                env_df = collect_env_data_plain_hierarchical_baseline(
                    trainer=trainer,
                    cfg=cfg,
                    scenario=scenario,
                    variant=label,
                    seed=seed,
                    tasks=env_task_subset,
                    episodes_per_task=env_eps,
                )
            else:
                ev = evaluate_heuristic_on_tasks(
                    trainer=trainer,
                    cfg=cfg,
                    tasks=eval_task_subset,
                    episodes_per_task=eval_eps,
                )
                env_df = collect_env_data_heuristic(
                    trainer=trainer,
                    cfg=cfg,
                    scenario=scenario,
                    variant=label,
                    seed=seed,
                    tasks=env_task_subset,
                    episodes_per_task=env_eps,
                )
            ev.update({"scenario": scenario, "variant": label, "seed": float(seed)})
            eval_rows.append(ev)
            env_all.append(env_df)
            baseline_meta = dict(cfg.get("baseline_metadata", {}))

            run_summaries.append(
                {
                    "scenario": scenario,
                    "variant": label,
                    "base_variant": variant,
                    "ablation": ablation,
                    "runner": runner,
                    "baseline_override": baseline_override,
                    "seed": int(seed),
                    "run_name": run_name,
                    "resolved_config": str(resolved_cfg_path),
                    "checkpoint_strategy": str(ckpt_pick.get("strategy", "none")),
                    "checkpoint_iter": int(ckpt_pick.get("selected_iter", -1)),
                    "checkpoint_score": (
                        float(ckpt_pick.get("selected_score"))
                        if np.isfinite(float(ckpt_pick.get("selected_score", float("nan"))))
                        else None
                    ),
                    "checkpoint_eval_reward": float(ckpt_pick.get("selected_metrics", {}).get("reward", float("nan"))) if ckpt_pick.get("selected_metrics") else None,
                    "checkpoint_eval_cost": float(ckpt_pick.get("selected_metrics", {}).get("cost", float("nan"))) if ckpt_pick.get("selected_metrics") else None,
                    "checkpoint_eval_violation_rate": float(ckpt_pick.get("selected_metrics", {}).get("violation_rate", float("nan"))) if ckpt_pick.get("selected_metrics") else None,
                    "power_weight": float(cfg["env"]["power_weight"]),
                    "action_smooth_weight": float(cfg["env"].get("action_smooth_weight", 0.0)),
                    "env_thermal_safe": float(cfg["env"]["thermal_safe"]),
                    "env_thermal_cutoff": float(cfg["env"]["thermal_cutoff"]),
                    "safety_thermal_safe": float(cfg["safety"]["thermal_safe"]),
                    "safety_thermal_cutoff": float(cfg["safety"]["thermal_cutoff"]),
                    "shared_init": bool(effective_shared_init),
                    "shared_init_pretrain_iters": int(pre_iters) if effective_shared_init else 0,
                    "shared_init_ckpt": str(shared_init_paths[int(seed)]) if effective_shared_init else "",
                    "shared_init_ckpt_pre_alignment": (
                        bool(getattr(trainer.agent, "loaded_alignment_meta", {}).get("pre_alignment", True))
                        if effective_shared_init and runner == "trainer"
                        else None
                    ),
                    "sampler_ranges": sampler_snapshot(cfg),
                    "selection_task_batch_hash": selection_task_hash,
                    "eval_task_batch_hash": eval_task_hash,
                    "env_task_batch_hash": env_task_hash,
                    "ordered_selection_task_batch_hash": ordered_selection_task_hash,
                    "ordered_eval_task_batch_hash": ordered_eval_task_hash,
                    "ordered_env_task_batch_hash": ordered_env_task_hash,
                    "baseline_family": str(baseline_meta.get("baseline_family", baseline_override or "")),
                    "exact_reproduction": baseline_meta.get("exact_reproduction"),
                    "external_baseline": baseline_meta.get("external_baseline"),
                    "safety_protocol": str(baseline_meta.get("safety_protocol", "")),
                    "comparison_role": str(baseline_meta.get("comparison_role", "")),
                    "lower_learned_action_dim": baseline_meta.get("lower_learned_action_dim"),
                    "fixed_mode_exec": baseline_meta.get("fixed_mode_exec"),
                    "fixed_mode_name": str(baseline_meta.get("fixed_mode_name", "")),
                    "fixed_current_template": str(baseline_meta.get("fixed_current_template", "")),
                    "fixed_current_fraction": baseline_meta.get("fixed_current_fraction"),
                    "formally_comparable": bool(is_formally_comparable_record(alignment_snapshot(cfg))),
                    **alignment_snapshot(cfg),
                    "eval_reward": float(ev["reward"]),
                    "eval_se": float(ev["se"]),
                    "eval_eh": float(ev["eh"]),
                    "eval_cost": float(ev["cost"]),
                    "eval_violation_rate": float(ev["violation_rate"]),
                    "env_cost_mean": float(env_df["cost"].mean()) if not env_df.empty else 0.0,
                    "env_step_violation_fraction": float((env_df["thermal_violation"] > 0).mean())
                    if not env_df.empty
                    else 0.0,
                    "env_temp_max_q90": float(env_df["temp_max_after"].quantile(0.9))
                    if not env_df.empty
                    else 0.0,
                    "tx_enabled_fraction_mean": float(env_df["tx_enabled_fraction"].mean())
                    if "tx_enabled_fraction" in env_df
                    else float("nan"),
                }
            )

    train_df = pd.concat(train_all, ignore_index=True) if train_all else pd.DataFrame(columns=["variant"])
    eval_df = pd.DataFrame(eval_rows)
    env_df = pd.concat(env_all, ignore_index=True)
    conv_df = pd.DataFrame(conv_rows)

    train_csv = scenario_dir / "training.csv"
    eval_csv = scenario_dir / "eval.csv"
    env_csv = scenario_dir / "env.csv"
    conv_csv = scenario_dir / "convergence.csv"
    stability_csv = scenario_dir / "stability.csv"
    requested_metrics_csv = scenario_dir / "requested_metrics.csv"
    current_trace_csv = scenario_dir / "current_trace.csv"
    run_summary_csv = scenario_dir / "run_summary.csv"
    run_summary_json = scenario_dir / "run_summary.json"

    train_df.to_csv(train_csv, index=False)
    eval_df.to_csv(eval_csv, index=False)
    env_df.to_csv(env_csv, index=False)
    conv_df.to_csv(conv_csv, index=False)
    stability_df = build_stability_table(eval_df, env_df)
    requested_metrics_df = build_requested_metrics_table(stability_df, scenario)
    current_trace_df = build_current_trace_table(env_df)
    stability_df.to_csv(stability_csv, index=False)
    requested_metrics_df.to_csv(requested_metrics_csv, index=False)
    current_trace_df.to_csv(current_trace_csv, index=False)
    pd.DataFrame(run_summaries).to_csv(run_summary_csv, index=False)
    with run_summary_json.open("w", encoding="utf-8") as f:
        json.dump(run_summaries, f, ensure_ascii=False, indent=2)

    conv_png = scenario_dir / "convergence.png"
    eval_png = scenario_dir / "final_metrics.png"
    se_eh_png = scenario_dir / "se_eh_metrics.png"
    env_png = scenario_dir / "env_realism.png"
    current_png = scenario_dir / "current_allocation.png"
    stepwise_png = scenario_dir / "stepwise_stability.png"
    tradeoff_png = scenario_dir / "utilization_tradeoff.png"

    plot_scenario_convergence(train_df, conv_png)
    plot_scenario_eval(eval_df, eval_png)
    plot_scenario_se_eh(eval_df, se_eh_png)
    se_png = scenario_dir / "se_metrics.png"
    eh_png = scenario_dir / "eh_metrics.png"
    plot_single_eval_metric(eval_df, "se", "SE", se_png, "Final Eval Metric: SE")
    plot_single_eval_metric(eval_df, "eh", "EH (x10^-3)", eh_png, "Final Eval Metric: EH", scale=1.0e3)
    plot_scenario_env(env_df, env_png)
    plot_current_allocation(current_trace_df, current_png)
    plot_stepwise_stability(env_df, stepwise_png)
    plot_utilization_tradeoff(env_df, tradeoff_png)

    env_step_violation = (env_df["thermal_violation"] > 0).astype(float)
    env_ep_violation = (
        env_df.groupby(["variant", "seed", "task_id", "episode"], as_index=False)["thermal_violation"]
        .max()
        .assign(episode_violation=lambda x: (x["thermal_violation"] > 0).astype(float))
    )
    activation_by_variant = []
    for variant, g in env_df.groupby("variant"):
        eval_g = eval_df[eval_df["variant"] == variant]
        ep_g = env_ep_violation[env_ep_violation["variant"] == variant]
        thermal_safe = float(g["thermal_safe"].iloc[0])
        temp_q90 = float(g["temp_max_after"].quantile(0.9))
        activation_by_variant.append(
            {
                "variant": variant,
                "eval_cost_mean": float(eval_g["cost"].mean()) if not eval_g.empty else 0.0,
                "eval_violation_rate_mean": float(eval_g["violation_rate"].mean()) if not eval_g.empty else 0.0,
                "env_step_violation_fraction": float((g["thermal_violation"] > 0).mean()),
                "env_episode_violation_fraction": float(ep_g["episode_violation"].mean())
                if not ep_g.empty
                else 0.0,
                "temp_max_q90": temp_q90,
                "temp_margin_to_safe_q90": float(thermal_safe - temp_q90),
                "thermal_safe": thermal_safe,
            }
        )

    variant_definitions = {}
    for spec in exp_specs:
        label = str(spec["label"])
        runner = str(spec["runner"])
        variant = str(spec["variant"])
        ablation = str(spec["ablation"])
        if runner == "heuristic":
            variant_definitions[label] = {
                "runner": "heuristic",
                "base_variant": variant,
                "ablation": ablation,
                "description": "Rule-based safe heuristic baseline on the hybrid transmitter structure",
            }
        elif runner == "sac_lagrangian":
            is_sac_dalal = label == "sac_dalal_safe"
            variant_definitions[label] = {
                "runner": "sac_lagrangian",
                "base_variant": variant,
                "ablation": ablation,
                "tx_device": ["LED", "LD", "LD"],
                "tx_enabled": [1.0, 1.0, 1.0],
                "baseline_family": "sac_dalal_safe" if is_sac_dalal else "sac_lagrangian",
                "exact_reproduction": False if is_sac_dalal else None,
                "external_baseline": True if is_sac_dalal else None,
                "safety_protocol": "dalal_style_projection" if is_sac_dalal else "common_smooth_projection",
                "comparison_role": "external_safety_layer_baseline" if is_sac_dalal else "learning_baseline",
                "description": (
                    "SAC-style external safety-layer baseline with context/meta/dual disabled and Dalal-style action correction."
                    if is_sac_dalal
                    else "Literature-style SAC-Lagrangian baseline on the fixed heterogeneous hybrid structure, without context latent z or inner/outer meta adaptation"
                ),
            }
        elif runner == "shin2024_matched":
            variant_definitions[label] = {
                "runner": "shin2024_matched",
                "base_variant": variant,
                "ablation": ablation,
                "tx_device": ["LED", "LD", "LD"],
                "tx_enabled": [1.0, 1.0, 1.0],
                "baseline_family": "shin2024_matched",
                "exact_reproduction": False,
                "safety_protocol": "common_smooth_projection",
                "lower_learned_action_dim": 2,
                "fixed_mode_exec": 2,
                "fixed_mode_name": "HY",
                "fixed_current_template": "sigmoid_logit_fraction",
                "fixed_current_fraction": 0.5,
                "description": "Matched Shin 2024-style hierarchical DQN-DDPG baseline under the common benchmark safety protocol: upper DQN selects HY-mode macro boost, lower DDPG learns only rho/tau, and currents come from a fixed feasible template.",
            }
        elif variant == "hybrid":
            variant_definitions[label] = {
                "runner": "trainer",
                "base_variant": variant,
                "ablation": ablation,
                "tx_device": ["LED", "LD", "LD"],
                "tx_enabled": [1.0, 1.0, 1.0],
                "description": "Full hierarchical method on the fixed heterogeneous hybrid structure"
                if label != "dalal2018_safe"
                else "Full hierarchical method with Dalal 2018-style local action correction replacing the default smooth predictive derating path",
            }
        elif variant == "single_led":
            variant_definitions[label] = {
                "runner": "trainer",
                "base_variant": variant,
                "ablation": ablation,
                "tx_device": ["LED", "LED", "LED"],
                "tx_enabled": [1.0, 0.0, 0.0],
            }
        elif variant == "single_ld":
            variant_definitions[label] = {
                "runner": "trainer",
                "base_variant": variant,
                "ablation": ablation,
                "tx_device": ["LD", "LD", "LD"],
                "tx_enabled": [1.0, 0.0, 0.0],
            }

    train_query_cost_mean = (
        float(train_df["query_cost"].mean()) if "query_cost" in train_df.columns and not train_df.empty else None
    )
    train_query_violation_mean = (
        float(train_df["query_violation_rate"].mean())
        if "query_violation_rate" in train_df.columns and not train_df.empty
        else None
    )

    summary = {
        "scenario": scenario,
        "alignment": alignment_snapshot(precheck_cfg),
        "task_distribution": task_distribution_summary(precheck_cfg),
        "variant_definitions": variant_definitions,
        "eval_summary": evaluate_summary(eval_df),
        "precheck": precheck_result,
        "requested_metrics_summary": (
            requested_metrics_df.set_index("variant").to_dict(orient="index") if not requested_metrics_df.empty else {}
        ),
        "stability_summary": stability_df.set_index("variant").to_dict(orient="index") if not stability_df.empty else {},
        "reward_component_means": (
            env_df.groupby("variant", as_index=False)[
                [
                    "reward_se_term",
                    "reward_eh_term",
                    "reward_margin_term",
                    "penalty_cost_term",
                    "penalty_power_term",
                    "penalty_smooth_term",
                    "penalty_switch_term",
                ]
            ]
            .mean()
            .set_index("variant")
            .to_dict(orient="index")
        ),
        "constraint_activation": {
            "train_query_cost_mean": train_query_cost_mean,
            "train_query_violation_mean": train_query_violation_mean,
            "env_cost_mean": float(env_df["cost"].mean()),
            "env_step_violation_fraction": float(env_step_violation.mean()),
            "env_episode_violation_fraction": float(env_ep_violation["episode_violation"].mean()),
            "env_temp_max_q90": float(env_df["temp_max_after"].quantile(0.9)),
        },
        "constraint_activation_by_variant": activation_by_variant,
        "task_batch_hashes": (
            {
                "selection": sorted({str(row["selection_task_batch_hash"]) for row in run_summaries}),
                "eval": sorted({str(row["eval_task_batch_hash"]) for row in run_summaries}),
                "env": sorted({str(row["env_task_batch_hash"]) for row in run_summaries}),
                "ordered_selection": sorted(
                    {str(row["ordered_selection_task_batch_hash"]) for row in run_summaries}
                ),
                "ordered_eval": sorted({str(row["ordered_eval_task_batch_hash"]) for row in run_summaries}),
                "ordered_env": sorted({str(row["ordered_env_task_batch_hash"]) for row in run_summaries}),
            }
            if run_summaries
            else {}
        ),
        "metric_notes": {
            "primary_constraint_metrics": [
                "eval_summary.*.cost_mean",
                "eval_summary.*.violation_rate_mean",
            ],
            "auxiliary_env_metrics": [
                "constraint_activation.env_step_violation_fraction",
                "constraint_activation.env_episode_violation_fraction",
            ],
        },
        "artifacts": {
            "training_csv": str(train_csv),
            "eval_csv": str(eval_csv),
            "env_csv": str(env_csv),
            "convergence_csv": str(conv_csv),
            "stability_csv": str(stability_csv),
            "requested_metrics_csv": str(requested_metrics_csv),
            "current_trace_csv": str(current_trace_csv),
            "run_summary_csv": str(run_summary_csv),
            "run_summary_json": str(run_summary_json),
            "precheck_json": str(precheck_path),
            "convergence_png": str(conv_png),
            "final_metrics_png": str(eval_png),
            "se_eh_metrics_png": str(se_eh_png),
            "se_metrics_png": str(se_png),
            "eh_metrics_png": str(eh_png),
            "env_png": str(env_png),
            "current_allocation_png": str(current_png),
            "stepwise_stability_png": str(stepwise_png),
            "utilization_tradeoff_png": str(tradeoff_png),
        },
    }

    summary_path = scenario_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def run_benchmark(
    cfg_path: str,
    out_dir: str,
    scenarios: List[str],
    meta_iters: int,
    fast_mode: bool,
    seeds: List[int],
    eval_tasks: int,
    eval_eps: int,
    env_tasks: int,
    env_eps: int,
    use_curriculum: bool = False,
    shared_init: bool = True,
    shared_init_pretrain_iters: int = 4,
    variants: List[str] | None = None,
    ablations: List[str] | None = None,
    baselines: List[str] | None = None,
    device: str | None = None,
    include_variants: bool = True,
) -> Path:
    base_cfg = apply_cli_overrides(load_cfg(cfg_path), device=device)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    scenario_summaries = []
    requested_metric_rows: List[pd.DataFrame] = []
    for scenario in scenarios:
        summary = run_one_scenario(
            base_cfg=base_cfg,
            out_root=out_root,
            scenario=scenario,
            meta_iters=meta_iters,
            fast_mode=fast_mode,
            seeds=seeds,
            eval_tasks=eval_tasks,
            eval_eps=eval_eps,
            env_tasks=env_tasks,
            env_eps=env_eps,
            use_curriculum=use_curriculum,
            shared_init=shared_init,
            shared_init_pretrain_iters=shared_init_pretrain_iters,
            variants=variants,
            ablations=ablations,
            baselines=baselines,
            include_variants=include_variants,
        )
        scenario_summaries.append(summary)
        requested_metrics_csv = out_root / scenario / "requested_metrics.csv"
        if requested_metrics_csv.exists():
            requested_metric_rows.append(pd.read_csv(requested_metrics_csv))

    all_run_summaries: List[Dict] = []
    for s in scenarios:
        p = out_root / s / "run_summary.json"
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                all_run_summaries.extend(filter_formally_comparable_records(json.load(f), strict=True))
    run_summary_path = out_root / "run_summary.json"
    with run_summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_run_summaries, f, ensure_ascii=False, indent=2)

    requested_metrics_path = out_root / "requested_metrics.csv"
    if requested_metric_rows:
        pd.concat(requested_metric_rows, ignore_index=True).to_csv(requested_metrics_path, index=False)

    stats_artifacts: Dict[str, Dict[str, str]] = {}
    for artifact in filter(
        None,
        [
            build_statistics_artifact(
                all_run_summaries,
                artifact_name="stats_main_benchmark",
                scenarios=list(MAIN_BENCHMARK_SCENARIOS),
                variant_order=STRUCTURAL_VARIANT_ORDER,
                metrics=["reward", "se", "eh", "cost", "violation_rate"],
            ),
            build_statistics_artifact(
                all_run_summaries,
                artifact_name="stats_hard_stress_targeted",
                scenarios=["hard_stress"],
                variant_order=HARD_TARGETED_VARIANT_ORDER,
                metrics=["reward", "se", "eh", "cost", "violation_rate"],
            ),
            build_statistics_artifact(
                all_run_summaries,
                artifact_name="stats_thermal_rebalanced",
                scenarios=["thermal_rebalanced"],
                variant_order=THERMAL_VARIANT_ORDER,
                metrics=["reward", "cost", "violation_rate", "temp_q90", "thermal_step_violation_fraction"],
            ),
        ],
    ):
        stats_artifacts[str(artifact["artifact"])] = write_statistics_artifact(out_root, artifact)

    report = {
        "cfg_path": cfg_path,
        "alignment": alignment_snapshot(base_cfg),
        "task_distribution_scope": "base_config_snapshot",
        "task_distribution": task_distribution_summary(base_cfg),
        "meta_iters": meta_iters,
        "fast_mode": bool(fast_mode),
        "use_curriculum": bool(use_curriculum),
        "shared_init": bool(shared_init),
        "shared_init_pretrain_iters": int(shared_init_pretrain_iters),
        "variants": list(variants or ["hybrid", "single_led", "single_ld"]),
        "ablations": list(ablations or ["full"]),
        "baselines": list(baselines or []),
        "include_variants": bool(include_variants),
        "baselines_only": bool(not include_variants),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seeds": seeds,
        "scenarios": scenarios,
        "run_summary_json": str(run_summary_path),
        "requested_metrics_csv": str(requested_metrics_path),
        "scenario_summaries": scenario_summaries,
        "stats_artifacts": stats_artifacts,
    }

    report_path = out_root / "report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("Constraint scenario benchmark complete")
    print(f"Report: {report_path}")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default=None, help="Override config device: auto/cpu/cuda/mps")
    parser.add_argument("--out-dir", type=str, default="logs/constraint_scenarios")
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=["moderate_practical", "practical_hard"],
        choices=[
            "easy_baseline",
            "baseline_easy",
            "moderate_practical",
            "practical_hard",
            "thermal_moderate",
            "hard_balanced",
            "balanced_hard",
            "hard_stress",
            "ld_adverse_hard",
            "thermal_tight",
            "channel_harsh",
            "thermal_rebalanced",
        ],
    )
    parser.add_argument("--meta-iters", type=int, default=45)
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Use reduced workload for quick smoke benchmarks.",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Enable easy->moderate->target sampler curriculum during training.",
    )
    parser.add_argument(
        "--no-shared-init",
        action="store_true",
        help="Disable shared initialization across variants for each (scenario, seed).",
    )
    parser.add_argument(
        "--shared-init-iters",
        type=int,
        default=4,
        help="Meta-iters for Hybrid pretraining before cloning init to all variants (per scenario, seed).",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303, 404, 505])
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=["hybrid", "single_led", "single_ld"],
        choices=["hybrid", "single_led", "single_ld"],
        help="Which transmitter variants to train/evaluate.",
    )
    parser.add_argument(
        "--ablations",
        type=str,
        nargs="+",
        default=["full"],
        choices=["full", "wo_meta", "wo_lagrangian", "hard_clip"],
        help="Ablation settings applied on top of each selected variant.",
    )
    parser.add_argument(
        "--baselines",
        type=str,
        nargs="*",
        default=[],
        choices=["heuristic_safe", "sac_lagrangian", "sac_dalal_safe", "shin2024_matched", "dalal2018_safe"],
        help="Optional heuristic/learning baselines to evaluate.",
    )
    parser.add_argument(
        "--baselines-only",
        action="store_true",
        help="Run only the requested baselines and skip variant/ablation experiments.",
    )
    parser.add_argument("--eval-tasks", type=int, default=8)
    parser.add_argument("--eval-eps", type=int, default=2)
    parser.add_argument("--env-tasks", type=int, default=6)
    parser.add_argument("--env-eps", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark(
        cfg_path=args.cfg,
        out_dir=args.out_dir,
        scenarios=args.scenarios,
        meta_iters=args.meta_iters,
        fast_mode=args.fast_mode,
        seeds=args.seeds,
        eval_tasks=args.eval_tasks,
        eval_eps=args.eval_eps,
        env_tasks=args.env_tasks,
        env_eps=args.env_eps,
        use_curriculum=args.curriculum,
        shared_init=(not args.no_shared_init),
        shared_init_pretrain_iters=args.shared_init_iters,
        variants=args.variants,
        ablations=args.ablations,
        baselines=args.baselines,
        device=args.device,
        include_variants=(not args.baselines_only),
    )


if __name__ == "__main__":
    main()
