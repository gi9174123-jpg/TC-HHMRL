from __future__ import annotations

import copy

import numpy as np

from scripts.benchmark_constraint_scenarios import (
    Shin2024MatchedBaseline,
    apply_common_settings,
    apply_ablation,
    apply_baseline_overrides,
    apply_scenario,
    apply_variant,
    inject_default_curriculum,
    sample_fixed_tasks,
    validate_training_config,
)
from tchhmrl.envs.uw_slipt_env import MultiTxUwSliptEnv
from tchhmrl.utils.config import load_cfg


def test_practical_hard_safety_is_earlier_than_env():
    cfg = load_cfg("configs/default.yaml")
    cfg = copy.deepcopy(cfg)
    apply_scenario(cfg, "practical_hard")

    assert float(cfg["env"]["thermal_safe"]) == 50.0
    assert float(cfg["env"]["thermal_cutoff"]) == 60.0
    assert float(cfg["safety"]["thermal_safe"]) == 49.0
    assert float(cfg["safety"]["thermal_cutoff"]) == 59.0
    assert float(cfg["env"]["power_weight"]) == 0.0032


def test_moderate_practical_is_distinct_from_practical_hard():
    moderate = load_cfg("configs/default.yaml")
    practical_hard = load_cfg("configs/default.yaml")
    moderate = copy.deepcopy(moderate)
    practical_hard = copy.deepcopy(practical_hard)

    apply_scenario(moderate, "moderate_practical")
    apply_scenario(practical_hard, "practical_hard")

    assert float(moderate["env"]["attenuation_c"]) < float(practical_hard["env"]["attenuation_c"])
    assert float(moderate["env"]["thermal_safe"]) > float(practical_hard["env"]["thermal_safe"])
    assert float(moderate["safety"]["thermal_safe"]) > float(practical_hard["safety"]["thermal_safe"])
    assert float(moderate["env"]["power_weight"]) > float(practical_hard["env"]["power_weight"])


def test_hard_balanced_scenario_override_exists():
    cfg = load_cfg("configs/default.yaml")
    cfg = copy.deepcopy(cfg)
    apply_scenario(cfg, "hard_balanced")

    assert float(cfg["env"]["attenuation_c"]) == 0.27
    assert float(cfg["env"]["misalign_std"]) == 0.12
    assert float(cfg["safety"]["thermal_safe"]) == 48.0
    assert float(cfg["safety"]["thermal_cutoff"]) == 58.0


def test_thermal_rebalanced_scenario_override_exists():
    cfg = load_cfg("configs/default.yaml")
    cfg = copy.deepcopy(cfg)
    apply_scenario(cfg, "thermal_rebalanced")

    assert float(cfg["env"]["attenuation_c"]) == 0.22
    assert float(cfg["env"]["qos_min_rate"]) == 0.008
    assert float(cfg["safety"]["thermal_safe"]) == 43.0
    assert float(cfg["safety"]["bus_current_max"]) == 7.6
    assert cfg["sampler"]["site_bank"][1]["distances"] == [4.6, 5.2, 5.8]
    assert cfg["sampler"]["site_bank"][0]["amb_temp_range"] == [40.5, 42.5]
    assert cfg["sampler"]["site_bank"][2]["delta_range"] == [7.2, 8.4]


def test_variant_fairness_only_changes_device_mapping():
    base = load_cfg("configs/default.yaml")
    base = copy.deepcopy(base)
    apply_scenario(base, "practical_hard")
    hybrid_base = copy.deepcopy(base["env"]["hybrid"])
    fixed_keys = [k for k in hybrid_base.keys() if k not in {"tx_device", "tx_enabled"}]

    for variant in ["hybrid", "single_led", "single_ld"]:
        cfg = copy.deepcopy(base)
        apply_variant(cfg, variant)

        for k in fixed_keys:
            assert cfg["env"]["hybrid"][k] == hybrid_base[k]

    led_cfg = copy.deepcopy(base)
    apply_variant(led_cfg, "single_led")
    assert led_cfg["env"]["hybrid"]["tx_enabled"] == [1.0, 0.0, 0.0]

    ld_cfg = copy.deepcopy(base)
    apply_variant(ld_cfg, "single_ld")
    assert ld_cfg["env"]["hybrid"]["tx_enabled"] == [1.0, 0.0, 0.0]


def test_safety_thresholds_not_later_than_env_in_key_scenarios():
    for scenario in ["easy_baseline", "moderate_practical", "practical_hard", "hard_balanced", "hard_stress"]:
        cfg = load_cfg("configs/default.yaml")
        cfg = copy.deepcopy(cfg)
        apply_scenario(cfg, scenario)
        assert float(cfg["safety"]["thermal_safe"]) <= float(cfg["env"]["thermal_safe"])
        assert float(cfg["safety"]["thermal_cutoff"]) <= float(cfg["env"]["thermal_cutoff"])


def test_requested_training_precheck_passes_for_main_two_scenarios():
    for scenario in ["moderate_practical", "practical_hard"]:
        cfg = load_cfg("configs/default.yaml")
        cfg = copy.deepcopy(cfg)
        apply_scenario(cfg, scenario)
        checks = validate_training_config(cfg, scenario)
        assert checks["all_passed"] is True


def test_curriculum_does_not_override_strict_site_bank():
    cfg = load_cfg("configs/default.yaml")
    cfg = copy.deepcopy(cfg)
    apply_scenario(cfg, "thermal_rebalanced")
    inject_default_curriculum(cfg)

    phases = cfg["meta"]["curriculum"]["phases"]
    assert phases
    for phase in phases:
        assert "site_bank" not in phase["sampler"]
        assert "strict_site_bank" not in phase["sampler"]


def test_ablation_wo_meta_disables_meta_adaptation_and_context():
    cfg = load_cfg("configs/default.yaml")
    cfg = copy.deepcopy(cfg)
    apply_ablation(cfg, "wo_meta")

    assert cfg["context"]["enabled"] is False
    assert cfg["meta"]["explicit_inner_outer"] is False
    assert cfg["meta"]["query_updates_enabled"] is False
    assert int(cfg["meta"]["query_episodes"]) == 0
    assert int(cfg["meta"]["support_episodes"]) > 0


def test_ablation_wo_lagrangian_disables_dual_updates():
    cfg = load_cfg("configs/default.yaml")
    cfg = copy.deepcopy(cfg)
    apply_ablation(cfg, "wo_lagrangian")

    assert cfg["meta"]["dual_enabled"] is False
    assert float(cfg["meta"]["dual_lr"]) == 0.0
    assert cfg["meta"]["dual_lrs"] == [0.0, 0.0, 0.0, 0.0]

    checks = validate_training_config(cfg, "hard_stress", strict_thermal=False)
    assert checks["all_passed"] is True


def test_ablation_hard_clip_switches_safety_projection_mode():
    cfg = load_cfg("configs/default.yaml")
    cfg = copy.deepcopy(cfg)
    apply_ablation(cfg, "hard_clip")

    assert cfg["safety"]["projection_mode"] == "hard_clip"


def test_baseline_shin2024_disables_meta_and_sets_ddpg_hparams():
    cfg = load_cfg("configs/default.yaml")
    cfg = copy.deepcopy(cfg)
    apply_baseline_overrides(cfg, "shin2024_matched")

    assert cfg["context"]["enabled"] is False
    assert int(cfg["agent"]["z_dim"]) == 0
    assert cfg["meta"]["explicit_inner_outer"] is False
    assert cfg["meta"]["dual_enabled"] is False
    assert float(cfg["upper_dqn"]["epsilon_start"]) == 0.01
    assert float(cfg["upper_dqn"]["epsilon_final"]) == 0.01
    assert int(cfg["upper_dqn"]["replay_size"]) == 2000
    assert int(cfg["lower_ddpg"]["replay_size"]) == 1000000
    assert int(cfg["lower_ddpg"]["batch_size"]) == 64
    assert cfg["lower_ddpg"]["action_contract"] == "rho_tau_fixed_current"
    assert int(cfg["lower_ddpg"]["learned_action_dim"]) == 2
    assert float(cfg["lower_ddpg"]["fixed_current_fraction"]) == 0.5
    assert cfg["baseline_metadata"]["baseline_family"] == "shin2024_matched"
    assert cfg["baseline_metadata"]["exact_reproduction"] is False
    assert cfg["baseline_metadata"]["safety_protocol"] == "common_smooth_projection"
    assert cfg["baseline_metadata"]["fixed_mode_name"] == "HY"


def test_shin2024_matched_action_contract_forces_hy_and_fixed_currents(tmp_path):
    cfg = load_cfg("configs/default.yaml")
    cfg = apply_common_settings(
        copy.deepcopy(cfg),
        meta_iters=1,
        out_dir=tmp_path,
        run_name="shin_contract",
        seed=101,
        fast_mode=True,
        use_curriculum=False,
    )
    cfg["agent"]["hidden_dim"] = 32
    cfg["env"]["episode_len"] = 4
    apply_scenario(cfg, "thermal_rebalanced")
    apply_variant(cfg, "hybrid")
    apply_ablation(cfg, "full")
    apply_baseline_overrides(cfg, "shin2024_matched")

    trainer = Shin2024MatchedBaseline(cfg)
    assert trainer.lower.action_contract == "rho_tau_fixed_current"
    assert trainer.lower.learned_act_dim == 2
    assert trainer.lower.act_dim == 5
    assert trainer.dual_enabled is False

    task = sample_fixed_tasks(cfg, seed=101, n_tasks=1, seed_offset=21_000)[0]
    env = MultiTxUwSliptEnv(cfg, overrides=task.to_env_overrides())
    obs, _ = env.reset(seed=101)
    action, aux = trainer.act(obs, env, eval_mode=True)

    assert int(action["mode_exec"]) == 2
    assert int(action["upper_idx_exec"]) % 3 == 2
    assert np.asarray(aux["act_raw"]).shape == (5,)
    assert np.allclose(np.asarray(aux["act_raw"])[:3], 0.0, atol=1.0e-7)

    _, _, _, _, info = env.step(action)
    assert int(info["mode_exec"]) == 2
    assert int(info["upper_idx_exec"]) % 3 == 2


def test_baseline_dalal_switches_projection_mode():
    cfg = load_cfg("configs/default.yaml")
    cfg = copy.deepcopy(cfg)
    apply_baseline_overrides(cfg, "dalal2018_safe")

    assert cfg["safety"]["projection_mode"] == "dalal_safe"
