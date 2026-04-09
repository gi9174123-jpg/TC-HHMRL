from __future__ import annotations

import copy

from scripts.benchmark_constraint_scenarios import apply_ablation, apply_scenario, apply_variant, validate_training_config
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
