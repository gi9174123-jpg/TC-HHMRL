from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch

from scripts.benchmark_constraint_scenarios import (
    Shin2024MatchedBaseline,
    apply_common_settings,
    apply_ablation,
    apply_baseline_overrides,
    apply_scenario,
    apply_variant,
    formal_metadata_snapshot,
    inject_default_curriculum,
    sample_fixed_tasks,
    select_checkpoint,
    validate_training_config,
)
from tchhmrl.agents.ddpg_lower import LowerDDPG
from tchhmrl.envs.uw_slipt_env import MultiTxUwSliptEnv
from tchhmrl.safety.safety_layer import SafetyLayer
from tchhmrl.utils.config import load_cfg


def test_default_safety_uses_corrected_action_mapping_and_main_projection():
    cfg = load_cfg("configs/default.yaml")

    assert cfg["physics"]["physics_version"] == "physics_v2"
    assert cfg["physics"]["eh_model"] == "logistic"
    assert cfg["physics"]["thermal_model"] == "independent"
    assert cfg["physics"]["safety_projection_version"] == "independent_thermal_cap_v1"
    assert cfg["safety"]["projection_mode"] == "thermal_cap"
    assert cfg["safety"]["action_decode_mode"] == "tanh_affine"
    assert cfg["safety"]["current_decoder"] == "structured_total_allocation"
    assert cfg["safety"]["allocation_logit_scale"] == 2.0
    assert cfg["safety"]["inactive_source_mask_mode"] == "hard_zero"
    assert float(cfg["safety"]["smooth_relaxed_margin_c"]) == 1.0
    assert float(cfg["safety"]["thermal_cap_margin_c"]) == 0.5
    assert cfg["adaptive_thermal"]["enabled"] is True
    assert float(cfg["adaptive_thermal"]["excitation_threshold"]) == 0.03
    assert float(cfg["adaptive_thermal"]["uncertainty_beta_min"]) <= float(
        cfg["adaptive_thermal"]["uncertainty_beta_max"]
    )
    assert float(cfg["adaptive_thermal"]["initial_std"]) == 0.5
    assert np.allclose(cfg["adaptive_thermal"]["initial_effective_gain"], cfg["safety"]["effective_gain_initial"])
    assert cfg["physical_context"]["enabled"] is True
    assert int(cfg["physical_context"]["input_dim"]) == 18
    assert int(cfg["physical_context"]["embedding_dim"]) == 32
    assert cfg["constraint_critics"]["enabled"] is True
    assert int(cfg["constraint_critics"]["out_dim"]) == 4
    assert cfg["constraint_critics"]["actor_weights"] == [0.05, 0.10, 0.10, 0.10]
    assert cfg["constraint_critics"]["actor_penalty_nonnegative"] is True
    assert cfg["constraint_replay"]["enabled"] is True
    assert cfg["constraint_replay"]["uniform_fraction"] == 0.50
    assert cfg["constraint_replay"]["boundary_fraction"] == 0.30
    assert cfg["constraint_replay"]["violation_fraction"] == 0.20
    assert cfg["constraint_replay"]["thermal_headroom_threshold"] == 2.0
    assert cfg["constraint_replay"]["bus_utilization_threshold"] == 0.85
    assert cfg["constraint_replay"]["projection_residual_threshold"] == 0.10
    assert cfg["constraint_replay"]["temperature_slope_threshold"] == 0.30
    assert cfg["constraint_replay"]["importance_weighting"] is True
    assert cfg["constraint_replay"]["importance_weight_clip"] == [0.25, 4.0]
    assert cfg["constraint_replay"]["require_boundary_thresholds"] is True
    assert cfg["upper_dqn"]["double_dqn"] is True
    assert cfg["upper_dqn"]["dueling"] is True


def test_main_cli_defaults_match_strict_meta_evaluation_scale(monkeypatch):
    from scripts import benchmark_constraint_scenarios as bench
    from scripts import benchmark_hybrid_vs_single as structural

    monkeypatch.setattr("sys.argv", ["benchmark_constraint_scenarios.py"])
    args = bench.parse_args()
    assert list(args.scenarios) == ["moderate_practical", "hard_stress"]
    assert int(args.eval_tasks) == 10
    assert int(args.eval_eps) == 3
    assert int(args.env_tasks) == 8
    assert int(args.env_eps) == 1

    monkeypatch.setattr("sys.argv", ["benchmark_hybrid_vs_single.py"])
    args = structural.parse_args()
    assert int(args.eval_tasks) == 10
    assert int(args.eval_eps) == 3
    assert int(args.env_tasks) == 8
    assert int(args.env_eps) == 1


def test_formal_metadata_reports_model_aware_lower_components():
    cfg = load_cfg("configs/default.yaml")
    meta = formal_metadata_snapshot(cfg)

    assert meta["adaptive_thermal_enabled"] is True
    assert meta["adaptive_thermal_estimator"] == "ema_effective_gain"
    assert meta["thermal_parameter_source"] == "nominal_plus_online_effective_gain"
    assert meta["controller_uses_task_gamma_delta"] is False
    assert np.allclose(meta["effective_gain_initial"], cfg["safety"]["effective_gain_initial"])
    assert meta["current_decoder"] == "structured_total_allocation"
    assert meta["inactive_source_mask_mode"] == "hard_zero"
    assert meta["policy_distribution_space"] == "latent_structured_action"
    assert meta["critic_action_space"] == "executed_physical_action"
    assert meta["entropy_space"] == "mode_boost_masked_latent_action"
    assert meta["physical_context_enabled"] is True
    assert meta["constraint_critics_enabled"] is True
    assert meta["constraint_critic_dim"] == 4
    assert meta["constraint_reward_target"] == "reward_task"
    assert meta["reward_training_target"] == "reward_task"
    assert meta["reward_benchmark_includes_fixed_cost_penalty"] is True
    assert meta["reward_critic_uses_fixed_cost_penalty"] is False
    assert meta["constraint_actor_weights"] == [0.05, 0.10, 0.10, 0.10]
    assert meta["constraint_actor_penalty_nonnegative"] is True
    assert meta["constraint_replay_enabled"] is True
    assert meta["constraint_replay_uniform_fraction"] == 0.50
    assert meta["constraint_replay_boundary_fraction"] == 0.30
    assert meta["constraint_replay_violation_fraction"] == 0.20
    assert meta["constraint_replay_importance_weighting"] is True
    assert meta["constraint_replay_importance_weight_clip"] == [0.25, 4.0]
    assert meta["constraint_replay_thermal_headroom_threshold"] == 2.0
    assert meta["constraint_replay_bus_utilization_threshold"] == 0.85
    assert meta["constraint_replay_projection_residual_threshold"] == 0.10
    assert meta["constraint_replay_temperature_slope_threshold"] == 0.30
    assert meta["constraint_replay_reward_batch"] == "unchanged_reward_critic_batch"
    assert meta["constraint_replay_constraint_batch"] == "stratified_uniform_boundary_violation"
    assert meta["residual_planner_enabled"] is True
    assert meta["residual_planner_candidate_count"] == 24
    assert meta["residual_planner_adaptive_budget_enabled"] is True
    assert meta["residual_planner_budget_candidates"] == [0, 8, 16, 24]
    assert meta["residual_planner_budget_rule"] == "rule_based_current_risk"
    assert "target_critic_disagreement" in meta["residual_planner_budget_inputs"]
    assert meta["residual_planner_thermal_horizon"] == 2
    assert meta["residual_planner_start_meta_iter"] == 60
    assert meta["residual_planner_thermal_horizon_start_meta_iter"] == 86
    assert meta["transition_schema_version"] == "transition_schema_v2_fail_fast"
    assert meta["reward_schema_version"] == "reward_task_plus_benchmark_and_constraint_cost_vec_v2"
    assert meta["action_contract_version"] == "policy_planner_executed_action_v1"
    assert meta["residual_planner_scoring"] == "target_critics_plus_constraint_incremental_h2_risk"
    assert meta["residual_planner_trust_region_enabled"] is True
    assert meta["residual_planner_replacement_margin_mode"] == "normalized"
    assert meta["residual_planner_replacement_margin"] is None
    assert meta["residual_planner_total_current_raw_step"] == 0.10
    assert meta["residual_planner_allocation_logit_raw_step"] == 0.10
    assert meta["residual_planner_ratio_raw_step"] == 0.10
    assert meta["residual_planner_trust_region_effective"] is False
    assert meta["residual_planner_normalized_margin_factor"] == 0.25
    assert meta["residual_planner_constraint_non_degradation"] is True
    assert meta["residual_planner_h2_veto_enabled"] is True
    assert meta["residual_planner_uses_env_reward_model"] is False
    assert meta["upper_double_dqn"] is True
    assert meta["upper_dueling_dqn"] is True
    assert meta["upper_physical_context_enabled"] is True


def test_moderate_config_keeps_model_aware_lower_components_enabled():
    cfg = load_cfg("configs/moderate.yaml")

    assert cfg["adaptive_thermal"]["enabled"] is True
    assert cfg["physical_context"]["enabled"] is True
    assert int(cfg["physical_context"]["input_dim"]) == 18
    assert cfg["constraint_critics"]["enabled"] is True
    assert int(cfg["constraint_critics"]["out_dim"]) == 4
    assert cfg["constraint_critics"]["reward_target"] == "reward_task"
    assert cfg["constraint_replay"]["enabled"] is True
    assert cfg["constraint_replay"]["uniform_fraction"] == 0.50
    assert cfg["constraint_replay"]["boundary_fraction"] == 0.30
    assert cfg["constraint_replay"]["violation_fraction"] == 0.20
    assert cfg["constraint_replay"]["thermal_headroom_threshold"] == 2.0
    assert cfg["constraint_replay"]["bus_utilization_threshold"] == 0.85
    assert cfg["constraint_replay"]["projection_residual_threshold"] == 0.10
    assert cfg["residual_planner"]["enabled"] is True
    assert int(cfg["residual_planner"]["candidate_count"]) == 24
    assert cfg["residual_planner"]["adaptive_budget_enabled"] is True
    assert cfg["residual_planner"]["budget_candidates"] == [0, 8, 16, 24]
    assert cfg["residual_planner"]["replacement_margin_mode"] == "normalized"
    assert cfg["residual_planner"]["replacement_margin"] is None
    assert int(cfg["residual_planner"]["thermal_horizon_start_meta_iter"]) == 86
    assert int(cfg["meta"]["checkpoint_selection"]["min_iter"]) == 86
    assert cfg["upper_dqn"]["double_dqn"] is True
    assert cfg["upper_dqn"]["dueling"] is True
    assert int(cfg["context"]["input_dim"]) == 37


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
    assert cfg["meta"]["support_gate"]["enabled"] is False
    assert int(cfg["meta"]["query_episodes"]) == 0
    assert int(cfg["meta"]["support_episodes"]) > 0


def test_full_hybrid_support_gate_default_and_ungated_ablation():
    cfg = load_cfg("configs/default.yaml")
    assert cfg["meta"]["support_gate"]["enabled"] is True
    assert cfg["meta"]["support_gate"]["role"] == "rollback_guard"
    assert cfg["meta"]["support_gate"]["rule"] == "safety_first"
    assert cfg["meta"]["support_gate"]["query_leakage"] is False
    assert float(cfg["meta"]["support_gate"]["max_violation_increase"]) == 1.0e-4
    assert float(cfg["meta"]["support_gate"]["max_cost_increase"]) == 1.0e-5
    assert int(cfg["meta"]["support_adaptation_episodes"]) == 3
    assert int(cfg["meta"]["support_gate_validation_episodes"]) == 2
    assert cfg["meta"]["support_gate"]["paired_validation"] is True
    assert int(cfg["meta"]["support_gate"]["extra_support_rollouts"]) == 2
    assert int(cfg["meta"]["support_gate"]["extra_gradient_updates"]) == 0

    ungated = copy.deepcopy(cfg)
    apply_ablation(ungated, "meta_ungated")
    assert ungated["meta"]["support_gate"]["enabled"] is False
    assert ungated["meta"]["support_update_acceptance"] == "unconditional"
    assert ungated["pilot_metadata"]["comparison_role"] == "ungated_meta_ablation"


def test_checkpoint_selection_honors_stage_aware_min_iter(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    for it in (40, 70, 90, 100):
        (ckpt_dir / f"iter_{it}.pt").write_bytes(b"checkpoint")
    run_df = pd.DataFrame(
        [
            {
                "iter": 40,
                "query_reward": 10.0,
                "query_se": 0.0,
                "query_eh": 0.0,
                "query_cost": 0.0,
                "query_violation_rate": 0.0,
            },
            {
                "iter": 90,
                "query_reward": 1.0,
                "query_se": 0.0,
                "query_eh": 0.0,
                "query_cost": 0.0,
                "query_violation_rate": 0.0,
            },
            {
                "iter": 100,
                "query_reward": 2.0,
                "query_se": 0.0,
                "query_eh": 0.0,
                "query_cost": 0.0,
                "query_violation_rate": 0.0,
            },
        ]
    )

    pick = select_checkpoint(
        run_df=run_df,
        ckpt_dir=ckpt_dir,
        score_cfg={
            "enabled": True,
            "mode": "training_curve",
            "min_iter": 86,
            "reward_w": 1.0,
            "se_w": 0.0,
            "eh_w": 0.0,
            "cost_w": 0.0,
            "violation_w": 0.0,
        },
    )

    assert pick["min_iter_satisfied"] is True
    assert int(pick["min_iter"]) == 86
    assert int(pick["selected_iter"]) == 100
    assert all(int(row["iter"]) >= 86 for row in pick["selection_rows"])


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
    assert cfg["pilot_metadata"]["projection_variant"] == "naive_component_wise_clip"
    assert cfg["pilot_metadata"]["comparison_role"] == "diagnostic_clip_ablation"
    assert cfg["pilot_metadata"]["strong_safety_baseline"] is False
    assert cfg["pilot_metadata"]["qos_recovery_rule"] == "none_componentwise_zero_if_thermal_infeasible"


def test_ablation_qos_aware_hard_clip_sets_fair_projection_metadata():
    cfg = load_cfg("configs/default.yaml")
    cfg = copy.deepcopy(cfg)
    apply_ablation(cfg, "qos_aware_hard_clip")

    assert cfg["safety"]["projection_mode"] == "qos_aware_hard_clip"
    assert cfg["pilot_metadata"]["projection_variant"] == "qos_aware_feasible_hard_projection"
    assert cfg["pilot_metadata"]["comparison_role"] == "fair_hard_projection_baseline"
    assert cfg["pilot_metadata"]["strong_safety_baseline"] is True
    assert cfg["pilot_metadata"]["oracle_future_disturbances"] is False
    assert "non_oracle_current_recovery" in cfg["pilot_metadata"]["qos_recovery_rule"]


def test_ablation_smooth_relaxed_sets_pilot_metadata():
    cfg = load_cfg("configs/default.yaml")
    cfg = copy.deepcopy(cfg)
    apply_ablation(cfg, "smooth_relaxed")

    assert cfg["safety"]["projection_mode"] == "smooth_relaxed"
    assert float(cfg["safety"]["smooth_relaxed_margin_c"]) == 1.0
    assert cfg["pilot_metadata"]["projection_variant"] == "smooth_relaxed"
    assert cfg["pilot_metadata"]["pilot_only"] is True
    assert cfg["pilot_metadata"]["formal_ranking_exclude"] is True
    assert cfg["pilot_metadata"]["comparison_role"] == "projection_sensitivity"


def test_ablation_thermal_cap_sets_pilot_metadata():
    cfg = load_cfg("configs/default.yaml")
    cfg = copy.deepcopy(cfg)
    apply_ablation(cfg, "thermal_cap")

    assert cfg["safety"]["projection_mode"] == "thermal_cap"
    assert float(cfg["safety"]["thermal_cap_margin_c"]) == 0.5
    assert cfg["pilot_metadata"]["projection_variant"] == "thermal_cap"
    assert cfg["pilot_metadata"]["pilot_only"] is True
    assert cfg["pilot_metadata"]["formal_ranking_exclude"] is True
    assert cfg["pilot_metadata"]["comparison_role"] == "projection_sensitivity"
    assert float(cfg["pilot_metadata"]["thermal_cap_margin_c"]) == 0.5


def test_baseline_shin2024_disables_meta_and_sets_ddpg_hparams():
    cfg = load_cfg("configs/default.yaml")
    cfg = copy.deepcopy(cfg)
    apply_baseline_overrides(cfg, "shin2024_matched")

    assert cfg["context"]["enabled"] is False
    assert int(cfg["agent"]["z_dim"]) == 0
    assert cfg["meta"]["explicit_inner_outer"] is False
    assert cfg["meta"]["support_gate"]["enabled"] is False
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
    assert cfg["baseline_metadata"]["safety_protocol"] == "common_thermal_cap_projection"
    assert cfg["baseline_metadata"]["fixed_mode_name"] == "HY"
    assert cfg["baseline_metadata"]["fixed_current_template"] == "tanh_affine_fraction"
    assert cfg["baseline_metadata"]["meta_learning"] is False
    assert cfg["baseline_metadata"]["support_gate"] is False


def test_non_meta_baseline_overrides_disable_support_gate():
    for baseline in [
        "shin2024_matched",
        "shin2024_adapted_codebook",
        "shin2024_adapted_codebook_tuned",
        "sac_lagrangian",
        "sac_dalal_safe",
        "uysal_policy_optimizer",
        "mpc_grid",
        "javadi_ppo_dimming",
        "deeprat_assignment_power",
        "pdqn_hybrid_action",
    ]:
        cfg = load_cfg("configs/default.yaml")
        cfg = copy.deepcopy(cfg)
        apply_baseline_overrides(cfg, baseline)

        assert cfg["meta"]["explicit_inner_outer"] is False
        assert cfg["meta"]["support_gate"]["enabled"] is False
        assert cfg["meta"]["query_updates_enabled"] is False
        assert cfg["context"]["enabled"] is False


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


def test_shin2024_adapted_codebook_uses_upper_current_template_and_hy(tmp_path):
    cfg = load_cfg("configs/default.yaml")
    cfg = apply_common_settings(
        copy.deepcopy(cfg),
        meta_iters=1,
        out_dir=tmp_path,
        run_name="shin_codebook_contract",
        seed=101,
        fast_mode=True,
        use_curriculum=False,
    )
    cfg["agent"]["hidden_dim"] = 32
    cfg["env"]["episode_len"] = 4
    apply_scenario(cfg, "thermal_rebalanced")
    apply_variant(cfg, "hybrid")
    apply_ablation(cfg, "full")
    apply_baseline_overrides(cfg, "shin2024_adapted_codebook")

    assert cfg["lower_ddpg"]["action_contract"] == "rho_tau_codebook_current"
    assert cfg["lower_ddpg"]["upper_contract"] == "boost_intensity_codeword"
    assert int(cfg["lower_ddpg"]["learned_action_dim"]) == 2
    assert cfg["lower_ddpg"]["current_template_codeword_names"] == ["low_safe", "balanced", "high_performance"]
    assert cfg["lower_ddpg"]["current_template_codewords"] == [
        [0.40, 0.25, 0.25],
        [0.55, 0.45, 0.45],
        [0.70, 0.65, 0.65],
    ]
    assert cfg["baseline_metadata"]["baseline_family"] == "shin2024_adapted_codebook"
    assert cfg["baseline_metadata"]["paper_inspired"] is True
    assert cfg["baseline_metadata"]["upper_action_contract"] == "boost_combo_intensity_codeword"
    assert cfg["baseline_metadata"]["lower_action_contract"] == "rho_tau_only"
    assert cfg["baseline_metadata"]["fixed_mode_name"] == "HY"
    assert cfg["baseline_metadata"]["learned_current_allocation"] is False
    assert (
        cfg["baseline_metadata"]["mapped_original_control"]
        == "beam_divergence_angle_to_source_intensity_codeword"
    )

    trainer = Shin2024MatchedBaseline(cfg)
    assert trainer.lower.action_contract == "rho_tau_codebook_current"
    assert trainer.lower.upper_contract == "boost_intensity_codeword"
    assert trainer.lower.learned_act_dim == 2

    task = sample_fixed_tasks(cfg, seed=101, n_tasks=1, seed_offset=21_000)[0]
    env = MultiTxUwSliptEnv(cfg, overrides=task.to_env_overrides())
    obs, _ = env.reset(seed=101)

    trainer.upper_plan = 3  # boost combo 1, low current template; safety execution remains HY.
    action, aux = trainer.act(obs, env, eval_mode=True)

    assert int(action["mode_exec"]) == 2
    assert int(action["upper_idx_exec"]) % 3 == 2
    assert int(aux["upper_idx_raw"]) == 3
    assert int(aux["upper_idx_train"]) == 3
    assert int(aux["upper_idx_safety_raw"]) == 5
    assert int(aux["current_template_level_exec"]) == 0
    assert np.asarray(aux["act_raw"]).shape == (5,)
    assert np.allclose(np.asarray(aux["act_raw"])[:3], [-0.20, -0.50, -0.50], atol=1.0e-6)

    _, _, _, _, info = env.step(action)
    assert int(info["mode_exec"]) == 2
    assert int(info["upper_idx_exec"]) % 3 == 2


def test_lower_ddpg_rejects_inconsistent_codebook_contracts():
    cfg = load_cfg("configs/default.yaml")
    cfg = copy.deepcopy(cfg)
    apply_baseline_overrides(cfg, "shin2024_adapted_codebook")
    cfg["lower_ddpg"]["upper_contract"] = "boost_mode"

    with np.testing.assert_raises(ValueError):
        LowerDDPG(cfg, SafetyLayer(cfg), device=torch.device("cpu"))


def test_baseline_dalal_switches_projection_mode():
    cfg = load_cfg("configs/default.yaml")
    cfg = copy.deepcopy(cfg)
    apply_baseline_overrides(cfg, "dalal2018_safe")

    assert cfg["safety"]["projection_mode"] == "dalal_safe"


def test_baseline_sac_dalal_safe_disables_meta_dual_and_sets_metadata():
    cfg = load_cfg("configs/default.yaml")
    cfg = copy.deepcopy(cfg)
    apply_baseline_overrides(cfg, "sac_dalal_safe")

    assert cfg["context"]["enabled"] is False
    assert int(cfg["agent"]["z_dim"]) == 0
    assert cfg["meta"]["explicit_inner_outer"] is False
    assert cfg["meta"]["query_updates_enabled"] is False
    assert cfg["meta"]["dual_enabled"] is False
    assert float(cfg["meta"]["dual_lr"]) == 0.0
    assert cfg["meta"]["dual_lrs"] == [0.0, 0.0, 0.0, 0.0]
    assert cfg["safety"]["projection_mode"] == "dalal_safe"
    assert cfg["baseline_metadata"]["baseline_family"] == "sac_dalal_safe"
    assert cfg["baseline_metadata"]["exact_reproduction"] is False
    assert cfg["baseline_metadata"]["external_baseline"] is True
    assert cfg["baseline_metadata"]["safety_protocol"] == "dalal_style_projection"
    assert cfg["baseline_metadata"]["comparison_role"] == "external_safety_layer_baseline"


def test_baseline_mpc_lite_oracle_sets_metadata():
    cfg = copy.deepcopy(load_cfg("configs/default.yaml"))
    apply_baseline_overrides(cfg, "mpc_lite_oracle")

    meta = cfg["baseline_metadata"]
    assert meta["baseline_family"] == "mpc_lite_oracle"
    assert meta["uses_task_oracle"] is True
    assert meta["uses_learned_policy"] is False
    assert meta["uses_same_safety_projection"] is True
    assert int(meta["horizon"]) == 1
    assert int(meta["candidate_count"]) == 256
    assert meta["comparison_role"] == "model_based_optimizer"
