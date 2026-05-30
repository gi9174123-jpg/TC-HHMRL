from __future__ import annotations

import copy

import numpy as np
import torch

from scripts.benchmark_constraint_scenarios import (
    apply_ablation,
    apply_baseline_overrides,
    apply_common_settings,
    apply_scenario,
    apply_variant,
)
from tchhmrl.agents.ddpg_lower import LowerDDPG
from tchhmrl.baselines import (
    DeepRATAssignmentPowerBaseline,
    JavadiPPODimmingBaseline,
    MpcGridBaseline,
    PDQNHybridActionBaseline,
    UysalPolicyOptimizer,
)
from tchhmrl.envs.uw_slipt_env import MultiTxUwSliptEnv
from tchhmrl.safety.safety_layer import SafetyLayer
from tchhmrl.utils.config import load_cfg


PAPER_EXPLANATION_FIELDS = {
    "paper_core_mechanism",
    "adapted_mapping_to_tc_hhmrl",
    "domain_match",
    "environment_dependency",
    "not_exact_reproduction_reason",
}


def _cfg_for_baseline(tmp_path, baseline: str):
    cfg = load_cfg("configs/default.yaml")
    cfg = apply_common_settings(
        copy.deepcopy(cfg),
        meta_iters=1,
        out_dir=tmp_path,
        run_name=f"{baseline}_contract",
        seed=101,
        fast_mode=True,
        use_curriculum=False,
    )
    cfg["agent"]["hidden_dim"] = 24
    cfg["env"]["episode_len"] = 2
    cfg["meta"]["n_tasks_per_iter"] = 1
    cfg["meta"]["support_episodes"] = 1
    cfg["meta"]["query_episodes"] = 0
    apply_scenario(cfg, "moderate_practical")
    apply_variant(cfg, "hybrid")
    apply_ablation(cfg, "full")
    apply_baseline_overrides(cfg, baseline)
    return cfg


def test_shin_adapted_codebook_contract_is_source_aware(tmp_path):
    cfg = _cfg_for_baseline(tmp_path, "shin2024_adapted_codebook")
    lower = LowerDDPG(cfg, SafetyLayer(cfg), device=torch.device("cpu"))

    assert lower.action_contract == "rho_tau_codebook_current"
    assert lower.upper_contract == "boost_intensity_codeword"
    assert lower.learned_act_dim == 2
    assert lower.current_template_levels.shape == (3, 3)
    assert np.allclose(lower.current_template_levels[0], [0.40, 0.25, 0.25])
    assert cfg["baseline_metadata"]["fixed_mode_exec"] == 2
    assert cfg["baseline_metadata"]["learned_current_allocation"] is False
    assert cfg["baseline_metadata"]["paper_inspired"] is True
    assert cfg["baseline_metadata"]["rho_symbol_mapping"] == (
        "paper_rho_is_id_fraction; env_rho_exec_is_eh_fraction; paper_rho=1-env_rho_exec"
    )
    assert PAPER_EXPLANATION_FIELDS.issubset(cfg["baseline_metadata"])


def test_uysal_policy_optimizer_is_threshold_rule_not_oracle(tmp_path):
    cfg = _cfg_for_baseline(tmp_path, "uysal_policy_optimizer")
    policy = UysalPolicyOptimizer(cfg)
    env = MultiTxUwSliptEnv(cfg)
    obs, _ = env.reset(seed=101)
    action, aux = policy.act(obs, env, eval_mode=True)

    assert set(policy.subpolicy_names) == {"uysal_ts", "uysal_ps", "uysal_tsps", "uysal_ads"}
    assert cfg["baseline_metadata"]["uses_learned_policy"] is False
    assert cfg["baseline_metadata"]["policy_selection_rule"] == "predefined_ads_threshold_not_oracle_best_of_four"
    assert cfg["baselines"]["uysal_policy_optimizer"]["eh_min_target"] == 0.002
    assert cfg["baseline_metadata"]["eh_threshold_default"] == 0.002
    assert cfg["baseline_metadata"]["eh_threshold_calibration"] == "fixed_metric_scale_from_smoke_not_reward_optimized"
    assert aux["selected_uysal_controller"] == "uysal_ads"
    assert aux["selected_uysal_subpolicy"] in {"uysal_ts", "uysal_ps", "uysal_tsps"}
    assert aux["uysal_policy_rule"] == "ads_threshold_scheduler"
    assert aux["eh_threshold_source"] == "baselines.uysal_policy_optimizer.eh_min_target"
    assert "ads_balanced_predicted_qos_rate" in aux
    assert "ads_balanced_predicted_eh_metric" in aux
    assert aux["ads_qos_threshold"] == cfg["env"]["qos_min_rate"]
    assert aux["ads_eh_threshold"] == 0.002
    assert aux["ads_qos_deficit"] >= 0.0
    assert aux["ads_eh_deficit"] >= 0.0
    assert aux["ads_decision_reason"] in {
        "qos_below_threshold_select_ts",
        "eh_below_threshold_select_ps",
        "qos_and_eh_satisfied_select_tsps",
    }
    assert aux["paper_rho_equiv"] == 1.0 - aux["rho_exec"]
    assert aux["selected_env_rho"] == aux["rho_exec"]
    assert aux["selected_paper_rho"] == 1.0 - aux["selected_env_rho"]
    assert int(action["mode_exec"]) in {0, 1, 2}
    assert PAPER_EXPLANATION_FIELDS.issubset(cfg["baseline_metadata"])


def test_mpc_grid_uses_structured_templates_and_preserves_state_rng(tmp_path):
    cfg = _cfg_for_baseline(tmp_path, "mpc_grid")
    policy = MpcGridBaseline(cfg)
    env = MultiTxUwSliptEnv(cfg)
    obs, _ = env.reset(seed=101)

    temps_before = env.temps.copy()
    t_before = int(env.t)
    rng_before = repr(env.rng.bit_generator.state)
    _ = policy.score_candidate(
        env,
        boost_combo=3,
        mode=2,
        template_name="balanced_hybrid",
        rho=0.5,
        tau=0.5,
    )
    assert int(env.t) == t_before
    assert np.allclose(env.temps, temps_before)
    assert repr(env.rng.bit_generator.state) == rng_before

    action, aux = policy.act(obs, env, eval_mode=True)
    assert cfg["baseline_metadata"]["uses_learned_policy"] is False
    assert policy.candidate_count == 700
    assert int(aux["candidate_count"]) == 700
    assert aux["selected_template"] in policy.current_templates
    assert "online_latency_ms" in aux
    assert "predicted_qos_rate" in aux
    assert "predicted_eh_metric" in aux
    assert "predicted_snr" in aux
    assert "predicted_bus_utilization" in aux
    assert aux["selected_env_rho"] == aux["rho_exec"]
    assert aux["selected_paper_rho"] == 1.0 - aux["selected_env_rho"]
    assert int(action["upper_idx_exec"]) == int(action["boost_combo_exec"]) * 3 + int(action["mode_exec"])
    assert PAPER_EXPLANATION_FIELDS.issubset(cfg["baseline_metadata"])


def test_javadi_ppo_dimming_contract(tmp_path):
    cfg = _cfg_for_baseline(tmp_path, "javadi_ppo_dimming")
    policy = JavadiPPODimmingBaseline(cfg)
    env = MultiTxUwSliptEnv(cfg)
    obs, _ = env.reset(seed=101)
    action, aux = policy.act(obs, env, eval_mode=True)

    assert cfg["baseline_metadata"]["policy_family"] == "PPO"
    assert cfg["baseline_metadata"]["source_selection_rl"] is True
    assert cfg["baseline_metadata"]["joint_dimming"] is True
    assert cfg["baseline_metadata"]["dimming_type"] == "common_dimming_scale"
    assert cfg["baseline_metadata"]["source_subset_contract"] == "anchor_plus_optional_ld_boosts"
    assert cfg["baseline_metadata"]["active_source_selection_mapping"] == "active_led_selection_to_hybrid_boost_subset"
    assert cfg["baseline_metadata"]["continuous_policy_dim"] == 3
    assert cfg["baseline_metadata"]["domain_match"] == "owc_slipt_not_underwater"
    assert int(action["mode_exec"]) == 2
    assert "source_subset_id" in aux
    assert "joint_dimming_scale" in aux
    assert "joint_dimming_scale_tx0" in aux
    assert aux["joint_dimming_scale_tx0"] == aux["joint_dimming_scale_tx1"] == aux["joint_dimming_scale_tx2"]
    assert aux["selected_env_rho"] == aux["rho_exec"]
    assert aux["selected_paper_rho"] == 1.0 - aux["selected_env_rho"]
    assert "predicted_qos_rate" in aux
    assert "predicted_eh_metric" in aux
    assert np.asarray(aux["ppo_cont_raw"]).shape == (3,)
    assert PAPER_EXPLANATION_FIELDS.issubset(cfg["baseline_metadata"])


def test_deeprat_assignment_power_contract(tmp_path):
    cfg = _cfg_for_baseline(tmp_path, "deeprat_assignment_power")
    policy = DeepRATAssignmentPowerBaseline(cfg)
    env = MultiTxUwSliptEnv(cfg)
    obs, _ = env.reset(seed=101)
    action, aux = policy.act(obs, env, eval_mode=True)

    assert int(cfg["agent"]["n_upper_actions"]) == 4
    assert policy.upper.n_actions == 4
    assert cfg["baseline_metadata"]["discrete_assignment_dim"] == 4
    assert policy.lower.action_contract == "current_allocation_only"
    assert policy.lower.upper_contract == "source_assignment"
    assert policy.lower.learned_act_dim == 3
    assert cfg["baseline_metadata"]["receiver_ratio_rule"] == "fixed_balanced_not_deeprat_core"
    assert cfg["baseline_metadata"]["assignment_mapping"] == "rat_assignment_to_hybrid_source_boost_assignment"
    assert cfg["baseline_metadata"]["power_allocation_mapping"] == "rat_power_allocation_to_tx_current_allocation"
    assert cfg["baseline_metadata"]["domain_match"] == "wireless_resource_allocation_not_slipt"
    assert int(action["mode_exec"]) == 2
    assert aux["selected_env_rho"] == aux["rho_exec"]
    assert aux["selected_paper_rho"] == 1.0 - aux["selected_env_rho"]
    assert "predicted_qos_rate" in aux
    assert "predicted_eh_metric" in aux
    assert PAPER_EXPLANATION_FIELDS.issubset(cfg["baseline_metadata"])


def test_pdqn_parameterized_action_contract(tmp_path):
    cfg = _cfg_for_baseline(tmp_path, "pdqn_hybrid_action")
    policy = PDQNHybridActionBaseline(cfg)
    obs = torch.zeros((2, int(cfg["agent"]["obs_dim"])), dtype=torch.float32, device=policy.device)

    params = policy.param_net(obs)
    action_idx, selected, q_all = policy.select_parameterized_action(obs)

    assert params.shape == (2, 12, 5)
    assert selected.shape == (2, 5)
    assert torch.equal(action_idx, torch.argmax(q_all, dim=1))
    assert cfg["baseline_metadata"]["parameterized_action"] is True

    env = MultiTxUwSliptEnv(cfg)
    obs_np, _ = env.reset(seed=101)
    _, aux = policy.act(obs_np, env, eval_mode=True)
    assert aux["selected_env_rho"] == aux["rho_exec"]
    assert aux["selected_paper_rho"] == 1.0 - aux["selected_env_rho"]
    assert "predicted_qos_rate" in aux
    assert "predicted_eh_metric" in aux
    assert PAPER_EXPLANATION_FIELDS.issubset(cfg["baseline_metadata"])


def test_new_baselines_can_run_one_episode_smoke(tmp_path):
    baseline_classes = {
        "uysal_policy_optimizer": UysalPolicyOptimizer,
        "mpc_grid": MpcGridBaseline,
        "javadi_ppo_dimming": JavadiPPODimmingBaseline,
        "deeprat_assignment_power": DeepRATAssignmentPowerBaseline,
        "pdqn_hybrid_action": PDQNHybridActionBaseline,
    }
    for baseline, cls in baseline_classes.items():
        cfg = _cfg_for_baseline(tmp_path / baseline, baseline)
        policy = cls(cfg)
        env = MultiTxUwSliptEnv(cfg)
        stats = policy._run_episode(env, train=False)

        assert np.isfinite(stats.reward)
        assert np.isfinite(stats.se)
        assert np.isfinite(stats.eh)
        assert np.isfinite(stats.cost)
        assert 0.0 <= stats.violation_rate <= 1.0
        assert np.isfinite(stats.temp_max)
        assert np.isfinite(stats.bus_utilization)
        assert cfg["baseline_metadata"]["paper_inspired"] is True
        assert cfg["baseline_metadata"]["exact_reproduction"] is False
