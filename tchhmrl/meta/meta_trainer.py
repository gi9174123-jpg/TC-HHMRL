from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from tchhmrl.agents.hierarchical_agent import HierarchicalAgent
from tchhmrl.constraints.dual_layer import DualLayer
from tchhmrl.envs.task_sampler import TaskSampler
from tchhmrl.envs.task_contract import build_context_task_summary_v2, task_defaults_from_cfg
from tchhmrl.envs.uw_slipt_env import MultiTxUwSliptEnv
from tchhmrl.meta.support_gate import SupportGateDecision, SupportGateStats, evaluate_support_gate
from tchhmrl.utils.config import resolve_device
from tchhmrl.utils.logger import Logger
from tchhmrl.utils.seed import set_seed


@dataclass
class EpisodeStats:
    reward: float
    reward_task: float
    reward_benchmark: float
    reward_dual_penalized: float
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
    eh_input_eff: float = 0.0
    eh_metric_raw_nonlinear: float = 0.0
    eh_saturation_fraction: float = 0.0
    eh_near_zero_fraction: float = 0.0
    residual_planner_active_rate: float = 0.0
    residual_planner_replacement_rate: float = 0.0
    residual_planner_fallback_rate: float = 0.0
    residual_planner_score_improvement: float = 0.0
    residual_planner_best_score_improvement: float = 0.0
    residual_planner_candidate_distance: float = 0.0
    residual_planner_disagreement: float = 0.0
    residual_planner_budget_mean: float = 0.0
    residual_planner_budget_k0_rate: float = 0.0
    residual_planner_budget_k8_rate: float = 0.0
    residual_planner_budget_k16_rate: float = 0.0
    residual_planner_budget_k24_rate: float = 0.0
    cost_qos: float = 0.0
    cost_temp_anchor: float = 0.0
    cost_temp_boost1: float = 0.0
    cost_temp_boost2: float = 0.0
    temp0_max: float = 0.0
    temp1_max: float = 0.0
    temp2_max: float = 0.0
    thermal_headroom_min: float = 0.0
    mode_ps_rate: float = 0.0
    mode_ts_rate: float = 0.0
    mode_hy_rate: float = 0.0
    boost_anchor_rate: float = 0.0
    boost_ld1_rate: float = 0.0
    boost_ld2_rate: float = 0.0
    boost_all_rate: float = 0.0
    projected_current_total_mean: float = 0.0
    bus_utilization_mean: float = 0.0
    projection_residual_mean: float = 0.0
    thermal_gain_mean0: float = 0.0
    thermal_gain_mean1: float = 0.0
    thermal_gain_mean2: float = 0.0
    thermal_gain_std0: float = 0.0
    thermal_gain_std1: float = 0.0
    thermal_gain_std2: float = 0.0
    thermal_gain_safe_scale0: float = 0.0
    thermal_gain_safe_scale1: float = 0.0
    thermal_gain_safe_scale2: float = 0.0
    thermal_valid_count0: float = 0.0
    thermal_valid_count1: float = 0.0
    thermal_valid_count2: float = 0.0
    planner_h2_veto_rate: float = 0.0
    planner_constraint_rejection_rate: float = 0.0
    planner_h1_thermal_risk: float = 0.0
    planner_h2_thermal_risk: float = 0.0
    planner_h2_max_temperature: float = 0.0
    planner_target_constraint_value: float = 0.0
    upper_shield_applied_rate: float = 0.0
    upper_shield_ld1_disabled_rate: float = 0.0
    upper_shield_ld2_disabled_rate: float = 0.0
    upper_shield_all_disabled_rate: float = 0.0
    upper_shield_ld1_locked_rate: float = 0.0
    upper_shield_ld2_locked_rate: float = 0.0
    current_requested0_mean: float = 0.0
    current_requested1_mean: float = 0.0
    current_requested2_mean: float = 0.0
    current_exec0_mean: float = 0.0
    current_exec1_mean: float = 0.0
    current_exec2_mean: float = 0.0
    thermal_cap_current0_mean: float = 0.0
    thermal_cap_current1_mean: float = 0.0
    thermal_cap_current2_mean: float = 0.0
    predicted_headroom0_mean: float = 0.0
    predicted_headroom1_mean: float = 0.0
    predicted_headroom2_mean: float = 0.0
    compression_ratio0_mean: float = 0.0
    compression_ratio1_mean: float = 0.0
    compression_ratio2_mean: float = 0.0
    qos_recovered_current_mean: float = 0.0
    execution_guard_applied_rate: float = 0.0
    execution_guard_downgrade_rate: float = 0.0
    execution_guard_anchor_clamp_rate: float = 0.0
    execution_guard_downgrade_ld1_rate: float = 0.0
    execution_guard_downgrade_ld2_rate: float = 0.0


class MetaTrainer:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        seed = int(cfg["experiment"]["seed"])
        set_seed(seed)

        requested_device = str(cfg["experiment"].get("device", "auto"))
        self.device = resolve_device(requested_device)
        self.cfg.setdefault("experiment", {})
        self.cfg["experiment"]["device_requested"] = requested_device
        self.cfg["experiment"]["device_resolved"] = str(self.device)
        self.agent = HierarchicalAgent(cfg, self.device)
        self.base_sampler_cfg = copy.deepcopy(cfg["sampler"])
        self.task_sampler = TaskSampler(
            copy.deepcopy(self.base_sampler_cfg),
            seed=seed,
            task_defaults=task_defaults_from_cfg(cfg),
        )

        curriculum_cfg = cfg.get("meta", {}).get("curriculum", {})
        self.curriculum_enabled = bool(curriculum_cfg.get("enabled", False))
        self.curriculum_phases = self._normalize_curriculum_phases(curriculum_cfg.get("phases", []))

        meta_cfg = cfg.get("meta", {})
        self.dual = DualLayer.from_meta_cfg(meta_cfg, n_tx=int(cfg["env"]["n_tx"]))
        self.dual_enabled = bool(meta_cfg.get("dual_enabled", True))
        self.explicit_inner_outer = bool(meta_cfg.get("explicit_inner_outer", True))
        self.outer_step_size = float(meta_cfg.get("outer_step_size", 0.15))
        self.reset_optimizer_after_outer_update = bool(meta_cfg.get("reset_optimizer_after_outer_update", False))
        self.query_updates_enabled = bool(meta_cfg.get("query_updates_enabled", True))
        self.query_context_updates_enabled = bool(meta_cfg.get("query_context_updates_enabled", True))
        support_gate_cfg = meta_cfg.get("support_gate", {})
        if not isinstance(support_gate_cfg, dict):
            support_gate_cfg = {}
        self.support_gate_cfg = copy.deepcopy(support_gate_cfg)
        self.support_gate_enabled = bool(support_gate_cfg.get("enabled", False))
        self.support_gate_role = str(support_gate_cfg.get("role", "rollback_guard"))
        self.support_gate_rule = str(support_gate_cfg.get("rule", "support_score_non_degradation"))
        self.support_gate_paired_validation = bool(support_gate_cfg.get("paired_validation", True))
        support_eps = int(meta_cfg.get("support_episodes", 0))
        default_adapt_eps = min(3, max(0, support_eps))
        self.support_adaptation_episodes = int(meta_cfg.get("support_adaptation_episodes", default_adapt_eps))
        self.support_adaptation_episodes = int(np.clip(self.support_adaptation_episodes, 0, max(0, support_eps)))
        default_validation_eps = max(0, support_eps - self.support_adaptation_episodes)
        self.support_gate_validation_episodes = int(
            meta_cfg.get("support_gate_validation_episodes", default_validation_eps)
        )
        self.support_gate_validation_episodes = int(
            np.clip(self.support_gate_validation_episodes, 0, max(0, support_eps - self.support_adaptation_episodes))
        )
        self.inner_warmup_steps = int(meta_cfg.get("inner_warmup_steps", max(self.agent.batch_size, 64)))
        self.inner_upper_warmup_steps = int(
            meta_cfg.get("inner_upper_warmup_steps", max(self.agent.batch_size // 2, 32))
        )
        if self.explicit_inner_outer:
            self.agent.warmup_steps = self.inner_warmup_steps
            self.agent.upper_warmup_steps = self.inner_upper_warmup_steps

        log_dir = cfg["experiment"]["log_dir"]
        run_name = cfg["experiment"]["run_name"]
        self.logger = Logger(log_dir=log_dir, run_name=run_name)

        self.ckpt_dir = Path(log_dir) / run_name / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalize_curriculum_phases(phases_raw) -> List[Dict]:
        phases: List[Dict] = []
        if not isinstance(phases_raw, list):
            return phases
        for idx, p in enumerate(phases_raw):
            if not isinstance(p, dict):
                continue
            sampler = p.get("sampler", {})
            if not isinstance(sampler, dict):
                continue
            until_frac = float(p.get("until_frac", 1.0))
            until_frac = float(np.clip(until_frac, 0.0, 1.0))
            phases.append(
                {
                    "name": str(p.get("name", f"phase_{idx+1}")),
                    "until_frac": until_frac,
                    "sampler": copy.deepcopy(sampler),
                }
            )
        phases.sort(key=lambda x: x["until_frac"])
        return phases

    def _sampler_cfg_for_iter(self, it: int, meta_iters: int) -> tuple[str, Dict]:
        if not self.curriculum_enabled or not self.curriculum_phases:
            return "base", copy.deepcopy(self.base_sampler_cfg)

        frac = float(it) / float(max(meta_iters, 1))
        picked = self.curriculum_phases[-1]
        for phase in self.curriculum_phases:
            if frac <= float(phase["until_frac"]):
                picked = phase
                break

        if bool(self.base_sampler_cfg.get("strict_site_bank", False)) and "site_bank" in picked["sampler"]:
            raise ValueError("strict_site_bank=true does not allow curriculum phases to override site_bank")
        merged = copy.deepcopy(self.base_sampler_cfg)
        merged.update(copy.deepcopy(picked["sampler"]))
        return str(picked["name"]), merged

    @staticmethod
    def _state_path_delta_norm(reference: Dict, current: Dict, paths: List[tuple[str, ...]]) -> float:
        total = 0.0

        def get_path(obj: Dict, path: tuple[str, ...]):
            out = obj
            for key in path:
                if not isinstance(out, dict) or key not in out:
                    return None
                out = out[key]
            return out

        def accumulate(ref_obj, cur_obj) -> None:
            nonlocal total
            if isinstance(ref_obj, dict) and isinstance(cur_obj, dict):
                for key in ref_obj.keys() & cur_obj.keys():
                    accumulate(ref_obj[key], cur_obj[key])
            elif torch.is_tensor(ref_obj) and torch.is_tensor(cur_obj):
                if torch.is_floating_point(ref_obj) and torch.is_floating_point(cur_obj):
                    diff = cur_obj.detach().float().cpu() - ref_obj.detach().float().cpu()
                    total += float(torch.sum(diff * diff).item())

        for path in paths:
            ref_obj = get_path(reference, path)
            cur_obj = get_path(current, path)
            if ref_obj is not None and cur_obj is not None:
                accumulate(ref_obj, cur_obj)
        return float(math.sqrt(max(total, 0.0)))

    @staticmethod
    def _numeric_state_delta_norm(reference, current) -> float:
        total = 0.0

        def mark_inf() -> None:
            nonlocal total
            total = float("inf")

        def accumulate(ref_obj, cur_obj) -> None:
            nonlocal total
            if math.isinf(total):
                return
            if ref_obj is None or cur_obj is None:
                if ref_obj is not None or cur_obj is not None:
                    mark_inf()
                return
            if isinstance(ref_obj, dict) and isinstance(cur_obj, dict):
                if set(ref_obj.keys()) != set(cur_obj.keys()):
                    mark_inf()
                    return
                for key in ref_obj.keys():
                    accumulate(ref_obj[key], cur_obj[key])
                return
            if isinstance(ref_obj, (list, tuple)) and isinstance(cur_obj, (list, tuple)):
                if len(ref_obj) != len(cur_obj):
                    mark_inf()
                    return
                for ref_item, cur_item in zip(ref_obj, cur_obj):
                    accumulate(ref_item, cur_item)
                return
            if torch.is_tensor(ref_obj) and torch.is_tensor(cur_obj):
                if tuple(ref_obj.shape) != tuple(cur_obj.shape):
                    mark_inf()
                    return
                ref = ref_obj.detach().float().cpu().numpy()
                cur = cur_obj.detach().float().cpu().numpy()
            elif isinstance(ref_obj, np.ndarray) and isinstance(cur_obj, np.ndarray):
                if ref_obj.shape != cur_obj.shape:
                    mark_inf()
                    return
                if not (np.issubdtype(ref_obj.dtype, np.number) or np.issubdtype(ref_obj.dtype, np.bool_)):
                    return
                ref = ref_obj.astype(np.float64, copy=False)
                cur = cur_obj.astype(np.float64, copy=False)
            elif isinstance(ref_obj, (int, float, bool, np.number)) and isinstance(
                cur_obj, (int, float, bool, np.number)
            ):
                ref = np.asarray([ref_obj], dtype=np.float64)
                cur = np.asarray([cur_obj], dtype=np.float64)
            else:
                if ref_obj != cur_obj:
                    return
                return

            ref_nan = np.isnan(ref)
            cur_nan = np.isnan(cur)
            if np.any(ref_nan != cur_nan):
                mark_inf()
                return
            diff = np.where(ref_nan & cur_nan, 0.0, cur - ref)
            total += float(np.sum(diff * diff))

        accumulate(reference, current)
        return float(total if math.isinf(total) else math.sqrt(max(total, 0.0)))

    @classmethod
    def _state_paths_numeric_delta_norm(cls, reference: Dict, current: Dict, paths: List[tuple[str, ...]]) -> float:
        total = 0.0

        def get_path(obj: Dict, path: tuple[str, ...]):
            out = obj
            for key in path:
                if not isinstance(out, dict) or key not in out:
                    return None
                out = out[key]
            return out

        for path in paths:
            ref_obj = get_path(reference, path)
            cur_obj = get_path(current, path)
            if ref_obj is None and cur_obj is None:
                continue
            delta = cls._numeric_state_delta_norm(ref_obj, cur_obj)
            if math.isinf(delta):
                return float("inf")
            total += float(delta * delta)
        return float(math.sqrt(max(total, 0.0)))

    @classmethod
    def _trainable_parameter_delta_norm(cls, reference: Dict, current: Dict) -> float:
        return cls._state_path_delta_norm(
            reference,
            current,
            paths=[
                ("upper", "q"),
                ("lower", "actor"),
                ("lower", "q1"),
                ("lower", "q2"),
                ("context_encoder",),
                ("context_predictor",),
            ],
        )

    @classmethod
    def _target_parameter_delta_norm(cls, reference: Dict, current: Dict) -> float:
        return cls._state_path_delta_norm(
            reference,
            current,
            paths=[
                ("upper", "q_tgt"),
                ("upper", "q_tgt_phys"),
                ("lower", "q1_tgt"),
                ("lower", "q2_tgt"),
                ("lower", "constraint_q_tgt"),
                ("lower", "q1_tgt_phys"),
                ("lower", "q2_tgt_phys"),
                ("lower", "constraint_tgt_phys"),
            ],
        )

    @classmethod
    def _optimizer_state_delta_norm(cls, reference: Dict, current: Dict) -> float:
        return cls._state_paths_numeric_delta_norm(
            reference,
            current,
            paths=[
                ("upper", "optim"),
                ("lower", "actor_optim"),
                ("lower", "critic_optim"),
                ("lower", "constraint_optim"),
                ("lower", "alpha_optim"),
                ("context_optim",),
            ],
        )

    @staticmethod
    def _dual_delta_norm(reference: Dict, current: Dict) -> float:
        ref = np.asarray(reference.get("values", []), dtype=np.float32).reshape(-1)
        cur = np.asarray(current.get("values", []), dtype=np.float32).reshape(-1)
        if ref.shape != cur.shape:
            return float("inf")
        diff = cur - ref
        return float(np.sqrt(float(np.sum(diff * diff))))

    @staticmethod
    def _safety_state_delta_norm(reference: Dict, current: Dict) -> float:
        ref_est = (reference or {}).get("thermal_estimator", {})
        cur_est = (current or {}).get("thermal_estimator", {})
        keys = ["gain_mean", "gain_var", "valid_count", "temperature_slope", "last_headroom"]
        total = 0.0
        for key in keys:
            ref = np.asarray(ref_est.get(key, []), dtype=np.float32).reshape(-1)
            cur = np.asarray(cur_est.get(key, []), dtype=np.float32).reshape(-1)
            if ref.shape != cur.shape:
                return float("inf")
            ref_nan = np.isnan(ref)
            cur_nan = np.isnan(cur)
            if np.any(ref_nan != cur_nan):
                return float("inf")
            diff = np.where(ref_nan & cur_nan, 0.0, cur - ref)
            total += float(np.sum(diff * diff))
        ref_locked = np.asarray((reference or {}).get("upper_shield_hot_locked", []), dtype=np.float32).reshape(-1)
        cur_locked = np.asarray((current or {}).get("upper_shield_hot_locked", []), dtype=np.float32).reshape(-1)
        if ref_locked.shape != cur_locked.shape:
            return float("inf")
        lock_diff = cur_locked - ref_locked
        total += float(np.sum(lock_diff * lock_diff))
        return float(np.sqrt(total))

    @staticmethod
    def _mean_gate_stats(stats: List[EpisodeStats]) -> SupportGateStats:
        if not stats:
            return SupportGateStats()
        return SupportGateStats(
            reward=float(np.mean([s.reward_task for s in stats])),
            cost=float(np.mean([s.cost for s in stats])),
            violation_rate=float(np.mean([s.violations for s in stats])),
        )

    def _support_gate_before_after(self, stats: List[EpisodeStats]) -> tuple[SupportGateStats, SupportGateStats]:
        if not stats:
            empty = SupportGateStats()
            return empty, empty
        if len(stats) == 1:
            one = self._mean_gate_stats(stats)
            return one, one
        if self.support_gate_validation_episodes > 0 and len(stats) > self.support_adaptation_episodes:
            split = int(np.clip(self.support_adaptation_episodes, 1, len(stats) - 1))
        else:
            split = max(1, len(stats) // 2)
        return self._mean_gate_stats(stats[:split]), self._mean_gate_stats(stats[split:])

    def _support_gate_active(self) -> bool:
        return bool(self.support_gate_enabled and self.explicit_inner_outer)

    def _evaluate_support_gate(
        self,
        *,
        support_stats: List[EpisodeStats],
        parameter_delta: float,
        pre_validation_stats: List[EpisodeStats] | None = None,
        post_validation_stats: List[EpisodeStats] | None = None,
    ) -> SupportGateDecision:
        if pre_validation_stats is not None and post_validation_stats is not None:
            before = self._mean_gate_stats(pre_validation_stats)
            after = self._mean_gate_stats(post_validation_stats)
        else:
            before, after = self._support_gate_before_after(support_stats)
        return evaluate_support_gate(
            pre_support_stats=before,
            post_support_stats=after,
            parameter_delta=float(parameter_delta),
            config=self.support_gate_cfg,
        )

    def _support_gate_validation_seed(self, iteration: int, task_idx: int, validation_idx: int) -> int:
        base_seed = int(self.cfg.get("experiment", {}).get("seed", 0))
        seed = (
            int(base_seed) * 1_000_003
            + int(iteration) * 10_007
            + int(task_idx) * 1_009
            + int(validation_idx) * 37
            + 17
        )
        return int(seed % (2**32 - 1))

    @staticmethod
    def _mutable_state_with_train_parameters(mutable_state: Dict, train_state: Dict) -> Dict:
        """Use mutable support context/state with an alternate trainable snapshot."""

        out = copy.deepcopy(mutable_state)
        for key in (
            "upper",
            "lower",
            "context_encoder",
            "context_predictor",
            "context_optim",
        ):
            if key in train_state:
                out[key] = copy.deepcopy(train_state[key])
        return out

    def _run_gate_validation_pair(
        self,
        *,
        task,
        iteration: int,
        task_idx: int,
        candidate_state: Dict,
        candidate_dual_state: Dict,
        pre_state: Dict,
        pre_dual_state: Dict,
    ) -> tuple[List[EpisodeStats], List[EpisodeStats], List[int], float]:
        seeds = [
            self._support_gate_validation_seed(iteration, task_idx, val_idx)
            for val_idx in range(int(self.support_gate_validation_episodes))
        ]
        if not seeds:
            return [], [], [], 0.0

        validation_start = time.perf_counter()
        pre_validation_state = self._mutable_state_with_train_parameters(candidate_state, pre_state)
        pre_stats: List[EpisodeStats] = []
        for seed in seeds:
            self.agent.restore_mutable_state(pre_validation_state)
            self.dual.load_state_dict(copy.deepcopy(pre_dual_state))
            val_env = MultiTxUwSliptEnv(self.cfg, overrides=task.to_env_overrides())
            pre_stats.append(
                self._run_episode(
                    val_env,
                    train=False,
                    clear_context=False,
                    reset_seed=seed,
                    update_context=False,
                )
            )

        post_stats: List[EpisodeStats] = []
        for seed in seeds:
            self.agent.restore_mutable_state(candidate_state)
            self.dual.load_state_dict(copy.deepcopy(candidate_dual_state))
            val_env = MultiTxUwSliptEnv(self.cfg, overrides=task.to_env_overrides())
            post_stats.append(
                self._run_episode(
                    val_env,
                    train=False,
                    clear_context=False,
                    reset_seed=seed,
                    update_context=False,
                )
            )

        self.agent.restore_mutable_state(candidate_state)
        self.dual.load_state_dict(copy.deepcopy(candidate_dual_state))
        validation_latency_ms = float((time.perf_counter() - validation_start) * 1000.0)
        return pre_stats, post_stats, seeds, validation_latency_ms

    def _run_episode(
        self,
        env: MultiTxUwSliptEnv,
        train: bool,
        clear_context: bool = True,
        reset_seed: int | None = None,
        update_context: bool | None = None,
    ) -> EpisodeStats:
        obs, _ = env.reset(seed=reset_seed)
        self.agent.reset_rollout_state(clear_context=clear_context)
        if update_context is None:
            update_context = bool((not train) and self.query_context_updates_enabled)

        ep_reward = 0.0
        ep_reward_task = 0.0
        ep_reward_benchmark = 0.0
        ep_reward_dual_penalized = 0.0
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
        ep_eh_input_eff = 0.0
        ep_eh_raw_nonlinear = 0.0
        ep_eh_sat = 0.0
        ep_eh_near_zero = 0.0
        ep_planner_active = 0.0
        ep_planner_replaced = 0.0
        ep_planner_fallback = 0.0
        ep_planner_score_improvement = 0.0
        ep_planner_best_score_improvement = 0.0
        ep_planner_candidate_distance = 0.0
        ep_planner_disagreement = 0.0
        ep_planner_budget = 0.0
        ep_planner_budget_k0 = 0.0
        ep_planner_budget_k8 = 0.0
        ep_planner_budget_k16 = 0.0
        ep_planner_budget_k24 = 0.0
        ep_cost_qos = 0.0
        ep_cost_temp = np.zeros(3, dtype=np.float32)
        ep_temp_max = np.full(3, -np.inf, dtype=np.float32)
        ep_headroom_min = float("inf")
        ep_mode_counts = np.zeros(3, dtype=np.float32)
        ep_boost_counts = np.zeros(4, dtype=np.float32)
        ep_projected_current_total = 0.0
        ep_bus_utilization = 0.0
        ep_projection_residual = 0.0
        ep_thermal_gain_mean = np.zeros(3, dtype=np.float32)
        ep_thermal_gain_std = np.zeros(3, dtype=np.float32)
        ep_thermal_gain_safe_scale = np.zeros(3, dtype=np.float32)
        ep_thermal_valid_count = np.zeros(3, dtype=np.float32)
        ep_planner_h2_veto = 0.0
        ep_planner_constraint_rejection = 0.0
        ep_planner_h1_risk = 0.0
        ep_planner_h2_risk = 0.0
        ep_planner_h2_max_temp = 0.0
        ep_planner_target_constraint = 0.0
        ep_upper_shield_applied = 0.0
        ep_upper_shield_ld1_disabled = 0.0
        ep_upper_shield_ld2_disabled = 0.0
        ep_upper_shield_all_disabled = 0.0
        ep_upper_shield_ld1_locked = 0.0
        ep_upper_shield_ld2_locked = 0.0
        ep_current_requested = np.zeros(3, dtype=np.float32)
        ep_current_exec = np.zeros(3, dtype=np.float32)
        ep_thermal_cap_current = np.zeros(3, dtype=np.float32)
        ep_predicted_headroom = np.zeros(3, dtype=np.float32)
        ep_compression_ratio = np.zeros(3, dtype=np.float32)
        ep_qos_recovered_current = 0.0
        ep_execution_guard_applied = 0.0
        ep_execution_guard_downgrade = 0.0
        ep_execution_guard_anchor_clamp = 0.0
        ep_execution_guard_downgrade_ld1 = 0.0
        ep_execution_guard_downgrade_ld2 = 0.0

        macro_start_obs = None
        macro_start_z = None
        macro_start_physical = None
        macro_upper_idx_raw = 0.0
        macro_upper_idx_exec = 0.0
        macro_reward = 0.0
        macro_steps = 0

        done = False
        while not done:
            z = self.agent.infer_z()
            temps_before = env.temps.copy().astype(np.float32)

            action, aux = self.agent.act(
                obs=obs,
                temps=temps_before,
                amb_temp=env.amb_temp,
                gamma=env.gamma,
                delta=env.delta,
                z=z,
                eval_mode=not train,
            )

            next_obs, reward, terminated, truncated, info = env.step(action)
            thermal_estimator_diag = self.agent.update_safety_estimator(
                temps_before=temps_before,
                info=info,
            )
            physical_features_next = self.agent.current_physical_features(
                temps=np.asarray(info.get("temps", temps_before), dtype=np.float32)
            )
            done = bool(terminated or truncated)
            ep_len += 1

            cost = float(info["cost"])
            cost_vec = np.asarray(info.get("cost_vec", [cost]), dtype=np.float32).reshape(-1)
            dual_penalty = self.dual.penalty(cost_vec) if self.dual_enabled else 0.0
            reward_benchmark = float(info.get("reward_benchmark", reward))
            reward_task = float(info.get("reward_task", reward_benchmark + float(info.get("reward_cost_penalty", 0.0))))
            reward_dual_penalized = float(reward_task - dual_penalty)

            lower_transition = {
                "obs": obs.astype(np.float32),
                "next_obs": next_obs.astype(np.float32),
                "upper_idx_raw": float(aux["upper_idx_raw"]),
                "upper_idx_exec": float(aux["upper_idx_exec"]),
                "reward": reward_dual_penalized,
                "reward_raw": reward_benchmark,
                "reward_task": reward_task,
                "reward_benchmark": reward_benchmark,
                "reward_dual_penalized": reward_dual_penalized,
                "done": float(done),
                "z": z.astype(np.float32),
                "act_exec": aux["act_exec"].astype(np.float32),
                "act_raw": aux["act_raw"].astype(np.float32),
                "act_refined_raw": aux.get("act_refined_raw", aux["act_raw"]).astype(np.float32),
                "act_policy_raw": aux.get("act_policy_raw", aux["act_raw"]).astype(np.float32),
                "policy_action_raw": aux.get("policy_action_raw", aux.get("act_policy_raw", aux["act_raw"])).astype(np.float32),
                "planner_action_raw": aux.get("planner_action_raw", aux.get("act_refined_raw", aux["act_raw"])).astype(np.float32),
                "executed_action": aux.get("executed_action", aux["act_exec"]).astype(np.float32),
                "planner_selected": float(bool(aux.get("planner_selected", False))),
                "act_desired": aux.get("act_desired", aux["act_exec"]).astype(np.float32),
                "projection_residual": aux.get("projection_residual", np.zeros(5, dtype=np.float32)).astype(np.float32),
                "upper_action_mask": aux.get(
                    "upper_action_mask",
                    np.ones(int(self.cfg["agent"]["n_upper_actions"]), dtype=np.float32),
                ).astype(np.float32),
                "residual_planner_score_improvement": float(aux.get("residual_planner_score_improvement", 0.0)),
                "physical_features": aux.get("physical_features", np.zeros(18, dtype=np.float32)).astype(np.float32),
                "physical_features_next": physical_features_next.astype(np.float32),
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
                "bus_utilization": float(info.get("bus_utilization", 0.0)),
                "projected_current_total": float(
                    aux.get("projected_current_total", info.get("current_total", 0.0))
                ),
                "qos_margin": float(info.get("qos_rate", 0.0) - env.qos_min_rate),
                "burst_active": float(info.get("burst_mean", 0.0) > 1.0e-8),
                "thermal_gain_mean": np.asarray(
                    thermal_estimator_diag.get("thermal_gain_mean", np.ones(3, dtype=np.float32)),
                    dtype=np.float32,
                ),
                "thermal_gain_std": np.asarray(
                    thermal_estimator_diag.get("thermal_gain_std", np.zeros(3, dtype=np.float32)),
                    dtype=np.float32,
                ),
                "thermal_gain_safe_scale": np.asarray(
                    thermal_estimator_diag.get("thermal_gain_safe_scale", np.ones(3, dtype=np.float32)),
                    dtype=np.float32,
                ),
                "temperature_slope": np.asarray(
                    thermal_estimator_diag.get("temperature_slope", np.zeros(3, dtype=np.float32)),
                    dtype=np.float32,
                ),
                "thermal_headroom": np.asarray(
                    thermal_estimator_diag.get("thermal_headroom", np.zeros(3, dtype=np.float32)),
                    dtype=np.float32,
                ),
            }

            if bool(aux.get("macro_new", False)) or macro_start_obs is None:
                macro_start_obs = obs.astype(np.float32)
                macro_start_z = z.astype(np.float32)
                macro_start_physical = aux.get("physical_features", np.zeros(18, dtype=np.float32)).astype(np.float32)
                macro_upper_idx_raw = float(aux["upper_idx_raw"])
                macro_upper_idx_exec = float(aux["upper_idx_exec"])
                macro_reward = 0.0
                macro_steps = 0

            macro_reward += (float(self.agent.upper.gamma) ** int(macro_steps)) * reward_dual_penalized
            macro_steps += 1
            macro_done = bool(done)
            macro_end = macro_done or (int(aux.get("hold_left", 0)) <= 0)

            if train:
                next_macro_info: Dict[str, float] = {}

                def _next_macro(z_next_val: np.ndarray) -> Dict[str, float]:
                    nonlocal next_macro_info
                    if done:
                        payload = {
                            "upper_idx_raw_next": float(aux["upper_idx_raw"]),
                            "upper_idx_exec_next": float(aux["upper_idx_exec"]),
                            "boost_combo_exec_next": float(aux["boost_combo_exec"]),
                            "mode_exec_next": float(aux["mode_exec"]),
                            "next_exec_map": np.arange(int(self.cfg["agent"]["n_upper_actions"]), dtype=np.float32),
                            "next_action_mask": np.ones(int(self.cfg["agent"]["n_upper_actions"]), dtype=np.float32),
                        }
                    else:
                        nxt = self.agent.preview_next_macro(
                            next_obs=next_obs.astype(np.float32),
                            z_next=z_next_val.astype(np.float32),
                            physical_features_next=physical_features_next.astype(np.float32),
                            temps_next=np.asarray(info["temps"], dtype=np.float32),
                            eval_mode=False,
                            commit_plan=True,
                        )
                        payload = {
                            "upper_idx_raw_next": float(nxt["upper_idx_raw_next"]),
                            "upper_idx_exec_next": float(nxt["upper_idx_exec_next"]),
                            "boost_combo_exec_next": float(nxt["boost_combo_exec_next"]),
                            "mode_exec_next": float(nxt["mode_exec_next"]),
                            "next_exec_map": np.asarray(nxt["next_exec_map"], dtype=np.float32),
                            "next_action_mask": np.asarray(nxt["next_action_mask"], dtype=np.float32),
                        }
                    next_macro_info = payload
                    return payload

                z_next = self.agent.observe_lower(lower_transition, next_macro_fn=_next_macro)
                if macro_end and macro_start_obs is not None and macro_start_z is not None:
                    upper_transition = {
                        "obs": macro_start_obs,
                        "next_obs": next_obs.astype(np.float32),
                        "physical_features": (
                            macro_start_physical
                            if macro_start_physical is not None
                            else np.zeros(18, dtype=np.float32)
                        ),
                        "physical_features_next": physical_features_next.astype(np.float32),
                        "upper_idx_raw": macro_upper_idx_raw,
                        "upper_idx_exec": macro_upper_idx_exec,
                        "reward": float(macro_reward),
                        "done": float(macro_done),
                        "z": macro_start_z,
                        "z_next": z_next.astype(np.float32),
                        "horizon": float(macro_steps),
                        "next_exec_map": np.asarray(
                            next_macro_info.get(
                                "next_exec_map",
                                np.arange(int(self.cfg["agent"]["n_upper_actions"]), dtype=np.float32),
                            ),
                            dtype=np.float32,
                        ),
                        "next_action_mask": np.asarray(
                            next_macro_info.get(
                                "next_action_mask",
                                np.ones(int(self.cfg["agent"]["n_upper_actions"]), dtype=np.float32),
                            ),
                            dtype=np.float32,
                        ),
                    }
                    self.agent.observe_upper(upper_transition)
                self.agent.learn()
            elif update_context:
                self.agent.episode.add(
                    {
                        "obs": lower_transition["obs"],
                        "upper_idx_exec": lower_transition["upper_idx_exec"],
                        "boost_combo_exec": lower_transition["boost_combo_exec"],
                        "mode_exec": lower_transition["mode_exec"],
                        "act_exec": lower_transition["act_exec"],
                        "reward": lower_transition["reward"],
                        "reward_raw": lower_transition["reward_raw"],
                        "reward_task": lower_transition["reward_task"],
                        "reward_benchmark": lower_transition["reward_benchmark"],
                        "reward_dual_penalized": lower_transition["reward_dual_penalized"],
                        "cost": lower_transition["cost"],
                        "cost_vec": lower_transition["cost_vec"],
                        "task_params": np.asarray(
                            build_context_task_summary_v2(lower_transition),
                            dtype=np.float32,
                        ),
                    }
                )

            ep_reward += reward_benchmark
            ep_reward_task += reward_task
            ep_reward_benchmark += reward_benchmark
            ep_reward_dual_penalized += reward_dual_penalized
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
            ep_eh_input_eff += float(info.get("eh_input_eff", info.get("eh_metric", 0.0)))
            ep_eh_raw_nonlinear += float(info.get("eh_metric_raw_nonlinear", info.get("eh_metric", 0.0)))
            ep_eh_sat += float(info.get("eh_saturation_fraction", 0.0))
            ep_eh_near_zero += float(info.get("eh_near_zero_fraction", 0.0))
            ep_planner_active += float(bool(aux.get("residual_planner_enabled", False)))
            ep_planner_replaced += float(bool(aux.get("residual_planner_replaced_policy", False)))
            ep_planner_fallback += float(bool(aux.get("residual_planner_fallback_to_policy", False)))
            ep_planner_score_improvement += float(aux.get("residual_planner_score_improvement", 0.0))
            ep_planner_best_score_improvement += float(aux.get("residual_planner_best_score_improvement", 0.0))
            ep_planner_candidate_distance += float(aux.get("residual_planner_candidate_distance", 0.0))
            ep_planner_disagreement += float(aux.get("residual_planner_disagreement", 0.0))
            planner_budget = int(aux.get("residual_planner_budget", aux.get("residual_planner_candidate_count", 0)))
            ep_planner_budget += float(planner_budget)
            ep_planner_budget_k0 += float(planner_budget == 0)
            ep_planner_budget_k8 += float(planner_budget == 8)
            ep_planner_budget_k16 += float(planner_budget == 16)
            ep_planner_budget_k24 += float(planner_budget == 24)
            ep_cost_qos += float(info.get("cost_qos", cost_vec[0] if cost_vec.size > 0 else 0.0))
            ep_cost_temp[0] += float(info.get("cost_temp_anchor", cost_vec[1] if cost_vec.size > 1 else 0.0))
            ep_cost_temp[1] += float(info.get("cost_temp_boost1", cost_vec[2] if cost_vec.size > 2 else 0.0))
            ep_cost_temp[2] += float(info.get("cost_temp_boost2", cost_vec[3] if cost_vec.size > 3 else 0.0))
            temps_after = np.asarray(info.get("temps", temps_before), dtype=np.float32).reshape(-1)[:3]
            if temps_after.size == 3:
                ep_temp_max = np.maximum(ep_temp_max, temps_after)
                ep_headroom_min = min(ep_headroom_min, float(np.min(float(env.thermal_safe) - temps_after)))
            mode_i = int(np.clip(int(aux.get("mode_exec", 0)), 0, 2))
            boost_i = int(np.clip(int(aux.get("boost_combo_exec", 0)), 0, 3))
            ep_mode_counts[mode_i] += 1.0
            ep_boost_counts[boost_i] += 1.0
            ep_projected_current_total += float(aux.get("projected_current_total", info.get("current_total", 0.0)))
            ep_bus_utilization += float(info.get("bus_utilization", 0.0))
            ep_projection_residual += float(
                np.linalg.norm(np.asarray(aux.get("projection_residual", np.zeros(5, dtype=np.float32)), dtype=np.float32))
            )
            gain_mean = np.asarray(
                thermal_estimator_diag.get("thermal_gain_mean", np.zeros(3, dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)[:3]
            gain_std = np.asarray(
                thermal_estimator_diag.get("thermal_gain_std", np.zeros(3, dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)[:3]
            gain_scale = np.asarray(
                thermal_estimator_diag.get("thermal_gain_safe_scale", np.zeros(3, dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)[:3]
            valid_count = np.asarray(
                thermal_estimator_diag.get("thermal_gain_valid_count", np.zeros(3, dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)[:3]
            if gain_mean.size == 3:
                ep_thermal_gain_mean += gain_mean
            if gain_std.size == 3:
                ep_thermal_gain_std += gain_std
            if gain_scale.size == 3:
                ep_thermal_gain_safe_scale += gain_scale
            if valid_count.size == 3:
                ep_thermal_valid_count += valid_count
            ep_planner_h2_veto += float(bool(aux.get("residual_planner_h2_veto", False)))
            ep_planner_constraint_rejection += float(aux.get("residual_planner_constraint_rejection_rate", 0.0))
            ep_planner_h1_risk += float(aux.get("residual_planner_h1_thermal_risk", 0.0))
            ep_planner_h2_risk += float(aux.get("residual_planner_h2_thermal_risk", 0.0))
            ep_planner_h2_max_temp += float(aux.get("residual_planner_h2_max_temperature", 0.0))
            ep_planner_target_constraint += float(aux.get("residual_planner_target_constraint_value", 0.0))
            ep_upper_shield_applied += float(bool(aux.get("upper_shield_applied", False)))
            ep_upper_shield_ld1_disabled += float(float(aux.get("upper_shield_allowed_ld1", 1.0)) < 0.5)
            ep_upper_shield_ld2_disabled += float(float(aux.get("upper_shield_allowed_ld2", 1.0)) < 0.5)
            ep_upper_shield_all_disabled += float(float(aux.get("upper_shield_allowed_all", 1.0)) < 0.5)
            ep_upper_shield_ld1_locked += float(float(aux.get("upper_shield_locked_ld1", 0.0)) > 0.5)
            ep_upper_shield_ld2_locked += float(float(aux.get("upper_shield_locked_ld2", 0.0)) > 0.5)
            current_requested = np.asarray(aux.get("current_requested", np.zeros(3, dtype=np.float32)), dtype=np.float32).reshape(-1)[:3]
            current_exec = np.asarray(aux.get("currents_exec", aux.get("act_exec", np.zeros(5, dtype=np.float32))[:3]), dtype=np.float32).reshape(-1)[:3]
            thermal_cap_current = np.asarray(aux.get("thermal_cap_current", np.zeros(3, dtype=np.float32)), dtype=np.float32).reshape(-1)[:3]
            predicted_headroom = np.asarray(aux.get("thermal_pred_margin", np.zeros(3, dtype=np.float32)), dtype=np.float32).reshape(-1)[:3]
            compression_ratio = np.asarray(
                aux.get("projection_compression_ratio_per_source", np.zeros(3, dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1)[:3]
            if current_requested.size == 3:
                ep_current_requested += current_requested
            if current_exec.size == 3:
                ep_current_exec += current_exec
            if thermal_cap_current.size == 3:
                ep_thermal_cap_current += thermal_cap_current
            if predicted_headroom.size == 3:
                ep_predicted_headroom += predicted_headroom
            if compression_ratio.size == 3:
                ep_compression_ratio += compression_ratio
            ep_qos_recovered_current += float(aux.get("qos_recovered_current", 0.0))
            ep_execution_guard_applied += float(bool(aux.get("execution_guard_applied", False)))
            ep_execution_guard_downgrade += float(bool(aux.get("execution_guard_downgrade_applied", False)))
            ep_execution_guard_anchor_clamp += float(bool(aux.get("execution_guard_anchor_clamp_applied", False)))
            ep_execution_guard_downgrade_ld1 += float(float(aux.get("execution_guard_downgrade_ld1", 0.0)) > 0.5)
            ep_execution_guard_downgrade_ld2 += float(float(aux.get("execution_guard_downgrade_ld2", 0.0)) > 0.5)

            obs = next_obs

        ep_len_safe = max(ep_len, 1)
        ep_temp_max = np.where(np.isfinite(ep_temp_max), ep_temp_max, 0.0).astype(np.float32)
        headroom_min = float(ep_headroom_min if math.isfinite(ep_headroom_min) else 0.0)
        return EpisodeStats(
            reward=ep_reward,
            reward_task=ep_reward_task,
            reward_benchmark=ep_reward_benchmark,
            reward_dual_penalized=ep_reward_dual_penalized,
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
            eh_input_eff=ep_eh_input_eff / max(ep_len, 1),
            eh_metric_raw_nonlinear=ep_eh_raw_nonlinear / max(ep_len, 1),
            eh_saturation_fraction=ep_eh_sat / max(ep_len, 1),
            eh_near_zero_fraction=ep_eh_near_zero / max(ep_len, 1),
            residual_planner_active_rate=ep_planner_active / max(ep_len, 1),
            residual_planner_replacement_rate=ep_planner_replaced / max(ep_len, 1),
            residual_planner_fallback_rate=ep_planner_fallback / max(ep_len, 1),
            residual_planner_score_improvement=ep_planner_score_improvement / max(ep_len, 1),
            residual_planner_best_score_improvement=ep_planner_best_score_improvement / max(ep_len, 1),
            residual_planner_candidate_distance=ep_planner_candidate_distance / max(ep_len, 1),
            residual_planner_disagreement=ep_planner_disagreement / max(ep_len, 1),
            residual_planner_budget_mean=ep_planner_budget / max(ep_len, 1),
            residual_planner_budget_k0_rate=ep_planner_budget_k0 / max(ep_len, 1),
            residual_planner_budget_k8_rate=ep_planner_budget_k8 / max(ep_len, 1),
            residual_planner_budget_k16_rate=ep_planner_budget_k16 / max(ep_len, 1),
            residual_planner_budget_k24_rate=ep_planner_budget_k24 / max(ep_len, 1),
            cost_qos=ep_cost_qos / ep_len_safe,
            cost_temp_anchor=float(ep_cost_temp[0]) / ep_len_safe,
            cost_temp_boost1=float(ep_cost_temp[1]) / ep_len_safe,
            cost_temp_boost2=float(ep_cost_temp[2]) / ep_len_safe,
            temp0_max=float(ep_temp_max[0]),
            temp1_max=float(ep_temp_max[1]),
            temp2_max=float(ep_temp_max[2]),
            thermal_headroom_min=headroom_min,
            mode_ps_rate=float(ep_mode_counts[0]) / ep_len_safe,
            mode_ts_rate=float(ep_mode_counts[1]) / ep_len_safe,
            mode_hy_rate=float(ep_mode_counts[2]) / ep_len_safe,
            boost_anchor_rate=float(ep_boost_counts[0]) / ep_len_safe,
            boost_ld1_rate=float(ep_boost_counts[1]) / ep_len_safe,
            boost_ld2_rate=float(ep_boost_counts[2]) / ep_len_safe,
            boost_all_rate=float(ep_boost_counts[3]) / ep_len_safe,
            projected_current_total_mean=ep_projected_current_total / ep_len_safe,
            bus_utilization_mean=ep_bus_utilization / ep_len_safe,
            projection_residual_mean=ep_projection_residual / ep_len_safe,
            thermal_gain_mean0=float(ep_thermal_gain_mean[0]) / ep_len_safe,
            thermal_gain_mean1=float(ep_thermal_gain_mean[1]) / ep_len_safe,
            thermal_gain_mean2=float(ep_thermal_gain_mean[2]) / ep_len_safe,
            thermal_gain_std0=float(ep_thermal_gain_std[0]) / ep_len_safe,
            thermal_gain_std1=float(ep_thermal_gain_std[1]) / ep_len_safe,
            thermal_gain_std2=float(ep_thermal_gain_std[2]) / ep_len_safe,
            thermal_gain_safe_scale0=float(ep_thermal_gain_safe_scale[0]) / ep_len_safe,
            thermal_gain_safe_scale1=float(ep_thermal_gain_safe_scale[1]) / ep_len_safe,
            thermal_gain_safe_scale2=float(ep_thermal_gain_safe_scale[2]) / ep_len_safe,
            thermal_valid_count0=float(ep_thermal_valid_count[0]) / ep_len_safe,
            thermal_valid_count1=float(ep_thermal_valid_count[1]) / ep_len_safe,
            thermal_valid_count2=float(ep_thermal_valid_count[2]) / ep_len_safe,
            planner_h2_veto_rate=ep_planner_h2_veto / ep_len_safe,
            planner_constraint_rejection_rate=ep_planner_constraint_rejection / ep_len_safe,
            planner_h1_thermal_risk=ep_planner_h1_risk / ep_len_safe,
            planner_h2_thermal_risk=ep_planner_h2_risk / ep_len_safe,
            planner_h2_max_temperature=ep_planner_h2_max_temp / ep_len_safe,
            planner_target_constraint_value=ep_planner_target_constraint / ep_len_safe,
            upper_shield_applied_rate=ep_upper_shield_applied / ep_len_safe,
            upper_shield_ld1_disabled_rate=ep_upper_shield_ld1_disabled / ep_len_safe,
            upper_shield_ld2_disabled_rate=ep_upper_shield_ld2_disabled / ep_len_safe,
            upper_shield_all_disabled_rate=ep_upper_shield_all_disabled / ep_len_safe,
            upper_shield_ld1_locked_rate=ep_upper_shield_ld1_locked / ep_len_safe,
            upper_shield_ld2_locked_rate=ep_upper_shield_ld2_locked / ep_len_safe,
            current_requested0_mean=float(ep_current_requested[0]) / ep_len_safe,
            current_requested1_mean=float(ep_current_requested[1]) / ep_len_safe,
            current_requested2_mean=float(ep_current_requested[2]) / ep_len_safe,
            current_exec0_mean=float(ep_current_exec[0]) / ep_len_safe,
            current_exec1_mean=float(ep_current_exec[1]) / ep_len_safe,
            current_exec2_mean=float(ep_current_exec[2]) / ep_len_safe,
            thermal_cap_current0_mean=float(ep_thermal_cap_current[0]) / ep_len_safe,
            thermal_cap_current1_mean=float(ep_thermal_cap_current[1]) / ep_len_safe,
            thermal_cap_current2_mean=float(ep_thermal_cap_current[2]) / ep_len_safe,
            predicted_headroom0_mean=float(ep_predicted_headroom[0]) / ep_len_safe,
            predicted_headroom1_mean=float(ep_predicted_headroom[1]) / ep_len_safe,
            predicted_headroom2_mean=float(ep_predicted_headroom[2]) / ep_len_safe,
            compression_ratio0_mean=float(ep_compression_ratio[0]) / ep_len_safe,
            compression_ratio1_mean=float(ep_compression_ratio[1]) / ep_len_safe,
            compression_ratio2_mean=float(ep_compression_ratio[2]) / ep_len_safe,
            qos_recovered_current_mean=ep_qos_recovered_current / ep_len_safe,
            execution_guard_applied_rate=ep_execution_guard_applied / ep_len_safe,
            execution_guard_downgrade_rate=ep_execution_guard_downgrade / ep_len_safe,
            execution_guard_anchor_clamp_rate=ep_execution_guard_anchor_clamp / ep_len_safe,
            execution_guard_downgrade_ld1_rate=ep_execution_guard_downgrade_ld1 / ep_len_safe,
            execution_guard_downgrade_ld2_rate=ep_execution_guard_downgrade_ld2 / ep_len_safe,
        )

    def train(self, meta_iters: int | None = None) -> Path:
        meta_cfg = self.cfg["meta"]
        meta_iters = int(meta_iters or meta_cfg["meta_iters"])

        for it in range(1, meta_iters + 1):
            self.agent.set_meta_iter(it)
            iter_global_step_start = int(self.agent.global_step)
            iter_upper_steps_start = int(self.agent.upper.update_steps)
            iter_lower_steps_start = int(self.agent.lower.update_steps)
            curriculum_stage, sampler_cfg = self._sampler_cfg_for_iter(it, meta_iters)
            self.task_sampler.cfg = sampler_cfg
            tasks = self.task_sampler.sample(int(meta_cfg["n_tasks_per_iter"]))

            support_stats: List[EpisodeStats] = []
            query_stats: List[EpisodeStats] = []
            adapted_states: List[Dict] = []
            adapted_dual_states: List[Dict] = []
            inner_delta_norms: List[float] = []
            target_delta_norms: List[float] = []
            support_gate_records: List[Dict[str, float | str | bool]] = []
            base_state = self.agent.snapshot_train_state() if self.explicit_inner_outer else None
            base_dual_state = copy.deepcopy(self.dual.state_dict()) if self.explicit_inner_outer else None
            shared_global_step = int(self.agent.global_step)
            shared_upper_steps = int(self.agent.upper.update_steps)
            shared_lower_steps = int(self.agent.lower.update_steps)

            for task_idx, task in enumerate(tasks):
                env = MultiTxUwSliptEnv(self.cfg, overrides=task.to_env_overrides())
                if self.explicit_inner_outer and base_state is not None:
                    self.agent.restore_train_state(base_state)
                    self.agent.global_step = shared_global_step
                    self.agent.upper.update_steps = shared_upper_steps
                    self.agent.lower.update_steps = shared_lower_steps
                if self.explicit_inner_outer and base_dual_state is not None:
                    self.dual.load_state_dict(copy.deepcopy(base_dual_state))
                if self.explicit_inner_outer:
                    self.agent.clear_learning_buffers()

                gate_active = self._support_gate_active()
                rollback_state = self.agent.snapshot_mutable_state() if gate_active else None
                rollback_dual_state = copy.deepcopy(self.dual.state_dict()) if gate_active else None
                support_adapt_start = time.perf_counter()
                task_support_stats: List[EpisodeStats] = []
                gate_pre_validation_stats: List[EpisodeStats] = []
                gate_post_validation_stats: List[EpisodeStats] = []
                gate_validation_seeds: List[int] = []
                gate_validation_latency_ms = 0.0
                paired_gate_validation = bool(
                    gate_active and self.support_gate_paired_validation and self.support_gate_validation_episodes > 0
                )
                support_train_episode_count = 0
                support_validation_episode_count = 0
                support_episode_budget = (
                    int(self.support_adaptation_episodes)
                    if gate_active
                    else int(meta_cfg["support_episodes"])
                )
                for ep_idx in range(support_episode_budget):
                    support_train = bool((not gate_active) and ep_idx < self.support_adaptation_episodes) or bool(gate_active)
                    support_train_episode_count += int(support_train)
                    support_validation_episode_count += int(not support_train)
                    ep_stats = self._run_episode(env, train=support_train, clear_context=(ep_idx == 0))
                    support_stats.append(ep_stats)
                    task_support_stats.append(ep_stats)
                support_adapt_latency_ms = float((time.perf_counter() - support_adapt_start) * 1000.0)

                gate_accepted = True
                gate_reason = "support_gate_disabled"
                gate_score = 0.0
                gate_threshold = 0.0
                gate_reward_before = 0.0
                gate_reward_after = 0.0
                gate_reward_delta = 0.0
                gate_cost_before = 0.0
                gate_cost_after = 0.0
                gate_cost_delta = 0.0
                gate_violation_before = 0.0
                gate_violation_after = 0.0
                gate_violation_delta = 0.0
                gate_parameter_delta_norm = 0.0
                rollback_performed = False
                rollback_residual = 0.0
                rollback_dual_residual = 0.0
                rollback_optimizer_residual = 0.0
                rollback_target_residual = 0.0
                rollback_context_residual = 0
                rollback_lower_replay_residual = 0
                rollback_upper_replay_residual = 0
                rollback_safety_residual = 0.0
                rng_state_restored = True
                gate_latency_ms = 0.0
                rollback_latency_ms = 0.0

                if gate_active and rollback_state is not None and rollback_dual_state is not None:
                    candidate_state = self.agent.snapshot_train_state()
                    candidate_mutable_state = self.agent.snapshot_mutable_state()
                    candidate_dual_state = copy.deepcopy(self.dual.state_dict())
                    if paired_gate_validation:
                        (
                            gate_pre_validation_stats,
                            gate_post_validation_stats,
                            gate_validation_seeds,
                            gate_validation_latency_ms,
                        ) = self._run_gate_validation_pair(
                            task=task,
                            iteration=it,
                            task_idx=task_idx,
                            candidate_state=candidate_mutable_state,
                            candidate_dual_state=candidate_dual_state,
                            pre_state=rollback_state,
                            pre_dual_state=rollback_dual_state,
                        )
                        support_validation_episode_count = len(gate_post_validation_stats)
                    gate_start = time.perf_counter()
                    gate_parameter_delta_norm = self._trainable_parameter_delta_norm(rollback_state, candidate_state)
                    decision = self._evaluate_support_gate(
                        support_stats=task_support_stats,
                        parameter_delta=gate_parameter_delta_norm,
                        pre_validation_stats=gate_pre_validation_stats if paired_gate_validation else None,
                        post_validation_stats=gate_post_validation_stats if paired_gate_validation else None,
                    )
                    gate_accepted = bool(decision.accepted)
                    gate_reason = str(decision.reason)
                    gate_score = float(decision.gate_score)
                    gate_threshold = float(decision.threshold)
                    gate_reward_before = float(decision.support_stats_before.reward)
                    gate_reward_after = float(decision.support_stats_after.reward)
                    gate_reward_delta = float(decision.reward_delta)
                    gate_cost_before = float(decision.support_stats_before.cost)
                    gate_cost_after = float(decision.support_stats_after.cost)
                    gate_cost_delta = float(decision.cost_delta)
                    gate_violation_before = float(decision.support_stats_before.violation_rate)
                    gate_violation_after = float(decision.support_stats_after.violation_rate)
                    gate_violation_delta = float(decision.violation_delta)
                    gate_latency_ms = float((time.perf_counter() - gate_start) * 1000.0)
                    if not gate_accepted:
                        rollback_start = time.perf_counter()
                        self.agent.restore_mutable_state(rollback_state)
                        self.dual.load_state_dict(copy.deepcopy(rollback_dual_state))
                        rollback_performed = True
                        rollback_latency_ms = float((time.perf_counter() - rollback_start) * 1000.0)
                        restored_state = self.agent.snapshot_train_state()
                        rollback_residual = self._trainable_parameter_delta_norm(rollback_state, restored_state)
                        rollback_dual_residual = self._dual_delta_norm(
                            rollback_dual_state,
                            self.dual.state_dict(),
                        )
                        rollback_optimizer_residual = self._optimizer_state_delta_norm(
                            rollback_state,
                            restored_state,
                        )
                        rollback_target_residual = self._target_parameter_delta_norm(
                            rollback_state,
                            restored_state,
                        )
                        rollback_safety_residual = self._safety_state_delta_norm(
                            rollback_state.get("safety", {}),
                            self.agent.safety.state_dict(),
                        )
                        rollback_context_residual = abs(
                            int(len(self.agent.episode))
                            - int(len(rollback_state.get("episode", {}).get("items", [])))
                        )
                        rollback_lower_replay_residual = abs(
                            int(len(self.agent.replay))
                            - int(len(rollback_state.get("replay", {}).get("items", [])))
                        )
                        rollback_upper_replay_residual = abs(
                            int(len(self.agent.upper_replay))
                            - int(len(rollback_state.get("upper_replay", {}).get("items", [])))
                        )
                        rng_state_restored = "rng" in rollback_state

                if self.explicit_inner_outer and self.dual_enabled and task_support_stats and gate_accepted:
                    task_support_mean_cost_vec = np.mean(
                        np.stack([s.cost_vec for s in task_support_stats], axis=0),
                        axis=0,
                    )
                    self.dual.update(task_support_mean_cost_vec)

                support_gate_records.append(
                    {
                        "support_gate_enabled": bool(gate_active),
                        "support_gate_accepted": bool(gate_accepted),
                        "rollback_performed": bool(rollback_performed),
                        "support_gate_reason": gate_reason,
                        "support_gate_score": float(gate_score),
                        "support_gate_threshold": float(gate_threshold),
                        "support_gate_budget_mode": str(
                            self.support_gate_cfg.get(
                                "budget_mode",
                                "paired_support_validation" if paired_gate_validation else "support_adaptation_only",
                            )
                        ),
                        "support_reward_before": float(gate_reward_before),
                        "support_reward_after": float(gate_reward_after),
                        "support_reward_delta": float(gate_reward_delta),
                        "support_cost_before": float(gate_cost_before),
                        "support_cost_after": float(gate_cost_after),
                        "support_cost_delta": float(gate_cost_delta),
                        "support_violation_before": float(gate_violation_before),
                        "support_violation_after": float(gate_violation_after),
                        "support_violation_delta": float(gate_violation_delta),
                        "support_parameter_delta_norm": float(gate_parameter_delta_norm),
                        "rollback_parameter_residual": float(rollback_residual),
                        "rollback_dual_residual": float(rollback_dual_residual),
                        "rollback_optimizer_residual": float(rollback_optimizer_residual),
                        "rollback_target_parameter_residual": float(rollback_target_residual),
                        "rollback_context_residual": int(rollback_context_residual),
                        "rollback_lower_replay_residual": int(rollback_lower_replay_residual),
                        "rollback_upper_replay_residual": int(rollback_upper_replay_residual),
                        "rollback_safety_estimator_residual": float(rollback_safety_residual),
                        "thermal_estimator_state_restored": bool(rollback_safety_residual <= 1.0e-8),
                        "rng_state_restored": bool(rng_state_restored),
                        "gate_latency_ms": float(gate_latency_ms),
                        "rollback_latency_ms": float(rollback_latency_ms),
                        "support_adaptation_latency_ms": float(support_adapt_latency_ms),
                        "support_adaptation_episodes": int(support_train_episode_count),
                        "support_gate_validation_episodes": int(support_validation_episode_count),
                        "support_gate_paired_validation": bool(paired_gate_validation),
                        "support_gate_pre_validation_episodes": int(len(gate_pre_validation_stats)),
                        "support_gate_post_validation_episodes": int(len(gate_post_validation_stats)),
                        "support_gate_validation_seed_pairs": int(len(gate_validation_seeds)),
                        "support_gate_same_validation_seeds": bool(
                            (not paired_gate_validation) or len(gate_pre_validation_stats) == len(gate_post_validation_stats)
                        ),
                        "support_gate_validation_latency_ms": float(gate_validation_latency_ms),
                        "total_adaptation_latency_ms": float(
                            support_adapt_latency_ms
                            + gate_validation_latency_ms
                            + gate_latency_ms
                            + rollback_latency_ms
                        ),
                        "ungated_support_adaptation_latency_ms": float(support_adapt_latency_ms),
                        "gated_support_adaptation_latency_ms": float(
                            support_adapt_latency_ms
                            + gate_validation_latency_ms
                            + gate_latency_ms
                            + rollback_latency_ms
                        )
                        if gate_active
                        else 0.0,
                        "extra_support_rollouts": int(len(gate_pre_validation_stats)),
                        "extra_gradient_updates": 0,
                        "extra_query_evaluations": 0,
                        "query_leakage": False,
                    }
                )

                for ep_idx in range(int(meta_cfg["query_episodes"])):
                    query_stats.append(
                        self._run_episode(
                            env,
                            train=bool(self.explicit_inner_outer and self.query_updates_enabled),
                            clear_context=(int(meta_cfg["support_episodes"]) <= 0 and ep_idx == 0),
                        )
                    )

                if self.explicit_inner_outer:
                    adapted_state = self.agent.snapshot_train_state()
                    adapted_states.append(adapted_state)
                    adapted_dual_states.append(copy.deepcopy(self.dual.state_dict()))
                    if base_state is not None:
                        inner_delta_norms.append(self._trainable_parameter_delta_norm(base_state, adapted_state))
                        target_delta_norms.append(self._target_parameter_delta_norm(base_state, adapted_state))
                shared_global_step = max(shared_global_step, int(self.agent.global_step))
                shared_upper_steps = max(shared_upper_steps, int(self.agent.upper.update_steps))
                shared_lower_steps = max(shared_lower_steps, int(self.agent.lower.update_steps))

            if self.explicit_inner_outer and base_state is not None:
                self.agent.restore_train_state(base_state)
                self.agent.global_step = shared_global_step
                self.agent.upper.update_steps = shared_upper_steps
                self.agent.lower.update_steps = shared_lower_steps
                self.agent.apply_outer_update(adapted_states, self.outer_step_size)
                if self.reset_optimizer_after_outer_update:
                    self.agent.reset_optimizer_states()
                if self.dual_enabled and base_dual_state is not None:
                    base_values = np.asarray(base_dual_state["values"], dtype=np.float32)
                    if adapted_dual_states:
                        avg_values = np.mean(
                            np.stack([np.asarray(s["values"], dtype=np.float32) for s in adapted_dual_states], axis=0),
                            axis=0,
                        )
                    else:
                        avg_values = base_values
                    blended_values = base_values + self.outer_step_size * (avg_values - base_values)
                    dual_state = copy.deepcopy(base_dual_state)
                    dual_state["values"] = np.clip(blended_values, 0.0, np.asarray(base_dual_state["max_lambdas"], dtype=np.float32))
                    self.dual.load_state_dict(dual_state)

            mean_support_cost = float(np.mean([s.cost for s in support_stats])) if support_stats else 0.0
            if support_stats:
                mean_support_cost_vec = np.mean(np.stack([s.cost_vec for s in support_stats], axis=0), axis=0)
            else:
                mean_support_cost_vec = np.zeros(self.dual.n_constraints, dtype=np.float32)
            if (not self.dual_enabled):
                lambda_val = 0.0
            elif self.explicit_inner_outer and base_dual_state is not None:
                lambda_val = float(np.mean(self.dual.values))
            else:
                lambda_val = self.dual.update(mean_support_cost_vec)

            def _gate_mean(key: str) -> float:
                vals = [r[key] for r in support_gate_records if isinstance(r.get(key), (int, float, bool))]
                return float(np.mean([float(v) for v in vals])) if vals else 0.0

            gate_enabled_fraction = _gate_mean("support_gate_enabled")
            gate_accept_rate = _gate_mean("support_gate_accepted") if support_gate_records else 0.0
            gate_reject_rate = float(max(0.0, gate_enabled_fraction - gate_accept_rate))
            accepted_update_count = int(
                sum(1 for r in support_gate_records if bool(r.get("support_gate_enabled")) and bool(r.get("support_gate_accepted")))
            )
            rejected_update_count = int(
                sum(1 for r in support_gate_records if bool(r.get("support_gate_enabled")) and not bool(r.get("support_gate_accepted")))
            )
            support_parameter_delta_logged = (
                _gate_mean("support_parameter_delta_norm")
                if gate_enabled_fraction > 0.0
                else (float(np.mean(inner_delta_norms)) if inner_delta_norms else 0.0)
            )
            first_gate_reason = next(
                (str(r.get("support_gate_reason", "")) for r in support_gate_records if str(r.get("support_gate_reason", ""))),
                "",
            )
            first_gate_budget_mode = next(
                (
                    str(r.get("support_gate_budget_mode", ""))
                    for r in support_gate_records
                    if str(r.get("support_gate_budget_mode", ""))
                ),
                "",
            )
            row = {
                "iter": float(it),
                "support_reward": float(np.mean([s.reward for s in support_stats])),
                "support_reward_task": float(np.mean([s.reward_task for s in support_stats])),
                "support_reward_benchmark": float(np.mean([s.reward_benchmark for s in support_stats])),
                "support_reward_dual_penalized": float(
                    np.mean([s.reward_dual_penalized for s in support_stats])
                ),
                "support_se": float(np.mean([s.se for s in support_stats])),
                "support_eh": float(np.mean([s.eh for s in support_stats])),
                "support_cost": mean_support_cost,
                "support_violation_rate": float(np.mean([s.violations for s in support_stats])),
                "support_se_term": float(np.mean([s.se_term for s in support_stats])),
                "support_eh_term": float(np.mean([s.eh_term for s in support_stats])),
                "support_cost_term": float(np.mean([s.cost_term for s in support_stats])),
                "support_power_term": float(np.mean([s.power_term for s in support_stats])),
                "support_smooth_term": float(np.mean([s.smooth_term for s in support_stats])),
                "support_eh_input_eff": float(np.mean([s.eh_input_eff for s in support_stats])),
                "support_eh_metric_raw_nonlinear": float(
                    np.mean([s.eh_metric_raw_nonlinear for s in support_stats])
                ),
                "support_eh_saturation_fraction": float(np.mean([s.eh_saturation_fraction for s in support_stats])),
                "support_eh_near_zero_fraction": float(np.mean([s.eh_near_zero_fraction for s in support_stats])),
                "support_residual_planner_active_rate": float(
                    np.mean([s.residual_planner_active_rate for s in support_stats])
                ),
                "support_residual_planner_replacement_rate": float(
                    np.mean([s.residual_planner_replacement_rate for s in support_stats])
                ),
                "support_residual_planner_fallback_rate": float(
                    np.mean([s.residual_planner_fallback_rate for s in support_stats])
                ),
                "support_residual_planner_score_improvement": float(
                    np.mean([s.residual_planner_score_improvement for s in support_stats])
                ),
                "support_residual_planner_best_score_improvement": float(
                    np.mean([s.residual_planner_best_score_improvement for s in support_stats])
                ),
                "support_residual_planner_candidate_distance": float(
                    np.mean([s.residual_planner_candidate_distance for s in support_stats])
                ),
                "support_residual_planner_disagreement": float(
                    np.mean([s.residual_planner_disagreement for s in support_stats])
                ),
                "support_residual_planner_budget_mean": float(
                    np.mean([s.residual_planner_budget_mean for s in support_stats])
                ),
                "support_residual_planner_budget_k0_rate": float(
                    np.mean([s.residual_planner_budget_k0_rate for s in support_stats])
                ),
                "support_residual_planner_budget_k8_rate": float(
                    np.mean([s.residual_planner_budget_k8_rate for s in support_stats])
                ),
                "support_residual_planner_budget_k16_rate": float(
                    np.mean([s.residual_planner_budget_k16_rate for s in support_stats])
                ),
                "support_residual_planner_budget_k24_rate": float(
                    np.mean([s.residual_planner_budget_k24_rate for s in support_stats])
                ),
                "query_reward": float(np.mean([s.reward for s in query_stats])) if query_stats else 0.0,
                "query_reward_task": float(np.mean([s.reward_task for s in query_stats])) if query_stats else 0.0,
                "query_reward_benchmark": float(np.mean([s.reward_benchmark for s in query_stats]))
                if query_stats
                else 0.0,
                "query_reward_dual_penalized": float(
                    np.mean([s.reward_dual_penalized for s in query_stats])
                )
                if query_stats
                else 0.0,
                "query_se": float(np.mean([s.se for s in query_stats])) if query_stats else 0.0,
                "query_eh": float(np.mean([s.eh for s in query_stats])) if query_stats else 0.0,
                "query_cost": float(np.mean([s.cost for s in query_stats])) if query_stats else 0.0,
                "query_violation_rate": float(np.mean([s.violations for s in query_stats]))
                if query_stats
                else 0.0,
                "query_se_term": float(np.mean([s.se_term for s in query_stats])) if query_stats else 0.0,
                "query_eh_term": float(np.mean([s.eh_term for s in query_stats])) if query_stats else 0.0,
                "query_cost_term": float(np.mean([s.cost_term for s in query_stats])) if query_stats else 0.0,
                "query_power_term": float(np.mean([s.power_term for s in query_stats])) if query_stats else 0.0,
                "query_smooth_term": float(np.mean([s.smooth_term for s in query_stats])) if query_stats else 0.0,
                "query_eh_input_eff": float(np.mean([s.eh_input_eff for s in query_stats])) if query_stats else 0.0,
                "query_eh_metric_raw_nonlinear": float(
                    np.mean([s.eh_metric_raw_nonlinear for s in query_stats])
                )
                if query_stats
                else 0.0,
                "query_eh_saturation_fraction": float(np.mean([s.eh_saturation_fraction for s in query_stats]))
                if query_stats
                else 0.0,
                "query_eh_near_zero_fraction": float(np.mean([s.eh_near_zero_fraction for s in query_stats]))
                if query_stats
                else 0.0,
                "query_residual_planner_active_rate": float(
                    np.mean([s.residual_planner_active_rate for s in query_stats])
                )
                if query_stats
                else 0.0,
                "query_residual_planner_replacement_rate": float(
                    np.mean([s.residual_planner_replacement_rate for s in query_stats])
                )
                if query_stats
                else 0.0,
                "query_residual_planner_fallback_rate": float(
                    np.mean([s.residual_planner_fallback_rate for s in query_stats])
                )
                if query_stats
                else 0.0,
                "query_residual_planner_score_improvement": float(
                    np.mean([s.residual_planner_score_improvement for s in query_stats])
                )
                if query_stats
                else 0.0,
                "query_residual_planner_best_score_improvement": float(
                    np.mean([s.residual_planner_best_score_improvement for s in query_stats])
                )
                if query_stats
                else 0.0,
                "query_residual_planner_candidate_distance": float(
                    np.mean([s.residual_planner_candidate_distance for s in query_stats])
                )
                if query_stats
                else 0.0,
                "query_residual_planner_disagreement": float(
                    np.mean([s.residual_planner_disagreement for s in query_stats])
                )
                if query_stats
                else 0.0,
                "query_residual_planner_budget_mean": float(
                    np.mean([s.residual_planner_budget_mean for s in query_stats])
                )
                if query_stats
                else 0.0,
                "query_residual_planner_budget_k0_rate": float(
                    np.mean([s.residual_planner_budget_k0_rate for s in query_stats])
                )
                if query_stats
                else 0.0,
                "query_residual_planner_budget_k8_rate": float(
                    np.mean([s.residual_planner_budget_k8_rate for s in query_stats])
                )
                if query_stats
                else 0.0,
                "query_residual_planner_budget_k16_rate": float(
                    np.mean([s.residual_planner_budget_k16_rate for s in query_stats])
                )
                if query_stats
                else 0.0,
                "query_residual_planner_budget_k24_rate": float(
                    np.mean([s.residual_planner_budget_k24_rate for s in query_stats])
                )
                if query_stats
                else 0.0,
                "lambda": lambda_val,
                "curriculum_stage": curriculum_stage,
                "outer_step_size": float(self.outer_step_size if self.explicit_inner_outer else 0.0),
                "reset_optimizer_after_outer_update": bool(
                    self.explicit_inner_outer and self.reset_optimizer_after_outer_update
                ),
                "explicit_inner_outer": bool(self.explicit_inner_outer),
                "query_updates_enabled": bool(self.query_updates_enabled),
                "query_context_updates_enabled": bool(self.query_context_updates_enabled),
                "heldout_query_evaluation": bool(self.explicit_inner_outer and not self.query_updates_enabled),
                "support_gate_enabled": bool(gate_enabled_fraction > 0.0),
                "support_gate_role": self.support_gate_role if gate_enabled_fraction > 0.0 else "",
                "support_gate_rule": self.support_gate_rule if gate_enabled_fraction > 0.0 else "",
                "support_update_acceptance": "support_side_gated" if gate_enabled_fraction > 0.0 else "unconditional",
                "support_gate_uses_query": False,
                "support_gate_budget_mode": first_gate_budget_mode if gate_enabled_fraction > 0.0 else "",
                "support_gate_paired_validation": bool(_gate_mean("support_gate_paired_validation") > 0.0),
                "support_gate_same_validation_seeds": bool(
                    gate_enabled_fraction == 0.0 or _gate_mean("support_gate_same_validation_seeds") >= 1.0
                ),
                "support_gate_pre_validation_episodes": int(round(_gate_mean("support_gate_pre_validation_episodes"))),
                "support_gate_post_validation_episodes": int(round(_gate_mean("support_gate_post_validation_episodes"))),
                "support_gate_validation_seed_pairs": int(round(_gate_mean("support_gate_validation_seed_pairs"))),
                "support_gate_extra_rollouts": int(round(_gate_mean("extra_support_rollouts"))),
                "support_gate_extra_gradient_updates": 0,
                "support_gate_extra_query_evaluations": 0,
                "support_adaptation_episodes": int(round(_gate_mean("support_adaptation_episodes"))),
                "support_gate_validation_episodes": int(round(_gate_mean("support_gate_validation_episodes"))),
                "support_gate_accept_rate": gate_accept_rate,
                "support_gate_reject_rate": gate_reject_rate,
                "accepted_update_count": accepted_update_count,
                "rejected_update_count": rejected_update_count,
                "support_gate_accepted": bool(gate_accept_rate >= 0.5) if gate_enabled_fraction > 0.0 else True,
                "support_gate_reason": first_gate_reason,
                "support_gate_score": _gate_mean("support_gate_score"),
                "support_gate_threshold": _gate_mean("support_gate_threshold"),
                "support_reward_before": _gate_mean("support_reward_before"),
                "support_reward_after": _gate_mean("support_reward_after"),
                "support_reward_delta": _gate_mean("support_reward_delta"),
                "support_cost_before": _gate_mean("support_cost_before"),
                "support_cost_after": _gate_mean("support_cost_after"),
                "support_cost_delta": _gate_mean("support_cost_delta"),
                "support_violation_before": _gate_mean("support_violation_before"),
                "support_violation_after": _gate_mean("support_violation_after"),
                "support_violation_delta": _gate_mean("support_violation_delta"),
                "support_parameter_delta_norm": float(support_parameter_delta_logged),
                "support_target_parameter_delta_norm": float(np.mean(target_delta_norms)) if target_delta_norms else 0.0,
                "support_gate_parameter_delta_norm": _gate_mean("support_parameter_delta_norm"),
                "rollback_performed": bool(_gate_mean("rollback_performed") > 0.0),
                "rollback_parameter_residual": _gate_mean("rollback_parameter_residual"),
                "rollback_dual_residual": _gate_mean("rollback_dual_residual"),
                "rollback_optimizer_residual": _gate_mean("rollback_optimizer_residual"),
                "rollback_target_parameter_residual": _gate_mean("rollback_target_parameter_residual"),
                "rollback_context_residual": _gate_mean("rollback_context_residual"),
                "rollback_lower_replay_residual": _gate_mean("rollback_lower_replay_residual"),
                "rollback_upper_replay_residual": _gate_mean("rollback_upper_replay_residual"),
                "rollback_safety_estimator_residual": _gate_mean("rollback_safety_estimator_residual"),
                "optimizer_state_restored": bool(gate_reject_rate == 0.0 or _gate_mean("rollback_optimizer_residual") <= 1.0e-8),
                "target_network_state_restored": bool(
                    gate_reject_rate == 0.0 or _gate_mean("rollback_target_parameter_residual") <= 1.0e-8
                ),
                "context_state_restored": bool(gate_reject_rate == 0.0 or _gate_mean("rollback_context_residual") == 0.0),
                "dual_state_restored": bool(gate_reject_rate == 0.0 or _gate_mean("rollback_dual_residual") <= 1.0e-8),
                "thermal_estimator_state_restored": bool(
                    gate_reject_rate == 0.0 or _gate_mean("rollback_safety_estimator_residual") <= 1.0e-8
                ),
                "rng_state_restored": bool(gate_reject_rate == 0.0 or _gate_mean("rng_state_restored") >= 1.0),
                "gate_latency_ms": _gate_mean("gate_latency_ms"),
                "support_gate_validation_latency_ms": _gate_mean("support_gate_validation_latency_ms"),
                "support_adaptation_latency_ms": _gate_mean("support_adaptation_latency_ms"),
                "total_adaptation_latency_ms": _gate_mean("total_adaptation_latency_ms"),
                "ungated_support_adaptation_latency_ms": _gate_mean("ungated_support_adaptation_latency_ms"),
                "gated_support_adaptation_latency_ms": _gate_mean("gated_support_adaptation_latency_ms"),
                "gate_only_latency_ms": _gate_mean("gate_latency_ms"),
                "rollback_latency_ms": _gate_mean("rollback_latency_ms"),
                "extra_support_rollouts": int(_gate_mean("extra_support_rollouts")),
                "extra_gradient_updates": int(_gate_mean("extra_gradient_updates")),
                "extra_query_evaluations": int(_gate_mean("extra_query_evaluations")),
                "query_leakage": False,
                "inner_warmup_steps": int(self.inner_warmup_steps),
                "inner_upper_warmup_steps": int(self.inner_upper_warmup_steps),
                "lower_batch_size": int(self.agent.batch_size),
                "upper_batch_size": int(self.agent.upper_batch_size),
                "lower_updates_per_step": int(self.agent.lower_updates_per_step),
                "upper_update_every": int(self.agent.upper_update_every),
                "upper_warmup_steps": int(self.agent.upper_warmup_steps),
                "global_step": int(self.agent.global_step),
                "upper_update_steps": int(self.agent.upper.update_steps),
                "lower_update_steps": int(self.agent.lower.update_steps),
                "iter_global_step_delta": int(self.agent.global_step - iter_global_step_start),
                "iter_upper_update_step_delta": int(self.agent.upper.update_steps - iter_upper_steps_start),
                "iter_lower_update_step_delta": int(self.agent.lower.update_steps - iter_lower_steps_start),
            }
            diagnostic_names = [
                "cost_qos",
                "cost_temp_anchor",
                "cost_temp_boost1",
                "cost_temp_boost2",
                "temp0_max",
                "temp1_max",
                "temp2_max",
                "thermal_headroom_min",
                "mode_ps_rate",
                "mode_ts_rate",
                "mode_hy_rate",
                "boost_anchor_rate",
                "boost_ld1_rate",
                "boost_ld2_rate",
                "boost_all_rate",
                "projected_current_total_mean",
                "bus_utilization_mean",
                "projection_residual_mean",
                "thermal_gain_mean0",
                "thermal_gain_mean1",
                "thermal_gain_mean2",
                "thermal_gain_std0",
                "thermal_gain_std1",
                "thermal_gain_std2",
                "thermal_gain_safe_scale0",
                "thermal_gain_safe_scale1",
                "thermal_gain_safe_scale2",
                "thermal_valid_count0",
                "thermal_valid_count1",
                "thermal_valid_count2",
                "planner_h2_veto_rate",
                "planner_constraint_rejection_rate",
                "planner_h1_thermal_risk",
                "planner_h2_thermal_risk",
                "planner_h2_max_temperature",
                "planner_target_constraint_value",
                "upper_shield_applied_rate",
                "upper_shield_ld1_disabled_rate",
                "upper_shield_ld2_disabled_rate",
                "upper_shield_all_disabled_rate",
                "upper_shield_ld1_locked_rate",
                "upper_shield_ld2_locked_rate",
                "current_requested0_mean",
                "current_requested1_mean",
                "current_requested2_mean",
                "current_exec0_mean",
                "current_exec1_mean",
                "current_exec2_mean",
                "thermal_cap_current0_mean",
                "thermal_cap_current1_mean",
                "thermal_cap_current2_mean",
                "predicted_headroom0_mean",
                "predicted_headroom1_mean",
                "predicted_headroom2_mean",
                "compression_ratio0_mean",
                "compression_ratio1_mean",
                "compression_ratio2_mean",
                "qos_recovered_current_mean",
                "execution_guard_applied_rate",
                "execution_guard_downgrade_rate",
                "execution_guard_anchor_clamp_rate",
                "execution_guard_downgrade_ld1_rate",
                "execution_guard_downgrade_ld2_rate",
            ]
            for prefix, stats_group in (("support", support_stats), ("query", query_stats)):
                for name in diagnostic_names:
                    row[f"{prefix}_{name}"] = (
                        float(np.mean([float(getattr(s, name)) for s in stats_group])) if stats_group else 0.0
                    )
            cost_component_names = list(self.dual.names)
            support_vec_mean = mean_support_cost_vec
            query_vec_mean = (
                np.mean(np.stack([s.cost_vec for s in query_stats], axis=0), axis=0)
                if query_stats
                else np.zeros(self.dual.n_constraints, dtype=np.float32)
            )
            for idx, name in enumerate(cost_component_names):
                row[f"support_cost_{name}"] = float(support_vec_mean[idx])
                row[f"query_cost_{name}"] = float(query_vec_mean[idx])
            if self.dual_enabled:
                row.update(self.dual.as_dict())
            else:
                row.update({f"lambda_{name}": 0.0 for name in self.dual.names})
            self.logger.log(row)

            if it % 10 == 0 or it == meta_iters:
                self.agent.save(self.ckpt_dir / f"iter_{it}.pt")

        return self.logger.csv_path

    def evaluate(self, n_tasks: int = 5, episodes_per_task: int = 2) -> Dict[str, float]:
        tasks = self.task_sampler.sample(n_tasks)
        stats: List[EpisodeStats] = []

        for task in tasks:
            env = MultiTxUwSliptEnv(self.cfg, overrides=task.to_env_overrides())
            for ep_idx in range(episodes_per_task):
                stats.append(self._run_episode(env, train=False, clear_context=(ep_idx == 0)))

        return {
            "reward": float(np.mean([s.reward for s in stats])),
            "se": float(np.mean([s.se for s in stats])),
            "eh": float(np.mean([s.eh for s in stats])),
            "cost": float(np.mean([s.cost for s in stats])),
            "violation_rate": float(np.mean([s.violations for s in stats])),
            "len": float(np.mean([s.length for s in stats])),
        }
