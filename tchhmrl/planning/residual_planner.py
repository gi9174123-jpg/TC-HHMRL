from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np
import torch

from tchhmrl.planning.residual_basis import residual_basis


class ResidualPlanner:
    """Policy-centered local planner for the lower continuous action.

    The planner does not roll the environment forward. It generates local raw
    lower-action candidates around the SAC policy output, projects every
    candidate through the same safety layer, and scores the executed candidates
    using learned reward/constraint critics plus current thermal state.
    """

    def __init__(self, cfg: Dict[str, Any], device: torch.device):
        planner_cfg = cfg.get("residual_planner", {}) or {}
        self.enabled = bool(planner_cfg.get("enabled", False))
        self.candidate_count = int(planner_cfg.get("candidate_count", 24))
        legacy_current_step = float(planner_cfg.get("current_step", 0.05)) * 2.0
        legacy_ratio_step = float(planner_cfg.get("ratio_step", 0.05)) * 2.0
        self.total_current_raw_step = float(planner_cfg.get("total_current_raw_step", legacy_current_step))
        self.allocation_logit_raw_step = float(planner_cfg.get("allocation_logit_raw_step", legacy_current_step))
        self.ratio_raw_step = float(planner_cfg.get("ratio_raw_step", legacy_ratio_step))
        self.disagreement_beta = float(planner_cfg.get("disagreement_beta", 0.10))
        self.projection_penalty = float(planner_cfg.get("projection_penalty", 0.10))
        self.constraint_beta = float(planner_cfg.get("constraint_beta", 1.0))
        self.thermal_risk_beta = float(planner_cfg.get("thermal_risk_beta", 0.05))
        self.h2_increment_beta = float(planner_cfg.get("h2_increment_beta", self.thermal_risk_beta))
        self.thermal_margin_target_c = float(planner_cfg.get("thermal_margin_target_c", 1.0))
        self.thermal_horizon = int(planner_cfg.get("thermal_horizon", 2))
        self.start_meta_iter = int(planner_cfg.get("start_meta_iter", 60))
        self.thermal_horizon_start_meta_iter = int(planner_cfg.get("thermal_horizon_start_meta_iter", 86))
        self.adaptive_budget_enabled = bool(planner_cfg.get("adaptive_budget_enabled", False))
        budget_candidates = planner_cfg.get("budget_candidates", [0, 8, 16, 24])
        self.budget_candidates = tuple(sorted({int(x) for x in budget_candidates if int(x) >= 0}))
        if not self.budget_candidates:
            self.budget_candidates = (0, 8, 16, 24)
        self.budget_low_periodic_interval = int(planner_cfg.get("budget_low_periodic_interval", 10))
        self.budget_low_periodic_k = int(planner_cfg.get("budget_low_periodic_k", 8))
        self.budget_medium_k = int(planner_cfg.get("budget_medium_k", 16))
        self.budget_high_k = int(planner_cfg.get("budget_high_k", 24))
        self.budget_high_headroom_c = float(planner_cfg.get("budget_high_headroom_c", 1.0))
        self.budget_medium_headroom_c = float(planner_cfg.get("budget_medium_headroom_c", 3.0))
        self.budget_high_gain_std = float(planner_cfg.get("budget_high_gain_std", 1.5))
        self.budget_medium_gain_std = float(planner_cfg.get("budget_medium_gain_std", 0.5))
        self.budget_high_disagreement = float(planner_cfg.get("budget_high_disagreement", 1.0))
        self.budget_medium_disagreement = float(planner_cfg.get("budget_medium_disagreement", 0.25))
        self.budget_high_constraint = float(planner_cfg.get("budget_high_constraint", 0.20))
        self.budget_medium_constraint = float(planner_cfg.get("budget_medium_constraint", 0.05))
        self.budget_high_projection_residual = float(planner_cfg.get("budget_high_projection_residual", 0.25))
        self.budget_medium_projection_residual = float(planner_cfg.get("budget_medium_projection_residual", 0.10))
        self.trust_region_enabled = bool(planner_cfg.get("trust_region_enabled", False))
        self.trust_region_mode = str(planner_cfg.get("trust_region_mode", "raw_and_executed"))
        self.trust_region_raw_l2 = planner_cfg.get("trust_region_raw_l2", None)
        self.trust_region_exec_l2 = planner_cfg.get("trust_region_exec_l2", None)
        self.trust_region_raw_l2 = None if self.trust_region_raw_l2 is None else float(self.trust_region_raw_l2)
        self.trust_region_exec_l2 = None if self.trust_region_exec_l2 is None else float(self.trust_region_exec_l2)
        self.replacement_margin_mode = str(planner_cfg.get("replacement_margin_mode", "normalized"))
        margin_cfg = planner_cfg.get("replacement_margin", None)
        self.replacement_margin = None if margin_cfg is None else float(margin_cfg)
        factor_cfg = planner_cfg.get("normalized_margin_factor", 0.25)
        self.normalized_margin_factor = None if factor_cfg is None else float(factor_cfg)
        self.constraint_tolerance = float(planner_cfg.get("constraint_tolerance", 0.0))
        projection_limit = planner_cfg.get("projection_residual_limit", None)
        self.projection_residual_limit = None if projection_limit is None else float(projection_limit)
        self.h2_veto_enabled = bool(planner_cfg.get("h2_veto_enabled", True))
        self.h2_margin = float(planner_cfg.get("h2_margin", 0.0))
        self.device = device

    def active(self, *, meta_iter: int) -> bool:
        return bool(self.enabled and int(meta_iter) >= self.start_meta_iter and self.candidate_count > 1)

    def _candidate_raw(
        self,
        policy_raw: np.ndarray,
        *,
        boost_combo: int,
        mode: int,
        thermal_headroom: np.ndarray | None,
        candidate_count: int | None = None,
    ) -> np.ndarray:
        raw = np.asarray(policy_raw, dtype=np.float32).reshape(-1)[:5]
        active = np.asarray([1.0, float(int(boost_combo) in {1, 3}), float(int(boost_combo) in {2, 3})], dtype=np.float32)
        basis = residual_basis(
            candidate_count=self.candidate_count if candidate_count is None else int(candidate_count),
            total_current_raw_step=self.total_current_raw_step,
            allocation_logit_raw_step=self.allocation_logit_raw_step,
            ratio_raw_step=self.ratio_raw_step,
            mode=int(mode),
            active_source_mask=active,
            thermal_headroom=thermal_headroom,
        )
        candidates = raw.reshape(1, -1) + basis
        return np.clip(candidates, -1.0, 1.0).astype(np.float32)

    def _nearest_budget(self, requested: int) -> int:
        requested = int(max(0, requested))
        candidates = sorted(self.budget_candidates)
        feasible = [k for k in candidates if k <= requested]
        if feasible:
            return int(max(feasible))
        return int(min(candidates))

    @staticmethod
    def _min_headroom(thermal_headroom: np.ndarray | None) -> float:
        if thermal_headroom is None:
            return float("inf")
        headroom = np.asarray(thermal_headroom, dtype=np.float32).reshape(-1)
        finite = headroom[np.isfinite(headroom)]
        return float(np.min(finite)) if finite.size else float("inf")

    def _policy_risk_probe(
        self,
        *,
        lower,
        safety,
        obs: np.ndarray,
        z: np.ndarray,
        upper_idx_exec: int,
        boost_combo: int,
        mode: int,
        policy_raw: np.ndarray,
        physical_features: np.ndarray,
        thermal_headroom: np.ndarray | None,
        temps: np.ndarray,
        amb_temp: float,
        previous_projection_residual: np.ndarray | None,
    ) -> Dict[str, float]:
        raw = torch.as_tensor(np.asarray(policy_raw, dtype=np.float32).reshape(1, -1), dtype=torch.float32, device=self.device)
        boost_t = torch.full((1,), int(boost_combo), dtype=torch.long, device=self.device)
        mode_t = torch.full((1,), int(mode), dtype=torch.long, device=self.device)
        temps_t = torch.as_tensor(np.asarray(temps, dtype=np.float32), dtype=torch.float32, device=self.device).view(1, -1)
        amb_t = torch.full((1,), float(amb_temp), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            safe = safety.project_torch(raw, boost_t, mode_t, temps_t, amb_t)
            executed = self._executed_action(safe)
            obs_aug = lower._augment_np(
                np.asarray(obs, dtype=np.float32),
                upper_idx=int(upper_idx_exec),
                physical_features=np.asarray(physical_features, dtype=np.float32),
                encoder=lower.q1_tgt_phys,
            )
            obs_aug_q2 = lower._augment_np(
                np.asarray(obs, dtype=np.float32),
                upper_idx=int(upper_idx_exec),
                physical_features=np.asarray(physical_features, dtype=np.float32),
                encoder=lower.q2_tgt_phys,
            )
            z_t = torch.as_tensor(np.asarray(z, dtype=np.float32), dtype=torch.float32, device=self.device).view(1, -1)
            q1 = lower.q1_tgt(obs_aug, z_t, executed)
            q2 = lower.q2_tgt(obs_aug_q2, z_t, executed)
            disagreement = float(torch.abs(q1 - q2).detach().cpu().item())
            constraint_penalty = 0.0
            if getattr(lower, "constraint_q_tgt", None) is not None:
                constraint_val = lower.constraint_q_tgt(obs_aug, z_t, executed)
                if getattr(lower, "constraint_actor_penalty_nonnegative", True):
                    constraint_val = torch.relu(constraint_val)
                constraint_penalty = float((constraint_val * lower.constraint_actor_weights).sum().detach().cpu().item())

        diag = safety.thermal_diagnostics()
        gain_std_arr = np.asarray(
            diag.get(
                "effective_gain_uncertainty",
                diag.get("effective_gain_std", diag.get("thermal_gain_std", np.zeros(3, dtype=np.float32))),
            ),
            dtype=np.float32,
        ).reshape(-1)
        gain_std = float(np.nanmax(gain_std_arr)) if gain_std_arr.size else 0.0
        residual = np.asarray(
            previous_projection_residual if previous_projection_residual is not None else np.zeros(5, dtype=np.float32),
            dtype=np.float32,
        ).reshape(-1)
        return {
            "min_thermal_headroom": self._min_headroom(thermal_headroom),
            "effective_gain_uncertainty": gain_std,
            "target_critic_disagreement": disagreement,
            "target_constraint_value": constraint_penalty,
            "previous_projection_residual_norm": float(np.linalg.norm(residual)),
        }

    def _adaptive_budget(self, risk: Dict[str, float], *, global_step: int) -> tuple[int, str]:
        if not self.adaptive_budget_enabled:
            return int(self.candidate_count), "fixed_candidate_count"

        high = (
            risk["min_thermal_headroom"] <= self.budget_high_headroom_c
            or risk["effective_gain_uncertainty"] >= self.budget_high_gain_std
            or risk["target_critic_disagreement"] >= self.budget_high_disagreement
            or risk["target_constraint_value"] >= self.budget_high_constraint
            or risk["previous_projection_residual_norm"] >= self.budget_high_projection_residual
        )
        if high:
            return self._nearest_budget(self.budget_high_k), "high_risk"

        medium = (
            risk["min_thermal_headroom"] <= self.budget_medium_headroom_c
            or risk["effective_gain_uncertainty"] >= self.budget_medium_gain_std
            or risk["target_critic_disagreement"] >= self.budget_medium_disagreement
            or risk["target_constraint_value"] >= self.budget_medium_constraint
            or risk["previous_projection_residual_norm"] >= self.budget_medium_projection_residual
        )
        if medium:
            return self._nearest_budget(self.budget_medium_k), "medium_risk"

        if self.budget_low_periodic_interval > 0 and int(global_step) % self.budget_low_periodic_interval == 0:
            return self._nearest_budget(self.budget_low_periodic_k), "low_risk_periodic_verification"
        return self._nearest_budget(0), "very_low_risk_policy_only"

    @staticmethod
    def _executed_action(safe: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([safe["currents_exec"], safe["rho_exec"], safe["tau_exec"]], dim=1)

    def _projection_residual_norm(self, *, lower_raw: torch.Tensor, safe: Dict[str, torch.Tensor], safety) -> torch.Tensor:
        current_max = torch.as_tensor(safety.current_max, dtype=lower_raw.dtype, device=lower_raw.device).view(1, -1)
        requested_current = safe.get("current_requested")
        if requested_current is None:
            decoded = safety._decode_frac_torch(lower_raw[:, :5])
            requested_current = decoded[:, :3] * current_max
            requested_rho_tau = decoded[:, 3:5]
        else:
            requested_rho_tau = torch.cat([safe["rho_raw_decoded"], safe["tau_raw_decoded"]], dim=1)
        desired = torch.cat([requested_current, requested_rho_tau], dim=1)
        executed = self._executed_action(safe)
        residual = executed - desired
        residual_current = residual[:, :3] / torch.clamp(current_max, min=1.0e-6)
        residual_normed = torch.cat([residual_current, residual[:, 3:]], dim=1)
        return torch.linalg.vector_norm(residual_normed, dim=1, keepdim=True)

    def _thermal_risk_terms(
        self,
        *,
        safe: Dict[str, torch.Tensor],
        safety,
        amb_temp: torch.Tensor,
        thermal_horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        margin1 = safe["thermal_margin"]
        risk1 = torch.relu(self.thermal_margin_target_c - margin1).mean(dim=1, keepdim=True)
        if int(thermal_horizon) < 2:
            zeros = torch.zeros_like(risk1)
            max_t1 = safe["t_pred"].max(dim=1, keepdim=True).values
            veto = torch.zeros_like(risk1, dtype=torch.bool)
            return risk1, zeros, zeros, max_t1, veto

        temps2_in = safe["t_pred"]
        base2, _ = safety._thermal_base_torch(temps2_in, amb_temp)
        effective_gain = safety._safe_thermal_coeff_torch(dtype=temps2_in.dtype, device=temps2_in.device)
        currents = safe["currents_exec"]
        t2 = base2 + effective_gain * (currents**2)
        margin2 = safety.thermal_safe - t2
        risk2 = torch.relu(self.thermal_margin_target_c - margin2).mean(dim=1, keepdim=True)
        incremental = torch.relu(risk2 - risk1)
        max_t2 = t2.max(dim=1, keepdim=True).values
        veto = max_t2 > float(safety.thermal_safe - self.h2_margin)
        return risk1, risk2, incremental, max_t2, veto

    def _replacement_margin(self, valid_scores: torch.Tensor) -> float:
        finite = valid_scores[torch.isfinite(valid_scores)]
        if self.replacement_margin_mode == "absolute":
            return float(self.replacement_margin if self.replacement_margin is not None else 1.0e-6)
        factor = float(self.normalized_margin_factor if self.normalized_margin_factor is not None else 0.25)
        if finite.numel() <= 1:
            normalized = 0.0
        else:
            normalized = float(torch.std(finite.detach(), unbiased=False).cpu().item()) * factor
        if self.replacement_margin_mode == "absolute_or_normalized" and self.replacement_margin is not None:
            return max(float(self.replacement_margin), normalized)
        if self.replacement_margin is not None and self.replacement_margin_mode == "normalized":
            return max(float(self.replacement_margin), normalized)
        return max(1.0e-6, normalized)

    def plan(
        self,
        *,
        lower,
        safety,
        obs: np.ndarray,
        z: np.ndarray,
        upper_idx_exec: int,
        boost_combo: int,
        mode: int,
        policy_raw: np.ndarray,
        physical_features: np.ndarray,
        thermal_headroom: np.ndarray | None,
        temps: np.ndarray,
        amb_temp: float,
        meta_iter: int,
        global_step: int = 0,
        previous_projection_residual: np.ndarray | None = None,
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        if not self.enabled:
            return np.asarray(policy_raw, dtype=np.float32), {"residual_planner_enabled": False}

        start = time.perf_counter()
        probe_start = time.perf_counter()
        risk = self._policy_risk_probe(
            lower=lower,
            safety=safety,
            obs=obs,
            z=z,
            upper_idx_exec=upper_idx_exec,
            boost_combo=boost_combo,
            mode=mode,
            policy_raw=policy_raw,
            physical_features=physical_features,
            thermal_headroom=thermal_headroom,
            temps=temps,
            amb_temp=amb_temp,
            previous_projection_residual=previous_projection_residual,
        )
        probe_latency_ms = float((time.perf_counter() - probe_start) * 1000.0)
        budget_k, budget_reason = self._adaptive_budget(risk, global_step=global_step)
        if budget_k <= 0:
            latency_ms = float((time.perf_counter() - start) * 1000.0)
            return np.asarray(policy_raw, dtype=np.float32), {
                "residual_planner_enabled": True,
                "residual_planner_candidate_count": 0,
                "residual_planner_budget": 0,
                "residual_planner_budget_reason": budget_reason,
                "residual_planner_adaptive_budget_enabled": bool(self.adaptive_budget_enabled),
                "residual_planner_latency_ms": latency_ms,
                "residual_planner_probe_latency_ms": probe_latency_ms,
                "residual_planner_candidate_search_latency_ms": 0.0,
                "residual_planner_total_latency_ms": latency_ms,
                "residual_planner_selected_idx": 0,
                "residual_planner_effective_thermal_horizon": 0,
                "residual_planner_score": 0.0,
                "residual_planner_score_improvement": 0.0,
                "residual_planner_best_idx": 0,
                "residual_planner_best_score_improvement": 0.0,
                "residual_planner_replacement_margin": 0.0,
                "residual_planner_replacement_margin_mode": self.replacement_margin_mode,
                "residual_planner_replaced_policy": False,
                "residual_planner_fallback_to_policy": True,
                "residual_planner_candidate_distance": 0.0,
                "residual_planner_candidate_raw_distance": 0.0,
                "residual_planner_candidate_exec_distance": 0.0,
                "residual_planner_trust_region_rejected_count": 0,
                "residual_planner_valid_candidate_count": 1,
                "residual_planner_max_valid_distance": 0.0,
                "residual_planner_margin_rejection_rate": 0.0,
                "residual_planner_constraint_rejection_rate": 0.0,
                "residual_planner_projection_rejection_rate": 0.0,
                "residual_planner_h2_veto_rate": 0.0,
                "residual_planner_fallback_rate": 1.0,
                "residual_planner_replacement_rate": 0.0,
                "residual_planner_reward_value": 0.0,
                "residual_planner_constraint_penalty": risk["target_constraint_value"],
                "residual_planner_disagreement": risk["target_critic_disagreement"],
                "residual_planner_projection_residual": risk["previous_projection_residual_norm"],
                "residual_planner_thermal_risk": 0.0,
                "residual_planner_h1_thermal_risk": 0.0,
                "residual_planner_h2_thermal_risk": 0.0,
                "residual_planner_incremental_h2_risk": 0.0,
                "residual_planner_h2_max_temperature": 0.0,
                "residual_planner_h2_veto": False,
                "residual_planner_trust_region_rejected": False,
                "residual_planner_min_thermal_headroom": risk["min_thermal_headroom"],
                "residual_planner_effective_gain_uncertainty": risk["effective_gain_uncertainty"],
                "residual_planner_target_critic_disagreement": risk["target_critic_disagreement"],
                "residual_planner_target_constraint_value": risk["target_constraint_value"],
                "residual_planner_previous_projection_residual_norm": risk["previous_projection_residual_norm"],
            }

        candidate_search_start = time.perf_counter()
        candidates_np = self._candidate_raw(
            policy_raw,
            boost_combo=int(boost_combo),
            mode=int(mode),
            thermal_headroom=thermal_headroom,
            candidate_count=budget_k,
        )
        k = int(candidates_np.shape[0])
        candidates = torch.as_tensor(candidates_np, dtype=torch.float32, device=self.device)
        boost_t = torch.full((k,), int(boost_combo), dtype=torch.long, device=self.device)
        mode_t = torch.full((k,), int(mode), dtype=torch.long, device=self.device)
        temps_t = torch.as_tensor(np.asarray(temps, dtype=np.float32), dtype=torch.float32, device=self.device).view(1, -1)
        temps_t = temps_t.expand(k, -1)
        amb_t = torch.full((k,), float(amb_temp), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            safe = safety.project_torch(candidates, boost_t, mode_t, temps_t, amb_t)
            executed = self._executed_action(safe)
            obs_aug = lower._augment_np(
                np.asarray(obs, dtype=np.float32),
                upper_idx=int(upper_idx_exec),
                physical_features=np.asarray(physical_features, dtype=np.float32),
                encoder=lower.q1_tgt_phys,
            ).expand(k, -1)
            obs_aug_q2 = lower._augment_np(
                np.asarray(obs, dtype=np.float32),
                upper_idx=int(upper_idx_exec),
                physical_features=np.asarray(physical_features, dtype=np.float32),
                encoder=lower.q2_tgt_phys,
            ).expand(k, -1)
            z_t = torch.as_tensor(np.asarray(z, dtype=np.float32), dtype=torch.float32, device=self.device).view(1, -1)
            z_t = z_t.expand(k, -1)
            q1 = lower.q1_tgt(obs_aug, z_t, executed)
            q2 = lower.q2_tgt(obs_aug_q2, z_t, executed)
            reward_value = torch.minimum(q1, q2)
            disagreement = torch.abs(q1 - q2)
            projection_residual = self._projection_residual_norm(lower_raw=candidates, safe=safe, safety=safety)
            raw_distance = torch.linalg.vector_norm(candidates - candidates[0:1], dim=1, keepdim=True)
            current_max = torch.as_tensor(safety.current_max, dtype=executed.dtype, device=executed.device).view(1, -1)
            exec_delta = executed - executed[0:1]
            exec_delta_normed = torch.cat(
                [exec_delta[:, :3] / torch.clamp(current_max, min=1.0e-6), exec_delta[:, 3:]],
                dim=1,
            )
            exec_distance = torch.linalg.vector_norm(exec_delta_normed, dim=1, keepdim=True)
            effective_thermal_horizon = (
                self.thermal_horizon if int(meta_iter) >= self.thermal_horizon_start_meta_iter else 1
            )
            h1_risk, h2_risk, incremental_h2_risk, h2_max_temperature, h2_veto = self._thermal_risk_terms(
                safe=safe,
                safety=safety,
                amb_temp=amb_t.view(-1, 1),
                thermal_horizon=effective_thermal_horizon,
            )

            constraint_penalty = torch.zeros_like(reward_value)
            if getattr(lower, "constraint_q_tgt", None) is not None:
                constraint_val = lower.constraint_q_tgt(obs_aug, z_t, executed)
                if getattr(lower, "constraint_actor_penalty_nonnegative", True):
                    constraint_val = torch.relu(constraint_val)
                constraint_penalty = (constraint_val * lower.constraint_actor_weights).sum(dim=1, keepdim=True)

            trust_valid = torch.ones_like(reward_value, dtype=torch.bool)
            if self.trust_region_enabled:
                if self.trust_region_raw_l2 is not None and self.trust_region_mode in {"raw", "raw_and_executed"}:
                    trust_valid = trust_valid & (raw_distance <= float(self.trust_region_raw_l2))
                if self.trust_region_exec_l2 is not None and self.trust_region_mode in {"executed", "raw_and_executed"}:
                    trust_valid = trust_valid & (exec_distance <= float(self.trust_region_exec_l2))
            constraint_valid = constraint_penalty <= (constraint_penalty[0:1] + float(self.constraint_tolerance))
            projection_valid = torch.ones_like(reward_value, dtype=torch.bool)
            if self.projection_residual_limit is not None:
                projection_valid = projection_residual <= float(self.projection_residual_limit)
            h2_valid = torch.ones_like(reward_value, dtype=torch.bool)
            if self.h2_veto_enabled and int(effective_thermal_horizon) >= 2:
                h2_valid = ~h2_veto
            valid = trust_valid & constraint_valid & projection_valid & h2_valid
            valid[0] = True

            score = (
                reward_value
                - self.constraint_beta * constraint_penalty
                - self.disagreement_beta * disagreement
                - self.projection_penalty * projection_residual
                - self.h2_increment_beta * incremental_h2_risk
            )
            valid_score = torch.where(valid, score, torch.full_like(score, -torch.inf))
            margin = self._replacement_margin(valid_score.view(-1))
            best_idx = int(torch.argmax(valid_score.view(-1)).item())
            best_improvement = float((score[best_idx] - score[0]).detach().cpu().item())
            selected_idx = best_idx if best_idx != 0 and best_improvement >= margin else 0
            score_improvement = float((score[selected_idx] - score[0]).detach().cpu().item())
            margin_rejected = bool(best_idx != 0 and best_improvement < margin)
            trust_rejected = (
                self.trust_region_enabled
                and (
                    ((self.trust_region_raw_l2 is not None) and bool((raw_distance[1:] > self.trust_region_raw_l2).any().item()))
                    or ((self.trust_region_exec_l2 is not None) and bool((exec_distance[1:] > self.trust_region_exec_l2).any().item()))
                )
            )

        latency_ms = float((time.perf_counter() - start) * 1000.0)
        candidate_search_latency_ms = float((time.perf_counter() - candidate_search_start) * 1000.0)
        diagnostics = {
            "residual_planner_enabled": True,
            "residual_planner_candidate_count": k,
            "residual_planner_budget": int(budget_k),
            "residual_planner_budget_reason": budget_reason,
            "residual_planner_adaptive_budget_enabled": bool(self.adaptive_budget_enabled),
            "residual_planner_effective_thermal_horizon": int(effective_thermal_horizon),
            "residual_planner_selected_idx": selected_idx,
            "residual_planner_latency_ms": latency_ms,
            "residual_planner_probe_latency_ms": probe_latency_ms,
            "residual_planner_candidate_search_latency_ms": candidate_search_latency_ms,
            "residual_planner_total_latency_ms": latency_ms,
            "residual_planner_score": float(score[selected_idx].detach().cpu().item()),
            "residual_planner_score_improvement": score_improvement,
            "residual_planner_best_idx": best_idx,
            "residual_planner_best_score_improvement": best_improvement,
            "residual_planner_replacement_margin": float(margin),
            "residual_planner_replacement_margin_mode": self.replacement_margin_mode,
            "residual_planner_replaced_policy": bool(selected_idx != 0),
            "residual_planner_fallback_to_policy": bool(selected_idx == 0),
            "residual_planner_candidate_distance": float(
                np.linalg.norm(candidates_np[selected_idx].astype(np.float32) - np.asarray(policy_raw, dtype=np.float32))
            ),
            "residual_planner_candidate_raw_distance": float(raw_distance[selected_idx].detach().cpu().item()),
            "residual_planner_candidate_exec_distance": float(exec_distance[selected_idx].detach().cpu().item()),
            "residual_planner_trust_region_rejected_count": int((~trust_valid[1:]).sum().detach().cpu().item())
            if self.trust_region_enabled
            else 0,
            "residual_planner_valid_candidate_count": int(valid.sum().detach().cpu().item()),
            "residual_planner_max_valid_distance": float(
                torch.where(valid, raw_distance, torch.zeros_like(raw_distance)).max().detach().cpu().item()
            ),
            "residual_planner_margin_rejection_rate": float(margin_rejected),
            "residual_planner_constraint_rejection_rate": float((~constraint_valid[1:]).float().mean().detach().cpu().item())
            if k > 1
            else 0.0,
            "residual_planner_projection_rejection_rate": float((~projection_valid[1:]).float().mean().detach().cpu().item())
            if self.projection_residual_limit is not None and k > 1
            else 0.0,
            "residual_planner_h2_veto_rate": float(h2_veto[1:].float().mean().detach().cpu().item()) if k > 1 else 0.0,
            "residual_planner_fallback_rate": float(selected_idx == 0),
            "residual_planner_replacement_rate": float(selected_idx != 0),
            "residual_planner_reward_value": float(reward_value[selected_idx].detach().cpu().item()),
            "residual_planner_constraint_penalty": float(constraint_penalty[selected_idx].detach().cpu().item()),
            "residual_planner_disagreement": float(disagreement[selected_idx].detach().cpu().item()),
            "residual_planner_projection_residual": float(projection_residual[selected_idx].detach().cpu().item()),
            "residual_planner_thermal_risk": float(h1_risk[selected_idx].detach().cpu().item()),
            "residual_planner_h1_thermal_risk": float(h1_risk[selected_idx].detach().cpu().item()),
            "residual_planner_h2_thermal_risk": float(h2_risk[selected_idx].detach().cpu().item()),
            "residual_planner_incremental_h2_risk": float(incremental_h2_risk[selected_idx].detach().cpu().item()),
            "residual_planner_h2_max_temperature": float(h2_max_temperature[selected_idx].detach().cpu().item()),
            "residual_planner_h2_veto": bool(h2_veto[selected_idx].detach().cpu().item()),
            "residual_planner_trust_region_rejected": bool(trust_rejected),
            "residual_planner_min_thermal_headroom": risk["min_thermal_headroom"],
            "residual_planner_effective_gain_uncertainty": risk["effective_gain_uncertainty"],
            "residual_planner_target_critic_disagreement": risk["target_critic_disagreement"],
            "residual_planner_target_constraint_value": risk["target_constraint_value"],
            "residual_planner_previous_projection_residual_norm": risk["previous_projection_residual_norm"],
        }
        return candidates_np[selected_idx].astype(np.float32), diagnostics
