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
        self.current_step = float(planner_cfg.get("current_step", 0.05))
        self.ratio_step = float(planner_cfg.get("ratio_step", 0.05))
        self.disagreement_beta = float(planner_cfg.get("disagreement_beta", 0.10))
        self.projection_penalty = float(planner_cfg.get("projection_penalty", 0.10))
        self.constraint_beta = float(planner_cfg.get("constraint_beta", 1.0))
        self.thermal_risk_beta = float(planner_cfg.get("thermal_risk_beta", 0.05))
        self.h2_increment_beta = float(planner_cfg.get("h2_increment_beta", self.thermal_risk_beta))
        self.thermal_margin_target_c = float(planner_cfg.get("thermal_margin_target_c", 1.0))
        self.thermal_horizon = int(planner_cfg.get("thermal_horizon", 2))
        self.start_meta_iter = int(planner_cfg.get("start_meta_iter", 60))
        self.thermal_horizon_start_meta_iter = int(planner_cfg.get("thermal_horizon_start_meta_iter", 86))
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
        thermal_headroom: np.ndarray | None,
    ) -> np.ndarray:
        raw = np.asarray(policy_raw, dtype=np.float32).reshape(-1)[:5]
        basis = residual_basis(
            candidate_count=self.candidate_count,
            current_step=self.current_step,
            ratio_step=self.ratio_step,
            thermal_headroom=thermal_headroom,
        )
        candidates = raw.reshape(1, -1) + basis
        return np.clip(candidates, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def _executed_action(safe: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([safe["currents_exec"], safe["rho_exec"], safe["tau_exec"]], dim=1)

    def _projection_residual_norm(self, *, lower_raw: torch.Tensor, safe: Dict[str, torch.Tensor], safety) -> torch.Tensor:
        decoded = safety._decode_frac_torch(lower_raw[:, :5])
        current_max = torch.as_tensor(safety.current_max, dtype=lower_raw.dtype, device=lower_raw.device).view(1, -1)
        desired = torch.cat([decoded[:, :3] * current_max, decoded[:, 3:4], decoded[:, 4:5]], dim=1)
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
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        if not self.enabled:
            return np.asarray(policy_raw, dtype=np.float32), {"residual_planner_enabled": False}

        start = time.perf_counter()
        candidates_np = self._candidate_raw(policy_raw, thermal_headroom=thermal_headroom)
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
        diagnostics = {
            "residual_planner_enabled": True,
            "residual_planner_candidate_count": k,
            "residual_planner_effective_thermal_horizon": int(effective_thermal_horizon),
            "residual_planner_selected_idx": selected_idx,
            "residual_planner_latency_ms": latency_ms,
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
        }
        return candidates_np[selected_idx].astype(np.float32), diagnostics
