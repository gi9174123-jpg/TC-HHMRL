from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class SupportGateStats:
    reward: float = 0.0
    cost: float = 0.0
    violation_rate: float = 0.0


@dataclass(frozen=True)
class SupportGateDecision:
    accepted: bool
    reason: str
    gate_score: float
    threshold: float
    support_stats_before: SupportGateStats
    support_stats_after: SupportGateStats
    reward_delta: float
    cost_delta: float
    violation_delta: float
    parameter_delta_norm: float
    query_leakage: bool = False
    extra_support_rollouts: int = 0
    extra_gradient_updates: int = 0
    extra_query_evaluations: int = 0


def support_gate_score(stats: SupportGateStats, config: Mapping | None = None) -> float:
    cfg = dict(config or {})
    reward_w = float(cfg.get("reward_weight", 1.0))
    cost_w = float(cfg.get("cost_weight", 1.0))
    violation_w = float(cfg.get("violation_weight", 1.0))
    return float(reward_w * stats.reward - cost_w * stats.cost - violation_w * stats.violation_rate)


def evaluate_support_gate(
    pre_support_stats: SupportGateStats,
    post_support_stats: SupportGateStats,
    parameter_delta: float,
    config: Mapping | None = None,
) -> SupportGateDecision:
    cfg = dict(config or {})
    rule = str(cfg.get("rule", "support_score_non_degradation"))
    reward_norm_eps = float(cfg.get("reward_normalization_eps", 1.0))
    threshold = float(
        cfg.get("normalized_reward_threshold", cfg.get("score_threshold", 0.0))
        if rule == "safety_first"
        else cfg.get("score_threshold", 0.0)
    )
    max_param_delta = float(cfg.get("max_parameter_delta_norm", float("inf")))
    min_param_delta = float(cfg.get("min_parameter_delta_norm", 0.0))
    max_cost_increase = float(cfg.get("max_cost_increase", float("inf")))
    max_violation_increase = float(cfg.get("max_violation_increase", float("inf")))

    before_score = support_gate_score(pre_support_stats, cfg)
    after_score = support_gate_score(post_support_stats, cfg)
    gate_score = float(after_score - before_score)
    reward_delta = float(post_support_stats.reward - pre_support_stats.reward)
    cost_delta = float(post_support_stats.cost - pre_support_stats.cost)
    violation_delta = float(post_support_stats.violation_rate - pre_support_stats.violation_rate)
    parameter_delta = float(parameter_delta)
    normalized_reward_delta = float(
        reward_delta / max(abs(float(pre_support_stats.reward)), reward_norm_eps)
    )
    if rule == "safety_first":
        gate_score = normalized_reward_delta

    finite_values = [
        before_score,
        after_score,
        gate_score,
        reward_delta,
        cost_delta,
        violation_delta,
        parameter_delta,
    ]
    if not all(math.isfinite(v) for v in finite_values):
        return SupportGateDecision(
            accepted=False,
            reason="reject_nonfinite_support_or_parameter_metric",
            gate_score=gate_score,
            threshold=threshold,
            support_stats_before=pre_support_stats,
            support_stats_after=post_support_stats,
            reward_delta=reward_delta,
            cost_delta=cost_delta,
            violation_delta=violation_delta,
            parameter_delta_norm=parameter_delta,
        )
    if rule == "safety_first":
        if violation_delta > max_violation_increase:
            return SupportGateDecision(
                accepted=False,
                reason="reject_support_violation_increase",
                gate_score=gate_score,
                threshold=threshold,
                support_stats_before=pre_support_stats,
                support_stats_after=post_support_stats,
                reward_delta=reward_delta,
                cost_delta=cost_delta,
                violation_delta=violation_delta,
                parameter_delta_norm=parameter_delta,
            )
        if cost_delta > max_cost_increase:
            return SupportGateDecision(
                accepted=False,
                reason="reject_support_cost_increase",
                gate_score=gate_score,
                threshold=threshold,
                support_stats_before=pre_support_stats,
                support_stats_after=post_support_stats,
                reward_delta=reward_delta,
                cost_delta=cost_delta,
                violation_delta=violation_delta,
                parameter_delta_norm=parameter_delta,
            )
        if parameter_delta > max_param_delta:
            return SupportGateDecision(
                accepted=False,
                reason="reject_parameter_delta_too_large",
                gate_score=gate_score,
                threshold=threshold,
                support_stats_before=pre_support_stats,
                support_stats_after=post_support_stats,
                reward_delta=reward_delta,
                cost_delta=cost_delta,
                violation_delta=violation_delta,
                parameter_delta_norm=parameter_delta,
            )
        if parameter_delta < min_param_delta:
            return SupportGateDecision(
                accepted=False,
                reason="reject_parameter_delta_too_small",
                gate_score=gate_score,
                threshold=threshold,
                support_stats_before=pre_support_stats,
                support_stats_after=post_support_stats,
                reward_delta=reward_delta,
                cost_delta=cost_delta,
                violation_delta=violation_delta,
                parameter_delta_norm=parameter_delta,
            )
        accepted = bool(normalized_reward_delta >= threshold)
        return SupportGateDecision(
            accepted=accepted,
            reason="accept_safety_first_support_gate" if accepted else "reject_support_reward_not_improved",
            gate_score=gate_score,
            threshold=threshold,
            support_stats_before=pre_support_stats,
            support_stats_after=post_support_stats,
            reward_delta=reward_delta,
            cost_delta=cost_delta,
            violation_delta=violation_delta,
            parameter_delta_norm=parameter_delta,
        )
    if parameter_delta > max_param_delta:
        return SupportGateDecision(
            accepted=False,
            reason="reject_parameter_delta_too_large",
            gate_score=gate_score,
            threshold=threshold,
            support_stats_before=pre_support_stats,
            support_stats_after=post_support_stats,
            reward_delta=reward_delta,
            cost_delta=cost_delta,
            violation_delta=violation_delta,
            parameter_delta_norm=parameter_delta,
        )
    if parameter_delta < min_param_delta:
        return SupportGateDecision(
            accepted=False,
            reason="reject_parameter_delta_too_small",
            gate_score=gate_score,
            threshold=threshold,
            support_stats_before=pre_support_stats,
            support_stats_after=post_support_stats,
            reward_delta=reward_delta,
            cost_delta=cost_delta,
            violation_delta=violation_delta,
            parameter_delta_norm=parameter_delta,
        )
    if cost_delta > max_cost_increase:
        return SupportGateDecision(
            accepted=False,
            reason="reject_support_cost_increase",
            gate_score=gate_score,
            threshold=threshold,
            support_stats_before=pre_support_stats,
            support_stats_after=post_support_stats,
            reward_delta=reward_delta,
            cost_delta=cost_delta,
            violation_delta=violation_delta,
            parameter_delta_norm=parameter_delta,
        )
    if violation_delta > max_violation_increase:
        return SupportGateDecision(
            accepted=False,
            reason="reject_support_violation_increase",
            gate_score=gate_score,
            threshold=threshold,
            support_stats_before=pre_support_stats,
            support_stats_after=post_support_stats,
            reward_delta=reward_delta,
            cost_delta=cost_delta,
            violation_delta=violation_delta,
            parameter_delta_norm=parameter_delta,
        )

    accepted = bool(gate_score >= threshold)
    return SupportGateDecision(
        accepted=accepted,
        reason="accept_support_score_non_degradation" if accepted else "reject_support_score_degradation",
        gate_score=gate_score,
        threshold=threshold,
        support_stats_before=pre_support_stats,
        support_stats_after=post_support_stats,
        reward_delta=reward_delta,
        cost_delta=cost_delta,
        violation_delta=violation_delta,
        parameter_delta_norm=parameter_delta,
    )
