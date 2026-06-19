from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.benchmark_constraint_scenarios import (
    apply_ablation,
    apply_common_settings,
    apply_scenario,
    sample_fixed_tasks,
)
from tchhmrl.envs.task_contract import ordered_task_batch_hash, task_batch_hash
from tchhmrl.envs.uw_slipt_env import MultiTxUwSliptEnv
from tchhmrl.meta.meta_trainer import EpisodeStats, MetaTrainer
from tchhmrl.utils.config import load_cfg


ROW_FIELDS = [
    "scenario",
    "seed",
    "variant",
    "task_id",
    "site_id",
    "phase",
    "episode_in_phase",
    "episode_global",
    "episode_seed",
    "reward",
    "se",
    "eh",
    "cost",
    "violation_rate",
    "length",
    "context_enabled",
    "explicit_inner_outer",
    "support_train_adapts",
    "support_gate_enabled",
    "support_gate_validation",
    "support_gate_accepted",
    "support_gate_selected",
    "support_gate_metric",
    "support_gate_score_no_context",
    "support_gate_score_before",
    "support_gate_score_after",
    "support_gate_score_delta",
    "query_eval_after_support",
    "pre_query_eval_before_support",
    "support_global_step_delta",
    "support_lower_update_delta",
    "support_upper_update_delta",
    "support_parameter_delta_norm",
    "support_upper_delta_norm",
    "support_lower_actor_delta_norm",
    "support_lower_critic_delta_norm",
    "support_context_encoder_delta_norm",
    "support_context_predictor_delta_norm",
    "support_lower_replay_len_after_support",
    "support_upper_replay_len_after_support",
    "query_global_step_delta",
    "query_lower_update_delta",
    "query_upper_update_delta",
    "query_parameter_delta_norm",
    "context_history_len_before_query",
    "query_has_support_context",
    "same_task_protocol",
]


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return float(np.mean(vals)) if vals else 0.0


def _state_delta_norm(left: object, right: object) -> float:
    if isinstance(left, dict) and isinstance(right, dict):
        keys = set(left).intersection(right)
        sq_sum = 0.0
        for key in keys:
            delta = _state_delta_norm(left[key], right[key])
            sq_sum += delta * delta
        return float(np.sqrt(sq_sum))
    if torch.is_tensor(left) and torch.is_tensor(right) and torch.is_floating_point(left):
        diff = left.detach().cpu().float() - right.detach().cpu().float()
        return float(torch.linalg.vector_norm(diff).item())
    return 0.0


def _module_delta_summary(base_state: Dict, current_state: Dict) -> Dict[str, float]:
    lower_base = base_state.get("lower", {})
    lower_current = current_state.get("lower", {})
    lower_critic_base = {
        key: lower_base.get(key, {})
        for key in ("q1", "q2", "q1_tgt", "q2_tgt")
    }
    lower_critic_current = {
        key: lower_current.get(key, {})
        for key in ("q1", "q2", "q1_tgt", "q2_tgt")
    }
    return {
        "support_upper_delta_norm": float(
            _state_delta_norm(base_state.get("upper", {}).get("q", {}), current_state.get("upper", {}).get("q", {}))
        ),
        "support_lower_actor_delta_norm": float(
            _state_delta_norm(lower_base.get("actor", {}), lower_current.get("actor", {}))
        ),
        "support_lower_critic_delta_norm": float(_state_delta_norm(lower_critic_base, lower_critic_current)),
        "support_context_encoder_delta_norm": float(
            _state_delta_norm(base_state.get("context_encoder", {}), current_state.get("context_encoder", {}))
        ),
        "support_context_predictor_delta_norm": float(
            _state_delta_norm(base_state.get("context_predictor", {}), current_state.get("context_predictor", {}))
        ),
    }


def _snapshot_context(trainer: MetaTrainer) -> List[Dict]:
    return copy.deepcopy(trainer.agent.episode.as_list())


def _restore_context(trainer: MetaTrainer, context_items: List[Dict]) -> None:
    trainer.agent.episode.clear()
    for item in context_items:
        trainer.agent.episode.add(copy.deepcopy(item))


def _gate_score(stats: EpisodeStats) -> float:
    # Use the same shaped episode reward used by the diagnostic; query rollouts
    # are never consulted by the gate.
    return float(stats.reward)


def _set_episode_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _episode_row(
    *,
    scenario: str,
    seed: int,
    variant: str,
    task_id: int,
    site_id: int,
    phase: str,
    episode_in_phase: int,
    episode_global: int,
    episode_seed: int,
    stats: EpisodeStats,
    context_enabled: bool,
    explicit_inner_outer: bool,
    support_train_adapts: bool,
    support_gate_enabled: bool = False,
    support_gate_validation: bool = False,
    support_gate_accepted: bool = False,
    support_gate_selected: str = "",
    support_gate_metric: str = "",
    support_gate_score_no_context: float = 0.0,
    support_gate_score_before: float = 0.0,
    support_gate_score_after: float = 0.0,
    support_gate_score_delta: float = 0.0,
    support_global_step_delta: int = 0,
    support_lower_update_delta: int = 0,
    support_upper_update_delta: int = 0,
    support_parameter_delta_norm: float = 0.0,
    support_upper_delta_norm: float = 0.0,
    support_lower_actor_delta_norm: float = 0.0,
    support_lower_critic_delta_norm: float = 0.0,
    support_context_encoder_delta_norm: float = 0.0,
    support_context_predictor_delta_norm: float = 0.0,
    support_lower_replay_len_after_support: int = 0,
    support_upper_replay_len_after_support: int = 0,
    query_global_step_delta: int = 0,
    query_lower_update_delta: int = 0,
    query_upper_update_delta: int = 0,
    query_parameter_delta_norm: float = 0.0,
    context_history_len_before_query: int = 0,
    query_has_support_context: bool = False,
) -> Dict[str, object]:
    return {
        "scenario": scenario,
        "seed": int(seed),
        "variant": variant,
        "task_id": int(task_id),
        "site_id": int(site_id),
        "phase": phase,
        "episode_in_phase": int(episode_in_phase),
        "episode_global": int(episode_global),
        "episode_seed": int(episode_seed),
        "reward": float(stats.reward),
        "se": float(stats.se),
        "eh": float(stats.eh),
        "cost": float(stats.cost),
        "violation_rate": float(stats.violations),
        "length": int(stats.length),
        "context_enabled": bool(context_enabled),
        "explicit_inner_outer": bool(explicit_inner_outer),
        "support_train_adapts": bool(support_train_adapts),
        "support_gate_enabled": bool(support_gate_enabled),
        "support_gate_validation": bool(support_gate_validation),
        "support_gate_accepted": bool(support_gate_accepted),
        "support_gate_selected": str(support_gate_selected),
        "support_gate_metric": str(support_gate_metric),
        "support_gate_score_no_context": float(support_gate_score_no_context),
        "support_gate_score_before": float(support_gate_score_before),
        "support_gate_score_after": float(support_gate_score_after),
        "support_gate_score_delta": float(support_gate_score_delta),
        "query_eval_after_support": phase == "query",
        "pre_query_eval_before_support": phase == "pre_query",
        "support_global_step_delta": int(support_global_step_delta),
        "support_lower_update_delta": int(support_lower_update_delta),
        "support_upper_update_delta": int(support_upper_update_delta),
        "support_parameter_delta_norm": float(support_parameter_delta_norm),
        "support_upper_delta_norm": float(support_upper_delta_norm),
        "support_lower_actor_delta_norm": float(support_lower_actor_delta_norm),
        "support_lower_critic_delta_norm": float(support_lower_critic_delta_norm),
        "support_context_encoder_delta_norm": float(support_context_encoder_delta_norm),
        "support_context_predictor_delta_norm": float(support_context_predictor_delta_norm),
        "support_lower_replay_len_after_support": int(support_lower_replay_len_after_support),
        "support_upper_replay_len_after_support": int(support_upper_replay_len_after_support),
        "query_global_step_delta": int(query_global_step_delta),
        "query_lower_update_delta": int(query_lower_update_delta),
        "query_upper_update_delta": int(query_upper_update_delta),
        "query_parameter_delta_norm": float(query_parameter_delta_norm),
        "context_history_len_before_query": int(context_history_len_before_query),
        "query_has_support_context": bool(query_has_support_context),
        "same_task_protocol": True,
    }


def _prepare_cfg(
    base_cfg: Dict,
    *,
    out_dir: Path,
    run_name: str,
    seed: int,
    scenario: str,
    train_iters: int,
    fast_mode: bool,
    support_episodes: int,
    query_episodes: int,
    episode_len: int | None,
    device: str,
    ablation: str | None = None,
) -> Dict:
    cfg = apply_common_settings(
        base_cfg,
        meta_iters=max(0, int(train_iters)),
        out_dir=out_dir,
        run_name=run_name,
        seed=seed,
        fast_mode=fast_mode,
        use_curriculum=False,
    )
    apply_scenario(cfg, scenario)
    cfg["experiment"]["device"] = str(device)
    if episode_len is not None:
        cfg["env"]["episode_len"] = int(episode_len)

    cfg["meta"]["support_episodes"] = int(support_episodes)
    cfg["meta"]["query_episodes"] = int(query_episodes)
    if ablation:
        apply_ablation(cfg, ablation)
    return cfg


def _collect_variant_rows(
    *,
    trainer: MetaTrainer,
    cfg: Dict,
    tasks: List[object],
    scenario: str,
    seed: int,
    variant: str,
    support_episodes: int,
    query_episodes: int,
    pre_query_episodes: int,
    support_train_override: bool | None = None,
    support_gate_enabled: bool = False,
    support_gate_validation_episodes: int = 1,
) -> List[Dict[str, object]]:
    base_state = trainer.agent.snapshot_train_state()
    base_dual_state = copy.deepcopy(trainer.dual.state_dict())
    base_global_step = int(trainer.agent.global_step)
    base_upper_steps = int(trainer.agent.upper.update_steps)
    base_lower_steps = int(trainer.agent.lower.update_steps)
    context_enabled = bool(cfg.get("context", {}).get("enabled", True))
    explicit_inner_outer = bool(cfg.get("meta", {}).get("explicit_inner_outer", False))
    default_support_train_adapts = bool(context_enabled and explicit_inner_outer)
    support_train_adapts = (
        bool(default_support_train_adapts)
        if support_train_override is None
        else bool(support_train_override and default_support_train_adapts)
    )

    rows: List[Dict[str, object]] = []
    for task_id, task in enumerate(tasks):
        trainer.agent.restore_train_state(base_state)
        trainer.agent.global_step = base_global_step
        trainer.agent.upper.update_steps = base_upper_steps
        trainer.agent.lower.update_steps = base_lower_steps
        trainer.dual.load_state_dict(copy.deepcopy(base_dual_state))
        trainer.agent.clear_learning_buffers()

        site_id = int(getattr(task, "site_id", -1))
        support_stats: List[EpisodeStats] = []
        episode_global = 0
        task_row_start = len(rows)

        for ep_idx in range(int(pre_query_episodes)):
            episode_seed = int(seed + task_id * 10_000 + ep_idx)
            _set_episode_seed(episode_seed)
            env = MultiTxUwSliptEnv(cfg, overrides=task.to_env_overrides())
            env.rng = np.random.default_rng(episode_seed)
            stats = trainer._run_episode(env, train=False, clear_context=(ep_idx == 0))
            rows.append(
                _episode_row(
                    scenario=scenario,
                    seed=seed,
                    variant=variant,
                    task_id=task_id,
                    site_id=site_id,
                    phase="pre_query",
                    episode_in_phase=ep_idx,
                    episode_global=episode_global,
                    episode_seed=episode_seed,
                    stats=stats,
                    context_enabled=context_enabled,
                    explicit_inner_outer=explicit_inner_outer,
                    support_train_adapts=support_train_adapts,
                )
            )
            episode_global += 1

        trainer.agent.reset_rollout_state(clear_context=True)
        task_row_start = len(rows)

        gate_active = bool(support_gate_enabled and support_train_adapts and int(support_episodes) >= 2)
        gate_validation_episodes = int(np.clip(int(support_gate_validation_episodes), 1, max(1, int(support_episodes) - 1)))
        n_adapt_support = int(support_episodes) - gate_validation_episodes if gate_active else int(support_episodes)
        support_context_snapshot: List[Dict] = []
        adapted_state: Dict | None = None
        adapted_dual_state: Dict | None = None
        gate_stats_no_context: EpisodeStats | None = None
        gate_stats_context_only: EpisodeStats | None = None
        gate_stats_after: EpisodeStats | None = None
        gate_accepted = False
        gate_selected = ""
        gate_score_no_context = 0.0
        gate_score_before = 0.0
        gate_score_after = 0.0
        gate_score_delta = 0.0

        for ep_idx in range(n_adapt_support):
            episode_seed = int(seed + task_id * 10_000 + 1_000 + ep_idx)
            _set_episode_seed(episode_seed)
            env = MultiTxUwSliptEnv(cfg, overrides=task.to_env_overrides())
            env.rng = np.random.default_rng(episode_seed)
            stats = trainer._run_episode(
                env,
                train=support_train_adapts,
                clear_context=(ep_idx == 0),
            )
            support_stats.append(stats)
            rows.append(
                _episode_row(
                    scenario=scenario,
                    seed=seed,
                    variant=variant,
                    task_id=task_id,
                    site_id=site_id,
                    phase="support",
                    episode_in_phase=ep_idx,
                    episode_global=episode_global,
                    episode_seed=episode_seed,
                    stats=stats,
                    context_enabled=context_enabled,
                    explicit_inner_outer=explicit_inner_outer,
                    support_train_adapts=support_train_adapts,
                )
            )
            episode_global += 1

        if gate_active:
            support_context_snapshot = _snapshot_context(trainer)
            adapted_state = trainer.agent.snapshot_train_state()
            adapted_dual_state = copy.deepcopy(trainer.dual.state_dict())
            validation_seeds = [
                int(seed + task_id * 10_000 + 1_000 + n_adapt_support + val_idx)
                for val_idx in range(gate_validation_episodes)
            ]

            def _score_candidate(
                *,
                train_state: Dict,
                dual_state: Dict,
                context_items: List[Dict],
                clear_context: bool,
            ) -> float:
                scores: List[float] = []
                for validation_seed in validation_seeds:
                    _set_episode_seed(validation_seed)
                    trainer.agent.restore_train_state(train_state)
                    trainer.dual.load_state_dict(copy.deepcopy(dual_state))
                    if clear_context:
                        trainer.agent.episode.clear()
                    else:
                        _restore_context(trainer, context_items)
                    gate_env = MultiTxUwSliptEnv(cfg, overrides=task.to_env_overrides())
                    gate_env.rng = np.random.default_rng(validation_seed)
                    stats_val = trainer._run_episode(gate_env, train=False, clear_context=clear_context)
                    scores.append(_gate_score(stats_val))
                return _mean(scores)

            gate_score_no_context = _score_candidate(
                train_state=base_state,
                dual_state=base_dual_state,
                context_items=[],
                clear_context=True,
            )
            gate_score_before = _score_candidate(
                train_state=base_state,
                dual_state=base_dual_state,
                context_items=support_context_snapshot,
                clear_context=False,
            )
            gate_score_after = _score_candidate(
                train_state=adapted_state,
                dual_state=adapted_dual_state,
                context_items=support_context_snapshot,
                clear_context=False,
            )
            candidates = {
                "no_support": gate_score_no_context,
                "context_only": gate_score_before,
                "adapted": gate_score_after,
            }
            if gate_score_after >= gate_score_no_context and gate_score_after >= gate_score_before:
                gate_selected = "adapted"
            elif gate_score_before >= gate_score_no_context:
                gate_selected = "context_only"
            else:
                gate_selected = "no_support"
            selected_score = float(candidates[gate_selected])
            gate_score_delta = float(selected_score - gate_score_no_context)
            gate_accepted = gate_selected == "adapted"

            if gate_selected == "adapted":
                trainer.agent.restore_train_state(adapted_state)
                trainer.dual.load_state_dict(copy.deepcopy(adapted_dual_state))
                _restore_context(trainer, support_context_snapshot)
                validation_clear_context = False
            elif gate_selected == "context_only":
                trainer.agent.restore_train_state(base_state)
                trainer.agent.upper.update_steps = base_upper_steps
                trainer.agent.lower.update_steps = base_lower_steps
                trainer.dual.load_state_dict(copy.deepcopy(base_dual_state))
                _restore_context(trainer, support_context_snapshot)
                validation_clear_context = False
            else:
                trainer.agent.restore_train_state(base_state)
                trainer.agent.upper.update_steps = base_upper_steps
                trainer.agent.lower.update_steps = base_lower_steps
                trainer.dual.load_state_dict(copy.deepcopy(base_dual_state))
                trainer.agent.episode.clear()
                validation_clear_context = True

            for val_idx, validation_seed in enumerate(validation_seeds):
                _set_episode_seed(validation_seed)
                gate_env = MultiTxUwSliptEnv(cfg, overrides=task.to_env_overrides())
                gate_env.rng = np.random.default_rng(validation_seed)
                validation_stats = trainer._run_episode(
                    gate_env,
                    train=False,
                    clear_context=validation_clear_context and val_idx == 0,
                )
                if gate_selected == "no_support":
                    trainer.agent.episode.clear()
                support_stats.append(validation_stats)
                rows.append(
                    _episode_row(
                        scenario=scenario,
                        seed=seed,
                        variant=variant,
                        task_id=task_id,
                        site_id=site_id,
                        phase="support",
                        episode_in_phase=n_adapt_support + val_idx,
                        episode_global=episode_global,
                        episode_seed=validation_seed,
                        stats=validation_stats,
                        context_enabled=context_enabled,
                        explicit_inner_outer=explicit_inner_outer,
                        support_train_adapts=False,
                        support_gate_enabled=True,
                        support_gate_validation=True,
                        support_gate_accepted=gate_accepted,
                        support_gate_selected=gate_selected,
                        support_gate_metric="support_validation_reward",
                        support_gate_score_no_context=gate_score_no_context,
                        support_gate_score_before=gate_score_before,
                        support_gate_score_after=selected_score,
                        support_gate_score_delta=gate_score_delta,
                    )
                )
                episode_global += 1

        if support_train_adapts and trainer.dual_enabled and support_stats and (not gate_active or gate_accepted):
            mean_cost_vec = np.mean(np.stack([s.cost_vec for s in support_stats], axis=0), axis=0)
            trainer.dual.update(mean_cost_vec)

        current_state = trainer.agent.snapshot_train_state()
        support_diag = {
            "support_gate_enabled": gate_active,
            "support_gate_accepted": gate_accepted,
            "support_gate_selected": gate_selected,
            "support_gate_metric": "support_validation_reward" if gate_active else "",
            "support_gate_score_no_context": gate_score_no_context,
            "support_gate_score_before": gate_score_before,
            "support_gate_score_after": float(gate_score_no_context + gate_score_delta) if gate_active else gate_score_after,
            "support_gate_score_delta": gate_score_delta,
            "support_global_step_delta": int(trainer.agent.global_step - base_global_step),
            "support_lower_update_delta": int(trainer.agent.lower.update_steps - base_lower_steps),
            "support_upper_update_delta": int(trainer.agent.upper.update_steps - base_upper_steps),
            "support_parameter_delta_norm": float(MetaTrainer._trainable_parameter_delta_norm(base_state, current_state)),
            **_module_delta_summary(base_state, current_state),
            "support_lower_replay_len_after_support": int(len(trainer.agent.replay)),
            "support_upper_replay_len_after_support": int(len(trainer.agent.upper_replay)),
            "context_history_len_before_query": int(len(trainer.agent.episode)) if context_enabled else 0,
            "query_has_support_context": bool(context_enabled and support_episodes > 0 and len(trainer.agent.episode) > 0),
        }
        for row in rows[task_row_start:]:
            row.update(support_diag)
            if bool(row.get("support_gate_validation", False)):
                row["support_gate_enabled"] = True
                row["support_gate_accepted"] = gate_accepted
                row["support_gate_selected"] = gate_selected
                row["support_gate_metric"] = "support_validation_reward"
                row["support_gate_score_no_context"] = gate_score_no_context
                row["support_gate_score_before"] = gate_score_before
                row["support_gate_score_after"] = float(gate_score_no_context + gate_score_delta)
                row["support_gate_score_delta"] = gate_score_delta

        query_base_state = trainer.agent.snapshot_train_state()
        query_base_global_step = int(trainer.agent.global_step)
        query_base_upper_steps = int(trainer.agent.upper.update_steps)
        query_base_lower_steps = int(trainer.agent.lower.update_steps)
        query_row_start = len(rows)
        for ep_idx in range(int(query_episodes)):
            episode_seed = int(seed + task_id * 10_000 + ep_idx)
            _set_episode_seed(episode_seed)
            env = MultiTxUwSliptEnv(cfg, overrides=task.to_env_overrides())
            env.rng = np.random.default_rng(episode_seed)
            stats = trainer._run_episode(env, train=False, clear_context=(support_episodes <= 0 and ep_idx == 0))
            row = _episode_row(
                scenario=scenario,
                seed=seed,
                variant=variant,
                task_id=task_id,
                site_id=site_id,
                phase="query",
                episode_in_phase=ep_idx,
                episode_global=episode_global,
                episode_seed=episode_seed,
                stats=stats,
                context_enabled=context_enabled,
                explicit_inner_outer=explicit_inner_outer,
                support_train_adapts=support_train_adapts,
            )
            row.update(support_diag)
            rows.append(row)
            episode_global += 1
        query_state = trainer.agent.snapshot_train_state()
        query_diag = {
            "query_global_step_delta": int(trainer.agent.global_step - query_base_global_step),
            "query_lower_update_delta": int(trainer.agent.lower.update_steps - query_base_lower_steps),
            "query_upper_update_delta": int(trainer.agent.upper.update_steps - query_base_upper_steps),
            "query_parameter_delta_norm": float(MetaTrainer._trainable_parameter_delta_norm(query_base_state, query_state)),
        }
        for row in rows[query_row_start:]:
            row.update(query_diag)

    trainer.agent.restore_train_state(base_state)
    trainer.dual.load_state_dict(copy.deepcopy(base_dual_state))
    return rows


def _summarize_rows(rows: List[Dict[str, object]]) -> Dict[str, object]:
    grouped: Dict[tuple, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], row["phase"])].append(row)

    phase_summary: Dict[str, Dict[str, float]] = {}
    for (variant, phase), phase_rows in sorted(grouped.items()):
        key = f"{variant}_{phase}"
        phase_summary[key] = {
            "reward": _mean(row["reward"] for row in phase_rows),
            "se": _mean(row["se"] for row in phase_rows),
            "eh": _mean(row["eh"] for row in phase_rows),
            "cost": _mean(row["cost"] for row in phase_rows),
            "violation_rate": _mean(row["violation_rate"] for row in phase_rows),
        }

    meta_query = phase_summary.get("hybrid_meta_query", {})
    gated_query = phase_summary.get("hybrid_meta_support_gated_query", {})
    meta_no_adapt_query = phase_summary.get("hybrid_meta_no_support_adapt_query", {})
    context_only_query = phase_summary.get("hybrid_context_only_query", {})
    wo_query = phase_summary.get("hybrid_wo_meta_query", {})
    meta_pre_query = phase_summary.get("hybrid_meta_pre_query", {})
    gated_pre_query = phase_summary.get("hybrid_meta_support_gated_pre_query", {})
    meta_no_adapt_pre_query = phase_summary.get("hybrid_meta_no_support_adapt_pre_query", {})
    context_only_pre_query = phase_summary.get("hybrid_context_only_pre_query", {})
    wo_pre_query = phase_summary.get("hybrid_wo_meta_pre_query", {})
    meta_query_reward_gain = float(meta_query.get("reward", 0.0) - meta_pre_query.get("reward", 0.0))
    gated_query_reward_gain = float(gated_query.get("reward", 0.0) - gated_pre_query.get("reward", 0.0))
    meta_no_adapt_query_reward_gain = float(
        meta_no_adapt_query.get("reward", 0.0) - meta_no_adapt_pre_query.get("reward", 0.0)
    )
    context_only_query_reward_gain = float(
        context_only_query.get("reward", 0.0) - context_only_pre_query.get("reward", 0.0)
    )
    wo_query_reward_gain = float(wo_query.get("reward", 0.0) - wo_pre_query.get("reward", 0.0))
    comparison = {
        "query_reward_delta_meta_minus_wo_meta": float(meta_query.get("reward", 0.0) - wo_query.get("reward", 0.0)),
        "query_reward_delta_meta_minus_no_support_adapt": float(
            meta_query.get("reward", 0.0) - meta_no_adapt_query.get("reward", 0.0)
        ),
        "query_reward_delta_gated_minus_meta": float(gated_query.get("reward", 0.0) - meta_query.get("reward", 0.0)),
        "query_reward_delta_gated_minus_no_support_adapt": float(
            gated_query.get("reward", 0.0) - meta_no_adapt_query.get("reward", 0.0)
        ),
        "query_reward_delta_gated_minus_context_only": float(
            gated_query.get("reward", 0.0) - context_only_query.get("reward", 0.0)
        ),
        "query_reward_delta_gated_minus_wo_meta": float(gated_query.get("reward", 0.0) - wo_query.get("reward", 0.0)),
        "query_reward_delta_meta_minus_context_only": float(
            meta_query.get("reward", 0.0) - context_only_query.get("reward", 0.0)
        ),
        "query_reward_delta_context_only_minus_wo_meta": float(
            context_only_query.get("reward", 0.0) - wo_query.get("reward", 0.0)
        ),
        "query_violation_delta_meta_minus_wo_meta": float(
            meta_query.get("violation_rate", 0.0) - wo_query.get("violation_rate", 0.0)
        ),
        "query_violation_delta_meta_minus_no_support_adapt": float(
            meta_query.get("violation_rate", 0.0) - meta_no_adapt_query.get("violation_rate", 0.0)
        ),
        "query_violation_delta_gated_minus_meta": float(
            gated_query.get("violation_rate", 0.0) - meta_query.get("violation_rate", 0.0)
        ),
        "query_violation_delta_gated_minus_context_only": float(
            gated_query.get("violation_rate", 0.0) - context_only_query.get("violation_rate", 0.0)
        ),
        "query_violation_delta_gated_minus_wo_meta": float(
            gated_query.get("violation_rate", 0.0) - wo_query.get("violation_rate", 0.0)
        ),
        "query_violation_delta_meta_minus_context_only": float(
            meta_query.get("violation_rate", 0.0) - context_only_query.get("violation_rate", 0.0)
        ),
        "query_violation_delta_context_only_minus_wo_meta": float(
            context_only_query.get("violation_rate", 0.0) - wo_query.get("violation_rate", 0.0)
        ),
        "query_cost_delta_meta_minus_wo_meta": float(meta_query.get("cost", 0.0) - wo_query.get("cost", 0.0)),
        "query_cost_delta_meta_minus_no_support_adapt": float(
            meta_query.get("cost", 0.0) - meta_no_adapt_query.get("cost", 0.0)
        ),
        "query_cost_delta_gated_minus_meta": float(gated_query.get("cost", 0.0) - meta_query.get("cost", 0.0)),
        "query_cost_delta_meta_minus_context_only": float(
            meta_query.get("cost", 0.0) - context_only_query.get("cost", 0.0)
        ),
        "meta_query_reward_after_minus_before_support": meta_query_reward_gain,
        "gated_query_reward_after_minus_before_support": gated_query_reward_gain,
        "meta_no_support_adapt_query_reward_after_minus_before_support": meta_no_adapt_query_reward_gain,
        "context_only_query_reward_after_minus_before_support": context_only_query_reward_gain,
        "meta_query_se_after_minus_before_support": float(meta_query.get("se", 0.0) - meta_pre_query.get("se", 0.0)),
        "meta_query_eh_after_minus_before_support": float(meta_query.get("eh", 0.0) - meta_pre_query.get("eh", 0.0)),
        "meta_query_cost_after_minus_before_support": float(meta_query.get("cost", 0.0) - meta_pre_query.get("cost", 0.0)),
        "meta_query_violation_after_minus_before_support": float(
            meta_query.get("violation_rate", 0.0) - meta_pre_query.get("violation_rate", 0.0)
        ),
        "wo_meta_query_reward_after_minus_before_support": wo_query_reward_gain,
        "wo_meta_query_violation_after_minus_before_support": float(
            wo_query.get("violation_rate", 0.0) - wo_pre_query.get("violation_rate", 0.0)
        ),
        "few_shot_reward_gain_meta_minus_wo_meta": meta_query_reward_gain - wo_query_reward_gain,
        "few_shot_reward_gain_gated_minus_wo_meta": gated_query_reward_gain - wo_query_reward_gain,
        "few_shot_reward_gain_gated_minus_meta": gated_query_reward_gain - meta_query_reward_gain,
        "few_shot_reward_gain_meta_minus_no_support_adapt": (
            meta_query_reward_gain - meta_no_adapt_query_reward_gain
        ),
        "few_shot_reward_gain_meta_minus_context_only": meta_query_reward_gain - context_only_query_reward_gain,
        "few_shot_reward_gain_context_only_minus_wo_meta": context_only_query_reward_gain - wo_query_reward_gain,
    }
    adaptation_summary: Dict[str, Dict[str, float]] = {}
    for variant in sorted({str(row["variant"]) for row in rows}):
        variant_rows = [row for row in rows if row["variant"] == variant and row["phase"] == "query"]
        if not variant_rows:
            variant_rows = [row for row in rows if row["variant"] == variant]
        support_metrics = phase_summary.get(f"{variant}_support", {})
        pre_query_metrics = phase_summary.get(f"{variant}_pre_query", {})
        query_metrics = phase_summary.get(f"{variant}_query", {})
        adaptation_summary[variant] = {
            "support_reward_mean": float(support_metrics.get("reward", 0.0)),
            "query_reward_before_support": float(pre_query_metrics.get("reward", 0.0)),
            "query_reward_after_support": float(query_metrics.get("reward", 0.0)),
            "query_reward_after_minus_before_support": float(
                query_metrics.get("reward", 0.0) - pre_query_metrics.get("reward", 0.0)
            ),
            "query_violation_before_support": float(pre_query_metrics.get("violation_rate", 0.0)),
            "query_violation_after_support": float(query_metrics.get("violation_rate", 0.0)),
            "query_violation_after_minus_before_support": float(
                query_metrics.get("violation_rate", 0.0) - pre_query_metrics.get("violation_rate", 0.0)
            ),
            "support_global_step_delta": _mean(row["support_global_step_delta"] for row in variant_rows),
            "support_lower_update_delta": _mean(row["support_lower_update_delta"] for row in variant_rows),
            "support_upper_update_delta": _mean(row["support_upper_update_delta"] for row in variant_rows),
            "support_parameter_delta_norm": _mean(row["support_parameter_delta_norm"] for row in variant_rows),
            "support_upper_delta_norm": _mean(row["support_upper_delta_norm"] for row in variant_rows),
            "support_lower_actor_delta_norm": _mean(row["support_lower_actor_delta_norm"] for row in variant_rows),
            "support_lower_critic_delta_norm": _mean(row["support_lower_critic_delta_norm"] for row in variant_rows),
            "support_context_encoder_delta_norm": _mean(
                row["support_context_encoder_delta_norm"] for row in variant_rows
            ),
            "support_context_predictor_delta_norm": _mean(
                row["support_context_predictor_delta_norm"] for row in variant_rows
            ),
            "support_lower_replay_len_after_support": _mean(
                row["support_lower_replay_len_after_support"] for row in variant_rows
            ),
            "support_upper_replay_len_after_support": _mean(
                row["support_upper_replay_len_after_support"] for row in variant_rows
            ),
            "query_global_step_delta": _mean(row["query_global_step_delta"] for row in variant_rows),
            "query_lower_update_delta": _mean(row["query_lower_update_delta"] for row in variant_rows),
            "query_upper_update_delta": _mean(row["query_upper_update_delta"] for row in variant_rows),
            "query_parameter_delta_norm": _mean(row["query_parameter_delta_norm"] for row in variant_rows),
            "context_history_len_before_query": _mean(row["context_history_len_before_query"] for row in variant_rows),
            "query_has_support_context_fraction": _mean(float(row["query_has_support_context"]) for row in variant_rows),
            "support_gate_enabled_fraction": _mean(float(row["support_gate_enabled"]) for row in variant_rows),
            "support_gate_accept_rate": _mean(float(row["support_gate_accepted"]) for row in variant_rows),
            "support_gate_context_only_rate": _mean(
                float(row.get("support_gate_selected", "") == "context_only") for row in variant_rows
            ),
            "support_gate_no_support_rate": _mean(
                float(row.get("support_gate_selected", "") == "no_support") for row in variant_rows
            ),
            "support_gate_adapted_rate": _mean(
                float(row.get("support_gate_selected", "") == "adapted") for row in variant_rows
            ),
            "support_gate_score_no_context": _mean(row["support_gate_score_no_context"] for row in variant_rows),
            "support_gate_score_before": _mean(row["support_gate_score_before"] for row in variant_rows),
            "support_gate_score_after": _mean(row["support_gate_score_after"] for row in variant_rows),
            "support_gate_score_delta": _mean(row["support_gate_score_delta"] for row in variant_rows),
        }
    return {
        "phase_summary": phase_summary,
        "comparison": comparison,
        "adaptation_summary": adaptation_summary,
        "diagnostic_contract": {
            "hybrid_meta": {
                "context_enabled": True,
                "explicit_inner_outer": True,
                "pre_query_eval_before_support": True,
                "support_train_adapts": True,
                "query_eval_after_support": True,
                "query_train_updates": False,
                "matched_pre_post_eval_episode_seeds": True,
            },
            "hybrid_meta_no_support_adapt": {
                "context_enabled": True,
                "explicit_inner_outer": True,
                "pre_query_eval_before_support": True,
                "support_train_adapts": False,
                "query_eval_after_support": True,
                "query_train_updates": False,
                "matched_pre_post_eval_episode_seeds": True,
                "same_checkpoint_as": "hybrid_meta",
            },
            "hybrid_meta_support_gated": {
                "context_enabled": True,
                "explicit_inner_outer": True,
                "pre_query_eval_before_support": True,
                "support_train_adapts": True,
                "support_gate_enabled": True,
                "support_gate_rule": "select_best_of_no_support_context_only_adapted_using_support_validation_reward",
                "query_used_by_gate": False,
                "query_eval_after_support": True,
                "query_train_updates": False,
                "matched_pre_post_eval_episode_seeds": True,
                "same_checkpoint_as": "hybrid_meta",
            },
            "hybrid_context_only": {
                "context_enabled": True,
                "explicit_inner_outer": False,
                "pre_query_eval_before_support": True,
                "support_train_adapts": False,
                "query_eval_after_support": True,
                "query_train_updates": False,
                "matched_pre_post_eval_episode_seeds": True,
                "purpose": "isolate GRU support-context conditioning without Reptile/outer meta update",
            },
            "hybrid_wo_meta": {
                "context_enabled": False,
                "explicit_inner_outer": False,
                "pre_query_eval_before_support": True,
                "support_train_adapts": False,
                "query_eval_after_support": True,
                "query_train_updates": False,
                "matched_pre_post_eval_episode_seeds": True,
            },
        },
    }


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _plot_metric(rows: List[Dict[str, object]], out_path: Path, metric: str, ylabel: str) -> None:
    grouped: Dict[tuple, List[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], int(row["episode_global"]))].append(float(row[metric]))

    by_variant: Dict[str, Dict[int, float]] = defaultdict(dict)
    for (variant, episode_global), vals in grouped.items():
        by_variant[variant][episode_global] = float(np.mean(vals))

    plt.figure(figsize=(7.0, 4.0))
    for variant in sorted(by_variant):
        xs = sorted(by_variant[variant])
        ys = [by_variant[variant][x] for x in xs]
        plt.plot(xs, ys, marker="o", label=variant)
    plt.xlabel("episode within task")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def run_meta_adaptation_diagnostics(
    *,
    cfg_path: str | Path = "configs/default.yaml",
    out_dir: str | Path,
    scenario: str = "hard_stress",
    seed: int = 101,
    train_iters: int = 1,
    n_tasks: int = 3,
    support_episodes: int = 5,
    query_episodes: int = 2,
    pre_query_episodes: int | None = None,
    episode_len: int | None = None,
    fast_mode: bool = False,
    device: str = "cpu",
    make_plots: bool = True,
) -> Dict[str, object]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    base_cfg = load_cfg(cfg_path)
    pre_query_episodes = int(query_episodes if pre_query_episodes is None else pre_query_episodes)

    task_cfg = _prepare_cfg(
        base_cfg,
        out_dir=out_path,
        run_name="meta_adaptation_task_bank",
        seed=seed,
        scenario=scenario,
        train_iters=train_iters,
        fast_mode=fast_mode,
        support_episodes=support_episodes,
        query_episodes=query_episodes,
        episode_len=episode_len,
        device=device,
    )
    tasks = sample_fixed_tasks(task_cfg, seed=seed, n_tasks=n_tasks, seed_offset=21_000)
    fixed_task_batch_hash = task_batch_hash(tasks)
    ordered_fixed_task_batch_hash = ordered_task_batch_hash(tasks)

    rows: List[Dict[str, object]] = []
    meta_cfg = _prepare_cfg(
        base_cfg,
        out_dir=out_path,
        run_name="meta_adaptation_hybrid_meta",
        seed=seed,
        scenario=scenario,
        train_iters=train_iters,
        fast_mode=fast_mode,
        support_episodes=support_episodes,
        query_episodes=query_episodes,
        episode_len=episode_len,
        device=device,
    )
    meta_trainer = MetaTrainer(meta_cfg)
    if int(train_iters) > 0:
        meta_trainer.train(meta_iters=int(train_iters))
    rows.extend(
        _collect_variant_rows(
            trainer=meta_trainer,
            cfg=meta_cfg,
            tasks=tasks,
            scenario=scenario,
            seed=seed,
            variant="hybrid_meta",
            support_episodes=support_episodes,
            query_episodes=query_episodes,
            pre_query_episodes=pre_query_episodes,
            support_train_override=True,
        )
    )
    rows.extend(
        _collect_variant_rows(
            trainer=meta_trainer,
            cfg=meta_cfg,
            tasks=tasks,
            scenario=scenario,
            seed=seed,
            variant="hybrid_meta_no_support_adapt",
            support_episodes=support_episodes,
            query_episodes=query_episodes,
            pre_query_episodes=pre_query_episodes,
            support_train_override=False,
        )
    )
    rows.extend(
        _collect_variant_rows(
            trainer=meta_trainer,
            cfg=meta_cfg,
            tasks=tasks,
            scenario=scenario,
            seed=seed,
            variant="hybrid_meta_support_gated",
            support_episodes=support_episodes,
            query_episodes=query_episodes,
            pre_query_episodes=pre_query_episodes,
            support_train_override=True,
            support_gate_enabled=True,
            support_gate_validation_episodes=min(2, max(1, int(support_episodes) - 1)),
        )
    )

    context_only_cfg = _prepare_cfg(
        base_cfg,
        out_dir=out_path,
        run_name="meta_adaptation_hybrid_context_only",
        seed=seed,
        scenario=scenario,
        train_iters=train_iters,
        fast_mode=fast_mode,
        support_episodes=support_episodes,
        query_episodes=query_episodes,
        episode_len=episode_len,
        device=device,
    )
    context_only_cfg.setdefault("context", {})["enabled"] = True
    context_only_cfg.setdefault("meta", {})["explicit_inner_outer"] = False
    context_only_cfg["meta"]["outer_step_size"] = 0.0
    context_only_cfg["meta"]["query_updates_enabled"] = False
    context_only_cfg["meta"]["query_context_updates_enabled"] = True
    context_only_cfg["meta"]["protocol_name"] = "context_only_no_outer_meta"
    context_only_trainer = MetaTrainer(context_only_cfg)
    if int(train_iters) > 0:
        context_only_trainer.train(meta_iters=int(train_iters))
    rows.extend(
        _collect_variant_rows(
            trainer=context_only_trainer,
            cfg=context_only_cfg,
            tasks=tasks,
            scenario=scenario,
            seed=seed,
            variant="hybrid_context_only",
            support_episodes=support_episodes,
            query_episodes=query_episodes,
            pre_query_episodes=pre_query_episodes,
            support_train_override=False,
        )
    )

    wo_meta_cfg = _prepare_cfg(
        base_cfg,
        out_dir=out_path,
        run_name="meta_adaptation_hybrid_wo_meta",
        seed=seed,
        scenario=scenario,
        train_iters=train_iters,
        fast_mode=fast_mode,
        support_episodes=support_episodes,
        query_episodes=query_episodes,
        episode_len=episode_len,
        device=device,
        ablation="wo_meta",
    )
    wo_meta_trainer = MetaTrainer(wo_meta_cfg)
    if int(train_iters) > 0:
        wo_meta_trainer.train(meta_iters=int(train_iters))
    rows.extend(
        _collect_variant_rows(
            trainer=wo_meta_trainer,
            cfg=wo_meta_cfg,
            tasks=tasks,
            scenario=scenario,
            seed=seed,
            variant="hybrid_wo_meta",
            support_episodes=support_episodes,
            query_episodes=query_episodes,
            pre_query_episodes=pre_query_episodes,
            support_train_override=False,
        )
    )

    csv_path = out_path / "meta_adaptation_episode_metrics.csv"
    _write_csv(csv_path, rows)

    summary = {
        "scenario": scenario,
        "seed": int(seed),
        "train_iters": int(train_iters),
        "n_tasks": int(n_tasks),
        "pre_query_episodes": int(pre_query_episodes),
        "support_episodes": int(support_episodes),
        "query_episodes": int(query_episodes),
        "episode_len": episode_len,
        "fixed_task_batch_hash": fixed_task_batch_hash,
        "ordered_fixed_task_batch_hash": ordered_fixed_task_batch_hash,
        "csv_path": str(csv_path),
        "plot_reward_path": str(out_path / "meta_vs_wo_meta_reward.png"),
        "plot_violation_path": str(out_path / "meta_vs_wo_meta_violation.png"),
        **_summarize_rows(rows),
    }
    summary_path = out_path / "meta_adaptation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if make_plots and rows:
        _plot_metric(rows, Path(summary["plot_reward_path"]), "reward", "episode reward")
        _plot_metric(rows, Path(summary["plot_violation_path"]), "violation_rate", "violation rate")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Meta vs w/o-meta few-shot adaptation diagnostics.")
    parser.add_argument("--cfg", default="configs/default.yaml")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--scenario", default="hard_stress")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--train-iters", type=int, default=1)
    parser.add_argument("--tasks", type=int, default=8)
    parser.add_argument("--pre-query-episodes", type=int, default=None)
    parser.add_argument("--support-episodes", type=int, default=5)
    parser.add_argument("--query-episodes", type=int, default=2)
    parser.add_argument("--episode-len", type=int, default=None)
    parser.add_argument("--fast-mode", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    summary = run_meta_adaptation_diagnostics(
        cfg_path=args.cfg,
        out_dir=args.out_dir,
        scenario=args.scenario,
        seed=args.seed,
        train_iters=args.train_iters,
        n_tasks=args.tasks,
        support_episodes=args.support_episodes,
        query_episodes=args.query_episodes,
        pre_query_episodes=args.pre_query_episodes,
        episode_len=args.episode_len,
        fast_mode=bool(args.fast_mode),
        device=args.device,
        make_plots=not bool(args.no_plots),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
