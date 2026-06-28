from __future__ import annotations

import copy
import random
import warnings
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tchhmrl.agents.dqn_upper import UpperDQN
from tchhmrl.agents.sac_lower import LowerSAC
from tchhmrl.buffers.replay_buffer import EpisodeBuffer, ReplayBuffer
from tchhmrl.models.context_encoder import ContextEncoder
from tchhmrl.envs.task_contract import (
    build_context_task_summary_v2,
    is_formally_comparable_record,
    physics_snapshot_from_cfg,
)
from tchhmrl.planning import ResidualPlanner
from tchhmrl.safety.safety_layer import SafetyLayer


class HierarchicalAgent:
    def __init__(self, cfg: Dict, device: torch.device):
        self.cfg = cfg
        self.device = device

        self.safety = SafetyLayer(cfg)
        self.upper = UpperDQN(cfg, device)
        self.lower = LowerSAC(cfg, self.safety, device)
        self.residual_planner = ResidualPlanner(cfg, device)

        replay_size = int(cfg["buffer"]["replay_size"])
        self.replay = ReplayBuffer(replay_size)
        self.upper_replay = ReplayBuffer(replay_size)
        self.episode = EpisodeBuffer(cfg["buffer"]["context_max_len"])
        self.z_dim = int(cfg["agent"]["z_dim"])
        self.context_enabled = bool(cfg.get("context", {}).get("enabled", True))
        self.context_cost_dim = len(cfg.get("meta", {}).get("dual_names", ["qos"] + [f"temp_tx{i}" for i in range(int(cfg["env"]["n_tx"]))]))
        self.context_task_dim = 9
        self.context_upper_dim = int(cfg["agent"].get("lower_upper_ctx_dim", 7))
        alignment_cfg = cfg.get("alignment", {})
        self.alignment_version = str(alignment_cfg.get("alignment_version", "system_model_v1"))
        self.task_summary_version = str(alignment_cfg.get("task_summary_version", "site_v2"))
        self.pre_alignment = bool(alignment_cfg.get("pre_alignment", False))
        self.physics_meta = physics_snapshot_from_cfg(cfg)
        self.loaded_alignment_meta = self._alignment_meta()

        ctx_cfg = cfg["context"]
        expected_context_input_dim = (
            int(cfg["agent"]["obs_dim"])
            + self.context_upper_dim
            + int(cfg["agent"]["act_lower_dim"])
            + 1
            + self.context_cost_dim
        )
        if int(ctx_cfg.get("input_dim", expected_context_input_dim)) != expected_context_input_dim:
            ctx_cfg["input_dim"] = expected_context_input_dim
        self.context_encoder = ContextEncoder(
            input_dim=int(ctx_cfg["input_dim"]),
            hidden_dim=int(ctx_cfg["gru_hidden"]),
            z_dim=int(ctx_cfg["z_dim"]),
        ).to(device)
        self.context_policy_deterministic = bool(ctx_cfg.get("policy_deterministic", True))
        self.context_updates_per_env_step = max(0, int(ctx_cfg.get("updates_per_env_step", 1)))
        self.context_train_window_len = int(ctx_cfg.get("train_window_len", 0))
        self.context_target_mask = self._context_target_vector(
            ctx_cfg.get("target_mask", [0, 0, 1, 0, 0, 1, 1, 1, 1]),
            default=1.0,
        )
        self.context_target_mean = self._context_target_vector(
            ctx_cfg.get("target_mean", [0.0, 0.0, 30.0, 0.0, 0.0, 0.05, 6.0, 6.0, 6.0]),
            default=0.0,
        )
        self.context_target_scale = np.maximum(
            self._context_target_vector(
                ctx_cfg.get("target_scale", [1.0, 1.0, 10.0, 1.0, 1.0, 0.05, 3.0, 3.0, 3.0]),
                default=1.0,
            ),
            1.0e-6,
        ).astype(np.float32)
        self.context_predictor = nn.Sequential(
            nn.Linear(self.z_dim, 32),
            nn.ReLU(),
            nn.Linear(32, self.context_task_dim),
        ).to(device)
        self.context_pred_w = float(ctx_cfg.get("pred_w", 0.5))
        self.context_optim = torch.optim.Adam(
            list(self.context_encoder.parameters()) + list(self.context_predictor.parameters()),
            lr=float(ctx_cfg["lr"]),
        )
        self.kl_beta = float(ctx_cfg["kl_beta"])
        self.batch_size = int(cfg["agent"]["batch_size"])
        self.upper_batch_size = int(
            cfg["agent"].get("upper_batch_size", cfg.get("upper_dqn", {}).get("batch_size", self.batch_size))
        )
        self.warmup_steps = int(cfg["agent"]["warmup_steps"])
        self.lower_updates_per_step = int(cfg["agent"].get("lower_updates_per_step", 1))
        self.upper_update_every = int(cfg["agent"].get("upper_update_every", 1))
        self.upper_warmup_steps = int(
            cfg["agent"].get(
                "upper_warmup_steps",
                max(self.upper_batch_size, self.warmup_steps // max(1, int(cfg["agent"].get("upper_hold_steps", 1)))),
            )
        )
        self.upper_hold_steps = int(cfg["agent"].get("upper_hold_steps", 1))

        hard_cfg = cfg.get("buffer", {}).get("hard_mining", {})
        self.hard_mining_enabled = bool(hard_cfg.get("enabled", False))
        self.hard_fraction = float(hard_cfg.get("fraction", 0.0))
        self.hard_cost_w = float(hard_cfg.get("cost_w", 1.0))
        self.hard_mode_switch_bonus = float(hard_cfg.get("mode_switch_bonus", 0.5))
        self.hard_boost_switch_bonus = float(hard_cfg.get("boost_switch_bonus", 0.5))
        constraint_replay_cfg = cfg.get("constraint_replay", {}) or {}
        self.constraint_replay_enabled = bool(constraint_replay_cfg.get("enabled", False))
        self.constraint_replay_uniform_fraction = float(constraint_replay_cfg.get("uniform_fraction", 0.50))
        self.constraint_replay_boundary_fraction = float(constraint_replay_cfg.get("boundary_fraction", 0.30))
        self.constraint_replay_violation_fraction = float(constraint_replay_cfg.get("violation_fraction", 0.20))
        self.constraint_replay_importance_weighting = bool(constraint_replay_cfg.get("importance_weighting", True))
        clip_cfg = constraint_replay_cfg.get("importance_weight_clip", [0.25, 4.0])
        self.constraint_replay_importance_weight_clip = (float(clip_cfg[0]), float(clip_cfg[1]))
        self.constraint_replay_thresholds = {
            "thermal_headroom_threshold": constraint_replay_cfg.get("thermal_headroom_threshold", None),
            "qos_margin_threshold": constraint_replay_cfg.get("qos_margin_threshold", None),
            "bus_utilization_threshold": constraint_replay_cfg.get("bus_utilization_threshold", None),
            "projection_residual_threshold": constraint_replay_cfg.get("projection_residual_threshold", None),
            "constraint_cost_threshold": constraint_replay_cfg.get("constraint_cost_threshold", 1.0e-8),
            "temperature_slope_threshold": constraint_replay_cfg.get("temperature_slope_threshold", None),
            "bus_current_max": float(self.safety.bus_current_max),
        }
        self.constraint_replay_empty_bucket_warn_after = int(
            constraint_replay_cfg.get("empty_bucket_warn_after", 20)
        )
        self.constraint_replay_empty_boundary_updates = 0
        self.constraint_replay_empty_violation_updates = 0
        if self.constraint_replay_enabled and bool(constraint_replay_cfg.get("require_boundary_thresholds", True)):
            boundary_keys = (
                "thermal_headroom_threshold",
                "qos_margin_threshold",
                "bus_utilization_threshold",
                "projection_residual_threshold",
                "temperature_slope_threshold",
            )
            if all(self.constraint_replay_thresholds.get(k) is None for k in boundary_keys):
                raise ValueError(
                    "constraint_replay.enabled requires at least one non-null boundary threshold"
                )

        self.global_step = 0
        self.rollout_step = 0
        self.current_meta_iter = 0
        self.safety_mem = {"current_boost": 0, "dwell_count": cfg["safety"]["min_dwell_steps"]}
        self.upper_mem = {"upper_idx": 0, "hold_left": 0}
        self.upper_plan: Optional[int] = None
        self.physical_feature_dim = int(cfg.get("physical_context", {}).get("input_dim", 18))
        self.prev_projection_residual = np.zeros(5, dtype=np.float32)
        self.prev_bus_headroom = 1.0

    def set_meta_iter(self, it: int) -> None:
        self.current_meta_iter = int(max(0, it))

    @staticmethod
    def _rng_state() -> Dict:
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng_state(state: Dict) -> None:
        if not state:
            return
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "torch" in state:
            torch.set_rng_state(state["torch"])
        if "torch_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])

    def _context_cost_vec(self, tr: Dict) -> np.ndarray:
        cost_vec = tr.get("cost_vec", None)
        if cost_vec is None:
            out = np.zeros(self.context_cost_dim, dtype=np.float32)
            out[0] = float(tr.get("cost", 0.0))
            return out
        arr = np.asarray(cost_vec, dtype=np.float32).reshape(-1)
        if arr.size == self.context_cost_dim:
            return arr.astype(np.float32)
        out = np.zeros(self.context_cost_dim, dtype=np.float32)
        out[: min(arr.size, self.context_cost_dim)] = arr[: self.context_cost_dim]
        return out

    def _context_upper_exec(self, tr: Dict) -> np.ndarray:
        if "boost_combo_exec" in tr and "mode_exec" in tr:
            boost = int(np.clip(int(round(float(tr.get("boost_combo_exec", 0.0)))), 0, 3))
            mode = int(np.clip(int(round(float(tr.get("mode_exec", 0.0)))), 0, 2))
        else:
            upper_idx_exec = int(np.clip(int(round(float(tr.get("upper_idx_exec", tr.get("upper_idx", 0.0))))), 0, 11))
            boost = upper_idx_exec // 3
            mode = upper_idx_exec % 3
        boost_oh = np.eye(4, dtype=np.float32)[boost]
        mode_oh = np.eye(3, dtype=np.float32)[mode]
        return np.concatenate([boost_oh, mode_oh]).astype(np.float32)

    def _context_feedback(self, tr: Dict) -> np.ndarray:
        reward_signal = float(tr.get("reward_task", tr.get("reward_raw", tr.get("reward", 0.0))))
        return np.concatenate(
            [
                np.asarray([reward_signal], dtype=np.float32),
                self._context_cost_vec(tr),
            ]
        ).astype(np.float32)

    def _task_params_from_transition(self, tr: Dict) -> np.ndarray:
        return build_context_task_summary_v2(tr)

    def _alignment_meta(self, *, pre_alignment: bool | None = None) -> Dict[str, object]:
        return {
            "alignment_version": self.alignment_version,
            "task_summary_version": self.task_summary_version,
            "pre_alignment": bool(self.pre_alignment if pre_alignment is None else pre_alignment),
            **self.physics_meta,
        }

    def is_formally_comparable(self) -> bool:
        return is_formally_comparable_record(self.loaded_alignment_meta)

    @staticmethod
    def _mean_metrics(metrics_list: list[Dict[str, float]]) -> Dict[str, float]:
        if not metrics_list:
            return {}
        keys = metrics_list[0].keys()
        return {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}

    def _context_target_vector(self, raw, *, default: float) -> np.ndarray:
        arr = np.asarray(raw if raw is not None else [], dtype=np.float32).reshape(-1)
        if arr.size < self.context_task_dim:
            arr = np.pad(arr, (0, self.context_task_dim - arr.size), constant_values=float(default))
        return arr[: self.context_task_dim].astype(np.float32)

    def _hard_score(self, tr: Dict) -> float:
        cost = max(float(tr.get("cost", 0.0)), 0.0)
        mode_cur = int(round(float(tr.get("mode_exec", 0.0))))
        mode_next = int(round(float(tr.get("mode_exec_next", mode_cur))))
        boost_cur = int(round(float(tr.get("boost_combo_exec", 0.0))))
        boost_next = int(round(float(tr.get("boost_combo_exec_next", boost_cur))))
        return (
            self.hard_cost_w * cost
            + self.hard_mode_switch_bonus * float(mode_cur != mode_next)
            + self.hard_boost_switch_bonus * float(boost_cur != boost_next)
        )

    def clear_learning_buffers(self) -> None:
        self.replay.clear()
        self.upper_replay.clear()
        self.episode.clear()

    def reset_rollout_state(self, clear_context: bool = True) -> None:
        if clear_context:
            self.episode.clear()
        self.safety_mem = {
            "current_boost": 0,
            "dwell_count": self.cfg["safety"]["min_dwell_steps"],
        }
        self.upper_mem = {"upper_idx": 0, "hold_left": 0}
        self.upper_plan = None
        self.rollout_step = 0
        self.prev_projection_residual = np.zeros(5, dtype=np.float32)
        self.prev_bus_headroom = 1.0

    def reset_episode_state(self) -> None:
        self.reset_rollout_state(clear_context=True)

    def snapshot_train_state(self) -> Dict:
        return {
            "upper": copy.deepcopy(self.upper.state_dict()),
            "lower": copy.deepcopy(self.lower.state_dict()),
            "context_encoder": copy.deepcopy(self.context_encoder.state_dict()),
            "context_predictor": copy.deepcopy(self.context_predictor.state_dict()),
            "context_optim": copy.deepcopy(self.context_optim.state_dict()),
            "safety": copy.deepcopy(self.safety.state_dict()),
            "global_step": int(self.global_step),
            "rollout_step": int(self.rollout_step),
            "current_meta_iter": int(self.current_meta_iter),
        }

    def restore_train_state(self, state: Dict) -> None:
        self.upper.load_state_dict(copy.deepcopy(state["upper"]))
        self.lower.load_state_dict(copy.deepcopy(state["lower"]))
        self.context_encoder.load_state_dict(copy.deepcopy(state["context_encoder"]))
        self.context_predictor.load_state_dict(copy.deepcopy(state["context_predictor"]))
        self.context_optim.load_state_dict(copy.deepcopy(state["context_optim"]))
        if "safety" in state:
            self.safety.load_state_dict(copy.deepcopy(state["safety"]))
        self.global_step = int(state.get("global_step", 0))
        self.rollout_step = int(state.get("rollout_step", 0))
        self.current_meta_iter = int(state.get("current_meta_iter", self.current_meta_iter))

    def snapshot_mutable_state(self) -> Dict:
        state = self.snapshot_train_state()
        state.update(
            {
                "replay": self.replay.state_dict(),
                "upper_replay": self.upper_replay.state_dict(),
                "episode": self.episode.state_dict(),
                "safety_mem": copy.deepcopy(self.safety_mem),
                "upper_mem": copy.deepcopy(self.upper_mem),
                "upper_plan": copy.deepcopy(self.upper_plan),
                "prev_projection_residual": self.prev_projection_residual.copy(),
                "prev_bus_headroom": float(self.prev_bus_headroom),
                "rng": self._rng_state(),
            }
        )
        return state

    def restore_mutable_state(self, state: Dict) -> None:
        self.restore_train_state(state)
        if "replay" in state:
            self.replay.load_state_dict(copy.deepcopy(state["replay"]))
        if "upper_replay" in state:
            self.upper_replay.load_state_dict(copy.deepcopy(state["upper_replay"]))
        if "episode" in state:
            self.episode.load_state_dict(copy.deepcopy(state["episode"]))
        self.safety_mem = copy.deepcopy(state.get("safety_mem", self.safety_mem))
        self.upper_mem = copy.deepcopy(state.get("upper_mem", self.upper_mem))
        self.upper_plan = copy.deepcopy(state.get("upper_plan", None))
        self.prev_projection_residual = np.asarray(
            state.get("prev_projection_residual", self.prev_projection_residual),
            dtype=np.float32,
        ).copy()
        self.prev_bus_headroom = float(state.get("prev_bus_headroom", self.prev_bus_headroom))
        self._restore_rng_state(state.get("rng", {}))

    def reset_optimizer_states(self) -> None:
        optimizers = [
            self.upper.optim,
            self.lower.actor_optim,
            self.lower.critic_optim,
            self.lower.constraint_optim,
            self.lower.alpha_optim,
            self.context_optim,
        ]
        for optim in optimizers:
            if optim is not None:
                optim.state.clear()

    def current_physical_features(self, temps: np.ndarray | None = None) -> np.ndarray:
        diag = self.safety.thermal_diagnostics()
        gain_mean = np.asarray(diag.get("thermal_gain_mean", np.ones(3)), dtype=np.float32).reshape(-1)[:3]
        gain_std = np.asarray(
            diag.get("thermal_gain_uncertainty", diag.get("thermal_gain_std", np.zeros(3))),
            dtype=np.float32,
        ).reshape(-1)[:3]
        nominal = np.asarray(
            self.cfg.get("safety", {}).get("effective_gain_initial", np.ones(3)),
            dtype=np.float32,
        ).reshape(-1)[:3]
        nominal = np.maximum(nominal, 1.0e-6)
        gain_mean = np.clip(gain_mean / nominal, 0.0, 3.0).astype(np.float32)
        gain_std = np.clip(gain_std / nominal, 0.0, 2.0).astype(np.float32)
        temp_slope = np.asarray(diag.get("temperature_slope", np.zeros(3)), dtype=np.float32).reshape(-1)[:3] / 10.0
        if temps is None:
            headroom = np.asarray(diag.get("thermal_headroom", np.zeros(3)), dtype=np.float32).reshape(-1)[:3]
        else:
            headroom = (float(self.safety.thermal_safe) - np.asarray(temps, dtype=np.float32).reshape(-1)[:3])
        headroom = np.nan_to_num(headroom / max(float(self.safety.thermal_safe), 1.0e-6), nan=0.0)
        bus_headroom = np.asarray([float(self.prev_bus_headroom)], dtype=np.float32)
        residual = np.asarray(self.prev_projection_residual, dtype=np.float32).reshape(-1)[:5]
        features = np.concatenate([gain_mean, gain_std, temp_slope, headroom, bus_headroom, residual]).astype(np.float32)
        if features.size != self.physical_feature_dim:
            out = np.zeros((self.physical_feature_dim,), dtype=np.float32)
            out[: min(features.size, self.physical_feature_dim)] = features[: self.physical_feature_dim]
            features = out
        return features

    def update_safety_estimator(
        self,
        *,
        temps_before: np.ndarray,
        info: Dict,
    ) -> Dict[str, np.ndarray | bool]:
        thermal_base, _ = self.safety._thermal_base_np(
            np.asarray(temps_before, dtype=np.float32),
            float(info.get("amb_temp", self.cfg.get("env", {}).get("amb_temp", 0.0))),
        )
        return self.safety.update_thermal_estimator(
            currents=np.asarray(info.get("currents_exec", np.zeros(3, dtype=np.float32)), dtype=np.float32),
            temps_before=np.asarray(temps_before, dtype=np.float32),
            temps_after=np.asarray(info.get("temps", temps_before), dtype=np.float32),
            thermal_base=thermal_base,
        )

    @staticmethod
    def _average_state_dict(states: list[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not states:
            raise ValueError("states must be non-empty")
        out: Dict[str, torch.Tensor] = {}
        keys = states[0].keys()
        for key in keys:
            first = states[0][key]
            if torch.is_tensor(first) and torch.is_floating_point(first):
                stacked = torch.stack([s[key].detach().float().cpu() for s in states], dim=0)
                out[key] = stacked.mean(dim=0)
            else:
                out[key] = copy.deepcopy(first)
        return out

    @staticmethod
    def _blend_module_state(module: nn.Module, target_state: Dict[str, torch.Tensor], step_size: float) -> None:
        current = module.state_dict()
        blended: Dict[str, torch.Tensor] = {}
        for key, cur_val in current.items():
            tgt_val = target_state[key]
            if torch.is_tensor(cur_val) and torch.is_floating_point(cur_val):
                tgt = tgt_val.to(device=cur_val.device, dtype=cur_val.dtype)
                blended[key] = cur_val + step_size * (tgt - cur_val)
            else:
                blended[key] = tgt_val
        module.load_state_dict(blended)

    def apply_outer_update(self, adapted_states: list[Dict], step_size: float) -> None:
        if not adapted_states or step_size <= 0.0:
            return
        step_size = float(np.clip(step_size, 0.0, 1.0))
        upper_q = self._average_state_dict([s["upper"]["q"] for s in adapted_states])
        upper_q_tgt = self._average_state_dict([s["upper"]["q_tgt"] for s in adapted_states])
        lower_actor = self._average_state_dict([s["lower"]["actor"] for s in adapted_states])
        lower_q1 = self._average_state_dict([s["lower"]["q1"] for s in adapted_states])
        lower_q2 = self._average_state_dict([s["lower"]["q2"] for s in adapted_states])
        lower_q1_tgt = self._average_state_dict([s["lower"]["q1_tgt"] for s in adapted_states])
        lower_q2_tgt = self._average_state_dict([s["lower"]["q2_tgt"] for s in adapted_states])
        ctx_enc = self._average_state_dict([s["context_encoder"] for s in adapted_states])
        ctx_pred = self._average_state_dict([s["context_predictor"] for s in adapted_states])

        self._blend_module_state(self.upper.q, upper_q, step_size)
        self._blend_module_state(self.upper.q_tgt, upper_q_tgt, step_size)
        if self.upper.q_phys is not None and all(s["upper"].get("q_phys") is not None for s in adapted_states):
            upper_q_phys = self._average_state_dict([s["upper"]["q_phys"] for s in adapted_states])
            self._blend_module_state(self.upper.q_phys, upper_q_phys, step_size)
        if self.upper.q_tgt_phys is not None and all(s["upper"].get("q_tgt_phys") is not None for s in adapted_states):
            upper_q_tgt_phys = self._average_state_dict([s["upper"]["q_tgt_phys"] for s in adapted_states])
            self._blend_module_state(self.upper.q_tgt_phys, upper_q_tgt_phys, step_size)
        self._blend_module_state(self.lower.actor, lower_actor, step_size)
        self._blend_module_state(self.lower.q1, lower_q1, step_size)
        self._blend_module_state(self.lower.q2, lower_q2, step_size)
        self._blend_module_state(self.lower.q1_tgt, lower_q1_tgt, step_size)
        self._blend_module_state(self.lower.q2_tgt, lower_q2_tgt, step_size)
        if self.lower.actor_phys is not None and all(s["lower"].get("actor_phys") is not None for s in adapted_states):
            actor_phys = self._average_state_dict([s["lower"]["actor_phys"] for s in adapted_states])
            self._blend_module_state(self.lower.actor_phys, actor_phys, step_size)
        if self.lower.q1_phys is not None and all(s["lower"].get("q1_phys") is not None for s in adapted_states):
            q1_phys = self._average_state_dict([s["lower"]["q1_phys"] for s in adapted_states])
            self._blend_module_state(self.lower.q1_phys, q1_phys, step_size)
        if self.lower.q2_phys is not None and all(s["lower"].get("q2_phys") is not None for s in adapted_states):
            q2_phys = self._average_state_dict([s["lower"]["q2_phys"] for s in adapted_states])
            self._blend_module_state(self.lower.q2_phys, q2_phys, step_size)
        if self.lower.q1_tgt_phys is not None and all(s["lower"].get("q1_tgt_phys") is not None for s in adapted_states):
            q1_tgt_phys = self._average_state_dict([s["lower"]["q1_tgt_phys"] for s in adapted_states])
            self._blend_module_state(self.lower.q1_tgt_phys, q1_tgt_phys, step_size)
        if self.lower.q2_tgt_phys is not None and all(s["lower"].get("q2_tgt_phys") is not None for s in adapted_states):
            q2_tgt_phys = self._average_state_dict([s["lower"]["q2_tgt_phys"] for s in adapted_states])
            self._blend_module_state(self.lower.q2_tgt_phys, q2_tgt_phys, step_size)
        if self.lower.constraint_q is not None and all(s["lower"].get("constraint_q") is not None for s in adapted_states):
            constraint_q = self._average_state_dict([s["lower"]["constraint_q"] for s in adapted_states])
            self._blend_module_state(self.lower.constraint_q, constraint_q, step_size)
        if self.lower.constraint_q_tgt is not None and all(s["lower"].get("constraint_q_tgt") is not None for s in adapted_states):
            constraint_q_tgt = self._average_state_dict([s["lower"]["constraint_q_tgt"] for s in adapted_states])
            self._blend_module_state(self.lower.constraint_q_tgt, constraint_q_tgt, step_size)
        if self.lower.constraint_phys is not None and all(s["lower"].get("constraint_phys") is not None for s in adapted_states):
            constraint_phys = self._average_state_dict([s["lower"]["constraint_phys"] for s in adapted_states])
            self._blend_module_state(self.lower.constraint_phys, constraint_phys, step_size)
        if self.lower.constraint_tgt_phys is not None and all(
            s["lower"].get("constraint_tgt_phys") is not None for s in adapted_states
        ):
            constraint_tgt_phys = self._average_state_dict([s["lower"]["constraint_tgt_phys"] for s in adapted_states])
            self._blend_module_state(self.lower.constraint_tgt_phys, constraint_tgt_phys, step_size)
        self._blend_module_state(self.context_encoder, ctx_enc, step_size)
        self._blend_module_state(self.context_predictor, ctx_pred, step_size)
        if self.lower.auto_alpha and self.lower.log_alpha is not None:
            alpha_vals = [float(s["lower"].get("log_alpha", self.lower.log_alpha.item())) for s in adapted_states]
            avg_alpha = float(np.mean(alpha_vals))
            with torch.no_grad():
                current = float(self.lower.log_alpha.item())
                self.lower.log_alpha.copy_(
                    torch.tensor(current + step_size * (avg_alpha - current), dtype=self.lower.log_alpha.dtype, device=self.lower.log_alpha.device)
                )

    def infer_z(self) -> np.ndarray:
        if not self.context_enabled:
            return np.zeros(self.z_dim, dtype=np.float32)
        if len(self.episode) < 2:
            return np.zeros(self.z_dim, dtype=np.float32)

        rows = []
        for tr in self.episode.as_list():
            rows.append(
                np.concatenate(
                    [
                        tr["obs"],
                        self._context_upper_exec(tr),
                        tr["act_exec"],
                        self._context_feedback(tr),
                    ]
                )
            )
        seq = torch.tensor(np.stack(rows), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            z, _, _, _ = self.context_encoder.infer(seq, deterministic=self.context_policy_deterministic)
        return z.squeeze(0).cpu().numpy().astype(np.float32)

    def act(
        self,
        obs: np.ndarray,
        temps: np.ndarray,
        amb_temp: float,
        gamma: float,
        delta: float,
        z: Optional[np.ndarray] = None,
        eval_mode: bool = False,
    ) -> Tuple[Dict, Dict]:
        z = np.zeros(self.z_dim, dtype=np.float32) if z is None else z.astype(np.float32)

        macro_new = self.upper_mem["hold_left"] <= 0
        exec_map = self.safety.raw_to_exec_map(self.safety_mem, temps=temps)
        action_mask = self.safety.upper_raw_allowed_mask(temps=temps)
        physical_features = self.current_physical_features(temps=temps)
        if macro_new:
            if self.upper_plan is not None:
                upper_idx_raw = int(self.upper_plan)
                self.upper_plan = None
            else:
                upper_idx_raw = self.upper.select_action(
                    obs,
                    z,
                    t=self.global_step,
                    eval_mode=eval_mode,
                    exec_map=exec_map,
                    action_mask=action_mask,
                    physical_features=physical_features,
                )
            self.upper_mem["upper_idx"] = int(upper_idx_raw)
            self.upper_mem["hold_left"] = max(1, self.upper_hold_steps)
        else:
            upper_idx_raw = int(self.upper_mem["upper_idx"])
        self.upper_mem["hold_left"] = max(0, int(self.upper_mem["hold_left"]) - 1)

        boost_preview, mode_preview = self.safety.preview_exec(upper_idx_raw, self.safety_mem, temps=temps)
        upper_idx_exec = self.safety.encode_exec(boost_preview, mode_preview)
        lower_raw = self.lower.select_action(
            obs,
            z,
            upper_idx=upper_idx_exec,
            physical_features=physical_features,
            eval_mode=eval_mode,
        )
        policy_lower_raw = lower_raw.copy().astype(np.float32)
        planner_aux: Dict[str, object] = {
            "residual_planner_enabled": False,
            "residual_planner_candidate_count": 0,
            "residual_planner_latency_ms": 0.0,
            "residual_planner_score_improvement": 0.0,
        }
        planner_step = int(self.rollout_step)
        if self.residual_planner.active(meta_iter=self.current_meta_iter):
            lower_raw, planner_aux = self.residual_planner.plan(
                lower=self.lower,
                safety=self.safety,
                obs=obs,
                z=z,
                upper_idx_exec=int(upper_idx_exec),
                boost_combo=int(boost_preview),
                mode=int(mode_preview),
                policy_raw=lower_raw,
                physical_features=physical_features,
                thermal_headroom=(float(self.safety.thermal_safe) - np.asarray(temps, dtype=np.float32)),
                temps=temps,
                amb_temp=amb_temp,
                meta_iter=int(self.current_meta_iter),
                global_step=planner_step,
                previous_projection_residual=self.prev_projection_residual,
            )

        safe, self.safety_mem = self.safety.project_np(
            upper_idx_raw,
            lower_raw,
            temps=temps,
            amb_temp=amb_temp,
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

        exec_vec = np.concatenate(
            [safe["currents_exec"], np.asarray([safe["rho_exec"], safe["tau_exec"]], dtype=np.float32)]
        ).astype(np.float32)
        desired_currents = np.asarray(
            safe.get("current_requested", np.asarray(safe["raw_current_frac"], dtype=np.float32) * self.safety.current_max),
            dtype=np.float32,
        )
        desired_currents_pre_static = np.asarray(
            safe.get("current_requested_pre_static_cap", desired_currents),
            dtype=np.float32,
        )
        mode_exec = int(safe["mode_exec"])
        rho_raw_decoded = float(safe["rho_raw_decoded"])
        tau_raw_decoded = float(safe["tau_raw_decoded"])
        if mode_exec == 0:
            receiver_desired = np.asarray([rho_raw_decoded, 1.0], dtype=np.float32)
        elif mode_exec == 1:
            receiver_desired = np.asarray([0.0, tau_raw_decoded], dtype=np.float32)
        else:
            receiver_desired = np.asarray([rho_raw_decoded, tau_raw_decoded], dtype=np.float32)
        desired_vec = np.concatenate(
            [
                desired_currents.astype(np.float32),
                receiver_desired,
            ]
        ).astype(np.float32)
        desired_total_vec = np.concatenate(
            [
                desired_currents_pre_static.astype(np.float32),
                np.asarray([rho_raw_decoded, tau_raw_decoded], dtype=np.float32),
            ]
        ).astype(np.float32)
        decoder_vec = np.concatenate(
            [
                desired_currents.astype(np.float32),
                np.asarray([rho_raw_decoded, tau_raw_decoded], dtype=np.float32),
            ]
        ).astype(np.float32)
        safety_residual = exec_vec - desired_vec
        total_projection_residual = exec_vec - desired_total_vec
        decoder_residual = decoder_vec - desired_total_vec

        def _normalize_residual(vec: np.ndarray) -> np.ndarray:
            out = np.asarray(vec, dtype=np.float32).copy()
            out[:3] = out[:3] / np.maximum(self.safety.current_max, 1.0e-6)
            return out.astype(np.float32)

        projection_residual_norm = _normalize_residual(safety_residual)
        total_projection_residual_norm = _normalize_residual(total_projection_residual)
        decoder_residual_norm = _normalize_residual(decoder_residual)
        self.prev_projection_residual = projection_residual_norm.astype(np.float32)
        self.prev_bus_headroom = float(
            np.clip(
                (self.safety.bus_current_max - float(safe.get("projected_current_total", 0.0)))
                / max(self.safety.bus_current_max, 1.0e-6),
                0.0,
                1.0,
            )
        )

        aux = {
            "z": z,
            "upper_idx_raw": int(upper_idx_raw),
            "upper_idx_exec": int(safe["upper_idx_exec"]),
            "boost_combo_exec": int(safe["boost_combo_exec"]),
            "mode_exec": int(safe["mode_exec"]),
            "act_exec": exec_vec,
            "act_raw": lower_raw.astype(np.float32),
            "act_refined_raw": lower_raw.astype(np.float32),
            "act_policy_raw": policy_lower_raw.astype(np.float32),
            "policy_action_raw": policy_lower_raw.astype(np.float32),
            "planner_action_raw": lower_raw.astype(np.float32),
            "executed_action": exec_vec.astype(np.float32),
            "planner_selected": bool(planner_aux.get("residual_planner_replaced_policy", False)),
            "act_desired": desired_vec.astype(np.float32),
            "projection_residual": projection_residual_norm.astype(np.float32),
            "decoder_residual": decoder_residual_norm.astype(np.float32),
            "safety_projection_residual": projection_residual_norm.astype(np.float32),
            "total_projection_residual": total_projection_residual_norm.astype(np.float32),
            "physical_features": physical_features.astype(np.float32),
            "upper_action_mask": action_mask.astype(np.float32),
            **planner_aux,
            "t_pred": safe["t_pred"],
            "thermal_scale": safe["thermal_scale"],
            "thermal_soft_scale": safe.get("thermal_soft_scale"),
            "thermal_cutoff_scale": safe.get("thermal_cutoff_scale"),
            "thermal_cap_current": safe.get("thermal_cap_current"),
            "thermal_cap_scale": safe.get("thermal_cap_scale"),
            "thermal_cap_margin_c": safe.get("thermal_cap_margin_c"),
            "thermal_margin_min": safe.get("thermal_margin_min"),
            "action_decode_mode": safe.get("action_decode_mode"),
            "raw_current_frac": safe.get("raw_current_frac"),
            "rho_raw_decoded": safe.get("rho_raw_decoded"),
            "tau_raw_decoded": safe.get("tau_raw_decoded"),
            "raw_current_total": safe.get("raw_current_total"),
            "current_decoder": safe.get("current_decoder"),
            "current_requested": safe.get("current_requested"),
            "current_requested_pre_static_cap": safe.get("current_requested_pre_static_cap"),
            "actor_total_current_requested": safe.get("actor_total_current_requested"),
            "actor_allocation_anchor": safe.get("actor_allocation_anchor"),
            "actor_allocation_ld1": safe.get("actor_allocation_ld1"),
            "actor_allocation_ld2": safe.get("actor_allocation_ld2"),
            "actor_inactive_allocation_sum": safe.get("actor_inactive_allocation_sum"),
            "actor_per_source_clip_count": safe.get("actor_per_source_clip_count"),
            "structured_actor_per_source_clip_rate": safe.get("structured_actor_per_source_clip_rate"),
            "structured_actor_bus_clip_rate": safe.get("structured_actor_bus_clip_rate"),
            "mode_effective_latent_dim": safe.get("mode_effective_latent_dim"),
            "masked_current_total": safe.get("masked_current_total"),
            "bus_projected_current_total": safe.get("bus_projected_current_total"),
            "projected_current_total": safe.get("projected_current_total"),
            "projection_compression_ratio": safe.get("projection_compression_ratio"),
            "adaptive_thermal_enabled": safe.get("adaptive_thermal_enabled"),
            "thermal_gain_mean": safe.get("thermal_gain_mean"),
            "thermal_gain_std": safe.get("thermal_gain_std"),
            "thermal_gain_safe_scale": safe.get("thermal_gain_safe_scale"),
            "thermal_gain_beta": safe.get("thermal_gain_beta"),
            "thermal_gain_valid_count": safe.get("thermal_gain_valid_count"),
            "temperature_slope": safe.get("temperature_slope"),
            "thermal_headroom": safe.get("thermal_headroom"),
            "upper_shield_enabled": safe.get("upper_shield_enabled", False),
            "upper_shield_applied": safe.get("upper_shield_applied", False),
            "upper_shield_requested_boost": safe.get("upper_shield_requested_boost", safe.get("boost_combo_exec", 0)),
            "upper_shield_selected_boost": safe.get("upper_shield_selected_boost", safe.get("boost_combo_exec", 0)),
            "upper_shield_allowed_anchor": safe.get("upper_shield_allowed_anchor", 1.0),
            "upper_shield_allowed_ld1": safe.get("upper_shield_allowed_ld1", 1.0),
            "upper_shield_allowed_ld2": safe.get("upper_shield_allowed_ld2", 1.0),
            "upper_shield_allowed_all": safe.get("upper_shield_allowed_all", 1.0),
            "macro_new": bool(macro_new),
            "hold_left": int(self.upper_mem["hold_left"]),
            "rollout_step": int(planner_step),
        }
        self.rollout_step += 1
        return action, aux

    def observe(self, transition: Dict) -> None:
        self.observe_lower(transition)

    def preview_next_macro(
        self,
        next_obs: np.ndarray,
        z_next: np.ndarray,
        physical_features_next: np.ndarray | None = None,
        temps_next: np.ndarray | None = None,
        eval_mode: bool = False,
        commit_plan: bool = False,
    ) -> Dict[str, int]:
        macro_new_next = self.upper_mem["hold_left"] <= 0
        next_exec_map = self.safety.raw_to_exec_map(self.safety_mem, temps=temps_next)
        next_action_mask = self.safety.upper_raw_allowed_mask(temps=temps_next)
        if macro_new_next:
            upper_idx_next_raw = int(
                self.upper.select_action(
                    next_obs.astype(np.float32),
                    z_next.astype(np.float32),
                    t=self.global_step + 1,
                    eval_mode=eval_mode,
                    exec_map=next_exec_map,
                    action_mask=next_action_mask,
                    physical_features=physical_features_next,
                )
            )
            if commit_plan:
                self.upper_plan = upper_idx_next_raw
        else:
            upper_idx_next_raw = int(self.upper_mem["upper_idx"])

        boost_next, mode_next = self.safety.preview_exec(upper_idx_next_raw, self.safety_mem, temps=temps_next)
        return {
            "upper_idx_raw_next": int(upper_idx_next_raw),
            "upper_idx_exec_next": int(self.safety.encode_exec(boost_next, mode_next)),
            "boost_combo_exec_next": int(boost_next),
            "mode_exec_next": int(mode_next),
            "next_exec_map": next_exec_map.astype(np.float32),
            "next_action_mask": next_action_mask.astype(np.float32),
        }

    def observe_lower(
        self,
        transition: Dict,
        next_macro_fn: Optional[Callable[[np.ndarray], Dict[str, float | int]]] = None,
    ) -> np.ndarray:
        transition = dict(transition)
        if "policy_action_raw" not in transition:
            transition["policy_action_raw"] = transition.get("act_policy_raw", transition.get("act_raw"))
        if "planner_action_raw" not in transition:
            transition["planner_action_raw"] = transition.get("act_refined_raw", transition.get("act_raw"))
        if "act_refined_raw" not in transition:
            transition["act_refined_raw"] = transition["planner_action_raw"]
        if "executed_action" not in transition:
            transition["executed_action"] = transition.get("act_exec")
        transition["planner_selected"] = bool(transition.get("planner_selected", False))
        task_params = self._task_params_from_transition(transition)
        self.episode.add(
            {
                "obs": transition["obs"],
                "upper_idx_exec": transition.get("upper_idx_exec", transition.get("upper_idx", 0.0)),
                "boost_combo_exec": transition.get("boost_combo_exec", 0.0),
                "mode_exec": transition.get("mode_exec", 0.0),
                "act_exec": transition["act_exec"],
                "executed_action": transition["executed_action"],
                "reward": transition["reward"],
                "reward_raw": transition.get("reward_raw", transition["reward"]),
                "reward_task": transition.get("reward_task", transition.get("reward_raw", transition["reward"])),
                "reward_benchmark": transition.get("reward_benchmark", transition.get("reward_raw", transition["reward"])),
                "reward_dual_penalized": transition.get("reward_dual_penalized", transition["reward"]),
                "cost": transition.get("cost", 0.0),
                "cost_vec": self._context_cost_vec(transition),
                "task_params": task_params,
            }
        )
        z_next = self.infer_z().astype(np.float32)
        tr = dict(transition)
        tr["z_next"] = z_next
        if next_macro_fn is not None:
            tr.update(next_macro_fn(z_next))
        else:
            tr.setdefault("upper_idx_raw_next", tr.get("upper_idx_raw", tr.get("upper_idx_exec", 0.0)))
            tr.setdefault("upper_idx_exec_next", tr.get("upper_idx_exec", tr.get("upper_idx", 0.0)))
            tr.setdefault("boost_combo_exec_next", tr.get("boost_combo_exec", 0.0))
            tr.setdefault("mode_exec_next", tr.get("mode_exec", 0.0))
            tr.setdefault(
                "next_exec_map",
                self.safety.raw_to_exec_map(self.safety_mem).astype(np.float32),
            )
            tr.setdefault("next_action_mask", self.safety.upper_raw_allowed_mask().astype(np.float32))
        self.replay.add(tr)
        return z_next

    def observe_upper(self, transition: Dict) -> None:
        self.upper_replay.add(transition)

    def update_context_encoder(self) -> Dict[str, float]:
        if not self.context_enabled:
            return {}
        if len(self.episode) < 3:
            return {}

        items = self.episode.as_list()
        if self.context_train_window_len > 0:
            items = items[-int(self.context_train_window_len) :]
        if len(items) < 3:
            return {}

        rows = []
        for tr in items:
            rows.append(
                np.concatenate(
                    [
                        tr["obs"],
                        self._context_upper_exec(tr),
                        tr["act_exec"],
                        self._context_feedback(tr),
                    ]
                )
            )
        seq = torch.tensor(np.stack(rows), dtype=torch.float32, device=self.device).unsqueeze(0)
        z, kl, _, _ = self.context_encoder.infer(seq)
        task_rows = np.stack(
            [
                np.asarray(tr.get("task_params", np.zeros(self.context_task_dim, dtype=np.float32)), dtype=np.float32)
                for tr in items
            ],
            axis=0,
        )
        target_raw = (
            torch.tensor(task_rows.mean(axis=0), dtype=torch.float32, device=self.device)
            .view(1, self.context_task_dim)
            .detach()
        )
        target_mean = torch.tensor(self.context_target_mean, dtype=torch.float32, device=self.device).view(1, -1)
        target_scale = torch.tensor(self.context_target_scale, dtype=torch.float32, device=self.device).view(1, -1)
        target_mask = torch.tensor(self.context_target_mask, dtype=torch.float32, device=self.device).view(1, -1)
        target = (target_raw - target_mean) / target_scale
        pred = self.context_predictor(z)
        pred_loss = (((pred - target) ** 2) * target_mask).sum() / torch.clamp(target_mask.sum(), min=1.0)
        # Keep latent norm bounded to avoid unstable conditioning.
        latent_reg = (z.pow(2).mean())
        loss = self.kl_beta * kl + self.context_pred_w * pred_loss + 1.0e-4 * latent_reg

        self.context_optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.context_encoder.parameters()) + list(self.context_predictor.parameters()),
            5.0,
        )
        self.context_optim.step()

        return {
            "context_loss": float(loss.item()),
            "kl": float(kl.item()),
            "ctx_pred_loss": float(pred_loss.item()),
            "ctx_latent_norm": float(latent_reg.item()),
            "ctx_target_mask_active_dim": float(target_mask.sum().detach().item()),
            "ctx_train_window_len": float(len(items)),
        }

    def learn(self) -> Dict[str, float]:
        self.global_step += 1

        merged = {}

        if len(self.replay) >= max(self.batch_size, self.warmup_steps):
            sac_list = []
            ctx_list = []
            ctx_budget = min(max(0, self.context_updates_per_env_step), max(1, self.lower_updates_per_step))
            for update_idx in range(max(1, self.lower_updates_per_step)):
                lower_batch = self.replay.sample(
                    self.batch_size,
                    hard_fraction=self.hard_fraction if self.hard_mining_enabled else 0.0,
                    scorer=self._hard_score if self.hard_mining_enabled else None,
                )
                constraint_batch = None
                constraint_stats = {}
                if self.constraint_replay_enabled and getattr(self.lower, "constraint_critics_enabled", False):
                    constraint_batch, constraint_stats = self.replay.sample_stratified_constraint(
                        self.batch_size,
                        uniform_fraction=self.constraint_replay_uniform_fraction,
                        boundary_fraction=self.constraint_replay_boundary_fraction,
                        violation_fraction=self.constraint_replay_violation_fraction,
                        thresholds=self.constraint_replay_thresholds,
                        importance_weighting=self.constraint_replay_importance_weighting,
                        importance_weight_clip=self.constraint_replay_importance_weight_clip,
                    )
                    boundary_count = float(constraint_stats.get("constraint_batch_boundary_count", 0.0))
                    violation_count = float(constraint_stats.get("constraint_batch_violation_count", 0.0))
                    self.constraint_replay_empty_boundary_updates = (
                        self.constraint_replay_empty_boundary_updates + 1 if boundary_count <= 0.0 else 0
                    )
                    self.constraint_replay_empty_violation_updates = (
                        self.constraint_replay_empty_violation_updates + 1 if violation_count <= 0.0 else 0
                    )
                    if (
                        self.constraint_replay_empty_bucket_warn_after > 0
                        and self.constraint_replay_empty_boundary_updates == self.constraint_replay_empty_bucket_warn_after
                    ):
                        warnings.warn(
                            "constraint replay boundary bucket has been empty for "
                            f"{self.constraint_replay_empty_boundary_updates} consecutive lower updates",
                            RuntimeWarning,
                        )
                    if (
                        self.constraint_replay_empty_bucket_warn_after > 0
                        and self.constraint_replay_empty_violation_updates == self.constraint_replay_empty_bucket_warn_after
                    ):
                        warnings.warn(
                            "constraint replay violation bucket has been empty for "
                            f"{self.constraint_replay_empty_violation_updates} consecutive lower updates",
                            RuntimeWarning,
                        )
                    constraint_stats["constraint_empty_boundary_update"] = float(boundary_count <= 0.0)
                    constraint_stats["constraint_empty_violation_update"] = float(violation_count <= 0.0)
                    constraint_stats["constraint_empty_boundary_update_streak"] = float(
                        self.constraint_replay_empty_boundary_updates
                    )
                    constraint_stats["constraint_empty_violation_update_streak"] = float(
                        self.constraint_replay_empty_violation_updates
                    )
                sac_stats = self.lower.update(lower_batch, constraint_batch=constraint_batch)
                sac_stats.update(constraint_stats)
                sac_stats["constraint_replay_enabled"] = float(
                    self.constraint_replay_enabled and constraint_batch is not None
                )
                sac_list.append(sac_stats)
                if update_idx < ctx_budget:
                    ctx_list.append(self.update_context_encoder())
            merged.update(self._mean_metrics(sac_list))
            merged.update(self._mean_metrics(ctx_list))

        if (
            len(self.upper_replay) >= max(self.upper_batch_size, self.upper_warmup_steps)
            and self.global_step % max(1, self.upper_update_every) == 0
        ):
            upper_batch = self.upper_replay.sample(self.upper_batch_size)
            dqn_metrics = self.upper.update(upper_batch)
            merged.update(dqn_metrics)

        return merged

    def save(self, ckpt_path: str | Path) -> None:
        ckpt = {
            "upper": self.upper.state_dict(),
            "lower": self.lower.state_dict(),
            "context_encoder": self.context_encoder.state_dict(),
            "context_predictor": self.context_predictor.state_dict(),
            "context_optim": self.context_optim.state_dict(),
            "safety": self.safety.state_dict(),
            "global_step": self.global_step,
            "rollout_step": int(self.rollout_step),
            "current_meta_iter": int(self.current_meta_iter),
            "alignment_meta": self._alignment_meta(pre_alignment=False),
        }
        torch.save(ckpt, ckpt_path)

    def load(self, ckpt_path: str | Path) -> None:
        try:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location=self.device)
        self.upper.load_state_dict(ckpt["upper"])
        self.lower.load_state_dict(ckpt["lower"])
        self.context_encoder.load_state_dict(ckpt["context_encoder"])
        if "context_predictor" in ckpt:
            try:
                self.context_predictor.load_state_dict(ckpt["context_predictor"])
            except RuntimeError:
                # Allow loading older predictor heads with different output dimensions.
                pass
        if "context_optim" in ckpt:
            try:
                self.context_optim.load_state_dict(ckpt["context_optim"])
            except ValueError:
                # Allow loading older checkpoints saved before predictor params were added.
                pass
        if "safety" in ckpt:
            self.safety.load_state_dict(ckpt["safety"])
        self.loaded_alignment_meta = dict(
            ckpt.get(
                "alignment_meta",
                self._alignment_meta(pre_alignment=True),
            )
        )
        if "pre_alignment" not in self.loaded_alignment_meta:
            self.loaded_alignment_meta["pre_alignment"] = True
        self.global_step = int(ckpt.get("global_step", 0))
        self.rollout_step = int(ckpt.get("rollout_step", 0))
        self.current_meta_iter = int(ckpt.get("current_meta_iter", self.current_meta_iter))
