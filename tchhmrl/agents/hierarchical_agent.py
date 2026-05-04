from __future__ import annotations

import copy
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
from tchhmrl.envs.task_contract import build_task_summary_v2, is_formally_comparable_record
from tchhmrl.safety.safety_layer import SafetyLayer


class HierarchicalAgent:
    def __init__(self, cfg: Dict, device: torch.device):
        self.cfg = cfg
        self.device = device

        self.safety = SafetyLayer(cfg)
        self.upper = UpperDQN(cfg, device)
        self.lower = LowerSAC(cfg, self.safety, device)

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
        self.warmup_steps = int(cfg["agent"]["warmup_steps"])
        self.lower_updates_per_step = int(cfg["agent"].get("lower_updates_per_step", 1))
        self.upper_update_every = int(cfg["agent"].get("upper_update_every", 1))
        self.upper_warmup_steps = int(
            cfg["agent"].get(
                "upper_warmup_steps",
                max(self.batch_size, self.warmup_steps // max(1, int(cfg["agent"].get("upper_hold_steps", 1)))),
            )
        )
        self.upper_hold_steps = int(cfg["agent"].get("upper_hold_steps", 1))

        hard_cfg = cfg.get("buffer", {}).get("hard_mining", {})
        self.hard_mining_enabled = bool(hard_cfg.get("enabled", False))
        self.hard_fraction = float(hard_cfg.get("fraction", 0.0))
        self.hard_cost_w = float(hard_cfg.get("cost_w", 1.0))
        self.hard_mode_switch_bonus = float(hard_cfg.get("mode_switch_bonus", 0.5))
        self.hard_boost_switch_bonus = float(hard_cfg.get("boost_switch_bonus", 0.5))

        self.global_step = 0
        self.safety_mem = {"current_boost": 0, "dwell_count": cfg["safety"]["min_dwell_steps"]}
        self.upper_mem = {"upper_idx": 0, "hold_left": 0}
        self.upper_plan: Optional[int] = None

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
        reward_raw = float(tr.get("reward_raw", tr.get("reward", 0.0)))
        return np.concatenate(
            [
                np.asarray([reward_raw], dtype=np.float32),
                self._context_cost_vec(tr),
            ]
        ).astype(np.float32)

    def _task_params_from_transition(self, tr: Dict) -> np.ndarray:
        return build_task_summary_v2(tr)

    def _alignment_meta(self, *, pre_alignment: bool | None = None) -> Dict[str, object]:
        return {
            "alignment_version": self.alignment_version,
            "task_summary_version": self.task_summary_version,
            "pre_alignment": bool(self.pre_alignment if pre_alignment is None else pre_alignment),
        }

    def is_formally_comparable(self) -> bool:
        return is_formally_comparable_record(self.loaded_alignment_meta)

    @staticmethod
    def _mean_metrics(metrics_list: list[Dict[str, float]]) -> Dict[str, float]:
        if not metrics_list:
            return {}
        keys = metrics_list[0].keys()
        return {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}

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

    def reset_episode_state(self) -> None:
        self.reset_rollout_state(clear_context=True)

    def snapshot_train_state(self) -> Dict:
        return {
            "upper": copy.deepcopy(self.upper.state_dict()),
            "lower": copy.deepcopy(self.lower.state_dict()),
            "context_encoder": copy.deepcopy(self.context_encoder.state_dict()),
            "context_predictor": copy.deepcopy(self.context_predictor.state_dict()),
            "context_optim": copy.deepcopy(self.context_optim.state_dict()),
            "global_step": int(self.global_step),
        }

    def restore_train_state(self, state: Dict) -> None:
        self.upper.load_state_dict(copy.deepcopy(state["upper"]))
        self.lower.load_state_dict(copy.deepcopy(state["lower"]))
        self.context_encoder.load_state_dict(copy.deepcopy(state["context_encoder"]))
        self.context_predictor.load_state_dict(copy.deepcopy(state["context_predictor"]))
        self.context_optim.load_state_dict(copy.deepcopy(state["context_optim"]))
        self.global_step = int(state.get("global_step", 0))

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
        self._blend_module_state(self.lower.actor, lower_actor, step_size)
        self._blend_module_state(self.lower.q1, lower_q1, step_size)
        self._blend_module_state(self.lower.q2, lower_q2, step_size)
        self._blend_module_state(self.lower.q1_tgt, lower_q1_tgt, step_size)
        self._blend_module_state(self.lower.q2_tgt, lower_q2_tgt, step_size)
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
            z, _, _, _ = self.context_encoder.infer(seq)
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
        exec_map = self.safety.raw_to_exec_map(self.safety_mem)
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
                )
            self.upper_mem["upper_idx"] = int(upper_idx_raw)
            self.upper_mem["hold_left"] = max(1, self.upper_hold_steps)
        else:
            upper_idx_raw = int(self.upper_mem["upper_idx"])
        self.upper_mem["hold_left"] = max(0, int(self.upper_mem["hold_left"]) - 1)

        boost_preview, mode_preview = self.safety.preview_exec(upper_idx_raw, self.safety_mem)
        upper_idx_exec = self.safety.encode_exec(boost_preview, mode_preview)
        lower_raw = self.lower.select_action(obs, z, upper_idx=upper_idx_exec, eval_mode=eval_mode)

        safe, self.safety_mem = self.safety.project_np(
            upper_idx_raw,
            lower_raw,
            temps=temps,
            amb_temp=amb_temp,
            gamma=gamma,
            delta=delta,
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

        aux = {
            "z": z,
            "upper_idx_raw": int(upper_idx_raw),
            "upper_idx_exec": int(safe["upper_idx_exec"]),
            "boost_combo_exec": int(safe["boost_combo_exec"]),
            "mode_exec": int(safe["mode_exec"]),
            "act_exec": exec_vec,
            "act_raw": lower_raw.astype(np.float32),
            "t_pred": safe["t_pred"],
            "macro_new": bool(macro_new),
            "hold_left": int(self.upper_mem["hold_left"]),
        }
        return action, aux

    def observe(self, transition: Dict) -> None:
        self.observe_lower(transition)

    def preview_next_macro(
        self,
        next_obs: np.ndarray,
        z_next: np.ndarray,
        eval_mode: bool = False,
        commit_plan: bool = False,
    ) -> Dict[str, int]:
        macro_new_next = self.upper_mem["hold_left"] <= 0
        next_exec_map = self.safety.raw_to_exec_map(self.safety_mem)
        if macro_new_next:
            upper_idx_next_raw = int(
                self.upper.select_action(
                    next_obs.astype(np.float32),
                    z_next.astype(np.float32),
                    t=self.global_step + 1,
                    eval_mode=eval_mode,
                    exec_map=next_exec_map,
                )
            )
            if commit_plan:
                self.upper_plan = upper_idx_next_raw
        else:
            upper_idx_next_raw = int(self.upper_mem["upper_idx"])

        boost_next, mode_next = self.safety.preview_exec(upper_idx_next_raw, self.safety_mem)
        return {
            "upper_idx_raw_next": int(upper_idx_next_raw),
            "upper_idx_exec_next": int(self.safety.encode_exec(boost_next, mode_next)),
            "boost_combo_exec_next": int(boost_next),
            "mode_exec_next": int(mode_next),
            "next_exec_map": next_exec_map.astype(np.float32),
        }

    def observe_lower(
        self,
        transition: Dict,
        next_macro_fn: Optional[Callable[[np.ndarray], Dict[str, float | int]]] = None,
    ) -> np.ndarray:
        task_params = self._task_params_from_transition(transition)
        self.episode.add(
            {
                "obs": transition["obs"],
                "upper_idx_exec": transition.get("upper_idx_exec", transition.get("upper_idx", 0.0)),
                "boost_combo_exec": transition.get("boost_combo_exec", 0.0),
                "mode_exec": transition.get("mode_exec", 0.0),
                "act_exec": transition["act_exec"],
                "reward": transition["reward"],
                "reward_raw": transition.get("reward_raw", transition["reward"]),
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
                np.tile(np.arange(12, dtype=np.float32).reshape(1, -1), (1, 1)).squeeze(0),
            )
        self.replay.add(tr)
        return z_next

    def observe_upper(self, transition: Dict) -> None:
        self.upper_replay.add(transition)

    def update_context_encoder(self) -> Dict[str, float]:
        if not self.context_enabled:
            return {}
        if len(self.episode) < 3:
            return {}

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
        z, kl, _, _ = self.context_encoder.infer(seq)
        task_rows = np.stack(
            [
                np.asarray(tr.get("task_params", np.zeros(self.context_task_dim, dtype=np.float32)), dtype=np.float32)
                for tr in self.episode.as_list()
            ],
            axis=0,
        )
        target = (
            torch.tensor(task_rows.mean(axis=0), dtype=torch.float32, device=self.device)
            .view(1, self.context_task_dim)
            .detach()
        )
        pred = self.context_predictor(z)
        pred_loss = F.mse_loss(pred, target)
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
        }

    def learn(self) -> Dict[str, float]:
        self.global_step += 1

        merged = {}

        if len(self.replay) >= max(self.batch_size, self.warmup_steps):
            sac_list = []
            ctx_list = []
            for _ in range(max(1, self.lower_updates_per_step)):
                lower_batch = self.replay.sample(
                    self.batch_size,
                    hard_fraction=self.hard_fraction if self.hard_mining_enabled else 0.0,
                    scorer=self._hard_score if self.hard_mining_enabled else None,
                )
                sac_list.append(self.lower.update(lower_batch))
                ctx_list.append(self.update_context_encoder())
            merged.update(self._mean_metrics(sac_list))
            merged.update(self._mean_metrics(ctx_list))

        if (
            len(self.upper_replay) >= max(self.batch_size, self.upper_warmup_steps)
            and self.global_step % max(1, self.upper_update_every) == 0
        ):
            upper_batch = self.upper_replay.sample(self.batch_size)
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
            "global_step": self.global_step,
            "alignment_meta": self._alignment_meta(pre_alignment=False),
        }
        torch.save(ckpt, ckpt_path)

    def load(self, ckpt_path: str | Path) -> None:
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
        self.loaded_alignment_meta = dict(
            ckpt.get(
                "alignment_meta",
                self._alignment_meta(pre_alignment=True),
            )
        )
        if "pre_alignment" not in self.loaded_alignment_meta:
            self.loaded_alignment_meta["pre_alignment"] = True
        self.global_step = int(ckpt.get("global_step", 0))
