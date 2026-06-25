from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from tchhmrl.agents.ddpg_lower import LowerDDPG
from tchhmrl.agents.dqn_upper import UpperDQN
from tchhmrl.baselines.common import BasePaperBaseline, expected_step_metrics
from tchhmrl.buffers.replay_buffer import ReplayBuffer
from tchhmrl.envs.uw_slipt_env import MultiTxUwSliptEnv


class DeepRATAssignmentPowerBaseline(BasePaperBaseline):
    """DeepRAT-style assignment plus power-allocation adapted baseline."""

    baseline_family = "deeprat_assignment_power"

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        opts = cfg.get("baselines", {}).get("deeprat_assignment_power", {})
        self.assignment_action_dim = int(opts.get("assignment_action_dim", 4))
        if self.assignment_action_dim != 4:
            raise ValueError("DeepRAT assignment-power adapted baseline requires exactly 4 assignment actions")
        self.cfg["agent"]["n_upper_actions"] = self.assignment_action_dim
        self.upper = UpperDQN(self.cfg, self.device)
        self.lower = LowerDDPG(self.cfg, self.safety, self.device)
        self.replay = ReplayBuffer(int(opts.get("replay_size", cfg.get("lower_ddpg", {}).get("replay_size", 100000))))
        self.upper_replay = ReplayBuffer(int(opts.get("upper_replay_size", cfg.get("upper_dqn", {}).get("replay_size", 2000))))
        self.batch_size = int(opts.get("batch_size", cfg.get("lower_ddpg", {}).get("batch_size", 64)))
        self.warmup_steps = int(opts.get("warmup_steps", cfg["agent"].get("warmup_steps", 100)))
        self.lower_updates_per_step = int(opts.get("lower_updates_per_step", 1))
        self.upper_update_every = int(opts.get("upper_update_every", cfg["agent"].get("upper_update_every", 1)))
        self.action_contract = "source_assignment_current_allocation_only"

    def _hy_exec_map(self) -> np.ndarray:
        exec_map = np.zeros(4, dtype=np.int64)
        for assignment_idx in range(4):
            hy_idx = int(assignment_idx * 3 + 2)
            boost_exec, _ = self.safety.preview_exec(hy_idx, mem=self.safety_mem)
            exec_map[assignment_idx] = int(boost_exec)
        return exec_map

    def act(self, obs: np.ndarray, env: MultiTxUwSliptEnv, eval_mode: bool = False) -> tuple[Dict, Dict]:
        z = self._empty_latent()
        exec_map = self._hy_exec_map()
        assignment_idx = self.upper.select_action(
            obs.astype(np.float32),
            z,
            t=self.global_step,
            eval_mode=eval_mode,
            exec_map=exec_map,
        )
        boost_combo = int(np.clip(assignment_idx, 0, 3))
        upper_raw = int(boost_combo * 3 + 2)
        boost_preview, _ = self.safety.preview_exec(upper_raw, self.safety_mem)
        upper_idx_exec = self.safety.encode_exec(boost_preview, 2)
        lower_raw = self.lower.select_action(obs, z, upper_idx=upper_idx_exec, eval_mode=eval_mode)
        safe, _ = self._project_raw_action(env, upper_raw, lower_raw, commit=True)
        if int(safe["mode_exec"]) != 2:
            raise RuntimeError("DeepRAT assignment-power adapted baseline must execute HY mode")
        predicted = expected_step_metrics(env, safe)
        action, aux = self._action_from_safe(
            upper_raw,
            lower_raw,
            safe,
            aux_extra={
                "upper_idx_train": int(boost_combo),
                "upper_idx_safety_raw": int(upper_raw),
                "source_assignment": int(boost_combo),
                "assignment_idx_train": int(boost_combo),
                "receiver_ratio_rule": "fixed_balanced_not_deeprat_core",
                "predicted_qos_rate": float(predicted["qos_rate"]),
                "predicted_eh_metric": float(predicted["eh_metric"]),
                "predicted_snr": float(predicted["snr"]),
                "predicted_bus_utilization": float(predicted["bus_utilization"]),
                "selected_action_contract": self.action_contract,
                "temps_before": env.temps.copy().astype(np.float32),
            },
        )
        return action, aux

    def record_transition(self, obs, aux, reward, done, next_obs, info) -> None:
        cost = float(info["cost"])
        cost_vec = np.asarray(info.get("cost_vec", [cost]), dtype=np.float32)
        transition = {
            "obs": obs.astype(np.float32),
            "next_obs": next_obs.astype(np.float32),
            "upper_idx_raw": float(aux["upper_idx_raw"]),
            "upper_idx_exec": float(aux["upper_idx_exec"]),
            "upper_idx_train": float(aux.get("upper_idx_train", aux["upper_idx_exec"])),
            "reward": float(reward),
            "reward_raw": float(reward),
            "reward_task": float(info.get("reward_task", reward)),
            "reward_benchmark": float(info.get("reward_benchmark", reward)),
            "reward_dual_penalized": float(info.get("reward_task", reward)),
            "done": float(done),
            "z": self._empty_latent(),
            "z_next": self._empty_latent(),
            "act_exec": np.asarray(aux["act_exec"], dtype=np.float32),
            "act_raw": np.asarray(aux["act_raw"], dtype=np.float32),
            "boost_combo_exec": float(aux["boost_combo_exec"]),
            "mode_exec": float(aux["mode_exec"]),
            "current_template_level_exec": 2.0,
            "temps": np.asarray(aux.get("temps_before", np.zeros(3)), dtype=np.float32),
            "next_temps": np.asarray(info.get("temps", np.zeros(3)), dtype=np.float32),
            "amb_temp": float(info["amb_temp"]),
            "gamma_env": float(info["gamma"]),
            "delta_env": float(info["delta"]),
            "cost": cost,
            "cost_vec": cost_vec,
            "upper_idx_exec_next": float(aux["upper_idx_exec"]),
            "upper_idx_train_next": float(aux.get("upper_idx_train", aux["upper_idx_exec"])),
            "boost_combo_exec_next": float(aux["boost_combo_exec"]),
            "mode_exec_next": 2.0,
            "current_template_level_exec_next": 2.0,
        }
        self.replay.add(transition)
        self.upper_replay.add(
            {
                "obs": obs.astype(np.float32),
                "next_obs": next_obs.astype(np.float32),
                "upper_idx_train": float(aux.get("upper_idx_train", aux["upper_idx_exec"])),
                "reward": float(reward),
                "done": float(done),
                "z": self._empty_latent(),
                "z_next": self._empty_latent(),
                "horizon": 1.0,
                "next_exec_map": self._hy_exec_map().astype(np.float32),
            }
        )

    def learn(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if len(self.replay) >= max(self.batch_size, self.warmup_steps):
            vals = []
            for _ in range(max(1, self.lower_updates_per_step)):
                vals.append(self.lower.update(self.replay.sample(self.batch_size)))
            for key in vals[0]:
                metrics[f"deeprat_{key}"] = float(np.mean([v[key] for v in vals]))
        if (
            len(self.upper_replay) >= max(self.batch_size, self.warmup_steps)
            and self.global_step % max(1, self.upper_update_every) == 0
        ):
            out = self.upper.update(self.upper_replay.sample(self.batch_size))
            metrics.update({f"deeprat_{k}": float(v) for k, v in out.items()})
        return metrics

    def save(self, ckpt_path):
        torch.save(
            {
                "baseline_metadata": self.metadata,
                "upper": self.upper.state_dict(),
                "lower": self.lower.state_dict(),
            },
            ckpt_path,
        )

    def load(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.upper.load_state_dict(ckpt["upper"])
        self.lower.load_state_dict(ckpt["lower"])
