from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from tchhmrl.envs.task_contract import task_defaults_from_cfg
from tchhmrl.envs.task_sampler import TaskSampler
from tchhmrl.envs.uw_slipt_env import MultiTxUwSliptEnv
from tchhmrl.safety.safety_layer import SafetyLayer, raw_from_frac01
from tchhmrl.utils.config import resolve_device
from tchhmrl.utils.logger import Logger
from tchhmrl.utils.seed import set_seed


@dataclass
class PolicyEpisodeStats:
    reward: float
    se: float
    eh: float
    cost: float
    violation_rate: float
    length: float
    temp_max: float
    bus_utilization: float
    online_latency_ms: float


def current_action_from_frac(
    current_frac: np.ndarray,
    rho: float,
    tau: float,
    action_decode_mode: str,
) -> np.ndarray:
    frac = np.concatenate(
        [
            np.asarray(current_frac, dtype=np.float32).reshape(3),
            np.asarray([rho, tau], dtype=np.float32),
        ],
        axis=0,
    )
    return raw_from_frac01(frac, action_decode_mode).astype(np.float32)


def expected_step_metrics(env: MultiTxUwSliptEnv, safe: Dict[str, object]) -> Dict[str, float | np.ndarray]:
    """Deterministic one-step prediction from the current observable state.

    This mirrors the environment reward calculation but intentionally omits all
    post-action random draws, channel updates, and observation noise. It is used
    by non-RL optimizers to avoid future-disturbance oracle behavior.
    """
    currents = np.asarray(safe["currents_exec"], dtype=np.float32).reshape(3)
    mode = int(safe["mode_exec"])
    boost_combo = int(safe["boost_combo_exec"])
    rho = float(safe["rho_exec"])
    tau = float(safe["tau_exec"])

    tx_signal = env._compute_tx_signal(currents)
    signal = float(np.sum(tx_signal))
    signal_led = float(np.sum(tx_signal * env.tx_is_led))
    signal_ld = float(np.sum(tx_signal * env.tx_is_ld))
    noise_power = env.noise_floor + env.noise_led_coeff * abs(signal_led) + env.noise_ld_coeff * abs(signal_ld)

    se_tx_weight = env._tx_vector(env.se_led_weight, env.se_ld_weight)
    eh_tx_weight = env._tx_vector(env.eh_led_weight, env.eh_ld_weight)
    info_signal = float(np.sum(tx_signal * se_tx_weight))
    eh_input = float(np.sum(tx_signal * eh_tx_weight))
    snr = max(info_signal / max(noise_power, 1.0e-6), 1.0e-6)
    mode_se = env._mode_gain(mode, env.mode_se_gain)
    mode_eh = env._mode_gain(mode, env.mode_eh_gain)

    if mode == 0:
        info_share, eh_share = 1.0 - rho, rho
    elif mode == 1:
        info_share, eh_share = tau, 1.0 - tau
    else:
        info_share = tau * (1.0 - rho)
        eh_share = 1.0 - info_share
    info_share = float(np.clip(info_share, 0.0, 1.0))
    eh_share = float(np.clip(eh_share, 0.0, 1.0))

    qos_rate = float(mode_se * info_share * np.log2(1.0 + snr))
    eh_input_eff = float(mode_eh * eh_share * eh_input)
    eh_diag = env._compute_eh_metric(eh_input_eff)
    eh_metric = float(eh_diag["eh_metric"])

    thermal_coeff = env._tx_vector(env.thermal_led_coeff, env.thermal_ld_coeff)
    temps_before = env.temps.copy().astype(np.float32)
    thermal_source_term = env._thermal_coupling_term(temps_before)
    thermal_base = (1.0 - env.gamma) * temps_before + env.gamma * env.amb_temp + thermal_source_term
    temps_next = (thermal_base + env.delta * thermal_coeff * (currents**2)).astype(np.float32)
    thermal_violation_vec = np.maximum(temps_next - env.thermal_safe, 0.0).astype(np.float32)
    qos_violation = float(max(env.qos_min_rate - qos_rate, 0.0))
    cost_vec = np.concatenate([np.asarray([qos_violation], dtype=np.float32), thermal_violation_vec], axis=0)
    cost = float(np.sum(cost_vec))
    power_penalty = float(np.sum(currents**2))

    delta_curr_norm = (currents - env.prev_currents) / np.maximum(env.current_max, 1.0e-6)
    smooth_raw = float(np.mean(delta_curr_norm**2) + 0.5 * ((rho - env.prev_rho) ** 2 + (tau - env.prev_tau) ** 2))
    smooth_penalty = env.action_smooth_weight * smooth_raw
    mode_switch = float(mode != env.prev_mode)
    boost_switch = float(boost_combo != env.prev_boost)
    switch_penalty = env.mode_switch_penalty * mode_switch + env.boost_switch_penalty * boost_switch
    temp_peak = float(np.max(temps_next))
    margin_norm = float(np.clip((env.thermal_safe - temp_peak) / max(env.thermal_safe, 1.0e-6), 0.0, 1.0))
    margin_reward = env.thermal_margin_weight * margin_norm
    se = float(env.se_weight * qos_rate)
    eh = float(env.eh_weight * eh_metric)
    reward = float(
        se
        + eh
        + margin_reward
        - env.cost_weight * cost
        - env.power_weight * power_penalty
        - smooth_penalty
        - switch_penalty
    )
    return {
        "reward": reward,
        "se": se,
        "eh": eh,
        "cost": cost,
        "cost_vec": cost_vec.astype(np.float32),
        "qos_rate": qos_rate,
        "eh_metric": eh_metric,
        "eh_input_eff": eh_input_eff,
        "snr": float(snr),
        "signal": signal,
        "temps_next": temps_next,
        "thermal_violation": float(np.sum(thermal_violation_vec)),
        "bus_utilization": float(np.sum(currents) / max(env.bus_current_max, 1.0e-6)),
    }


class BasePaperBaseline:
    """Small baseline runner interface shared by adapted paper baselines."""

    baseline_family = "paper_baseline"

    def __init__(self, cfg: Dict):
        self.cfg = copy.deepcopy(cfg)
        self.cfg.setdefault("context", {})["enabled"] = False
        self.cfg.setdefault("agent", {})["z_dim"] = 0
        self.cfg.setdefault("meta", {})["explicit_inner_outer"] = False
        self.cfg.setdefault("meta", {})["query_updates_enabled"] = False
        self.cfg.setdefault("meta", {})["dual_enabled"] = False
        seed = int(self.cfg["experiment"]["seed"])
        set_seed(seed)
        self.device = resolve_device(str(self.cfg["experiment"].get("device", "auto")))
        self.safety = SafetyLayer(self.cfg)
        self.task_sampler = TaskSampler(
            copy.deepcopy(self.cfg["sampler"]),
            seed=seed,
            task_defaults=task_defaults_from_cfg(self.cfg),
        )
        self.logger = Logger(
            log_dir=self.cfg["experiment"]["log_dir"],
            run_name=self.cfg["experiment"]["run_name"],
        )
        self.ckpt_dir = Path(self.cfg["experiment"]["log_dir"]) / self.cfg["experiment"]["run_name"] / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.global_step = 0
        self.safety_mem = {"current_boost": 0, "dwell_count": self.cfg["safety"]["min_dwell_steps"]}

    @property
    def metadata(self) -> Dict[str, object]:
        return dict(self.cfg.get("baseline_metadata", {}))

    def reset_episode_state(self) -> None:
        self.safety_mem = {"current_boost": 0, "dwell_count": self.cfg["safety"]["min_dwell_steps"]}

    def _empty_latent(self) -> np.ndarray:
        return np.zeros((int(self.cfg["agent"].get("z_dim", 0)),), dtype=np.float32)

    def _project_raw_action(
        self,
        env: MultiTxUwSliptEnv,
        upper_raw: int,
        lower_raw: np.ndarray,
        *,
        commit: bool,
    ) -> Tuple[Dict[str, object], Dict[str, int]]:
        mem = self.safety_mem if commit else dict(self.safety_mem)
        safe, new_mem = self.safety.project_np(
            upper_raw,
            np.asarray(lower_raw, dtype=np.float32),
            temps=env.temps.copy().astype(np.float32),
            amb_temp=float(env.amb_temp),
            gamma=float(env.gamma),
            delta=float(env.delta),
            mem=mem,
        )
        if commit:
            self.safety_mem = new_mem
        return safe, new_mem

    def _action_from_safe(
        self,
        upper_raw: int,
        lower_raw: np.ndarray,
        safe: Dict[str, object],
        *,
        aux_extra: Dict[str, object] | None = None,
    ) -> Tuple[Dict[str, object], Dict[str, object]]:
        currents = np.asarray(safe["currents_exec"], dtype=np.float32)
        action = {
            "upper_idx": int(upper_raw),
            "upper_idx_exec": int(safe["upper_idx_exec"]),
            "boost_combo_exec": int(safe["boost_combo_exec"]),
            "mode_exec": int(safe["mode_exec"]),
            "currents_exec": currents,
            "rho_exec": np.asarray([safe["rho_exec"]], dtype=np.float32),
            "tau_exec": np.asarray([safe["tau_exec"]], dtype=np.float32),
        }
        aux = {
            "z": self._empty_latent(),
            "upper_idx_raw": int(upper_raw),
            "upper_idx_exec": int(safe["upper_idx_exec"]),
            "upper_idx_train": int(safe["upper_idx_exec"]),
            "upper_idx_safety_raw": int(upper_raw),
            "boost_combo_exec": int(safe["boost_combo_exec"]),
            "mode_exec": int(safe["mode_exec"]),
            "act_raw": np.asarray(lower_raw, dtype=np.float32),
            "act_exec": np.concatenate(
                [currents, np.asarray([safe["rho_exec"], safe["tau_exec"]], dtype=np.float32)]
            ).astype(np.float32),
            "macro_new": True,
            "hold_left": 0,
            "selected_action_contract": str(self.metadata.get("selected_action_contract", "")),
            "online_latency_ms": 0.0,
        }
        for key, value in safe.items():
            if key in {
                "t_pred",
                "thermal_scale",
                "thermal_soft_scale",
                "thermal_cutoff_scale",
                "thermal_cap_current",
                "thermal_cap_scale",
                "thermal_cap_margin_c",
                "thermal_source_model",
                "thermal_source_term",
                "thermal_base",
                "thermal_pred_temp",
                "thermal_pred_margin",
                "thermal_model",
                "thermal_coupling_matrix_hash",
                "safety_projection_version",
                "thermal_margin_min",
                "action_decode_mode",
                "raw_current_frac",
                "rho_raw_decoded",
                "tau_raw_decoded",
                "raw_current_total",
                "masked_current_total",
                "bus_projected_current_total",
                "projected_current_total",
                "projection_compression_ratio",
                "thermal_coupling_term",
                "thermal_base_coupled",
            }:
                aux[key] = value
        if aux_extra:
            aux.update(aux_extra)
        return action, aux

    def act(self, obs: np.ndarray, env: MultiTxUwSliptEnv, eval_mode: bool = False) -> tuple[Dict, Dict]:
        raise NotImplementedError

    def record_transition(
        self,
        obs: np.ndarray,
        aux: Dict[str, object],
        reward: float,
        done: bool,
        next_obs: np.ndarray,
        info: Dict[str, object],
    ) -> None:
        return None

    def learn(self) -> Dict[str, float]:
        return {}

    def after_training_iteration(self) -> Dict[str, float]:
        return {}

    def _run_episode(self, env: MultiTxUwSliptEnv, train: bool) -> PolicyEpisodeStats:
        obs, _ = env.reset()
        self.reset_episode_state()
        done = False
        ep_reward = ep_se = ep_eh = ep_cost = ep_viol = 0.0
        ep_temp_max = ep_bus_util = ep_latency = 0.0
        ep_len = 0
        while not done:
            action, aux = self.act(obs, env, eval_mode=not train)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            if train:
                self.record_transition(obs, aux, float(reward), done, next_obs, info)
                self.learn()
            cost_vec = np.asarray(info.get("cost_vec", [info["cost"]]), dtype=np.float32)
            ep_reward += float(reward)
            ep_se += float(info["se"])
            ep_eh += float(info["eh"])
            ep_cost += float(info["cost"])
            ep_viol += float(np.any(cost_vec > 0.0))
            ep_temp_max += float(np.max(info["temps"]))
            ep_bus_util += float(info.get("bus_utilization", 0.0))
            ep_latency += float(aux.get("online_latency_ms", 0.0))
            ep_len += 1
            obs = next_obs
            self.global_step += 1
        denom = max(ep_len, 1)
        return PolicyEpisodeStats(
            reward=ep_reward / denom,
            se=ep_se / denom,
            eh=ep_eh / denom,
            cost=ep_cost / denom,
            violation_rate=ep_viol / denom,
            length=float(ep_len),
            temp_max=ep_temp_max / denom,
            bus_utilization=ep_bus_util / denom,
            online_latency_ms=ep_latency / denom,
        )

    def train(self, meta_iters: int | None = None) -> Path:
        meta_cfg = self.cfg["meta"]
        meta_iters = int(meta_cfg.get("meta_iters", 1) if meta_iters is None else meta_iters)
        if meta_iters <= 0:
            return self.write_noop_training_csv()
        episodes_per_task = max(1, int(meta_cfg.get("support_episodes", 0)) + int(meta_cfg.get("query_episodes", 0)))
        n_tasks = int(meta_cfg.get("n_tasks_per_iter", 1))
        for it in range(1, meta_iters + 1):
            stats: List[PolicyEpisodeStats] = []
            for task in self.task_sampler.sample(n_tasks):
                env = MultiTxUwSliptEnv(self.cfg, overrides=task.to_env_overrides())
                for _ in range(episodes_per_task):
                    stats.append(self._run_episode(env, train=True))
            learn_metrics = self.after_training_iteration()
            row = {
                "iter": float(it),
                "support_reward": float(np.mean([s.reward for s in stats])) if stats else 0.0,
                "support_se": float(np.mean([s.se for s in stats])) if stats else 0.0,
                "support_eh": float(np.mean([s.eh for s in stats])) if stats else 0.0,
                "support_cost": float(np.mean([s.cost for s in stats])) if stats else 0.0,
                "support_violation_rate": float(np.mean([s.violation_rate for s in stats])) if stats else 0.0,
                "query_reward": float(np.mean([s.reward for s in stats])) if stats else 0.0,
                "query_se": float(np.mean([s.se for s in stats])) if stats else 0.0,
                "query_eh": float(np.mean([s.eh for s in stats])) if stats else 0.0,
                "query_cost": float(np.mean([s.cost for s in stats])) if stats else 0.0,
                "query_violation_rate": float(np.mean([s.violation_rate for s in stats])) if stats else 0.0,
                "query_temp_max": float(np.mean([s.temp_max for s in stats])) if stats else 0.0,
                "query_bus_utilization": float(np.mean([s.bus_utilization for s in stats])) if stats else 0.0,
                "query_online_latency_ms": float(np.mean([s.online_latency_ms for s in stats])) if stats else 0.0,
                "lambda": 0.0,
                "curriculum_stage": "base",
                "outer_step_size": 0.0,
            }
            row.update({k: float(v) for k, v in learn_metrics.items()})
            self.logger.log(row)
            if it % 10 == 0 or it == meta_iters:
                self.save(self.ckpt_dir / f"iter_{it}.pt")
        return self.logger.csv_path

    def save(self, ckpt_path: str | Path) -> None:
        Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"baseline_metadata": self.metadata, "baseline_family": self.baseline_family}, ckpt_path)

    def load(self, ckpt_path: str | Path) -> None:
        return None

    def write_noop_training_csv(self) -> Path:
        out = Path(self.cfg["experiment"]["log_dir"]) / self.cfg["experiment"]["run_name"] / "training.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "iter": 0.0,
                    "query_reward": 0.0,
                    "query_se": 0.0,
                    "query_eh": 0.0,
                    "query_cost": 0.0,
                    "query_violation_rate": 0.0,
                    "query_online_latency_ms": 0.0,
                }
            ]
        ).to_csv(out, index=False)
        return out


def monotonic_latency_ms(start: float) -> float:
    return float((time.perf_counter() - start) * 1000.0)
