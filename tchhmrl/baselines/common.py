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


def _model_scalar(model: Dict[str, object] | None, key: str, default: float) -> float:
    val = None if not model else model.get(key)
    if val is None or (isinstance(val, str) and val == ""):
        return float(default)
    return float(val)


def _model_vector(
    model: Dict[str, object] | None,
    key: str,
    default: np.ndarray,
    *,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    default_arr = np.asarray(default, dtype=np.float32)
    val = None if not model else model.get(key)
    if val is None or (isinstance(val, str) and val == ""):
        out = default_arr.copy()
    else:
        out = np.asarray(val, dtype=np.float32)
    if shape is not None:
        out = out.reshape(shape)
    return out.astype(np.float32)


def mpc_nominal_scoring_model_from_cfg(cfg: Dict, env: MultiTxUwSliptEnv) -> Dict[str, object]:
    """Build the pre-registered model used by nominal-model MPC scoring.

    The snapshot intentionally comes from the configuration rather than the
    task-reset values inside ``env``. This gives MPC a fixed model for candidate
    scoring while the final selected action is still executed in the real task
    environment.
    """
    env_cfg = cfg.get("env", {}) or {}
    safety_cfg = cfg.get("safety", {}) or {}
    hybrid_cfg = env_cfg.get("hybrid", {}) or {}
    opts = cfg.get("baselines", {}).get("mpc_grid", {}) or {}
    nominal = dict(opts.get("nominal_model", {}) or {})

    gamma = float(nominal.get("gamma", env_cfg.get("gamma", env.gamma)))
    delta = float(nominal.get("delta", env_cfg.get("delta", env.delta)))
    amb_temp = float(nominal.get("amb_temp", env_cfg.get("amb_temp", env.amb_temp)))
    thermal_safe = float(nominal.get("thermal_safe", env_cfg.get("thermal_safe", env.thermal_safe)))
    qos_min_rate = float(nominal.get("qos_min_rate", env_cfg.get("qos_min_rate", env.qos_min_rate)))
    current_max = np.asarray(nominal.get("current_max", safety_cfg.get("current_max", env.current_max)), dtype=np.float32)
    tx_is_led = np.asarray(env.tx_is_led, dtype=np.float32)
    tx_is_ld = np.asarray(env.tx_is_ld, dtype=np.float32)

    def _tx_vector_cfg(led_key: str, ld_key: str, env_led: float, env_ld: float) -> np.ndarray:
        led = float(nominal.get(led_key, hybrid_cfg.get(led_key, env_led)))
        ld = float(nominal.get(ld_key, hybrid_cfg.get(ld_key, env_ld)))
        return (tx_is_led * led + tx_is_ld * ld).astype(np.float32)

    return {
        "model_name": "nominal_model",
        "gamma": gamma,
        "delta": delta,
        "amb_temp": amb_temp,
        "thermal_safe": thermal_safe,
        "qos_min_rate": qos_min_rate,
        "current_max": current_max,
        "bus_current_max": float(nominal.get("bus_current_max", safety_cfg.get("bus_current_max", env.bus_current_max))),
        "eta_tx": _tx_vector_cfg("eta_led", "eta_ld", env.eta_led, env.eta_ld),
        "se_tx_weight": _tx_vector_cfg("se_led_weight", "se_ld_weight", env.se_led_weight, env.se_ld_weight),
        "eh_tx_weight": _tx_vector_cfg("eh_led_weight", "eh_ld_weight", env.eh_led_weight, env.eh_ld_weight),
        "thermal_coeff": _tx_vector_cfg("thermal_led_coeff", "thermal_ld_coeff", env.thermal_led_coeff, env.thermal_ld_coeff),
        "noise_floor": float(nominal.get("noise_floor", hybrid_cfg.get("noise_floor", env.noise_floor))),
        "noise_led_coeff": float(nominal.get("noise_led_coeff", hybrid_cfg.get("noise_led_coeff", env.noise_led_coeff))),
        "noise_ld_coeff": float(nominal.get("noise_ld_coeff", hybrid_cfg.get("noise_ld_coeff", env.noise_ld_coeff))),
        "mode_se_gain": np.asarray(nominal.get("mode_se_gain", env_cfg.get("mode_se_gain", env.mode_se_gain)), dtype=np.float32),
        "mode_eh_gain": np.asarray(nominal.get("mode_eh_gain", env_cfg.get("mode_eh_gain", env.mode_eh_gain)), dtype=np.float32),
        "channel_gain_scale": float(nominal.get("channel_gain_scale", 1.0)),
        "thermal_source_term": np.asarray(nominal.get("thermal_source_term", np.zeros(3, dtype=np.float32)), dtype=np.float32),
        "uses_true_gamma_delta": False,
        "uses_future_disturbance": False,
    }


def expected_step_metrics(
    env: MultiTxUwSliptEnv,
    safe: Dict[str, object],
    *,
    model: Dict[str, object] | None = None,
) -> Dict[str, float | np.ndarray]:
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

    eta_tx = _model_vector(model, "eta_tx", env._tx_vector(env.eta_led, env.eta_ld), shape=(3,))
    channel_gain_scale = _model_scalar(model, "channel_gain_scale", 1.0)
    tx_signal = currents * eta_tx * env.channel.astype(np.float32) * float(channel_gain_scale)
    signal = float(np.sum(tx_signal))
    signal_led = float(np.sum(tx_signal * env.tx_is_led))
    signal_ld = float(np.sum(tx_signal * env.tx_is_ld))
    noise_floor = _model_scalar(model, "noise_floor", env.noise_floor)
    noise_led_coeff = _model_scalar(model, "noise_led_coeff", env.noise_led_coeff)
    noise_ld_coeff = _model_scalar(model, "noise_ld_coeff", env.noise_ld_coeff)
    noise_power = noise_floor + noise_led_coeff * abs(signal_led) + noise_ld_coeff * abs(signal_ld)

    se_tx_weight = _model_vector(model, "se_tx_weight", env._tx_vector(env.se_led_weight, env.se_ld_weight), shape=(3,))
    eh_tx_weight = _model_vector(model, "eh_tx_weight", env._tx_vector(env.eh_led_weight, env.eh_ld_weight), shape=(3,))
    info_signal = float(np.sum(tx_signal * se_tx_weight))
    eh_input = float(np.sum(tx_signal * eh_tx_weight))
    snr = max(info_signal / max(noise_power, 1.0e-6), 1.0e-6)
    mode_se_gain = _model_vector(model, "mode_se_gain", env.mode_se_gain)
    mode_eh_gain = _model_vector(model, "mode_eh_gain", env.mode_eh_gain)
    mode_se = env._mode_gain(mode, mode_se_gain)
    mode_eh = env._mode_gain(mode, mode_eh_gain)

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

    thermal_coeff = _model_vector(
        model,
        "thermal_coeff",
        env._tx_vector(env.thermal_led_coeff, env.thermal_ld_coeff),
        shape=(3,),
    )
    temps_before = env.temps.copy().astype(np.float32)
    thermal_source_term = _model_vector(model, "thermal_source_term", env._thermal_coupling_term(temps_before), shape=(3,))
    gamma = _model_scalar(model, "gamma", env.gamma)
    delta = _model_scalar(model, "delta", env.delta)
    amb_temp = _model_scalar(model, "amb_temp", env.amb_temp)
    thermal_safe = _model_scalar(model, "thermal_safe", env.thermal_safe)
    qos_min_rate = _model_scalar(model, "qos_min_rate", env.qos_min_rate)
    current_max = _model_vector(model, "current_max", env.current_max, shape=(3,))
    bus_current_max = _model_scalar(model, "bus_current_max", env.bus_current_max)
    thermal_base = (1.0 - gamma) * temps_before + gamma * amb_temp + thermal_source_term
    temps_next = (thermal_base + delta * thermal_coeff * (currents**2)).astype(np.float32)
    thermal_violation_vec = np.maximum(temps_next - thermal_safe, 0.0).astype(np.float32)
    qos_violation = float(max(qos_min_rate - qos_rate, 0.0))
    cost_vec = np.concatenate([np.asarray([qos_violation], dtype=np.float32), thermal_violation_vec], axis=0)
    cost = float(np.sum(cost_vec))
    power_penalty = float(np.sum(currents**2))

    delta_curr_norm = (currents - env.prev_currents) / np.maximum(current_max, 1.0e-6)
    smooth_raw = float(np.mean(delta_curr_norm**2) + 0.5 * ((rho - env.prev_rho) ** 2 + (tau - env.prev_tau) ** 2))
    smooth_penalty = env.action_smooth_weight * smooth_raw
    mode_switch = float(mode != env.prev_mode)
    boost_switch = float(boost_combo != env.prev_boost)
    switch_penalty = env.mode_switch_penalty * mode_switch + env.boost_switch_penalty * boost_switch
    temp_peak = float(np.max(temps_next))
    margin_norm = float(np.clip((thermal_safe - temp_peak) / max(thermal_safe, 1.0e-6), 0.0, 1.0))
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
        "bus_utilization": float(np.sum(currents) / max(bus_current_max, 1.0e-6)),
        "scoring_model": str((model or {}).get("model_name", "perfect_env")),
        "scoring_gamma": float(gamma),
        "scoring_delta": float(delta),
        "scoring_amb_temp": float(amb_temp),
        "scoring_thermal_safe": float(thermal_safe),
        "scoring_qos_min_rate": float(qos_min_rate),
        "true_gamma": float(env.gamma),
        "true_delta": float(env.delta),
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

    def _update_safety_estimator(self, temps_before: np.ndarray, info: Dict[str, object]) -> Dict[str, object]:
        currents = np.asarray(info.get("currents_exec", np.zeros(3, dtype=np.float32)), dtype=np.float32)
        temps_after = np.asarray(info.get("temps", temps_before), dtype=np.float32)
        amb_temp = float(info.get("amb_temp", self.cfg["env"].get("amb_temp", 10.0)))
        gamma = float(info.get("gamma", self.cfg["env"].get("gamma", 0.95)))
        thermal_base, _ = self.safety._thermal_base_np(
            np.asarray(temps_before, dtype=np.float32),
            amb_temp,
            gamma,
        )
        return self.safety.update_thermal_estimator(
            currents=currents,
            temps_before=np.asarray(temps_before, dtype=np.float32),
            temps_after=temps_after,
            thermal_base=thermal_base,
        )

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
            "rho_exec": float(safe["rho_exec"]),
            "tau_exec": float(safe["tau_exec"]),
            "env_rho_exec": float(safe["rho_exec"]),
            "paper_rho_exec": float(1.0 - float(safe["rho_exec"])),
            "paper_rho_equiv": float(1.0 - float(safe["rho_exec"])),
            "paper_tau_equiv": float(safe["tau_exec"]),
            "selected_env_rho": float(safe["rho_exec"]),
            "selected_paper_rho": float(1.0 - float(safe["rho_exec"])),
            "selected_tau": float(safe["tau_exec"]),
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
                "current_requested",
                "current_requested_pre_static_cap",
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
            temps_before = env.temps.copy().astype(np.float32)
            action, aux = self.act(obs, env, eval_mode=not train)
            next_obs, reward, terminated, truncated, info = env.step(action)
            self._update_safety_estimator(temps_before, info)
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
