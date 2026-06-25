from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class ThermalEstimatorSnapshot:
    gain_mean: np.ndarray
    gain_var: np.ndarray
    valid_count: np.ndarray
    temperature_slope: np.ndarray
    last_headroom: np.ndarray


class ThermalGainEstimator:
    """Online EMA estimator for per-source effective thermal gain.

    The estimator tracks the full effective heating gain in

        T_next = base_nominal + g_i * I_i^2.

    Here ``base_nominal`` uses the controller's calibrated cooling coefficient,
    not the task-level environment truth.  The task-specific ``delta`` is
    intentionally absent from this estimator interface so the controller cannot
    receive thermal dynamics oracle information through the safety layer.

    It only updates when the current provides enough excitation to identify
    the thermal response. The state is intentionally small so it can be copied
    and restored by support-gated rollback without extra training budget.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        *,
        n_tx: int,
        thermal_safe: float,
        initial_effective_gain: np.ndarray | None = None,
    ):
        self.cfg = dict(cfg or {})
        self.n_tx = int(n_tx)
        self.thermal_safe = float(thermal_safe)
        self.enabled = bool(self.cfg.get("enabled", False))
        self.excitation_threshold = float(self.cfg.get("excitation_threshold", 0.03))
        self.ema_alpha = float(self.cfg.get("ema_alpha", 0.10))
        self.innovation_clip = float(self.cfg.get("innovation_clip", 3.0))
        self.uncertainty_beta_min = float(self.cfg.get("uncertainty_beta_min", 0.5))
        self.uncertainty_beta_max = float(self.cfg.get("uncertainty_beta_max", 1.5))
        self.min_gain = float(self.cfg.get("min_gain", 0.5))
        self.max_gain = float(self.cfg.get("max_gain", 16.0))
        self.prior_std = float(self.cfg.get("initial_std", 0.0))
        if initial_effective_gain is None:
            configured = self.cfg.get("initial_effective_gain", self.cfg.get("initial_mean", 1.0))
            initial_effective_gain = np.asarray(configured, dtype=np.float32)
        initial_effective_gain = np.asarray(initial_effective_gain, dtype=np.float32).reshape(-1)
        if initial_effective_gain.size == 1:
            initial_effective_gain = np.full((self.n_tx,), float(initial_effective_gain[0]), dtype=np.float32)
        if initial_effective_gain.size != self.n_tx:
            raise ValueError("adaptive_thermal.initial_effective_gain must be scalar or length n_tx")

        self.gain_mean = np.clip(initial_effective_gain, self.min_gain, self.max_gain).astype(np.float32)
        self.gain_var = np.zeros((self.n_tx,), dtype=np.float32)
        self.valid_count = np.zeros((self.n_tx,), dtype=np.int64)
        self.temperature_slope = np.zeros((self.n_tx,), dtype=np.float32)
        self.last_headroom = np.full((self.n_tx,), np.nan, dtype=np.float32)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "gain_mean": self.gain_mean.copy(),
            "gain_var": self.gain_var.copy(),
            "valid_count": self.valid_count.copy(),
            "temperature_slope": self.temperature_slope.copy(),
            "last_headroom": self.last_headroom.copy(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.gain_mean = np.asarray(state["gain_mean"], dtype=np.float32).copy()
        self.gain_var = np.asarray(state["gain_var"], dtype=np.float32).copy()
        self.valid_count = np.asarray(state["valid_count"], dtype=np.int64).copy()
        self.temperature_slope = np.asarray(state["temperature_slope"], dtype=np.float32).copy()
        self.last_headroom = np.asarray(state["last_headroom"], dtype=np.float32).copy()

    def clone_state(self) -> Dict[str, Any]:
        return copy.deepcopy(self.state_dict())

    def reset(self) -> None:
        fresh = ThermalGainEstimator(self.cfg, n_tx=self.n_tx, thermal_safe=self.thermal_safe)
        self.load_state_dict(fresh.state_dict())

    def beta(self) -> np.ndarray:
        if not self.enabled:
            return np.zeros((self.n_tx,), dtype=np.float32)
        sample_uncertainty = 1.0 / np.sqrt(np.maximum(self.valid_count.astype(np.float32), 0.0) + 1.0)
        beta = self.uncertainty_beta_min + (self.uncertainty_beta_max - self.uncertainty_beta_min) * sample_uncertainty
        return np.clip(beta, self.uncertainty_beta_min, self.uncertainty_beta_max).astype(np.float32)

    def gain_std(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.gain_var, 0.0)).astype(np.float32)

    def gain_uncertainty(self) -> np.ndarray:
        prior = max(float(self.prior_std), 0.0) / np.sqrt(self.valid_count.astype(np.float32) + 1.0)
        return (self.gain_std() + prior).astype(np.float32)

    def effective_gain_safe(self) -> np.ndarray:
        if not self.enabled:
            return self.gain_mean.astype(np.float32).copy()
        safe = self.gain_mean + self.beta() * self.gain_uncertainty()
        return np.clip(safe, self.min_gain, self.max_gain).astype(np.float32)

    def safe_gain_scale(self) -> np.ndarray:
        # Backward-compatible alias.  The returned value is now an absolute
        # effective heating gain, not a multiplicative scale.
        return self.effective_gain_safe()

    def observe_temperature(self, temps: np.ndarray) -> None:
        temps_arr = np.asarray(temps, dtype=np.float32).reshape(-1)[: self.n_tx]
        if temps_arr.size == self.n_tx:
            self.last_headroom = (self.thermal_safe - temps_arr).astype(np.float32)

    def diagnostics(self) -> Dict[str, np.ndarray | bool]:
        return {
            "adaptive_thermal_enabled": bool(self.enabled),
            "thermal_gain_mean": self.gain_mean.astype(np.float32).copy(),
            "thermal_gain_std": self.gain_std(),
            "thermal_gain_uncertainty": self.gain_uncertainty(),
            "thermal_gain_safe_scale": self.effective_gain_safe(),
            "effective_gain_mean": self.gain_mean.astype(np.float32).copy(),
            "effective_gain_std": self.gain_std(),
            "effective_gain_uncertainty": self.gain_uncertainty(),
            "effective_gain_safe": self.effective_gain_safe(),
            "thermal_gain_beta": self.beta(),
            "thermal_gain_valid_count": self.valid_count.astype(np.float32).copy(),
            "temperature_slope": self.temperature_slope.astype(np.float32).copy(),
            "thermal_headroom": self.last_headroom.astype(np.float32).copy(),
        }

    def update(
        self,
        *,
        currents: np.ndarray,
        temps_before: np.ndarray,
        temps_after: np.ndarray,
        thermal_base: np.ndarray,
    ) -> Dict[str, np.ndarray | bool]:
        temps_before = np.asarray(temps_before, dtype=np.float32).reshape(-1)[: self.n_tx]
        temps_after = np.asarray(temps_after, dtype=np.float32).reshape(-1)[: self.n_tx]
        self.temperature_slope = (temps_after - temps_before).astype(np.float32)
        self.last_headroom = (self.thermal_safe - temps_after).astype(np.float32)
        if not self.enabled:
            return self.diagnostics()

        currents = np.asarray(currents, dtype=np.float32).reshape(-1)[: self.n_tx]
        thermal_base = np.asarray(thermal_base, dtype=np.float32).reshape(-1)[: self.n_tx]
        current_sq = currents * currents
        valid = current_sq >= self.excitation_threshold

        for idx in range(self.n_tx):
            if not bool(valid[idx]):
                continue
            instant_eff_gain = float((temps_after[idx] - thermal_base[idx]) / max(float(current_sq[idx]), 1.0e-8))
            if not np.isfinite(instant_eff_gain):
                continue
            instant_eff_gain = float(np.clip(instant_eff_gain, self.min_gain, self.max_gain))
            prev_mean = float(self.gain_mean[idx])
            prev_std = float(np.sqrt(max(float(self.gain_var[idx]), 0.0)))
            clip_radius = max(self.innovation_clip * max(prev_std, 0.05), 0.05)
            clipped = float(np.clip(instant_eff_gain, prev_mean - clip_radius, prev_mean + clip_radius))

            alpha = float(np.clip(self.ema_alpha, 0.0, 1.0))
            delta_mean = clipped - prev_mean
            new_mean = prev_mean + alpha * delta_mean
            new_var = (1.0 - alpha) * (float(self.gain_var[idx]) + alpha * delta_mean * delta_mean)
            self.gain_mean[idx] = float(np.clip(new_mean, self.min_gain, self.max_gain))
            self.gain_var[idx] = float(max(new_var, 0.0))
            self.valid_count[idx] += 1

        return self.diagnostics()
