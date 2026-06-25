from __future__ import annotations

from typing import Dict, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tchhmrl.envs.physics_v2 import (
    DEFAULT_EH_MODEL,
    DEFAULT_PHYSICS_VERSION,
    DEFAULT_SAFETY_PROJECTION_VERSION,
    DEFAULT_THERMAL_COUPLING_MATRIX,
    DEFAULT_THERMAL_MODEL,
    coupling_matrix_hash,
    eh_calibration_hash,
    logistic_eh_metric,
    normalize_safety_projection_version,
    thermal_coupling_term_np,
    validate_coupling_matrix,
)
from tchhmrl.envs.task_contract import TaskSpec


class MultiTxUwSliptEnv(gym.Env):
    """Multi-transmitter underwater SLIPT simulation with fixed heterogeneous devices.

    Default mapping follows the paper narrative:
    - Tx0 (Anchor / wide beam): LED
    - Tx1, Tx2 (Boost / narrow beams): LD
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: Dict, overrides: Optional[Dict[str, float]] = None):
        super().__init__()
        env_cfg = dict(cfg["env"])
        if overrides:
            env_cfg.update(overrides)
        alignment_cfg = dict(cfg.get("alignment", {}))
        self.alignment_version = str(alignment_cfg.get("alignment_version", "system_model_v1"))
        self.task_summary_version = str(alignment_cfg.get("task_summary_version", "site_v2"))
        self.pre_alignment = bool(alignment_cfg.get("pre_alignment", False))
        self.default_alignment_version = str(self.alignment_version)
        self.default_task_summary_version = str(self.task_summary_version)
        self.n_tx = int(env_cfg["n_tx"])

        physics_cfg = dict(cfg.get("physics", {}))
        self.physics_version = str(physics_cfg.get("physics_version", DEFAULT_PHYSICS_VERSION))
        self.eh_model = str(physics_cfg.get("eh_model", DEFAULT_EH_MODEL)).lower()
        if self.eh_model not in {"linear", "logistic"}:
            raise ValueError(f"unsupported physics.eh_model={self.eh_model}")
        self.eh_nl_M = float(physics_cfg.get("eh_nl_M", 0.0))
        self.eh_nl_a = float(physics_cfg.get("eh_nl_a", 1.0))
        self.eh_nl_b = float(physics_cfg.get("eh_nl_b", 0.0))
        self.eh_calibration_hash = str(
            physics_cfg.get(
                "eh_calibration_hash",
                eh_calibration_hash(
                    {
                        "eh_model": self.eh_model,
                        "eh_nl_M": self.eh_nl_M,
                        "eh_nl_a": self.eh_nl_a,
                        "eh_nl_b": self.eh_nl_b,
                    }
                ),
            )
        )
        self.thermal_model = str(physics_cfg.get("thermal_model", DEFAULT_THERMAL_MODEL)).lower()
        if self.thermal_model not in {"independent", "coupled"}:
            raise ValueError(f"unsupported physics.thermal_model={self.thermal_model}")
        self.safety_projection_version = normalize_safety_projection_version(
            self.thermal_model,
            physics_cfg.get("safety_projection_version", DEFAULT_SAFETY_PROJECTION_VERSION),
        )
        coupling_matrix = physics_cfg.get("thermal_coupling_matrix", DEFAULT_THERMAL_COUPLING_MATRIX.tolist())
        self.thermal_coupling_matrix = validate_coupling_matrix(coupling_matrix, n_tx=3)
        if self.n_tx != 3:
            raise ValueError("physics_v2 currently requires n_tx=3")
        self.thermal_coupling_matrix_hash = coupling_matrix_hash(self.thermal_coupling_matrix)

        self.episode_len = int(env_cfg["episode_len"])
        self.obs_noise_std = float(env_cfg["obs_noise_std"])
        hybrid_cfg = env_cfg.get("hybrid", {})

        self.attenuation_c = float(env_cfg["attenuation_c"])
        self.d0 = float(env_cfg["d0"])
        self.distances = np.asarray(env_cfg["distances"], dtype=np.float32)
        self.default_distances = self.distances.copy()
        self.misalign_std = float(env_cfg["misalign_std"])
        self.temporal_misalign_rho = float(env_cfg.get("temporal_misalign_rho", 0.0))
        self.attenuation_drift_rho = float(env_cfg.get("attenuation_drift_rho", 0.0))
        self.attenuation_drift_std = float(env_cfg.get("attenuation_drift_std", 0.0))
        self.burst_prob = float(env_cfg.get("burst_prob", 0.0))
        self.burst_decay = float(env_cfg.get("burst_decay", 0.9))
        burst_strength = env_cfg.get("burst_strength_range", [0.0, 0.0])
        self.burst_strength_range = np.asarray(burst_strength, dtype=np.float32)
        if self.burst_strength_range.shape != (2,):
            raise ValueError("burst_strength_range must be [min, max]")
        self.burst_led_factor = float(hybrid_cfg.get("burst_led_factor", env_cfg.get("burst_led_factor", 1.0)))
        self.burst_ld_factor = float(hybrid_cfg.get("burst_ld_factor", env_cfg.get("burst_ld_factor", 1.0)))
        self.obs_bias_rho = float(env_cfg.get("obs_bias_rho", 0.0))
        self.obs_bias_step_std = float(env_cfg.get("obs_bias_step_std", 0.0))

        self.thermal_safe = float(env_cfg["thermal_safe"])
        self.thermal_cutoff = float(env_cfg["thermal_cutoff"])
        self.temp_init = float(env_cfg["temp_init"])
        self.amb_temp = float(env_cfg["amb_temp"])
        self.gamma = float(env_cfg["gamma"])
        self.delta = float(env_cfg["delta"])

        self.se_weight = float(env_cfg["se_weight"])
        self.eh_weight = float(env_cfg["eh_weight"])
        self.cost_weight = float(env_cfg["cost_weight"])
        self.power_weight = float(env_cfg["power_weight"])
        self.qos_min_rate = float(env_cfg.get("qos_min_rate", 0.0))
        self.action_smooth_weight = float(env_cfg.get("action_smooth_weight", 0.0))
        self.mode_switch_penalty = float(env_cfg.get("mode_switch_penalty", 0.0))
        self.boost_switch_penalty = float(env_cfg.get("boost_switch_penalty", 0.0))
        self.thermal_margin_weight = float(env_cfg.get("thermal_margin_weight", 0.0))
        self.mode_se_gain = np.asarray(env_cfg.get("mode_se_gain", [1.0, 1.2, 0.85]), dtype=np.float32)
        self.mode_eh_gain = np.asarray(env_cfg.get("mode_eh_gain", [1.0, 0.85, 1.2]), dtype=np.float32)
        # Optional supplementary EH override. If absent, the formal physics.* EH
        # model remains active.
        env_eh_override = env_cfg.get("eh_model", None)
        if env_eh_override not in (None, ""):
            env_eh_override = str(env_eh_override).strip().lower()
            if env_eh_override not in {"linear", "nonlinear"}:
                raise ValueError(
                    f"env.eh_model override must be 'linear' or 'nonlinear', got {env_eh_override!r}"
                )
            self.eh_model = env_eh_override
        self.eh_nonlinear_cfg = dict(env_cfg.get("eh_nonlinear", {}) or {})
        self.eh_robust_type = str(self.eh_nonlinear_cfg.get("type", "logistic_normalized")).strip().lower()
        if self.eh_robust_type != "logistic_normalized":
            raise ValueError(f"Only logistic_normalized nonlinear EH is supported, got {self.eh_robust_type!r}")
        self.eh_robust_e_max = float(self.eh_nonlinear_cfg.get("e_max", 1.0))
        self.eh_robust_a = float(self.eh_nonlinear_cfg.get("a", 12.0))
        self.eh_robust_b = float(self.eh_nonlinear_cfg.get("b", 0.10))
        self.eh_robust_scale_match = bool(self.eh_nonlinear_cfg.get("scale_match", True))
        scale_cfg = self.eh_nonlinear_cfg.get("scale", None)
        self.eh_robust_scale = 1.0 if scale_cfg in (None, "") else float(scale_cfg)

        tx_device = hybrid_cfg.get("tx_device", ["LED", "LD", "LD"])
        if len(tx_device) != self.n_tx:
            raise ValueError(f"len(tx_device)={len(tx_device)} must equal n_tx={self.n_tx}")
        tx_enabled = hybrid_cfg.get("tx_enabled", [1.0] * self.n_tx)
        if len(tx_enabled) != self.n_tx:
            raise ValueError(f"len(tx_enabled)={len(tx_enabled)} must equal n_tx={self.n_tx}")
        self.tx_enabled = np.asarray(tx_enabled, dtype=np.float32)
        self.tx_enabled = np.where(self.tx_enabled > 0.5, 1.0, 0.0).astype(np.float32)
        self.tx_device = np.asarray([str(x).upper() for x in tx_device])
        valid = {"LED", "LD"}
        if not set(self.tx_device).issubset(valid):
            raise ValueError(f"tx_device only supports {valid}, got {set(self.tx_device)}")

        self.tx_is_led = (self.tx_device == "LED").astype(np.float32)
        self.tx_is_ld = 1.0 - self.tx_is_led
        active_sum = float(np.sum(self.tx_enabled))
        if active_sum > 0.0:
            self.led_tx_fraction = float(np.sum(self.tx_is_led * self.tx_enabled) / active_sum)
        else:
            self.led_tx_fraction = 0.0

        self.eta_led = float(hybrid_cfg.get("eta_led", 0.90))
        self.eta_ld = float(hybrid_cfg.get("eta_ld", 1.35))
        self.attenuation_led_factor = float(hybrid_cfg.get("attenuation_led_factor", 1.15))
        self.attenuation_ld_factor = float(hybrid_cfg.get("attenuation_ld_factor", 0.85))
        self.misalign_led_scale = float(hybrid_cfg.get("misalign_led_scale", 1.30))
        self.misalign_ld_scale = float(hybrid_cfg.get("misalign_ld_scale", 0.75))
        self.noise_floor = float(hybrid_cfg.get("noise_floor", 0.02))
        self.noise_led_coeff = float(hybrid_cfg.get("noise_led_coeff", 0.015))
        self.noise_ld_coeff = float(hybrid_cfg.get("noise_ld_coeff", 0.025))
        self.se_led_weight = float(hybrid_cfg.get("se_led_weight", 0.85))
        self.se_ld_weight = float(hybrid_cfg.get("se_ld_weight", 1.20))
        self.eh_led_weight = float(hybrid_cfg.get("eh_led_weight", 1.20))
        self.eh_ld_weight = float(hybrid_cfg.get("eh_ld_weight", 0.80))
        self.thermal_led_coeff = float(hybrid_cfg.get("thermal_led_coeff", 1.00))
        self.thermal_ld_coeff = float(hybrid_cfg.get("thermal_ld_coeff", 1.25))

        current_max = np.asarray(cfg["safety"]["current_max"], dtype=np.float32)
        self.current_max = current_max
        self.bus_current_max = float(cfg["safety"]["bus_current_max"])

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4 * self.n_tx + 8,),
            dtype=np.float32,
        )

        self.action_space = spaces.Dict(
            {
                "upper_idx": spaces.Discrete(12),
                "upper_idx_exec": spaces.Discrete(12),
                "boost_combo_exec": spaces.Discrete(4),
                "mode_exec": spaces.Discrete(3),
                "currents_exec": spaces.Box(
                    low=np.zeros(self.n_tx, dtype=np.float32),
                    high=self.current_max,
                    dtype=np.float32,
                ),
                "rho_exec": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
                "tau_exec": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            }
        )

        self.rng = np.random.default_rng(0)
        self.site_id = int(env_cfg.get("site_id", -1))
        self.default_site_id = int(self.site_id)
        self.task_source = str(env_cfg.get("task_source", "global_fallback"))
        self.default_task_source = str(self.task_source)
        self.t = 0
        self.temps = np.full(self.n_tx, self.temp_init, dtype=np.float32)
        self.misalign = np.zeros(self.n_tx, dtype=np.float32)
        self.channel = np.zeros(self.n_tx, dtype=np.float32)
        self.channel_led = np.zeros(self.n_tx, dtype=np.float32)
        self.channel_ld = np.zeros(self.n_tx, dtype=np.float32)
        self.attenuation_eff = float(self.attenuation_c)
        self.burst_state = np.zeros(self.n_tx, dtype=np.float32)
        self.prev_currents = np.zeros(self.n_tx, dtype=np.float32)
        self.prev_rho = 0.5
        self.prev_tau = 0.5
        self.prev_mode = 0
        self.prev_boost = 0
        self.boost_hold_steps = 0
        self.prev_qos_rate = 0.0
        self.prev_eh_metric = 0.0
        self.obs_bias = np.zeros(self.observation_space.shape[0], dtype=np.float32)

    @staticmethod
    def _boost_mask(combo: int) -> np.ndarray:
        table = np.asarray(
            [
                [1.0, 0.0, 0.0],  # Anchor only
                [1.0, 1.0, 0.0],  # Anchor + Boost1
                [1.0, 0.0, 1.0],  # Anchor + Boost2
                [1.0, 1.0, 1.0],  # Anchor + Boost1 + Boost2
            ],
            dtype=np.float32,
        )
        combo = int(np.clip(combo, 0, len(table) - 1))
        return table[combo]

    @staticmethod
    def _mode_gain(mode: int, gain_table: np.ndarray) -> float:
        mode = int(np.clip(mode, 0, len(gain_table) - 1))
        return float(gain_table[mode])

    def _tx_vector(self, led_value: float, ld_value: float) -> np.ndarray:
        return self.tx_is_led * led_value + self.tx_is_ld * ld_value

    def _compute_tx_signal(self, currents: np.ndarray) -> np.ndarray:
        eta_tx = self._tx_vector(self.eta_led, self.eta_ld)
        return currents * eta_tx * self.channel

    @staticmethod
    def _stable_sigmoid(x: float) -> float:
        x = float(np.clip(x, -60.0, 60.0))
        return float(1.0 / (1.0 + np.exp(-x)))

    def _compute_nonlinear_eh_raw(self, eh_input_eff: float) -> float:
        """Normalized logistic EH response before the global robustness-check scale.

        This is only used when env.eh_model="nonlinear"; the main-paper default remains
        the original linear EH mapping.
        """
        x = float(max(eh_input_eff, 0.0))
        e_max = float(max(self.eh_robust_e_max, 0.0))
        if e_max <= 0.0:
            return 0.0
        omega = self._stable_sigmoid(-self.eh_robust_a * self.eh_robust_b)
        y = self._stable_sigmoid(self.eh_robust_a * (x - self.eh_robust_b))
        denom = max(1.0 - omega, 1.0e-12)
        raw = e_max * (y - omega) / denom
        return float(np.clip(raw, 0.0, e_max))

    def _compute_eh_metric(self, eh_input_eff: float) -> Dict[str, float | str]:
        linear_proxy = float(eh_input_eff)
        raw_nonlinear = self._compute_nonlinear_eh_raw(linear_proxy)
        scaled_nonlinear = float(raw_nonlinear * self.eh_robust_scale)
        if self.eh_model == "linear":
            eh_metric = linear_proxy
        elif self.eh_model == "logistic":
            eh_metric = float(
                logistic_eh_metric(
                    linear_proxy,
                    M=self.eh_nl_M,
                    a=self.eh_nl_a,
                    b=self.eh_nl_b,
                )
            )
        else:
            eh_metric = scaled_nonlinear
        sat_threshold = 0.99 * max(self.eh_robust_e_max, 1.0e-12)
        near_zero_threshold = max(1.0e-12, 1.0e-4 * max(self.eh_robust_e_max, 1.0e-12))
        return {
            "eh_model": self.eh_model,
            "eh_input_eff": float(linear_proxy),
            "eh_metric_linear_proxy": float(linear_proxy),
            "eh_metric_raw_nonlinear": float(raw_nonlinear),
            "eh_metric": float(eh_metric),
            "eh_saturation_fraction": float(raw_nonlinear >= sat_threshold),
            "eh_near_zero_fraction": float(raw_nonlinear <= near_zero_threshold),
            "eh_scale": float(self.eh_robust_scale),
        }

    def _update_channel(self) -> None:
        mis_rho = float(np.clip(self.temporal_misalign_rho, 0.0, 0.995))
        if mis_rho <= 1e-6:
            self.misalign = self.rng.normal(0.0, self.misalign_std, size=self.n_tx).astype(np.float32)
        else:
            eps_std = self.misalign_std * np.sqrt(max(1.0 - mis_rho * mis_rho, 1e-4))
            eps = self.rng.normal(0.0, eps_std, size=self.n_tx).astype(np.float32)
            self.misalign = (mis_rho * self.misalign + eps).astype(np.float32)

        att_rho = float(np.clip(self.attenuation_drift_rho, 0.0, 0.995))
        att_noise = float(self.rng.normal(0.0, self.attenuation_drift_std))
        self.attenuation_eff = (
            att_rho * float(self.attenuation_eff) + (1.0 - att_rho) * float(self.attenuation_c) + att_noise
        )
        self.attenuation_eff = float(np.clip(self.attenuation_eff, 0.02, 1.2))

        self.burst_state = (self.burst_decay * self.burst_state).astype(np.float32)
        if self.burst_prob > 0.0 and self.burst_strength_range[1] > 0.0:
            lo = float(min(self.burst_strength_range[0], self.burst_strength_range[1]))
            hi = float(max(self.burst_strength_range[0], self.burst_strength_range[1]))
            trigger = (self.rng.random(self.n_tx) < self.burst_prob).astype(np.float32)
            shock = self.rng.uniform(lo, hi, size=self.n_tx).astype(np.float32) * trigger
            self.burst_state = (self.burst_state + shock).astype(np.float32)

        geom = (self.d0 / (self.distances + 1e-6)) ** 2

        att_factor = self.tx_is_led * self.attenuation_led_factor + self.tx_is_ld * self.attenuation_ld_factor
        sigma_scale = self.tx_is_led * self.misalign_led_scale + self.tx_is_ld * self.misalign_ld_scale
        burst_factor = self.tx_is_led * self.burst_led_factor + self.tx_is_ld * self.burst_ld_factor

        water = np.exp(-(self.attenuation_eff * att_factor) * self.distances)
        water *= np.exp(-(self.burst_state * burst_factor))
        sigma = self.misalign_std * sigma_scale + 1e-6
        mis = np.exp(-0.5 * (self.misalign / sigma) ** 2)

        self.channel = (geom * water * mis * self.tx_enabled).astype(np.float32)
        self.channel_led = (self.channel * self.tx_is_led).astype(np.float32)
        self.channel_ld = (self.channel * self.tx_is_ld).astype(np.float32)

    def _obs(self) -> np.ndarray:
        # Normalize mixed-scale features for stabler value/policy learning.
        channel_norm = np.log1p(np.clip(self.channel, 0.0, None) * 500.0) / np.log1p(500.0)
        misalign_scale = max(self.misalign_std * 3.0, 1.0e-3)
        misalign_norm = np.clip(self.misalign / misalign_scale, -2.0, 2.0)
        temp_norm = np.clip((self.temps - self.amb_temp) / 20.0, -2.0, 2.0)
        amb_norm = float(np.clip((self.amb_temp - 25.0) / 15.0, -2.0, 2.0))
        prev_curr_norm = self.prev_currents / np.maximum(self.current_max, 1e-6)
        mode_norm = float(self.prev_mode) / 2.0
        boost_norm = float(self.prev_boost) / 3.0
        hold_norm = float(min(self.boost_hold_steps, self.episode_len)) / float(max(1, self.episode_len))
        se_norm = float(np.tanh(self.prev_qos_rate / 4.0))
        eh_norm = float(np.tanh(self.prev_eh_metric / 4.0))

        obs = np.concatenate(
            [
                channel_norm,
                misalign_norm,
                temp_norm,
                [amb_norm],
                prev_curr_norm,
                [self.prev_rho, self.prev_tau, mode_norm, boost_norm, hold_norm, se_norm, eh_norm],
            ]
        ).astype(np.float32)
        if self.obs_bias_step_std > 0.0:
            rho = float(np.clip(self.obs_bias_rho, 0.0, 0.999))
            self.obs_bias = (
                rho * self.obs_bias
                + self.rng.normal(0.0, self.obs_bias_step_std, size=self.obs_bias.shape).astype(np.float32)
            )
            obs += self.obs_bias
        obs += self.rng.normal(0.0, self.obs_noise_std, size=obs.shape).astype(np.float32)
        return obs

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if options:
            for key in ["attenuation_c", "misalign_std", "amb_temp", "gamma", "delta", "qos_min_rate"]:
                if key in options:
                    setattr(self, key, float(options[key]))
            if "distances" in options:
                distances = np.asarray(options["distances"], dtype=np.float32)
                if distances.shape != (self.n_tx,):
                    raise ValueError(f"distances must have shape ({self.n_tx},), got {distances.shape}")
                self.distances = distances
            else:
                self.distances = self.default_distances.copy()
            if "site_id" in options:
                self.site_id = int(options["site_id"])
            else:
                self.site_id = -1
            self.task_source = str(options.get("task_source", "global_fallback"))
            if "alignment_version" in options:
                self.alignment_version = str(options["alignment_version"])
            if "task_summary_version" in options:
                self.task_summary_version = str(options["task_summary_version"])
        else:
            self.distances = self.default_distances.copy()
            self.site_id = int(getattr(self, "default_site_id", -1))
            self.task_source = str(getattr(self, "default_task_source", "global_fallback"))
            self.alignment_version = str(getattr(self, "default_alignment_version", self.alignment_version))
            self.task_summary_version = str(
                getattr(self, "default_task_summary_version", self.task_summary_version)
            )

        self.t = 0
        self.temps = np.full(self.n_tx, self.temp_init, dtype=np.float32)
        self.attenuation_eff = float(self.attenuation_c)
        self.burst_state = np.zeros(self.n_tx, dtype=np.float32)
        self.prev_currents = np.zeros(self.n_tx, dtype=np.float32)
        self.prev_rho = 0.5
        self.prev_tau = 0.5
        self.prev_mode = 0
        self.prev_boost = 0
        self.boost_hold_steps = 0
        self.prev_qos_rate = 0.0
        self.prev_eh_metric = 0.0
        self.obs_bias = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        self._update_channel()
        return self._obs(), {"task": self._task_dict()}

    def _task_dict(self) -> Dict[str, float]:
        spec = TaskSpec(
            site_id=int(getattr(self, "site_id", -1)),
            task_source=str(getattr(self, "task_source", "global_fallback")),
            distances=tuple(float(x) for x in self.distances[:3]),
            attenuation_c=float(self.attenuation_c),
            misalign_std=float(self.misalign_std),
            amb_temp=float(self.amb_temp),
            gamma=float(self.gamma),
            delta=float(self.delta),
            qos_min_rate=float(self.qos_min_rate),
            alignment_version=self.alignment_version,
            task_summary_version=self.task_summary_version,
        )
        return spec.to_task_dict(pre_alignment=self.pre_alignment)

    def debug_hy_snapshot(
        self,
        *,
        currents_exec: np.ndarray,
        rho_exec: float,
        tau_exec: float,
        mode_exec: int = 2,
    ) -> Dict[str, float]:
        currents = np.asarray(currents_exec, dtype=np.float32).copy()
        rho = float(rho_exec)
        tau = float(tau_exec)
        tx_signal = self._compute_tx_signal(currents)
        signal_led = float(np.sum(tx_signal * self.tx_is_led))
        signal_ld = float(np.sum(tx_signal * self.tx_is_ld))
        se_tx_weight = self._tx_vector(self.se_led_weight, self.se_ld_weight)
        eh_tx_weight = self._tx_vector(self.eh_led_weight, self.eh_ld_weight)
        info_signal = float(np.sum(tx_signal * se_tx_weight))
        eh_input = float(np.sum(tx_signal * eh_tx_weight))
        noise_power = (
            self.noise_floor
            + self.noise_led_coeff * abs(signal_led)
            + self.noise_ld_coeff * abs(signal_ld)
        )
        info_share = tau * (1.0 - rho)
        eh_share = 1.0 - info_share
        snr = max(info_signal / max(noise_power, 1.0e-6), 1.0e-6)
        qos_rate = float(self._mode_gain(mode_exec, self.mode_se_gain) * info_share * np.log2(1.0 + snr))
        eh_input_eff = float(self._mode_gain(mode_exec, self.mode_eh_gain) * eh_share * eh_input)
        eh_diag = self._compute_eh_metric(eh_input_eff)
        eh_metric = float(eh_diag["eh_metric"])
        return {
            "info_share": float(info_share),
            "eh_share": float(eh_share),
            "qos_rate": float(qos_rate),
            "eh_metric": float(eh_metric),
            "eh_model": str(eh_diag["eh_model"]),
            "eh_input_eff": float(eh_diag["eh_input_eff"]),
            "eh_metric_linear_proxy": float(eh_diag["eh_metric_linear_proxy"]),
            "eh_metric_raw_nonlinear": float(eh_diag["eh_metric_raw_nonlinear"]),
            "eh_saturation_fraction": float(eh_diag["eh_saturation_fraction"]),
            "eh_near_zero_fraction": float(eh_diag["eh_near_zero_fraction"]),
            "eh_scale": float(eh_diag["eh_scale"]),
            "reward_id_term": float(self.se_weight * qos_rate),
            "reward_eh_term": float(self.eh_weight * eh_metric),
            "total_reward_no_penalty": float(self.se_weight * qos_rate + self.eh_weight * eh_metric),
        }

    def _thermal_coupling_term(self, temps: np.ndarray) -> np.ndarray:
        return thermal_coupling_term_np(
            np.asarray(temps, dtype=np.float32),
            self.thermal_coupling_matrix,
            thermal_model=self.thermal_model,
        )

    def step(self, action: Dict):
        self.t += 1
        boost_combo = int(action.get("boost_combo_exec", 0))
        mode = int(action.get("mode_exec", 0))

        currents = np.asarray(action.get("currents_exec", np.zeros(self.n_tx)), dtype=np.float32)
        currents = np.clip(currents, 0.0, self.current_max)
        currents *= self.tx_enabled
        # `currents_exec` is expected to be already projected by the safety layer.
        # Keep env-side clipping/total-current check as a defensive fallback only.

        total = float(np.sum(currents))
        if total > self.bus_current_max:
            currents *= self.bus_current_max / (total + 1e-6)
            total = float(np.sum(currents))

        rho = float(np.clip(np.asarray(action.get("rho_exec", [0.5])).item(), 0.0, 1.0))
        tau = float(np.clip(np.asarray(action.get("tau_exec", [0.5])).item(), 0.0, 1.0))

        tx_signal = self._compute_tx_signal(currents)

        signal = float(np.sum(tx_signal))
        signal_led = float(np.sum(tx_signal * self.tx_is_led))
        signal_ld = float(np.sum(tx_signal * self.tx_is_ld))

        noise_power = (
            self.noise_floor
            + self.noise_led_coeff * abs(signal_led)
            + self.noise_ld_coeff * abs(signal_ld)
            + abs(float(self.rng.normal(0.0, 0.005)))
        )

        se_tx_weight = self._tx_vector(self.se_led_weight, self.se_ld_weight)
        eh_tx_weight = self._tx_vector(self.eh_led_weight, self.eh_ld_weight)

        info_signal = float(np.sum(tx_signal * se_tx_weight))
        eh_input = float(np.sum(tx_signal * eh_tx_weight))

        snr = max(info_signal / max(noise_power, 1e-6), 1e-6)
        mode_se = self._mode_gain(mode, self.mode_se_gain)
        mode_eh = self._mode_gain(mode, self.mode_eh_gain)

        # Mode semantics:
        # mode=0: PS  -> info=(1-rho), EH=rho
        # mode=1: TS  -> info=tau, EH=(1-tau)
        # mode=2: HY  -> info=tau*(1-rho), EH=1-tau*(1-rho)
        if mode == 0:
            info_share = 1.0 - rho
            eh_share = rho
        elif mode == 1:
            info_share = tau
            eh_share = 1.0 - tau
        else:
            info_share = tau * (1.0 - rho)
            eh_share = 1.0 - info_share

        info_share = float(np.clip(info_share, 0.0, 1.0))
        eh_share = float(np.clip(eh_share, 0.0, 1.0))
        if mode == 2:
            assert abs((info_share + eh_share) - 1.0) < 1.0e-6

        qos_rate = float(mode_se * info_share * np.log2(1.0 + snr))
        eh_input_eff = float(mode_eh * eh_share * eh_input)
        eh_diag = self._compute_eh_metric(eh_input_eff)
        eh_metric = float(eh_diag["eh_metric"])
        eh_metric_linear_proxy = float(eh_diag["eh_metric_linear_proxy"])
        id_term = self.se_weight * qos_rate
        eh_term = self.eh_weight * eh_metric

        thermal_coeff = self._tx_vector(self.thermal_led_coeff, self.thermal_ld_coeff)
        temps_before_update = self.temps.copy()
        thermal_source_term = self._thermal_coupling_term(temps_before_update)
        thermal_base = (1.0 - self.gamma) * temps_before_update + self.gamma * self.amb_temp + thermal_source_term
        thermal_drive = self.delta * thermal_coeff * (currents**2)
        self.temps = (thermal_base + thermal_drive).astype(np.float32)

        thermal_violation_vec = np.maximum(self.temps - self.thermal_safe, 0.0).astype(np.float32)
        thermal_violation = float(thermal_violation_vec.sum())
        qos_violation = float(max(self.qos_min_rate - qos_rate, 0.0))
        cost_vec = np.concatenate([np.asarray([qos_violation], dtype=np.float32), thermal_violation_vec], axis=0)
        power_penalty = float(np.sum(currents**2))
        delta_curr_norm = (currents - self.prev_currents) / np.maximum(self.current_max, 1.0e-6)
        smooth_raw = float(np.mean(delta_curr_norm**2) + 0.5 * ((rho - self.prev_rho) ** 2 + (tau - self.prev_tau) ** 2))
        smooth_penalty = self.action_smooth_weight * smooth_raw
        mode_switch = float(mode != self.prev_mode)
        boost_switch = float(boost_combo != self.prev_boost)
        switch_penalty = self.mode_switch_penalty * mode_switch + self.boost_switch_penalty * boost_switch
        temp_peak = float(np.max(self.temps))
        margin_norm = float(np.clip((self.thermal_safe - temp_peak) / max(self.thermal_safe, 1e-6), 0.0, 1.0))
        margin_reward = self.thermal_margin_weight * margin_norm
        cost = float(np.sum(cost_vec))
        cost_penalty = self.cost_weight * cost
        power_penalty_term = self.power_weight * power_penalty
        reward_task = float(id_term + eh_term + margin_reward - power_penalty_term - smooth_penalty - switch_penalty)
        reward = float(reward_task - cost_penalty)

        if boost_combo == self.prev_boost:
            self.boost_hold_steps += 1
        else:
            self.boost_hold_steps = 1
        self.prev_currents = currents.copy()
        self.prev_rho = rho
        self.prev_tau = tau
        self.prev_mode = mode
        self.prev_boost = boost_combo
        self.prev_qos_rate = float(qos_rate)
        self.prev_eh_metric = float(eh_metric)

        terminated = bool(np.any(self.temps >= self.thermal_cutoff))
        truncated = bool(self.t >= self.episode_len)

        self._update_channel()
        obs = self._obs()

        info = {
            "se": float(id_term),
            "eh": float(eh_term),
            "qos_rate": float(qos_rate),
            "eh_input_eff": float(eh_input_eff),
            "eh_metric_linear_proxy": float(eh_metric_linear_proxy),
            "eh_metric": float(eh_metric),
            "eh_model": self.eh_model,
            "eh_nl_M": float(self.eh_nl_M),
            "eh_nl_a": float(self.eh_nl_a),
            "eh_nl_b": float(self.eh_nl_b),
            "eh_calibration_hash": str(self.eh_calibration_hash),
            "eh_metric_raw_nonlinear": float(eh_diag["eh_metric_raw_nonlinear"]),
            "eh_saturation_fraction": float(eh_diag["eh_saturation_fraction"]),
            "eh_near_zero_fraction": float(eh_diag["eh_near_zero_fraction"]),
            "eh_scale": float(eh_diag["eh_scale"]),
            "reward_se_term": float(id_term),
            "reward_eh_term": float(eh_term),
            "reward_id_term": float(id_term),
            "reward_cost_penalty": float(cost_penalty),
            "reward_power_penalty": float(power_penalty_term),
            "reward_task": float(reward_task),
            "reward_benchmark": float(reward),
            "reward_fixed_constraint_penalty": float(cost_penalty),
            "penalty_cost_term": float(cost_penalty),
            "penalty_power_term": float(power_penalty_term),
            "penalty_smooth_term": float(smooth_penalty),
            "penalty_switch_term": float(switch_penalty),
            "reward_margin_term": float(margin_reward),
            "snr": float(snr),
            "info_share": float(info_share),
            "eh_share": float(eh_share),
            "signal": float(signal),
            "signal_led": float(signal_led),
            "signal_ld": float(signal_ld),
            "signal_ld_share": float(signal_ld / (abs(signal) + 1e-6)),
            "tx_signal0": float(tx_signal[0]) if self.n_tx > 0 else 0.0,
            "tx_signal1": float(tx_signal[1]) if self.n_tx > 1 else 0.0,
            "tx_signal2": float(tx_signal[2]) if self.n_tx > 2 else 0.0,
            "led_tx_fraction": self.led_tx_fraction,
            "tx_enabled_fraction": float(np.mean(self.tx_enabled)),
            "boost_combo_exec": int(boost_combo),
            "mode_exec": int(mode),
            "upper_idx_exec": int(boost_combo * 3 + mode),
            "mode_switch": float(mode_switch),
            "boost_switch": float(boost_switch),
            "cost": float(cost),
            "cost_vec": cost_vec.astype(np.float32),
            "cost_qos": float(qos_violation),
            "cost_temp_anchor": float(thermal_violation_vec[0]) if self.n_tx > 0 else 0.0,
            "cost_temp_boost1": float(thermal_violation_vec[1]) if self.n_tx > 1 else 0.0,
            "cost_temp_boost2": float(thermal_violation_vec[2]) if self.n_tx > 2 else 0.0,
            "thermal_violation": thermal_violation,
            "thermal_violation_vec": thermal_violation_vec.astype(np.float32),
            "power_penalty": power_penalty,
            "currents_exec": currents.copy(),
            "current_total": total,
            "bus_current_max": float(self.bus_current_max),
            "bus_utilization": float(total / max(self.bus_current_max, 1.0e-6)),
            "temps": self.temps.copy(),
            "thermal_model": self.thermal_model,
            "thermal_coupling_matrix_hash": self.thermal_coupling_matrix_hash,
            "thermal_source_model": self.thermal_model,
            "thermal_source_term_tx0": float(thermal_source_term[0]) if self.n_tx > 0 else 0.0,
            "thermal_source_term_tx1": float(thermal_source_term[1]) if self.n_tx > 1 else 0.0,
            "thermal_source_term_tx2": float(thermal_source_term[2]) if self.n_tx > 2 else 0.0,
            "thermal_base_tx0": float(thermal_base[0]) if self.n_tx > 0 else 0.0,
            "thermal_base_tx1": float(thermal_base[1]) if self.n_tx > 1 else 0.0,
            "thermal_base_tx2": float(thermal_base[2]) if self.n_tx > 2 else 0.0,
            "gamma": float(self.gamma),
            "delta": float(self.delta),
            "attenuation_eff": float(self.attenuation_eff),
            "burst_mean": float(np.mean(self.burst_state)),
            "amb_temp": float(self.amb_temp),
            "amb_temp_env": float(self.amb_temp),
            "site_id": int(getattr(self, "site_id", -1)),
            "task_source": str(getattr(self, "task_source", "global_fallback")),
            "distance_tx0": float(self.distances[0]) if self.n_tx > 0 else 0.0,
            "distance_tx1": float(self.distances[1]) if self.n_tx > 1 else 0.0,
            "distance_tx2": float(self.distances[2]) if self.n_tx > 2 else 0.0,
            "alignment_version": self.alignment_version,
            "task_summary_version": self.task_summary_version,
            "pre_alignment": bool(self.pre_alignment),
            "physics_version": self.physics_version,
            "safety_projection_version": self.safety_projection_version,
        }
        if self.thermal_model == "coupled":
            info.update(
                {
                    "thermal_coupling_term_tx0": float(thermal_source_term[0]) if self.n_tx > 0 else 0.0,
                    "thermal_coupling_term_tx1": float(thermal_source_term[1]) if self.n_tx > 1 else 0.0,
                    "thermal_coupling_term_tx2": float(thermal_source_term[2]) if self.n_tx > 2 else 0.0,
                    "thermal_base_coupled_tx0": float(thermal_base[0]) if self.n_tx > 0 else 0.0,
                    "thermal_base_coupled_tx1": float(thermal_base[1]) if self.n_tx > 1 else 0.0,
                    "thermal_base_coupled_tx2": float(thermal_base[2]) if self.n_tx > 2 else 0.0,
                }
            )
        return obs, reward, terminated, truncated, info
