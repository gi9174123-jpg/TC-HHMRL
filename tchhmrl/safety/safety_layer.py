from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from tchhmrl.envs.physics_v2 import (
    DEFAULT_SAFETY_PROJECTION_VERSION,
    DEFAULT_THERMAL_COUPLING_MATRIX,
    DEFAULT_THERMAL_MODEL,
    coupling_matrix_hash,
    normalize_safety_projection_version,
    validate_coupling_matrix,
)


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    pos = x >= 0
    neg = ~pos
    out = np.empty_like(x, dtype=np.float32)
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out


def raw_from_frac01(x: np.ndarray | float, action_decode_mode: str = "tanh_affine") -> np.ndarray:
    """Encode physical [0, 1] fractions back to lower-policy raw action space."""
    frac = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    mode = str(action_decode_mode).lower()
    if mode == "sigmoid_logit":
        clipped = np.clip(frac, 1.0e-4, 1.0 - 1.0e-4)
        return np.log(clipped / (1.0 - clipped)).astype(np.float32)
    if mode == "tanh_affine":
        return (2.0 * frac - 1.0).astype(np.float32)
    raise ValueError(f"unsupported safety.action_decode_mode={mode}")


def _softplus_np(x: np.ndarray, beta: float) -> np.ndarray:
    return np.logaddexp(0.0, beta * x) / beta


class SafetyLayer:
    """Safety projection for hierarchical actions."""

    def __init__(self, cfg: Dict):
        safety_cfg = cfg["safety"]
        env_cfg = cfg.get("env", {})
        hybrid_cfg = env_cfg.get("hybrid", {})
        physics_cfg = cfg.get("physics", {})
        self.thermal_model = str(physics_cfg.get("thermal_model", DEFAULT_THERMAL_MODEL)).lower()
        if self.thermal_model not in {"independent", "coupled"}:
            raise ValueError(f"unsupported physics.thermal_model={self.thermal_model}")
        self.safety_projection_version = normalize_safety_projection_version(
            self.thermal_model,
            physics_cfg.get("safety_projection_version", DEFAULT_SAFETY_PROJECTION_VERSION),
        )
        coupling_matrix = physics_cfg.get("thermal_coupling_matrix", DEFAULT_THERMAL_COUPLING_MATRIX.tolist())
        self.thermal_coupling_matrix = validate_coupling_matrix(coupling_matrix, n_tx=len(safety_cfg["current_max"]))
        self.thermal_coupling_matrix_hash = coupling_matrix_hash(self.thermal_coupling_matrix)
        self.projection_mode = str(safety_cfg.get("projection_mode", "thermal_cap")).lower()
        if self.projection_mode not in {
            "smooth",
            "smooth_relaxed",
            "thermal_cap",
            "hard_clip",
            "qos_aware_hard_clip",
            "dalal_safe",
        }:
            raise ValueError(f"unsupported safety.projection_mode={self.projection_mode}")
        self.action_decode_mode = str(safety_cfg.get("action_decode_mode", "tanh_affine")).lower()
        if self.action_decode_mode not in {"tanh_affine", "sigmoid_logit"}:
            raise ValueError(f"unsupported safety.action_decode_mode={self.action_decode_mode}")
        self.min_dwell_steps = int(safety_cfg["min_dwell_steps"])
        self.soft_alpha = float(safety_cfg["soft_alpha"])
        self.smooth_relaxed_margin_c = float(safety_cfg.get("smooth_relaxed_margin_c", 1.0))
        self.thermal_cap_margin_c = float(safety_cfg.get("thermal_cap_margin_c", 0.5))
        self.cutoff_alpha = float(safety_cfg["cutoff_alpha"])
        self.projection_beta = float(safety_cfg["projection_beta"])
        self.mask_floor = float(safety_cfg["mask_floor"])
        self.current_max = np.asarray(safety_cfg["current_max"], dtype=np.float32)
        self.bus_current_max = float(safety_cfg["bus_current_max"])
        self.thermal_safe = float(safety_cfg["thermal_safe"])
        self.thermal_cutoff = float(safety_cfg["thermal_cutoff"])

        tx_device = hybrid_cfg.get("tx_device", ["LED", "LD", "LD"])
        if len(tx_device) != int(self.current_max.shape[0]):
            raise ValueError("len(env.hybrid.tx_device) must match len(safety.current_max)")
        tx_enabled = hybrid_cfg.get("tx_enabled", [1.0] * int(self.current_max.shape[0]))
        if len(tx_enabled) != int(self.current_max.shape[0]):
            raise ValueError("len(env.hybrid.tx_enabled) must match len(safety.current_max)")
        self.tx_enabled = np.asarray(tx_enabled, dtype=np.float32)
        self.tx_enabled = np.where(self.tx_enabled > 0.5, 1.0, 0.0).astype(np.float32)
        tx_is_led = np.asarray([str(x).upper() == "LED" for x in tx_device], dtype=np.float32)
        tx_is_ld = 1.0 - tx_is_led
        thermal_led_coeff = float(hybrid_cfg.get("thermal_led_coeff", 1.00))
        thermal_ld_coeff = float(hybrid_cfg.get("thermal_ld_coeff", 1.25))
        self.tx_thermal_coeff = tx_is_led * thermal_led_coeff + tx_is_ld * thermal_ld_coeff
        self._dalal_eps = 1.0e-6
        self._dalal_iters = 4

    @staticmethod
    def decode_upper(upper_raw: int | np.ndarray) -> Tuple[int, int]:
        if isinstance(upper_raw, np.ndarray):
            idx = int(np.asarray(upper_raw).reshape(-1)[0])
        else:
            idx = int(upper_raw)
        idx = int(np.clip(idx, 0, 11))
        boost_combo = idx // 3
        mode = idx % 3
        return boost_combo, mode

    @staticmethod
    def encode_exec(boost_combo: int, mode: int) -> int:
        boost_combo = int(np.clip(boost_combo, 0, 3))
        mode = int(np.clip(mode, 0, 2))
        return int(boost_combo * 3 + mode)

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

    def _apply_dwell(self, desired_boost: int, mem: Dict[str, int]) -> Tuple[int, Dict[str, int]]:
        current = int(mem.get("current_boost", 0))
        dwell = int(mem.get("dwell_count", 0))

        if desired_boost == current:
            dwell += 1
            mem["dwell_count"] = dwell
            return current, mem

        if dwell < self.min_dwell_steps:
            mem["dwell_count"] = dwell + 1
            return current, mem

        mem["current_boost"] = desired_boost
        mem["dwell_count"] = 1
        return desired_boost, mem

    def preview_exec(self, upper_raw: int | np.ndarray, mem: Optional[Dict[str, int]] = None) -> Tuple[int, int]:
        """Preview next executed (boost, mode) without mutating agent memory."""
        mem_preview = dict(mem or {"current_boost": 0, "dwell_count": self.min_dwell_steps})
        desired_boost, mode = self.decode_upper(upper_raw)
        exec_boost, _ = self._apply_dwell(desired_boost, mem_preview)
        return int(exec_boost), int(mode)

    def raw_to_exec_map(self, mem: Optional[Dict[str, int]] = None) -> np.ndarray:
        exec_map = np.zeros(12, dtype=np.int64)
        for raw_idx in range(12):
            boost_exec, mode_exec = self.preview_exec(raw_idx, mem=mem)
            exec_map[raw_idx] = self.encode_exec(boost_exec, mode_exec)
        return exec_map

    def _smooth_bus_scale_np(self, total: float) -> float:
        total_arr = np.asarray([total], dtype=np.float32)
        overflow = _softplus_np(total_arr - self.bus_current_max, self.projection_beta)
        ratio = self.bus_current_max / (total_arr + overflow + 1e-6)
        gate = _sigmoid_np(self.projection_beta * (total_arr - self.bus_current_max))
        scale = (1.0 - gate) + gate * ratio
        return float(scale.item())

    def _smooth_bus_scale_torch(self, total: torch.Tensor) -> torch.Tensor:
        overflow = F.softplus(total - self.bus_current_max, beta=self.projection_beta)
        ratio = self.bus_current_max / (total + overflow + 1e-6)
        gate = torch.sigmoid(self.projection_beta * (total - self.bus_current_max))
        return (1.0 - gate) + gate * ratio

    def _hard_bus_scale_np(self, total: float) -> float:
        return float(min(1.0, self.bus_current_max / (float(total) + 1.0e-6)))

    def _hard_bus_scale_torch(self, total: torch.Tensor) -> torch.Tensor:
        ratio = self.bus_current_max / (total + 1.0e-6)
        return torch.clamp(ratio, max=1.0)

    def _relaxed_soft_scale_np(self, gap: np.ndarray, base_scale: np.ndarray) -> np.ndarray:
        if self.smooth_relaxed_margin_c <= 0.0:
            return base_scale.astype(np.float32)
        t = np.clip(gap / self.smooth_relaxed_margin_c, 0.0, 1.0).astype(np.float32)
        return (base_scale * (1.0 - t) + t).astype(np.float32)

    def _relaxed_soft_scale_torch(self, gap: torch.Tensor, base_scale: torch.Tensor) -> torch.Tensor:
        if self.smooth_relaxed_margin_c <= 0.0:
            return base_scale
        t = torch.clamp(gap / self.smooth_relaxed_margin_c, min=0.0, max=1.0)
        return base_scale * (1.0 - t) + t

    def _decode_frac_np(self, lower_raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(lower_raw, dtype=np.float32)
        if self.action_decode_mode == "sigmoid_logit":
            return _sigmoid_np(raw)
        return np.clip((raw + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)

    def _decode_frac_torch(self, lower_raw: torch.Tensor) -> torch.Tensor:
        if self.action_decode_mode == "sigmoid_logit":
            return torch.sigmoid(lower_raw)
        return torch.clamp((lower_raw + 1.0) * 0.5, min=0.0, max=1.0)

    def _thermal_coupling_term_np(self, temps: np.ndarray) -> np.ndarray:
        temps = np.asarray(temps, dtype=np.float32)
        if self.thermal_model != "coupled":
            return np.zeros_like(temps, dtype=np.float32)
        diff = temps.reshape(1, -1) - temps.reshape(-1, 1)
        return np.sum(self.thermal_coupling_matrix * diff, axis=1).astype(np.float32)

    def _thermal_base_np(self, temps: np.ndarray, amb_temp: float, gamma: float) -> Tuple[np.ndarray, np.ndarray]:
        temps = np.asarray(temps, dtype=np.float32)
        coupling = self._thermal_coupling_term_np(temps)
        base = ((1.0 - gamma) * temps + gamma * float(amb_temp) + coupling).astype(np.float32)
        return base, coupling

    def _thermal_coupling_term_torch(self, temps: torch.Tensor) -> torch.Tensor:
        if self.thermal_model != "coupled":
            return torch.zeros_like(temps)
        matrix = torch.as_tensor(self.thermal_coupling_matrix, dtype=temps.dtype, device=temps.device)
        diff = temps.unsqueeze(1) - temps.unsqueeze(2)
        return (matrix.unsqueeze(0) * diff).sum(dim=2)

    def _thermal_base_torch(self, temps: torch.Tensor, amb_temp: torch.Tensor, gamma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        coupling = self._thermal_coupling_term_torch(temps)
        base = (1.0 - gamma) * temps + gamma * amb_temp + coupling
        return base, coupling

    def _thermal_cap_np(
        self,
        *,
        currents: np.ndarray,
        temps: np.ndarray,
        amb_temp: float,
        gamma: float,
        delta: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        base, coupling = self._thermal_base_np(temps, amb_temp, gamma)
        allowed_rise = self.thermal_safe - self.thermal_cap_margin_c - base
        denom = max(float(delta), 0.0) * self.tx_thermal_coeff + 1.0e-6
        cap = np.sqrt(np.maximum(allowed_rise / denom, 0.0)).astype(np.float32)
        cap = np.minimum(cap, self.current_max).astype(np.float32)
        safe_currents = np.minimum(currents, cap).astype(np.float32)
        scale = safe_currents / np.maximum(currents, 1.0e-6)
        scale = np.where(currents > 1.0e-6, scale, 1.0).astype(np.float32)
        return safe_currents, cap, scale, base, coupling

    def _qos_aware_hard_clip_np(
        self,
        *,
        currents: np.ndarray,
        target_total: float,
        active_mask: np.ndarray,
        temps: np.ndarray,
        amb_temp: float,
        gamma: float,
        delta: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Hard per-source clipping with non-oracle current recovery.

        Naive hard clipping drops a source to zero whenever the one-step
        prediction crosses the thermal limit. This tuned hard-clip baseline is
        still a hard feasibility rule, but clips each active source to its
        current thermal cap and redistributes any lost current only to currently
        active sources with remaining thermal and bus headroom. It does not
        inspect future disturbances or realized reward.
        """
        safe_currents, cap, scale, base, coupling = self._thermal_cap_np(
            currents=currents,
            temps=temps,
            amb_temp=amb_temp,
            gamma=gamma,
            delta=delta,
        )
        active = (np.asarray(active_mask, dtype=np.float32) > 0.5).astype(np.float32)
        cap = np.minimum(cap, self.current_max * active).astype(np.float32)
        safe_currents = np.minimum(safe_currents, cap).astype(np.float32)
        shortage = float(max(0.0, min(float(target_total), self.bus_current_max) - float(np.sum(safe_currents))))
        if shortage > 1.0e-8:
            available = np.maximum(cap - safe_currents, 0.0).astype(np.float32) * active
            # Fill cooler / larger-headroom sources first. This is deterministic
            # and uses only the current thermal state.
            order = np.argsort(-available)
            for idx in order:
                room = float(available[idx])
                if room <= 1.0e-8 or shortage <= 1.0e-8:
                    continue
                add = min(room, shortage)
                safe_currents[idx] += add
                shortage -= add
        recovered_total = float(np.sum(safe_currents))
        if recovered_total > self.bus_current_max:
            safe_currents *= self._hard_bus_scale_np(recovered_total)
        scale = np.where(
            currents > 1.0e-6,
            safe_currents / np.maximum(currents, 1.0e-6),
            np.where(safe_currents > 1.0e-6, 1.0, 0.0),
        ).astype(np.float32)
        return safe_currents.astype(np.float32), cap.astype(np.float32), scale, base, coupling

    def _thermal_cap_torch(
        self,
        *,
        currents: torch.Tensor,
        temps: torch.Tensor,
        amb_temp: torch.Tensor,
        gamma: torch.Tensor,
        delta: torch.Tensor,
        current_max: torch.Tensor,
        thermal_coeff: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        base, coupling = self._thermal_base_torch(temps, amb_temp, gamma)
        allowed_rise = self.thermal_safe - self.thermal_cap_margin_c - base
        denom = torch.clamp(delta, min=0.0) * thermal_coeff + 1.0e-6
        cap = torch.sqrt(torch.clamp(allowed_rise / denom, min=0.0))
        cap = torch.minimum(cap, current_max)
        safe_currents = torch.minimum(currents, cap)
        scale = torch.where(
            currents > 1.0e-6,
            safe_currents / torch.clamp(currents, min=1.0e-6),
            torch.ones_like(currents),
        )
        return safe_currents, cap, scale, base, coupling

    def _qos_aware_hard_clip_torch(
        self,
        *,
        currents: torch.Tensor,
        target_total: torch.Tensor,
        active_mask: torch.Tensor,
        temps: torch.Tensor,
        amb_temp: torch.Tensor,
        gamma: torch.Tensor,
        delta: torch.Tensor,
        current_max: torch.Tensor,
        thermal_coeff: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        safe_currents, cap, scale, base, coupling = self._thermal_cap_torch(
            currents=currents,
            temps=temps,
            amb_temp=amb_temp,
            gamma=gamma,
            delta=delta,
            current_max=current_max,
            thermal_coeff=thermal_coeff,
        )
        active = (active_mask > 0.5).to(currents.dtype)
        cap = torch.minimum(cap, current_max * active)
        safe_currents = torch.minimum(safe_currents, cap)
        shortage = torch.clamp(
            torch.minimum(target_total, torch.full_like(target_total, self.bus_current_max))
            - safe_currents.sum(dim=1, keepdim=True),
            min=0.0,
        )
        available = torch.clamp(cap - safe_currents, min=0.0) * active
        weights = available / torch.clamp(available.sum(dim=1, keepdim=True), min=1.0e-6)
        safe_currents = safe_currents + weights * shortage
        total = safe_currents.sum(dim=1, keepdim=True)
        safe_currents = safe_currents * self._hard_bus_scale_torch(total)
        scale = torch.where(
            currents > 1.0e-6,
            safe_currents / torch.clamp(currents, min=1.0e-6),
            torch.where(safe_currents > 1.0e-6, torch.ones_like(currents), torch.zeros_like(currents)),
        )
        return safe_currents, cap, scale, base, coupling

    @staticmethod
    def _project_mode_params_np(mode: int, rho_raw: float, tau_raw: float) -> Tuple[float, float]:
        rho = float(np.clip(rho_raw, 0.0, 1.0))
        tau = float(np.clip(tau_raw, 0.0, 1.0))
        if mode == 0:  # PS
            return rho, 1.0
        if mode == 1:  # TS
            return 0.0, tau
        return rho, tau

    @staticmethod
    def _project_mode_params_torch(mode: torch.Tensor, rho_raw: torch.Tensor, tau_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mode = torch.clamp(mode.long().view(-1, 1), 0, 2)
        rho = torch.clamp(rho_raw, 0.0, 1.0)
        tau = torch.clamp(tau_raw, 0.0, 1.0)
        is_ps = (mode == 0).float()
        is_ts = (mode == 1).float()
        is_hy = 1.0 - is_ps - is_ts
        rho_exec = is_ps * rho + is_hy * rho
        tau_exec = is_ps * torch.ones_like(tau) + is_ts * tau + is_hy * tau
        rho_exec = rho_exec * (1.0 - is_ts)
        return rho_exec, tau_exec

    def project_np(
        self,
        upper_raw: int | np.ndarray,
        lower_raw: np.ndarray,
        temps: np.ndarray,
        amb_temp: float,
        gamma: float,
        delta: float,
        mem: Optional[Dict[str, int]] = None,
    ) -> Tuple[Dict[str, np.ndarray | int | float], Dict[str, int]]:
        mem = dict(mem or {"current_boost": 0, "dwell_count": self.min_dwell_steps})
        desired_boost, mode = self.decode_upper(upper_raw)
        exec_boost, mem = self._apply_dwell(desired_boost, mem)

        lower_raw = np.asarray(lower_raw, dtype=np.float32)
        decoded = self._decode_frac_np(lower_raw[:5])
        current_frac = decoded[:3].astype(np.float32)
        currents = current_frac * self.current_max
        raw_current_total = float(np.sum(currents))

        mask = self._boost_mask(exec_boost)
        if self.projection_mode in {"smooth", "smooth_relaxed", "thermal_cap", "dalal_safe"}:
            mask = self.mask_floor + (1.0 - self.mask_floor) * mask
        mask = self.tx_enabled * mask
        currents *= mask.astype(np.float32)
        masked_current_total = float(np.sum(currents))
        active_mask = mask.astype(np.float32)

        total = float(np.sum(currents))
        if self.projection_mode in {"smooth", "smooth_relaxed", "thermal_cap"}:
            currents *= self._smooth_bus_scale_np(total)
        elif self.projection_mode == "dalal_safe":
            currents = self._dalal_correct_currents_np(
                currents=currents,
                temps=np.asarray(temps, dtype=np.float32),
                amb_temp=float(amb_temp),
                gamma=float(gamma),
                delta=float(delta),
            )
        else:
            currents *= self._hard_bus_scale_np(total)
        bus_projected_current_total = float(np.sum(currents))

        rho_raw = float(decoded[3])
        tau_raw = float(decoded[4])
        rho, tau = self._project_mode_params_np(mode, rho_raw, tau_raw)

        temps = np.asarray(temps, dtype=np.float32)
        thermal_base, thermal_source_term = self._thermal_base_np(temps, float(amb_temp), float(gamma))
        T_pred = thermal_base + delta * self.tx_thermal_coeff * (currents**2)
        thermal_cap_current = self.current_max.astype(np.float32)
        thermal_cap_scale = np.ones_like(currents, dtype=np.float32)

        if self.projection_mode in {"smooth", "smooth_relaxed"}:
            gap = self.thermal_safe - T_pred
            soft_scale = _sigmoid_np(self.soft_alpha * gap)
            if self.projection_mode == "smooth_relaxed":
                soft_scale = self._relaxed_soft_scale_np(gap, soft_scale)
            cutoff_scale = _sigmoid_np(self.cutoff_alpha * (self.thermal_cutoff - T_pred))
            thermal_scale = (soft_scale * cutoff_scale).astype(np.float32)
        elif self.projection_mode == "thermal_cap":
            currents, thermal_cap_current, thermal_scale, thermal_base, thermal_source_term = self._thermal_cap_np(
                currents=currents,
                temps=temps,
                amb_temp=float(amb_temp),
                gamma=float(gamma),
                delta=float(delta),
            )
            T_pred = thermal_base + delta * self.tx_thermal_coeff * (currents**2)
            thermal_cap_scale = thermal_scale.astype(np.float32)
            soft_scale = thermal_scale.astype(np.float32)
            cutoff_scale = np.ones_like(currents, dtype=np.float32)
        elif self.projection_mode == "qos_aware_hard_clip":
            currents, thermal_cap_current, thermal_scale, thermal_base, thermal_source_term = self._qos_aware_hard_clip_np(
                currents=currents,
                target_total=bus_projected_current_total,
                active_mask=active_mask,
                temps=temps,
                amb_temp=float(amb_temp),
                gamma=float(gamma),
                delta=float(delta),
            )
            T_pred = thermal_base + delta * self.tx_thermal_coeff * (currents**2)
            thermal_cap_scale = thermal_scale.astype(np.float32)
            soft_scale = thermal_scale.astype(np.float32)
            cutoff_scale = np.ones_like(currents, dtype=np.float32)
        elif self.projection_mode == "dalal_safe":
            soft_scale = np.ones_like(currents, dtype=np.float32)
            cutoff_scale = np.ones_like(currents, dtype=np.float32)
            thermal_scale = np.clip(currents / np.maximum(self.current_max, 1.0e-6), 0.0, 1.0).astype(np.float32)
        else:
            soft_scale = (T_pred <= self.thermal_safe).astype(np.float32)
            cutoff_scale = np.ones_like(currents, dtype=np.float32)
            thermal_scale = (T_pred <= self.thermal_safe).astype(np.float32)
        if self.projection_mode not in {"dalal_safe", "thermal_cap", "qos_aware_hard_clip"}:
            currents *= thermal_scale
        projected_current_total = float(np.sum(currents))
        projection_compression_ratio = float(projected_current_total / max(raw_current_total, 1.0e-6))
        thermal_margin = (self.thermal_safe - T_pred).astype(np.float32)

        out = {
            "boost_combo_exec": int(exec_boost),
            "mode_exec": int(mode),
            "upper_idx_exec": int(self.encode_exec(exec_boost, mode)),
            "currents_exec": currents.astype(np.float32),
            "rho_exec": float(rho),
            "tau_exec": float(tau),
            "t_pred": T_pred.astype(np.float32),
            "thermal_scale": thermal_scale,
            "thermal_soft_scale": soft_scale.astype(np.float32),
            "thermal_cutoff_scale": cutoff_scale.astype(np.float32),
            "thermal_cap_current": thermal_cap_current.astype(np.float32),
            "thermal_cap_scale": thermal_cap_scale.astype(np.float32),
            "thermal_cap_margin_c": float(self.thermal_cap_margin_c),
            "thermal_source_model": self.thermal_model,
            "thermal_source_term": thermal_source_term.astype(np.float32),
            "thermal_base": thermal_base.astype(np.float32),
            "thermal_pred_temp": T_pred.astype(np.float32),
            "thermal_pred_margin": thermal_margin.astype(np.float32),
            "thermal_model": self.thermal_model,
            "thermal_coupling_matrix_hash": self.thermal_coupling_matrix_hash,
            "safety_projection_version": self.safety_projection_version,
            "thermal_margin": thermal_margin,
            "thermal_margin_min": float(np.min(thermal_margin)),
            "action_decode_mode": self.action_decode_mode,
            "raw_current_frac": current_frac.astype(np.float32),
            "rho_raw_decoded": float(rho_raw),
            "tau_raw_decoded": float(tau_raw),
            "raw_current_total": raw_current_total,
            "masked_current_total": masked_current_total,
            "bus_projected_current_total": bus_projected_current_total,
            "projected_current_total": projected_current_total,
            "projection_compression_ratio": projection_compression_ratio,
        }
        if self.thermal_model == "coupled":
            out["thermal_coupling_term"] = thermal_source_term.astype(np.float32)
            out["thermal_base_coupled"] = thermal_base.astype(np.float32)
        return out, mem

    def project_torch(
        self,
        lower_raw: torch.Tensor,
        boost_combo: torch.Tensor,
        mode: torch.Tensor,
        temps: torch.Tensor,
        amb_temp: torch.Tensor,
        gamma: torch.Tensor,
        delta: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if lower_raw.dim() == 1:
            lower_raw = lower_raw.unsqueeze(0)
        device = lower_raw.device

        current_max = torch.as_tensor(self.current_max, device=device, dtype=lower_raw.dtype).view(1, -1)
        decoded = self._decode_frac_torch(lower_raw[:, :5])
        current_frac = decoded[:, :3]
        currents = current_frac * current_max
        raw_current_total = currents.sum(dim=1)

        boost_combo = boost_combo.long().view(-1)

        table = torch.tensor(
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
            device=device,
            dtype=lower_raw.dtype,
        )
        boost_combo = torch.clamp(boost_combo, 0, table.shape[0] - 1)
        mask = table.index_select(0, boost_combo)
        if self.projection_mode in {"smooth", "smooth_relaxed", "thermal_cap", "dalal_safe"}:
            mask = self.mask_floor + (1.0 - self.mask_floor) * mask
        tx_enabled = torch.as_tensor(self.tx_enabled, dtype=lower_raw.dtype, device=device).view(1, -1)
        mask = tx_enabled * mask
        currents = currents * mask
        masked_current_total = currents.sum(dim=1)
        active_mask = mask

        total = currents.sum(dim=1, keepdim=True)
        if self.projection_mode in {"smooth", "smooth_relaxed", "thermal_cap"}:
            currents = currents * self._smooth_bus_scale_torch(total)
        elif self.projection_mode == "dalal_safe":
            currents = self._dalal_correct_currents_torch(
                currents=currents,
                temps=temps if temps.dim() > 1 else temps.unsqueeze(0),
                amb_temp=amb_temp if amb_temp.dim() > 1 else amb_temp.unsqueeze(1),
                gamma=gamma if gamma.dim() > 1 else gamma.unsqueeze(1),
                delta=delta if delta.dim() > 1 else delta.unsqueeze(1),
            )
        else:
            currents = currents * self._hard_bus_scale_torch(total)
        bus_projected_current_total = currents.sum(dim=1)

        rho_raw = decoded[:, 3:4]
        tau_raw = decoded[:, 4:5]
        rho, tau = self._project_mode_params_torch(mode, rho_raw, tau_raw)

        if temps.dim() == 1:
            temps = temps.unsqueeze(0)
        if amb_temp.dim() == 1:
            amb_temp = amb_temp.unsqueeze(1)
        if gamma.dim() == 1:
            gamma = gamma.unsqueeze(1)
        if delta.dim() == 1:
            delta = delta.unsqueeze(1)

        thermal_coeff = torch.as_tensor(
            self.tx_thermal_coeff,
            dtype=lower_raw.dtype,
            device=device,
        ).view(1, -1)
        thermal_base, thermal_source_term = self._thermal_base_torch(temps, amb_temp, gamma)
        T_pred = thermal_base + delta * thermal_coeff * (currents**2)
        thermal_cap_current = current_max.expand_as(currents)
        thermal_cap_scale = torch.ones_like(currents)
        if self.projection_mode in {"smooth", "smooth_relaxed"}:
            gap = self.thermal_safe - T_pred
            soft_scale = torch.sigmoid(self.soft_alpha * gap)
            if self.projection_mode == "smooth_relaxed":
                soft_scale = self._relaxed_soft_scale_torch(gap, soft_scale)
            cutoff_scale = torch.sigmoid(self.cutoff_alpha * (self.thermal_cutoff - T_pred))
            thermal_scale = soft_scale * cutoff_scale
        elif self.projection_mode == "thermal_cap":
            currents, thermal_cap_current, thermal_scale, thermal_base, thermal_source_term = self._thermal_cap_torch(
                currents=currents,
                temps=temps,
                amb_temp=amb_temp,
                gamma=gamma,
                delta=delta,
                current_max=current_max,
                thermal_coeff=thermal_coeff,
            )
            T_pred = thermal_base + delta * thermal_coeff * (currents**2)
            thermal_cap_scale = thermal_scale
            soft_scale = thermal_scale
            cutoff_scale = torch.ones_like(currents)
        elif self.projection_mode == "qos_aware_hard_clip":
            currents, thermal_cap_current, thermal_scale, thermal_base, thermal_source_term = self._qos_aware_hard_clip_torch(
                currents=currents,
                target_total=bus_projected_current_total.view(-1, 1),
                active_mask=active_mask,
                temps=temps,
                amb_temp=amb_temp,
                gamma=gamma,
                delta=delta,
                current_max=current_max,
                thermal_coeff=thermal_coeff,
            )
            T_pred = thermal_base + delta * thermal_coeff * (currents**2)
            thermal_cap_scale = thermal_scale
            soft_scale = thermal_scale
            cutoff_scale = torch.ones_like(currents)
        elif self.projection_mode == "dalal_safe":
            soft_scale = torch.ones_like(currents)
            cutoff_scale = torch.ones_like(currents)
            thermal_scale = torch.clamp(
                currents / torch.clamp(current_max, min=1.0e-6),
                min=0.0,
                max=1.0,
            )
        else:
            soft_scale = (T_pred <= self.thermal_safe).to(lower_raw.dtype)
            cutoff_scale = torch.ones_like(currents)
            thermal_scale = (T_pred <= self.thermal_safe).to(lower_raw.dtype)
        if self.projection_mode not in {"dalal_safe", "thermal_cap", "qos_aware_hard_clip"}:
            currents = currents * thermal_scale
        projected_current_total = currents.sum(dim=1)
        projection_compression_ratio = projected_current_total / torch.clamp(raw_current_total, min=1.0e-6)
        thermal_margin = self.thermal_safe - T_pred

        out = {
            "boost_combo_exec": boost_combo,
            "mode_exec": torch.clamp(mode.long().view(-1), 0, 2),
            "upper_idx_exec": torch.clamp(boost_combo.long().view(-1), 0, 3) * 3 + torch.clamp(mode.long().view(-1), 0, 2),
            "currents_exec": currents,
            "rho_exec": rho,
            "tau_exec": tau,
            "t_pred": T_pred,
            "thermal_scale": thermal_scale,
            "thermal_soft_scale": soft_scale,
            "thermal_cutoff_scale": cutoff_scale,
            "thermal_cap_current": thermal_cap_current,
            "thermal_cap_scale": thermal_cap_scale,
            "thermal_cap_margin_c": torch.full(
                (lower_raw.shape[0],),
                float(self.thermal_cap_margin_c),
                dtype=lower_raw.dtype,
                device=device,
            ),
            "thermal_source_model": self.thermal_model,
            "thermal_source_term": thermal_source_term,
            "thermal_base": thermal_base,
            "thermal_pred_temp": T_pred,
            "thermal_pred_margin": thermal_margin,
            "thermal_model": self.thermal_model,
            "thermal_coupling_matrix_hash": self.thermal_coupling_matrix_hash,
            "safety_projection_version": self.safety_projection_version,
            "thermal_margin": thermal_margin,
            "thermal_margin_min": thermal_margin.min(dim=1).values,
            "raw_current_frac": current_frac,
            "rho_raw_decoded": rho_raw,
            "tau_raw_decoded": tau_raw,
            "raw_current_total": raw_current_total,
            "masked_current_total": masked_current_total,
            "bus_projected_current_total": bus_projected_current_total,
            "projected_current_total": projected_current_total,
            "projection_compression_ratio": projection_compression_ratio,
        }
        if self.thermal_model == "coupled":
            out["thermal_coupling_term"] = thermal_source_term
            out["thermal_base_coupled"] = thermal_base
        return out

    def _dalal_correct_currents_np(
        self,
        *,
        currents: np.ndarray,
        temps: np.ndarray,
        amb_temp: float,
        gamma: float,
        delta: float,
    ) -> np.ndarray:
        currents = np.clip(np.asarray(currents, dtype=np.float32), 0.0, self.current_max)
        for _ in range(self._dalal_iters):
            total = float(np.sum(currents))
            g_bus = total - self.bus_current_max
            if g_bus > 0.0:
                grad_bus = np.ones_like(currents, dtype=np.float32)
                step = float(g_bus / (np.dot(grad_bus, grad_bus) + self._dalal_eps))
                currents = np.clip(currents - step * grad_bus, 0.0, self.current_max)

            base, _ = self._thermal_base_np(temps, amb_temp, gamma)
            t_pred = base + delta * self.tx_thermal_coeff * (currents**2)
            for idx in range(currents.shape[0]):
                g_temp = float(t_pred[idx] - self.thermal_safe)
                if g_temp <= 0.0:
                    continue
                grad_i = float(2.0 * delta * self.tx_thermal_coeff[idx] * currents[idx])
                step = float(g_temp / (grad_i * grad_i + self._dalal_eps))
                currents[idx] = float(np.clip(currents[idx] - step * grad_i, 0.0, self.current_max[idx]))
        return currents.astype(np.float32)

    def _dalal_correct_currents_torch(
        self,
        *,
        currents: torch.Tensor,
        temps: torch.Tensor,
        amb_temp: torch.Tensor,
        gamma: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        current_max = torch.as_tensor(self.current_max, dtype=currents.dtype, device=currents.device).view(1, -1)
        thermal_coeff = torch.as_tensor(self.tx_thermal_coeff, dtype=currents.dtype, device=currents.device).view(1, -1)
        currents = torch.minimum(torch.clamp(currents, min=0.0), current_max)
        eps = torch.as_tensor(self._dalal_eps, dtype=currents.dtype, device=currents.device)
        for _ in range(self._dalal_iters):
            total = currents.sum(dim=1, keepdim=True)
            g_bus = torch.clamp(total - self.bus_current_max, min=0.0)
            grad_bus = torch.ones_like(currents)
            step_bus = g_bus / (grad_bus.square().sum(dim=1, keepdim=True) + eps)
            currents = torch.minimum(torch.clamp(currents - step_bus * grad_bus, min=0.0), current_max)

            base, _ = self._thermal_base_torch(temps, amb_temp, gamma)
            t_pred = base + delta * thermal_coeff * (currents**2)
            g_temp = torch.clamp(t_pred - self.thermal_safe, min=0.0)
            grad_temp = 2.0 * delta * thermal_coeff * currents
            step_temp = g_temp / (grad_temp.square() + eps)
            currents = torch.minimum(torch.clamp(currents - step_temp * grad_temp, min=0.0), current_max)
        return currents
