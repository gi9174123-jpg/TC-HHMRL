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
from tchhmrl.safety.thermal_estimator import ThermalGainEstimator


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
        self.current_decoder = str(safety_cfg.get("current_decoder", "per_source")).lower()
        if self.current_decoder not in {"per_source", "structured_total_allocation"}:
            raise ValueError(f"unsupported safety.current_decoder={self.current_decoder}")
        self.allocation_logit_scale = float(safety_cfg.get("allocation_logit_scale", 1.0))
        self.inactive_source_mask_mode = str(safety_cfg.get("inactive_source_mask_mode", "hard_zero")).lower()
        if self.inactive_source_mask_mode not in {"hard_zero", "soft_floor"}:
            raise ValueError(f"unsupported safety.inactive_source_mask_mode={self.inactive_source_mask_mode}")
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
        self.thermal_parameter_source = str(
            safety_cfg.get("thermal_parameter_source", "nominal_plus_online_effective_gain")
        )
        self.gamma_nominal = float(safety_cfg.get("gamma_nominal", env_cfg.get("gamma", 0.07)))

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
        configured_gain = safety_cfg.get("effective_gain_initial", None)
        if configured_gain is None:
            configured_gain = float(env_cfg.get("delta", 1.0)) * self.tx_thermal_coeff
        self.thermal_estimator = ThermalGainEstimator(
            cfg.get("adaptive_thermal", {}),
            n_tx=int(self.current_max.shape[0]),
            thermal_safe=self.thermal_safe,
            initial_effective_gain=np.asarray(configured_gain, dtype=np.float32),
        )
        self._dalal_eps = 1.0e-6
        self._dalal_iters = 4
        shield_cfg = cfg.get("upper_safety_shield", {}) or {}
        self.upper_shield_enabled = bool(shield_cfg.get("enabled", False))
        self.upper_shield_ld_headroom_disable_c = float(shield_cfg.get("ld_headroom_disable_c", 1.0))
        self.upper_shield_ld_headroom_reenable_c = float(shield_cfg.get("ld_headroom_reenable_c", 2.0))
        self.upper_shield_led_headroom_disable_c = float(shield_cfg.get("led_headroom_disable_c", 0.5))
        self.upper_shield_critical_headroom_c = float(shield_cfg.get("critical_headroom_c", 0.25))
        self.upper_shield_always_allow_minimal_combo = bool(
            shield_cfg.get("always_allow_minimal_combo", shield_cfg.get("always_allow_anchor_only", True))
        )
        self.upper_shield_emergency_bypass_dwell = bool(shield_cfg.get("emergency_bypass_dwell", True))
        self.upper_shield_hot_locked = np.zeros(3, dtype=bool)

    def state_dict(self) -> Dict:
        return {
            "thermal_estimator": self.thermal_estimator.state_dict(),
            "upper_shield_hot_locked": self.upper_shield_hot_locked.astype(bool).copy(),
        }

    def load_state_dict(self, state: Dict) -> None:
        if not state:
            return
        if "thermal_estimator" in state:
            self.thermal_estimator.load_state_dict(state["thermal_estimator"])
        if "upper_shield_hot_locked" in state:
            locked = np.asarray(state["upper_shield_hot_locked"], dtype=bool).reshape(-1)
            self.upper_shield_hot_locked[:] = False
            self.upper_shield_hot_locked[: min(3, locked.size)] = locked[:3]

    def reset_runtime_state(self) -> None:
        self.upper_shield_hot_locked[:] = False

    def update_thermal_estimator(
        self,
        *,
        currents: np.ndarray,
        temps_before: np.ndarray,
        temps_after: np.ndarray,
        thermal_base: np.ndarray,
    ) -> Dict[str, np.ndarray | bool]:
        return self.thermal_estimator.update(
            currents=currents,
            temps_before=temps_before,
            temps_after=temps_after,
            thermal_base=thermal_base,
        )

    def observe_temperature(self, temps: np.ndarray) -> None:
        self.thermal_estimator.observe_temperature(temps)

    def thermal_diagnostics(self) -> Dict[str, np.ndarray | bool]:
        return self.thermal_estimator.diagnostics()

    def _safe_thermal_coeff_np(self) -> np.ndarray:
        return self.thermal_estimator.effective_gain_safe().astype(np.float32)

    def _safe_thermal_coeff_torch(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return torch.as_tensor(self.thermal_estimator.effective_gain_safe(), dtype=dtype, device=device).view(1, -1)

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

    def _upper_headroom(self, temps: Optional[np.ndarray] = None, headroom: Optional[np.ndarray] = None) -> np.ndarray | None:
        if headroom is not None:
            arr = np.asarray(headroom, dtype=np.float32).reshape(-1)
        elif temps is not None:
            arr = (float(self.thermal_safe) - np.asarray(temps, dtype=np.float32)).reshape(-1)
        else:
            return None
        if arr.size < 3:
            out = np.full(3, np.inf, dtype=np.float32)
            out[: arr.size] = arr
            return out
        return arr[:3].astype(np.float32)

    def upper_boost_allowed_mask(
        self,
        *,
        temps: Optional[np.ndarray] = None,
        headroom: Optional[np.ndarray] = None,
        update_latch: bool = True,
    ) -> np.ndarray:
        """Return a boost-combo mask from current thermal headroom.

        The shield is conservative and only removes LD-containing upper
        actions when the corresponding source is near the thermal boundary.
        The minimal active-source combo remains available as the emergency
        fallback because all valid upper combos contain source 0 by contract.
        """
        allowed = np.ones(4, dtype=bool)
        if not self.upper_shield_enabled:
            return allowed
        hr = self._upper_headroom(temps=temps, headroom=headroom)
        if hr is None:
            return allowed

        locked = self.upper_shield_hot_locked.copy()
        for src in (1, 2):
            if bool(hr[src] < self.upper_shield_ld_headroom_disable_c):
                locked[src] = True
            elif bool(hr[src] > self.upper_shield_ld_headroom_reenable_c):
                locked[src] = False
        if update_latch:
            self.upper_shield_hot_locked[:] = locked
        disable_ld1 = bool(locked[1] or hr[1] < self.upper_shield_critical_headroom_c)
        disable_ld2 = bool(locked[2] or hr[2] < self.upper_shield_critical_headroom_c)
        if disable_ld1:
            allowed[[1, 3]] = False
        if disable_ld2:
            allowed[[2, 3]] = False
        if bool(hr[0] < self.upper_shield_led_headroom_disable_c) and not self.upper_shield_always_allow_minimal_combo:
            allowed[[0, 1, 2, 3]] = False
        if self.upper_shield_always_allow_minimal_combo:
            allowed[0] = True
        if not np.any(allowed):
            allowed[0] = True
        return allowed

    def upper_raw_allowed_mask(
        self,
        *,
        temps: Optional[np.ndarray] = None,
        headroom: Optional[np.ndarray] = None,
        update_latch: bool = True,
    ) -> np.ndarray:
        allowed_boost = self.upper_boost_allowed_mask(temps=temps, headroom=headroom, update_latch=update_latch)
        mask = np.zeros(12, dtype=bool)
        for raw_idx in range(12):
            boost, _mode = self.decode_upper(raw_idx)
            mask[raw_idx] = bool(allowed_boost[boost])
        if not np.any(mask):
            mask[:3] = True
        return mask

    def _shield_boost_combo(
        self,
        desired_boost: int,
        *,
        temps: Optional[np.ndarray] = None,
        headroom: Optional[np.ndarray] = None,
        update_latch: bool = True,
    ) -> Tuple[int, np.ndarray, bool]:
        desired_boost = int(np.clip(desired_boost, 0, 3))
        allowed = self.upper_boost_allowed_mask(temps=temps, headroom=headroom, update_latch=update_latch)
        if bool(allowed[desired_boost]):
            return desired_boost, allowed, False

        desired_mask = self._boost_mask(desired_boost) > 0.0
        candidates = [idx for idx in range(4) if bool(allowed[idx])]
        if not candidates:
            return 0, allowed, desired_boost != 0
        # This is only the execution fallback for an already requested unsafe
        # combo. Normal selection receives the same mask and still chooses by Q.
        subset_candidates = [
            idx
            for idx in candidates
            if np.all((self._boost_mask(idx) > 0.0) <= desired_mask)
        ]
        if subset_candidates:
            best = max(subset_candidates, key=lambda idx: int(np.sum(self._boost_mask(idx) > 0.0)))
        else:
            best = 0 if 0 in candidates else candidates[0]
        return int(best), allowed, int(best) != desired_boost

    def _is_emergency_downshift(
        self,
        *,
        current_boost: int,
        desired_boost: int,
        temps: Optional[np.ndarray] = None,
        headroom: Optional[np.ndarray] = None,
    ) -> bool:
        if not self.upper_shield_emergency_bypass_dwell or not self.upper_shield_enabled:
            return False
        current_mask = self._boost_mask(current_boost) > 0.0
        desired_mask = self._boost_mask(desired_boost) > 0.0
        if not np.all(desired_mask <= current_mask) or np.array_equal(current_mask, desired_mask):
            return False
        hr = self._upper_headroom(temps=temps, headroom=headroom)
        if hr is None:
            return False
        removed = current_mask & (~desired_mask)
        hot_removed_ld = bool((removed[1] and hr[1] < self.upper_shield_ld_headroom_disable_c) or (removed[2] and hr[2] < self.upper_shield_ld_headroom_disable_c))
        critical = bool(float(np.nanmin(hr[1:3])) < self.upper_shield_critical_headroom_c)
        return bool(hot_removed_ld or critical)

    def _apply_dwell(
        self,
        desired_boost: int,
        mem: Dict[str, int],
        *,
        temps: Optional[np.ndarray] = None,
        headroom: Optional[np.ndarray] = None,
    ) -> Tuple[int, Dict[str, int]]:
        current = int(mem.get("current_boost", 0))
        dwell = int(mem.get("dwell_count", 0))

        if desired_boost == current:
            dwell += 1
            mem["dwell_count"] = dwell
            return current, mem

        if self._is_emergency_downshift(
            current_boost=current,
            desired_boost=desired_boost,
            temps=temps,
            headroom=headroom,
        ):
            mem["current_boost"] = desired_boost
            mem["dwell_count"] = 1
            return desired_boost, mem

        if dwell < self.min_dwell_steps:
            mem["dwell_count"] = dwell + 1
            return current, mem

        mem["current_boost"] = desired_boost
        mem["dwell_count"] = 1
        return desired_boost, mem

    def preview_exec(
        self,
        upper_raw: int | np.ndarray,
        mem: Optional[Dict[str, int]] = None,
        *,
        temps: Optional[np.ndarray] = None,
        headroom: Optional[np.ndarray] = None,
        update_latch: bool = True,
    ) -> Tuple[int, int]:
        """Preview next executed (boost, mode) without mutating agent memory."""
        mem_preview = dict(mem or {"current_boost": 0, "dwell_count": self.min_dwell_steps})
        desired_boost, mode = self.decode_upper(upper_raw)
        desired_boost, _allowed, _shielded = self._shield_boost_combo(
            desired_boost,
            temps=temps,
            headroom=headroom,
            update_latch=update_latch,
        )
        exec_boost, _ = self._apply_dwell(desired_boost, mem_preview, temps=temps, headroom=headroom)
        return int(exec_boost), int(mode)

    def raw_to_exec_map(
        self,
        mem: Optional[Dict[str, int]] = None,
        *,
        temps: Optional[np.ndarray] = None,
        headroom: Optional[np.ndarray] = None,
        update_latch: bool = True,
    ) -> np.ndarray:
        exec_map = np.zeros(12, dtype=np.int64)
        if update_latch:
            # Update the hysteresis latch once for this state, then reuse the
            # resulting mask for each raw-action preview below.
            self.upper_boost_allowed_mask(temps=temps, headroom=headroom, update_latch=True)
        for raw_idx in range(12):
            boost_exec, mode_exec = self.preview_exec(
                raw_idx,
                mem=mem,
                temps=temps,
                headroom=headroom,
                update_latch=False,
            )
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

    def _static_cap_redistribute_np(
        self,
        *,
        total_current: float,
        allocation: np.ndarray,
        active: np.ndarray,
    ) -> np.ndarray:
        cap = (self.current_max * (np.asarray(active, dtype=np.float32) > 0.0)).astype(np.float32)
        currents = np.zeros_like(cap, dtype=np.float32)
        remaining = float(min(max(total_current, 0.0), float(np.sum(cap)), self.bus_current_max))
        weights = np.asarray(allocation, dtype=np.float32) * (cap > 0.0).astype(np.float32)
        for _ in range(int(cap.size) + 1):
            if remaining <= 1.0e-8:
                break
            room = np.maximum(cap - currents, 0.0).astype(np.float32)
            unsat = (room > 1.0e-8).astype(np.float32)
            w = weights * unsat
            denom = float(np.sum(w))
            if denom <= 1.0e-8:
                w = unsat
                denom = float(np.sum(w))
            if denom <= 1.0e-8:
                break
            proposed = remaining * (w / denom)
            add = np.minimum(proposed, room).astype(np.float32)
            currents += add
            remaining -= float(np.sum(add))
        return currents.astype(np.float32)

    def _static_cap_redistribute_torch(
        self,
        *,
        total_current: torch.Tensor,
        allocation: torch.Tensor,
        active: torch.Tensor,
        current_max: torch.Tensor,
    ) -> torch.Tensor:
        cap = current_max * (active > 0.0).to(allocation.dtype)
        currents = torch.zeros_like(cap)
        remaining = torch.minimum(
            torch.clamp(total_current, min=0.0),
            torch.minimum(
                cap.sum(dim=1, keepdim=True),
                torch.full_like(total_current, float(self.bus_current_max)),
            ),
        )
        weights = allocation * (cap > 0.0).to(allocation.dtype)
        for _ in range(int(cap.shape[1]) + 1):
            room = torch.clamp(cap - currents, min=0.0)
            unsat = (room > 1.0e-8).to(allocation.dtype)
            w = weights * unsat
            denom = w.sum(dim=1, keepdim=True)
            w = torch.where(denom > 1.0e-8, w, unsat)
            denom = w.sum(dim=1, keepdim=True)
            proposed = remaining * w / torch.clamp(denom, min=1.0e-8)
            add = torch.minimum(proposed, room)
            add = torch.where((remaining > 1.0e-8) & (denom > 1.0e-8), add, torch.zeros_like(add))
            currents = currents + add
            remaining = remaining - add.sum(dim=1, keepdim=True)
        return currents

    def _decode_current_request_np(
        self,
        lower_raw: np.ndarray,
        *,
        exec_boost: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray | float]]:
        decoded = self._decode_frac_np(lower_raw[:5])
        active = (self._boost_mask(exec_boost) * self.tx_enabled).astype(np.float32)
        if self.current_decoder != "structured_total_allocation":
            current_frac = decoded[:3].astype(np.float32)
            currents = (current_frac * self.current_max).astype(np.float32)
            return currents, current_frac, decoded, {
                "current_requested": currents.copy(),
                "actor_total_current_requested": float(np.sum(currents)),
                "actor_allocation": np.full(3, np.nan, dtype=np.float32),
                "actor_inactive_allocation_sum": 0.0,
                "actor_per_source_clip_count": 0.0,
                "structured_actor_per_source_clip_rate": 0.0,
                "active_source_mask": active,
            }

        active_capacity = float(min(self.bus_current_max, float(np.sum(active * self.current_max))))
        total_requested = float(decoded[0]) * active_capacity
        logits = np.asarray(
            [0.0, self.allocation_logit_scale * lower_raw[1], self.allocation_logit_scale * lower_raw[2]],
            dtype=np.float32,
        )
        masked_logits = np.where(active > 0.0, logits, -1.0e9).astype(np.float32)
        shifted = masked_logits - np.max(masked_logits)
        exp_logits = np.exp(shifted) * active
        denom = float(np.sum(exp_logits))
        allocation = (
            np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
            if denom <= 0.0
            else (exp_logits / denom).astype(np.float32)
        )
        requested_initial = (total_requested * allocation).astype(np.float32)
        clipped = self._static_cap_redistribute_np(
            total_current=total_requested,
            allocation=allocation,
            active=active,
        )
        clip_count = float(np.sum(requested_initial > (self.current_max + 1.0e-6)))
        current_frac = (clipped / np.maximum(self.current_max, 1.0e-6)).astype(np.float32)
        inactive_sum = float(np.sum(allocation[active <= 0.0]))
        return clipped, current_frac, decoded, {
            "current_requested": clipped.astype(np.float32),
            "current_requested_pre_static_cap": requested_initial.astype(np.float32),
            "actor_total_current_requested": float(total_requested),
            "actor_active_current_capacity": float(active_capacity),
            "actor_allocation": allocation.astype(np.float32),
            "actor_inactive_allocation_sum": inactive_sum,
            "actor_per_source_clip_count": clip_count,
            "structured_actor_per_source_clip_rate": float(clip_count / max(float(np.sum(active > 0.0)), 1.0)),
            "active_source_mask": active,
        }

    def _decode_current_request_torch(
        self,
        lower_raw: torch.Tensor,
        *,
        boost_combo: torch.Tensor,
        current_max: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        decoded = self._decode_frac_torch(lower_raw[:, :5])
        table = torch.tensor(
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
            device=lower_raw.device,
            dtype=lower_raw.dtype,
        )
        boost_idx = torch.clamp(boost_combo.long().view(-1), 0, table.shape[0] - 1)
        active = table.index_select(0, boost_idx)
        tx_enabled = torch.as_tensor(self.tx_enabled, dtype=lower_raw.dtype, device=lower_raw.device).view(1, -1)
        active = active * tx_enabled
        if self.current_decoder != "structured_total_allocation":
            current_frac = decoded[:, :3]
            currents = current_frac * current_max
            return currents, current_frac, decoded, {
                "current_requested": currents,
                "actor_total_current_requested": currents.sum(dim=1),
                "actor_allocation": torch.full_like(currents, float("nan")),
                "actor_inactive_allocation_sum": torch.zeros((lower_raw.shape[0],), dtype=lower_raw.dtype, device=lower_raw.device),
                "actor_per_source_clip_count": torch.zeros((lower_raw.shape[0],), dtype=lower_raw.dtype, device=lower_raw.device),
                "structured_actor_per_source_clip_rate": torch.zeros((lower_raw.shape[0],), dtype=lower_raw.dtype, device=lower_raw.device),
                "active_source_mask": active,
            }

        active_capacity = torch.minimum(
            torch.full((lower_raw.shape[0], 1), float(self.bus_current_max), dtype=lower_raw.dtype, device=lower_raw.device),
            (active * current_max).sum(dim=1, keepdim=True),
        )
        total_requested = decoded[:, 0:1] * active_capacity
        logits = torch.cat(
            [
                torch.zeros_like(lower_raw[:, 1:2]),
                self.allocation_logit_scale * lower_raw[:, 1:3],
            ],
            dim=1,
        )
        masked_logits = torch.where(active > 0.0, logits, torch.full_like(logits, -1.0e9))
        allocation = torch.softmax(masked_logits, dim=1) * active
        allocation = allocation / torch.clamp(allocation.sum(dim=1, keepdim=True), min=1.0e-6)
        requested_initial = total_requested * allocation
        clipped = self._static_cap_redistribute_torch(
            total_current=total_requested,
            allocation=allocation,
            active=active,
            current_max=current_max,
        )
        clip_count = (requested_initial > (current_max + 1.0e-6)).to(lower_raw.dtype).sum(dim=1)
        active_count = torch.clamp((active > 0.0).to(lower_raw.dtype).sum(dim=1), min=1.0)
        return clipped, clipped / torch.clamp(current_max, min=1.0e-6), decoded, {
            "current_requested": clipped,
            "current_requested_pre_static_cap": requested_initial,
            "actor_total_current_requested": total_requested.view(-1),
            "actor_active_current_capacity": active_capacity.view(-1),
            "actor_allocation": allocation,
            "actor_inactive_allocation_sum": (allocation * (active <= 0.0).to(lower_raw.dtype)).sum(dim=1),
            "actor_per_source_clip_count": clip_count,
            "structured_actor_per_source_clip_rate": clip_count / active_count,
            "active_source_mask": active,
        }

    def _thermal_coupling_term_np(self, temps: np.ndarray) -> np.ndarray:
        temps = np.asarray(temps, dtype=np.float32)
        if self.thermal_model != "coupled":
            return np.zeros_like(temps, dtype=np.float32)
        diff = temps.reshape(1, -1) - temps.reshape(-1, 1)
        return np.sum(self.thermal_coupling_matrix * diff, axis=1).astype(np.float32)

    def _thermal_base_np(self, temps: np.ndarray, amb_temp: float, gamma: float | None = None) -> Tuple[np.ndarray, np.ndarray]:
        temps = np.asarray(temps, dtype=np.float32)
        coupling = self._thermal_coupling_term_np(temps)
        gamma_nominal = float(self.gamma_nominal)
        base = ((1.0 - gamma_nominal) * temps + gamma_nominal * float(amb_temp) + coupling).astype(np.float32)
        return base, coupling

    def _thermal_coupling_term_torch(self, temps: torch.Tensor) -> torch.Tensor:
        if self.thermal_model != "coupled":
            return torch.zeros_like(temps)
        matrix = torch.as_tensor(self.thermal_coupling_matrix, dtype=temps.dtype, device=temps.device)
        diff = temps.unsqueeze(1) - temps.unsqueeze(2)
        return (matrix.unsqueeze(0) * diff).sum(dim=2)

    def _thermal_base_torch(
        self,
        temps: torch.Tensor,
        amb_temp: torch.Tensor,
        gamma: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        coupling = self._thermal_coupling_term_torch(temps)
        gamma_nominal = torch.as_tensor(float(self.gamma_nominal), dtype=temps.dtype, device=temps.device)
        base = (1.0 - gamma_nominal) * temps + gamma_nominal * amb_temp + coupling
        return base, coupling

    def _thermal_cap_np(
        self,
        *,
        currents: np.ndarray,
        temps: np.ndarray,
        amb_temp: float,
        gamma: float | None = None,
        delta: float | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        base, coupling = self._thermal_base_np(temps, amb_temp, gamma)
        allowed_rise = self.thermal_safe - self.thermal_cap_margin_c - base
        denom = self._safe_thermal_coeff_np() + 1.0e-6
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
        gamma: float | None = None,
        delta: float | None = None,
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
        gamma: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
        current_max: torch.Tensor,
        thermal_coeff: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        base, coupling = self._thermal_base_torch(temps, amb_temp, gamma)
        allowed_rise = self.thermal_safe - self.thermal_cap_margin_c - base
        denom = thermal_coeff + 1.0e-6
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
        gamma: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
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
        for _ in range(currents.shape[1]):
            available = torch.clamp(cap - safe_currents, min=0.0) * active
            idx = torch.argmax(available, dim=1, keepdim=True)
            room = torch.gather(available, dim=1, index=idx)
            add = torch.minimum(room, shortage)
            add = torch.where((room > 1.0e-8) & (shortage > 1.0e-8), add, torch.zeros_like(add))
            safe_currents = safe_currents.scatter_add(1, idx, add)
            shortage = torch.clamp(shortage - add, min=0.0)
        safe_currents = torch.minimum(safe_currents, cap)
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
        gamma: float | None = None,
        delta: float | None = None,
        mem: Optional[Dict[str, int]] = None,
    ) -> Tuple[Dict[str, np.ndarray | int | float], Dict[str, int]]:
        mem = dict(mem or {"current_boost": 0, "dwell_count": self.min_dwell_steps})
        desired_boost, mode = self.decode_upper(upper_raw)
        shielded_boost, shield_allowed, shield_changed = self._shield_boost_combo(
            desired_boost,
            temps=temps,
            headroom=None,
        )
        exec_boost, mem = self._apply_dwell(shielded_boost, mem, temps=temps, headroom=None)

        lower_raw = np.asarray(lower_raw, dtype=np.float32)
        currents, current_frac, decoded, current_aux = self._decode_current_request_np(lower_raw, exec_boost=exec_boost)
        current_requested = np.asarray(current_aux["current_requested"], dtype=np.float32)
        raw_current_total = float(np.sum(currents))

        mask = self._boost_mask(exec_boost)
        if (
            self.inactive_source_mask_mode == "soft_floor"
            and self.projection_mode in {"smooth", "smooth_relaxed", "thermal_cap", "dalal_safe"}
        ):
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
            )
        else:
            currents *= self._hard_bus_scale_np(total)
        bus_projected_current_total = float(np.sum(currents))

        rho_raw = float(decoded[3])
        tau_raw = float(decoded[4])
        rho, tau = self._project_mode_params_np(mode, rho_raw, tau_raw)

        temps = np.asarray(temps, dtype=np.float32)
        thermal_base, thermal_source_term = self._thermal_base_np(temps, float(amb_temp))
        thermal_coeff_safe_np = self._safe_thermal_coeff_np()
        T_pred = thermal_base + thermal_coeff_safe_np * (currents**2)
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
            )
            thermal_coeff_safe_np = self._safe_thermal_coeff_np()
            T_pred = thermal_base + thermal_coeff_safe_np * (currents**2)
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
            )
            thermal_coeff_safe_np = self._safe_thermal_coeff_np()
            T_pred = thermal_base + thermal_coeff_safe_np * (currents**2)
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
            "upper_shield_enabled": bool(self.upper_shield_enabled),
            "upper_shield_applied": bool(shield_changed or exec_boost != desired_boost),
            "upper_shield_requested_boost": int(desired_boost),
            "upper_shield_selected_boost": int(shielded_boost),
            "upper_shield_allowed_anchor": float(bool(shield_allowed[0])),
            "upper_shield_allowed_ld1": float(bool(shield_allowed[1])),
            "upper_shield_allowed_ld2": float(bool(shield_allowed[2])),
            "upper_shield_allowed_all": float(bool(shield_allowed[3])),
            "upper_shield_locked_ld1": float(bool(self.upper_shield_hot_locked[1])),
            "upper_shield_locked_ld2": float(bool(self.upper_shield_hot_locked[2])),
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
            "thermal_parameter_source": self.thermal_parameter_source,
            "gamma_nominal": float(self.gamma_nominal),
            "current_decoder": self.current_decoder,
            "inactive_source_mask_mode": self.inactive_source_mask_mode,
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
            "current_requested": current_requested.astype(np.float32),
            "current_requested_pre_static_cap": np.asarray(
                current_aux.get("current_requested_pre_static_cap", current_requested),
                dtype=np.float32,
            ),
            "actor_total_current_requested": float(current_aux["actor_total_current_requested"]),
            "actor_active_current_capacity": float(current_aux.get("actor_active_current_capacity", self.bus_current_max)),
            "actor_allocation_anchor": float(np.asarray(current_aux["actor_allocation"])[0]),
            "actor_allocation_ld1": float(np.asarray(current_aux["actor_allocation"])[1]),
            "actor_allocation_ld2": float(np.asarray(current_aux["actor_allocation"])[2]),
            "actor_inactive_allocation_sum": float(current_aux["actor_inactive_allocation_sum"]),
            "actor_per_source_clip_count": float(current_aux["actor_per_source_clip_count"]),
            "structured_actor_per_source_clip_rate": float(current_aux["structured_actor_per_source_clip_rate"]),
            "structured_actor_bus_clip_rate": float(raw_current_total > self.bus_current_max + 1.0e-6),
            "mode_effective_latent_dim": float(4 if mode in {0, 1} else 5),
        }
        diag = self.thermal_diagnostics()
        local_headroom = (self.thermal_safe - temps).astype(np.float32)
        out.update(diag)
        out["thermal_headroom_observed"] = local_headroom
        out["thermal_headroom"] = local_headroom
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
        gamma: torch.Tensor | None = None,
        delta: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if lower_raw.dim() == 1:
            lower_raw = lower_raw.unsqueeze(0)
        device = lower_raw.device

        current_max = torch.as_tensor(self.current_max, device=device, dtype=lower_raw.dtype).view(1, -1)
        boost_combo = boost_combo.long().view(-1)
        currents, current_frac, decoded, current_aux = self._decode_current_request_torch(
            lower_raw,
            boost_combo=boost_combo,
            current_max=current_max,
        )
        current_requested = current_aux["current_requested"]
        raw_current_total = currents.sum(dim=1)

        table = torch.tensor(
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
            device=device,
            dtype=lower_raw.dtype,
        )
        boost_combo = torch.clamp(boost_combo, 0, table.shape[0] - 1)
        mask = table.index_select(0, boost_combo)
        if (
            self.inactive_source_mask_mode == "soft_floor"
            and self.projection_mode in {"smooth", "smooth_relaxed", "thermal_cap", "dalal_safe"}
        ):
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
        thermal_coeff = self._safe_thermal_coeff_torch(dtype=lower_raw.dtype, device=device)
        thermal_base, thermal_source_term = self._thermal_base_torch(temps, amb_temp)
        T_pred = thermal_base + thermal_coeff * (currents**2)
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
                current_max=current_max,
                thermal_coeff=thermal_coeff,
            )
            T_pred = thermal_base + thermal_coeff * (currents**2)
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
                current_max=current_max,
                thermal_coeff=thermal_coeff,
            )
            T_pred = thermal_base + thermal_coeff * (currents**2)
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
            "thermal_parameter_source": self.thermal_parameter_source,
            "gamma_nominal": torch.full((lower_raw.shape[0],), float(self.gamma_nominal), dtype=lower_raw.dtype, device=device),
            "current_decoder": self.current_decoder,
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
            "current_requested": current_requested,
            "current_requested_pre_static_cap": current_aux.get("current_requested_pre_static_cap", current_requested),
            "actor_total_current_requested": current_aux["actor_total_current_requested"],
            "actor_active_current_capacity": current_aux.get(
                "actor_active_current_capacity",
                torch.full((lower_raw.shape[0],), float(self.bus_current_max), dtype=lower_raw.dtype, device=device),
            ),
            "actor_allocation": current_aux["actor_allocation"],
            "actor_inactive_allocation_sum": current_aux["actor_inactive_allocation_sum"],
            "actor_per_source_clip_count": current_aux["actor_per_source_clip_count"],
            "structured_actor_per_source_clip_rate": current_aux["structured_actor_per_source_clip_rate"],
            "structured_actor_bus_clip_rate": (raw_current_total > float(self.bus_current_max + 1.0e-6)).to(lower_raw.dtype),
            "mode_effective_latent_dim": torch.where(
                torch.clamp(mode.long().view(-1), 0, 2) == 2,
                torch.full((lower_raw.shape[0],), 5.0, dtype=lower_raw.dtype, device=device),
                torch.full((lower_raw.shape[0],), 4.0, dtype=lower_raw.dtype, device=device),
            ),
        }
        diag = self.thermal_diagnostics()
        for key, val in diag.items():
            if isinstance(val, np.ndarray):
                tensor = torch.as_tensor(val, dtype=lower_raw.dtype, device=device)
                if tensor.dim() == 1:
                    tensor = tensor.view(1, -1).expand(lower_raw.shape[0], -1)
                out[key] = tensor
            else:
                out[key] = torch.full((lower_raw.shape[0],), float(bool(val)), dtype=lower_raw.dtype, device=device)
        out["thermal_headroom_observed"] = self.thermal_safe - temps
        out["thermal_headroom"] = self.thermal_safe - temps
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
    ) -> np.ndarray:
        currents = np.clip(np.asarray(currents, dtype=np.float32), 0.0, self.current_max)
        for _ in range(self._dalal_iters):
            total = float(np.sum(currents))
            g_bus = total - self.bus_current_max
            if g_bus > 0.0:
                grad_bus = np.ones_like(currents, dtype=np.float32)
                step = float(g_bus / (np.dot(grad_bus, grad_bus) + self._dalal_eps))
                currents = np.clip(currents - step * grad_bus, 0.0, self.current_max)

            base, _ = self._thermal_base_np(temps, amb_temp)
            effective_gain = self._safe_thermal_coeff_np()
            t_pred = base + effective_gain * (currents**2)
            for idx in range(currents.shape[0]):
                g_temp = float(t_pred[idx] - self.thermal_safe)
                if g_temp <= 0.0:
                    continue
                grad_i = float(2.0 * effective_gain[idx] * currents[idx])
                step = float(g_temp / (grad_i * grad_i + self._dalal_eps))
                currents[idx] = float(np.clip(currents[idx] - step * grad_i, 0.0, self.current_max[idx]))
        base, _ = self._thermal_base_np(temps, amb_temp)
        effective_gain = self._safe_thermal_coeff_np()
        cap = np.sqrt(
            np.maximum((self.thermal_safe - base) / np.maximum(effective_gain, 1.0e-6), 0.0)
        ).astype(np.float32)
        currents = np.minimum(currents, np.minimum(cap, self.current_max)).astype(np.float32)
        return currents.astype(np.float32)

    def _dalal_correct_currents_torch(
        self,
        *,
        currents: torch.Tensor,
        temps: torch.Tensor,
        amb_temp: torch.Tensor,
    ) -> torch.Tensor:
        current_max = torch.as_tensor(self.current_max, dtype=currents.dtype, device=currents.device).view(1, -1)
        effective_gain = self._safe_thermal_coeff_torch(dtype=currents.dtype, device=currents.device)
        currents = torch.minimum(torch.clamp(currents, min=0.0), current_max)
        eps = torch.as_tensor(self._dalal_eps, dtype=currents.dtype, device=currents.device)
        for _ in range(self._dalal_iters):
            total = currents.sum(dim=1, keepdim=True)
            g_bus = torch.clamp(total - self.bus_current_max, min=0.0)
            grad_bus = torch.ones_like(currents)
            step_bus = g_bus / (grad_bus.square().sum(dim=1, keepdim=True) + eps)
            currents = torch.minimum(torch.clamp(currents - step_bus * grad_bus, min=0.0), current_max)

            base, _ = self._thermal_base_torch(temps, amb_temp)
            t_pred = base + effective_gain * (currents**2)
            g_temp = torch.clamp(t_pred - self.thermal_safe, min=0.0)
            grad_temp = 2.0 * effective_gain * currents
            step_temp = g_temp / (grad_temp.square() + eps)
            currents = torch.minimum(torch.clamp(currents - step_temp * grad_temp, min=0.0), current_max)
        base, _ = self._thermal_base_torch(temps, amb_temp)
        cap = torch.sqrt(torch.clamp((self.thermal_safe - base) / torch.clamp(effective_gain, min=1.0e-6), min=0.0))
        currents = torch.minimum(currents, torch.minimum(cap, current_max))
        return currents
