from __future__ import annotations

import time
from typing import Dict

import numpy as np

from tchhmrl.baselines.common import (
    BasePaperBaseline,
    current_action_from_frac,
    expected_step_metrics,
    mpc_nominal_scoring_model_from_cfg,
    monotonic_latency_ms,
)
from tchhmrl.envs.uw_slipt_env import MultiTxUwSliptEnv


class MpcGridBaseline(BasePaperBaseline):
    """Deterministic one-step MPC-Grid optimizer without future-disturbance oracle access."""

    baseline_family = "mpc_grid"

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        opts = cfg.get("baselines", {}).get("mpc_grid", {})
        self.current_templates = dict(
            opts.get(
                "current_templates",
                {
                    "safe_anchor": [0.55, 0.00, 0.00],
                    "balanced_hybrid": [0.45, 0.40, 0.40],
                    "ld1_preferred": [0.40, 0.55, 0.20],
                    "ld2_preferred": [0.40, 0.20, 0.55],
                    "thermal_spread": [0.35, 0.45, 0.45],
                },
            )
        )
        self.rho_grid = np.asarray(opts.get("rho_grid", [0.10, 0.30, 0.50, 0.70, 0.90]), dtype=np.float32)
        self.tau_grid = np.asarray(opts.get("tau_grid", [0.10, 0.30, 0.50, 0.70, 0.90]), dtype=np.float32)
        self.scoring_model = str(opts.get("scoring_model", "perfect_env")).strip().lower()
        if self.scoring_model not in {"perfect_env", "nominal_model"}:
            raise ValueError(
                "baselines.mpc_grid.scoring_model must be 'perfect_env' or 'nominal_model', "
                f"got {self.scoring_model!r}"
            )
        self.action_contract = "boost_mode_structured_current_template_receiver_grid"
        self.candidate_count = self._count_candidates()

    def _count_candidates(self) -> int:
        count = 0
        for _boost in range(4):
            for mode in range(3):
                for _template in self.current_templates:
                    if mode == 0:
                        count += len(self.rho_grid)
                    elif mode == 1:
                        count += len(self.tau_grid)
                    else:
                        count += len(self.rho_grid) * len(self.tau_grid)
        return int(count)

    def _mode_receiver_grid(self, mode: int):
        if mode == 0:
            for rho in self.rho_grid:
                yield float(rho), 1.0
        elif mode == 1:
            for tau in self.tau_grid:
                yield 0.0, float(tau)
        else:
            for rho in self.rho_grid:
                for tau in self.tau_grid:
                    yield float(rho), float(tau)

    def _scoring_model_snapshot(self, env: MultiTxUwSliptEnv) -> Dict[str, object] | None:
        if self.scoring_model == "perfect_env":
            return None
        return mpc_nominal_scoring_model_from_cfg(self.cfg, env)

    def _project_raw_action_for_scoring(
        self,
        env: MultiTxUwSliptEnv,
        upper_raw: int,
        lower_raw: np.ndarray,
        scoring_model: Dict[str, object] | None,
    ) -> Dict[str, object]:
        if scoring_model is None:
            safe, _ = self._project_raw_action(env, upper_raw, lower_raw, commit=False)
            return safe
        safe, _ = self.safety.project_np(
            upper_raw,
            np.asarray(lower_raw, dtype=np.float32),
            temps=env.temps.copy().astype(np.float32),
            amb_temp=float(scoring_model.get("amb_temp", env.amb_temp)),
            gamma=float(scoring_model.get("gamma", env.gamma)),
            delta=float(scoring_model.get("delta", env.delta)),
            mem=dict(self.safety_mem),
        )
        return safe

    def score_candidate(
        self,
        env: MultiTxUwSliptEnv,
        *,
        boost_combo: int,
        mode: int,
        template_name: str,
        rho: float,
        tau: float,
    ) -> tuple[int, np.ndarray, Dict[str, object], Dict[str, float | np.ndarray]]:
        upper_raw = int(boost_combo) * 3 + int(mode)
        lower_raw = current_action_from_frac(
            np.asarray(self.current_templates[template_name], dtype=np.float32),
            rho=rho,
            tau=tau,
            action_decode_mode=self.safety.action_decode_mode,
        )
        scoring_model = self._scoring_model_snapshot(env)
        safe = self._project_raw_action_for_scoring(env, upper_raw, lower_raw, scoring_model)
        metrics = expected_step_metrics(env, safe, model=scoring_model)
        return upper_raw, lower_raw, safe, metrics

    def act(self, obs: np.ndarray, env: MultiTxUwSliptEnv, eval_mode: bool = False) -> tuple[Dict, Dict]:
        del obs, eval_mode
        start = time.perf_counter()
        best = None
        seen = 0
        for boost_combo in range(4):
            for mode in range(3):
                for template_name in self.current_templates:
                    for rho, tau in self._mode_receiver_grid(mode):
                        cand = self.score_candidate(
                            env,
                            boost_combo=boost_combo,
                            mode=mode,
                            template_name=template_name,
                            rho=rho,
                            tau=tau,
                        )
                        seen += 1
                        score = float(cand[3]["reward"])
                        if best is None or score > best[0]:
                            best = (score, boost_combo, mode, template_name, rho, tau, cand)
        assert best is not None
        _, boost_combo, mode, template_name, rho, tau, cand = best
        upper_raw, lower_raw, _, metrics = cand
        safe, _ = self._project_raw_action(env, upper_raw, lower_raw, commit=True)
        action, aux = self._action_from_safe(
            upper_raw,
            lower_raw,
            safe,
            aux_extra={
                "candidate_count": int(seen),
                "online_latency_ms": monotonic_latency_ms(start),
                "selected_boost_combo": int(boost_combo),
                "selected_mode": int(mode),
                "selected_template": str(template_name),
                "selected_rho": float(rho),
                "selected_env_rho": float(safe["rho_exec"]),
                "selected_paper_rho": float(1.0 - float(safe["rho_exec"])),
                "selected_tau": float(tau),
                "mpc_grid_score": float(metrics["reward"]),
                "predicted_qos_rate": float(metrics["qos_rate"]),
                "predicted_eh_metric": float(metrics["eh_metric"]),
                "predicted_snr": float(metrics["snr"]),
                "predicted_bus_utilization": float(metrics["bus_utilization"]),
                "candidate_state_protocol": "deterministic_expected_one_step_no_rng_mutation",
                "mpc_scoring_model": str(metrics.get("scoring_model", self.scoring_model)),
                "mpc_information_boundary": (
                    "perfect_model_upper_bound_uses_true_current_model"
                    if self.scoring_model == "perfect_env"
                    else "nominal_model_candidate_scoring_no_true_gamma_delta"
                ),
                "mpc_scoring_uses_true_gamma_delta": bool(self.scoring_model == "perfect_env"),
                "mpc_scoring_gamma": float(metrics.get("scoring_gamma", env.gamma)),
                "mpc_scoring_delta": float(metrics.get("scoring_delta", env.delta)),
                "mpc_true_gamma": float(env.gamma),
                "mpc_true_delta": float(env.delta),
                "mpc_gamma_mismatch": float(env.gamma - float(metrics.get("scoring_gamma", env.gamma))),
                "mpc_delta_mismatch": float(env.delta - float(metrics.get("scoring_delta", env.delta))),
                "selected_action_contract": self.action_contract,
            },
        )
        return action, aux
