from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np

from tchhmrl.baselines.common import (
    BasePaperBaseline,
    current_action_from_frac,
    expected_step_metrics,
    monotonic_latency_ms,
)
from tchhmrl.envs.uw_slipt_env import MultiTxUwSliptEnv


class UysalPolicyOptimizer(BasePaperBaseline):
    """Uysal-style underwater SLIPT receiver-policy optimizer.

    The baseline uses a fixed transmitter template and a predefined
    activation-driven rule over TS, PS, and TS-PS/HY receiver policies. It is not
    a best-of-four oracle selector.
    """

    baseline_family = "uysal_policy_optimizer"
    subpolicy_names = ("uysal_ts", "uysal_ps", "uysal_tsps", "uysal_ads")

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        opts = cfg.get("baselines", {}).get("uysal_policy_optimizer", {})
        self.fixed_boost_combo = int(opts.get("fixed_boost_combo", 3))
        self.current_template_name = str(opts.get("fixed_current_template_name", "balanced_receiver_policy"))
        self.current_template = np.asarray(opts.get("fixed_current_template", [0.50, 0.40, 0.40]), dtype=np.float32)
        self.rho_grid = np.asarray(opts.get("rho_grid", [0.10, 0.30, 0.50, 0.70, 0.90]), dtype=np.float32)
        self.tau_grid = np.asarray(opts.get("tau_grid", [0.10, 0.30, 0.50, 0.70, 0.90]), dtype=np.float32)
        self.eh_min_target = float(opts.get("eh_min_target", cfg.get("env", {}).get("eh_min_target", 0.02)))
        self.action_contract = "uysal_ads_threshold_receiver_policy"

    def _candidate(
        self,
        env: MultiTxUwSliptEnv,
        *,
        mode: int,
        rho: float,
        tau: float,
    ) -> Tuple[int, np.ndarray, Dict[str, object], Dict[str, float | np.ndarray]]:
        upper_raw = int(np.clip(self.fixed_boost_combo, 0, 3)) * 3 + int(np.clip(mode, 0, 2))
        lower_raw = current_action_from_frac(
            self.current_template,
            rho=float(rho),
            tau=float(tau),
            action_decode_mode=self.safety.action_decode_mode,
        )
        safe, _ = self._project_raw_action(env, upper_raw, lower_raw, commit=False)
        metrics = expected_step_metrics(env, safe)
        return upper_raw, lower_raw, safe, metrics

    def _deficit_score(self, metrics: Dict[str, float | np.ndarray], env: MultiTxUwSliptEnv) -> float:
        qos_deficit = max(float(env.qos_min_rate) - float(metrics["qos_rate"]), 0.0)
        eh_deficit = max(self.eh_min_target - float(metrics["eh_metric"]), 0.0)
        return float(qos_deficit * qos_deficit + eh_deficit * eh_deficit)

    def _optimize_ts(self, env: MultiTxUwSliptEnv):
        best = None
        for tau in self.tau_grid:
            cand = self._candidate(env, mode=1, rho=0.0, tau=float(tau))
            key = (self._deficit_score(cand[3], env), abs(float(tau) - 0.5))
            if best is None or key < best[0]:
                best = (key, cand)
        return best[1]

    def _optimize_ps(self, env: MultiTxUwSliptEnv):
        best = None
        for rho in self.rho_grid:
            cand = self._candidate(env, mode=0, rho=float(rho), tau=1.0)
            key = (self._deficit_score(cand[3], env), abs(float(rho) - 0.5))
            if best is None or key < best[0]:
                best = (key, cand)
        return best[1]

    def _optimize_tsps(self, env: MultiTxUwSliptEnv):
        best = None
        for rho in self.rho_grid:
            for tau in self.tau_grid:
                cand = self._candidate(env, mode=2, rho=float(rho), tau=float(tau))
                key = (self._deficit_score(cand[3], env), abs(float(rho) - 0.5) + abs(float(tau) - 0.5))
                if best is None or key < best[0]:
                    best = (key, cand)
        return best[1]

    def _ads_select_subpolicy(self, env: MultiTxUwSliptEnv) -> str:
        _, _, _, balanced = self._candidate(env, mode=2, rho=0.5, tau=0.5)
        if float(balanced["qos_rate"]) < float(env.qos_min_rate):
            return "uysal_ts"
        if float(balanced["eh_metric"]) < self.eh_min_target:
            return "uysal_ps"
        return "uysal_tsps"

    def act(self, obs: np.ndarray, env: MultiTxUwSliptEnv, eval_mode: bool = False) -> tuple[Dict, Dict]:
        del obs, eval_mode
        start = time.perf_counter()
        selected_policy = self._ads_select_subpolicy(env)
        if selected_policy == "uysal_ts":
            upper_raw, lower_raw, _, metrics = self._optimize_ts(env)
        elif selected_policy == "uysal_ps":
            upper_raw, lower_raw, _, metrics = self._optimize_ps(env)
        else:
            upper_raw, lower_raw, _, metrics = self._optimize_tsps(env)
        safe, _ = self._project_raw_action(env, upper_raw, lower_raw, commit=True)
        action, aux = self._action_from_safe(
            upper_raw,
            lower_raw,
            safe,
            aux_extra={
                "selected_uysal_policy": "uysal_ads",
                "selected_uysal_controller": "uysal_ads",
                "selected_uysal_subpolicy": selected_policy,
                "uysal_policy_rule": "ads_threshold_scheduler",
                "qos_threshold": float(env.qos_min_rate),
                "eh_threshold": float(self.eh_min_target),
                "eh_threshold_source": "baselines.uysal_policy_optimizer.eh_min_target",
                "fixed_current_template_name": self.current_template_name,
                "selected_action_contract": self.action_contract,
                "online_latency_ms": monotonic_latency_ms(start),
                "predicted_qos_rate": float(metrics["qos_rate"]),
                "predicted_eh_metric": float(metrics["eh_metric"]),
            },
        )
        return action, aux
