from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from tchhmrl.agents.hierarchical_agent import HierarchicalAgent
from tchhmrl.constraints.dual_layer import DualLayer
from tchhmrl.envs.task_sampler import TaskSampler
from tchhmrl.envs.uw_slipt_env import MultiTxUwSliptEnv
from tchhmrl.utils.config import resolve_device
from tchhmrl.utils.logger import Logger
from tchhmrl.utils.seed import set_seed


@dataclass
class EpisodeStats:
    reward: float
    se: float
    eh: float
    cost: float
    cost_vec: np.ndarray
    violations: float
    length: int
    se_term: float
    eh_term: float
    cost_term: float
    power_term: float
    smooth_term: float


class MetaTrainer:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        seed = int(cfg["experiment"]["seed"])
        set_seed(seed)

        requested_device = str(cfg["experiment"].get("device", "auto"))
        self.device = resolve_device(requested_device)
        self.cfg.setdefault("experiment", {})
        self.cfg["experiment"]["device_requested"] = requested_device
        self.cfg["experiment"]["device_resolved"] = str(self.device)
        self.agent = HierarchicalAgent(cfg, self.device)
        self.base_sampler_cfg = copy.deepcopy(cfg["sampler"])
        self.task_sampler = TaskSampler(copy.deepcopy(self.base_sampler_cfg), seed=seed)

        curriculum_cfg = cfg.get("meta", {}).get("curriculum", {})
        self.curriculum_enabled = bool(curriculum_cfg.get("enabled", False))
        self.curriculum_phases = self._normalize_curriculum_phases(curriculum_cfg.get("phases", []))

        meta_cfg = cfg.get("meta", {})
        self.dual = DualLayer.from_meta_cfg(meta_cfg, n_tx=int(cfg["env"]["n_tx"]))
        self.dual_enabled = bool(meta_cfg.get("dual_enabled", True))
        self.explicit_inner_outer = bool(meta_cfg.get("explicit_inner_outer", True))
        self.outer_step_size = float(meta_cfg.get("outer_step_size", 0.15))
        self.query_updates_enabled = bool(meta_cfg.get("query_updates_enabled", True))
        self.inner_warmup_steps = int(meta_cfg.get("inner_warmup_steps", max(self.agent.batch_size, 64)))
        self.inner_upper_warmup_steps = int(
            meta_cfg.get("inner_upper_warmup_steps", max(self.agent.batch_size // 2, 32))
        )
        if self.explicit_inner_outer:
            self.agent.warmup_steps = self.inner_warmup_steps
            self.agent.upper_warmup_steps = self.inner_upper_warmup_steps

        log_dir = cfg["experiment"]["log_dir"]
        run_name = cfg["experiment"]["run_name"]
        self.logger = Logger(log_dir=log_dir, run_name=run_name)

        self.ckpt_dir = Path(log_dir) / run_name / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalize_curriculum_phases(phases_raw) -> List[Dict]:
        phases: List[Dict] = []
        if not isinstance(phases_raw, list):
            return phases
        for idx, p in enumerate(phases_raw):
            if not isinstance(p, dict):
                continue
            sampler = p.get("sampler", {})
            if not isinstance(sampler, dict):
                continue
            until_frac = float(p.get("until_frac", 1.0))
            until_frac = float(np.clip(until_frac, 0.0, 1.0))
            phases.append(
                {
                    "name": str(p.get("name", f"phase_{idx+1}")),
                    "until_frac": until_frac,
                    "sampler": copy.deepcopy(sampler),
                }
            )
        phases.sort(key=lambda x: x["until_frac"])
        return phases

    def _sampler_cfg_for_iter(self, it: int, meta_iters: int) -> tuple[str, Dict]:
        if not self.curriculum_enabled or not self.curriculum_phases:
            return "base", copy.deepcopy(self.base_sampler_cfg)

        frac = float(it) / float(max(meta_iters, 1))
        picked = self.curriculum_phases[-1]
        for phase in self.curriculum_phases:
            if frac <= float(phase["until_frac"]):
                picked = phase
                break

        merged = copy.deepcopy(self.base_sampler_cfg)
        merged.update(copy.deepcopy(picked["sampler"]))
        return str(picked["name"]), merged

    def _run_episode(self, env: MultiTxUwSliptEnv, train: bool, clear_context: bool = True) -> EpisodeStats:
        obs, _ = env.reset()
        self.agent.reset_rollout_state(clear_context=clear_context)

        ep_reward = 0.0
        ep_se = 0.0
        ep_eh = 0.0
        ep_cost = 0.0
        ep_cost_vec = np.zeros(self.dual.n_constraints, dtype=np.float32)
        ep_viol = 0.0
        ep_len = 0
        ep_se_term = 0.0
        ep_eh_term = 0.0
        ep_cost_term = 0.0
        ep_power_term = 0.0
        ep_smooth_term = 0.0

        macro_start_obs = None
        macro_start_z = None
        macro_upper_idx_raw = 0.0
        macro_upper_idx_exec = 0.0
        macro_reward = 0.0
        macro_steps = 0

        done = False
        while not done:
            z = self.agent.infer_z()
            temps_before = env.temps.copy().astype(np.float32)

            action, aux = self.agent.act(
                obs=obs,
                temps=temps_before,
                amb_temp=env.amb_temp,
                gamma=env.gamma,
                delta=env.delta,
                z=z,
                eval_mode=not train,
            )

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_len += 1

            cost = float(info["cost"])
            cost_vec = np.asarray(info.get("cost_vec", [cost]), dtype=np.float32).reshape(-1)
            dual_penalty = self.dual.penalty(cost_vec) if self.dual_enabled else 0.0
            penalized_reward = float(reward - dual_penalty)

            lower_transition = {
                "obs": obs.astype(np.float32),
                "next_obs": next_obs.astype(np.float32),
                "upper_idx_raw": float(aux["upper_idx_raw"]),
                "upper_idx_exec": float(aux["upper_idx_exec"]),
                "reward": penalized_reward,
                "reward_raw": float(reward),
                "done": float(done),
                "z": z.astype(np.float32),
                "act_exec": aux["act_exec"].astype(np.float32),
                "act_raw": aux["act_raw"].astype(np.float32),
                "boost_combo_exec": float(aux["boost_combo_exec"]),
                "mode_exec": float(aux["mode_exec"]),
                "temps": temps_before.astype(np.float32),
                "next_temps": info["temps"].astype(np.float32),
                "amb_temp": float(info["amb_temp"]),
                "gamma_env": float(info["gamma"]),
                "delta_env": float(info["delta"]),
                "attenuation_c_env": float(env.attenuation_c),
                "misalign_std_env": float(env.misalign_std),
                "qos_min_rate_env": float(env.qos_min_rate),
                "cost": cost,
                "cost_vec": cost_vec.astype(np.float32),
            }

            if bool(aux.get("macro_new", False)) or macro_start_obs is None:
                macro_start_obs = obs.astype(np.float32)
                macro_start_z = z.astype(np.float32)
                macro_upper_idx_raw = float(aux["upper_idx_raw"])
                macro_upper_idx_exec = float(aux["upper_idx_exec"])
                macro_reward = 0.0
                macro_steps = 0

            macro_reward += penalized_reward
            macro_steps += 1
            macro_done = bool(done)
            macro_end = macro_done or (int(aux.get("hold_left", 0)) <= 0)

            if train:
                next_macro_info: Dict[str, float] = {}

                def _next_macro(z_next_val: np.ndarray) -> Dict[str, float]:
                    nonlocal next_macro_info
                    if done:
                        payload = {
                            "upper_idx_raw_next": float(aux["upper_idx_raw"]),
                            "upper_idx_exec_next": float(aux["upper_idx_exec"]),
                            "boost_combo_exec_next": float(aux["boost_combo_exec"]),
                            "mode_exec_next": float(aux["mode_exec"]),
                            "next_exec_map": np.arange(int(self.cfg["agent"]["n_upper_actions"]), dtype=np.float32),
                        }
                    else:
                        nxt = self.agent.preview_next_macro(
                            next_obs=next_obs.astype(np.float32),
                            z_next=z_next_val.astype(np.float32),
                            eval_mode=False,
                            commit_plan=True,
                        )
                        payload = {
                            "upper_idx_raw_next": float(nxt["upper_idx_raw_next"]),
                            "upper_idx_exec_next": float(nxt["upper_idx_exec_next"]),
                            "boost_combo_exec_next": float(nxt["boost_combo_exec_next"]),
                            "mode_exec_next": float(nxt["mode_exec_next"]),
                            "next_exec_map": np.asarray(nxt["next_exec_map"], dtype=np.float32),
                        }
                    next_macro_info = payload
                    return payload

                z_next = self.agent.observe_lower(lower_transition, next_macro_fn=_next_macro)
                if macro_end and macro_start_obs is not None and macro_start_z is not None:
                    upper_transition = {
                        "obs": macro_start_obs,
                        "next_obs": next_obs.astype(np.float32),
                        "upper_idx_raw": macro_upper_idx_raw,
                        "upper_idx_exec": macro_upper_idx_exec,
                        "reward": float(macro_reward),
                        "done": float(macro_done),
                        "z": macro_start_z,
                        "z_next": z_next.astype(np.float32),
                        "horizon": float(macro_steps),
                        "next_exec_map": np.asarray(
                            next_macro_info.get(
                                "next_exec_map",
                                np.arange(int(self.cfg["agent"]["n_upper_actions"]), dtype=np.float32),
                            ),
                            dtype=np.float32,
                        ),
                    }
                    self.agent.observe_upper(upper_transition)
                self.agent.learn()
            else:
                self.agent.episode.add(
                    {
                        "obs": lower_transition["obs"],
                        "upper_idx_exec": lower_transition["upper_idx_exec"],
                        "boost_combo_exec": lower_transition["boost_combo_exec"],
                        "mode_exec": lower_transition["mode_exec"],
                        "act_exec": lower_transition["act_exec"],
                        "reward": lower_transition["reward"],
                        "reward_raw": lower_transition["reward_raw"],
                        "cost": lower_transition["cost"],
                        "cost_vec": lower_transition["cost_vec"],
                        "task_params": np.asarray(
                            [
                                float(lower_transition["attenuation_c_env"]),
                                float(lower_transition["misalign_std_env"]),
                                float(lower_transition["amb_temp"]),
                                float(lower_transition["gamma_env"]),
                                float(lower_transition["delta_env"]),
                                float(lower_transition["qos_min_rate_env"]),
                            ],
                            dtype=np.float32,
                        ),
                    }
                )

            ep_reward += penalized_reward
            ep_se += float(info["se"])
            ep_eh += float(info["eh"])
            ep_cost += cost
            ep_cost_vec += cost_vec
            ep_viol += float(np.any(cost_vec > 0.0))
            ep_se_term += float(info.get("reward_se_term", info["se"]))
            ep_eh_term += float(info.get("reward_eh_term", info["eh"]))
            ep_cost_term += float(info.get("penalty_cost_term", 0.0))
            ep_power_term += float(info.get("penalty_power_term", 0.0))
            ep_smooth_term += float(info.get("penalty_smooth_term", 0.0))

            obs = next_obs

        return EpisodeStats(
            reward=ep_reward,
            se=ep_se / max(ep_len, 1),
            eh=ep_eh / max(ep_len, 1),
            cost=ep_cost / max(ep_len, 1),
            cost_vec=ep_cost_vec / max(ep_len, 1),
            violations=ep_viol / max(ep_len, 1),
            length=ep_len,
            se_term=ep_se_term / max(ep_len, 1),
            eh_term=ep_eh_term / max(ep_len, 1),
            cost_term=ep_cost_term / max(ep_len, 1),
            power_term=ep_power_term / max(ep_len, 1),
            smooth_term=ep_smooth_term / max(ep_len, 1),
        )

    def train(self, meta_iters: int | None = None) -> Path:
        meta_cfg = self.cfg["meta"]
        meta_iters = int(meta_iters or meta_cfg["meta_iters"])

        for it in range(1, meta_iters + 1):
            curriculum_stage, sampler_cfg = self._sampler_cfg_for_iter(it, meta_iters)
            self.task_sampler.cfg = sampler_cfg
            tasks = self.task_sampler.sample(int(meta_cfg["n_tasks_per_iter"]))

            support_stats: List[EpisodeStats] = []
            query_stats: List[EpisodeStats] = []
            adapted_states: List[Dict] = []
            adapted_dual_states: List[Dict] = []
            base_state = self.agent.snapshot_train_state() if self.explicit_inner_outer else None
            base_dual_state = copy.deepcopy(self.dual.state_dict()) if self.explicit_inner_outer else None
            shared_global_step = int(self.agent.global_step)
            shared_upper_steps = int(self.agent.upper.update_steps)
            shared_lower_steps = int(self.agent.lower.update_steps)

            for task in tasks:
                env = MultiTxUwSliptEnv(self.cfg, overrides=task.to_env_overrides())
                if self.explicit_inner_outer and base_state is not None:
                    self.agent.restore_train_state(base_state)
                    self.agent.global_step = shared_global_step
                    self.agent.upper.update_steps = shared_upper_steps
                    self.agent.lower.update_steps = shared_lower_steps
                if self.explicit_inner_outer and base_dual_state is not None:
                    self.dual.load_state_dict(copy.deepcopy(base_dual_state))
                if self.explicit_inner_outer:
                    self.agent.clear_learning_buffers()

                task_support_stats: List[EpisodeStats] = []
                for ep_idx in range(int(meta_cfg["support_episodes"])):
                    ep_stats = self._run_episode(env, train=True, clear_context=(ep_idx == 0))
                    support_stats.append(ep_stats)
                    task_support_stats.append(ep_stats)

                if self.explicit_inner_outer and self.dual_enabled and task_support_stats:
                    task_support_mean_cost_vec = np.mean(
                        np.stack([s.cost_vec for s in task_support_stats], axis=0),
                        axis=0,
                    )
                    self.dual.update(task_support_mean_cost_vec)

                for ep_idx in range(int(meta_cfg["query_episodes"])):
                    query_stats.append(
                        self._run_episode(
                            env,
                            train=bool(self.explicit_inner_outer and self.query_updates_enabled),
                            clear_context=(int(meta_cfg["support_episodes"]) <= 0 and ep_idx == 0),
                        )
                    )

                if self.explicit_inner_outer:
                    adapted_states.append(self.agent.snapshot_train_state())
                    adapted_dual_states.append(copy.deepcopy(self.dual.state_dict()))
                shared_global_step = max(shared_global_step, int(self.agent.global_step))
                shared_upper_steps = max(shared_upper_steps, int(self.agent.upper.update_steps))
                shared_lower_steps = max(shared_lower_steps, int(self.agent.lower.update_steps))

            if self.explicit_inner_outer and base_state is not None:
                self.agent.restore_train_state(base_state)
                self.agent.global_step = shared_global_step
                self.agent.upper.update_steps = shared_upper_steps
                self.agent.lower.update_steps = shared_lower_steps
                self.agent.apply_outer_update(adapted_states, self.outer_step_size)
                if self.dual_enabled and base_dual_state is not None:
                    base_values = np.asarray(base_dual_state["values"], dtype=np.float32)
                    if adapted_dual_states:
                        avg_values = np.mean(
                            np.stack([np.asarray(s["values"], dtype=np.float32) for s in adapted_dual_states], axis=0),
                            axis=0,
                        )
                    else:
                        avg_values = base_values
                    blended_values = base_values + self.outer_step_size * (avg_values - base_values)
                    dual_state = copy.deepcopy(base_dual_state)
                    dual_state["values"] = np.clip(blended_values, 0.0, np.asarray(base_dual_state["max_lambdas"], dtype=np.float32))
                    self.dual.load_state_dict(dual_state)

            mean_support_cost = float(np.mean([s.cost for s in support_stats])) if support_stats else 0.0
            if support_stats:
                mean_support_cost_vec = np.mean(np.stack([s.cost_vec for s in support_stats], axis=0), axis=0)
            else:
                mean_support_cost_vec = np.zeros(self.dual.n_constraints, dtype=np.float32)
            if (not self.dual_enabled):
                lambda_val = 0.0
            elif self.explicit_inner_outer and base_dual_state is not None:
                lambda_val = float(np.mean(self.dual.values))
            else:
                lambda_val = self.dual.update(mean_support_cost_vec)

            row = {
                "iter": float(it),
                "support_reward": float(np.mean([s.reward for s in support_stats])),
                "support_se": float(np.mean([s.se for s in support_stats])),
                "support_eh": float(np.mean([s.eh for s in support_stats])),
                "support_cost": mean_support_cost,
                "support_violation_rate": float(np.mean([s.violations for s in support_stats])),
                "support_se_term": float(np.mean([s.se_term for s in support_stats])),
                "support_eh_term": float(np.mean([s.eh_term for s in support_stats])),
                "support_cost_term": float(np.mean([s.cost_term for s in support_stats])),
                "support_power_term": float(np.mean([s.power_term for s in support_stats])),
                "support_smooth_term": float(np.mean([s.smooth_term for s in support_stats])),
                "query_reward": float(np.mean([s.reward for s in query_stats])) if query_stats else 0.0,
                "query_se": float(np.mean([s.se for s in query_stats])) if query_stats else 0.0,
                "query_eh": float(np.mean([s.eh for s in query_stats])) if query_stats else 0.0,
                "query_cost": float(np.mean([s.cost for s in query_stats])) if query_stats else 0.0,
                "query_violation_rate": float(np.mean([s.violations for s in query_stats]))
                if query_stats
                else 0.0,
                "query_se_term": float(np.mean([s.se_term for s in query_stats])) if query_stats else 0.0,
                "query_eh_term": float(np.mean([s.eh_term for s in query_stats])) if query_stats else 0.0,
                "query_cost_term": float(np.mean([s.cost_term for s in query_stats])) if query_stats else 0.0,
                "query_power_term": float(np.mean([s.power_term for s in query_stats])) if query_stats else 0.0,
                "query_smooth_term": float(np.mean([s.smooth_term for s in query_stats])) if query_stats else 0.0,
                "lambda": lambda_val,
                "curriculum_stage": curriculum_stage,
                "outer_step_size": float(self.outer_step_size if self.explicit_inner_outer else 0.0),
            }
            cost_component_names = list(self.dual.names)
            support_vec_mean = mean_support_cost_vec
            query_vec_mean = (
                np.mean(np.stack([s.cost_vec for s in query_stats], axis=0), axis=0)
                if query_stats
                else np.zeros(self.dual.n_constraints, dtype=np.float32)
            )
            for idx, name in enumerate(cost_component_names):
                row[f"support_cost_{name}"] = float(support_vec_mean[idx])
                row[f"query_cost_{name}"] = float(query_vec_mean[idx])
            if self.dual_enabled:
                row.update(self.dual.as_dict())
            else:
                row.update({f"lambda_{name}": 0.0 for name in self.dual.names})
            self.logger.log(row)

            if it % 10 == 0 or it == meta_iters:
                self.agent.save(self.ckpt_dir / f"iter_{it}.pt")

        return self.logger.csv_path

    def evaluate(self, n_tasks: int = 5, episodes_per_task: int = 2) -> Dict[str, float]:
        tasks = self.task_sampler.sample(n_tasks)
        stats: List[EpisodeStats] = []

        for task in tasks:
            env = MultiTxUwSliptEnv(self.cfg, overrides=task.to_env_overrides())
            for ep_idx in range(episodes_per_task):
                stats.append(self._run_episode(env, train=False, clear_context=(ep_idx == 0)))

        return {
            "reward": float(np.mean([s.reward for s in stats])),
            "se": float(np.mean([s.se for s in stats])),
            "eh": float(np.mean([s.eh for s in stats])),
            "cost": float(np.mean([s.cost for s in stats])),
            "violation_rate": float(np.mean([s.violations for s in stats])),
            "len": float(np.mean([s.length for s in stats])),
        }
