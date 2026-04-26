from __future__ import annotations

import copy

import numpy as np

from scripts.benchmark_constraint_scenarios import apply_scenario, apply_variant, sample_fixed_tasks, validate_training_config
from tchhmrl.agents.hierarchical_agent import HierarchicalAgent
from tchhmrl.envs.task_contract import (
    build_task_summary_v2,
    filter_formally_comparable_records,
    is_formally_comparable_record,
    ordered_task_batch_hash,
    task_batch_hash,
    task_defaults_from_cfg,
)
from tchhmrl.envs.task_sampler import TaskSampler, validate_site_bank
from tchhmrl.envs.uw_slipt_env import MultiTxUwSliptEnv
from tchhmrl.meta.meta_trainer import MetaTrainer
from tchhmrl.utils.config import load_cfg


def _small_meta_cfg(tmp_path):
    cfg = load_cfg("configs/default.yaml")
    cfg["experiment"]["log_dir"] = str(tmp_path)
    cfg["experiment"]["run_name"] = "site_v2_smoke"
    cfg["experiment"]["seed"] = 7
    cfg["env"]["episode_len"] = 4
    cfg["agent"]["hidden_dim"] = 32
    cfg["agent"]["batch_size"] = 4
    cfg["agent"]["warmup_steps"] = 4
    cfg["buffer"]["replay_size"] = 64
    cfg["buffer"]["context_max_len"] = 16
    cfg["meta"]["meta_iters"] = 1
    cfg["meta"]["n_tasks_per_iter"] = 1
    cfg["meta"]["support_episodes"] = 1
    cfg["meta"]["query_episodes"] = 1
    cfg["meta"]["inner_warmup_steps"] = 4
    cfg["meta"]["inner_upper_warmup_steps"] = 2
    cfg["upper_dqn"]["epsilon_decay_steps"] = 10
    cfg["context"]["gru_hidden"] = 16
    return cfg


def test_task_sampler_site_bank_env_reset_metadata():
    cfg = load_cfg("configs/default.yaml")
    issues = validate_site_bank(cfg["sampler"]["site_bank"])
    assert issues == []

    sampler = TaskSampler(copy.deepcopy(cfg["sampler"]), seed=11, task_defaults=task_defaults_from_cfg(cfg))
    task = sampler.sample_task(bucket=1)

    assert task.site_id == 1
    assert len(task.distances) == 3
    assert task.task_source == "site_bank"

    env = MultiTxUwSliptEnv(cfg)
    _, info = env.reset(seed=5, options=task.to_env_overrides())
    task_info = info["task"]
    assert np.allclose(env.distances, np.asarray(task.distances, dtype=np.float32))
    assert task_info["site_id"] == task.site_id
    assert task_info["task_source"] == "site_bank"
    assert task_info["alignment_version"] == "teacher_model_v1"
    assert task_info["task_summary_version"] == "site_v2"
    assert task_info["pre_alignment"] is False
    assert task_info["qos_min_rate"] == task.qos_min_rate

    roundtrip = build_task_summary_v2(task_info)
    assert np.allclose(roundtrip, build_task_summary_v2(task))


def test_hy_energy_conserving_chain_updates_reward_terms():
    cfg = load_cfg("configs/default.yaml")
    env = MultiTxUwSliptEnv(cfg)
    env.reset(seed=3)
    env.channel = np.array([0.45, 0.25, 0.15], dtype=np.float32)
    env.prev_currents = np.zeros(env.n_tx, dtype=np.float32)
    env.prev_rho = 0.5
    env.prev_tau = 0.5
    env.prev_mode = 0

    currents = np.array([0.8, 0.4, 0.2], dtype=np.float32)
    rho = 0.30
    tau = 0.80
    action = {
        "upper_idx": 11,
        "upper_idx_exec": 11,
        "boost_combo_exec": 3,
        "mode_exec": 2,
        "currents_exec": currents,
        "rho_exec": np.array([rho], dtype=np.float32),
        "tau_exec": np.array([tau], dtype=np.float32),
    }

    tx_signal = env._compute_tx_signal(currents)
    se_tx_weight = env._tx_vector(env.se_led_weight, env.se_ld_weight)
    eh_tx_weight = env._tx_vector(env.eh_led_weight, env.eh_ld_weight)
    signal_led = float(np.sum(tx_signal * env.tx_is_led))
    signal_ld = float(np.sum(tx_signal * env.tx_is_ld))
    info_signal = float(np.sum(tx_signal * se_tx_weight))
    eh_input = float(np.sum(tx_signal * eh_tx_weight))

    expected_info_share = tau * (1.0 - rho)
    expected_eh_share = 1.0 - expected_info_share
    expected_eh_metric = float(env.mode_eh_gain[2] * expected_eh_share * eh_input)

    _, _, _, _, info = env.step(action)
    expected_qos_rate = float(env.mode_se_gain[2] * expected_info_share * np.log2(1.0 + info["snr"]))

    assert np.isclose(info["info_share"], expected_info_share, atol=1.0e-6)
    assert np.isclose(info["eh_share"], expected_eh_share, atol=1.0e-6)
    assert np.isclose(info["info_share"] + info["eh_share"], 1.0, atol=1.0e-6)
    assert np.isclose(info["qos_rate"], expected_qos_rate, atol=1.5e-2)
    assert np.isclose(info["eh_metric"], expected_eh_metric, atol=1.5e-2)
    assert np.isclose(info["reward_id_term"], env.se_weight * info["qos_rate"], atol=1.0e-6)
    assert np.isclose(info["reward_eh_term"], env.eh_weight * info["eh_metric"], atol=1.0e-6)


def test_context_site_v2_task_summary_reaches_eval_and_train_paths(tmp_path):
    cfg = _small_meta_cfg(tmp_path)
    trainer = MetaTrainer(cfg)
    task = trainer.task_sampler.sample_task(bucket=0)

    assert trainer.agent.context_task_dim == 9

    env_eval = MultiTxUwSliptEnv(cfg, overrides=task.to_env_overrides())
    trainer._run_episode(env_eval, train=False)
    eval_episode = trainer.agent.episode.as_list()
    assert eval_episode
    assert np.asarray(eval_episode[0]["task_params"]).shape == (9,)
    assert np.allclose(eval_episode[0]["task_params"][-3:], np.asarray(task.distances, dtype=np.float32))

    trainer.agent.replay.clear()
    trainer.agent.upper_replay.clear()
    env_train = MultiTxUwSliptEnv(cfg, overrides=task.to_env_overrides())
    trainer._run_episode(env_train, train=True)
    assert len(trainer.agent.replay) > 0
    tr = trainer.agent.replay._buf[0]
    assert np.asarray(tr["distances_env"]).shape == (3,)
    assert np.allclose(np.asarray(tr["distances_env"]), np.asarray(task.distances, dtype=np.float32))
    extracted = trainer.agent._task_params_from_transition(tr)
    assert extracted.shape == (9,)
    assert np.allclose(extracted[-3:], np.asarray(task.distances, dtype=np.float32))


def test_site_bank_validation_and_fixed_task_sharing_across_variants():
    cfg = load_cfg("configs/default.yaml")
    cfg = copy.deepcopy(cfg)
    apply_scenario(cfg, "moderate_practical")
    checks = validate_training_config(cfg, "moderate_practical")
    assert checks["task_source"] == "site_bank"
    assert checks["site_bank_valid"] is True

    hybrid_cfg = copy.deepcopy(cfg)
    apply_variant(hybrid_cfg, "hybrid")
    single_led_cfg = copy.deepcopy(cfg)
    apply_variant(single_led_cfg, "single_led")

    hybrid_tasks = sample_fixed_tasks(hybrid_cfg, seed=101, n_tasks=3, seed_offset=11_000)
    led_tasks = sample_fixed_tasks(single_led_cfg, seed=101, n_tasks=3, seed_offset=11_000)

    assert [t.site_id for t in hybrid_tasks] == [t.site_id for t in led_tasks]
    assert [tuple(t.distances) for t in hybrid_tasks] == [tuple(t.distances) for t in led_tasks]

    bad_site_bank = copy.deepcopy(cfg["sampler"]["site_bank"])
    bad_site_bank[1]["distances"] = list(bad_site_bank[0]["distances"])
    bad_site_bank[1]["attenuation_c_range"] = list(bad_site_bank[0]["attenuation_c_range"])
    bad_site_bank[1]["misalign_std_range"] = list(bad_site_bank[0]["misalign_std_range"])
    bad_site_bank[1]["amb_temp_range"] = list(bad_site_bank[0]["amb_temp_range"])
    bad_site_bank[1]["gamma_range"] = list(bad_site_bank[0]["gamma_range"])
    bad_site_bank[1]["delta_range"] = list(bad_site_bank[0]["delta_range"])
    bad_site_bank[2] = copy.deepcopy(bad_site_bank[1])
    bad_site_bank[2]["site_id"] = 2
    issues = validate_site_bank(bad_site_bank)
    assert any("identical" in issue for issue in issues)


def test_task_batch_hash_and_formal_consumer_filter():
    cfg = load_cfg("configs/default.yaml")
    cfg = copy.deepcopy(cfg)
    apply_scenario(cfg, "moderate_practical")
    tasks_a = sample_fixed_tasks(cfg, seed=101, n_tasks=4, seed_offset=21_000)
    tasks_b = sample_fixed_tasks(cfg, seed=101, n_tasks=4, seed_offset=21_000)
    tasks_c = sample_fixed_tasks(cfg, seed=101, n_tasks=4, seed_offset=31_000)

    hash_a = task_batch_hash(tasks_a)
    hash_b = task_batch_hash(tasks_b)
    hash_c = task_batch_hash(tasks_c)

    assert hash_a == hash_b
    assert hash_a != hash_c
    assert task_batch_hash(list(reversed(tasks_a))) == hash_a
    assert ordered_task_batch_hash(tasks_a) == ordered_task_batch_hash(tasks_b)
    assert ordered_task_batch_hash(list(reversed(tasks_a))) != ordered_task_batch_hash(tasks_a)

    good = {
        "alignment_version": "teacher_model_v1",
        "task_summary_version": "site_v2",
        "pre_alignment": False,
    }
    old = {"pre_alignment": True}
    assert is_formally_comparable_record(good) is True
    assert is_formally_comparable_record(old) is False
    assert filter_formally_comparable_records([good, old], strict=False) == [good]


def test_strict_site_bank_and_hy_debug_snapshot():
    cfg = load_cfg("configs/default.yaml")
    strict_cfg = copy.deepcopy(cfg["sampler"])
    strict_cfg["strict_site_bank"] = True
    strict_cfg["site_bank"] = []
    try:
        TaskSampler(strict_cfg, seed=0)
    except ValueError as exc:
        assert "strict_site_bank" in str(exc)
    else:
        raise AssertionError("strict_site_bank=True should reject missing site_bank")

    env = MultiTxUwSliptEnv(cfg)
    env.reset(seed=1)
    env.channel = np.array([0.45, 0.25, 0.15], dtype=np.float32)
    snap = env.debug_hy_snapshot(
        currents_exec=np.array([0.8, 0.4, 0.2], dtype=np.float32),
        rho_exec=0.30,
        tau_exec=0.80,
        mode_exec=2,
    )
    assert np.isclose(snap["info_share"] + snap["eh_share"], 1.0, atol=1.0e-6)
    assert snap["reward_id_term"] >= 0.0
    assert snap["reward_eh_term"] >= 0.0
