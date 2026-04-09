from __future__ import annotations

import csv

from tchhmrl.meta.meta_trainer import MetaTrainer
from tchhmrl.utils.config import load_cfg


def test_meta_trainer_one_iter_explicit_inner_outer_smoke(tmp_path):
    cfg = load_cfg("configs/default.yaml")
    cfg["experiment"]["log_dir"] = str(tmp_path)
    cfg["experiment"]["run_name"] = "meta_smoke"
    cfg["experiment"]["seed"] = 7
    cfg["env"]["episode_len"] = 6
    cfg["agent"]["hidden_dim"] = 32
    cfg["agent"]["batch_size"] = 4
    cfg["agent"]["warmup_steps"] = 4
    cfg["buffer"]["replay_size"] = 128
    cfg["buffer"]["context_max_len"] = 32
    cfg["meta"]["meta_iters"] = 1
    cfg["meta"]["n_tasks_per_iter"] = 1
    cfg["meta"]["support_episodes"] = 1
    cfg["meta"]["query_episodes"] = 1
    cfg["meta"]["inner_warmup_steps"] = 4
    cfg["meta"]["inner_upper_warmup_steps"] = 2
    cfg["upper_dqn"]["epsilon_decay_steps"] = 10
    cfg["context"]["gru_hidden"] = 16

    trainer = MetaTrainer(cfg)
    csv_path = trainer.train(meta_iters=1)

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows
    assert "query_reward" in rows[0]
    assert "lambda_qos" in rows[0]
    assert "query_cost_temp_anchor" in rows[0]


def test_meta_trainer_no_dual_and_no_context_smoke(tmp_path):
    cfg = load_cfg("configs/default.yaml")
    cfg["experiment"]["log_dir"] = str(tmp_path)
    cfg["experiment"]["run_name"] = "ablation_smoke"
    cfg["experiment"]["seed"] = 11
    cfg["env"]["episode_len"] = 6
    cfg["agent"]["hidden_dim"] = 32
    cfg["agent"]["batch_size"] = 4
    cfg["agent"]["warmup_steps"] = 4
    cfg["buffer"]["replay_size"] = 128
    cfg["buffer"]["context_max_len"] = 32
    cfg["meta"]["meta_iters"] = 1
    cfg["meta"]["n_tasks_per_iter"] = 1
    cfg["meta"]["support_episodes"] = 2
    cfg["meta"]["query_episodes"] = 0
    cfg["meta"]["explicit_inner_outer"] = False
    cfg["meta"]["query_updates_enabled"] = False
    cfg["meta"]["dual_enabled"] = False
    cfg["meta"]["dual_lr"] = 0.0
    cfg["meta"]["dual_lrs"] = [0.0, 0.0, 0.0, 0.0]
    cfg["context"]["enabled"] = False
    cfg["upper_dqn"]["epsilon_decay_steps"] = 10
    cfg["context"]["gru_hidden"] = 16

    trainer = MetaTrainer(cfg)
    csv_path = trainer.train(meta_iters=1)

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    assert rows
    assert float(rows[0]["lambda_qos"]) == 0.0
