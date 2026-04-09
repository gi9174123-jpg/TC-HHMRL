from __future__ import annotations

import argparse

from tchhmrl.meta.meta_trainer import MetaTrainer
from tchhmrl.utils.config import apply_cli_overrides, load_cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/default.yaml")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--n-tasks", type=int, default=5)
    parser.add_argument("--episodes-per-task", type=int, default=2)
    parser.add_argument("--device", type=str, default=None, help="Override config device: auto/cpu/cuda/mps")
    args = parser.parse_args()

    cfg = apply_cli_overrides(load_cfg(args.cfg), device=args.device)
    trainer = MetaTrainer(cfg)
    print(
        "Using device:"
        f" requested={trainer.cfg['experiment']['device_requested']}"
        f" resolved={trainer.cfg['experiment']['device_resolved']}"
    )

    if args.ckpt:
        trainer.agent.load(args.ckpt)

    metrics = trainer.evaluate(n_tasks=args.n_tasks, episodes_per_task=args.episodes_per_task)
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()
