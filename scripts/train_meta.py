from __future__ import annotations

import argparse

from tchhmrl.meta.meta_trainer import MetaTrainer
from tchhmrl.utils.config import apply_cli_overrides, load_cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/default.yaml")
    parser.add_argument("--meta-iters", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="Override config device: auto/cpu/cuda/mps")
    args = parser.parse_args()

    cfg = apply_cli_overrides(load_cfg(args.cfg), device=args.device)
    trainer = MetaTrainer(cfg)
    print(
        "Using device:"
        f" requested={trainer.cfg['experiment']['device_requested']}"
        f" resolved={trainer.cfg['experiment']['device_resolved']}"
    )
    csv_path = trainer.train(meta_iters=args.meta_iters)
    print(f"Training finished. Metrics saved to: {csv_path}")


if __name__ == "__main__":
    main()
