from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_csv(csv_path: str, out_dir: str | None = None) -> Path:
    df = pd.read_csv(csv_path)
    out_base = Path(out_dir) if out_dir else Path(csv_path).parent
    out_base.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes.flatten()

    ax[0].plot(df["iter"], df["support_reward"], label="support")
    ax[0].plot(df["iter"], df["query_reward"], label="query")
    ax[0].set_title("Reward")
    ax[0].legend()

    ax[1].plot(df["iter"], df["support_se"], label="SE")
    ax[1].plot(df["iter"], df["support_eh"], label="EH")
    ax[1].set_title("Support SE/EH")
    ax[1].legend()

    ax[2].plot(df["iter"], df["support_cost"], label="support cost")
    ax[2].plot(df["iter"], df["query_cost"], label="query cost")
    ax[2].set_title("Cost")
    ax[2].legend()

    ax[3].plot(df["iter"], df["support_violation_rate"], label="support")
    ax[3].plot(df["iter"], df["query_violation_rate"], label="query")
    ax[3].plot(df["iter"], df["lambda"], label="lambda")
    ax[3].set_title("Constraint & Lambda")
    ax[3].legend()

    for a in ax:
        a.grid(True, alpha=0.3)
        a.set_xlabel("Iteration")

    fig.tight_layout()
    out_path = out_base / "training_curves.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out = plot_csv(args.csv, args.out_dir)
    print(f"Saved plot to: {out}")


if __name__ == "__main__":
    main()
