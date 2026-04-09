from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable


class Logger:
    """Simple CSV logger for training and evaluation metrics."""

    def __init__(self, log_dir: str, run_name: str):
        self.base = Path(log_dir) / run_name
        self.base.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.base / "metrics.csv"
        self._fieldnames: list[str] | None = None

    def log(self, metrics: Dict[str, float]) -> None:
        write_header = not self.csv_path.exists()
        if self._fieldnames is None:
            self._fieldnames = list(metrics.keys())
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(metrics)

    def read_rows(self) -> Iterable[Dict[str, str]]:
        if not self.csv_path.exists():
            return []
        with self.csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)
