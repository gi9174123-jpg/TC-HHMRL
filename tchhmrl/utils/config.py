from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import torch
import yaml


def load_cfg(path: str | Path) -> Dict[str, Any]:
    """Load YAML config into a plain dictionary."""
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_cli_overrides(
    cfg: Dict[str, Any],
    *,
    device: str | None = None,
) -> Dict[str, Any]:
    """Return a copied config with supported runtime overrides applied."""
    out = copy.deepcopy(cfg)
    out.setdefault("experiment", {})
    if device is not None:
        out["experiment"]["device"] = str(device)
    return out


def resolve_device(device: str | None) -> torch.device:
    """Resolve a requested device string to an available torch device.

    Supported values:
    - `auto`: prefer CUDA, then MPS, then CPU
    - `cpu`
    - `cuda` / `cuda:0`
    - `mps`
    - `gpu` (alias for `cuda`)
    """
    requested = str(device or "auto").strip().lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "gpu":
        requested = "cuda"

    if requested == "cpu":
        return torch.device("cpu")

    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available. Use --device auto or --device cpu.")
        return torch.device(requested)

    if requested == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise RuntimeError("MPS requested but not available. Use --device auto or --device cpu.")
        return torch.device("mps")

    raise ValueError(f"Unsupported device setting: {device!r}")
