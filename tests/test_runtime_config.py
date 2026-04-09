from __future__ import annotations

import copy

import pytest
import torch

from tchhmrl.utils.config import apply_cli_overrides, load_cfg, resolve_device


def test_apply_cli_overrides_keeps_original_config_untouched():
    cfg = load_cfg("configs/default.yaml")
    base_device = cfg["experiment"]["device"]

    overridden = apply_cli_overrides(cfg, device="cpu")

    assert cfg["experiment"]["device"] == base_device
    assert overridden["experiment"]["device"] == "cpu"


def test_moderate_config_is_less_stressful_than_default():
    hard_cfg = copy.deepcopy(load_cfg("configs/default.yaml"))
    moderate_cfg = load_cfg("configs/moderate.yaml")

    assert float(moderate_cfg["env"]["attenuation_c"]) < float(hard_cfg["env"]["attenuation_c"])
    assert float(moderate_cfg["env"]["misalign_std"]) < float(hard_cfg["env"]["misalign_std"])
    assert float(moderate_cfg["env"]["thermal_safe"]) > float(hard_cfg["env"]["thermal_safe"])
    assert float(moderate_cfg["safety"]["thermal_safe"]) > float(hard_cfg["safety"]["thermal_safe"])
    assert float(moderate_cfg["env"]["cost_weight"]) < float(hard_cfg["env"]["cost_weight"])
    assert float(moderate_cfg["sampler"]["gamma_range"][1]) < float(hard_cfg["sampler"]["gamma_range"][1])


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None:
        monkeypatch.setattr(mps_backend, "is_available", lambda: False)

    assert resolve_device("auto").type == "cpu"


def test_resolve_device_raises_when_cuda_forced_but_missing(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError):
        resolve_device("cuda")
