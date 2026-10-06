from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def load_config(name: str) -> dict[str, Any]:
    with (CONFIG_DIR / name).open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"config {name} must be a mapping")
    return data


def _update(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _update(base[key], value)
        else:
            base[key] = value
    return base


_BASELINE_OVERRIDE: dict[str, Any] | None = None


def set_baseline_override(override: dict[str, Any] | None) -> None:
    """Temporarily replace the tuned baseline values (used only by the tuning search)."""
    global _BASELINE_OVERRIDE
    _BASELINE_OVERRIDE = override


def load_baselines() -> dict[str, Any]:
    """baselines.yaml with the TRAIN-tuned overrides from baselines_tuned.yaml (if present)."""
    cfg = load_config("baselines.yaml")
    if _BASELINE_OVERRIDE is not None:
        return _update(cfg, copy.deepcopy(_BASELINE_OVERRIDE))
    if (CONFIG_DIR / "baselines_tuned.yaml").exists():
        cfg = _update(cfg, load_config("baselines_tuned.yaml"))
    return cfg


_ESTIMATOR_OVERRIDE: dict[str, Any] | None = None
_IGNORE_TUNED_ESTIMATORS = False


def set_estimator_override(override: dict[str, Any] | None) -> None:
    """Temporarily replace the calibrated estimator values (used only by calibration)."""
    global _ESTIMATOR_OVERRIDE
    _ESTIMATOR_OVERRIDE = override


def load_estimators() -> dict[str, Any]:
    """estimators.yaml with TRAIN-calibrated overrides from estimators_tuned.yaml (if present)."""
    cfg = load_config("estimators.yaml")
    if (CONFIG_DIR / "estimators_tuned.yaml").exists() and not _IGNORE_TUNED_ESTIMATORS:
        cfg = _update(cfg, load_config("estimators_tuned.yaml"))
    if _ESTIMATOR_OVERRIDE is not None:
        cfg = _update(cfg, copy.deepcopy(_ESTIMATOR_OVERRIDE))
    return cfg


def config_hash(names: list[str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(names):
        digest.update(name.encode("utf-8"))
        digest.update((CONFIG_DIR / name).read_bytes())
    return digest.hexdigest()[:16]
