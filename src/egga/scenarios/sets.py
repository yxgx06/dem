from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from egga.config import REPO_ROOT, load_config

SET_DIR = REPO_ROOT / "experiments" / "scenario_sets"
GENERATOR_VERSION = 1
_SLICED = (
    "speed_mps",
    "demand_ratio",
    "mu_low",
    "mu_high",
    "mu_level",
    "mass_scale",
    "delay_s",
    "noise_std_m",
    "mu_belief_error",
    "sine_period_s",
    "step_time_s",
    "patch_start_m",
    "patch_len_m",
)


def _draw(rng: np.random.Generator, lo: float, hi: float, slices: int, allowed: list[int]) -> tuple:
    index = int(rng.choice(allowed))
    value = lo + (hi - lo) * (index + float(rng.random())) / slices
    return round(float(value), 6), index


def _geometry(
    speed: float, mu_min: float, ratio: float, period: float, cfg: dict[str, Any], g: float,
    wheelbase: float,
) -> dict[str, float]:
    geo = cfg["geometry"]
    tan_delta = ratio * mu_min * g * wheelbase / speed**2
    lc_amp = min(float(np.arctan(tan_delta)), float(geo["max_lc_amp_rad"]))
    half = period / 2.0
    k = max(2, int(round(float(geo["lc_target_start_s"]) / half)))
    return {
        "sine_amp_rad": round(float(geo["sine_to_lc_ratio"]) * lc_amp, 6),
        "sine_period_s": round(period, 6),
        "lc_start_s": round(k * half, 6),
        "lc_amp_rad": round(lc_amp, 6),
        "lc_period_s": 4.0,
    }


def _friction(
    family: str, p: dict[str, float], cfg: dict[str, Any]
) -> tuple[dict[str, Any], float]:
    ramp = float(cfg["geometry"]["ramp_duration_s"])
    if family == "constant":
        return {"type": "constant", "mu": p["mu_level"]}, p["mu_level"]
    lo, hi = p["mu_low"], p["mu_high"]
    if family == "step":
        prof = {"type": "step", "t_s": p["step_time_s"], "before": hi, "after": lo}
    elif family == "ramp":
        t0 = p["step_time_s"]
        prof = {"type": "ramp", "t0_s": t0, "t1_s": round(t0 + ramp, 6), "before": hi, "after": lo}
    else:
        s0 = p["patch_start_m"]
        prof = {
            "type": "patches",
            "default": hi,
            "patches": [{"s0_m": s0, "s1_m": round(s0 + p["patch_len_m"], 6), "mu": lo}],
        }
    return prof, lo


def generate_set(name: str) -> list[dict[str, Any]]:
    cfg = load_config("scenario_sets.yaml")
    spec = cfg["sets"][name]
    vehicle = load_config("vehicle.yaml")
    g = float(vehicle["gravity_mps2"])
    wheelbase = float(vehicle["lf_m"]) + float(vehicle["lr_m"])
    slices = int(cfg["slices"])
    allowed = [int(i) for i in spec["slice_indices"]]
    out: list[dict[str, Any]] = []
    for i in range(int(spec["n"])):
        rng = np.random.default_rng([int(spec["base_seed"]), i])
        values: dict[str, float] = {}
        used: dict[str, int] = {}
        for key in _SLICED:
            lo, hi = (float(v) for v in cfg["ranges"][key])
            values[key], used[key] = _draw(rng, lo, hi, slices, allowed)
        family = str(rng.choice(cfg["families"]))
        profile, mu_min = _friction(family, values, cfg)
        geometry = _geometry(
            values["speed_mps"], mu_min, values["demand_ratio"], values["sine_period_s"], cfg, g,
            wheelbase,
        )
        out.append(
            {
                "id": f"{name}-{i:03d}",
                "set": name,
                "seed": int(spec["base_seed"]) + i,
                "duration_s": float(cfg["duration_s"]),
                "speed_mps": values["speed_mps"],
                "mass_scale": values["mass_scale"],
                "delay_s": values["delay_s"],
                "noise_std_m": values["noise_std_m"],
                "mu_belief_error": values["mu_belief_error"],
                "demand_ratio": values["demand_ratio"],
                "mu_min": round(mu_min, 6),
                "friction_family": family,
                "friction": profile,
                "geometry": geometry,
                "slice_indices": used,
            }
        )
    return out


def set_hash(scenarios: list[dict[str, Any]]) -> str:
    blob = json.dumps(scenarios, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def freeze() -> dict[str, str]:
    """Generate all sets, write them under experiments/scenario_sets/, and return the hashes."""
    SET_DIR.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for name in load_config("scenario_sets.yaml")["sets"]:
        scenarios = generate_set(name)
        hashes[name] = set_hash(scenarios)
        payload = {
            "name": name,
            "hash": hashes[name],
            "generator_version": GENERATOR_VERSION,
            "scenarios": scenarios,
        }
        (SET_DIR / f"{name}.json").write_text(
            json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8"
        )
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "hashes": hashes,
        "note": "test set is locked until Phase 7; load it only with unlock_test=True",
    }
    (SET_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return hashes


def load_scenarios(name: str, unlock_test: bool = False) -> list[dict[str, Any]]:
    """Load a frozen set, verifying its hash. The test set requires unlock_test=True (Phase 7)."""
    if name == "test" and not unlock_test:
        raise PermissionError("the test scenario set is locked until the Phase 7 evaluation")
    manifest = json.loads((SET_DIR / "manifest.json").read_text(encoding="utf-8"))
    payload = json.loads((SET_DIR / f"{name}.json").read_text(encoding="utf-8"))
    scenarios: list[dict[str, Any]] = payload["scenarios"]
    frozen = manifest["hashes"][name]
    if set_hash(scenarios) != frozen or payload["hash"] != frozen:
        raise ValueError(f"scenario set {name} does not match its frozen hash")
    return scenarios


def run_arguments(spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], float, int]:
    """Plant overrides, mission config, friction belief error and seed for one scenario."""
    mission = load_config("mission.yaml")
    mission["t_final_s"] = spec["duration_s"]
    mission["vx_mps"] = spec["speed_mps"]
    mission["stage_bounds_s"] = [0.0, 1e6, 2e6, 3e6, 4e6, 5e6]
    geo = spec["geometry"]
    mission["reference"] = {
        "sine_amp_rad": geo["sine_amp_rad"],
        "sine_period_s": geo["sine_period_s"],
        "lc_start_s": geo["lc_start_s"],
        "lc_amp_rad": geo["lc_amp_rad"],
        "lc_period_s": geo["lc_period_s"],
        "lc_segments_s": [2.5, 6.5, 10.5],
    }
    plant = {
        "speed": {"constant_mps": spec["speed_mps"]},
        "mass": {"mass_scale": spec["mass_scale"]},
        "actuator": {"delay_s": spec["delay_s"]},
        "sensors": {"lateral_error": {"noise_std": spec["noise_std_m"]}},
        "friction": {"profile": spec["friction"]},
    }
    return plant, mission, float(spec["mu_belief_error"]), int(spec["seed"])
