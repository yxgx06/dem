from __future__ import annotations

import copy
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
import yaml

from egga.config import CONFIG_DIR, REPO_ROOT, load_config, set_baseline_override
from egga.eval.baselines import run_scenario
from egga.eval.metrics2 import run_metrics, tuning_cost
from egga.eval.provenance import provenance
from egga.scenarios.sets import load_scenarios, set_hash

RESULTS_DIR = REPO_ROOT / "results" / "phase2"
SUBSET_STRIDE = 4
START_STEPS = (2.0, 0.5)
SHRUNK_STEPS = (1.41, 0.71)

# controller name -> {parameter label -> (section, key path)}; lqr q/r use the whole table.
TARGETS: dict[str, tuple[str, dict[str, tuple[str, ...]]]] = {
    "pid_ff": ("b0_pid_ff", {"kp": ("kp",), "ki": ("ki",), "kd": ("kd",), "khead": ("khead",)}),
    "pd_ff": ("b1_pd_ff", {"kp": ("kp",), "kd": ("kd",), "khead": ("khead",)}),
    "lqr": ("b7_lqr_true_mass", {"q_e": ("q_e",), "q_psi": ("q_psi",), "r": ("r",)}),
    "mpc": (
        "b6_mpc",
        {"q_lat": ("q_lat",), "q_head": ("q_head",), "r_u": ("r_u",), "r_du": ("r_du",)},
    ),
}


def _apply(section: str, base: dict[str, Any], factors: dict[str, float]) -> dict[str, Any]:
    """Return a baselines override dict for the given multiplicative factors."""
    out: dict[str, Any] = {section: {}}
    if section == "lqr":
        q = np.asarray(base["lqr"]["q_diag"], dtype=float)
        q[:, 0] *= factors["q_e"]
        q[:, 2] *= factors["q_psi"]
        out["lqr"] = {"q_diag": q.tolist(), "r": float(base["lqr"]["r"]) * factors["r"]}
    else:
        out[section] = {k: float(base[section][k]) * f for k, f in factors.items()}
    return out


def _cost_one(args: tuple[str, dict[str, Any], dict[str, Any]]) -> float:
    controller, override, spec = args
    set_baseline_override(override)
    try:
        return tuning_cost(run_metrics(run_scenario(controller, spec), spec["mu_min"]))
    finally:
        set_baseline_override(None)


def _eval(
    pool: ProcessPoolExecutor, controller: str, override: dict[str, Any], specs: list[dict]
) -> float:
    costs = list(pool.map(_cost_one, [(controller, override, s) for s in specs]))
    return float(np.mean(costs))


def tune_target(
    pool: ProcessPoolExecutor,
    key: str,
    base: dict[str, Any],
    specs: list[dict[str, Any]],
    budget: int,
    fmin: float,
    fmax: float,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    controller, labels = TARGETS[key]
    section = "lqr" if key == "lqr" else key
    factors = {k: 1.0 for k in labels}
    cache: dict[tuple[float, ...], float] = {}
    log: list[dict[str, Any]] = []

    def cost(f: dict[str, float]) -> float:
        ident = tuple(round(f[k], 6) for k in labels)
        if ident not in cache:
            cache[ident] = _eval(pool, controller, _apply(section, base, f), specs)
            log.append({"target": key, "eval": len(cache), **dict(zip(labels, ident, strict=True)),
                        "cost": cache[ident]})
        return cache[ident]

    best = cost(factors)
    steps = START_STEPS
    while len(cache) < budget:
        improved = False
        for k in labels:
            for step in steps:
                if len(cache) >= budget:
                    break
                trial = dict(factors)
                trial[k] = float(np.clip(factors[k] * step, fmin, fmax))
                if trial[k] == factors[k]:
                    continue
                c = cost(trial)
                if c < best - 1e-9:
                    best, factors, improved = c, trial, True
        if not improved:
            if steps == SHRUNK_STEPS:
                break
            steps = SHRUNK_STEPS
    return factors, log


def main() -> None:
    cfg = load_config("baselines.yaml")
    tcfg = cfg["tuning"]
    train = load_scenarios("train")  # the ONLY set ever used here
    specs = train[:: SUBSET_STRIDE][: int(tcfg["scenarios_per_eval"])]
    train_hash = set_hash(train)
    workers = max(1, min(8, (os.cpu_count() or 2) - 1))
    tuned: dict[str, Any] = {}
    full_log: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for key in TARGETS:
            factors, log = tune_target(
                pool, key, cfg, specs, int(tcfg["budget_evals"]),
                float(tcfg["factor_min"]), float(tcfg["factor_max"]),
            )
            section = "lqr" if key == "lqr" else key
            override = _apply(section, cfg, factors)
            tuned_section = {"mpc": "mpc"}.get(key, section)
            tuned[tuned_section if key != "lqr" else "lqr"] = override[section]
            full_log.extend(log)
            summary[key] = {
                "factors": factors,
                "evals": len(log),
                "initial_cost": log[0]["cost"],
                "tuned_cost": min(r["cost"] for r in log),
            }
            print(key, summary[key], flush=True)
    meta = provenance()
    header = (
        "# TRAIN-tuned baseline values (generated by `python -m egga.eval.tuning`; do not edit).\n"
        f"# tuned on train set hash {train_hash}, {len(specs)} scenarios per evaluation.\n"
        "# Overrides configs/baselines.yaml. Val and test sets were never used for tuning.\n"
    )
    (CONFIG_DIR / "baselines_tuned.yaml").write_text(
        header + yaml.safe_dump(copy.deepcopy(tuned), sort_keys=True), encoding="utf-8"
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(full_log)
    for k, v in meta.items():
        df[k] = v
    df["train_set_hash"] = train_hash
    df.to_parquet(RESULTS_DIR / "tuning_log.parquet", index=False)


if __name__ == "__main__":
    main()
