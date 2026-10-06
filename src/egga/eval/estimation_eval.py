from __future__ import annotations

import copy
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from egga.config import (
    CONFIG_DIR,
    REPO_ROOT,
    load_config,
    load_estimators,
    set_estimator_override,
)
from egga.estimation.suite import EstimatorSuite
from egga.eval.closed_loop import (
    ClosedLoopResult,
    deep_merge,
    load_plant_config,
    run_closed_loop,
)
from egga.eval.provenance import provenance
from egga.scenarios.sets import load_scenarios, run_arguments, set_hash

RESULTS_DIR = REPO_ROOT / "results" / "phase3"
DOCS = REPO_ROOT / "docs"
WARMUP_S = 5.0
CONVERGENCE_WIDTH = 0.25
CONVERGENCE_HOLD_S = 2.0
TAU_CONVERGED = 0.12
COVERAGE_RUN_THRESHOLD = 0.95
EVAL_CONTROLLER = "b0_pid_ff"
DT = 0.01
MU_GRID = [(z, q) for z in (2.0, 2.5, 3.0, 3.5) for q in (0.03, 0.05, 0.1, 0.2)]
MASS_GRID = [(z, q) for z in (2.0, 3.0, 4.0, 6.0) for q in (0.002, 0.005, 0.01, 0.02)]


def run_estimation(
    spec: dict[str, Any],
    controller: str = EVAL_CONTROLLER,
    plant_extra: dict[str, Any] | None = None,
) -> ClosedLoopResult:
    overrides, mission_cfg, belief_error, seed = run_arguments(spec)
    cfg = load_plant_config(deep_merge(overrides, plant_extra or {}))
    suite = EstimatorSuite(load_config("vehicle.yaml"), load_estimators(), DT)
    return run_closed_loop(
        controller,
        cfg,
        seed=seed,
        mu_belief_error=belief_error,
        mission_cfg=mission_cfg,
        estimator=suite,
    )


def _first_sustained(mask: np.ndarray, t: np.ndarray, hold: float) -> float:
    run = int(round(hold / DT))
    count = 0
    for i, ok in enumerate(mask):
        count = count + 1 if ok else 0
        if count >= run:
            return float(t[i - run + 1])
    return float("nan")


def estimation_metrics(run: ClosedLoopResult) -> dict[str, Any]:
    e = run.estimates
    t = run.t
    valid = np.isfinite(run.mu) & np.isfinite(e["mu_lo"]) & (t >= WARMUP_S)
    mu_true = run.mu
    lo, hi = e["mu_lo"], e["mu_hi"]
    hat = 0.5 * (lo + hi)
    inside = (lo <= mu_true) & (mu_true <= hi)
    width = hi - lo
    tau_true = run.truth["latency_s"]
    tau_cov = e["tau_bar"] >= tau_true
    mass_true = run.truth["mass_scale"]
    mass_in = (e["mass_lo"] <= mass_true) & (mass_true <= e["mass_hi"])
    converged = np.isfinite(lo) & inside & (width <= CONVERGENCE_WIDTH)
    tau_conv = np.isfinite(e["tau_bar"]) & (e["tau_bar"] <= TAU_CONVERGED) & tau_cov
    n = int(valid.sum())

    def frac(mask: np.ndarray) -> float:
        return float(mask[valid].mean()) if n else float("nan")

    return {
        "mu_coverage": frac(inside),
        "mu_width": float(width[valid].mean()) if n else float("nan"),
        "mu_bias": float((hat - mu_true)[valid].mean()) if n else float("nan"),
        "mu_rmse": float(np.sqrt(np.mean((hat - mu_true)[valid] ** 2))) if n else float("nan"),
        "mu_converge_s": _first_sustained(converged, t, CONVERGENCE_HOLD_S),
        "tau_coverage": frac(tau_cov),
        "tau_excess": float((e["tau_bar"] - tau_true)[valid].mean()) if n else float("nan"),
        "tau_converge_s": _first_sustained(tau_conv, t, CONVERGENCE_HOLD_S),
        "mass_coverage": frac(mass_in),
        "mass_width": float((e["mass_hi"] - e["mass_lo"])[valid].mean()) if n else float("nan"),
        "ok_fraction": frac(e["ok"] > 0.5),
        "diverged": run.diverged_at_s is not None,
    }


def evaluate_set(set_name: str, limit: int | None = None) -> pd.DataFrame:
    specs = load_scenarios(set_name)
    if limit is not None:
        specs = specs[:limit]
    rows = []
    for spec in specs:
        run = run_estimation(spec)
        rows.append(
            {
                "set": set_name,
                "scenario": spec["id"],
                "family": spec["friction_family"],
                "demand_ratio": spec["demand_ratio"],
                "mu_min": spec["mu_min"],
                "mass_scale": spec["mass_scale"],
                "delay_s": spec["delay_s"],
                **estimation_metrics(run),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, reps: int = 2000, seed: int = 0) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, values.size).mean() for _ in range(reps)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def clopper_pearson(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    lo = 0.0 if successes == 0 else stats.beta.ppf(alpha / 2, successes, n - successes + 1)
    hi = 1.0 if successes == n else stats.beta.ppf(1 - alpha / 2, successes + 1, n - successes)
    return float(lo), float(hi)


# ---------------------------------------------------------------- calibration (TRAIN only)
def _calibration_job(args: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    spec, override = args
    set_estimator_override(override)
    try:
        return estimation_metrics(run_estimation(spec))
    finally:
        set_estimator_override(None)


def _score(rows: list[dict[str, Any]], cov_key: str, width_key: str) -> dict[str, float]:
    cov = np.array([r[cov_key] for r in rows])
    return {
        "mean_coverage": float(cov.mean()),
        "run_coverage": float((cov >= COVERAGE_RUN_THRESHOLD).mean()),
        "width": float(np.mean([r[width_key] for r in rows])),
    }


def _select(points: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [p for p in points if p["run_coverage"] >= COVERAGE_RUN_THRESHOLD]
    if ok:
        return min(ok, key=lambda p: p["width"])
    return max(points, key=lambda p: (p["run_coverage"], p["mean_coverage"], -p["width"]))


def calibrate() -> None:
    train = load_scenarios("train")  # TRAIN only
    workers = max(1, min(8, (os.cpu_count() or 2) - 1))
    log: list[dict[str, Any]] = []
    chosen: dict[str, dict[str, Any]] = {"friction": {}}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        mass_points = []
        for z_mass, q_theta in MASS_GRID:
            override = {"friction": {"z_mass": z_mass, "q_theta_per_sqrt_s": q_theta}}
            rows = list(pool.map(_calibration_job, [(s, override) for s in train]))
            point = {
                "z_mass": z_mass,
                "q_theta": q_theta,
                **_score(rows, "mass_coverage", "mass_width"),
            }
            mass_points.append(point)
            log.append({"stage": "mass", **point})
            print("mass", point, flush=True)
        best_mass = _select(mass_points)
        chosen["friction"].update(
            {"z_mass": best_mass["z_mass"], "q_theta_per_sqrt_s": best_mass["q_theta"]}
        )
        mu_points = []
        for z, q_mu in MU_GRID:
            override = {"friction": {**chosen["friction"], "z": z, "q_mu_per_sqrt_s": q_mu}}
            rows = list(pool.map(_calibration_job, [(s, override) for s in train]))
            point = {"z": z, "q_mu": q_mu, **_score(rows, "mu_coverage", "mu_width")}
            mu_points.append(point)
            log.append({"stage": "mu", **point})
            print("mu", point, flush=True)
        best_mu = _select(mu_points)
        chosen["friction"].update({"z": best_mu["z"], "q_mu_per_sqrt_s": best_mu["q_mu"]})
    header = (
        "# TRAIN-calibrated estimator values (generated by `python -m egga.eval.estimation_eval "
        "--calibrate`; do not edit).\n"
        f"# calibrated on train set hash {set_hash(train)}. Val and test were never used.\n"
    )
    (CONFIG_DIR / "estimators_tuned.yaml").write_text(
        header + yaml.safe_dump(copy.deepcopy(chosen), sort_keys=True), encoding="utf-8"
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(log)
    for k, v in provenance().items():
        df[k] = v
    df["train_set_hash"] = set_hash(train)
    df.to_parquet(RESULTS_DIR / "calibration_log.parquet", index=False)


# ---------------------------------------------------------------- condition tests
def _base_spec() -> dict[str, Any]:
    spec = copy.deepcopy(load_scenarios("val")[0])
    spec.update(
        {
            "speed_mps": 10.0,
            "mass_scale": 1.2,
            "delay_s": 0.03,
            "noise_std_m": 0.0,
            "mu_belief_error": 0.0,
            "friction": {"type": "constant", "mu": 0.6},
        }
    )
    spec["geometry"] = {
        "sine_amp_rad": 0.01,
        "sine_period_s": 10.0,
        "lc_start_s": 10.0,
        "lc_amp_rad": 0.05,
        "lc_period_s": 4.0,
    }
    return spec


def _straight(spec: dict[str, Any]) -> None:
    spec["geometry"].update({"sine_amp_rad": 0.0, "lc_amp_rad": 0.0})


def _step(spec: dict[str, Any], t_s: float, lc_start: float, before: float, after: float) -> None:
    spec["friction"] = {"type": "step", "t_s": t_s, "before": before, "after": after}
    spec["mu_min"] = min(before, after)
    spec["geometry"].update({"lc_start_s": lc_start, "sine_amp_rad": 0.0, "lc_amp_rad": 0.09})


def condition_specs() -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    """Named estimator-condition scenarios: (spec, extra plant overrides)."""
    out: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}

    def add(name: str, edit: Any = None, extra: dict[str, Any] | None = None) -> None:
        spec = _base_spec()
        if edit:
            edit(spec)
        out[name] = (spec, extra or {})

    add("nominal_excited")
    add("low_excitation_straight", _straight)
    add("drop_0.85_to_0.30_while_cornering", lambda s: _step(s, 12.0, 10.0, 0.85, 0.30))
    add("drop_0.85_to_0.30_while_straight", lambda s: _step(s, 6.0, 18.0, 0.85, 0.30))
    add("rise_0.30_to_0.85_while_cornering", lambda s: _step(s, 12.0, 10.0, 0.30, 0.85))
    for delay in (0.0, 0.06, 0.10):
        add(f"delay_{int(delay * 1000)}ms", lambda s, d=delay: s.update({"delay_s": d}))
    for mass in (1.0, 1.9):
        add(f"mass_x{mass}", lambda s, m=mass: s.update({"mass_scale": m}))
    add(
        "sensor_noise_realistic",
        extra={
            "sensors": {
                "yaw_rate": {"noise_std": 0.005},
                "lateral_accel": {"noise_std": 0.1},
                "steering_angle": {"noise_std": 0.002},
            }
        },
    )
    add(
        "sensor_dropout_5pct",
        extra={
            "sensors": {
                "yaw_rate": {"dropout_prob": 0.05},
                "lateral_accel": {"dropout_prob": 0.05},
                "steering_angle": {"dropout_prob": 0.05},
            }
        },
    )
    add("actuator_lag_40ms", extra={"actuator": {"lag_tau_s": 0.04}})
    return out


def detection_latency(run: ClosedLoopResult, t_event: float, old: float, new: float) -> float:
    """Seconds after t_event until the interval covers `new` and excludes `old` (NaN if never)."""
    e = run.estimates
    excludes_old = (e["mu_hi"] < old) if new < old else (e["mu_lo"] > old)
    mask = (run.t >= t_event) & (e["mu_lo"] <= new) & (new <= e["mu_hi"]) & excludes_old
    idx = np.flatnonzero(mask)
    return float(run.t[idx[0]] - t_event) if idx.size else float("nan")


def conditions_table() -> pd.DataFrame:
    rows = []
    for name, (spec, extra) in condition_specs().items():
        run = run_estimation(spec, plant_extra=extra)
        step = spec["friction"]
        latency = float("nan")
        if step["type"] == "step":
            latency = detection_latency(run, step["t_s"], step["before"], step["after"])
        rows.append({"condition": name, "detect_latency_s": latency, **estimation_metrics(run)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- report
def _ci_text(values: np.ndarray) -> str:
    lo, hi = bootstrap_ci(values)
    return f"{np.nanmean(values):.3f} [{lo:.3f}, {hi:.3f}]"


def report() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    meta = provenance()
    val = evaluate_set("val")
    cond = conditions_table()
    for name, df in (("val_runs", val), ("conditions", cond)):
        for k, v in meta.items():
            df[k] = v
        df.to_parquet(RESULTS_DIR / f"{name}.parquet", index=False)

    n = len(val)
    lines = [
        "# Phase 3 estimators",
        "",
        f"Status: MEASURED. Generated by `python -m egga.eval.estimation_eval` (git "
        f"{meta['git_sha'][:8]}, config hash {meta['config_hash']}, dirty={meta['git_dirty']}). "
        "Estimators see only yaw rate, lateral acceleration, steering (measured and commanded) and "
        "speed; calibrated on the TRAIN set (`configs/estimators_tuned.yaml`); evaluated on the "
        "VAL set under closed loop with the B0 controller (oracle mu belief).",
        "",
        f"## Val set ({n} scenarios, warm-up {WARMUP_S:g} s excluded)",
        "",
        "Values are means over scenarios with 95% bootstrap CIs.",
        "",
        "| quantity | mean [95% CI] |",
        "|---|---|",
    ]
    for label, col in (
        ("friction interval coverage (time fraction)", "mu_coverage"),
        ("friction interval width", "mu_width"),
        ("friction bias (mu_hat - mu)", "mu_bias"),
        ("friction RMSE", "mu_rmse"),
        ("delay bound coverage (tau_bar >= latency)", "tau_coverage"),
        ("delay bound excess over true latency (s)", "tau_excess"),
        ("mass interval coverage", "mass_coverage"),
        ("mass interval width (scale)", "mass_width"),
        ("fraction of time status ok", "ok_fraction"),
    ):
        lines.append(f"| {label} | {_ci_text(val[col].to_numpy())} |")
    for label, col in (
        ("friction converged (width <= 0.25 and covers, 2 s)", "mu_converge_s"),
        ("delay bound converged (<= 0.12 s and covers, 2 s)", "tau_converge_s"),
    ):
        got = val[col].notna()
        median = val.loc[got, col].median() if got.any() else float("nan")
        lines.append(
            f"| {label}: runs reaching it | {int(got.sum())}/{n}, median time {median:.1f} s |"
        )
    lines += [
        "",
        "### Run-level coverage (runs whose interval covers truth >= 95% of the time)",
        "",
        "| quantity | runs | rate [Clopper-Pearson 95%] | target |",
        "|---|---|---|---|",
    ]
    for label, col in (
        ("friction", "mu_coverage"),
        ("delay bound", "tau_coverage"),
        ("mass", "mass_coverage"),
    ):
        k = int((val[col] >= COVERAGE_RUN_THRESHOLD).sum())
        lo, hi = clopper_pearson(k, n)
        lines.append(f"| {label} | {k}/{n} | {k / n:.3f} [{lo:.3f}, {hi:.3f}] | >= 0.95 |")
    lines += [
        "",
        "## Conditions",
        "",
        "Dedicated scenarios (not part of the frozen sets). `latency` is the time after a "
        "friction step until the interval covers the new value and excludes the old.",
        "",
        "| condition | mu cov | mu width | mu bias | tau cov | mass cov | latency (s) "
        "| ok frac | diverged |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in cond.itertuples():
        lines.append(
            f"| {r.condition} | {r.mu_coverage:.2f} | {r.mu_width:.2f} | {r.mu_bias:+.2f} | "
            f"{r.tau_coverage:.2f} | {r.mass_coverage:.2f} | {r.detect_latency_s:.1f} | "
            f"{r.ok_fraction:.2f} | {r.diverged} |"
        )
    failing = cond[
        (cond["mu_coverage"] < COVERAGE_RUN_THRESHOLD)
        | (cond["tau_coverage"] < COVERAGE_RUN_THRESHOLD)
        | (cond["mass_coverage"] < COVERAGE_RUN_THRESHOLD)
    ]
    lines += ["", "## Conditions where coverage fails (below 0.95)", ""]
    if failing.empty:
        lines.append("None in the condition list above.")
    for r in failing.itertuples():
        bits = []
        if r.mu_coverage < COVERAGE_RUN_THRESHOLD:
            bits.append(f"friction {r.mu_coverage:.2f}")
        if r.tau_coverage < COVERAGE_RUN_THRESHOLD:
            bits.append(f"delay {r.tau_coverage:.2f}")
        if r.mass_coverage < COVERAGE_RUN_THRESHOLD:
            bits.append(f"mass {r.mass_coverage:.2f}")
        lines.append(f"- {r.condition}: " + ", ".join(bits))
    low = val[val["mu_coverage"] < COVERAGE_RUN_THRESHOLD]
    names = " (" + ", ".join(low["scenario"]) + ")" if len(low) else ""
    lines += ["", f"Val scenarios with friction coverage < 0.95: {len(low)} of {n}{names}."]
    (DOCS / "phase3_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote docs/phase3_report.md")


if __name__ == "__main__":
    if "--calibrate" in sys.argv:
        calibrate()
    else:
        report()
