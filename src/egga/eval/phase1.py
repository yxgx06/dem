from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from egga.config import REPO_ROOT, load_config
from egga.eval.closed_loop import deep_merge, load_case, load_plant_config, run_closed_loop
from egga.eval.provenance import provenance
from egga.eval.simulate import Scenario, run_mission
from egga.plant.vehicle import PlantParams, Vehicle

RESULTS_DIR = REPO_ROOT / "results" / "phase1"
DOCS = REPO_ROOT / "docs"
LIMIT_MUS = (0.2, 0.3, 0.5, 0.7, 0.85, 1.0)
LIMIT_SPEEDS = (10.0, 20.0)
LIMIT_DELTAS = np.linspace(0.005, 0.5, 100)
CASE_CONTROLLERS = ("pid", "pd_ff", "rl_handtyped")
SWEEP_CUTOFFS: tuple[float | None, ...] = (None, 10.0, 5.0, 2.0, 1.0, 0.5)
SWEEP_SEEDS = tuple(range(5))
DT = 0.01


def max_steady_lateral_acceleration(
    mu: float, vx: float, tyre_model: str, settle_s: float = 15.0
) -> dict[str, Any]:
    """Largest steady-turn lateral acceleration reached over a sweep of constant steer angles."""
    cfg = load_plant_config({"tyre": {"model": tyre_model}})
    params = PlantParams.from_configs(load_config("vehicle.yaml"), cfg)
    best = 0.0
    for delta in LIMIT_DELTAS:
        veh = Vehicle(params)
        tail: list[float] = []
        steps = int(settle_s / DT)
        for k in range(steps):
            out = veh.step(float(delta), mu, mu, 0.0, DT, vx, 0.0, k * DT)
            if k >= steps - 100:
                tail.append(out.ay)
        ay = float(np.mean(tail))
        if np.isfinite(ay):
            best = max(best, abs(ay))
    g = params.gravity
    return {
        "mu": mu,
        "vx_mps": vx,
        "tyre": tyre_model,
        "max_ay_mps2": best,
        "mu_g_mps2": mu * g,
        "fraction_of_mu_g": best / (mu * g),
    }


def plant_limits() -> pd.DataFrame:
    rows = [
        max_steady_lateral_acceleration(mu, vx, model)
        for model in ("pacejka", "linear_clip")
        for vx in LIMIT_SPEEDS
        for mu in LIMIT_MUS
    ]
    return pd.DataFrame(rows)


def _row(case: str, ctrl: str, run: Any) -> dict[str, Any]:
    finite = run.ey[np.isfinite(run.ey)]
    return {
        "case": case,
        "controller": ctrl,
        "max_abs_ey_cm": float(np.max(np.abs(finite)) * 100.0),
        "diverged": run.diverged_at_s is not None,
        "diverged_at_s": float(run.diverged_at_s) if run.diverged_at_s else float("nan"),
        "peak_ay_mps2": float(np.nanmax(np.abs(run.ay))),
        "meas_valid_fraction": run.meas_valid_fraction,
    }


def case_table() -> pd.DataFrame:
    cases = load_config("plant_cases.yaml")["cases"]
    rows = [
        _row(name, ctrl, run_closed_loop(ctrl, load_case(name), seed=0))
        for name in cases
        for ctrl in CASE_CONTROLLERS
    ]
    return pd.DataFrame(rows)


def noise_filter_sweep() -> pd.DataFrame:
    base = load_case("sensor_noise_2cm")
    rows: list[dict[str, Any]] = []
    for cutoff in SWEEP_CUTOFFS:
        cfg = deep_merge(base, {"controller": {"derivative_cutoff_hz": cutoff}})
        for ctrl in CASE_CONTROLLERS:
            for seed in SWEEP_SEEDS:
                row = _row("sensor_noise_2cm", ctrl, run_closed_loop(ctrl, cfg, seed=seed))
                row["cutoff_hz"] = float("nan") if cutoff is None else cutoff
                row["seed"] = seed
                rows.append(row)
    return pd.DataFrame(rows)


def equivalence_error() -> dict[str, float]:
    out: dict[str, float] = {}
    for ctrl in CASE_CONTROLLERS:
        new = run_closed_loop(ctrl, load_case("linear_equiv"))
        old = run_mission(ctrl, Scenario())
        out[ctrl] = float(np.nanmax(np.abs(new.ey - old.ey)))
    return out


def _md_limits(df: pd.DataFrame) -> str:
    lines = [
        "| tyre | v (m/s) | mu | max steady ay (m/s^2) | mu*g (m/s^2) | ay / (mu*g) |",
        "|---|---|---|---|---|---|",
    ]
    for r in df.itertuples():
        lines.append(
            f"| {r.tyre} | {r.vx_mps:.0f} | {r.mu:.2f} | {r.max_ay_mps2:.2f} | "
            f"{r.mu_g_mps2:.2f} | {r.fraction_of_mu_g:.3f} |"
        )
    return "\n".join(lines)


def _md_cases(df: pd.DataFrame) -> str:
    lines = [
        "| case | " + " | ".join(CASE_CONTROLLERS) + " |",
        "|---|" + "---|" * len(CASE_CONTROLLERS),
    ]
    for case in df["case"].drop_duplicates():
        cells = []
        for ctrl in CASE_CONTROLLERS:
            r = df[(df["case"] == case) & (df["controller"] == ctrl)].iloc[0]
            cells.append(
                f"DIV @{r['diverged_at_s']:.1f}s" if r["diverged"] else f"{r['max_abs_ey_cm']:.2f}"
            )
        lines.append(f"| {case} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _md_sweep(df: pd.DataFrame) -> str:
    lines = [
        "| derivative cutoff (Hz) | " + " | ".join(CASE_CONTROLLERS) + " |",
        "|---|" + "---|" * len(CASE_CONTROLLERS),
    ]
    for cutoff in df["cutoff_hz"].drop_duplicates():
        is_none = bool(np.isnan(cutoff))
        label = "none (finite difference)" if is_none else f"{cutoff:g}"
        cells = []
        for ctrl in CASE_CONTROLLERS:
            match = df["cutoff_hz"].isna() if is_none else df["cutoff_hz"] == cutoff
            sub = df[(df["controller"] == ctrl) & match]
            n_div = int(sub["diverged"].sum())
            ok = sub[~sub["diverged"]]["max_abs_ey_cm"]
            text = f"{n_div}/{len(sub)} DIV"
            cells.append(text + (f", median {ok.median():.2f}" if len(ok) else ""))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    meta = provenance()
    limits = plant_limits()
    cases = case_table()
    sweep = noise_filter_sweep()
    equiv = equivalence_error()
    for name, df in (
        ("plant_limits", limits),
        ("case_table", cases),
        ("noise_filter_sweep", sweep),
    ):
        for key, value in meta.items():
            df[key] = value
        df.to_parquet(RESULTS_DIR / f"{name}.parquet", index=False)
    header = (
        f"Status: MEASURED. Generated by `python -m egga.eval.phase1` from "
        f"`results/phase1/*.parquet` (git {meta['git_sha'][:8]}, config hash "
        f"{meta['config_hash']}, dirty={meta['git_dirty']}).\n"
    )
    equiv_lines = "\n".join(f"- {c}: max |e_y difference| = {v:.3e} m" for c, v in equiv.items())
    text = (
        "# Phase 1 plant report\n\n"
        + header
        + "\n## Linear-regime equivalence with the Phase 0 plant\n\n"
        "Case `linear_equiv` (linear_clip tyre, no load transfer, ideal actuator and sensors, "
        "finite-difference derivative) against the Phase 0 simulation, nominal mission:\n\n"
        + equiv_lines
        + "\n\n## Plant limits: maximum steady lateral acceleration\n\n"
        "Largest steady-turn lateral acceleration over constant steer angles 0.005-0.5 rad "
        "(100 values), 15 s settle, flat road. A ratio below 1 reflects load sensitivity and "
        "load transfer in the Pacejka tyre.\n\n"
        + _md_limits(limits)
        + "\n\n## Max cross-track error (cm) per plant case, seed 0\n\n"
        "Controllers use ground-truth mu (oracle) and a 5 Hz filtered derivative unless the case "
        "overrides it. DIV = |e_y| exceeded the divergence threshold.\n\n"
        + _md_cases(cases)
        + "\n\n## 2 cm lateral-error noise: derivative filter cutoff sweep\n\n"
        f"{len(SWEEP_SEEDS)} seeds per cell. `none` is the legacy finite difference.\n\n"
        + _md_sweep(sweep)
        + "\n"
    )
    (DOCS / "phase1_report.md").write_text(text, encoding="utf-8")
    print(f"wrote docs/phase1_report.md; equivalence max error {max(equiv.values()):.3e} m")


if __name__ == "__main__":
    main()
