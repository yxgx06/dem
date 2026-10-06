from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from egga.config import REPO_ROOT, load_baselines, set_baseline_override
from egga.eval.closed_loop import load_plant_config, run_closed_loop
from egga.eval.envelope_build import ENVELOPE_DIR
from egga.eval.provenance import provenance
from egga.scenarios.mission import build_mission
from egga.supervisor.envelope import Envelope

RESULTS_DIR = REPO_ROOT / "results" / "phase4"
DOCS = REPO_ROOT / "docs"
DESIGN_BOUNDS_S = (0.0, 0.04, 0.06, 0.08, 0.10)
NOMINAL_SPEED = 10.0
NOMINAL_MU_LO = 0.2  # lowest grid value at or below the mission's ice friction (0.25)
NOMINAL_MASS = 1.0


def cell_counts(env: Envelope) -> dict[str, Any]:
    shape = env.verified.shape
    cells = int(np.prod(shape[:4]))
    verified = env.verified.reshape(cells, -1)
    linear = env.linear_accepted.reshape(cells, -1)
    return {
        "cells": cells,
        "cells_linear": int(linear.any(axis=1).sum()),
        "cells_verified": int(verified.any(axis=1).sum()),
        "gains_per_cell_linear": float(linear.sum(axis=1).mean()),
        "gains_per_cell_verified": float(verified.sum(axis=1).mean()),
        "candidates": int(np.prod(shape[4:])),
    }


def per_axis_table(env: Envelope) -> pd.DataFrame:
    rows = []
    for axis, name in enumerate(("speed", "mu", "tau", "mass")):
        n_cells_each = int(np.prod([env.verified.shape[i] for i in range(4) if i != axis]))
        for k, value in enumerate(env.axes[name]):
            flat = np.take(env.verified, k, axis=axis).reshape(n_cells_each, -1)
            flat_lin = np.take(env.linear_accepted, k, axis=axis).reshape(n_cells_each, -1)
            rows.append(
                {
                    "axis": name,
                    "value": float(value),
                    "cells_with_set": f"{int(flat.any(axis=1).sum())}/{n_cells_each}",
                    "mean_verified_gains": float(flat.sum(axis=1).mean()),
                    "mean_linear_gains": float(flat_lin.sum(axis=1).mean()),
                }
            )
    return pd.DataFrame(rows)


def _run(speed: float, tau: float, gains: tuple[float, float, float, float]) -> dict[str, Any]:
    ref = load_baselines()["pid_ff"]
    set_baseline_override(
        {
            "pid_ff": {
                "kp": gains[0],
                "ki": gains[1],
                "kd": gains[2],
                "khead": gains[3],
                "integ_limit": float(ref["integ_limit"]),
            }
        }
    )
    try:
        cfg = load_plant_config({"actuator": {"delay_s": tau}, "speed": {"constant_mps": speed}})
        run = run_closed_loop("b0_pid_ff", cfg)
    finally:
        set_baseline_override(None)
    finite = run.ey[np.isfinite(run.ey)]
    return {
        "max_ey_cm": float(np.max(np.abs(finite)) * 100.0),
        "rms_ey_cm": float(np.sqrt(np.mean(finite**2)) * 100.0),
        "diverged": run.diverged_at_s is not None,
    }


def mission_peak_curvature() -> float:
    """Largest path curvature (1/m) the 75 s mission asks for."""
    mission = build_mission()
    return float(np.max(np.abs(np.tan(mission.delta_ref))) / mission.wheelbase)


def conservatism_curve(env: Envelope) -> pd.DataFrame:
    """Tracking error on the 75 s mission when the true steering delay equals the design bound.

    mu_lo is the lowest grid value at or below the mission's lowest friction (ice, 0.25).
    """
    ref = load_baselines()["pid_ff"]
    tuned = (float(ref["kp"]), float(ref["ki"]), float(ref["kd"]), float(ref["khead"]))
    kappa = mission_peak_curvature()
    nan = float("nan")
    rows: list[dict[str, Any]] = []
    for tau in DESIGN_BOUNDS_S:
        cap = env.speed_cap(kappa, NOMINAL_MU_LO)
        v_ver = env.max_verified_speed(NOMINAL_MU_LO, tau, NOMINAL_MASS)
        governed = None if cap is None or v_ver is None else min(NOMINAL_SPEED, cap, v_ver)
        plans: list[tuple[str, float | None, tuple[float, float, float, float] | None]] = []
        for case, speed in (
            ("envelope_gain_at_10mps", NOMINAL_SPEED),
            ("envelope_gain_speed_governed", governed),
        ):
            mask = (
                env.cell_mask(speed, NOMINAL_MU_LO, tau, NOMINAL_MASS)
                if speed is not None
                else None
            )
            plans.append((case, speed, env.nominal(mask) if mask is not None else None))
        plans.append(("tuned_b0_at_10mps_unprotected", NOMINAL_SPEED, tuned))
        for case, speed, gains in plans:
            row: dict[str, Any] = {
                "design_bound_s": tau,
                "case": case,
                "speed_mps": nan if speed is None else speed,
            }
            if gains is None or speed is None:
                row.update(
                    {
                        "kp": nan,
                        "ki": nan,
                        "kd": nan,
                        "khead": nan,
                        "max_ey_cm": nan,
                        "rms_ey_cm": nan,
                        "diverged": None,
                        "note": "no verified gain",
                    }
                )
            else:
                row.update(
                    {
                        "kp": gains[0],
                        "ki": gains[1],
                        "kd": gains[2],
                        "khead": gains[3],
                        **_run(speed, tau, gains),
                        "note": "",
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _md(df: pd.DataFrame, floats: int = 3) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        cells = [f"{v:.{floats}f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    env = Envelope.load(ENVELOPE_DIR / "envelope_v1.npz", ENVELOPE_DIR / "envelope_v1.json")
    manifest = json.loads((ENVELOPE_DIR / "envelope_v1.json").read_text(encoding="utf-8"))
    meta = provenance()
    counts = cell_counts(env)
    axis = per_axis_table(env)
    curve = conservatism_curve(env)
    axis_md, curve_md = _md(axis), _md(curve)
    for name, df in (("per_axis", axis), ("conservatism", curve)):
        for k, v in meta.items():
            df[k] = v
        df.to_parquet(RESULTS_DIR / f"{name}.parquet", index=False)
    nl = manifest["counts"]
    types = nl["nonlinear_by_pick_type"]
    type_rows = "\n".join(
        f"| {k} | {v['tests']} | {v['failures']} | "
        f"{(v['failures'] / v['tests'] if v['tests'] else float('nan')):.3f} |"
        for k, v in types.items()
    )
    text = (
        "# Phase 4 envelope\n\n"
        f"Status: MEASURED. Generated by `python -m egga.eval.phase4` from the committed envelope "
        f"`experiments/envelope/envelope_v{manifest['version']}.npz` "
        f"(hash `{manifest['hash'][:16]}`, "
        f"built at git {manifest['git_sha'][:8]}); this report at git "
        f"{meta['git_sha'][:8]}. The claim "
        f"is 'stable under assumptions A1..A12' (docs/ASSUMPTIONS.md), nothing more.\n\n"
        "## Cell counts\n\n"
        f"- Grid cells: {counts['cells']} (speed x mu_lo x tau_bar x mass).\n"
        f"- Cells with at least one linearly accepted gain: {counts['cells_linear']} of "
        f"{counts['cells']}.\n"
        f"- Cells with at least one VERIFIED gain (after nonlinear confirmation and pruning): "
        f"{counts['cells_verified']} of {counts['cells']}.\n"
        f"- Mean accepted gains per cell: linear {counts['gains_per_cell_linear']:.1f}, verified "
        f"{counts['gains_per_cell_verified']:.1f} of {counts['candidates']} candidates.\n"
        f"- Nonlinear confirmation: {nl['nonlinear_failures']} failing tests of "
        f"{nl['nonlinear_tests']} (each failure prunes the gain and every gain above it).\n\n"
        "### Nonlinear confirmation outcomes by test type\n\n"
        "Only the first five types are chosen to be aggressive; `random` tests a seeded random "
        "linearly accepted gain, so its failure rate estimates how often the linear analysis "
        "over-accepts. Untested accepted gains rely on the linear analysis (see the design "
        "notes).\n\n"
        "| test type | tests | failures | failure rate |\n|---|---|---|---|\n"
        + type_rows
        + "\n\n## Verified set by axis value\n\n"
        + axis_md
        + "\n\n## Conservatism cost\n\n"
        "75 s mission on the Phase 1 plant with the true steering delay equal to the design bound "
        f"(mu_lo {NOMINAL_MU_LO:g}, mass {NOMINAL_MASS:g}). `envelope_gain_at_10mps` uses "
        "the verified "
        "gain nearest the tuned B0 gains; an empty row means no gain is verified at that speed and "
        "bound. `envelope_gain_speed_governed` runs at min(10 m/s, the speed cap for the "
        "mission's peak "
        "curvature, the highest verified speed) (the degraded-mode speed). "
        "`tuned_b0_at_10mps_unprotected` keeps the "
        "train-tuned B0 gains regardless of delay.\n\n" + curve_md + "\n"
    )
    (DOCS / "phase4_report.md").write_text(text, encoding="utf-8")
    print("wrote docs/phase4_report.md")


if __name__ == "__main__":
    main()
