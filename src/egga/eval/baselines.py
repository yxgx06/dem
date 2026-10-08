from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from egga.config import REPO_ROOT, load_config, load_estimators, set_baseline_override
from egga.estimation.suite import EstimatorSuite
from egga.eval.closed_loop import ClosedLoopResult, load_plant_config, run_closed_loop
from egga.eval.metrics2 import run_metrics
from egga.eval.provenance import provenance
from egga.eval.simulate import Scenario, run_mission
from egga.scenarios.sets import load_scenarios, run_arguments

RESULTS_DIR = REPO_ROOT / "results" / "phase2"
DOCS = REPO_ROOT / "docs"
BASELINES = (
    "b0_pid_ff",
    "b1_pd_ff",
    "b2_rl_oracle",
    "b6_mpc",
    "b7_lqr_true_mass",
    "b7_lqr_nominal_mass",
    "b7_lqr_delay_aug",
)
EVAL_SETS = ("train", "val")


def run_scenario(controller: str, spec: dict[str, Any]) -> ClosedLoopResult:
    overrides, mission_cfg, belief_error, seed = run_arguments(spec)
    cfg = load_plant_config(overrides)
    estimator = (
        EstimatorSuite(load_config("vehicle.yaml"), load_estimators(), float(mission_cfg["dt_s"]))
        if controller.startswith("b4_")
        else None
    )
    return run_closed_loop(
        controller,
        cfg,
        seed=seed,
        mu_belief_error=belief_error,
        mission_cfg=mission_cfg,
        estimator=estimator,
    )


def evaluate_set(
    controllers: tuple[str, ...], set_name: str, limit: int | None = None
) -> pd.DataFrame:
    specs = load_scenarios(set_name)
    if limit is not None:
        specs = specs[:limit]
    meta = provenance()
    rows: list[dict[str, Any]] = []
    for spec in specs:
        for ctrl in controllers:
            run = run_scenario(ctrl, spec)
            rows.append(
                {
                    "set": set_name,
                    "scenario": spec["id"],
                    "controller": ctrl,
                    "family": spec["friction_family"],
                    "demand_ratio": spec["demand_ratio"],
                    "speed_mps": spec["speed_mps"],
                    "mass_scale": spec["mass_scale"],
                    "delay_s": spec["delay_s"],
                    **run_metrics(run, spec["mu_min"]),
                    **meta,
                }
            )
    return pd.DataFrame(rows)


def audit_lqr_reproduction() -> pd.DataFrame:
    """Reproduce the audit's LQR claims on the Phase 0 simulator (linear tyre, constant inertia)."""
    ref = load_config("audit_reference.yaml")["lqr"]
    cases = {
        "nominal": ("b7_lqr_true_mass", Scenario()),
        "mass_x1.3_nominal_design": ("b7_lqr_nominal_mass", Scenario(mass_scale=1.3)),
        "mass_x1.9_nominal_design": ("b7_lqr_nominal_mass", Scenario(mass_scale=1.9)),
        "delay_200ms": ("b7_lqr_true_mass", Scenario(delay_s=0.2)),
    }
    rows = []
    set_baseline_override({})  # the audit's original Q/R, not the TRAIN-tuned values
    for name, (ctrl, scenario) in cases.items():
        run = run_mission(ctrl, scenario)
        rows.append(
            {
                "case": name,
                "controller": ctrl,
                "audit_claim_cm": ref[name],
                "measured_max_cm": float(np.nanmax(np.abs(run.ey)) * 100.0),
                "diverged": run.diverged_at_s is not None,
            }
        )
    set_baseline_override(None)
    return pd.DataFrame(rows)


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ctrl in BASELINES:
        sub = df[df["controller"] == ctrl]
        ok = sub[~sub["diverged"]]
        rows.append(
            {
                "controller": ctrl,
                "n": len(sub),
                "diverged": int(sub["diverged"].sum()),
                "median_max_cm": float(ok["max_abs_ey_cm"].median()),
                "mean_rms_cm": float(ok["rms_ey_cm"].mean()),
                "median_p95_cm": float(ok["p95_abs_ey_cm"].median()),
                "mean_worst1pct_cm": float(ok["worst1pct_ey_cm"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame, floats: int = 2) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        cells = [f"{v:.{floats}f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(exist_ok=True)
    meta = provenance()
    parts = []
    frames = {}
    for set_name in EVAL_SETS:
        df = evaluate_set(BASELINES, set_name)
        df.to_parquet(RESULTS_DIR / f"{set_name}_runs.parquet", index=False)
        frames[set_name] = df
        parts.append(
            f"## {set_name} set ({df['scenario'].nunique()} scenarios)\n\n"
            + _md_table(_aggregate(df))
        )
    audit = audit_lqr_reproduction()
    audit.to_parquet(RESULTS_DIR / "lqr_audit_reproduction.parquet", index=False)

    mpc = pd.concat(frames.values())
    mpc = mpc[mpc["controller"] == "b6_mpc"]
    solve = (
        f"B6 MPC solve time over {len(mpc)} runs (HOST measurement, Python + OSQP on this "
        f"machine; not an ECU number): mean of run means {mpc['solve_mean_ms'].mean():.3f} ms, "
        f"worst p99 {mpc['solve_p99_ms'].max():.3f} ms, "
        f"worst max {mpc['solve_max_ms'].max():.3f} ms."
    )
    header = (
        f"Status: MEASURED. Generated by `python -m egga.eval.baselines` from "
        f"`results/phase2/*.parquet` (git {meta['git_sha'][:8]}, config hash "
        f"{meta['config_hash']}, dirty={meta['git_dirty']}). Controllers use ground-truth mu "
        f"(oracle) and true mass where noted; tuned on the TRAIN set only.\n"
    )
    text = (
        "# Phase 2 baselines\n\n"
        + header
        + "\n"
        + "\n\n".join(parts)
        + "\n\nMax/RMS/p95/worst-1% are over non-diverged runs only; the divergence count is "
        "reported separately. Cross-track error in cm.\n\n## B6 solve time\n\n"
        + solve
        + "\n\n## LQR audit reproduction (Phase 0 simulator)\n\n"
        "Audit claims come from `configs/audit_reference.yaml`; they are claims, not results.\n\n"
        + _md_table(audit)
        + "\n"
    )
    (DOCS / "phase2_report.md").write_text(text, encoding="utf-8")
    print("wrote docs/phase2_report.md")


if __name__ == "__main__":
    main()
