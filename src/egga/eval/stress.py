from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from egga.config import REPO_ROOT, load_config
from egga.eval.metrics import summarize
from egga.eval.provenance import provenance
from egga.eval.simulate import Scenario, run_mission

RESULTS_DIR = REPO_ROOT / "results" / "phase0"


def _scenario(case: dict[str, Any], seed: int, base_vx: float) -> Scenario:
    return Scenario(
        vx=float(case.get("vx_mps", base_vx)),
        mass_scale=float(case.get("mass_scale", 1.0)),
        mu_belief_error=float(case.get("mu_belief_error", 0.0)),
        delay_s=float(case.get("delay_s", 0.0)),
        noise_std_m=float(case.get("noise_std_m", 0.0)),
        seed=seed,
    )


def run_stress(out_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    stress = load_config("stress.yaml")
    base_vx = float(load_config("mission.yaml")["vx_mps"])
    threshold = float(stress["divergence_threshold_m"])
    controllers = list(stress["controllers"])
    meta = provenance()
    rows: list[dict[str, Any]] = []
    traces_dir = out_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    for case_name, case in stress["cases"].items():
        case = case or {}
        for seed in case.get("seeds", [0]):
            scenario = _scenario(case, int(seed), base_vx)
            trace: dict[str, np.ndarray] = {}
            t_axis: np.ndarray | None = None
            for ctrl in controllers:
                run = run_mission(ctrl, scenario, divergence_threshold_m=threshold)
                t_axis = run.t
                trace[f"{ctrl}_ey_m"] = run.ey
                trace[f"{ctrl}_steer_rad"] = run.steer
                rows.append(
                    {
                        "case": case_name,
                        "controller": ctrl,
                        "seed": int(seed),
                        "vx_mps": scenario.vx,
                        "mass_scale": scenario.mass_scale,
                        "mu_belief_error": scenario.mu_belief_error,
                        "delay_s": scenario.delay_s,
                        "noise_std_m": scenario.noise_std_m,
                        **summarize(run),
                        **meta,
                    }
                )
            assert t_axis is not None
            frame = pd.DataFrame({"t_s": t_axis, **trace})
            frame.to_parquet(traces_dir / f"{case_name}__seed{int(seed)}.parquet", index=False)

    summary = pd.DataFrame(rows)
    summary.to_parquet(out_dir / "summary.parquet", index=False)
    return summary


def main() -> None:
    summary = run_stress()
    diverged = int(summary["diverged"].sum())
    target = RESULTS_DIR / "summary.parquet"
    print(f"wrote {len(summary)} rows to {target}; {diverged} diverged runs")


if __name__ == "__main__":
    main()
