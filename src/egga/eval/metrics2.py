from __future__ import annotations

from typing import Any

import numpy as np

from egga.config import load_config
from egga.eval.closed_loop import ClosedLoopResult

UTILISATION_THRESHOLD = 0.95


def run_metrics(run: ClosedLoopResult, mu_min: float) -> dict[str, Any]:
    """Per-run tracking, tyre and actuator metrics over the samples before any divergence."""
    vehicle = load_config("vehicle.yaml")
    g = float(vehicle["gravity_mps2"])
    rate_max = float(vehicle["steer_rate_max_rps"])
    angle_max = float(vehicle["delta_max_rad"])
    dt = float(run.t[1] - run.t[0])
    valid = np.isfinite(run.ey)
    ey = np.abs(run.ey[valid])
    diverged = run.diverged_at_s is not None
    steer = run.steer[valid & np.isfinite(run.steer)]
    steer_rate = np.diff(steer) / dt if steer.size > 1 else np.zeros(1)
    ay = np.abs(run.ay[np.isfinite(run.ay)])
    mu = run.mu[np.isfinite(run.mu)]
    util = ay / np.maximum(mu * g, 1e-6) if ay.size else np.zeros(1)
    top = max(1, int(np.ceil(0.01 * ey.size)))
    yaw_err = (run.yaw_rate - run.yaw_rate_ref)[valid & np.isfinite(run.yaw_rate)]
    return {
        "max_abs_ey_cm": float(ey.max() * 100.0),
        "rms_ey_cm": float(np.sqrt(np.mean(ey**2)) * 100.0),
        "p95_abs_ey_cm": float(np.percentile(ey, 95) * 100.0),
        "worst1pct_ey_cm": float(np.sort(ey)[-top:].mean() * 100.0),
        "yaw_rate_err_rms": float(np.sqrt(np.mean(yaw_err**2))) if yaw_err.size else float("nan"),
        "max_sideslip_deg": float(np.degrees(np.nanmax(np.abs(run.sideslip)))),
        "peak_utilisation": float(util.max()),
        "utilisation_over_frac": float(np.mean(util > UTILISATION_THRESHOLD)),
        "saturation_frac": float(
            np.mean(
                (np.abs(steer[1:]) >= 0.999 * angle_max) | (np.abs(steer_rate) >= 0.99 * rate_max)
            )
            if steer.size > 1
            else 0.0
        ),
        "steer_rate_rms": float(np.sqrt(np.mean(steer_rate**2))),
        "demand_over_limit_mu_min": float(
            np.nanmax(np.abs(run.ay)) / max(mu_min * g, 1e-6)
        ),
        "diverged": diverged,
        "diverged_at_s": float(run.diverged_at_s) if diverged else float("nan"),
        "solve_mean_ms": float(run.solve_times.mean() * 1e3) if run.solve_times.size else 0.0,
        "solve_p99_ms": float(np.percentile(run.solve_times, 99) * 1e3)
        if run.solve_times.size
        else 0.0,
        "solve_max_ms": float(run.solve_times.max() * 1e3) if run.solve_times.size else 0.0,
    }


def tuning_cost(metrics: dict[str, Any]) -> float:
    """Scalar objective used only for tuning on the TRAIN set (lower is better)."""
    penalty = 100.0 if metrics["diverged"] else 0.0
    return metrics["rms_ey_cm"] + 0.25 * metrics["max_abs_ey_cm"] + penalty
