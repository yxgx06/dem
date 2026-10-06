from __future__ import annotations

from typing import Any

import numpy as np

from egga.config import load_config
from egga.eval.simulate import RunResult

# Index into mission stage_bounds_s where each reported window starts (stage_bounds_s is
# [0, 15, 30, 45, 60, 75]; rain = [30, 45), ice = [45, 60), lane change = [60, 75]).
RAIN_START_INDEX = 2
ICE_START_INDEX = 3
LANE_CHANGE_START_INDEX = 4


def _window_max_cm(run: RunResult, bounds: list[float], start_index: int) -> float:
    lo = bounds[start_index]
    hi = bounds[start_index + 1]
    mask = (run.t >= lo) & (run.t < hi)
    values = np.abs(run.ey[mask])
    values = values[np.isfinite(values)]
    return float(values.max() * 100.0) if values.size else float("nan")


def summarize(run: RunResult) -> dict[str, Any]:
    cfg = load_config("mission.yaml")
    bounds = [float(v) for v in cfg["stage_bounds_s"]]
    finite = run.ey[np.isfinite(run.ey)]
    diverged = run.diverged_at_s is not None
    return {
        "max_abs_ey_cm": float(np.max(np.abs(finite)) * 100.0) if finite.size else float("nan"),
        "rms_ey_cm": float(np.sqrt(np.mean(finite**2)) * 100.0) if finite.size else float("nan"),
        "rain_max_cm": _window_max_cm(run, bounds, RAIN_START_INDEX),
        "ice_max_cm": _window_max_cm(run, bounds, ICE_START_INDEX),
        "lane_change_max_cm": _window_max_cm(run, bounds, LANE_CHANGE_START_INDEX),
        "diverged": diverged,
        "diverged_at_s": float(run.diverged_at_s) if diverged else float("nan"),
    }
