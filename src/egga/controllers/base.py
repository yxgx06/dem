from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Observation:
    e_y: float
    de_y: float
    heading_error: float
    yaw_rate: float
    yaw_rate_ref: float
    slope_deg: float
    mu_belief: float
    vx: float = 0.0  # 0 -> controller falls back to its configured nominal speed
    r_ref_preview: tuple[float, ...] = ()  # future reference yaw rate (planner preview)
    estimate: Any = None  # latest egga.estimation Estimate (previous tick), when an estimator runs
    steer_meas: float = 0.0  # measured steering angle
    speed_request: float = 0.0  # planned speed (0 -> same as vx)
    curvature_ahead: float = 0.0  # largest planned path curvature ahead


@dataclass(frozen=True)
class Command:
    steer: float
    gains: tuple[float, float, float, float]
    solve_time_s: float = 0.0  # wall time of an optimisation solve on this tick, 0 otherwise
    speed_cmd: float | None = None  # speed the controller asks the vehicle to drive (None: no ask)
    mode: int = -1  # supervisor mode, -1 when there is no supervisor


class Controller(Protocol):
    name: str

    def reset(self) -> None: ...

    def command(self, obs: Observation) -> Command: ...


def clip(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)
