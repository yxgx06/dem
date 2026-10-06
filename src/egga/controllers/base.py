from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


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


@dataclass(frozen=True)
class Command:
    steer: float
    gains: tuple[float, float, float, float]
    solve_time_s: float = 0.0  # wall time of an optimisation solve on this tick, 0 otherwise


class Controller(Protocol):
    name: str

    def reset(self) -> None: ...

    def command(self, obs: Observation) -> Command: ...


def clip(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)
