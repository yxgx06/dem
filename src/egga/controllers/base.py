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


@dataclass(frozen=True)
class Command:
    steer: float
    gains: tuple[float, float, float, float]


class Controller(Protocol):
    name: str

    def reset(self) -> None: ...

    def command(self, obs: Observation) -> Command: ...


def clip(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)
