from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FrictionProfile:
    """Tyre-road friction as a function of time t and distance s along the path.

    Profile types: constant, step, ramp, patches (by path distance), array (per-sample values).
    The split-mu option scales the left and right wheel friction separately (known-failure flag).
    """

    kind: str
    params: dict[str, Any]
    left_factor: float = 1.0
    right_factor: float = 1.0

    @classmethod
    def from_config(cls, profile: dict[str, Any], split: dict[str, Any]) -> FrictionProfile:
        kind = str(profile["type"])
        if kind not in {"constant", "step", "ramp", "patches", "array"}:
            raise ValueError(f"unknown friction profile type: {kind}")
        left = right = 1.0
        if split.get("enabled", False):
            left = float(split["left_factor"])
            right = float(split["right_factor"])
        return cls(kind=kind, params=dict(profile), left_factor=left, right_factor=right)

    @classmethod
    def from_array(cls, values: np.ndarray, dt: float) -> FrictionProfile:
        return cls(kind="array", params={"type": "array", "values": values, "dt": dt})

    def mu(self, t: float, s: float) -> float:
        p = self.params
        if self.kind == "constant":
            return float(p["mu"])
        if self.kind == "step":
            return float(p["after"] if t >= float(p["t_s"]) else p["before"])
        if self.kind == "ramp":
            t0, t1 = float(p["t0_s"]), float(p["t1_s"])
            frac = float(np.clip((t - t0) / (t1 - t0), 0.0, 1.0))
            return float(p["before"]) + frac * (float(p["after"]) - float(p["before"]))
        if self.kind == "patches":
            for patch in p["patches"]:
                if float(patch["s0_m"]) <= s < float(patch["s1_m"]):
                    return float(patch["mu"])
            return float(p["default"])
        values = p["values"]
        idx = min(int(round(t / float(p["dt"]))), len(values) - 1)
        return float(values[idx])

    def mu_wheels(self, t: float, s: float) -> tuple[float, float]:
        base = self.mu(t, s)
        return base * self.left_factor, base * self.right_factor
