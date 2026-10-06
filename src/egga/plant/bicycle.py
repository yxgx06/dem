from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VehicleParams:
    mass: float
    yaw_inertia: float
    lf: float
    lr: float
    cg_height: float
    cf: float
    cr: float
    gravity: float
    delta_max: float
    steer_rate_max: float
    min_tyre_force_cap: float

    @property
    def wheelbase(self) -> float:
        return self.lf + self.lr

    @classmethod
    def from_config(cls, cfg: dict[str, Any], mass_scale: float = 1.0) -> VehicleParams:
        return cls(
            mass=float(cfg["mass_kg"]) * mass_scale,
            yaw_inertia=float(cfg["yaw_inertia_kgm2"]),
            lf=float(cfg["lf_m"]),
            lr=float(cfg["lr_m"]),
            cg_height=float(cfg["cg_height_m"]),
            cf=float(cfg["cf_n_per_rad"]),
            cr=float(cfg["cr_n_per_rad"]),
            gravity=float(cfg["gravity_mps2"]),
            delta_max=float(cfg["delta_max_rad"]),
            steer_rate_max=float(cfg["steer_rate_max_rps"]),
            min_tyre_force_cap=float(cfg["min_tyre_force_cap_n"]),
        )


@dataclass(frozen=True)
class PlantStep:
    ay: float
    alpha_f: float
    alpha_r: float


class BicyclePlant:
    """Two-DOF linear-tyre bicycle with friction-limited lateral forces.

    Update order follows legacy/mission_proving_ground_rl.m lines 197-215 exactly:
    every derivative is evaluated at the old state, and psi integrates the old yaw rate.
    """

    def __init__(
        self,
        params: VehicleParams,
        vx: float,
        x0: float = 0.0,
        y0: float = 0.0,
        psi0: float = 0.0,
    ) -> None:
        if vx <= 0.0:
            raise ValueError("longitudinal speed must be positive")
        self.params = params
        self.vx = vx
        self.x = x0
        self.y = y0
        self.psi = psi0
        self.vy = 0.0
        self.r = 0.0

    def step(self, delta: float, mu: float, slope_deg: float, dt: float) -> PlantStep:
        p = self.params
        vx, vy, r = self.vx, self.vy, self.r
        theta = np.radians(slope_deg)
        weight = p.mass * p.gravity
        fz_front = weight * (p.lr * np.cos(theta) - p.cg_height * np.sin(theta)) / p.wheelbase
        fz_rear = weight * (p.lf * np.cos(theta) + p.cg_height * np.sin(theta)) / p.wheelbase
        fy_front_max = max(mu * fz_front, p.min_tyre_force_cap)
        fy_rear_max = max(mu * fz_rear, p.min_tyre_force_cap)

        alpha_f = delta - np.arctan2(vy + p.lf * r, vx)
        alpha_r = -np.arctan2(vy - p.lr * r, vx)
        fyf = float(np.clip(p.cf * alpha_f, -fy_front_max, fy_front_max))
        fyr = float(np.clip(p.cr * alpha_r, -fy_rear_max, fy_rear_max))

        ay = (fyf * np.cos(delta) + fyr) / p.mass
        d_vy = ay - vx * r
        d_r = (p.lf * fyf * np.cos(delta) - p.lr * fyr) / p.yaw_inertia
        d_x = vx * np.cos(self.psi) - vy * np.sin(self.psi)
        d_y = vx * np.sin(self.psi) + vy * np.cos(self.psi)

        self.vy = vy + d_vy * dt
        self.r = r + d_r * dt
        self.psi = self.psi + r * dt
        self.x = self.x + d_x * dt
        self.y = self.y + d_y * dt
        return PlantStep(ay=float(ay), alpha_f=float(alpha_f), alpha_r=float(alpha_r))
