from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from egga.plant.tyre import TyreParams, linear_clip_force, pacejka_force


@dataclass(frozen=True)
class WindParams:
    speed: float
    gust_amp: float
    gust_period: float
    side_force_area: float
    air_density: float
    aero_offset: float

    def side_force(self, t: float) -> float:
        w = self.speed + self.gust_amp * np.sin(2.0 * np.pi * t / self.gust_period)
        return float(0.5 * self.air_density * self.side_force_area * w * abs(w))


@dataclass(frozen=True)
class PlantParams:
    mass: float
    yaw_inertia: float
    lf: float
    lr: float
    cg_height: float
    cf: float
    cr: float
    gravity: float
    min_force_cap: float
    tyre: TyreParams
    track: float
    front_roll_share: float
    lateral_transfer: bool
    longitudinal_transfer: bool
    wind: WindParams

    @property
    def wheelbase(self) -> float:
        return self.lf + self.lr

    @classmethod
    def from_configs(cls, vehicle: dict[str, Any], plant: dict[str, Any]) -> PlantParams:
        mass_cfg = plant["mass"]
        mass_scale = float(mass_cfg["mass_scale"])
        inertia_scale = mass_cfg["inertia_scale"]
        inertia_scale = mass_scale if inertia_scale is None else float(inertia_scale)
        t = plant["tyre"]
        stiff = float(t["stiffness_scale"])
        lt = plant["load_transfer"]
        w = plant["crosswind"]
        return cls(
            mass=float(vehicle["mass_kg"]) * mass_scale,
            yaw_inertia=float(vehicle["yaw_inertia_kgm2"]) * inertia_scale,
            lf=float(vehicle["lf_m"]),
            lr=float(vehicle["lr_m"]),
            cg_height=float(vehicle["cg_height_m"]),
            cf=float(vehicle["cf_n_per_rad"]) * stiff,
            cr=float(vehicle["cr_n_per_rad"]) * stiff,
            gravity=float(vehicle["gravity_mps2"]),
            min_force_cap=float(vehicle["min_tyre_force_cap_n"]),
            tyre=TyreParams(
                model=str(t["model"]),
                shape_c=float(t["shape_c"]),
                curvature_e=float(t["curvature_e"]),
                load_sensitivity=float(t["load_sensitivity"]),
                min_force_cap=float(vehicle["min_tyre_force_cap_n"]),
            ),
            track=float(lt["track_width_m"]),
            front_roll_share=float(lt["front_roll_share"]),
            lateral_transfer=bool(lt["enabled"]),
            longitudinal_transfer=bool(lt["longitudinal"]),
            wind=WindParams(
                speed=float(w["speed_mps"]),
                gust_amp=float(w["gust_amp_mps"]),
                gust_period=float(w["gust_period_s"]),
                side_force_area=float(w["side_force_area_m2"]),
                air_density=float(w["air_density"]),
                aero_offset=float(w["aero_centre_offset_m"]),
            ),
        )


@dataclass(frozen=True)
class VehicleStep:
    ay: float
    alpha_f: float
    alpha_r: float
    fyf: float
    fyr: float
    wheel_loads: tuple[float, float, float, float]


class Vehicle:
    """Two-DOF bicycle with nonlinear per-wheel tyres, load transfer, variable speed and wind.

    Independent of the Phase 0 plant (egga.plant.bicycle). The integration order is the same
    explicit-Euler order, so the linear_clip tyre with every extra effect off reproduces Phase 0.
    Lateral load transfer uses the previous step's lateral acceleration (quasi-static).
    """

    def __init__(
        self, params: PlantParams, x0: float = 0.0, y0: float = 0.0, psi0: float = 0.0
    ) -> None:
        self.params = params
        self.x = x0
        self.y = y0
        self.psi = psi0
        self.vy = 0.0
        self.r = 0.0
        self.vx = 0.0
        self._ay_prev = 0.0

    def _axle_loads(self, slope_deg: float, ax: float) -> tuple[float, float]:
        p = self.params
        theta = np.radians(slope_deg)
        weight = p.mass * p.gravity
        fz_f = weight * (p.lr * np.cos(theta) - p.cg_height * np.sin(theta)) / p.wheelbase
        fz_r = weight * (p.lf * np.cos(theta) + p.cg_height * np.sin(theta)) / p.wheelbase
        return float(fz_f), float(fz_r)

    def step(
        self,
        delta: float,
        mu_left: float,
        mu_right: float,
        slope_deg: float,
        dt: float,
        vx: float,
        ax: float = 0.0,
        t: float = 0.0,
    ) -> VehicleStep:
        if vx <= 0.0:
            raise ValueError("longitudinal speed must be positive")
        p = self.params
        self.vx = vx
        vy, r = self.vy, self.r

        fz_f_static, fz_r_static = self._axle_loads(slope_deg, ax)
        fz_f, fz_r = fz_f_static, fz_r_static
        if p.longitudinal_transfer:
            shift = p.mass * ax * p.cg_height / p.wheelbase
            fz_f -= shift
            fz_r += shift
        d_f = d_r = 0.0
        if p.lateral_transfer:
            total = p.mass * self._ay_prev * p.cg_height / p.track
            d_f = p.front_roll_share * total
            d_r = (1.0 - p.front_roll_share) * total
        loads = (
            max(fz_f / 2.0 - d_f, 0.0),
            max(fz_f / 2.0 + d_f, 0.0),
            max(fz_r / 2.0 - d_r, 0.0),
            max(fz_r / 2.0 + d_r, 0.0),
        )

        alpha_f = delta - np.arctan2(vy + p.lf * r, vx)
        alpha_r = -np.arctan2(vy - p.lr * r, vx)

        if p.tyre.model == "linear_clip":
            mu_axle = 0.5 * (mu_left + mu_right)
            fyf = linear_clip_force(alpha_f, p.cf, fz_f, mu_axle, p.min_force_cap)
            fyr = linear_clip_force(alpha_r, p.cr, fz_r, mu_axle, p.min_force_cap)
        elif p.tyre.model == "pacejka":
            kc_f = p.cf / fz_f_static
            kc_r = p.cr / fz_r_static
            nom_f, nom_r = fz_f_static / 2.0, fz_r_static / 2.0
            fyf = pacejka_force(alpha_f, loads[0], mu_left, kc_f, nom_f, p.tyre) + pacejka_force(
                alpha_f, loads[1], mu_right, kc_f, nom_f, p.tyre
            )
            fyr = pacejka_force(alpha_r, loads[2], mu_left, kc_r, nom_r, p.tyre) + pacejka_force(
                alpha_r, loads[3], mu_right, kc_r, nom_r, p.tyre
            )
        else:
            raise ValueError(f"unknown tyre model: {p.tyre.model}")

        f_wind = p.wind.side_force(t)
        ay = (fyf * np.cos(delta) + fyr + f_wind) / p.mass
        d_vy = ay - vx * r
        d_r_dot = (p.lf * fyf * np.cos(delta) - p.lr * fyr + f_wind * p.wind.aero_offset) / (
            p.yaw_inertia
        )
        d_x = vx * np.cos(self.psi) - vy * np.sin(self.psi)
        d_y = vx * np.sin(self.psi) + vy * np.cos(self.psi)

        self.vy = vy + d_vy * dt
        self.r = r + d_r_dot * dt
        self.psi = self.psi + r * dt
        self.x = self.x + d_x * dt
        self.y = self.y + d_y * dt
        self._ay_prev = float(ay)
        return VehicleStep(
            ay=float(ay),
            alpha_f=float(alpha_f),
            alpha_r=float(alpha_r),
            fyf=float(fyf),
            fyr=float(fyr),
            wheel_loads=loads,
        )
