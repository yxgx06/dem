from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from egga.config import load_baselines, load_config
from egga.controllers.base import Controller, Observation
from egga.controllers.classical import (
    PDFeedforwardController,
    PIDController,
    PIDFeedforwardController,
)
from egga.controllers.lqr import DesignVehicle, LQRController
from egga.controllers.mpc import MPCController
from egga.controllers.rl_actor import RLScheduler, load_rl_weights
from egga.plant.actuator import TransportDelay, steer_actuator
from egga.plant.bicycle import BicyclePlant, VehicleParams
from egga.scenarios.mission import Mission, build_mission

CONTROLLER_NAMES = ("pid", "pd_ff", "rl_handtyped", "rl_trained")
MU_BELIEF_FLOOR = 0.05
MU_BELIEF_CEIL = 1.0


@dataclass(frozen=True)
class Scenario:
    vx: float = 10.0
    mass_scale: float = 1.0
    mu_belief_error: float = 0.0
    delay_s: float = 0.0
    noise_std_m: float = 0.0
    seed: int = 0


@dataclass(frozen=True)
class RunResult:
    controller: str
    scenario: Scenario
    t: np.ndarray
    ey: np.ndarray
    steer: np.ndarray
    gains: np.ndarray
    diverged_at_s: float | None


@lru_cache(maxsize=8)
def _mission(vx: float) -> Mission:
    return build_mission(vx=vx)


def make_controller(
    name: str,
    mission: Mission,
    dt: float,
    derivative_cutoff_hz: float | None = None,
    mass_scale: float = 1.0,
    inertia_follows_mass: bool | None = None,
) -> Controller:
    """Build a named controller.

    Legacy: pid, pd_ff, rl_handtyped, rl_trained. Phase 2 baselines: b0_pid_ff, b1_pd_ff,
    b2_rl_oracle, b6_mpc, b7_lqr_true_mass, b7_lqr_nominal_mass, b7_lqr_delay_aug. mass_scale is
    the mass the design is allowed to know (true-mass variants); nominal-mass variants ignore it.
    """
    ctrl_cfg = load_config("controllers.yaml")
    if name.startswith(("b0_", "b1_", "b6_", "b7_", "b2_")):
        return _make_baseline(
            name, mission, dt, derivative_cutoff_hz, mass_scale, inertia_follows_mass, ctrl_cfg
        )
    if name == "pid":
        return PIDController(ctrl_cfg["pid"], dt, derivative_cutoff_hz)
    if name == "pd_ff":
        return PDFeedforwardController(
            ctrl_cfg["pd_ff"], mission.wheelbase, mission.vx, dt, derivative_cutoff_hz
        )
    if name in ("rl_handtyped", "rl_trained"):
        weights = load_rl_weights(name.removeprefix("rl_"))
        return RLScheduler(
            weights, ctrl_cfg["rl"], dt, mission.wheelbase, mission.vx, derivative_cutoff_hz
        )
    raise KeyError(f"unknown controller: {name}")


def _make_baseline(
    name: str,
    mission: Mission,
    dt: float,
    cutoff: float | None,
    mass_scale: float,
    inertia_follows_mass: bool | None,
    ctrl_cfg: dict[str, Any],
) -> Controller:
    base = load_baselines()
    vehicle = load_config("vehicle.yaml")
    if name == "b0_pid_ff":
        return PIDFeedforwardController(base["pid_ff"], dt, mission.wheelbase, mission.vx, cutoff)
    if name == "b1_pd_ff":
        return PDFeedforwardController(base["pd_ff"], mission.wheelbase, mission.vx, dt, cutoff)
    if name == "b2_rl_oracle":
        weights = load_rl_weights("handtyped")
        return RLScheduler(weights, ctrl_cfg["rl"], dt, mission.wheelbase, mission.vx, cutoff)
    lqr_cfg = base["lqr"]
    follows = bool(lqr_cfg["inertia_follows_mass"]) if inertia_follows_mass is None else (
        inertia_follows_mass
    )
    if name == "b6_mpc":
        design = DesignVehicle.from_config(vehicle, mass_scale, follows)
        return MPCController(
            design,
            base["mpc"],
            dt,
            float(vehicle["delta_max_rad"]),
            float(vehicle["steer_rate_max_rps"]),
            cutoff,
            mission.vx,
        )
    variants = {
        "b7_lqr_true_mass": (mass_scale, 0.0),
        "b7_lqr_nominal_mass": (1.0, 0.0),
        "b7_lqr_delay_aug": (mass_scale, float(lqr_cfg["delay_design_s"])),
    }
    if name in variants:
        scale, delay = variants[name]
        design = DesignVehicle.from_config(vehicle, scale, follows)
        return LQRController(name, design, lqr_cfg, dt, cutoff, delay, mission.vx)
    raise KeyError(f"unknown controller: {name}")


def _wrap(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def run_mission(
    controller_name: str,
    scenario: Scenario,
    divergence_threshold_m: float | None = None,
) -> RunResult:
    if divergence_threshold_m is None:
        divergence_threshold_m = float(load_config("stress.yaml")["divergence_threshold_m"])
    mission = _mission(scenario.vx)
    vehicle = load_config("vehicle.yaml")
    params = VehicleParams.from_config(vehicle, scenario.mass_scale)
    plant = BicyclePlant(
        params,
        mission.vx,
        x0=float(mission.x_ref[0]),
        y0=float(mission.y_ref[0]),
        psi0=float(mission.psi_ref[0]),
    )
    controller = make_controller(
        controller_name,
        mission,
        mission.dt,
        mass_scale=scenario.mass_scale,
        inertia_follows_mass=False,  # the Phase 0 plant keeps yaw inertia fixed
    )
    controller.reset()
    delay = TransportDelay(round(scenario.delay_s / mission.dt))
    rng = np.random.default_rng(scenario.seed)
    rate_max = float(vehicle["steer_rate_max_rps"])
    angle_max = float(vehicle["delta_max_rad"])

    n = mission.n
    dt = mission.dt
    ey = np.full(n, np.nan)
    steer = np.full(n, np.nan)
    gains = np.full((n, 4), np.nan)
    delta = 0.0
    e_prev = 0.0
    diverged_at: float | None = None

    with np.errstate(all="ignore"):
        for k in range(n - 1):
            ex_g = mission.x_ref[k] - plant.x
            ey_g = mission.y_ref[k] - plant.y
            e_true = ey_g * np.cos(plant.psi) - ex_g * np.sin(plant.psi)
            if not np.isfinite(e_true) or abs(e_true) > divergence_threshold_m:
                diverged_at = float(mission.t[k])
                ey[k] = e_true
                break

            noise = rng.normal() * scenario.noise_std_m
            e_meas = e_true + noise
            de_y = (e_meas - e_prev) / dt if k > 0 else 0.0
            e_prev = e_meas
            mu_belief = float(
                np.clip(mission.mu[k] + scenario.mu_belief_error, MU_BELIEF_FLOOR, MU_BELIEF_CEIL)
            )

            obs = Observation(
                e_y=float(e_meas),
                de_y=float(de_y),
                heading_error=_wrap(mission.psi_ref[k] - plant.psi),
                yaw_rate=plant.r,
                yaw_rate_ref=float(mission.r_ref[k]),
                slope_deg=float(mission.slope_deg[k]),
                mu_belief=mu_belief,
            )
            cmd = controller.command(obs)
            delta = steer_actuator(delay.step(cmd.steer), delta, dt, rate_max, angle_max)
            plant.step(delta, float(mission.mu[k]), float(mission.slope_deg[k]), dt)

            ey[k] = e_true
            steer[k] = delta
            gains[k] = cmd.gains

    if diverged_at is None:
        ey[n - 1] = ey[n - 2]
        steer[n - 1] = steer[n - 2]
        gains[n - 1] = gains[n - 2]

    return RunResult(
        controller=controller_name,
        scenario=scenario,
        t=mission.t,
        ey=ey,
        steer=steer,
        gains=gains,
        diverged_at_s=diverged_at,
    )
