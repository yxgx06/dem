from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from egga.config import load_config
from egga.controllers.base import Observation
from egga.estimation.suite import EstimatorSuite
from egga.estimation.types import Measurements
from egga.eval.simulate import make_controller
from egga.plant.actuator import SteeringActuator
from egga.plant.friction import FrictionProfile
from egga.plant.sensors import Sensor
from egga.plant.vehicle import PlantParams, Vehicle
from egga.scenarios.mission import Mission, build_mission

MU_BELIEF_FLOOR = 0.05
MU_BELIEF_CEIL = 1.0


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_plant_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = load_config("plant.yaml")
    return deep_merge(cfg, overrides) if overrides else cfg


def load_case(name: str) -> dict[str, Any]:
    cases = load_config("plant_cases.yaml")["cases"]
    if name not in cases:
        raise KeyError(f"unknown plant case: {name}")
    return load_plant_config(cases[name] or {})


def speed_profile(
    cfg: dict[str, Any], n: int, dt: float, mission_cfg: dict[str, Any] | None = None
) -> np.ndarray:
    speed = cfg["speed"]
    if speed["profile"] == "constant":
        nominal = speed.get("constant_mps")
        default = (mission_cfg or load_config("mission.yaml"))["vx_mps"]
        value = float(default) if nominal is None else float(nominal)
        return np.full(n, value)
    if speed["profile"] == "ramp":
        r = speed["ramp"]
        t = np.arange(n) * dt
        span = float(r["t_end_s"]) - float(r["t_start_s"])
        frac = np.clip((t - float(r["t_start_s"])) / span, 0, 1)
        return float(r["v_start_mps"]) + frac * (float(r["v_end_mps"]) - float(r["v_start_mps"]))
    raise ValueError(f"unknown speed profile: {speed['profile']}")


@dataclass(frozen=True)
class ClosedLoopResult:
    controller: str
    t: np.ndarray
    ey: np.ndarray
    yaw_rate: np.ndarray
    steer: np.ndarray
    ay: np.ndarray
    alpha_f: np.ndarray
    alpha_r: np.ndarray
    mu: np.ndarray
    vx: np.ndarray
    gains: np.ndarray
    meas_valid_fraction: float
    diverged_at_s: float | None
    oracle_mu: bool = True  # controllers receive true mu (+ belief error) until Phase 3
    solve_times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    yaw_rate_ref: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sideslip: np.ndarray = field(default_factory=lambda: np.zeros(0))
    estimates: dict[str, np.ndarray] = field(default_factory=dict)
    truth: dict[str, float] = field(default_factory=dict)
    modes: np.ndarray = field(default_factory=lambda: np.zeros(0))


def _wrap(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def run_closed_loop(
    controller_name: str,
    plant_cfg: dict[str, Any] | None = None,
    seed: int = 0,
    mu_belief_error: float = 0.0,
    divergence_threshold_m: float | None = None,
    mission: Mission | None = None,
    mission_cfg: dict[str, Any] | None = None,
    estimator: EstimatorSuite | None = None,
    mu_source: str = "oracle",
    controller_instance: Any = None,
) -> ClosedLoopResult:
    cfg = plant_cfg if plant_cfg is not None else load_plant_config()
    vehicle_cfg = load_config("vehicle.yaml")
    if divergence_threshold_m is None:
        divergence_threshold_m = float(load_config("stress.yaml")["divergence_threshold_m"])

    if mission_cfg is None:
        mission_cfg = load_config("mission.yaml")
    n = int(round(float(mission_cfg["t_final_s"]) / float(mission_cfg["dt_s"]))) + 1
    dt = float(mission_cfg["dt_s"])
    vx_t = speed_profile(cfg, n, dt, mission_cfg)
    ax_t = np.empty(n)
    ax_t[:-1] = np.diff(vx_t) / dt
    ax_t[-1] = ax_t[-2]
    if mission is None:
        mission = build_mission(vx=vx_t, cfg=mission_cfg)

    params = PlantParams.from_configs(vehicle_cfg, cfg)
    vehicle = Vehicle(
        params,
        x0=float(mission.x_ref[0]),
        y0=float(mission.y_ref[0]),
        psi0=float(mission.psi_ref[0]),
    )
    children = np.random.SeedSequence(seed).spawn(5)
    rngs = [np.random.default_rng(c) for c in children]
    actuator = SteeringActuator(
        cfg["actuator"],
        dt,
        float(vehicle_cfg["steer_rate_max_rps"]),
        float(vehicle_cfg["delta_max_rad"]),
        rngs[0],
    )
    sensors = {
        "lateral_error": Sensor(cfg["sensors"]["lateral_error"], rngs[1]),
        "yaw_rate": Sensor(cfg["sensors"]["yaw_rate"], rngs[2]),
        "steering_angle": Sensor(cfg["sensors"]["steering_angle"], rngs[3]),
        "lateral_accel": Sensor(cfg["sensors"]["lateral_accel"], rngs[4]),
    }
    if mu_source not in ("oracle", "estimate_safe"):
        raise ValueError(f"unknown mu_source: {mu_source}")
    if mu_source == "estimate_safe" and estimator is None:
        raise ValueError("mu_source='estimate_safe' needs an estimator")
    if estimator is not None:
        estimator.reset()
    est_names = ("mu_lo", "mu_hi", "tau_bar", "mass_lo", "mass_hi", "quality", "ok")
    est_log = {k: np.full(n, np.nan) for k in est_names}
    last_estimate = estimator.widest(0.0, "stale") if estimator is not None else None
    ay_prev = 0.0
    friction_cfg = cfg["friction"]
    if friction_cfg["profile"] is None:
        friction = FrictionProfile.from_array(mission.mu, dt)
        if friction_cfg["split_mu"]["enabled"]:
            friction = FrictionProfile(
                kind="array",
                params=friction.params,
                left_factor=float(friction_cfg["split_mu"]["left_factor"]),
                right_factor=float(friction_cfg["split_mu"]["right_factor"]),
            )
    else:
        friction = FrictionProfile.from_config(friction_cfg["profile"], friction_cfg["split_mu"])

    cutoff = cfg["controller"]["derivative_cutoff_hz"]
    if controller_instance is not None:
        controller = controller_instance
    else:
        controller = make_controller(
            controller_name,
            mission,
            dt,
            derivative_cutoff_hz=None if cutoff is None else float(cutoff),
            mass_scale=float(cfg["mass"]["mass_scale"]),
            inertia_follows_mass=cfg["mass"]["inertia_scale"] is None,
        )
    stride = int(getattr(controller, "preview_spacing_steps", 0))
    preview_count = int(getattr(controller, "preview_count", 0))
    solve_times: list[float] = []
    controller.reset()
    governed = bool(getattr(controller, "governs_speed", False))
    arclength = np.concatenate(
        [[0.0], np.cumsum(np.hypot(np.diff(mission.x_ref), np.diff(mission.y_ref)))]
    )
    curvature_ref = mission.r_ref / mission.vx_t
    v_cur = float(vx_t[0])
    mode_log = np.full(n, np.nan)

    names = ("ey", "yaw_rate", "steer", "ay", "alpha_f", "alpha_r", "mu", "vx", "r_ref", "beta")
    log = {k: np.full(n, np.nan) for k in names}
    gains = np.full((n, 4), np.nan)
    valid = 0
    total = 0
    path_s = 0.0
    e_prev = 0.0
    diverged_at: float | None = None

    with np.errstate(all="ignore"):
        for k in range(n - 1):
            t_k = float(mission.t[k])
            if governed:  # path-indexed reference: the vehicle may be slower than the plan
                s_v = path_s
                x_r = float(np.interp(s_v, arclength, mission.x_ref))
                y_r = float(np.interp(s_v, arclength, mission.y_ref))
                psi_r = float(np.interp(s_v, arclength, mission.psi_ref))
                vx_k = v_cur
                r_ref_k = float(np.interp(s_v, arclength, curvature_ref)) * vx_k
                slope_k = float(np.interp(s_v, arclength, mission.slope_deg))
                v_plan = float(np.interp(s_v, arclength, vx_t))
                look = max(30.0, 3.0 * vx_k)
                j0 = int(np.searchsorted(arclength, s_v))
                j1 = int(np.searchsorted(arclength, s_v + look)) + 1
                curv_ahead = float(np.max(np.abs(curvature_ref[j0:j1]))) if j1 > j0 else 0.0
            else:
                x_r, y_r, psi_r = (
                    float(mission.x_ref[k]),
                    float(mission.y_ref[k]),
                    float(mission.psi_ref[k]),
                )
                vx_k, r_ref_k, slope_k = (
                    float(vx_t[k]),
                    float(mission.r_ref[k]),
                    float(mission.slope_deg[k]),
                )
                v_plan, curv_ahead = vx_k, 0.0
            ex_g = x_r - vehicle.x
            ey_g = y_r - vehicle.y
            e_true = float(ey_g * np.cos(vehicle.psi) - ex_g * np.sin(vehicle.psi))
            if not np.isfinite(e_true) or abs(e_true) > divergence_threshold_m:
                diverged_at = t_k
                log["ey"][k] = e_true
                break

            m_ey = sensors["lateral_error"].measure(e_true)
            m_r = sensors["yaw_rate"].measure(vehicle.r)
            m_st = sensors["steering_angle"].measure(actuator.angle)
            m_ay = sensors["lateral_accel"].measure(ay_prev)
            valid += int(m_ey.valid) + int(m_r.valid)
            total += 2

            mu_l, mu_r = friction.mu_wheels(t_k, path_s)
            mu_true = friction.mu(t_k, path_s)
            belief = float(np.clip(mu_true + mu_belief_error, MU_BELIEF_FLOOR, MU_BELIEF_CEIL))
            if mu_source == "estimate_safe" and estimator is not None and last_estimate is not None:
                belief = float(
                    np.clip(
                        last_estimate.mu_safe(estimator.k_safe, estimator.mu_min),
                        MU_BELIEF_FLOOR,
                        MU_BELIEF_CEIL,
                    )
                )
            de_fd = (m_ey.value - e_prev) / dt if k > 0 else 0.0
            e_prev = m_ey.value
            preview: tuple[float, ...] = ()
            if stride:
                idx = np.minimum(k + stride * np.arange(preview_count), n - 1)
                preview = tuple(float(v) for v in mission.r_ref[idx])
            obs = Observation(
                e_y=m_ey.value,
                de_y=de_fd,
                heading_error=_wrap(psi_r - vehicle.psi),
                yaw_rate=m_r.value,
                yaw_rate_ref=r_ref_k,
                slope_deg=slope_k,
                mu_belief=belief,
                vx=vx_k,
                r_ref_preview=preview,
                estimate=last_estimate,
                steer_meas=m_st.value,
                speed_request=v_plan,
                curvature_ahead=curv_ahead,
            )
            cmd = controller.command(obs)
            if cmd.solve_time_s > 0.0:
                solve_times.append(cmd.solve_time_s)
            if estimator is not None:
                last_estimate = estimator.update(
                    Measurements(
                        t=t_k,
                        vx=vx_k,
                        yaw_rate=m_r.value,
                        yaw_rate_valid=m_r.valid,
                        lateral_accel=m_ay.value,
                        lateral_accel_valid=m_ay.valid,
                        steer_meas=m_st.value,
                        steer_meas_valid=m_st.valid,
                        steer_cmd=cmd.steer,
                    )
                )
                for key, val in (
                    ("mu_lo", last_estimate.mu_lo),
                    ("mu_hi", last_estimate.mu_hi),
                    ("tau_bar", last_estimate.tau_bar),
                    ("mass_lo", last_estimate.mass_lo),
                    ("mass_hi", last_estimate.mass_hi),
                    ("quality", last_estimate.quality),
                    ("ok", 1.0 if last_estimate.status == "ok" else 0.0),
                ):
                    est_log[key][k] = val
            delta = actuator.step(cmd.steer)
            if governed and cmd.speed_cmd is not None:
                v_new = max(float(cmd.speed_cmd), 1.0)
                ax_k = (v_new - vx_k) / dt
                v_cur = v_new
            else:
                v_new, ax_k = vx_k, float(ax_t[k])
            out = vehicle.step(delta, mu_l, mu_r, slope_k, dt, v_new, ax_k, t_k)
            path_s += v_new * dt
            ay_prev = out.ay
            mode_log[k] = cmd.mode

            log["ey"][k] = e_true
            log["yaw_rate"][k] = vehicle.r
            log["steer"][k] = delta
            log["ay"][k] = out.ay
            log["alpha_f"][k] = out.alpha_f
            log["alpha_r"][k] = out.alpha_r
            log["mu"][k] = mu_true
            log["vx"][k] = v_new
            log["r_ref"][k] = r_ref_k
            log["beta"][k] = np.arctan2(vehicle.vy, v_new)
            gains[k] = cmd.gains

    if diverged_at is None:
        for arr in (*log.values(), *est_log.values(), mode_log):
            arr[n - 1] = arr[n - 2]
        gains[n - 1] = gains[n - 2]

    return ClosedLoopResult(
        controller=controller_name,
        t=mission.t,
        ey=log["ey"],
        yaw_rate=log["yaw_rate"],
        steer=log["steer"],
        ay=log["ay"],
        alpha_f=log["alpha_f"],
        alpha_r=log["alpha_r"],
        mu=log["mu"],
        vx=log["vx"],
        gains=gains,
        meas_valid_fraction=valid / total if total else 1.0,
        diverged_at_s=diverged_at,
        solve_times=np.asarray(solve_times),
        yaw_rate_ref=log["r_ref"],
        sideslip=log["beta"],
        estimates=est_log if estimator is not None else {},
        modes=mode_log,
        truth={
            "distance_m": path_s,
            "path_length_m": float(arclength[-1]),
            "mass_scale": float(cfg["mass"]["mass_scale"]),
            "latency_s": float(cfg["actuator"]["delay_s"])
            + 0.5 * float(cfg["actuator"]["jitter_s"])
            + float(cfg["actuator"]["lag_tau_s"]),
        },
    )
