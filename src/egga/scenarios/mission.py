from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from egga.config import load_config


@dataclass(frozen=True)
class Mission:
    dt: float
    vx: float
    wheelbase: float
    t: np.ndarray
    slope_deg: np.ndarray
    mu: np.ndarray
    delta_ref: np.ndarray
    r_ref: np.ndarray
    psi_ref: np.ndarray
    x_ref: np.ndarray
    y_ref: np.ndarray

    @property
    def n(self) -> int:
        return int(self.t.shape[0])


def _stage_profiles(cfg: dict[str, Any], t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bounds = [float(v) for v in cfg["stage_bounds_s"]]
    b1, b2, b3, b4 = bounds[1], bounds[2], bounds[3], bounds[4]
    s = cfg["stages"]
    slope = np.zeros_like(t)
    mu = np.zeros_like(t)
    for k, tk in enumerate(t):
        if tk < b1:
            slope[k] = s["flat_dry"]["slope_deg"]
            mu[k] = s["flat_dry"]["mu"]
        elif tk < b2:
            up = s["uphill_dry"]
            slope[k] = up["slope_max_deg"] * (1.0 - np.exp(-(tk - b1) / up["slope_tau_s"]))
            mu[k] = up["mu"]
        elif tk < b3:
            wet = s["uphill_wet"]
            slope[k] = wet["slope_deg"]
            mu[k] = wet["mu_start"] - wet["mu_drop"] * (1.0 - np.exp(-(tk - b2) / wet["mu_tau_s"]))
        elif tk < b4:
            ice = s["downhill_ice"]
            ramp = (tk - b3) / ice["slope_ramp_s"]
            if ramp <= 1.0:
                span = ice["slope_end_deg"] - ice["slope_start_deg"]
                slope[k] = ice["slope_start_deg"] + span * ramp
            else:
                slope[k] = ice["slope_end_deg"]
            mu[k] = ice["mu_start"] - ice["mu_drop"] * (1.0 - np.exp(-(tk - b3) / ice["mu_tau_s"]))
        else:
            lc = s["lane_change"]
            ramp = min(1.0, (tk - b4) / lc["ramp_s"])
            slope[k] = lc["slope_start_deg"] + (lc["slope_end_deg"] - lc["slope_start_deg"]) * ramp
            mu[k] = lc["mu_start"] + (lc["mu_end"] - lc["mu_start"]) * ramp
    return slope, mu


def _reference_steer(cfg: dict[str, Any], t: np.ndarray) -> np.ndarray:
    ref = cfg["reference"]
    amp = float(ref["sine_amp_rad"])
    period = float(ref["sine_period_s"])
    lc_start = float(ref["lc_start_s"])
    lc_amp = float(ref["lc_amp_rad"])
    lc_period = float(ref["lc_period_s"])
    seg = [float(v) for v in ref["lc_segments_s"]]
    signs = (1.0, -1.0)
    delta = np.zeros_like(t)
    for k, tk in enumerate(t):
        if tk <= lc_start:
            delta[k] = amp * np.sin(2.0 * np.pi / period * tk)
            continue
        t_lc = tk - lc_start
        for i, sign in enumerate(signs):
            if seg[i] <= t_lc < seg[i + 1]:
                delta[k] = sign * lc_amp * np.sin(2.0 * np.pi / lc_period * (t_lc - seg[i]))
                break
    return delta


def build_mission(vx: float | None = None, cfg: dict[str, Any] | None = None) -> Mission:
    if cfg is None:
        cfg = load_config("mission.yaml")
    vehicle = load_config("vehicle.yaml")
    speed = float(cfg["vx_mps"]) if vx is None else float(vx)
    if speed <= 0.0:
        raise ValueError("vx must be positive")
    dt = float(cfg["dt_s"])
    t_final = float(cfg["t_final_s"])
    t = np.arange(0.0, t_final + dt / 2.0, dt)
    n = t.shape[0]
    wheelbase = float(vehicle["lf_m"]) + float(vehicle["lr_m"])

    slope, mu = _stage_profiles(cfg, t)
    delta_ref = _reference_steer(cfg, t)

    x_ref = np.zeros(n)
    y_ref = np.zeros(n)
    psi_ref = np.zeros(n)
    r_ref = np.zeros(n)
    for k in range(1, n):
        r_ref[k - 1] = speed / wheelbase * np.tan(delta_ref[k - 1])
        psi_ref[k] = psi_ref[k - 1] + r_ref[k - 1] * dt
        x_ref[k] = x_ref[k - 1] + speed * np.cos(psi_ref[k - 1]) * dt
        y_ref[k] = y_ref[k - 1] + speed * np.sin(psi_ref[k - 1]) * dt
    r_ref[n - 1] = speed / wheelbase * np.tan(delta_ref[n - 1])

    return Mission(
        dt=dt,
        vx=speed,
        wheelbase=wheelbase,
        t=t,
        slope_deg=slope,
        mu=mu,
        delta_ref=delta_ref,
        r_ref=r_ref,
        psi_ref=psi_ref,
        x_ref=x_ref,
        y_ref=y_ref,
    )
