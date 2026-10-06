from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

LOG_CAPACITY = 256
HISTORY = 32


class Mode(IntEnum):
    NOMINAL = 0
    CAUTIOUS = 1
    LOW_MU = 2
    DEGRADED_ACTUATOR = 3
    FALLBACK = 4
    MINIMAL_RISK = 5


class EstStatus(IntEnum):
    OK = 0
    LOW_EXCITATION = 1
    STALE = 2
    INVALID = 3


class Reason(IntEnum):
    NONE = 0
    MODE_CHANGE = 1
    GAIN_FORCED = 2
    RL_REJECTED_RANGE = 3
    RL_REJECTED_RATE = 4
    SPEED_CAPPED = 5
    NO_VERIFIED_SET = 6
    ESTIMATOR_FAULT = 7
    YAW_RESIDUAL = 8
    ACTUATOR_RESIDUAL = 9
    SATURATION = 10
    LATERAL_MARGIN = 11
    INVALID_INPUT = 12
    RECOVERY = 13
    DOMAIN_VIOLATION = 14
    RESET = 15


@dataclass(frozen=True)
class Config:
    """Flat configuration (all floats) so the C port can use a plain struct."""

    dt: float
    gain_hop_period: float
    min_dwell: float
    recover: float
    stale_grace: float
    mu_floor: float
    tau_prior: float
    mass_prior_max: float
    tau_fallback: float
    mass_fallback: float
    tau_degraded_enter: float
    tau_degraded_exit: float
    mu_low_enter: float
    mu_low_exit: float
    quality_enter: float
    quality_exit: float
    speed_scale: tuple[float, float, float, float, float, float]
    yaw_lpf_tau: float
    yaw_abs: float
    yaw_rel: float
    yaw_persist: float
    yaw_min_speed: float
    actuator_abs: float
    actuator_persist: float
    saturation_level: float
    saturation_persist: float
    rl_rate: tuple[float, float, float, float]
    rl_persist_rejects: int
    lane_margin: float
    margin_speed_scale: float
    min_speed_for_limit: float
    decel: float
    accel: float
    wheelbase: float
    lf: float
    lr: float
    cf: float
    cr: float
    gravity: float
    mass_nominal: float
    steer_hw_max: float
    steer_rate_hw_max: float

    def understeer_coeff(self, mass_scale: float) -> float:
        """K_us in delta = L * kappa + K_us * a_y for the given mass scale."""
        mass = self.mass_nominal * mass_scale
        return mass / self.wheelbase * (self.lr / self.cr - self.lf / self.cf)

    @classmethod
    def from_dict(cls, sup: dict[str, Any], vehicle: dict[str, Any]) -> Config:
        t, d, m, h = sup["timing"], sup["domain"], sup["modes"], sup["health"]
        scale = tuple(float(v) for v in m["speed_scale"])
        rate = tuple(float(v) for v in h["rl_rate_per_s"])
        if len(scale) != 6 or len(rate) != 4:
            raise ValueError("speed_scale needs 6 entries and rl_rate_per_s needs 4")
        return cls(
            dt=float(t["dt_s"]),
            gain_hop_period=float(t["gain_hop_period_s"]),
            min_dwell=float(t["min_dwell_s"]),
            recover=float(t["recover_s"]),
            stale_grace=float(t["stale_grace_s"]),
            mu_floor=float(d["mu_floor"]),
            tau_prior=float(d["tau_prior_s"]),
            mass_prior_max=float(d["mass_prior_max"]),
            tau_fallback=float(d["tau_fallback_s"]),
            mass_fallback=float(d["mass_fallback"]),
            tau_degraded_enter=float(m["tau_degraded_enter_s"]),
            tau_degraded_exit=float(m["tau_degraded_exit_s"]),
            mu_low_enter=float(m["mu_low_enter"]),
            mu_low_exit=float(m["mu_low_exit"]),
            quality_enter=float(m["quality_enter"]),
            quality_exit=float(m["quality_exit"]),
            speed_scale=(scale[0], scale[1], scale[2], scale[3], scale[4], scale[5]),
            yaw_lpf_tau=float(h["yaw_lpf_tau_s"]),
            yaw_abs=float(h["yaw_abs"]),
            yaw_rel=float(h["yaw_rel"]),
            yaw_persist=float(h["yaw_persist_s"]),
            yaw_min_speed=float(h["yaw_min_speed_mps"]),
            actuator_abs=float(h["actuator_abs_rad"]),
            actuator_persist=float(h["actuator_persist_s"]),
            saturation_level=float(h["saturation_level"]),
            saturation_persist=float(h["saturation_persist_s"]),
            rl_rate=(rate[0], rate[1], rate[2], rate[3]),
            rl_persist_rejects=int(h["rl_persist_rejects"]),
            lane_margin=float(sup["guard"]["lane_margin_m"]),
            margin_speed_scale=float(sup["guard"]["margin_speed_scale"]),
            min_speed_for_limit=float(sup["guard"]["min_speed_for_limit_mps"]),
            decel=float(sup["speed"]["decel_mps2"]),
            accel=float(sup["speed"]["accel_mps2"]),
            wheelbase=float(vehicle["lf_m"]) + float(vehicle["lr_m"]),
            lf=float(vehicle["lf_m"]),
            lr=float(vehicle["lr_m"]),
            cf=float(vehicle["cf_n_per_rad"]),
            cr=float(vehicle["cr_n_per_rad"]),
            gravity=float(vehicle["gravity_mps2"]),
            mass_nominal=float(vehicle["mass_kg"]),
            steer_hw_max=float(vehicle["delta_max_rad"]),
            steer_rate_hw_max=float(vehicle["steer_rate_max_rps"]),
        )


@dataclass(frozen=True)
class Inputs:
    """One supervisor tick. Plain scalars only; est_status is an EstStatus value."""

    t: float
    speed: float
    speed_request: float
    curvature_ahead: float
    mu_lo: float
    mu_hi: float
    tau_bar: float
    mass_hi: float
    quality: float
    est_status: int
    yaw_rate: float
    steer_meas: float
    steer_cmd: float
    e_abs: float
    e_rate_abs: float
    rl_valid: bool
    rl_gain: tuple[float, float, float, float]
    reference_gain: tuple[float, float, float, float]
    request_reset: bool = False


@dataclass(frozen=True)
class Outputs:
    gains: tuple[float, float, float, float]
    speed_cmd: float
    steer_limit: float
    rate_limit: float
    mode: int
    rl_applied: bool
    forced_gain_jump: bool
    verified: bool
    flags: int


# bit positions of Outputs.flags and State.prev_flags
FLAG_YAW = 1
FLAG_ACTUATOR = 2
FLAG_SATURATION = 4
FLAG_ESTIMATOR = 8
FLAG_MARGIN = 16
FLAG_INVALID = 32
FLAG_NO_SET = 64
FLAG_DOMAIN = 128
FLAG_CAPPED = 256


@dataclass
class State:
    """Fixed-size supervisor state (no dynamic allocation after construction)."""

    mode: int = int(Mode.NOMINAL)
    mode_entry_t: float = 0.0
    last_t: float = float("nan")
    healthy_for: float = 0.0
    yaw_pred: float = 0.0
    yaw_bad: float = 0.0
    act_bad: float = 0.0
    sat_bad: float = 0.0
    est_bad: float = 0.0
    rl_rejects: int = 0
    last_rl_valid: bool = False
    speed_cmd: float = float("nan")
    gain_idx: NDArray[np.int64] = field(default_factory=lambda: np.zeros(4, dtype=np.int64))
    have_gain: bool = False
    last_hop_t: float = float("-inf")
    last_rl: NDArray[np.float64] = field(default_factory=lambda: np.zeros(4))
    hist: NDArray[np.float64] = field(default_factory=lambda: np.zeros(HISTORY))
    hist_n: int = 0
    prev_flags: int = 0
    log_t: NDArray[np.float64] = field(default_factory=lambda: np.zeros(LOG_CAPACITY))
    log_code: NDArray[np.int64] = field(
        default_factory=lambda: np.zeros(LOG_CAPACITY, dtype=np.int64)
    )
    log_val: NDArray[np.float64] = field(default_factory=lambda: np.zeros(LOG_CAPACITY))
    log_n: int = 0
    mode_changes: int = 0
