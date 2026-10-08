from __future__ import annotations

import math
from dataclasses import dataclass

FLAG_JACKKNIFE_OK = 0
FLAG_JACKKNIFE_MARGIN = 1
FLAG_JACKKNIFE_CRITICAL = 2
FLAG_ROLLOVER_WARNING = 4


@dataclass(frozen=True)
class JackknifeConfig:
    """Thresholds and geometry for articulated commercial vehicle safety guard."""

    l2: float = 8.5  # Kingpin to trailer axle distance (m)
    tau_air: float = 0.25  # Pneumatic brake actuation delay (s)
    gravity: float = 9.80665  # Gravitational acceleration (m/s^2)
    ltr_warning: float = 0.70  # LTR threshold for rollover damping
    ltr_critical: float = 0.85  # LTR threshold for strict roll-rate clamping
    steer_rate_max: float = 0.60  # Maximum hardware steer rate (rad/s)
    min_theta_crit: float = 0.20  # Minimum critical angle floor (~11.5 deg)
    max_theta_crit: float = 0.785  # Maximum critical angle ceiling (45 deg)


@dataclass(frozen=True)
class JackknifeInputs:
    """Instantaneous vehicle states for jackknife and rollover evaluation."""

    theta_a: float  # Current articulation hitch angle (rad)
    theta_a_dot: float  # Articulation angular rate (rad/s)
    vx: float  # Forward speed (m/s)
    mu: float  # Surface friction coefficient
    ltr: float  # Current Load Transfer Ratio
    steer_cmd_req: float  # Incoming requested steering command (rad)
    steer_rate_req: float  # Incoming requested steering rate (rad/s)


@dataclass(frozen=True)
class JackknifeOutputs:
    """Safe, guarded outputs preventing jackknife and rollover incidents."""

    theta_crit: float  # Critical articulation threshold (rad)
    h_jackknife: float  # Barrier margin (rad): theta_crit - predicted_theta
    is_jackknife_critical: bool
    is_rollover_critical: bool
    steer_rate_safe: float  # Safe, rate-governed steering rate (rad/s)
    steer_cmd_safe: float  # Guarded steering command (rad)
    trailer_brake_pressure: float  # Differential trailer drag (0.0 to 1.0)
    status_flags: int  # Bitmask of safety flags


def compute_critical_articulation_angle(
    vx: float, mu: float, l2: float, g: float, min_val: float, max_val: float
) -> float:
    """Computes physical critical articulation angle above which yaw instability snaps:

    theta_crit = arcsin(mu * g * L2 / v^2)
    """
    v_eff = max(vx, 1.0) if math.isfinite(vx) else 1.0
    mu_eff = max(mu, 0.05) if math.isfinite(mu) else 0.05
    arg = (mu_eff * g * l2) / (v_eff * v_eff)
    arg_clamped = min(max(arg, -1.0), 1.0)
    theta_raw = math.asin(arg_clamped)
    return min(max(abs(theta_raw), min_val), max_val)


def step_jackknife_guard(cfg: JackknifeConfig, inp: JackknifeInputs) -> JackknifeOutputs:
    """Evaluates the dynamic anti-jackknife barrier and rollover Load Transfer Ratio.

    Enforces active trailer drag and steering rate throttling to maintain articulation stability.
    """
    theta_a = inp.theta_a if math.isfinite(inp.theta_a) else 0.0
    theta_a_dot = inp.theta_a_dot if math.isfinite(inp.theta_a_dot) else 0.0
    ltr = min(max(inp.ltr, 0.0), 1.0) if math.isfinite(inp.ltr) else 0.0
    steer_cmd = inp.steer_cmd_req if math.isfinite(inp.steer_cmd_req) else 0.0
    steer_rate = inp.steer_rate_req if math.isfinite(inp.steer_rate_req) else 0.0

    # 1. Critical Articulation Angle & Predictive Barrier
    theta_crit = compute_critical_articulation_angle(
        vx=inp.vx,
        mu=inp.mu,
        l2=cfg.l2,
        g=cfg.gravity,
        min_val=cfg.min_theta_crit,
        max_val=cfg.max_theta_crit,
    )

    predicted_theta = abs(theta_a) + cfg.tau_air * abs(theta_a_dot)
    h_jackknife = theta_crit - predicted_theta

    flags = FLAG_JACKKNIFE_OK
    trailer_drag = 0.0
    is_jk_crit = False

    if h_jackknife <= 0.0:
        is_jk_crit = True
        flags |= FLAG_JACKKNIFE_CRITICAL
        # Maximum trailer brake pressure to create stabilizing tension (parachute effect)
        trailer_drag = 1.0
    elif h_jackknife < 0.08:
        flags |= FLAG_JACKKNIFE_MARGIN
        # Proportional trailer drag application
        trailer_drag = min(1.0, (0.08 - h_jackknife) / 0.08)

    # 2. Rollover Load Transfer Ratio (LTR) Guard
    is_ro_crit = False
    rate_scale = 1.0

    if ltr >= cfg.ltr_critical:
        is_ro_crit = True
        flags |= FLAG_ROLLOVER_WARNING
        # Absolute steer rate lock: cannot increase lateral acceleration
        rate_scale = 0.0
    elif ltr >= cfg.ltr_warning:
        flags |= FLAG_ROLLOVER_WARNING
        # Progressive damping as LTR approaches critical
        rate_scale = max(0.0, (cfg.ltr_critical - ltr) / (cfg.ltr_critical - cfg.ltr_warning))

    # Steer rate throttling under jackknife or rollover danger
    if is_jk_crit:
        # Heavily dampen steering inputs that drive further articulation
        rate_scale = min(rate_scale, 0.2)

    steer_rate_safe = steer_rate * rate_scale
    steer_rate_safe = min(max(steer_rate_safe, -cfg.steer_rate_max), cfg.steer_rate_max)

    # Commanded steering angle is clamped if rollover or jackknife is critical
    if is_ro_crit:
        steer_cmd_safe = steer_cmd * 0.8  # Counter-roll reduction
    elif is_jk_crit:
        # Prevent increasing steering in direction of articulation
        if (theta_a > 0.0 and steer_cmd > 0.0) or (theta_a < 0.0 and steer_cmd < 0.0):
            steer_cmd_safe = steer_cmd * 0.5
        else:
            steer_cmd_safe = steer_cmd
    else:
        steer_cmd_safe = steer_cmd

    return JackknifeOutputs(
        theta_crit=theta_crit,
        h_jackknife=h_jackknife,
        is_jackknife_critical=is_jk_crit,
        is_rollover_critical=is_ro_crit,
        steer_rate_safe=steer_rate_safe,
        steer_cmd_safe=steer_cmd_safe,
        trailer_brake_pressure=trailer_drag,
        status_flags=flags,
    )
