from __future__ import annotations

from egga.supervisor.types import Config


def steer_angle_limit(cfg: Config, speed: float, ay_max: float, mass_hi: float) -> float:
    """Steering angle that produces a_y,max in steady state, capped by the hardware limit."""
    v = max(speed, cfg.min_speed_for_limit)
    angle = (cfg.wheelbase / (v * v) + cfg.understeer_coeff(mass_hi)) * ay_max
    return min(cfg.steer_hw_max, angle)


def lateral_margin_violated(
    cfg: Config, e_abs: float, e_rate_abs: float, ay_max: float, tau_bar: float
) -> bool:
    """Worst-case extra deviation during the delay bound must stay inside the lane margin.

    deviation = |e_dot| * tau_bar + 0.5 * a_y,max * tau_bar^2 (a design analogy to a safety
    distance check, not a formal safety model).
    """
    deviation = e_rate_abs * tau_bar + 0.5 * ay_max * tau_bar * tau_bar
    return e_abs + deviation > cfg.lane_margin


def clip_symmetric(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))
