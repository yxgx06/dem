from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum


class FrictionCirclePolicy(IntEnum):
    """Priority policy when combined longitudinal and lateral acceleration demands

    exceed the available friction circle (Kamm's circle).
    """

    STEERING_PRIORITY = 0  # Preserve steering/evasion capacity; clamp braking/acceleration
    BALANCED = 1  # Uniformly scale both longitudinal and lateral demands along the vector
    BRAKING_PRIORITY = 2  # Preserve stopping distance; clamp lateral steering authority


@dataclass(frozen=True)
class FrictionDemand:
    """Demands and physical parameters for coupled 2D friction circle allocation."""

    ax_req: float
    ay_req: float
    mu: float
    gravity: float = 9.80665
    safety_factor: float = 0.95
    policy: FrictionCirclePolicy = FrictionCirclePolicy.STEERING_PRIORITY


@dataclass(frozen=True)
class FrictionAllocated:
    """Safe, friction-circle-compliant allocated accelerations and steering limits."""

    ax_safe: float
    ay_safe: float
    utilisation: float
    is_clamped: bool
    clamped_axis: str  # "none", "longitudinal", "lateral", "both"
    delta_max_coupled: float  # Physical steering angle bound under current lateral capacity


def allocate_friction_circle(
    demand: FrictionDemand,
    speed: float,
    wheelbase: float,
    steer_hw_max: float = 0.60,
) -> FrictionAllocated:
    """Allocates longitudinal and lateral accelerations within Kamm's friction circle:

        (ax / (mu * g))^2 + (ay / (mu * g))^2 <= gamma^2

    Guarantees that tire saturation cannot be exceeded by combined braking and cornering.
    """
    g = demand.gravity if demand.gravity > 0.0 else 9.80665
    mu_eff = max(demand.mu, 1e-4) if math.isfinite(demand.mu) else 1e-4
    gamma = min(max(demand.safety_factor, 0.1), 1.0)
    a_max_total = gamma * mu_eff * g

    ax_raw = demand.ax_req if math.isfinite(demand.ax_req) else 0.0
    ay_raw = demand.ay_req if math.isfinite(demand.ay_req) else 0.0

    ax_abs = abs(ax_raw)
    ay_abs = abs(ay_raw)

    rho = math.hypot(ax_abs, ay_abs) / (mu_eff * g)

    if rho <= gamma:
        # Operating strictly within the friction circle
        ax_safe = ax_raw
        ay_safe = ay_raw
        is_clamped = False
        clamped_axis = "none"
    else:
        is_clamped = True
        if demand.policy == FrictionCirclePolicy.STEERING_PRIORITY:
            # Preserve lateral steering capacity up to a_max_total, clamp remaining to longitudinal
            ay_clamped = min(ay_abs, a_max_total)
            rem_sq = max(0.0, a_max_total * a_max_total - ay_clamped * ay_clamped)
            ax_clamped = min(ax_abs, math.sqrt(rem_sq))
            clamped_axis = "longitudinal" if ay_abs <= a_max_total else "both"
            ax_safe = math.copysign(ax_clamped, ax_raw) if ax_raw != 0.0 else 0.0
            ay_safe = math.copysign(ay_clamped, ay_raw) if ay_raw != 0.0 else 0.0

        elif demand.policy == FrictionCirclePolicy.BRAKING_PRIORITY:
            # Preserve longitudinal braking capacity up to a_max_total, clamp remaining to lateral
            ax_clamped = min(ax_abs, a_max_total)
            rem_sq = max(0.0, a_max_total * a_max_total - ax_clamped * ax_clamped)
            ay_clamped = min(ay_abs, math.sqrt(rem_sq))
            clamped_axis = "lateral" if ax_abs <= a_max_total else "both"
            ax_safe = math.copysign(ax_clamped, ax_raw) if ax_raw != 0.0 else 0.0
            ay_safe = math.copysign(ay_clamped, ay_raw) if ay_raw != 0.0 else 0.0

        else:  # FrictionCirclePolicy.BALANCED
            # Proportionally scale both axes by gamma / rho
            scale = gamma / rho
            ax_safe = ax_raw * scale
            ay_safe = ay_raw * scale
            clamped_axis = "both"

    # Compute coupled steering angle bound from allocated lateral acceleration
    v = max(speed, 0.5) if math.isfinite(speed) else 0.5
    wb = max(wheelbase, 0.5) if math.isfinite(wheelbase) else 2.8
    steer_limit_kinematic = math.atan2(abs(ay_safe) * wb, v * v)
    delta_max_coupled = min(steer_limit_kinematic, steer_hw_max)

    return FrictionAllocated(
        ax_safe=ax_safe,
        ay_safe=ay_safe,
        utilisation=rho,
        is_clamped=is_clamped,
        clamped_axis=clamped_axis,
        delta_max_coupled=delta_max_coupled,
    )
