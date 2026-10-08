from __future__ import annotations

import math

import pytest

from egga.supervisor.friction_circle import (
    FrictionCirclePolicy,
    FrictionDemand,
    allocate_friction_circle,
)


def test_within_friction_circle_unclamped() -> None:
    # 1.0 m/s^2 long, 2.0 m/s^2 lat on mu = 0.8 surface (mu * g ~ 7.84, rho ~ 0.285 < 0.95)
    demand = FrictionDemand(
        ax_req=1.0,
        ay_req=-2.0,
        mu=0.8,
        gravity=9.81,
        safety_factor=0.95,
        policy=FrictionCirclePolicy.STEERING_PRIORITY,
    )
    alloc = allocate_friction_circle(demand, speed=10.0, wheelbase=2.8)

    assert not alloc.is_clamped
    assert alloc.clamped_axis == "none"
    assert alloc.ax_safe == pytest.approx(1.0)
    assert alloc.ay_safe == pytest.approx(-2.0)
    assert alloc.utilisation < 0.95
    assert alloc.delta_max_coupled > 0.0


def test_steering_priority_clamps_longitudinal() -> None:
    # Extreme demand: ax = 6.0 m/s^2, ay = 5.0 m/s^2 on mu = 0.8 (mu*g = 7.848, a_max = 7.455)
    demand = FrictionDemand(
        ax_req=-6.0,
        ay_req=5.0,
        mu=0.8,
        safety_factor=0.95,
        policy=FrictionCirclePolicy.STEERING_PRIORITY,
    )
    alloc = allocate_friction_circle(demand, speed=12.0, wheelbase=2.8)

    assert alloc.is_clamped
    assert alloc.clamped_axis == "longitudinal"
    assert alloc.ay_safe == pytest.approx(5.0)  # Preserves lateral demand
    assert abs(alloc.ax_safe) < 6.0  # Long demand clamped
    # Verify exact point lies on friction circle boundary
    a_max = 0.95 * 0.8 * 9.80665
    hyp = math.hypot(alloc.ax_safe, alloc.ay_safe)
    assert hyp == pytest.approx(a_max, rel=1e-5)


def test_steering_priority_extreme_lateral_clamps_both() -> None:
    # Lateral demand alone exceeds a_max: ay = 10.0 m/s^2 on mu = 0.8 (a_max ~ 7.45)
    demand = FrictionDemand(
        ax_req=2.0,
        ay_req=10.0,
        mu=0.8,
        safety_factor=0.95,
        policy=FrictionCirclePolicy.STEERING_PRIORITY,
    )
    alloc = allocate_friction_circle(demand, speed=10.0, wheelbase=2.8)

    assert alloc.is_clamped
    assert alloc.clamped_axis == "both"
    a_max = 0.95 * 0.8 * 9.80665
    assert alloc.ay_safe == pytest.approx(a_max)
    assert alloc.ax_safe == pytest.approx(0.0)  # Zero remaining capacity for longitudinal


def test_braking_priority_clamps_lateral() -> None:
    # ax = 6.0 m/s^2, ay = 5.0 m/s^2 under BRAKING_PRIORITY
    demand = FrictionDemand(
        ax_req=-6.0,
        ay_req=5.0,
        mu=0.8,
        safety_factor=0.95,
        policy=FrictionCirclePolicy.BRAKING_PRIORITY,
    )
    alloc = allocate_friction_circle(demand, speed=10.0, wheelbase=2.8)

    assert alloc.is_clamped
    assert alloc.clamped_axis == "lateral"
    assert alloc.ax_safe == pytest.approx(-6.0)  # Preserves braking demand
    assert abs(alloc.ay_safe) < 5.0  # Steer demand clamped
    a_max = 0.95 * 0.8 * 9.80665
    hyp = math.hypot(alloc.ax_safe, alloc.ay_safe)
    assert hyp == pytest.approx(a_max, rel=1e-5)


def test_braking_priority_extreme_braking_clamps_both() -> None:
    # Longitudinal demand alone exceeds a_max: ax = -12.0 m/s^2 on mu = 0.8
    demand = FrictionDemand(
        ax_req=-12.0,
        ay_req=3.0,
        mu=0.8,
        safety_factor=0.95,
        policy=FrictionCirclePolicy.BRAKING_PRIORITY,
    )
    alloc = allocate_friction_circle(demand, speed=10.0, wheelbase=2.8)

    assert alloc.is_clamped
    assert alloc.clamped_axis == "both"
    a_max = 0.95 * 0.8 * 9.80665
    assert alloc.ax_safe == pytest.approx(-a_max)
    assert alloc.ay_safe == pytest.approx(0.0)


def test_balanced_policy_proportional_scaling() -> None:
    # ax = 6.0, ay = 8.0 (vector magnitude 10.0) on mu = 0.5 (a_max = 0.95 * 0.5 * 9.80665 ~ 4.658)
    demand = FrictionDemand(
        ax_req=6.0,
        ay_req=8.0,
        mu=0.5,
        safety_factor=0.95,
        policy=FrictionCirclePolicy.BALANCED,
    )
    alloc = allocate_friction_circle(demand, speed=8.0, wheelbase=2.5)

    assert alloc.is_clamped
    assert alloc.clamped_axis == "both"
    # Ratio between ax and ay must be preserved (6 / 8 = 0.75)
    assert (alloc.ax_safe / alloc.ay_safe) == pytest.approx(6.0 / 8.0)
    a_max = 0.95 * 0.5 * 9.80665
    assert math.hypot(alloc.ax_safe, alloc.ay_safe) == pytest.approx(a_max, rel=1e-5)


def test_edge_cases_and_defensive_fallbacks() -> None:
    # 1. Invalid gravity, mu, and nan demands
    demand_invalid = FrictionDemand(
        ax_req=float("nan"),
        ay_req=float("inf"),
        mu=-0.5,  # negative mu
        gravity=-9.8,  # negative gravity
        safety_factor=1.5,  # > 1.0
        policy=FrictionCirclePolicy.STEERING_PRIORITY,
    )
    alloc = allocate_friction_circle(
        demand_invalid,
        speed=-5.0,  # negative speed
        wheelbase=-1.0,  # negative wheelbase
        steer_hw_max=0.5,
    )
    assert math.isfinite(alloc.ax_safe)
    assert math.isfinite(alloc.ay_safe)
    assert math.isfinite(alloc.delta_max_coupled)
    assert alloc.delta_max_coupled <= 0.5

    # 2. Safety factor below 0.1
    demand_low_gamma = FrictionDemand(
        ax_req=0.0,
        ay_req=0.0,
        mu=float("nan"),
        safety_factor=0.01,
        policy=FrictionCirclePolicy.BRAKING_PRIORITY,
    )
    alloc2 = allocate_friction_circle(demand_low_gamma, speed=float("nan"), wheelbase=float("nan"))
    assert alloc2.ax_safe == 0.0
    assert alloc2.ay_safe == 0.0

    # 3. High steer kinematic angle clamped by steer_hw_max
    demand_high_steer = FrictionDemand(
        ax_req=0.0,
        ay_req=7.0,
        mu=0.9,
    )
    alloc3 = allocate_friction_circle(
        demand_high_steer, speed=1.0, wheelbase=3.0, steer_hw_max=0.35
    )
    assert alloc3.delta_max_coupled == pytest.approx(0.35)
