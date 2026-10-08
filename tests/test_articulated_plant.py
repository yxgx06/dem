from __future__ import annotations

import math

from egga.plant.articulated import ArticulatedParams, ArticulatedVehicle


def test_articulated_vehicle_straight_line_stability() -> None:
    veh = ArticulatedVehicle()
    s0 = veh.reset(vx=20.0, theta_a0=0.0)
    assert s0.vx == 20.0
    assert s0.theta_a == 0.0

    # Step for 1 second with 0 steer and 0 accel
    for _ in range(100):
        res = veh.step(delta=0.0, ax=0.0, dt=0.01, mu=0.8)

    assert abs(res.theta_a) < 1e-3
    assert abs(res.theta_a_dot) < 1e-3
    assert abs(res.ay1) < 1e-3
    assert res.ltr < 0.05
    assert res.state.x > 15.0


def test_articulated_vehicle_steady_turn_and_load_transfer() -> None:
    veh = ArticulatedVehicle()
    veh.reset(vx=12.0)

    # Step into steady turn
    for _ in range(150):
        res = veh.step(delta=0.04, ax=0.0, dt=0.01, mu=0.8)

    # Lateral acceleration and articulation angle should develop
    assert abs(res.ay1) > 0.5
    assert abs(res.theta_a) > 0.01
    assert 0.05 < res.ltr < 0.85
    assert math.isfinite(res.state.y)


def test_articulated_vehicle_severe_swerve_high_ltr() -> None:
    veh = ArticulatedVehicle(ArticulatedParams(cg_height2=2.8))  # Top-heavy trailer
    veh.reset(vx=22.0)

    max_ltr = 0.0
    # Aggressive high-speed steer impulse
    for step_i in range(100):
        steer = 0.08 if step_i < 30 else -0.08
        res = veh.step(delta=steer, ax=-1.0, dt=0.01, mu=0.9)
        if res.ltr > max_ltr:
            max_ltr = res.ltr

    # High load transfer ratio should be observed
    assert max_ltr > 0.35
