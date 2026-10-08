from __future__ import annotations

import sys

import pytest

from egga.config import REPO_ROOT

sys.path.insert(0, str(REPO_ROOT / "tooling" / "ros2" / "egga_supervisor_ros2"))

from egga_supervisor_ros2.supervisor_node import EggaSupervisorController  # noqa: E402


def test_ros2_supervisor_controller_nominal_command() -> None:
    ctrl = EggaSupervisorController(wheelbase=2.8, steer_hw_max=0.60)
    ctrl.set_friction_estimate(0.85)

    # Benign demand: 10 m/s, slight turn
    res = ctrl.filter_command(
        v_req=10.0,
        omega_req=0.05,
        current_speed=10.0,
    )

    assert not res["is_friction_clamped"]
    assert not res["is_jackknife_critical"]
    assert not res["is_rollover_critical"]
    assert not res["emergency_stop"]
    assert res["linear_velocity_safe"] == pytest.approx(10.0)
    assert res["angular_velocity_safe"] > 0.0


def test_ros2_supervisor_controller_extreme_command_clamping() -> None:
    ctrl = EggaSupervisorController(wheelbase=2.8, steer_hw_max=0.60)
    ctrl.set_friction_estimate(0.3)  # Slippery surface

    # Harsh braking + aggressive swerve
    res = ctrl.filter_command(
        v_req=0.0,
        omega_req=0.80,
        current_speed=20.0,
        theta_a=0.30,
        theta_a_dot=0.40,
        ltr=0.75,
        dt=0.01,
    )

    # Friction circle and/or jackknife barrier should intervene
    assert res["is_friction_clamped"] or res["is_jackknife_critical"]
    assert res["trailer_brake_pressure"] > 0.0
    assert abs(res["steering_angle_safe"]) <= 0.60
