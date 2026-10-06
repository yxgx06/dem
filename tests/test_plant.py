from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from egga.config import load_config
from egga.plant.actuator import TransportDelay, steer_actuator
from egga.plant.bicycle import BicyclePlant, VehicleParams

DT = 0.01


@pytest.fixture(scope="module")
def params() -> VehicleParams:
    return VehicleParams.from_config(load_config("vehicle.yaml"))


def _settle(plant: BicyclePlant, delta: float, mu: float, seconds: float = 20.0) -> None:
    for _ in range(int(seconds / DT)):
        plant.step(delta, mu, 0.0, DT)


def test_steady_state_yaw_rate_gain_matches_understeer_formula(params: VehicleParams) -> None:
    v, delta = 10.0, 0.01
    plant = BicyclePlant(params, v)
    _settle(plant, delta, mu=1.0)
    k_us = params.mass / params.wheelbase * (params.lr / params.cr - params.lf / params.cf)
    expected = v / (params.wheelbase + k_us * v**2) * delta
    assert plant.r == pytest.approx(expected, rel=0.02)


def test_understeer_gradient_is_positive_for_rear_biased_cg(params: VehicleParams) -> None:
    v = 10.0
    gains = []
    for delta in (0.005, 0.01):
        plant = BicyclePlant(params, v)
        _settle(plant, delta, mu=1.0)
        gains.append(plant.r / delta)
    neutral_gain = v / params.wheelbase
    assert gains[0] < neutral_gain
    assert gains[0] == pytest.approx(gains[1], rel=0.01)


def test_lateral_acceleration_saturates_at_mu_g(params: VehicleParams) -> None:
    mu = 0.2
    plant = BicyclePlant(params, 10.0)
    peak = 0.0
    for _ in range(int(10.0 / DT)):
        step = plant.step(0.4, mu, 0.0, DT)
        peak = max(peak, abs(step.ay))
    assert peak <= mu * params.gravity * 1.001
    assert peak > 0.5 * mu * params.gravity


def test_zero_input_stays_at_rest(params: VehicleParams) -> None:
    plant = BicyclePlant(params, 10.0)
    for _ in range(int(20.0 / DT)):
        plant.step(0.0, 0.85, 0.0, DT)
    assert plant.y == 0.0
    assert plant.psi == 0.0
    assert plant.vy == 0.0
    assert plant.r == 0.0


def test_free_response_decays(params: VehicleParams) -> None:
    plant = BicyclePlant(params, 10.0)
    plant.vy, plant.r = 0.3, 0.1
    for _ in range(int(20.0 / DT)):
        plant.step(0.0, 0.85, 0.0, DT)
    assert abs(plant.vy) < 1e-3
    assert abs(plant.r) < 1e-3


def test_plant_is_deterministic(params: VehicleParams) -> None:
    def run() -> tuple[float, float]:
        plant = BicyclePlant(params, 10.0)
        for k in range(500):
            plant.step(0.05 * np.sin(0.01 * k), 0.6, 3.0, DT)
        return plant.y, plant.r

    assert run() == run()


def test_rejects_non_positive_speed(params: VehicleParams) -> None:
    with pytest.raises(ValueError):
        BicyclePlant(params, 0.0)


@settings(max_examples=60, deadline=None)
@given(
    delta=st.floats(-0.5, 0.5),
    mu=st.floats(0.1, 1.0),
    slope=st.floats(-10.0, 10.0),
)
def test_lateral_acceleration_never_exceeds_friction_limit(
    delta: float, mu: float, slope: float
) -> None:
    p = VehicleParams.from_config(load_config("vehicle.yaml"))
    plant = BicyclePlant(p, 10.0)
    for _ in range(300):
        step = plant.step(delta, mu, slope, DT)
        assert abs(step.ay) <= mu * p.gravity * 1.02


def test_transport_delay_shifts_by_whole_steps() -> None:
    delay = TransportDelay(3)
    out = [delay.step(float(k)) for k in range(1, 8)]
    assert out == [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0]


def test_transport_delay_zero_is_passthrough() -> None:
    delay = TransportDelay(0)
    assert delay.step(1.5) == 1.5


def test_transport_delay_rejects_negative() -> None:
    with pytest.raises(ValueError):
        TransportDelay(-1)


def test_steer_actuator_applies_rate_and_angle_limits() -> None:
    assert steer_actuator(1.0, 0.0, 0.01, 0.6, 0.5) == pytest.approx(0.006)
    assert steer_actuator(1.0, 0.499, 0.01, 0.6, 0.5) == 0.5
    assert steer_actuator(-1.0, 0.0, 0.01, 0.6, 0.5) == pytest.approx(-0.006)
