from __future__ import annotations

import numpy as np
import pytest
from scipy import signal

from egga.config import load_config
from egga.controllers.derivative import FilteredDerivative
from egga.eval.closed_loop import deep_merge, load_plant_config
from egga.plant.actuator import SteeringActuator, TransportDelay, steer_actuator
from egga.plant.friction import FrictionProfile
from egga.plant.sensors import Sensor
from egga.plant.tyre import TyreParams, linear_clip_force, pacejka_force
from egga.plant.vehicle import PlantParams, Vehicle, WindParams

DT = 0.01
VEHICLE = load_config("vehicle.yaml")


def params(overrides: dict | None = None) -> PlantParams:
    return PlantParams.from_configs(VEHICLE, load_plant_config(overrides))


def settle(veh: Vehicle, delta: float, mu: float = 1.0, vx: float = 10.0, seconds: float = 15.0):
    out = None
    for k in range(int(seconds / DT)):
        out = veh.step(delta, mu, mu, 0.0, DT, vx, 0.0, k * DT)
    return out


# ---------------------------------------------------------------- tyre
TYRE = TyreParams("pacejka", 1.3, 0.0, 0.0, 100.0)


def test_pacejka_is_odd_and_zero_at_zero_slip() -> None:
    f = pacejka_force(0.05, 4000.0, 0.8, 9.5, 4000.0, TYRE)
    assert pacejka_force(-0.05, 4000.0, 0.8, 9.5, 4000.0, TYRE) == pytest.approx(-f)
    assert pacejka_force(0.0, 4000.0, 0.8, 9.5, 4000.0, TYRE) == 0.0


def test_pacejka_small_slip_stiffness_is_kc_times_load() -> None:
    a = 1e-5
    f = pacejka_force(a, 4000.0, 0.8, 9.5, 4000.0, TYRE)
    assert f / a == pytest.approx(9.5 * 4000.0, rel=1e-3)


def test_pacejka_peak_equals_mu_times_load() -> None:
    alphas = np.linspace(0.0, 1.4, 4000)
    peak = max(pacejka_force(a, 4000.0, 0.6, 9.5, 4000.0, TYRE) for a in alphas)
    assert peak == pytest.approx(0.6 * 4000.0, rel=2e-3)


def test_pacejka_load_sensitivity_lowers_total_force_under_transfer() -> None:
    p = TyreParams("pacejka", 1.3, 0.0, 0.2, 100.0)
    big = 0.5  # rad, deep saturation
    equal = 2 * pacejka_force(big, 4000.0, 0.8, 9.5, 4000.0, p)
    shifted = pacejka_force(big, 5000.0, 0.8, 9.5, 4000.0, p) + pacejka_force(
        big, 3000.0, 0.8, 9.5, 4000.0, p
    )
    assert shifted < equal


def test_pacejka_unloaded_wheel_gives_no_force() -> None:
    assert pacejka_force(0.1, 0.0, 0.8, 9.5, 4000.0, TYRE) == 0.0


def test_linear_clip_force_clips_at_mu_load() -> None:
    assert linear_clip_force(1.0, 80000.0, 5000.0, 0.5, 100.0) == 2500.0
    assert linear_clip_force(-1.0, 80000.0, 5000.0, 0.5, 100.0) == -2500.0
    assert linear_clip_force(0.001, 80000.0, 5000.0, 0.5, 100.0) == pytest.approx(80.0)


def test_unknown_tyre_model_rejected() -> None:
    veh = Vehicle(params({"tyre": {"model": "magic"}}))
    with pytest.raises(ValueError):
        veh.step(0.01, 0.8, 0.8, 0.0, DT, 10.0)


# ---------------------------------------------------------------- vehicle
def test_pacejka_steady_yaw_gain_matches_understeer_formula() -> None:
    p = params({"load_transfer": {"enabled": False, "longitudinal": False}})
    veh = Vehicle(p)
    delta, v = 0.005, 10.0
    settle(veh, delta, vx=v)
    k_us = p.mass / p.wheelbase * (p.lr / p.cr - p.lf / p.cf)
    assert veh.r == pytest.approx(v / (p.wheelbase + k_us * v**2) * delta, rel=0.02)


def test_step_response_matches_linear_state_space() -> None:
    """Pacejka small-slip response vs the exact linear 2-DOF state-space step response.

    The plant is explicit Euler, so the transient error is O(dt): it must shrink about 2x when dt
    is halved (proving it is integrator error, not a model mismatch), and the steady state matches.
    """
    p = params({"load_transfer": {"enabled": False, "longitudinal": False}})
    v, delta = 10.0, 0.002
    a = np.array(
        [
            [-(p.cf + p.cr) / (p.mass * v), (-p.cf * p.lf + p.cr * p.lr) / (p.mass * v) - v],
            [
                (-p.lf * p.cf + p.lr * p.cr) / (p.yaw_inertia * v),
                -(p.lf**2 * p.cf + p.lr**2 * p.cr) / (p.yaw_inertia * v),
            ],
        ]
    )
    b = np.array([[p.cf / p.mass], [p.lf * p.cf / p.yaw_inertia]])
    system = signal.StateSpace(a, b, np.array([[0.0, 1.0]]), [[0.0]])

    def max_error(dt: float) -> tuple[float, float]:
        t = np.arange(0.0, 3.0, dt)
        _, y, _ = signal.lsim(system, delta * np.ones_like(t), t)
        veh = Vehicle(p)
        sim = []
        for k in range(len(t)):
            veh.step(delta, 1.0, 1.0, 0.0, dt, v, 0.0, k * dt)
            sim.append(veh.r)  # state after step k, i.e. time (k + 1) * dt
        err = np.abs(np.asarray(sim)[:-1] - y[1:])
        return float(err.max() / np.abs(y).max()), float(abs(sim[-1] - y[-1]) / abs(y[-1]))

    coarse, final_coarse = max_error(0.01)
    fine, _ = max_error(0.005)
    assert coarse < 0.03
    assert final_coarse < 1e-3
    assert fine < 0.6 * coarse


def test_lateral_load_transfer_moves_load_to_outside_wheels_and_conserves_total() -> None:
    veh = Vehicle(params())
    for k in range(200):
        out = veh.step(0.05, 0.9, 0.9, 0.0, DT, 10.0, 0.0, k * DT)
    fl, fr, rl, rr = out.wheel_loads
    assert fr > fl and rr > rl  # positive ay (left turn) loads the right wheels
    p = veh.params
    assert fl + fr + rl + rr == pytest.approx(p.mass * p.gravity, rel=1e-6)


def test_longitudinal_acceleration_unloads_front_axle() -> None:
    veh = Vehicle(params())
    fl, fr, rl, rr = veh.step(0.0, 0.9, 0.9, 0.0, DT, 10.0, 3.0).wheel_loads
    base = Vehicle(params()).step(0.0, 0.9, 0.9, 0.0, DT, 10.0, 0.0).wheel_loads
    p = veh.params
    shift = p.mass * 3.0 * p.cg_height / p.wheelbase
    assert fl + fr == pytest.approx(base[0] + base[1] - shift)
    assert rl + rr == pytest.approx(base[2] + base[3] + shift)


def test_wheel_load_is_clamped_at_zero_on_lift_off() -> None:
    p = params({"load_transfer": {"track_width_m": 0.2}})
    veh = Vehicle(p)
    veh._ay_prev = 9.0
    loads = veh.step(0.0, 0.9, 0.9, 0.0, DT, 10.0).wheel_loads
    assert min(loads) == 0.0


def test_mass_scale_increases_understeer() -> None:
    gains = []
    for scale in (1.0, 1.9):
        veh = Vehicle(params({"mass": {"mass_scale": scale}, "load_transfer": {"enabled": False}}))
        settle(veh, 0.005)
        gains.append(veh.r)
    assert gains[1] < gains[0]


def test_stiffness_scale_changes_yaw_gain() -> None:
    soft = Vehicle(params({"tyre": {"stiffness_scale": 0.6}, "load_transfer": {"enabled": False}}))
    stiff = Vehicle(params({"load_transfer": {"enabled": False}}))
    settle(soft, 0.005)
    settle(stiff, 0.005)
    assert soft.r != pytest.approx(stiff.r, rel=0.01)


def test_crosswind_pushes_a_straight_vehicle_sideways() -> None:
    veh = Vehicle(params({"crosswind": {"speed_mps": 10.0}}))
    for k in range(300):
        veh.step(0.0, 0.9, 0.9, 0.0, DT, 10.0, 0.0, k * DT)
    assert veh.vy != 0.0
    calm = Vehicle(params())
    for k in range(300):
        calm.step(0.0, 0.9, 0.9, 0.0, DT, 10.0, 0.0, k * DT)
    assert calm.vy == 0.0


def test_wind_force_sign_and_gust() -> None:
    w = WindParams(8.0, 0.0, 5.0, 2.2, 1.2, 0.3)
    assert w.side_force(0.0) > 0.0
    assert WindParams(-8.0, 0.0, 5.0, 2.2, 1.2, 0.3).side_force(0.0) < 0.0
    gust = WindParams(0.0, 5.0, 4.0, 2.2, 1.2, 0.3)
    assert gust.side_force(1.0) > 0.0 > gust.side_force(3.0)


def test_variable_speed_accepts_changing_vx_and_rejects_non_positive() -> None:
    veh = Vehicle(params())
    for k in range(100):
        veh.step(0.02, 0.9, 0.9, 0.0, DT, 10.0 + 0.02 * k, 2.0, k * DT)
    assert veh.vx == pytest.approx(10.0 + 0.02 * 99)
    with pytest.raises(ValueError):
        veh.step(0.0, 0.9, 0.9, 0.0, DT, 0.0)


def test_split_mu_reduces_axle_force_at_saturation() -> None:
    uniform = Vehicle(params({"load_transfer": {"enabled": False}}))
    split = Vehicle(params({"load_transfer": {"enabled": False}}))
    a = uniform.step(0.4, 0.8, 0.8, 0.0, DT, 10.0)
    b = split.step(0.4, 0.8, 0.2, 0.0, DT, 10.0)
    assert b.fyf < a.fyf


def test_linear_clip_model_with_effects_off_is_pure_axle_model() -> None:
    p = params({"tyre": {"model": "linear_clip"}, "load_transfer": {"enabled": False,
                                                                     "longitudinal": False}})
    out = Vehicle(p).step(0.01, 0.9, 0.9, 0.0, DT, 10.0)
    assert out.fyf == pytest.approx(p.cf * 0.01)


# ---------------------------------------------------------------- actuator
def act_cfg(**kw: float) -> dict:
    return deep_merge(load_plant_config()["actuator"], kw)


def make_act(cfg: dict, seed: int = 0) -> SteeringActuator:
    return SteeringActuator(cfg, DT, 0.6, 0.5, np.random.default_rng(seed))


def test_actuator_matches_phase0_chain_for_integer_delay() -> None:
    rng = np.random.default_rng(1)
    cmds = rng.normal(0.0, 0.2, 400)
    new = make_act(act_cfg(delay_s=0.05))
    delay, prev = TransportDelay(5), 0.0
    for u in cmds:
        prev = steer_actuator(delay.step(float(u)), prev, DT, 0.6, 0.5)
        assert new.step(float(u)) == pytest.approx(prev, abs=1e-12)


def test_actuator_fractional_delay_interpolates() -> None:
    act = make_act(act_cfg(delay_s=0.025))
    outs = [act.step(0.01) for _ in range(6)]
    assert outs[0] == 0.0
    assert 0.0 < outs[2] < 0.01 or outs[2] > 0.0
    assert outs[-1] == pytest.approx(0.01, abs=1e-9) or outs[-1] > outs[2]


def test_actuator_lag_reaches_63_percent_at_one_time_constant() -> None:
    tau = 0.2
    act = make_act(act_cfg(lag_tau_s=tau))
    target = 0.05
    value = 0.0
    for _ in range(int(tau / DT)):
        value = act.step(target)
    assert value == pytest.approx(0.632 * target, rel=0.03)


def test_actuator_gain_and_bias() -> None:
    act = make_act(act_cfg(gain=0.8, bias_rad=0.01))
    value = 0.0
    for _ in range(300):
        value = act.step(0.1)
    assert value == pytest.approx(0.8 * 0.1 + 0.01)


def test_actuator_rate_and_angle_limits() -> None:
    act = make_act(act_cfg())
    assert act.step(1.0) == pytest.approx(0.6 * DT)
    for _ in range(300):
        v = act.step(1.0)
    assert v == 0.5


def test_actuator_jitter_is_bounded_reproducible_and_seeded() -> None:
    cfg = act_cfg(delay_s=0.03, jitter_s=0.04)
    cmds = np.sin(np.linspace(0, 6, 300)) * 0.1
    a = [make_act(cfg, 7).step(float(u)) for u in cmds]
    first = make_act(cfg, 7)
    second = make_act(cfg, 7)
    other = make_act(cfg, 8)
    ra = [first.step(float(u)) for u in cmds]
    rb = [second.step(float(u)) for u in cmds]
    rc = [other.step(float(u)) for u in cmds]
    assert ra == rb
    assert ra != rc
    assert len(a) == 300


def test_actuator_rejects_negative_parameters() -> None:
    with pytest.raises(ValueError):
        make_act(act_cfg(delay_s=-0.01))


def test_transport_delay_rejects_negative_steps() -> None:
    with pytest.raises(ValueError):
        TransportDelay(-2)


# ---------------------------------------------------------------- sensors
def sens_cfg(**kw: float) -> dict:
    base = {"noise_std": 0.0, "bias": 0.0, "quant_step": 0.0, "dropout_prob": 0.0,
            "delay_steps": 0}
    base.update(kw)
    return base


def test_sensor_ideal_passes_value() -> None:
    m = Sensor(sens_cfg(), np.random.default_rng(0)).measure(1.25)
    assert m.value == 1.25 and m.valid


def test_sensor_noise_std_and_bias() -> None:
    s = Sensor(sens_cfg(noise_std=0.1, bias=0.5), np.random.default_rng(3))
    vals = np.array([s.measure(0.0).value for _ in range(20000)])
    assert vals.mean() == pytest.approx(0.5, abs=0.01)
    assert vals.std() == pytest.approx(0.1, rel=0.03)


def test_sensor_quantisation_lands_on_grid() -> None:
    s = Sensor(sens_cfg(quant_step=0.05), np.random.default_rng(0))
    for x in (0.011, 0.037, -0.121):
        v = s.measure(x).value
        assert v / 0.05 == pytest.approx(round(v / 0.05))


def test_sensor_dropout_rate_and_hold() -> None:
    s = Sensor(sens_cfg(dropout_prob=0.2), np.random.default_rng(5))
    results = [s.measure(float(k)) for k in range(20000)]
    frac = 1.0 - np.mean([r.valid for r in results])
    assert frac == pytest.approx(0.2, abs=0.01)
    last_good = results[0].value
    for cur in results[1:]:
        if cur.valid:
            last_good = cur.value
        else:
            assert cur.value == last_good


def test_sensor_sample_delay() -> None:
    s = Sensor(sens_cfg(delay_steps=3), np.random.default_rng(0))
    out = [s.measure(float(k)).value for k in range(8)]
    assert out[3:] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_sensor_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        Sensor(sens_cfg(dropout_prob=1.5), np.random.default_rng(0))
    with pytest.raises(ValueError):
        Sensor(sens_cfg(delay_steps=-1), np.random.default_rng(0))


# ---------------------------------------------------------------- friction
NO_SPLIT = {"enabled": False, "left_factor": 1.0, "right_factor": 1.0}


def test_friction_profiles() -> None:
    const = FrictionProfile.from_config({"type": "constant", "mu": 0.7}, NO_SPLIT)
    assert const.mu(5.0, 100.0) == 0.7
    step = FrictionProfile.from_config(
        {"type": "step", "t_s": 2.0, "before": 0.9, "after": 0.3}, NO_SPLIT
    )
    assert step.mu(1.9, 0) == 0.9 and step.mu(2.0, 0) == 0.3
    ramp = FrictionProfile.from_config(
        {"type": "ramp", "t0_s": 0.0, "t1_s": 10.0, "before": 0.8, "after": 0.4}, NO_SPLIT
    )
    assert ramp.mu(5.0, 0) == pytest.approx(0.6) and ramp.mu(99.0, 0) == pytest.approx(0.4)
    patches = FrictionProfile.from_config(
        {"type": "patches", "default": 0.8, "patches": [{"s0_m": 10, "s1_m": 20, "mu": 0.2}]},
        NO_SPLIT,
    )
    assert patches.mu(0, 15.0) == 0.2 and patches.mu(0, 25.0) == 0.8
    arr = FrictionProfile.from_array(np.array([0.9, 0.5, 0.1]), 1.0)
    assert arr.mu(1.0, 0) == 0.5 and arr.mu(50.0, 0) == 0.1


def test_split_mu_scales_each_side() -> None:
    f = FrictionProfile.from_config(
        {"type": "constant", "mu": 0.8}, {"enabled": True, "left_factor": 1.0, "right_factor": 0.5}
    )
    assert f.mu_wheels(0, 0) == (0.8, 0.4)


def test_unknown_friction_profile_rejected() -> None:
    with pytest.raises(ValueError):
        FrictionProfile.from_config({"type": "oil"}, NO_SPLIT)


# ---------------------------------------------------------------- derivative filter
def test_filtered_derivative_recovers_ramp_slope() -> None:
    f = FilteredDerivative(5.0, DT)
    y = 0.0
    for k in range(400):
        y = f.update(2.0 * k * DT)
    assert y == pytest.approx(2.0, rel=1e-3)


def test_filtered_derivative_attenuates_noise_versus_finite_difference() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 0.02, 5000)
    raw = np.diff(x) / DT
    f = FilteredDerivative(2.0, DT)
    filt = np.array([f.update(float(v)) for v in x])
    assert filt.std() < 0.3 * raw.std()


def test_filtered_derivative_first_sample_is_zero_and_reset_works() -> None:
    f = FilteredDerivative(5.0, DT)
    assert f.update(3.0) == 0.0
    f.update(4.0)
    f.reset()
    assert f.update(10.0) == 0.0


def test_filtered_derivative_rejects_bad_parameters() -> None:
    with pytest.raises(ValueError):
        FilteredDerivative(0.0, DT)
