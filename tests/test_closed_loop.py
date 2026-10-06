from __future__ import annotations

import numpy as np
import pytest

from egga.config import load_config
from egga.controllers.classical import PDFeedforwardController, PIDController
from egga.eval.closed_loop import (
    deep_merge,
    load_case,
    load_plant_config,
    run_closed_loop,
    speed_profile,
)
from egga.eval.simulate import Scenario, run_mission


@pytest.mark.parametrize("controller", ["pid", "pd_ff", "rl_handtyped", "rl_trained"])
def test_linear_equiv_case_reproduces_phase0_exactly(controller: str) -> None:
    new = run_closed_loop(controller, load_case("linear_equiv"))
    old = run_mission(controller, Scenario())
    np.testing.assert_array_equal(new.ey, old.ey)


def test_closed_loop_is_deterministic_with_noise_and_dropout() -> None:
    cfg = load_case("sensor_dropout_5pct")
    a = run_closed_loop("pd_ff", cfg, seed=4)
    b = run_closed_loop("pd_ff", cfg, seed=4)
    c = run_closed_loop("pd_ff", cfg, seed=5)
    np.testing.assert_array_equal(a.ey, b.ey)
    assert a.meas_valid_fraction == b.meas_valid_fraction
    assert a.meas_valid_fraction != c.meas_valid_fraction


def test_dropout_case_reports_valid_fraction_near_expected() -> None:
    run = run_closed_loop("pd_ff", load_case("sensor_dropout_5pct"), seed=0)
    assert run.meas_valid_fraction == pytest.approx(0.95, abs=0.01)


def test_new_effects_degrade_tracking_versus_nominal() -> None:
    nominal = np.nanmax(np.abs(run_closed_loop("pd_ff", load_case("nominal")).ey))
    for case in ("delay_80ms", "gain_loss_20pct", "steer_bias_1deg", "mass_x1.9"):
        worse = np.nanmax(np.abs(run_closed_loop("pd_ff", load_case(case)).ey))
        assert worse > nominal, case


def test_infeasible_speed_ramp_diverges_and_split_mu_is_a_known_failure() -> None:
    assert run_closed_loop("pd_ff", load_case("speed_ramp")).diverged_at_s is not None
    assert run_closed_loop("pd_ff", load_case("split_mu")).diverged_at_s is not None


def test_nominal_new_plant_stays_bounded_for_pd_ff() -> None:
    run = run_closed_loop("pd_ff", load_case("nominal"))
    assert run.diverged_at_s is None
    assert np.nanmax(np.abs(run.ey)) < 0.05


def test_speed_profiles() -> None:
    cfg = load_plant_config()
    n = 7501
    assert np.all(speed_profile(cfg, n, 0.01) == 10.0)
    ramp = speed_profile(load_case("speed_ramp"), n, 0.01)
    assert ramp[0] == 10.0 and ramp[-1] == pytest.approx(12.0)
    with pytest.raises(ValueError):
        speed_profile(deep_merge(cfg, {"speed": {"profile": "zigzag"}}), n, 0.01)


def test_unknown_case_rejected() -> None:
    with pytest.raises(KeyError):
        load_case("nope")


def test_deep_merge_does_not_mutate_inputs() -> None:
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    out = deep_merge(base, {"a": {"b": 9}})
    assert out == {"a": {"b": 9, "c": 2}, "d": 3}
    assert base["a"]["b"] == 1


def test_filtered_derivative_is_used_when_cutoff_set() -> None:
    cfg = load_config("controllers.yaml")
    pid = PIDController(cfg["pid"], 0.01, derivative_cutoff_hz=2.0)
    from egga.controllers.base import Observation

    obs = Observation(0.1, 99.0, 0.0, 0.0, 0.0, 0.0, 0.8)
    assert pid.command(obs).steer == pytest.approx(0.8 * 0.1 + 0.05 * 0.001, abs=1e-6)


def test_pd_requires_dt_when_filtering() -> None:
    cfg = load_config("controllers.yaml")["pd_ff"]
    with pytest.raises(ValueError):
        PDFeedforwardController(cfg, 2.8, 10.0, dt=None, derivative_cutoff_hz=3.0)


def test_controller_uses_observation_speed_for_feedforward() -> None:
    from egga.controllers.base import Observation

    cfg = load_config("controllers.yaml")["pd_ff"]
    ctrl = PDFeedforwardController(cfg, 2.8, 10.0)
    slow = ctrl.command(Observation(0, 0, 0, 0, 0.3, 0, 0.8, vx=5.0)).steer
    nominal = ctrl.command(Observation(0, 0, 0, 0, 0.3, 0, 0.8, vx=0.0)).steer
    assert slow == pytest.approx(2.8 / 5.0 * 0.3)
    assert nominal == pytest.approx(2.8 / 10.0 * 0.3)
