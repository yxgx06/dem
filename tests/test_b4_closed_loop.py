from __future__ import annotations

import numpy as np
import pytest

from egga.config import load_config
from egga.controllers.base import Observation
from egga.eval.baselines import evaluate_set, run_scenario
from egga.eval.metrics2 import MODE_NAMES, run_metrics
from egga.eval.phase5 import CONTROLLERS, aggregate, mode_table, paired_table
from egga.eval.simulate import make_controller
from egga.scenarios.mission import build_mission
from egga.scenarios.sets import load_scenarios
from egga.supervisor.types import Mode

VAL = load_scenarios("val")


def test_b4_is_deterministic_for_a_scenario() -> None:
    a = run_scenario("b4_supervised", VAL[0])
    b = run_scenario("b4_supervised", VAL[0])
    np.testing.assert_array_equal(a.ey, b.ey)
    np.testing.assert_array_equal(a.vx, b.vx)
    np.testing.assert_array_equal(a.modes, b.modes)


def test_b4_survives_scenarios_where_the_unprotected_controller_diverges() -> None:
    for index in (3, 5):  # val-003 and val-005 diverge with B0 (see docs/phase5_report.md)
        spec = VAL[index]
        assert run_scenario("b0_pid_ff", spec).diverged_at_s is not None
        b4 = run_scenario("b4_supervised", spec)
        assert b4.diverged_at_s is None
        assert np.nanmax(np.abs(b4.ey)) < 0.15


def test_b4_pays_for_caution_with_progress_and_reports_a_mode_trace() -> None:
    spec = VAL[0]
    run = run_scenario("b4_supervised", spec)
    metrics = run_metrics(run, spec["mu_min"])
    assert metrics["progress_fraction"] < 1.0
    assert np.nanmax(run.vx) <= max(spec["speed_mps"], run.vx[0]) + 1e-6
    modes = run.modes[np.isfinite(run.modes)].astype(int)
    assert modes.min() >= 0 and modes.max() < len(MODE_NAMES)
    fractions = [metrics[f"mode_{name}"] for name in MODE_NAMES]
    assert sum(fractions) == pytest.approx(1.0)


def test_ungoverned_controllers_have_no_mode_trace_and_full_progress() -> None:
    spec = VAL[2]
    run = run_scenario("b0_pid_ff", spec)
    metrics = run_metrics(run, spec["mu_min"])
    assert all(np.isnan(metrics[f"mode_{name}"]) for name in MODE_NAMES)
    assert metrics["progress_fraction"] == pytest.approx(1.0, abs=0.02)


def test_b4_without_an_estimate_falls_back_and_stays_inside_the_limits() -> None:
    mission = build_mission()
    controller = make_controller("b4_supervised", mission, 0.01)
    controller.reset()
    last = None
    for _ in range(80):
        obs = Observation(0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, vx=7.5, speed_request=7.5)
        last = controller.command(obs)
    assert last is not None and last.mode == int(Mode.FALLBACK)
    assert last.speed_cmd is not None and last.speed_cmd <= 7.5
    assert abs(last.steer) <= load_config("vehicle.yaml")["delta_max_rad"]


def test_phase5_tables_have_the_expected_shape() -> None:
    df = evaluate_set(CONTROLLERS, "val", limit=2)
    agg = aggregate(df)
    assert list(agg["controller"]) == list(CONTROLLERS)
    assert len(paired_table(df)) == len(CONTROLLERS) - 1
    modes = mode_table(df)
    assert list(modes["mode"]) == list(MODE_NAMES)
    assert modes["mean_fraction"].sum() == pytest.approx(1.0)
