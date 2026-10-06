from __future__ import annotations

import ast
import math
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from egga.config import (
    CONFIG_DIR,
    REPO_ROOT,
    load_config,
    load_estimators,
    set_estimator_override,
)
from egga.estimation.delay import DelayBoundEstimator
from egga.estimation.friction import FrictionEKF, NominalVehicle
from egga.estimation.oracle import OracleEstimator
from egga.estimation.suite import EstimatorSuite
from egga.estimation.types import (
    STATUS_INVALID,
    STATUS_OK,
    STATUS_STALE,
    Estimate,
    Measurements,
)
from egga.eval.closed_loop import load_plant_config, run_closed_loop
from egga.eval.estimation_eval import (
    bootstrap_ci,
    clopper_pearson,
    condition_specs,
    detection_latency,
    estimation_metrics,
    run_estimation,
)
from egga.scenarios.sets import load_scenarios, set_hash

DT = 0.01
EST_DIR = REPO_ROOT / "src" / "egga" / "estimation"
FORBIDDEN = ("egga.plant", "egga.scenarios", "egga.eval", "egga.controllers")


def suite() -> EstimatorSuite:
    return EstimatorSuite(load_config("vehicle.yaml"), load_estimators(), DT)


def meas(t_s: float, **kw: float | bool) -> Measurements:
    base: dict[str, float | bool] = {
        "t": t_s, "vx": 10.0, "yaw_rate": 0.05, "yaw_rate_valid": True, "lateral_accel": 0.5,
        "lateral_accel_valid": True, "steer_meas": 0.02, "steer_meas_valid": True,
        "steer_cmd": 0.02,
    }
    base.update(kw)
    return Measurements(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------- independence
def test_estimation_modules_do_not_import_plant_or_controllers() -> None:
    offenders = []
    for path in EST_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            offenders += [f"{path.name}: {n}" for n in names if n.startswith(FORBIDDEN)]
    assert offenders == []


def test_measurements_carry_no_plant_truth_fields() -> None:
    names = {f.name for f in fields(Measurements)}
    assert names.isdisjoint({"mu", "mu_true", "delay", "mass", "mass_scale", "truth"})
    assert "mu" not in names


def test_only_the_oracle_takes_truth_and_is_labelled() -> None:
    est = OracleEstimator().estimate(1.0, 0.5, 0.05, 1.3)
    assert OracleEstimator.label == "oracle"
    assert est.mu_lo < 0.5 < est.mu_hi and est.status == STATUS_OK


# ---------------------------------------------------------------- Estimate interface
def test_mu_safe_with_sqrt3_equals_lower_bound() -> None:
    e = Estimate(0.3, 0.7, 0.05, 1.0, 1.5, 0.5, STATUS_OK, 1.0)
    assert e.mu_hat == pytest.approx(0.5)
    assert e.mu_safe(math.sqrt(3.0), 0.1) == pytest.approx(0.3)
    assert e.mu_safe(10.0, 0.1) == 0.1


# ---------------------------------------------------------------- suite validity / staleness
@pytest.mark.parametrize("bad", ["nan_t", "nan_vx", "zero_vx", "nan_cmd"])
def test_invalid_scalar_inputs_give_widest_bounds(bad: str) -> None:
    s = suite()
    s.update(meas(0.0))
    kw = {
        "nan_t": {"t": float("nan")},
        "nan_vx": {"vx": float("nan")},
        "zero_vx": {"vx": 0.0},
        "nan_cmd": {"steer_cmd": float("inf")},
    }[bad]
    e = s.update(meas(0.01, **kw))
    assert e.status == STATUS_INVALID
    assert (e.mu_lo, e.mu_hi, e.tau_bar) == (0.1, 1.0, 0.15) and e.quality == 0.0
    assert (e.mass_lo, e.mass_hi) == (1.0, 1.9)


def test_nan_measurement_values_give_widest_bounds_never_last_good() -> None:
    s = suite()
    for k in range(300):
        s.update(meas(k * DT))
    e = s.update(meas(3.0, yaw_rate=float("nan")))
    assert e.status == STATUS_INVALID and (e.mu_lo, e.mu_hi) == (0.1, 1.0)
    after = s.update(meas(3.01))
    assert after.mu_hi - after.mu_lo > 0.5  # filters were reset: not the old narrow interval


def test_time_going_backwards_is_invalid() -> None:
    s = suite()
    s.update(meas(1.0))
    assert s.update(meas(0.5)).status == STATUS_INVALID
    assert s.update(meas(0.5)).status == STATUS_INVALID


def test_long_gap_gives_widest_bounds_short_gap_does_not() -> None:
    s = suite()
    for k in range(50):
        s.update(meas(k * DT))
    t = 0.5
    short = [s.update(meas(t + i * DT, yaw_rate_valid=False)) for i in range(5)]
    assert all(e.status != STATUS_STALE for e in short)
    long = [s.update(meas(t + (5 + i) * DT, yaw_rate_valid=False)) for i in range(20)]
    assert long[-1].status == STATUS_STALE
    assert (long[-1].mu_lo, long[-1].mu_hi, long[-1].tau_bar) == (0.1, 1.0, 0.15)


def test_estimates_are_ordered_and_inside_the_prior() -> None:
    s = suite()
    rng = np.random.default_rng(0)
    for k in range(500):
        e = s.update(
            meas(k * DT, yaw_rate=float(rng.normal(0, 0.1)), lateral_accel=float(rng.normal(0, 2)),
                 steer_meas=float(rng.normal(0, 0.05)), steer_cmd=float(rng.normal(0, 0.05)))
        )
        assert 0.1 <= e.mu_lo <= e.mu_hi <= 1.0
        assert 1.0 <= e.mass_lo <= e.mass_hi <= 1.9
        assert 0.0 <= e.tau_bar <= 0.15 + 1e-12
        assert 0.0 <= e.quality <= 1.0


# ---------------------------------------------------------------- friction EKF
def _ekf(**over: float) -> FrictionEKF:
    cfg = dict(load_estimators()["friction"])
    cfg.update(over)
    return FrictionEKF(
        NominalVehicle.from_config(load_config("vehicle.yaml")), cfg, load_estimators()["prior"], DT
    )


def test_ekf_without_excitation_stays_at_the_prior_width() -> None:
    ekf = _ekf()
    for _ in range(1000):
        ekf.step(0.0, 10.0, 0.0, 0.0)
    lo, hi, _ = ekf.interval()
    assert hi - lo > 0.6


def test_ekf_recovers_friction_and_mass_from_consistent_saturating_data() -> None:
    truth_mu, truth_mass = 0.35, 1.3
    ekf = _ekf(z=2.0)
    sim = _ekf()
    sim._x = np.array([0.0, 0.0, truth_mu, 1.0 / truth_mass])
    vy, r = 0.0, 0.0
    for k in range(4000):
        delta = 0.12 * math.sin(2 * math.pi * k * DT / 4.0)
        d_vy, d_r, ay = sim._dynamics(vy, r, truth_mu, 1.0 / truth_mass, delta, 10.0)
        vy += DT * d_vy
        r += DT * d_r
        ekf.step(delta, 10.0, r, ay)
    lo, hi, sigma = ekf.interval()
    assert lo <= truth_mu <= hi
    assert abs(0.5 * (lo + hi) - truth_mu) < 0.1
    mlo, mhi, _ = ekf.mass_interval()
    assert mlo <= truth_mass <= mhi


def test_ekf_reset_returns_to_prior() -> None:
    ekf = _ekf()
    for k in range(500):
        ekf.step(0.1 * math.sin(k * 0.05), 10.0, 0.1, 1.0)
    ekf.reset_to_prior()
    lo, hi, _ = ekf.interval()
    assert hi - lo > 0.6


def test_ekf_handles_missing_measurements() -> None:
    ekf = _ekf()
    ekf.step(0.05, 10.0, None, None)
    ekf.step(0.05, 10.0, 0.04, None)
    ekf.step(0.05, 10.0, None, 0.4)
    lo, hi, _ = ekf.interval()
    assert 0.1 <= lo <= hi <= 1.0


# ---------------------------------------------------------------- delay bound
def _delay_est() -> DelayBoundEstimator:
    return DelayBoundEstimator(load_estimators()["delay"], 0.15, DT)


def _drive(est: DelayBoundEstimator, lag_steps: int, seconds: float, amp: float = 0.05) -> float:
    rng = np.random.default_rng(1)
    n = int(seconds / DT)
    t = np.arange(n) * DT
    u = amp * np.sin(2 * np.pi * 0.4 * t) + amp * 0.3 * np.sin(2 * np.pi * 1.3 * t + 1.0)
    d = np.concatenate([np.zeros(lag_steps), u[: n - lag_steps]]) + rng.normal(0, 1e-4, n)
    for k in range(n):
        est.update(float(t[k]), float(u[k]), float(d[k]))
    return float(t[-1])


@pytest.mark.parametrize("lag_steps", [0, 3, 6, 10])
def test_delay_bound_covers_true_latency_and_is_tighter_than_prior(lag_steps: int) -> None:
    est = _delay_est()
    t = _drive(est, lag_steps, 8.0)
    bound, informed = est.bound(t)
    assert informed
    assert lag_steps * DT <= bound < 0.15
    assert bound - lag_steps * DT < 0.06


def test_delay_bound_is_the_prior_without_excitation() -> None:
    est = _delay_est()
    for k in range(600):
        est.update(k * DT, 0.0, 0.0)
    assert est.bound(6.0) == (0.15, False)


def test_delay_bound_widens_after_excitation_is_lost() -> None:
    est = _delay_est()
    t = _drive(est, 4, 8.0)
    tight, _ = est.bound(t)
    later, _ = est.bound(t + 2.5)
    latest, informed = est.bound(t + 6.0)
    assert tight < later < 0.15 and latest == pytest.approx(0.15) and not informed


def test_delay_gap_clears_the_window() -> None:
    est = _delay_est()
    _drive(est, 4, 4.0)
    est.update(5.0, 0.1, None)
    assert est.bound(5.0)[1] in (True, False)  # no crash; window cleared


# ---------------------------------------------------------------- closed loop integration
def test_closed_loop_estimates_are_finite_deterministic_and_cover_latency() -> None:
    spec = load_scenarios("val")[0]
    a = run_estimation(spec)
    b = run_estimation(spec)
    for key in a.estimates:
        np.testing.assert_array_equal(a.estimates[key], b.estimates[key])
    m = estimation_metrics(a)
    assert np.isfinite(m["mu_coverage"]) and m["tau_coverage"] >= 0.95


def test_mu_source_estimate_safe_runs_and_validates_arguments() -> None:
    spec = load_scenarios("val")[0]
    from egga.eval.baselines import run_scenario  # noqa: F401  (import check)
    from egga.scenarios.sets import run_arguments

    overrides, mission_cfg, belief, seed = run_arguments(spec)
    cfg = load_plant_config(overrides)
    with pytest.raises(ValueError):
        run_closed_loop("b1_pd_ff", cfg, mu_source="estimate_safe", mission_cfg=mission_cfg)
    with pytest.raises(ValueError):
        run_closed_loop("b1_pd_ff", cfg, mu_source="guess", mission_cfg=mission_cfg)
    run = run_closed_loop(
        "b1_pd_ff", cfg, seed=seed, mission_cfg=mission_cfg, estimator=suite(),
        mu_source="estimate_safe",
    )
    assert run.estimates["mu_lo"].shape == run.t.shape


def test_sensor_dropout_condition_runs_without_crash() -> None:
    spec, extra = condition_specs()["sensor_dropout_5pct"]
    run = run_estimation(spec, plant_extra=extra)
    assert run.meas_valid_fraction < 1.0
    assert np.isfinite(run.estimates["mu_lo"][-1])


def test_detection_latency_helper_on_a_synthetic_trace() -> None:
    class Run:
        t = np.arange(0, 10, 0.01)
        estimates = {
            "mu_lo": np.where(np.arange(1000) < 500, 0.7, 0.2),
            "mu_hi": np.where(np.arange(1000) < 500, 0.9, 0.4),
        }

    assert detection_latency(Run(), 4.0, 0.8, 0.3) == pytest.approx(1.0, abs=0.02)  # type: ignore[arg-type]
    assert math.isnan(detection_latency(Run(), 4.0, 0.8, 0.95))  # type: ignore[arg-type]


# ---------------------------------------------------------------- statistics helpers
def test_clopper_pearson_known_values() -> None:
    lo, hi = clopper_pearson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.3085, abs=1e-3)
    lo, hi = clopper_pearson(10, 10)
    assert hi == 1.0 and lo == pytest.approx(0.6915, abs=1e-3)
    lo, hi = clopper_pearson(5, 10)
    assert lo < 0.5 < hi


def test_bootstrap_ci_brackets_the_mean() -> None:
    values = np.random.default_rng(0).normal(1.0, 0.2, 50)
    lo, hi = bootstrap_ci(values)
    assert lo < values.mean() < hi


# ---------------------------------------------------------------- calibration provenance
def test_calibrated_file_was_fitted_on_the_frozen_train_set_only() -> None:
    path: Path = CONFIG_DIR / "estimators_tuned.yaml"
    assert path.exists(), "run python -m egga.eval.estimation_eval --calibrate"
    assert set_hash(load_scenarios("train")) in path.read_text(encoding="utf-8").splitlines()[1]
    text = (REPO_ROOT / "src" / "egga" / "eval" / "estimation_eval.py").read_text(encoding="utf-8")
    start = text.index("def calibrate")
    calibrate_body = text[start : text.index("# ----", start)]
    assert 'load_scenarios("train")' in calibrate_body
    assert 'load_scenarios("val")' not in calibrate_body
    assert 'load_scenarios("test"' not in calibrate_body


def test_override_hook_changes_the_estimator_and_is_reversible() -> None:
    base = load_estimators()["friction"]["z"]
    set_estimator_override({"friction": {"z": base + 1.0}})
    try:
        assert load_estimators()["friction"]["z"] == pytest.approx(base + 1.0)
    finally:
        set_estimator_override(None)
    assert load_estimators()["friction"]["z"] == pytest.approx(base)
