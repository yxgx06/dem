from __future__ import annotations

import math

import pytest

from egga.supervisor.jackknife_guard import (
    FLAG_JACKKNIFE_CRITICAL,
    FLAG_JACKKNIFE_MARGIN,
    FLAG_JACKKNIFE_OK,
    FLAG_ROLLOVER_WARNING,
    JackknifeConfig,
    JackknifeInputs,
    compute_critical_articulation_angle,
    step_jackknife_guard,
)


def test_jackknife_guard_nominal_safe_state() -> None:
    cfg = JackknifeConfig()
    inp = JackknifeInputs(
        theta_a=0.02,
        theta_a_dot=0.01,
        vx=15.0,
        mu=0.8,
        ltr=0.20,
        steer_cmd_req=0.05,
        steer_rate_req=0.10,
    )
    out = step_jackknife_guard(cfg, inp)

    assert out.status_flags == FLAG_JACKKNIFE_OK
    assert not out.is_jackknife_critical
    assert not out.is_rollover_critical
    assert out.trailer_brake_pressure == 0.0
    assert out.steer_cmd_safe == pytest.approx(0.05)
    assert out.steer_rate_safe == pytest.approx(0.10)
    assert out.h_jackknife > 0.08


def test_jackknife_guard_margin_warning_applies_trailer_drag() -> None:
    cfg = JackknifeConfig()
    # Critical angle at 20 m/s, mu=0.3 is small: theta_crit ~ 0.20 rad
    # Set theta_a close to theta_crit so 0 < h_jackknife < 0.08
    inp = JackknifeInputs(
        theta_a=0.15,
        theta_a_dot=0.05,
        vx=20.0,
        mu=0.3,
        ltr=0.30,
        steer_cmd_req=0.05,
        steer_rate_req=0.10,
    )
    out = step_jackknife_guard(cfg, inp)

    assert out.status_flags & FLAG_JACKKNIFE_MARGIN
    assert not out.is_jackknife_critical
    assert 0.0 < out.trailer_brake_pressure <= 1.0


def test_jackknife_guard_critical_intervention_and_steer_clamping() -> None:
    cfg = JackknifeConfig()
    # High articulation angle and rate exceeding critical angle
    inp = JackknifeInputs(
        theta_a=0.35,
        theta_a_dot=0.50,
        vx=18.0,
        mu=0.4,
        ltr=0.40,
        steer_cmd_req=0.20,  # Same direction as theta_a > 0
        steer_rate_req=0.50,
    )
    out = step_jackknife_guard(cfg, inp)

    assert out.is_jackknife_critical
    assert out.status_flags & FLAG_JACKKNIFE_CRITICAL
    assert out.trailer_brake_pressure == 1.0
    assert out.steer_cmd_safe == pytest.approx(0.10)  # Clamped by 0.5
    assert abs(out.steer_rate_safe) < 0.50

    # Negative articulation and negative steer command (same sign)
    inp_neg = JackknifeInputs(
        theta_a=-0.35,
        theta_a_dot=-0.50,
        vx=18.0,
        mu=0.4,
        ltr=0.40,
        steer_cmd_req=-0.20,
        steer_rate_req=0.50,
    )
    out_neg = step_jackknife_guard(cfg, inp_neg)
    assert out_neg.steer_cmd_safe == pytest.approx(-0.10)

    # Negative articulation but opposite steer command (counter-steering allowed)
    inp_counter = JackknifeInputs(
        theta_a=-0.35,
        theta_a_dot=-0.50,
        vx=18.0,
        mu=0.4,
        ltr=0.40,
        steer_cmd_req=0.20,
        steer_rate_req=0.50,
    )
    out_counter = step_jackknife_guard(cfg, inp_counter)
    assert out_counter.steer_cmd_safe == pytest.approx(0.20)


def test_rollover_warning_and_critical_locking() -> None:
    cfg = JackknifeConfig(ltr_warning=0.70, ltr_critical=0.85)

    # 1. Warning range: LTR = 0.775 (progressive damping)
    inp_warn = JackknifeInputs(
        theta_a=0.01,
        theta_a_dot=0.0,
        vx=15.0,
        mu=0.8,
        ltr=0.775,
        steer_cmd_req=0.10,
        steer_rate_req=0.40,
    )
    out_warn = step_jackknife_guard(cfg, inp_warn)
    assert out_warn.status_flags & FLAG_ROLLOVER_WARNING
    assert not out_warn.is_rollover_critical
    assert 0.0 < out_warn.steer_rate_safe < 0.40

    # 2. Critical range: LTR = 0.90 (complete lock & command reduction)
    inp_crit = JackknifeInputs(
        theta_a=0.01,
        theta_a_dot=0.0,
        vx=15.0,
        mu=0.8,
        ltr=0.90,
        steer_cmd_req=0.10,
        steer_rate_req=0.40,
    )
    out_crit = step_jackknife_guard(cfg, inp_crit)
    assert out_crit.is_rollover_critical
    assert out_crit.steer_rate_safe == 0.0
    assert out_crit.steer_cmd_safe == pytest.approx(0.08)  # 0.8 * 0.10


def test_edge_cases_and_non_finite_inputs() -> None:
    cfg = JackknifeConfig()

    # Non-finite numbers should be handled safely without exceptions
    inp_nan = JackknifeInputs(
        theta_a=float("nan"),
        theta_a_dot=float("inf"),
        vx=-5.0,
        mu=-0.2,
        ltr=float("nan"),
        steer_cmd_req=float("nan"),
        steer_rate_req=float("nan"),
    )
    out_nan = step_jackknife_guard(cfg, inp_nan)
    assert math.isfinite(out_nan.steer_cmd_safe)
    assert math.isfinite(out_nan.steer_rate_safe)
    assert math.isfinite(out_nan.trailer_brake_pressure)

    # Steer rate clamped to steer_rate_max
    inp_high_rate = JackknifeInputs(
        theta_a=0.0,
        theta_a_dot=0.0,
        vx=10.0,
        mu=0.8,
        ltr=0.0,
        steer_cmd_req=0.0,
        steer_rate_req=5.0,  # exceeds steer_rate_max (0.60)
    )
    out_high = step_jackknife_guard(cfg, inp_high_rate)
    assert out_high.steer_rate_safe == pytest.approx(cfg.steer_rate_max)


def test_compute_critical_articulation_angle_bounds() -> None:
    # High speed, low mu -> hit floor
    floor_th = compute_critical_articulation_angle(
        vx=50.0, mu=0.05, l2=8.5, g=9.81, min_val=0.20, max_val=0.785
    )
    assert floor_th == pytest.approx(0.20)

    # Low speed, high mu -> hit ceiling
    ceil_th = compute_critical_articulation_angle(
        vx=1.0, mu=1.0, l2=8.5, g=9.81, min_val=0.20, max_val=0.785
    )
    assert ceil_th == pytest.approx(0.785)
