"""Red-team tests for the runtime supervisor. Each asserts the DESIRED safe property."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_supervisor import CFG, DT, ENV, make, run  # noqa: E402

from egga.supervisor.core import step  # noqa: E402
from egga.supervisor.types import Mode, State  # noqa: E402

MU_TOP = float(ENV.axes["mu"][-1])


def _applied_is_verified(out, speed, mu, tau, mass) -> bool:  # type: ignore[no-untyped-def]
    mask = ENV.cell_mask(speed, mu, tau, mass)
    return mask is not None and ENV.contains(mask, out.gains)


# ---- (d) unsafe action: out-of-grid speed reported as verified ----------------------
def test_speed_below_grid_is_not_reported_verified() -> None:
    s = State()
    out, _ = run(s, 1.0, speed=1.0, speed_request=1.0)
    assert not (out.verified and not _applied_is_verified(out, 1.0, 0.5, 0.04, 1.3)), (
        "gains from a different speed cell are claimed verified at 1 m/s (below the grid)"
    )


def test_speed_above_grid_is_not_reported_verified() -> None:
    s = State()
    out, _ = run(s, 1.0, speed=25.0, speed_request=25.0)
    assert not (out.verified and not _applied_is_verified(out, 25.0, 0.5, 0.04, 1.3)), (
        "gains from the top cell are claimed verified at 25 m/s (above the grid)"
    )


# ---- (d) mu_lo above the grid is not clamped for the curvature cap ------------------
def test_mu_lo_above_grid_cannot_relax_the_curvature_cap() -> None:
    kappa = 0.3
    assert ENV.speed_cap(kappa, MU_TOP) is None  # infeasible even at the best verified mu
    s = State()
    out, _ = run(s, 1.0, mu_lo=3.0, mu_hi=3.0, curvature_ahead=kappa)
    assert out.mode == int(Mode.MINIMAL_RISK), (
        f"mu_lo=3.0 (outside grid, physically impossible) keeps mode {out.mode} at kappa={kappa}"
    )


def test_ay_demand_limit_never_exceeds_verified_mu_axis() -> None:
    s = State()
    lo, _ = run(s, 0.5, mu_lo=MU_TOP, mu_hi=MU_TOP)
    s2 = State()
    hi, _ = run(s2, 0.5, mu_lo=5.0, mu_hi=5.0)
    assert hi.steer_limit <= lo.steer_limit + 1e-12


# ---- (d) non-positive speed request -------------------------------------------------
def test_negative_speed_request_never_commands_below_min_feasible_speed() -> None:
    s = State()
    out, _ = run(s, 6.0, speed_request=-5.0)
    assert out.speed_cmd >= min(ENV.min_feasible_speed, 7.5) - 1e-9


# ---- (d) speed command above the curvature cap after a sudden curvature jump --------
def test_speed_cmd_respects_curvature_cap_after_sudden_curvature() -> None:
    s = State()
    run(s, 1.0, speed=12.0, speed_request=12.0, curvature_ahead=0.0)
    cap = ENV.speed_cap(0.08, 0.5)
    assert cap is not None and cap < 12.0
    out, t = run(s, 0.01, start=1.0, speed=12.0, speed_request=12.0, curvature_ahead=0.08)
    assert out.flags & 256  # FLAG_CAPPED is raised immediately on the onset tick
    out, _ = run(s, 2.0, start=t, speed=12.0, speed_request=12.0, curvature_ahead=0.08)
    assert out.speed_cmd <= cap + 1e-6, (
        f"speed_cmd {out.speed_cmd:.2f} > cap {cap:.2f} after deceleration interval"
    )


# ---- (d) RL rate check is defeated by repeating the same proposal -------------------
def test_rl_rate_limit_cannot_be_bypassed_by_repeating_a_jump() -> None:
    s = State()
    a = (1.0, 0.05, 0.15, 1.0)
    b = (1.9, 0.15, 0.45, 1.8)  # far beyond rl_rate_per_s over one tick
    out = None
    for k in range(50):
        out = step(s, CFG, ENV, make(k * DT, rl_valid=True, rl_gain=a))
    assert out is not None and out.rl_applied
    accepted_after_jump = False
    for k in range(50, 55):
        out = step(s, CFG, ENV, make(k * DT, rl_valid=True, rl_gain=b))
        accepted_after_jump |= out.rl_applied
    assert not accepted_after_jump, "a 5-tick-old step of 0.9 in Kp was accepted as rate-valid"


# ---- (c)/(a) intermittent faults evade persistence timers ----------------------------
def test_duty_cycled_stale_estimator_is_not_treated_as_healthy() -> None:
    s = State()
    modes = []
    for k in range(1500):
        stale = (k % 10) != 0  # 9 stale ticks (0.09 s) < stale_grace 0.3 s, then one OK tick
        out = step(s, CFG, ENV, make(k * DT, est_status=2 if stale else 0))
        modes.append(out.mode)
    assert max(modes) >= int(Mode.FALLBACK), "90 % stale estimator never leaves NOMINAL"


def test_duty_cycled_actuator_fault_is_detected() -> None:
    s = State()
    modes = []
    for k in range(1500):
        bad = (k % 20) < 19  # 0.19 s bad then 1 clean tick; actuator_persist is 0.4 s
        out = step(s, CFG, ENV, make(k * DT, steer_meas=0.3 if bad else 0.0, steer_cmd=0.0))
        modes.append(out.mode)
    assert max(modes) >= int(Mode.FALLBACK), "95 % actuator-residual duty cycle never detected"


# ---- (b) MINIMAL_RISK latch defeated by reset spam on a flickering no-set -----------
def test_reset_spam_cannot_flap_minimal_risk_and_fallback_every_other_tick() -> None:
    s = State()
    changes_before = None
    for k in range(400):
        mu_hi = 0.1 if (k % 2) == 1 else 0.9
        step(s, CFG, ENV, make(k * DT, mu_hi=mu_hi, request_reset=True))
        if k == 10:
            changes_before = s.mode_changes
    assert changes_before is not None
    changes = s.mode_changes - changes_before
    assert changes <= 4, f"{changes} mode changes in 3.9 s with a persistent domain violation"


# ---- time / invalid channels (attacks that did NOT break it; kept as regression) ------
def test_huge_time_gap_with_invalid_estimator_is_a_fault() -> None:
    s = State()
    run(s, 1.0)
    out = step(s, CFG, ENV, make(500.0, est_status=3))
    assert out.mode >= int(Mode.FALLBACK)


# ---- invalid-input channels --------------------------------------------------------
def test_unknown_estimator_status_is_not_ok() -> None:
    s = State()
    modes = [step(s, CFG, ENV, make(k * DT, est_status=99)).mode for k in range(100)]
    assert max(modes) >= int(Mode.FALLBACK)


def test_sudden_mass_jump_does_not_leave_gain_unverified_at_actual_speed() -> None:
    """mass_hi jumps 1.0 -> 1.9 at 15 m/s: v_ver drops, but the plant is still at 15 m/s while the
    slew-limited speed_cmd decelerates. The applied gain must stay verified at the ACTUAL speed
    (or the output must not claim verified)."""
    s = State()
    run(s, 1.0, speed=15.0, speed_request=15.0, mass_hi=1.0, curvature_ahead=0.0)
    bad = 0
    for k in range(100):
        out = step(s, CFG, ENV, make(1.0 + k * DT, speed=15.0, speed_request=15.0, mass_hi=1.9,
                                     curvature_ahead=0.0, tau_bar=0.15))
        if out.verified and not _applied_is_verified(out, 15.0, 0.5, 0.15, 1.9):
            bad += 1
    assert bad == 0, f"{bad}/100 ticks claim verified with a gain not verified at 15 m/s"


# ---- closed loop: in-domain delay plus lateral-error noise ---------------------------
def test_closed_loop_in_grid_delay_with_4cm_noise_does_not_diverge() -> None:
    import copy

    from egga.config import load_config
    from egga.estimation.suite import EstimatorSuite
    from egga.eval.baselines import load_estimators
    from egga.eval.closed_loop import load_plant_config, run_closed_loop
    from egga.scenarios.sets import load_scenarios, run_arguments

    spec = copy.deepcopy(load_scenarios("val")[22])  # never the locked test set
    spec["duration_s"] = 20.0
    spec["mass_scale"] = 1.2293826
    spec["mu_belief_error"] = -0.03295
    plant, mission, belief, seed = run_arguments(spec)
    plant["actuator"]["delay_s"] = 0.07  # well inside the verified tau axis (<= 0.15 s)
    plant["sensors"]["lateral_error"]["noise_std"] = 0.04
    est = EstimatorSuite(load_config("vehicle.yaml"), load_estimators(), float(mission["dt_s"]))
    run = run_closed_loop(
        "b4_supervised", load_plant_config(plant), seed=seed, mu_belief_error=belief,
        mission_cfg=mission, estimator=est,
    )
    assert run.diverged_at_s is None, (
        f"diverged at {run.diverged_at_s}s, max|ey|={np.nanmax(np.abs(run.ey)):.2f} m, "
        f"modes seen={sorted(set(run.modes[np.isfinite(run.modes)].astype(int)))}"
    )
