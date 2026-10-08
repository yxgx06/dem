from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from egga.config import load_config
from egga.eval.envelope_build import ENVELOPE_DIR
from egga.supervisor import guard, health, modes
from egga.supervisor.core import log_event, step
from egga.supervisor.envelope import Envelope
from egga.supervisor.gains import gain_bounds, next_index
from egga.supervisor.types import (
    FLAG_ACTUATOR,
    FLAG_DOMAIN,
    FLAG_ESTIMATOR,
    FLAG_INVALID,
    FLAG_MARGIN,
    FLAG_NO_SET,
    FLAG_SATURATION,
    FLAG_YAW,
    HISTORY,
    LOG_CAPACITY,
    Config,
    Inputs,
    Mode,
    Reason,
    State,
)

ENV = Envelope.load(ENVELOPE_DIR / "envelope_v1.npz", ENVELOPE_DIR / "envelope_v1.json")
CFG = Config.from_dict(load_config("supervisor.yaml"), load_config("vehicle.yaml"))
REF = tuple(float(x) for x in ENV.reference_gain)
DT = CFG.dt


def make(t: float, **kw: object) -> Inputs:
    base: dict[str, object] = {
        "t": t, "speed": 7.5, "speed_request": 7.5, "curvature_ahead": 0.01, "mu_lo": 0.5,
        "mu_hi": 0.9, "tau_bar": 0.04, "mass_hi": 1.3, "quality": 0.3, "est_status": 0,
        "yaw_rate": 0.0, "steer_meas": 0.0, "steer_cmd": 0.0, "e_abs": 0.0, "e_rate_abs": 0.0,
        "rl_valid": False, "rl_gain": REF, "reference_gain": REF,
    }  # fmt: skip
    base.update(kw)
    return Inputs(**base)  # type: ignore[arg-type]


def run(state: State, seconds: float, start: float = 0.0, **kw: object):  # type: ignore[no-untyped-def]
    out = None
    n = int(round(seconds / DT))
    for k in range(n):
        out = step(state, CFG, ENV, make(start + k * DT, **kw))
    return out, start + n * DT


# ---------------------------------------------------------------- types / config
def test_config_rejects_wrong_vector_lengths() -> None:
    sup = load_config("supervisor.yaml")
    sup["modes"]["speed_scale"] = [1.0] * 5
    with pytest.raises(ValueError):
        Config.from_dict(sup, load_config("vehicle.yaml"))
    sup = load_config("supervisor.yaml")
    sup["health"]["rl_rate_per_s"] = [1.0] * 3
    with pytest.raises(ValueError):
        Config.from_dict(sup, load_config("vehicle.yaml"))


def test_understeer_coefficient_grows_with_mass() -> None:
    assert CFG.understeer_coeff(1.9) > CFG.understeer_coeff(1.0) > 0.0


# ---------------------------------------------------------------- guard
def test_steer_limit_grows_with_ay_falls_with_speed_and_respects_hardware() -> None:
    assert guard.steer_angle_limit(CFG, 10.0, 4.0, 1.0) > guard.steer_angle_limit(
        CFG, 10.0, 2.0, 1.0
    )
    assert guard.steer_angle_limit(CFG, 8.0, 3.0, 1.0) > guard.steer_angle_limit(
        CFG, 14.0, 3.0, 1.0
    )
    assert guard.steer_angle_limit(CFG, 0.1, 50.0, 1.0) == CFG.steer_hw_max


def test_lateral_margin_check_both_outcomes() -> None:
    assert not guard.lateral_margin_violated(CFG, 0.0, 0.0, 2.0, 0.05)
    assert guard.lateral_margin_violated(CFG, 0.95, 1.0, 5.0, 0.15)
    assert guard.clip_symmetric(2.0, 0.5) == 0.5 and guard.clip_symmetric(-2.0, 0.5) == -0.5
    assert guard.clip_symmetric(0.1, 0.5) == 0.1


# ---------------------------------------------------------------- health
def test_accumulate_and_command_history() -> None:
    assert health.accumulate(1.0, True, 0.5) == 1.5 and health.accumulate(1.0, False, 0.5) == 0.0
    s = State()
    assert health.command_range(s, 5) == (0.0, 0.0)
    for v in range(HISTORY + 5):  # wraps the ring
        health.push_command(s, float(v))
    low, high = health.command_range(s, 3)
    assert (low, high) == (HISTORY + 2.0, HISTORY + 4.0)


def test_yaw_residual_flags_only_a_real_mismatch_above_minimum_speed() -> None:
    s = State()
    for _ in range(200):
        bad = health.yaw_residual_bad(s, CFG, 7.5, 0.0, 0.1, 1.3, DT)
    assert bad
    s2 = State()
    for _ in range(200):
        ok = health.yaw_residual_bad(s2, CFG, 7.5, s2.yaw_pred, 0.1, 1.3, DT)
    assert not ok
    s3 = State()
    for _ in range(200):
        slow = health.yaw_residual_bad(s3, CFG, 1.0, 0.0, 0.1, 1.3, DT)
    assert not slow


def test_actuator_residual_allows_lag_inside_the_command_range() -> None:
    s = State()
    for cmd in (0.0, 0.05, 0.1):
        health.push_command(s, cmd)
    assert not health.actuator_residual_bad(s, CFG, 0.05, 0.04, DT)
    assert health.actuator_residual_bad(s, CFG, 0.3, 0.04, DT)


def test_saturation_check() -> None:
    assert health.saturation_bad(CFG, 0.2, 0.2) and not health.saturation_bad(CFG, 0.1, 0.2)


def test_rl_check_reasons() -> None:
    lower, upper = gain_bounds(ENV)
    s = State()
    ok = (1.0, 0.05, 0.1, 1.0)
    assert health.rl_check(s, CFG, ok, DT, lower, upper) == Reason.NONE  # first proposal
    assert health.rl_check(s, CFG, ok, DT, lower, upper) == Reason.NONE  # unchanged
    jump = (2.0, 0.05, 0.1, 1.0)
    assert health.rl_check(s, CFG, jump, DT, lower, upper) == Reason.RL_REJECTED_RATE
    out_of_range = (9.0, 0.05, 0.1, 1.0)
    assert health.rl_check(s, CFG, out_of_range, DT, lower, upper) == Reason.RL_REJECTED_RANGE
    nan = (float("nan"), 0.05, 0.1, 1.0)
    assert health.rl_check(s, CFG, nan, DT, lower, upper) == Reason.RL_REJECTED_RANGE
    assert not s.last_rl_valid


# ---------------------------------------------------------------- gains
def test_next_index_forced_hop_and_neighbour_filtering() -> None:
    mask = np.zeros((3, 3, 3, 3), dtype=bool)
    mask[0, 0, 0, 0] = mask[1, 0, 0, 0] = mask[2, 0, 0, 0] = True
    target = (2, 0, 0, 0)
    assert next_index(ENV, mask, None, target, True) == (target, True)
    assert next_index(ENV, mask, (0, 1, 1, 1), target, True) == (target, True)  # unverified now
    assert next_index(ENV, mask, (0, 0, 0, 0), target, False) == ((0, 0, 0, 0), False)
    assert next_index(ENV, mask, (0, 0, 0, 0), target, True) == ((1, 0, 0, 0), False)
    assert next_index(ENV, mask, (2, 0, 0, 0), target, True) == ((2, 0, 0, 0), False)  # at target
    lower, upper = gain_bounds(ENV)
    assert all(lo < hi for lo, hi in zip(lower, upper, strict=True))


# ---------------------------------------------------------------- modes
def test_desired_mode_branches_and_hysteresis() -> None:
    d = modes.desired_mode
    assert d(CFG, Mode.NOMINAL, 0.09, 0.9, 0.3) == Mode.DEGRADED_ACTUATOR
    assert d(CFG, Mode.DEGRADED_ACTUATOR, 0.07, 0.9, 0.3) == Mode.DEGRADED_ACTUATOR  # hysteresis
    assert d(CFG, Mode.DEGRADED_ACTUATOR, 0.05, 0.9, 0.3) == Mode.NOMINAL
    assert d(CFG, Mode.NOMINAL, 0.0, 0.3, 0.3) == Mode.LOW_MU
    assert d(CFG, Mode.LOW_MU, 0.0, 0.45, 0.3) == Mode.LOW_MU  # hysteresis
    assert d(CFG, Mode.LOW_MU, 0.0, 0.6, 0.3) == Mode.NOMINAL
    assert d(CFG, Mode.NOMINAL, 0.0, 0.9, 0.01) == Mode.CAUTIOUS
    assert d(CFG, Mode.CAUTIOUS, 0.0, 0.9, 0.1) == Mode.CAUTIOUS  # hysteresis
    assert d(CFG, Mode.CAUTIOUS, 0.0, 0.9, 0.3) == Mode.NOMINAL


def _state(mode: Mode, entry: float = 0.0, healthy: float = 0.0) -> State:
    return State(mode=int(mode), mode_entry_t=entry, healthy_for=healthy)


def test_next_mode_latching_escalation_and_recovery() -> None:
    nm = modes.next_mode
    args = {"tau": 0.0, "mu_hi": 0.9, "quality": 0.3}
    latched = _state(Mode.MINIMAL_RISK)
    assert nm(latched, CFG, 9.0, False, False, reset=False, **args)[0] == Mode.MINIMAL_RISK
    assert nm(latched, CFG, 9.0, False, False, reset=True, **args) == (Mode.FALLBACK, Reason.RESET)
    assert nm(latched, CFG, 9.0, True, False, reset=True, **args)[0] == Mode.MINIMAL_RISK
    assert nm(latched, CFG, 9.0, False, True, reset=True, **args)[0] == Mode.MINIMAL_RISK
    nominal = _state(Mode.NOMINAL)
    assert nm(nominal, CFG, 9.0, True, False, reset=False, **args) == (
        Mode.MINIMAL_RISK, Reason.NO_VERIFIED_SET)  # fmt: skip
    assert nm(nominal, CFG, 9.0, False, True, reset=False, **args) == (
        Mode.FALLBACK, Reason.MODE_CHANGE)  # fmt: skip
    fallback = _state(Mode.FALLBACK)
    assert nm(fallback, CFG, 9.0, False, True, reset=False, **args) == (Mode.FALLBACK, Reason.NONE)
    assert nm(fallback, CFG, 9.0, False, False, reset=False, **args)[0] == Mode.FALLBACK
    healthy = _state(Mode.FALLBACK, healthy=6.0)
    assert nm(healthy, CFG, 9.0, False, False, reset=False, **args) == (
        Mode.CAUTIOUS,
        Reason.RECOVERY,
    )
    low = {"tau": 0.0, "mu_hi": 0.3, "quality": 0.3}
    early = nm(_state(Mode.NOMINAL, entry=0.0), CFG, 1.0, False, False, reset=False, **low)
    late = nm(_state(Mode.NOMINAL, entry=0.0), CFG, 3.0, False, False, reset=False, **low)
    assert early[0] == Mode.NOMINAL and late == (Mode.LOW_MU, Reason.MODE_CHANGE)


# ---------------------------------------------------------------- core behaviour
def test_healthy_run_is_nominal_verified_and_logs_nothing() -> None:
    s = State()
    out, _ = run(s, 3.0)
    assert out.mode == Mode.NOMINAL and out.verified and out.flags == 0
    mask = ENV.cell_mask(7.5, 0.5, 0.04, 1.3)
    assert s.log_n == 0 and mask is not None and ENV.contains(mask, out.gains)
    assert out.speed_cmd == pytest.approx(7.5)  # request is below the caps


def test_invalid_inputs_go_to_fallback_with_safe_speed_and_are_logged() -> None:
    s = State()
    run(s, 0.5)
    out = step(s, CFG, ENV, make(0.6, speed=float("nan")))
    assert out.mode == Mode.FALLBACK and out.flags & FLAG_INVALID
    assert s.log_code[: s.log_n].tolist().count(int(Reason.INVALID_INPUT)) == 1
    back = step(s, CFG, ENV, make(0.4))  # time going backwards
    assert back.flags & FLAG_INVALID
    out_nan_t = step(State(), CFG, ENV, make(float("nan")))
    assert out_nan_t.flags & FLAG_INVALID and out_nan_t.mode == Mode.FALLBACK
    assert step(State(), CFG, ENV, make(0.0, speed=0.0)).flags & FLAG_INVALID


@pytest.mark.parametrize(
    ("kw", "flag"),
    [
        ({"est_status": 2}, FLAG_ESTIMATOR),
        ({"steer_meas": 0.1, "yaw_rate": 0.0}, FLAG_YAW),
        ({"steer_meas": 0.3}, FLAG_ACTUATOR),
        ({"steer_cmd": 0.5, "steer_meas": 0.5}, FLAG_SATURATION),
    ],
)
def test_persistent_faults_escalate_to_fallback_then_recover(
    kw: dict[str, object], flag: int
) -> None:
    s = State()
    out, t = run(s, 2.5, **kw)
    assert out.flags & flag and out.mode == Mode.FALLBACK
    out, t = run(s, 4.0, start=t)
    assert out.mode == Mode.FALLBACK  # latched: not yet healthy long enough
    out, t = run(s, 2.0, start=t)
    assert out.mode == Mode.CAUTIOUS
    out, t = run(s, 3.0, start=t)
    assert out.mode == Mode.NOMINAL  # after the dwell time
    assert Reason.RECOVERY in s.log_code[: s.log_n].tolist()


def test_estimator_low_excitation_alone_is_not_a_fault() -> None:
    out, _ = run(State(), 2.0, est_status=1)
    assert out.mode == Mode.NOMINAL and not out.flags & FLAG_ESTIMATOR


def test_domain_violation_and_unreachable_demand_go_to_minimal_risk_until_reset() -> None:
    s = State()
    out, t = run(s, 0.5, mu_hi=0.1)
    assert out.mode == Mode.MINIMAL_RISK and out.flags & (FLAG_DOMAIN | FLAG_NO_SET)
    assert out.speed_cmd <= 7.5
    out, t = run(s, 1.0, start=t)
    assert out.mode == Mode.MINIMAL_RISK  # latched even though the inputs are fine again
    out = step(s, CFG, ENV, make(t, request_reset=True))
    assert out.mode == Mode.FALLBACK
    out2 = step(State(), CFG, ENV, make(0.0, curvature_ahead=1.0))  # the speed cap is None
    assert out2.mode == Mode.MINIMAL_RISK and out2.flags & FLAG_NO_SET
    out3 = step(State(), CFG, ENV, make(0.0, tau_bar=0.5))  # delay beyond the grid
    assert out3.mode == Mode.MINIMAL_RISK


def test_rl_proposals_are_projected_rejected_and_never_applied_in_latched_modes() -> None:
    s = State()
    proposal = (1.0, 0.05, 0.15, 1.0)
    out = None
    for k in range(60):
        out = step(s, CFG, ENV, make(k * DT, rl_valid=True, rl_gain=proposal))
    assert out is not None and out.rl_applied
    mask = ENV.cell_mask(7.5, 0.5, 0.04, 1.3)
    assert mask is not None and ENV.contains(mask, out.gains)
    k0 = 60
    bad = (9.0, 0.05, 0.15, 1.0)
    out = step(s, CFG, ENV, make(k0 * DT, rl_valid=True, rl_gain=bad))
    assert not out.rl_applied and s.rl_rejects == 1
    for k in range(1, 30):  # persistent rejections are logged once at the threshold
        step(s, CFG, ENV, make((k0 + k) * DT, rl_valid=True, rl_gain=bad))
    assert s.rl_rejects >= CFG.rl_persist_rejects
    assert s.log_code[: s.log_n].tolist().count(int(Reason.RL_REJECTED_RANGE)) == 2
    back = step(s, CFG, ENV, make((k0 + 40) * DT, rl_valid=True, rl_gain=proposal))
    assert back.rl_applied  # an acceptable proposal after rejections is accepted again
    fast = step(s, CFG, ENV, make((k0 + 41) * DT, rl_valid=True, rl_gain=(2.1, 0.05, 0.15, 1.0)))
    assert not fast.rl_applied  # changes too fast
    latched = State()
    out, _ = run(latched, 2.5, est_status=2, rl_valid=True, rl_gain=proposal)
    assert out.mode == Mode.FALLBACK and not out.rl_applied


def test_lateral_margin_flag_reduces_the_speed_command() -> None:
    out, _ = run(State(), 3.0, speed_request=7.5, e_abs=0.99, e_rate_abs=1.0)
    assert out.flags & FLAG_MARGIN
    assert out.speed_cmd < 7.5


def test_speed_command_is_slew_limited_and_capped_by_curvature_and_verified_speed() -> None:
    s = State()
    fast = {"speed": 12.0, "speed_request": 12.0, "curvature_ahead": 0.05, "mu_lo": 0.5}
    out = step(s, CFG, ENV, make(0.0, **fast))
    assert 12.0 - CFG.decel * DT * 1.01 <= out.speed_cmd <= 12.0  # starts at the actual speed
    out = step(s, CFG, ENV, make(DT, **fast))
    assert 12.0 - 2 * CFG.decel * DT * 1.01 <= out.speed_cmd < 12.0 - CFG.decel * DT * 0.99
    out, _ = run(State(), 8.0, speed=7.5, speed_request=15.0, curvature_ahead=0.05, mu_lo=0.5)
    cap = ENV.speed_cap(0.05, 0.5)
    assert cap is not None and out.speed_cmd <= cap + 1e-9


def test_gain_changes_are_rate_limited_and_stay_verified_when_the_envelope_shrinks() -> None:
    s = State()
    out, t = run(s, 1.0, rl_valid=False)
    hops = 0
    last = out.gains
    for k in range(100):
        out = step(s, CFG, ENV, make(t + k * DT, rl_valid=True, rl_gain=(0.7, 0.01, 0.07, 0.6)))
        if out.gains != last:
            hops += 1
            last = out.gains
    assert hops <= 10  # at most one hop per gain_hop_period
    s2 = State()
    out, t = run(s2, 1.0, tau_bar=0.0)
    out = step(s2, CFG, ENV, make(t, tau_bar=0.1))
    v_ver = ENV.max_verified_speed(0.5, 0.1, 1.3)
    assert v_ver is not None
    mask = ENV.cell_mask(v_ver, 0.5, 0.1, 1.3)  # gains come from the highest verified speed
    assert not out.verified and mask is not None and ENV.contains(mask, out.gains)


def test_speed_above_the_grid_uses_the_hardware_rate_limit_and_verified_speed_mask() -> None:
    out = step(State(), CFG, ENV, make(0.0, speed=20.0, speed_request=20.0))
    assert out.rate_limit == CFG.steer_rate_hw_max
    assert not out.verified  # out-of-grid speed is not verified (per redteam finding V2)


def test_event_log_is_a_ring_buffer() -> None:
    s = State()
    for k in range(LOG_CAPACITY + 10):
        log_event(s, float(k), int(Reason.MODE_CHANGE), float(k))
    assert s.log_n == LOG_CAPACITY + 10 and s.log_t[9] == LOG_CAPACITY + 9


def test_outputs_are_immutable_value_objects() -> None:
    out = step(State(), CFG, ENV, make(0.0))
    with pytest.raises(dataclasses.FrozenInstanceError):
        out.mode = 3  # type: ignore[misc]
    assert math.isfinite(out.speed_cmd) and out.steer_limit > 0 and out.rate_limit > 0
