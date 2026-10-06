from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from egga.config import load_config
from egga.eval.envelope_build import ENVELOPE_DIR
from egga.supervisor.core import step
from egga.supervisor.envelope import ARRAY_KEYS, Envelope, EnvelopeError, array_hash
from egga.supervisor.types import Config, Inputs, Mode, State

ENV = Envelope.load(ENVELOPE_DIR / "envelope_v1.npz", ENVELOPE_DIR / "envelope_v1.json")
CFG = Config.from_dict(load_config("supervisor.yaml"), load_config("vehicle.yaml"))
REF = tuple(float(x) for x in ENV.reference_gain)
DT = CFG.dt
LATCHED = (int(Mode.FALLBACK), int(Mode.MINIMAL_RISK))
NASTY = [float("nan"), float("inf"), -float("inf"), 0.0, -1.0, 1e9, -1e9, 1e-12]

finite = st.floats(-50.0, 50.0, allow_nan=False, allow_infinity=False)
anything = st.one_of(finite, st.sampled_from(NASTY))
gain = st.tuples(*[st.one_of(st.floats(0.0, 3.0), st.sampled_from(NASTY)) for _ in range(4)])


def verified_somewhere(out_gains: tuple[float, ...], inp: Inputs, state_speed: float) -> bool:
    """The applied gain is verified in the cell used for the lookup (or the verified-speed cell)."""
    mu_eff = max(inp.mu_lo, CFG.mu_floor)
    speeds = {max(state_speed, inp.speed), ENV.max_verified_speed(mu_eff, inp.tau_bar, inp.mass_hi)}
    bounds = (
        (inp.tau_bar, inp.mass_hi),
        (max(inp.tau_bar, CFG.tau_fallback), max(inp.mass_hi, CFG.mass_fallback)),
    )
    for speed in speeds:
        if speed is None:
            continue
        for tau, mass in bounds:
            mask = ENV.cell_mask(speed, mu_eff, tau, max(mass, 1.0))
            if mask is not None and ENV.contains(mask, out_gains):  # type: ignore[arg-type]
                return True
    return False


def inputs_from(t: float, v: tuple[float, ...], g: tuple[float, ...]) -> Inputs:
    status = int(abs(v[8])) % 4 if math.isfinite(v[8]) and abs(v[8]) < 1e6 else 3
    return Inputs(
        t=t, speed=v[0], speed_request=v[1], curvature_ahead=v[2], mu_lo=v[3], mu_hi=v[4],
        tau_bar=v[5], mass_hi=v[6], quality=v[7], est_status=status, yaw_rate=v[9],
        steer_meas=v[10], steer_cmd=v[11], e_abs=v[12], e_rate_abs=v[13], rl_valid=v[14] > 0,
        rl_gain=(g[0], g[1], g[2], g[3]), reference_gain=REF, request_reset=v[15] > 0,
    )  # fmt: skip


@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
@given(st.lists(st.tuples(*[anything for _ in range(16)]), min_size=1, max_size=40), gain)
def test_fuzz_random_nan_stale_and_extreme_inputs_never_crash_and_outputs_stay_sane(  # type: ignore[no-untyped-def]
    rows, g
) -> None:
    state = State()
    for k, v in enumerate(rows):
        inp = inputs_from(k * DT, v, g)
        out = step(state, CFG, ENV, inp)
        assert out.mode in {int(m) for m in Mode}
        assert all(math.isfinite(x) for x in out.gains)
        assert math.isfinite(out.speed_cmd) and out.speed_cmd >= 0.0
        assert out.steer_limit > 0.0 and out.rate_limit > 0.0


@settings(max_examples=40, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.floats(5.0, 14.0),
            st.floats(0.25, 1.0),
            st.floats(0.0, 0.12),
            st.floats(1.0, 1.8),
            gain,
        ),
        min_size=5,
        max_size=60,
    )
)
def test_supervisor_gains_are_always_verified_and_unsafe_rl_proposals_never_applied(rows) -> None:  # type: ignore[no-untyped-def]
    state = State()
    for k, (speed, mu, tau, mass, rl) in enumerate(rows):
        inp = Inputs(
            t=k * DT, speed=speed, speed_request=speed, curvature_ahead=0.01, mu_lo=mu,
            mu_hi=min(1.0, mu + 0.3), tau_bar=tau, mass_hi=mass, quality=0.3, est_status=0,
            yaw_rate=0.0, steer_meas=0.0, steer_cmd=0.0, e_abs=0.0, e_rate_abs=0.0,
            rl_valid=True, rl_gain=rl, reference_gain=REF,
        )  # fmt: skip
        out = step(state, CFG, ENV, inp)
        if out.verified:
            assert verified_somewhere(out.gains, inp, state.speed_cmd)
        if out.rl_applied:
            assert all(math.isfinite(x) for x in rl) and out.verified


@settings(max_examples=25, deadline=None)
@given(st.integers(0, 2**31 - 1), st.integers(200, 600))
def test_noisy_estimates_do_not_cause_mode_chatter(seed, steps) -> None:  # type: ignore[no-untyped-def]
    """Estimates jittering around thresholds change the non-latched mode at most once per dwell."""
    noise = np.random.default_rng(seed).random((steps, 3))
    noise[:, 2] = 0.06 * noise[:, 2] - 0.03
    state = State()
    changes: list[float] = []
    last = state.mode
    for k, (a, b, c) in enumerate(noise):
        inp = Inputs(
            t=k * DT, speed=7.5, speed_request=7.5, curvature_ahead=0.01, mu_lo=0.5,
            mu_hi=0.45 + 0.1 * a - 0.05, tau_bar=0.07 + 0.02 * b - 0.01 + c, mass_hi=1.3,
            quality=0.1 + 0.1 * (a - b), est_status=0, yaw_rate=0.0, steer_meas=0.0,
            steer_cmd=0.0, e_abs=0.0, e_rate_abs=0.0, rl_valid=False, rl_gain=REF,
            reference_gain=REF,
        )  # fmt: skip
        out = step(state, CFG, ENV, inp)
        if out.mode != last:
            if out.mode not in LATCHED and last not in LATCHED:
                changes.append(k * DT)
            last = out.mode
    assert np.all(np.diff(changes) >= CFG.min_dwell - 1e-9)


def _healthy(k: int, est_status: int) -> Inputs:
    return Inputs(
        t=k * DT, speed=7.5, speed_request=7.5, curvature_ahead=0.01, mu_lo=0.5, mu_hi=0.9,
        tau_bar=0.04, mass_hi=1.3, quality=0.3, est_status=est_status, yaw_rate=0.0,
        steer_meas=0.0, steer_cmd=0.0, e_abs=0.0, e_rate_abs=0.0, rl_valid=False, rl_gain=REF,
        reference_gain=REF,
    )  # fmt: skip


def test_fallback_latches_until_healthy_for_the_recovery_time() -> None:
    state = State()
    for k in range(250):
        step(state, CFG, ENV, _healthy(k, 2))
    assert state.mode == Mode.FALLBACK
    seen = []
    for k in range(250, 250 + 650):
        seen.append(step(state, CFG, ENV, _healthy(k, 0)).mode)
    assert int(Mode.FALLBACK) in seen and int(Mode.CAUTIOUS) in seen
    first_cautious = seen.index(int(Mode.CAUTIOUS))
    assert first_cautious * DT >= CFG.recover - 2 * DT  # not before the recovery time


# ---------------------------------------------------------------- envelope error paths
def _data() -> dict[str, np.ndarray]:
    with np.load(ENVELOPE_DIR / "envelope_v1.npz") as loaded:
        return {k: loaded[k] for k in loaded.files}


def test_envelope_rejects_a_mask_of_the_wrong_shape() -> None:
    data = _data()
    data["verified"] = data["verified"][:, :, :, :, :-1]
    with pytest.raises(EnvelopeError):
        Envelope(data)


def test_envelope_load_rejects_missing_arrays(tmp_path: Path) -> None:
    data = _data()
    del data["rate_limit"]
    npz = tmp_path / "e.npz"
    np.savez_compressed(npz, **data)
    meta = tmp_path / "e.json"
    meta.write_text('{"hash": "x"}', encoding="utf-8")
    with pytest.raises(EnvelopeError, match="missing arrays"):
        Envelope.load(npz, meta)
    assert set(ARRAY_KEYS) >= {"verified", "rate_limit"} and array_hash(_data())
