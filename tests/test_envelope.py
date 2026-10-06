from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from egga.config import REPO_ROOT, load_config
from egga.controllers.base import Observation
from egga.controllers.classical import PIDController
from egga.controllers.lqr import discretize, error_model
from egga.eval.envelope_build import (
    ENVELOPE_DIR,
    LinearLoop,
    build_envelope,
    design_vehicle,
    envelope_config,
    pole_metrics,
    write_envelope,
)
from egga.supervisor.envelope import ARRAY_KEYS, Envelope, EnvelopeError, array_hash

DT = 0.01
NPZ = ENVELOPE_DIR / "envelope_v1.npz"
META = ENVELOPE_DIR / "envelope_v1.json"

SMALL = {
    "grid": {
        "speed_mps": [10.0, 12.5],
        "mu_lo": [0.5, 0.9],
        "tau_bar_s": [0.0, 0.06],
        "mass_scale": [1.0, 1.9],
    },
    "gains": {"kp": [0.7, 1.6], "ki": [0.05], "kd": [0.07, 0.25], "khead": [1.0, 1.4]},
    "nonlinear": {"tests_per_cell": 4},
}


@pytest.fixture(scope="module")
def small_envelope() -> tuple[Envelope, dict, dict]:
    data, manifest = build_envelope(SMALL, workers=1)
    return Envelope(data), data, manifest


@pytest.fixture(scope="module")
def full_envelope() -> Envelope:
    return Envelope.load(NPZ, META)


# ---------------------------------------------------------------- linear model
@pytest.mark.parametrize("gains", [(1.3, 0.10, 0.25, 1.2), (0.7, 0.05, 0.07, 0.8)])
@pytest.mark.parametrize("delay", [0, 4, 10])
def test_linear_model_matches_the_real_controller_and_plant_algebra(
    gains: tuple[float, float, float, float], delay: int
) -> None:
    vehicle = load_config("vehicle.yaml")
    design = design_vehicle(vehicle, 1.3, 1.0)
    loop = LinearLoop(design, 10.0, delay, DT, 5.0)
    a_d, b_d = discretize(*error_model(design, 10.0), DT)
    pid = PIDController(
        {"kp": gains[0], "ki": gains[1], "kd": gains[2], "khead": gains[3], "integ_limit": 1e9},
        DT,
        5.0,
    )
    x = np.array([0.05, 0.01, -0.02, 0.0])
    buffer = [0.0] * delay
    z = np.zeros(loop.n)
    z[:4] = x
    z[5] = -x[0]  # the controller's first derivative is zero: previous e starts at the first e
    a_cl = loop.matrices(np.array([gains]))[0]
    worst = 0.0
    for _ in range(300):
        obs = Observation(-x[0], 0.0, -x[2], 0.0, 0.0, 0.0, 0.8)
        cmd = pid.command(obs).steer
        applied = buffer[-1] if delay else cmd
        if delay:
            buffer = [cmd, *buffer[:-1]]
        x = a_d @ x + b_d[:, 0] * applied
        z = a_cl @ z
        worst = max(worst, float(np.abs(z[:4] - x).max()))
    assert worst < 1e-9


def test_pole_metrics_excludes_the_integrator_and_reports_damping() -> None:
    eig = np.array([[0.9999 + 0j, 0.9 + 0.2j, 0.9 - 0.2j, 0.5 + 0j]])
    rho, fast, zeta = pole_metrics(eig, 0.8)
    assert rho[0] == pytest.approx(0.9999)
    assert fast[0] == pytest.approx(abs(0.9 + 0.2j))
    assert 0.0 < zeta[0] < 1.0


def test_slow_gain_is_rejected_and_aggressive_gain_fails_only_with_delay() -> None:
    from egga.eval.envelope_build import linear_assessment

    cfg = envelope_config()
    vehicle = load_config("vehicle.yaml")
    slow, balanced, aggressive = (
        (0.4, 0.01, 0.02, 0.6),
        (1.6, 0.05, 0.25, 0.8),
        (2.2, 0.15, 0.45, 1.8),
    )
    gains = np.array([slow, balanced, aggressive])
    no_delay, rho = linear_assessment(cfg, vehicle, 10.0, 0.0, 1.0, gains)
    delayed, _ = linear_assessment(cfg, vehicle, 10.0, 0.10, 1.0, gains)
    assert not no_delay[0]  # too slow: fails the fast-mode decay criterion
    assert no_delay[1] and no_delay[2]
    assert rho[2] < rho[1]  # without delay the aggressive gain is the better damped one
    assert not delayed[2]  # but it is rejected once the delay bound is 100 ms
    assert not delayed[0]


# ---------------------------------------------------------------- lookup semantics (small build)
def test_cell_mask_is_the_and_of_bracketing_cells(small_envelope) -> None:  # type: ignore[no-untyped-def]
    env, _, _ = small_envelope
    mask = env.cell_mask(11.0, 0.7, 0.03, 1.5)
    assert mask is not None
    expected = np.ones(env.verified.shape[4:], dtype=bool)
    for iv in (0, 1):
        for imu in (0, 1):
            for it in (0, 1):
                for im in (0, 1):
                    expected &= env.verified[iv, imu, it, im]
    np.testing.assert_array_equal(mask, expected)


@pytest.mark.parametrize(
    ("args", "none"),
    [
        ((9.0, 0.7, 0.03, 1.5), True),  # speed below grid
        ((13.0, 0.7, 0.03, 1.5), True),  # speed above grid
        ((11.0, 0.3, 0.03, 1.5), True),  # mu_lo below grid
        ((11.0, 0.7, 0.08, 1.5), True),  # tau_bar above grid
        ((11.0, 0.7, 0.03, 2.2), True),  # mass above grid
        ((11.0, 0.95, 0.03, 1.5), False),  # mu above grid: clamp
        ((11.0, 0.7, -0.01, 1.5), False),  # tau below grid: clamp
        ((11.0, 0.7, 0.03, 0.9), False),  # mass below grid: clamp
        ((float("nan"), 0.7, 0.03, 1.5), True),
        ((11.0, float("nan"), 0.03, 1.5), True),
    ],
)
def test_out_of_range_policy(small_envelope, args, none) -> None:  # type: ignore[no-untyped-def]
    env, _, _ = small_envelope
    assert (env.cell_mask(*args) is None) is none


def test_clamped_queries_use_the_nearest_cell(small_envelope) -> None:  # type: ignore[no-untyped-def]
    env, _, _ = small_envelope
    np.testing.assert_array_equal(
        env.cell_mask(11.0, 0.95, 0.03, 1.5), env.cell_mask(11.0, 0.9, 0.03, 1.5)
    )


def test_exact_grid_point_equals_its_single_cell(small_envelope) -> None:  # type: ignore[no-untyped-def]
    env, _, _ = small_envelope
    np.testing.assert_array_equal(env.cell_mask(10.0, 0.5, 0.0, 1.0), env.verified[0, 0, 0, 0])


def test_verified_is_a_subset_of_linear_accepted(small_envelope) -> None:  # type: ignore[no-untyped-def]
    env, _, _ = small_envelope
    assert not (env.verified & ~env.linear_accepted).any()


def test_project_returns_a_verified_grid_gain_nearest_to_the_target(small_envelope) -> None:  # type: ignore[no-untyped-def]
    env, _, _ = small_envelope
    mask = env.cell_mask(10.0, 0.5, 0.0, 1.0)
    assert mask is not None and mask.any()
    gain = env.project(mask, (1.0, 0.05, 0.1, 1.2))
    assert gain is not None and env.contains(mask, gain)
    empty = np.zeros_like(mask)
    assert env.project(empty, (1.0, 0.05, 0.1, 1.2)) is None
    assert not env.contains(mask, (9.0, 9.0, 9.0, 9.0))


def test_demand_limits_and_speed_cap(small_envelope) -> None:  # type: ignore[no-untyped-def]
    env, _, _ = small_envelope
    assert env.ay_max(0.5) == pytest.approx(env.ay_fraction_k * 0.5 * env.gravity)
    assert env.ay_max(0.9) > env.ay_max(0.5)
    straight = env.speed_cap(0.0, 0.5)
    assert straight == pytest.approx(12.5)
    tight = env.speed_cap(0.05, 0.5)
    assert tight is not None and tight < 12.5
    assert tight**2 * 0.05 <= env.ay_max(0.5) + 1e-9
    assert env.speed_cap(1.0, 0.5) is None  # infeasible even at the lowest speed: no relaxation
    assert env.speed_cap(float("nan"), 0.5) is None
    assert env.rate_limit(11.0, 1.5) is not None and env.rate_limit(20.0, 1.5) is None


def test_max_verified_speed_is_a_grid_speed_with_a_verified_gain(small_envelope) -> None:  # type: ignore[no-untyped-def]
    env, _, _ = small_envelope
    speed = env.max_verified_speed(0.9, 0.0, 1.0)
    assert speed in (10.0, 12.5)
    mask = env.cell_mask(speed, 0.9, 0.0, 1.0)  # type: ignore[arg-type]
    assert mask is not None and mask.any()
    higher = [v for v in (10.0, 12.5) if v > speed]  # type: ignore[operator]
    for v in higher:
        m = env.cell_mask(v, 0.9, 0.0, 1.0)
        assert m is None or not m.any()
    assert env.max_verified_speed(0.9, 0.5, 1.0) is None  # tau above the grid
    assert env.max_verified_speed(float("nan"), 0.0, 1.0) is None


@settings(max_examples=200, deadline=None)
@given(
    v=st.floats(8.0, 14.0),
    mu=st.floats(0.3, 1.0),
    tau=st.floats(-0.02, 0.09),
    mass=st.floats(0.8, 2.1),
)
def test_returned_gains_are_always_inside_every_bracketing_verified_cell(  # type: ignore[no-untyped-def]
    small_envelope, v, mu, tau, mass
) -> None:
    env, _, _ = small_envelope
    mask = env.cell_mask(v, mu, tau, mass)
    if mask is None:
        return
    for pick in np.argwhere(mask):
        gain = env.gain_at((int(pick[0]), int(pick[1]), int(pick[2]), int(pick[3])))
        # every bracketing cell must contain it
        for name, value in zip(("speed", "mu", "tau", "mass"), (v, mu, tau, mass), strict=True):
            assert env._bracket(name, value) is not None
        projected = env.project(mask, gain)
        assert projected is not None and env.contains(mask, projected)
    nominal = env.nominal(mask)
    assert nominal is None or env.contains(mask, nominal)


# ---------------------------------------------------------------- hash and integrity
def test_rebuild_with_the_same_inputs_gives_the_same_hash(small_envelope) -> None:  # type: ignore[no-untyped-def]
    _, _, manifest = small_envelope
    _, again = build_envelope(SMALL, workers=1)
    assert again["hash"] == manifest["hash"]


def test_written_envelope_round_trips_and_detects_corruption(  # type: ignore[no-untyped-def]
    small_envelope, tmp_path: Path
) -> None:
    _, data, manifest = small_envelope
    npz, meta = write_envelope(data, manifest, tmp_path)
    loaded = Envelope.load(npz, meta)
    np.testing.assert_array_equal(loaded.verified, data["verified"])
    tampered = dict(data)
    tampered["verified"] = ~np.asarray(data["verified"])
    np.savez_compressed(npz, **tampered)
    with pytest.raises(EnvelopeError):
        Envelope.load(npz, meta)
    with pytest.raises(EnvelopeError):
        Envelope.load(tmp_path / "missing.npz", meta)


def test_array_hash_changes_with_any_array(small_envelope) -> None:  # type: ignore[no-untyped-def]
    _, data, _ = small_envelope
    base = array_hash(data)
    changed = dict(data)
    changed["reference_gain"] = np.asarray(data["reference_gain"]) + 1e-6
    assert array_hash(changed) != base


# ---------------------------------------------------------------- committed envelope
def test_committed_envelope_loads_and_matches_its_manifest(full_envelope: Envelope) -> None:
    manifest = json.loads(META.read_text(encoding="utf-8"))
    assert manifest["hash"] == array_hash({k: np.load(NPZ)[k] for k in ARRAY_KEYS})
    assert manifest["counts"]["cells"] == int(np.prod(full_envelope.verified.shape[:4]))


def test_committed_envelope_has_a_verified_set_and_conservatism_structure(
    full_envelope: Envelope,
) -> None:
    env = full_envelope
    cells = int(np.prod(env.verified.shape[:4]))
    with_set = env.verified.reshape(cells, -1).any(axis=1)
    assert with_set.any() and not with_set.all()
    # longer delay bounds can only shrink the average verified set
    per_tau = env.verified.sum(axis=(0, 1, 3, 4, 5, 6, 7))
    assert per_tau[0] >= per_tau[-1]


def test_every_gain_the_committed_table_returns_is_in_a_verified_cell(  # type: ignore[no-untyped-def]
    full_envelope: Envelope,
) -> None:
    env = full_envelope
    rng = np.random.default_rng(0)
    for _ in range(300):
        v = rng.uniform(5.0, 15.0)
        mu = rng.uniform(0.2, 1.0)
        tau = rng.uniform(0.0, 0.15)
        mass = rng.uniform(1.0, 1.9)
        mask = env.cell_mask(v, mu, tau, mass)
        assert mask is not None
        assert np.all(~mask | _all_bracketing(env, v, mu, tau, mass))
        gain = env.nominal(mask)
        if gain is not None:
            assert env.contains(mask, gain)


def _all_bracketing(env: Envelope, v: float, mu: float, tau: float, mass: float) -> np.ndarray:
    out = np.ones(env.verified.shape[4:], dtype=bool)
    brackets = [
        env._bracket(n, x)
        for n, x in zip(("speed", "mu", "tau", "mass"), (v, mu, tau, mass), strict=True)
    ]
    for iv in {brackets[0][0], brackets[0][1]}:  # type: ignore[index]
        for imu in {brackets[1][0], brackets[1][1]}:  # type: ignore[index]
            for it in {brackets[2][0], brackets[2][1]}:  # type: ignore[index]
                for im in {brackets[3][0], brackets[3][1]}:  # type: ignore[index]
                    out &= env.verified[iv, imu, it, im]
    return out


def test_manifest_lists_assumptions_that_match_the_assumptions_document() -> None:
    manifest = json.loads(META.read_text(encoding="utf-8"))
    doc = (REPO_ROOT / "docs" / "ASSUMPTIONS.md").read_text(encoding="utf-8")
    for key in manifest["assumptions"]:
        assert f"| {key} |" in doc
    assert set(manifest["assumptions"]) == set(load_config("envelope.yaml")["assumptions"])
    assert manifest["assumptions"] == load_config("envelope.yaml")["assumptions"]
    assert math.isfinite(manifest["counts"]["mean_verified_gains_per_cell"])
