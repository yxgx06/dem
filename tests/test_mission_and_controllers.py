from __future__ import annotations

import json

import numpy as np
import pytest

from egga.config import REPO_ROOT, load_config
from egga.controllers.base import Observation
from egga.controllers.rl_actor import RLScheduler, load_rl_weights
from egga.eval.simulate import Scenario, run_mission
from egga.scenarios.mission import build_mission

GOLDEN = REPO_ROOT / "results" / "phase0" / "matlab_golden.json"


def test_mission_has_75_seconds_at_100_hz() -> None:
    m = build_mission()
    assert m.n == 7501
    assert m.t[-1] == pytest.approx(75.0)


def test_mission_stage_values() -> None:
    m = build_mission()

    def at(s: float) -> int:
        return int(round(s / m.dt))

    assert m.mu[at(5)] == 0.85 and m.slope_deg[at(5)] == 0.0
    assert m.slope_deg[at(29.9)] == pytest.approx(10.0, abs=0.01)
    assert m.mu[at(44.9)] == pytest.approx(0.5, abs=0.01)
    assert m.slope_deg[at(50)] == -10.0
    assert m.mu[at(59.9)] == pytest.approx(0.25, abs=0.01)
    assert m.mu[at(75)] == pytest.approx(0.60)
    assert m.slope_deg[at(75)] == 0.0


def test_ice_stage_reference_is_infeasible_at_12_mps_but_not_10() -> None:
    cfg = load_config("mission.yaml")
    g = float(load_config("vehicle.yaml")["gravity_mps2"])
    ratios = {}
    for vx in (10.0, 12.0):
        m = build_mission(vx=vx, cfg=cfg)
        ice = (m.t >= 45.0) & (m.t < 60.0)
        demand = np.max(np.abs(vx * m.r_ref[ice]))
        ratios[vx] = demand / (0.25 * g)
    assert ratios[10.0] < 1.0
    assert ratios[12.0] > 1.0


def test_run_is_deterministic_for_a_seed() -> None:
    sc = Scenario(noise_std_m=0.005, seed=3)
    a = run_mission("pd_ff", sc)
    b = run_mission("pd_ff", sc)
    np.testing.assert_array_equal(a.ey, b.ey)


def test_different_noise_seeds_give_different_runs() -> None:
    a = run_mission("pd_ff", Scenario(noise_std_m=0.005, seed=1))
    b = run_mission("pd_ff", Scenario(noise_std_m=0.005, seed=2))
    assert not np.array_equal(a.ey, b.ey)


def test_divergence_is_detected_and_stops_the_run() -> None:
    run = run_mission("pid", Scenario(delay_s=0.12))
    assert run.diverged_at_s is not None
    assert np.isnan(run.ey[-1])


def test_unknown_controller_rejected() -> None:
    with pytest.raises(KeyError):
        run_mission("nope", Scenario())


def test_weight_sets_differ_and_have_expected_shapes() -> None:
    hand = load_rl_weights("handtyped")
    trained = load_rl_weights("trained")
    assert hand.W1.shape == trained.W1.shape == (8, 6)
    n_params = sum(a.size for a in (hand.W1, hand.b1, hand.W2, hand.b2))
    assert n_params == 92
    assert not np.allclose(hand.W1, trained.W1)


def test_unknown_weight_set_rejected() -> None:
    with pytest.raises(KeyError):
        load_rl_weights("missing")


def test_rl_gains_respect_bounds_before_rules() -> None:
    cfg = load_config("controllers.yaml")["rl"]
    ctrl = RLScheduler(load_rl_weights("handtyped"), cfg, 0.01, 2.8, 10.0)
    obs = Observation(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.85)
    kp, ki, kd, kh = ctrl.command(obs).gains
    assert 0.4 <= kp <= 2.2 and 0.01 <= ki <= 0.15 and 0.02 <= kd <= 0.45 and 0.6 <= kh <= 1.8


@pytest.mark.parametrize("controller", ["pid", "pd_ff", "rl_handtyped"])
def test_python_port_matches_matlab_golden(controller: str) -> None:
    assert GOLDEN.exists(), "regenerate with matlab/crosscheck.m (make matlab-golden)"
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))[controller]
    run = run_mission(controller, Scenario())
    assert np.nanmax(np.abs(run.ey)) * 100 == pytest.approx(golden["max_abs_ey_cm"], abs=0.01)
    rms = float(np.sqrt(np.mean(run.ey**2)) * 100)
    assert rms == pytest.approx(golden["rms_ey_cm"], abs=0.01)
    np.testing.assert_allclose(run.ey[::100], np.asarray(golden["ey_m_every_1s"]), atol=1e-4)
