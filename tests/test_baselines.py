from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from egga.config import CONFIG_DIR, load_baselines, load_config, set_baseline_override
from egga.controllers.base import Observation
from egga.controllers.lqr import (
    DesignVehicle,
    LQRController,
    discretize,
    error_model,
    lqr_gain,
)
from egga.controllers.mpc import MPCController
from egga.eval.baselines import BASELINES, audit_lqr_reproduction
from egga.eval.metrics2 import run_metrics, tuning_cost
from egga.eval.simulate import Scenario, make_controller, run_mission
from egga.scenarios.mission import build_mission
from egga.scenarios.sets import load_scenarios, set_hash

DT = 0.01
VEHICLE = load_config("vehicle.yaml")
OBS0 = Observation(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, vx=10.0)


def design(scale: float = 1.0, follows: bool = False) -> DesignVehicle:
    return DesignVehicle.from_config(VEHICLE, scale, follows)


# ---------------------------------------------------------------- LQR
@pytest.mark.parametrize("speed", [5.0, 10.0, 15.0, 20.0])
@pytest.mark.parametrize("delay_steps", [0, 3, 10])
def test_lqr_closed_loop_is_stable(speed: float, delay_steps: int) -> None:
    d = design()
    q = [0.05, 0.0, 1.0, 0.0]
    k = lqr_gain(d, speed, q, 1.0, DT, delay_steps)
    ad, bd = discretize(*error_model(d, speed), DT)
    n = 4 + delay_steps
    if delay_steps == 0:
        closed = ad - bd @ k[None, :]
    else:
        az = np.zeros((n, n))
        bz = np.zeros((n, 1))
        az[:4, :4] = ad
        az[:4, n - 1 : n] = bd
        bz[4, 0] = 1.0
        for i in range(1, delay_steps):
            az[4 + i, 4 + i - 1] = 1.0
        closed = az - bz @ k[None, :]
    assert k.shape == (n,)
    assert np.max(np.abs(np.linalg.eigvals(closed))) < 1.0


def test_delay_augmented_design_is_not_worse_under_true_delay_in_linear_simulation() -> None:
    d, v, n_delay = design(), 10.0, 10
    q = [0.05, 0.0, 1.0, 0.0]
    ad, bd = discretize(*error_model(d, v), DT)
    plain = lqr_gain(d, v, q, 1.0, DT, 0)
    aug = lqr_gain(d, v, q, 1.0, DT, n_delay)

    def cost(use_aug: bool) -> float:
        x = np.array([0.1, 0.0, 0.0, 0.0])
        hist = [0.0] * n_delay
        total = 0.0
        for _ in range(600):
            u = -float(aug @ np.concatenate([x, hist])) if use_aug else -float(plain @ x)
            applied = hist[-1]
            hist = [u, *hist[:-1]]
            x = ad @ x + bd[:, 0] * applied
            total += q[0] * x[0] ** 2 + q[2] * x[2] ** 2 + u * u
        return total

    assert cost(True) <= cost(False) * 1.01


def test_lqr_steer_limit_grows_with_friction_and_falls_with_speed() -> None:
    cfg = load_baselines()["lqr"]
    ctrl = LQRController("t", design(), cfg, DT)
    assert ctrl.steer_limit(10.0, 0.8) > ctrl.steer_limit(10.0, 0.3)
    assert ctrl.steer_limit(8.0, 0.8) > ctrl.steer_limit(14.0, 0.8)


def test_lqr_output_is_clipped_to_the_friction_limit() -> None:
    cfg = load_baselines()["lqr"]
    ctrl = LQRController("t", design(), cfg, DT)
    big = Observation(2.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.3, vx=10.0)
    limit = min(ctrl.steer_limit(10.0, 0.3), 0.5)
    assert abs(ctrl.command(big).steer) == pytest.approx(limit)


def test_lqr_feedforward_only_follows_reference_yaw_rate() -> None:
    cfg = load_baselines()["lqr"]
    ctrl = LQRController("t", design(), cfg, DT)
    cmd = ctrl.command(Observation(0.0, 0.0, 0.0, 0.2, 0.2, 0.0, 0.9, vx=10.0))
    assert cmd.steer > 0.0


def test_true_mass_and_nominal_mass_designs_differ_only_when_mass_changes() -> None:
    mission = build_mission()
    same_a = make_controller("b7_lqr_true_mass", mission, DT, mass_scale=1.0)
    same_b = make_controller("b7_lqr_nominal_mass", mission, DT, mass_scale=1.0)
    heavy_a = make_controller("b7_lqr_true_mass", mission, DT, mass_scale=1.9)
    heavy_b = make_controller("b7_lqr_nominal_mass", mission, DT, mass_scale=1.9)
    obs = Observation(0.05, 0.0, 0.01, 0.0, 0.0, 0.0, 0.8, vx=10.0)
    assert same_a.command(obs).steer == pytest.approx(same_b.command(obs).steer)
    assert heavy_a.command(obs).steer != pytest.approx(heavy_b.command(obs).steer)


def test_audit_lqr_claims_reproduce_within_tolerance() -> None:
    df = audit_lqr_reproduction().set_index("case")
    assert abs(df.loc["nominal", "measured_max_cm"] - 0.87) < 0.15
    assert abs(df.loc["mass_x1.3_nominal_design", "measured_max_cm"] - 3.0) < 0.3
    assert abs(df.loc["mass_x1.9_nominal_design", "measured_max_cm"] - 8.8) < 0.5
    assert not bool(df.loc["delay_200ms", "diverged"])


def test_nominal_mass_design_degrades_with_mass_and_true_mass_does_not() -> None:
    blind = run_mission("b7_lqr_nominal_mass", Scenario(mass_scale=1.9))
    aware = run_mission("b7_lqr_true_mass", Scenario(mass_scale=1.9))
    assert np.nanmax(np.abs(blind.ey)) > 3.0 * np.nanmax(np.abs(aware.ey))


# ---------------------------------------------------------------- MPC
def mpc(**over: float) -> MPCController:
    cfg = load_baselines()["mpc"] | over
    return MPCController(
        design(), cfg, DT, float(VEHICLE["delta_max_rad"]), float(VEHICLE["steer_rate_max_rps"])
    )


def test_mpc_is_zero_at_zero_error_and_reference() -> None:
    assert abs(mpc().command(OBS0).steer) < 1e-4


def test_mpc_steers_toward_positive_error_and_respects_box_limit() -> None:
    ctrl = mpc()
    cmd = ctrl.command(Observation(0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, vx=10.0))
    assert cmd.steer > 0.0
    huge = mpc().command(Observation(5.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.3, vx=10.0))
    assert abs(huge.steer) <= mpc().steer_limit(10.0, 0.3) + 1e-9


def test_mpc_respects_rate_limit_between_solves() -> None:
    ctrl = mpc()
    step = float(VEHICLE["steer_rate_max_rps"]) * float(load_baselines()["mpc"]["ts_s"])
    prev = 0.0
    for _ in range(60):
        u = ctrl.command(Observation(0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, vx=10.0)).steer
        assert abs(u - prev) <= step + 1e-6
        prev = u


def test_mpc_solves_once_per_sample_and_reports_time() -> None:
    ctrl = mpc()
    times = [ctrl.command(OBS0).solve_time_s for _ in range(50)]
    stride = int(round(float(load_baselines()["mpc"]["ts_s"]) / DT))
    assert sum(t > 0 for t in times) == int(np.ceil(50 / stride))
    assert ctrl.solve_failures == 0


def test_mpc_uses_reference_preview_for_feedforward() -> None:
    ctrl = mpc()
    flat = ctrl.command(OBS0).steer
    ctrl2 = mpc()
    curved = Observation(
        0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.8, vx=10.0, r_ref_preview=tuple([0.2] * 15)
    )
    assert ctrl2.command(curved).steer > flat + 0.01


def test_mpc_reset_clears_state() -> None:
    ctrl = mpc()
    ctrl.command(Observation(0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, vx=10.0))
    ctrl.reset()
    assert ctrl.solve_times == [] and ctrl.command(OBS0).steer == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------- B0 / B1 / factory / hooks
def test_b0_adds_curvature_feedforward_to_the_pid() -> None:
    mission = build_mission()
    b0 = make_controller("b0_pid_ff", mission, DT)
    obs = Observation(0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.8, vx=10.0)
    assert b0.command(obs).steer == pytest.approx(mission.wheelbase / 10.0 * 0.1)


def test_all_named_baselines_build_and_unknown_is_rejected() -> None:
    mission = build_mission()
    for name in BASELINES:
        assert make_controller(name, mission, DT) is not None
    with pytest.raises(KeyError):
        make_controller("b9_magic", mission, DT)


def test_baseline_override_hook_changes_parameters_and_is_reversible() -> None:
    base = load_baselines()["pid_ff"]["kp"]
    set_baseline_override({"pid_ff": {"kp": base * 2}})
    try:
        assert load_baselines()["pid_ff"]["kp"] == pytest.approx(base * 2)
    finally:
        set_baseline_override(None)
    assert load_baselines()["pid_ff"]["kp"] == pytest.approx(base)


def test_tuned_file_was_fitted_on_the_frozen_train_set_only() -> None:
    path = CONFIG_DIR / "baselines_tuned.yaml"
    assert path.exists(), "run python -m egga.eval.tuning"
    header = path.read_text(encoding="utf-8").splitlines()[1]
    assert set_hash(load_scenarios("train")) in header


def test_tuning_module_never_touches_val_or_test() -> None:
    from egga.config import REPO_ROOT

    text = (REPO_ROOT / "src" / "egga" / "eval" / "tuning.py").read_text(encoding="utf-8")
    assert 'load_scenarios("train")' in text
    assert 'load_scenarios("val")' not in text and "load_scenarios('val')" not in text
    assert '"test"' not in text.replace('"test set"', "")


# ---------------------------------------------------------------- metrics
def test_run_metrics_are_internally_consistent() -> None:
    from egga.eval.baselines import run_scenario

    spec = load_scenarios("val")[0]
    m = run_metrics(run_scenario("b1_pd_ff", spec), spec["mu_min"])
    assert m["rms_ey_cm"] <= m["max_abs_ey_cm"]
    assert m["p95_abs_ey_cm"] <= m["max_abs_ey_cm"] + 1e-9
    assert m["worst1pct_ey_cm"] <= m["max_abs_ey_cm"] + 1e-9
    assert 0.0 <= m["saturation_frac"] <= 1.0
    assert tuning_cost(m) >= m["rms_ey_cm"]


def test_tuning_cost_penalises_divergence() -> None:
    ok = {"rms_ey_cm": 1.0, "max_abs_ey_cm": 2.0, "diverged": False}
    bad = dict(ok, diverged=True)
    assert tuning_cost(bad) == tuning_cost(ok) + 100.0


def test_phase2_tuning_log_lists_only_train_evaluations() -> None:
    from egga.config import REPO_ROOT

    log = pd.read_parquet(REPO_ROOT / "results" / "phase2" / "tuning_log.parquet")
    assert set(log["train_set_hash"]) == {set_hash(load_scenarios("train"))}
    assert (log["cost"].groupby(log["target"]).min() <= log.groupby("target")["cost"].first()).all()
