from __future__ import annotations

import ctypes
import json

import numpy as np

from egga.c_bridge import CEggaInputs, CEggaOutputs, CEggaState, load_c_lib, py_config_to_c
from egga.config import REPO_ROOT, load_config
from egga.eval.envelope_build import ENVELOPE_DIR
from egga.rl.model import ActorCritic
from egga.supervisor.core import step as py_supervisor_step
from egga.supervisor.envelope import Envelope
from egga.supervisor.types import Config as PyConfig
from egga.supervisor.types import Inputs as PyInputs
from egga.supervisor.types import State as PyState


def test_actor_c99_matches_python_actor() -> None:
    lib = load_c_lib()

    with open(REPO_ROOT / "results" / "phase6" / "b5_best.json", encoding="utf-8") as f:
        weights = json.load(f)
    py_model = ActorCritic.from_weights_dict(weights)

    rng = np.random.default_rng(12345)
    max_diff = 0.0

    import torch
    scale = np.array([0.6, 0.04, 0.15, 0.6], dtype=np.float32)

    for _ in range(1000):
        obs = rng.uniform(-5.0, 5.0, size=6).astype(np.float32)
        c_obs = (ctypes.c_float * 6)(*obs)
        c_delta_K = (ctypes.c_float * 4)()

        lib.egga_actor_predict(c_obs, c_delta_K)
        c_res = np.array([c_delta_K[i] for i in range(4)], dtype=np.float32)

        action, _, _ = py_model.get_action(torch.from_numpy(obs).float(), deterministic=True)
        py_delta_K = action.detach().cpu().numpy() * scale

        diff = np.max(np.abs(c_res - py_delta_K))
        if diff > max_diff:
            max_diff = float(diff)

    assert max_diff < 1e-6, f"Actor C99 max diff {max_diff:.8e} exceeds 1e-6 tolerance"


def test_supervisor_c99_matches_python_supervisor_step_by_step() -> None:
    lib = load_c_lib()
    sup_raw = load_config("supervisor.yaml")
    veh_raw = load_config("vehicle.yaml")
    py_cfg = PyConfig.from_dict(sup_raw, veh_raw)
    c_cfg = py_config_to_c(py_cfg)
    env = Envelope.load(ENVELOPE_DIR / "envelope_v1.npz", ENVELOPE_DIR / "envelope_v1.json")

    py_state = PyState()
    c_state = CEggaState()
    lib.egga_supervisor_init(ctypes.byref(c_state))

    ref = env.reference_gain
    ref_tuple = (float(ref[0]), float(ref[1]), float(ref[2]), float(ref[3]))

    rng = np.random.default_rng(42)
    dt = 0.01

    for step_idx in range(5000):
        t_clean = round((step_idx + 1) * dt, 4)

        # Generate varying operational conditions
        speed = float(rng.uniform(6.0, 14.0))
        speed_req = float(rng.uniform(8.0, 12.0))
        curvature = float(rng.choice([0.0, 0.01, 0.03, -0.02]))
        mu_lo = float(rng.uniform(0.25, 0.85))
        mu_hi = mu_lo + float(rng.uniform(0.05, 0.2))
        tau = float(rng.uniform(0.01, 0.09))
        mass = float(rng.uniform(1.0, 1.5))
        quality = float(rng.uniform(0.5, 1.0))
        yaw_rate = float(rng.uniform(-0.1, 0.1))
        steer_meas = float(rng.uniform(-0.05, 0.05))
        steer_cmd = steer_meas
        e_abs = float(rng.uniform(0.0, 0.15))
        e_rate_abs = float(rng.uniform(0.0, 0.1))

        rl_prop = (
            ref_tuple[0] + float(rng.uniform(-0.3, 0.3)),
            ref_tuple[1] + float(rng.uniform(-0.01, 0.01)),
            ref_tuple[2] + float(rng.uniform(-0.05, 0.05)),
            ref_tuple[3] + float(rng.uniform(-0.2, 0.2)),
        )

        py_in = PyInputs(
            t=t_clean,
            speed=speed,
            speed_request=speed_req,
            curvature_ahead=curvature,
            mu_lo=mu_lo,
            mu_hi=mu_hi,
            tau_bar=tau,
            mass_hi=mass,
            quality=quality,
            est_status=0,
            yaw_rate=yaw_rate,
            steer_meas=steer_meas,
            steer_cmd=steer_cmd,
            e_abs=e_abs,
            e_rate_abs=e_rate_abs,
            rl_valid=True,
            rl_gain=rl_prop,
            reference_gain=ref_tuple,
            request_reset=False,
        )

        c_in = CEggaInputs()
        c_in.t = t_clean
        c_in.speed = speed
        c_in.speed_request = speed_req
        c_in.curvature_ahead = curvature
        c_in.mu_lo = mu_lo
        c_in.mu_hi = mu_hi
        c_in.tau_bar = tau
        c_in.mass_hi = mass
        c_in.quality = quality
        c_in.est_status = 0
        c_in.yaw_rate = yaw_rate
        c_in.steer_meas = steer_meas
        c_in.steer_cmd = steer_cmd
        c_in.e_abs = e_abs
        c_in.e_rate_abs = e_rate_abs
        c_in.rl_valid = True
        for i in range(4):
            c_in.rl_gain[i] = rl_prop[i]
            c_in.reference_gain[i] = ref_tuple[i]
        c_in.request_reset = False

        c_out = CEggaOutputs()
        lib.egga_supervisor_step(
            ctypes.byref(c_state),
            ctypes.byref(c_cfg),
            ctypes.byref(c_in),
            ctypes.byref(c_out),
        )

        py_out = py_supervisor_step(py_state, py_cfg, env, py_in)

        # Check discrete fields (mode, verified, flags, forced_jump)
        msg_mode = f"Step {step_idx}: mode mismatch {c_out.mode} vs {py_out.mode}"
        assert c_out.mode == int(py_out.mode), msg_mode
        msg_flags = f"Step {step_idx}: flags mismatch {c_out.flags} vs {py_out.flags}"
        assert c_out.flags == py_out.flags, msg_flags
        assert c_out.verified == py_out.verified, f"Step {step_idx}: verified mismatch"
        assert c_out.forced_gain_jump == py_out.forced_gain_jump, f"Step {step_idx}: forced jump"

        # Check continuous outputs
        diff_speed = abs(c_out.speed_cmd - py_out.speed_cmd)
        assert diff_speed < 5e-4, f"Step {step_idx}: speed_cmd mismatch {diff_speed}"
        diff_steer = abs(c_out.steer_limit - py_out.steer_limit)
        assert diff_steer < 1e-4, f"Step {step_idx}: steer_limit mismatch {diff_steer}"
        diff_rate = abs(c_out.rate_limit - py_out.rate_limit)
        assert diff_rate < 1e-4, f"Step {step_idx}: rate_limit mismatch {diff_rate}"
        for i in range(4):
            diff_g = abs(c_out.gains[i] - py_out.gains[i])
            assert diff_g < 1e-4, f"Step {step_idx}: gain[{i}] mismatch {diff_g}"


def test_friction_circle_c99_matches_python() -> None:
    from egga.c_bridge import call_c_friction_circle
    from egga.supervisor.friction_circle import (
        FrictionCirclePolicy,
        FrictionDemand,
        allocate_friction_circle,
    )

    rng = np.random.default_rng(2026)
    policies = [
        FrictionCirclePolicy.STEERING_PRIORITY,
        FrictionCirclePolicy.BALANCED,
        FrictionCirclePolicy.BRAKING_PRIORITY,
    ]

    for _ in range(1000):
        policy = policies[int(rng.integers(0, 3))]
        ax = float(rng.uniform(-10.0, 10.0))
        ay = float(rng.uniform(-10.0, 10.0))
        mu = float(rng.uniform(0.15, 0.95))
        speed = float(rng.uniform(1.0, 25.0))
        wheelbase = float(rng.uniform(2.0, 4.0))

        py_demand = FrictionDemand(
            ax_req=ax,
            ay_req=ay,
            mu=mu,
            policy=policy,
        )
        py_alloc = allocate_friction_circle(py_demand, speed=speed, wheelbase=wheelbase)

        c_alloc = call_c_friction_circle(
            ax_req=ax,
            ay_req=ay,
            mu=mu,
            gravity=9.80665,
            safety_factor=0.95,
            policy=int(policy),
            speed=speed,
            wheelbase=wheelbase,
            steer_hw_max=0.60,
        )

        assert abs(c_alloc.ax_safe - py_alloc.ax_safe) < 1e-4
        assert abs(c_alloc.ay_safe - py_alloc.ay_safe) < 1e-4
        assert abs(c_alloc.utilisation - py_alloc.utilisation) < 1e-4
        assert bool(c_alloc.is_clamped) == py_alloc.is_clamped
        assert abs(c_alloc.delta_max_coupled - py_alloc.delta_max_coupled) < 1e-4


def test_jackknife_guard_c99_matches_python() -> None:
    from egga.c_bridge import call_c_jackknife_guard
    from egga.supervisor.jackknife_guard import (
        JackknifeConfig,
        JackknifeInputs,
        step_jackknife_guard,
    )

    rng = np.random.default_rng(2027)
    cfg = JackknifeConfig()

    for _ in range(1000):
        th_a = float(rng.uniform(-0.6, 0.6))
        th_a_dot = float(rng.uniform(-1.0, 1.0))
        vx = float(rng.uniform(2.0, 30.0))
        mu = float(rng.uniform(0.1, 0.95))
        ltr = float(rng.uniform(0.0, 1.0))
        steer_cmd = float(rng.uniform(-0.5, 0.5))
        steer_rate = float(rng.uniform(-0.6, 0.6))

        py_in = JackknifeInputs(
            theta_a=th_a,
            theta_a_dot=th_a_dot,
            vx=vx,
            mu=mu,
            ltr=ltr,
            steer_cmd_req=steer_cmd,
            steer_rate_req=steer_rate,
        )
        py_out = step_jackknife_guard(cfg, py_in)

        c_out = call_c_jackknife_guard(
            l2=cfg.l2,
            tau_air=cfg.tau_air,
            gravity=cfg.gravity,
            ltr_warning=cfg.ltr_warning,
            ltr_critical=cfg.ltr_critical,
            steer_rate_max=cfg.steer_rate_max,
            min_theta_crit=cfg.min_theta_crit,
            max_theta_crit=cfg.max_theta_crit,
            theta_a=th_a,
            theta_a_dot=th_a_dot,
            vx=vx,
            mu=mu,
            ltr=ltr,
            steer_cmd_req=steer_cmd,
            steer_rate_req=steer_rate,
        )

        assert abs(c_out.theta_crit - py_out.theta_crit) < 1e-4
        assert abs(c_out.h_jackknife - py_out.h_jackknife) < 1e-4
        assert bool(c_out.is_jackknife_critical) == py_out.is_jackknife_critical
        assert bool(c_out.is_rollover_critical) == py_out.is_rollover_critical
        assert abs(c_out.steer_rate_safe - py_out.steer_rate_safe) < 1e-4
        assert abs(c_out.steer_cmd_safe - py_out.steer_cmd_safe) < 1e-4
        assert abs(c_out.trailer_brake_pressure - py_out.trailer_brake_pressure) < 1e-4
        assert c_out.status_flags == py_out.status_flags


