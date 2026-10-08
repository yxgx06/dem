from __future__ import annotations

import math

from egga.supervisor import guard, health, modes
from egga.supervisor.envelope import Envelope, Gain
from egga.supervisor.gains import Index, gain_bounds, next_index
from egga.supervisor.types import (
    FLAG_ACTUATOR,
    FLAG_CAPPED,
    FLAG_DOMAIN,
    FLAG_ESTIMATOR,
    FLAG_INVALID,
    FLAG_MARGIN,
    FLAG_NO_SET,
    FLAG_SATURATION,
    FLAG_YAW,
    LOG_CAPACITY,
    Config,
    EstStatus,
    Inputs,
    Mode,
    Outputs,
    Reason,
    State,
)

_LATCHED = (int(Mode.FALLBACK), int(Mode.MINIMAL_RISK))


def log_event(state: State, t: float, code: int, value: float) -> None:
    """Append to the fixed-size ring buffer (oldest entries are overwritten)."""
    i = state.log_n % LOG_CAPACITY
    state.log_t[i] = t
    state.log_code[i] = code
    state.log_val[i] = value
    state.log_n += 1


def _edge(state: State, t: float, flags: int, bit: int, code: int, value: float) -> None:
    if flags & bit and not state.prev_flags & bit:
        log_event(state, t, code, value)


def _finite_inputs(inp: Inputs) -> bool:
    values = (
        inp.speed, inp.speed_request, inp.curvature_ahead, inp.mu_lo, inp.mu_hi, inp.tau_bar,
        inp.mass_hi, inp.quality, inp.yaw_rate, inp.steer_meas, inp.steer_cmd, inp.e_abs,
        inp.e_rate_abs,
    )  # fmt: skip
    return all(math.isfinite(v) for v in values) and inp.speed > 0.0


def step(state: State, cfg: Config, env: Envelope, inp: Inputs) -> Outputs:
    """One supervisor tick. Mutates `state` in place; everything else is a pure function of the
    inputs. The supervisor never commands steering: it selects verified gains, a speed command and
    the demand limits that the caller must apply to the classical controller's command.
    """
    time_ok = math.isfinite(inp.t) and (math.isnan(state.last_t) or inp.t > state.last_t)
    invalid = not (time_ok and _finite_inputs(inp))
    if time_ok:
        dt = cfg.dt if math.isnan(state.last_t) else inp.t - state.last_t
        t = inp.t
        state.last_t = t
    else:
        dt = cfg.dt
        t = 0.0 if math.isnan(state.last_t) else state.last_t

    safe = env.min_feasible_speed
    if invalid:
        speed, request, curvature = safe, safe, 0.0
        mu_lo, mu_hi, tau, mass_hi, quality = (
            cfg.mu_floor,
            1.0,
            cfg.tau_prior,
            cfg.mass_prior_max,
            0.0,
        )
        est_status = int(EstStatus.INVALID)
        yaw_rate = steer_meas = steer_cmd = e_abs = e_rate = 0.0
    else:
        speed, request, curvature = inp.speed, inp.speed_request, inp.curvature_ahead
        mu_lo, mu_hi, tau, mass_hi, quality = (
            inp.mu_lo,
            inp.mu_hi,
            inp.tau_bar,
            inp.mass_hi,
            inp.quality,
        )
        est_status = inp.est_status
        yaw_rate, steer_meas, steer_cmd = inp.yaw_rate, inp.steer_meas, inp.steer_cmd
        e_abs, e_rate = inp.e_abs, inp.e_rate_abs

    mu_top = float(env.axes["mu"][-1])
    mu_eff = min(max(mu_lo, cfg.mu_floor), mu_top)
    conservative = state.mode in _LATCHED
    tau_q = max(tau, cfg.tau_fallback) if conservative else tau
    mass_q = max(mass_hi, cfg.mass_fallback) if conservative else mass_hi
    mass_q = max(mass_q, 1.0)

    ay_max = env.ay_max(mu_eff)
    steer_limit = guard.steer_angle_limit(cfg, speed, ay_max, mass_q)

    if not invalid:
        health.push_command(state, steer_cmd)
        yaw_now = health.yaw_residual_bad(state, cfg, speed, yaw_rate, steer_meas, mass_q, dt)
        state.yaw_bad = health.accumulate(state.yaw_bad, yaw_now, dt)
        act_now = health.actuator_residual_bad(state, cfg, steer_meas, tau_q, dt)
        state.act_bad = health.accumulate(state.act_bad, act_now, dt)
        sat_now = health.saturation_bad(cfg, steer_cmd, steer_limit)
        state.sat_bad = health.accumulate(state.sat_bad, sat_now, dt)
        stale_now = est_status >= int(EstStatus.STALE)
        state.est_bad = health.accumulate_leaky(state.est_bad, stale_now, dt)
    yaw_flag = state.yaw_bad >= cfg.yaw_persist
    act_flag = state.act_bad >= cfg.actuator_persist
    sat_flag = state.sat_bad >= cfg.saturation_persist
    est_flag = state.est_bad >= cfg.stale_grace
    hard_fault = invalid or yaw_flag or act_flag or sat_flag or est_flag
    cap = env.speed_cap(curvature, mu_eff)
    v_ver = env.max_verified_speed(mu_eff, tau_q, mass_q)
    domain_violation = (not invalid) and mu_hi < cfg.mu_floor
    no_set = domain_violation or cap is None or v_ver is None

    state.healthy_for = 0.0 if (hard_fault or no_set) else state.healthy_for + dt

    previous_mode = state.mode
    reset_ok = inp.request_reset and not invalid and state.healthy_for >= cfg.stale_grace - 1e-4
    mode, reason = modes.next_mode(
        state, cfg, t, no_set, hard_fault, tau, mu_hi, quality, reset_ok
    )
    if mode != previous_mode:
        state.mode = mode
        state.mode_entry_t = t
        state.mode_changes += 1
        log_event(state, t, reason, float(mode))

    rl_ok = False
    if inp.rl_valid and not invalid:
        lower, upper = gain_bounds(env)
        rl_reason = health.rl_check(state, cfg, inp.rl_gain, dt, lower, upper)
        if rl_reason != int(Reason.NONE):
            state.rl_rejects += 1
            if state.rl_rejects in (1, cfg.rl_persist_rejects):
                log_event(state, t, rl_reason, float(state.rl_rejects))
        else:
            state.rl_rejects = 0
        rl_ok = rl_reason == int(Reason.NONE) and state.mode not in _LATCHED

    margin_flag = guard.lateral_margin_violated(cfg, e_abs, e_rate, ay_max, tau)
    scale = cfg.speed_scale[state.mode] * (cfg.margin_speed_scale if margin_flag else 1.0)
    if cap is None or v_ver is None or state.mode == int(Mode.MINIMAL_RISK):
        target = safe
    else:
        target = max(safe, min(request * scale, cap, v_ver))
    if math.isnan(state.speed_cmd):
        state.speed_cmd = speed
    state.speed_cmd += max(-cfg.decel * dt, min(cfg.accel * dt, target - state.speed_cmd))
    state.speed_cmd = max(safe, state.speed_cmd)

    mask = env.cell_mask(max(state.speed_cmd, speed), mu_eff, tau_q, mass_q)
    if (mask is None or not bool(mask.any())) and v_ver is not None:
        mask = env.cell_mask(v_ver, mu_eff, tau_q, mass_q)
    have_mask = mask is not None and bool(mask.any())
    forced = False
    if mask is not None and have_mask:
        proposal: Gain = inp.rl_gain if rl_ok else inp.reference_gain
        target_idx = env.project_index(mask, proposal)
        assert target_idx is not None
        current: Index | None = None
        if state.have_gain:
            current = (
                int(state.gain_idx[0]), int(state.gain_idx[1]),
                int(state.gain_idx[2]), int(state.gain_idx[3]),
            )  # fmt: skip
        may_hop = t - state.last_hop_t >= cfg.gain_hop_period
        new_idx, forced = next_index(env, mask, current, target_idx, may_hop)
        if current is None or new_idx != current:
            state.last_hop_t = t
        for i in range(4):
            state.gain_idx[i] = new_idx[i]
        if forced and current is not None:
            log_event(state, t, int(Reason.GAIN_FORCED), 1.0)
        state.have_gain = True
    if state.have_gain:
        applied: Gain = env.gain_at(
            (
                int(state.gain_idx[0]),
                int(state.gain_idx[1]),
                int(state.gain_idx[2]),
                int(state.gain_idx[3]),
            )  # fmt: skip
        )
    else:
        applied = inp.reference_gain

    actual_mask = env.cell_mask(speed, mu_eff, tau_q, mass_q)
    verified = (
        actual_mask is not None
        and bool(actual_mask.any())
        and state.have_gain
        and env.contains(actual_mask, applied)
    )

    rate_limit = env.rate_limit(max(speed, state.speed_cmd), mass_q)
    flags = (
        (FLAG_YAW if yaw_flag else 0)
        | (FLAG_ACTUATOR if act_flag else 0)
        | (FLAG_SATURATION if sat_flag else 0)
        | (FLAG_ESTIMATOR if est_flag else 0)
        | (FLAG_MARGIN if margin_flag else 0)
        | (FLAG_INVALID if invalid else 0)
        | (FLAG_NO_SET if no_set else 0)
        | (FLAG_DOMAIN if domain_violation else 0)
        | (FLAG_CAPPED if target < request - 1e-9 else 0)
    )
    for bit, code, value in (
        (FLAG_YAW, Reason.YAW_RESIDUAL, state.yaw_bad),
        (FLAG_ACTUATOR, Reason.ACTUATOR_RESIDUAL, state.act_bad),
        (FLAG_SATURATION, Reason.SATURATION, state.sat_bad),
        (FLAG_ESTIMATOR, Reason.ESTIMATOR_FAULT, state.est_bad),
        (FLAG_MARGIN, Reason.LATERAL_MARGIN, e_abs),
        (FLAG_INVALID, Reason.INVALID_INPUT, 0.0),
        (FLAG_DOMAIN, Reason.DOMAIN_VIOLATION, mu_hi),
        (FLAG_CAPPED, Reason.SPEED_CAPPED, target),
    ):
        _edge(state, t, flags, bit, int(code), value)
    state.prev_flags = flags

    return Outputs(
        gains=applied,
        speed_cmd=state.speed_cmd,
        steer_limit=steer_limit,
        rate_limit=cfg.steer_rate_hw_max if rate_limit is None else rate_limit,
        mode=state.mode,
        rl_applied=rl_ok and verified,
        forced_gain_jump=forced,
        verified=verified,
        flags=flags,
    )
