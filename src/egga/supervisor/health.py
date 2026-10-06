from __future__ import annotations

import math

from egga.supervisor.types import HISTORY, Config, Reason, State


def accumulate(accumulated: float, bad: bool, dt: float) -> float:
    """Seconds the condition has persisted without interruption."""
    return accumulated + dt if bad else 0.0


def push_command(state: State, command: float) -> None:
    state.hist[state.hist_n % HISTORY] = command
    state.hist_n += 1


def command_range(state: State, window: int) -> tuple[float, float]:
    """Smallest and largest command among the last `window` samples (0, 0 when none yet)."""
    count = min(window, state.hist_n, HISTORY)
    if count <= 0:
        return 0.0, 0.0
    values = [float(state.hist[(state.hist_n - 1 - i) % HISTORY]) for i in range(count)]
    return min(values), max(values)


def yaw_residual_bad(
    state: State,
    cfg: Config,
    speed: float,
    yaw_rate: float,
    steer_meas: float,
    mass_hi: float,
    dt: float,
) -> bool:
    """Measured yaw rate vs the steady-state bicycle response to the measured steering.

    The prediction is low-passed (the real response lags). Returns True while the residual is
    above abs + rel * |predicted|; the caller applies the persistence time.
    """
    predicted = speed * steer_meas / (cfg.wheelbase + cfg.understeer_coeff(mass_hi) * speed * speed)
    alpha = dt / (cfg.yaw_lpf_tau + dt)
    state.yaw_pred += alpha * (predicted - state.yaw_pred)
    residual = abs(yaw_rate - state.yaw_pred)
    return speed >= cfg.yaw_min_speed and residual > cfg.yaw_abs + cfg.yaw_rel * abs(state.yaw_pred)


def actuator_residual_bad(
    state: State, cfg: Config, steer_meas: float, tau_bar: float, dt: float
) -> bool:
    """Measured steering outside the range of recent commands (a lag within tau_bar is allowed)."""
    window = int(round(tau_bar / dt)) + 1
    low, high = command_range(state, window)
    residual = max(0.0, low - steer_meas, steer_meas - high)
    return residual > cfg.actuator_abs


def saturation_bad(cfg: Config, steer_cmd: float, steer_limit: float) -> bool:
    return abs(steer_cmd) >= cfg.saturation_level * steer_limit


def rl_check(
    state: State,
    cfg: Config,
    gain: tuple[float, float, float, float],
    dt: float,
    lower: tuple[float, float, float, float],
    upper: tuple[float, float, float, float],
) -> int:
    """Reason code for an RL gain proposal: NONE when acceptable, otherwise why it is rejected."""
    finite = all(math.isfinite(g) for g in gain)
    in_range = finite and all(lo <= g <= hi for g, lo, hi in zip(gain, lower, upper, strict=True))
    if not in_range:
        state.last_rl_valid = False
        return int(Reason.RL_REJECTED_RANGE)
    too_fast = state.last_rl_valid and any(
        abs(g - float(prev)) / dt > rate
        for g, prev, rate in zip(gain, state.last_rl, cfg.rl_rate, strict=True)
    )
    for i in range(4):
        state.last_rl[i] = gain[i]
    state.last_rl_valid = True
    return int(Reason.RL_REJECTED_RATE) if too_fast else int(Reason.NONE)
