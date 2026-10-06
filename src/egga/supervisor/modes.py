from __future__ import annotations

from egga.supervisor.types import Config, Mode, Reason, State


def desired_mode(cfg: Config, current: int, tau: float, mu_hi: float, quality: float) -> int:
    """Non-latched mode implied by the estimates, with hysteresis around every threshold."""
    tau_threshold = (
        cfg.tau_degraded_exit if current == Mode.DEGRADED_ACTUATOR else cfg.tau_degraded_enter
    )
    if tau > tau_threshold:
        return int(Mode.DEGRADED_ACTUATOR)
    mu_threshold = cfg.mu_low_exit if current == Mode.LOW_MU else cfg.mu_low_enter
    if mu_hi < mu_threshold:
        return int(Mode.LOW_MU)
    quality_threshold = cfg.quality_exit if current == Mode.CAUTIOUS else cfg.quality_enter
    if quality < quality_threshold:
        return int(Mode.CAUTIOUS)
    return int(Mode.NOMINAL)


def next_mode(
    state: State,
    cfg: Config,
    t: float,
    no_set: bool,
    hard_fault: bool,
    tau: float,
    mu_hi: float,
    quality: float,
    reset: bool,
) -> tuple[int, int]:
    """(mode, reason) for this tick.

    Escalation to FALLBACK / MINIMAL_RISK is immediate. MINIMAL_RISK latches until an explicit reset
    with no fault, then goes to FALLBACK. FALLBACK latches until the loop has been healthy for
    `recover` seconds, then goes to CAUTIOUS. The other modes change only after `min_dwell`.
    """
    current = state.mode
    if current == Mode.MINIMAL_RISK:
        if reset and not no_set and not hard_fault:
            return int(Mode.FALLBACK), int(Reason.RESET)
        return current, int(Reason.NONE)
    if no_set:
        return int(Mode.MINIMAL_RISK), int(Reason.NO_VERIFIED_SET)
    if hard_fault:
        return int(Mode.FALLBACK), int(
            Reason.MODE_CHANGE if current != Mode.FALLBACK else Reason.NONE
        )
    if current == Mode.FALLBACK:
        if state.healthy_for >= cfg.recover:
            return int(Mode.CAUTIOUS), int(Reason.RECOVERY)
        return current, int(Reason.NONE)
    desired = desired_mode(cfg, current, tau, mu_hi, quality)
    if desired != current and t - state.mode_entry_t >= cfg.min_dwell:
        return desired, int(Reason.MODE_CHANGE)
    return current, int(Reason.NONE)
