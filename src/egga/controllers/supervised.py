from __future__ import annotations

from typing import Any

from egga.controllers.base import Command, Observation, clip
from egga.controllers.classical import PIDFeedforwardController
from egga.controllers.derivative import FilteredDerivative
from egga.supervisor.core import step
from egga.supervisor.envelope import Envelope
from egga.supervisor.types import Config, EstStatus, Inputs, State

_STATUS = {
    "ok": int(EstStatus.OK),
    "low_excitation": int(EstStatus.LOW_EXCITATION),
    "stale": int(EstStatus.STALE),
    "invalid": int(EstStatus.INVALID),
}


class SupervisedController:
    """B4: classical PID + curvature feedforward whose gains and speed are set by the supervisor.

    The proposal is the reference gain (no RL), projected onto the verified set. The steering
    command is limited by the envelope guard (steering angle for a_y,max and the rate limit).
    The supervisor needs an estimator: obs.estimate is the latest Estimate; without one the
    supervisor sees an invalid estimator and falls back.
    """

    name = "b4_supervised"
    governs_speed = True

    def __init__(
        self,
        pid_cfg: dict[str, Any],
        wheelbase: float,
        vx: float,
        dt: float,
        cutoff_hz: float | None,
        sup_cfg: Config,
        envelope: Envelope,
    ) -> None:
        self._dt = dt
        self._cfg = sup_cfg
        self._env = envelope
        self._pid = PIDFeedforwardController(pid_cfg, dt, wheelbase, vx, cutoff_hz)
        self._rate = FilteredDerivative(cutoff_hz or 5.0, dt)
        ref = envelope.reference_gain
        self._reference = (float(ref[0]), float(ref[1]), float(ref[2]), float(ref[3]))
        self.reset()

    def reset(self) -> None:
        self._pid.reset()
        self._rate.reset()
        self.state = State()
        self._prev_cmd = 0.0
        self._t = 0.0

    def command(self, obs: Observation) -> Command:
        est = obs.estimate
        e_rate = abs(self._rate.update(obs.e_y))
        if est is None:
            mu_lo, mu_hi, tau, mass_hi, quality, status = 0.1, 1.0, 0.15, 1.9, 0.0, 3
        else:
            mu_lo, mu_hi, tau = est.mu_lo, est.mu_hi, est.tau_bar
            mass_hi, quality, status = est.mass_hi, est.quality, _STATUS[est.status]
        speed = obs.vx if obs.vx > 0.0 else 1.0
        inputs = Inputs(
            t=self._t,
            speed=speed,
            speed_request=obs.speed_request if obs.speed_request > 0.0 else speed,
            curvature_ahead=obs.curvature_ahead,
            mu_lo=mu_lo,
            mu_hi=mu_hi,
            tau_bar=tau,
            mass_hi=mass_hi,
            quality=quality,
            est_status=status,
            yaw_rate=obs.yaw_rate,
            steer_meas=obs.steer_meas,
            steer_cmd=self._prev_cmd,
            e_abs=abs(obs.e_y),
            e_rate_abs=e_rate,
            rl_valid=False,
            rl_gain=self._reference,
            reference_gain=self._reference,
        )
        self._t += self._dt
        out = step(self.state, self._cfg, self._env, inputs)
        self._pid.set_gains(*out.gains)
        raw = self._pid.command(obs).steer
        cmd = clip(raw, -out.steer_limit, out.steer_limit)
        self._prev_cmd = cmd
        return Command(steer=cmd, gains=out.gains, speed_cmd=out.speed_cmd, mode=out.mode)
