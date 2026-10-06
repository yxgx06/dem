from __future__ import annotations

from typing import Any

from egga.controllers.base import Command, Observation, clip
from egga.controllers.derivative import FilteredDerivative


class PIDController:
    """Steering PID with heading term. Source: legacy/mission_proving_ground_rl.m lines 173-177."""

    name = "pid"

    def __init__(
        self, cfg: dict[str, Any], dt: float, derivative_cutoff_hz: float | None = None
    ) -> None:
        self._kp = float(cfg["kp"])
        self._ki = float(cfg["ki"])
        self._kd = float(cfg["kd"])
        self._khead = float(cfg["khead"])
        self._integ_limit = float(cfg["integ_limit"])
        self._dt = dt
        self._integ = 0.0
        self._deriv = (
            FilteredDerivative(derivative_cutoff_hz, dt) if derivative_cutoff_hz else None
        )

    def reset(self) -> None:
        self._integ = 0.0
        if self._deriv:
            self._deriv.reset()

    def command(self, obs: Observation) -> Command:
        de_y = self._deriv.update(obs.e_y) if self._deriv else obs.de_y
        self._integ = clip(self._integ + obs.e_y * self._dt, -self._integ_limit, self._integ_limit)
        steer = (
            self._kp * obs.e_y
            + self._ki * self._integ
            + self._kd * de_y
            + self._khead * obs.heading_error
        )
        return Command(steer=steer, gains=(self._kp, self._ki, self._kd, self._khead))


class PDFeedforwardController:
    """Fixed PD with heading term and curvature feedforward; the repo calls it "MPC".

    Source: legacy/mission_proving_ground_rl.m lines 180-184. The low-friction scale uses the
    friction belief the controller is given, which equals the true mu when no belief error is set.
    """

    name = "pd_ff"

    def __init__(
        self,
        cfg: dict[str, Any],
        wheelbase: float,
        vx: float,
        dt: float | None = None,
        derivative_cutoff_hz: float | None = None,
    ) -> None:
        self._kp = float(cfg["kp"])
        self._kd = float(cfg["kd"])
        self._khead = float(cfg["khead"])
        self._low_mu_threshold = float(cfg["low_mu_threshold"])
        self._low_mu_scale = float(cfg["low_mu_scale"])
        self._wheelbase = wheelbase
        self._vx = vx
        if derivative_cutoff_hz and dt is None:
            raise ValueError("dt is required when a derivative cutoff is set")
        self._deriv = (
            FilteredDerivative(derivative_cutoff_hz, dt)
            if derivative_cutoff_hz and dt is not None
            else None
        )

    def reset(self) -> None:
        if self._deriv:
            self._deriv.reset()

    def command(self, obs: Observation) -> Command:
        de_y = self._deriv.update(obs.e_y) if self._deriv else obs.de_y
        vx = obs.vx if obs.vx > 0.0 else self._vx
        steer = (
            self._kp * obs.e_y
            + self._kd * de_y
            + self._khead * obs.heading_error
            + (self._wheelbase / vx) * obs.yaw_rate_ref
        )
        if obs.mu_belief < self._low_mu_threshold:
            steer *= self._low_mu_scale
        return Command(steer=steer, gains=(self._kp, 0.0, self._kd, self._khead))
