from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from egga.config import load_config
from egga.controllers.base import Command, Observation, clip
from egga.controllers.derivative import FilteredDerivative


@dataclass(frozen=True)
class RLWeights:
    name: str
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray


def load_rl_weights(set_name: str) -> RLWeights:
    raw = load_config("rl_weights.yaml")
    if set_name not in raw:
        raise KeyError(f"unknown RL weight set: {set_name}")
    block = raw[set_name]
    w1 = np.asarray(block["W1"], dtype=float)
    w2 = np.asarray(block["W2"], dtype=float)
    b1 = np.asarray(block["b1"], dtype=float)
    b2 = np.asarray(block["b2"], dtype=float)
    if w1.shape != (8, 6) or b1.shape != (8,) or w2.shape != (4, 8) or b2.shape != (4,):
        raise ValueError(f"RL weight set {set_name} has the wrong shapes")
    return RLWeights(name=set_name, W1=w1, b1=b1, W2=w2, b2=b2)


class RLScheduler:
    """Actor network scheduling [Kp, Ki, Kd, Khead] of a steering PID.

    Source: legacy/rl_adaptive_controller.m. The network proposes gains; the hand rules and
    clipping run after it, as in the repo. Friction belief is used for the observation and rules.
    """

    def __init__(
        self,
        weights: RLWeights,
        cfg: dict[str, Any],
        dt: float,
        wheelbase: float,
        vx: float,
        derivative_cutoff_hz: float | None = None,
    ) -> None:
        self.name = f"rl_{weights.name}"
        self._deriv = (
            FilteredDerivative(derivative_cutoff_hz, dt) if derivative_cutoff_hz else None
        )
        self._weights = weights
        self._dt = dt
        self._wheelbase = wheelbase
        self._vx = vx
        self._integ_limit = float(cfg["integ_limit"])
        bounds = cfg["gain_bounds"]
        self._bounds = {k: (float(v[0]), float(v[1])) for k, v in bounds.items()}
        fr = cfg["friction_rule"]
        self._mu_threshold = float(fr["mu_threshold"])
        self._kd_gain = float(fr["kd_gain"])
        self._ki_mu_ref = float(fr["ki_mu_ref"])
        sr = cfg["slope_rule"]
        self._slope_threshold = float(sr["slope_threshold_deg"])
        self._kp_per_deg = float(sr["kp_per_deg"])
        self._khead_per_deg = float(sr["khead_per_deg"])
        ob = cfg["observation"]
        self._slope_scale = float(ob["slope_scale_deg"])
        self._mu_offset = float(ob["mu_offset"])
        self._mu_scale = float(ob["mu_scale"])
        self._integ = 0.0

    def reset(self) -> None:
        self._integ = 0.0
        if self._deriv:
            self._deriv.reset()

    def command(self, obs: Observation) -> Command:
        de_y = self._deriv.update(obs.e_y) if self._deriv else obs.de_y
        vx = obs.vx if obs.vx > 0.0 else self._vx
        x = np.array(
            [
                obs.e_y,
                de_y,
                obs.heading_error,
                obs.yaw_rate - obs.yaw_rate_ref,
                obs.slope_deg / self._slope_scale,
                (obs.mu_belief - self._mu_offset) / self._mu_scale,
            ]
        )
        hidden = np.tanh(self._weights.W1 @ x + self._weights.b1)
        raw = self._weights.W2 @ hidden + self._weights.b2

        kp = clip(float(raw[0]), *self._bounds["kp"])
        ki = clip(float(raw[1]), *self._bounds["ki"])
        kd = clip(float(raw[2]), *self._bounds["kd"])
        khead = clip(float(raw[3]), *self._bounds["khead"])

        mu = obs.mu_belief
        if mu < self._mu_threshold:
            kd *= 1.0 + self._kd_gain * (self._mu_threshold - mu)
            ki *= mu / self._ki_mu_ref
        if obs.slope_deg > self._slope_threshold:
            kp *= 1.0 + self._kp_per_deg * obs.slope_deg
            khead *= 1.0 + self._khead_per_deg * obs.slope_deg

        self._integ = clip(self._integ + obs.e_y * self._dt, -self._integ_limit, self._integ_limit)
        steer = (
            kp * obs.e_y
            + ki * self._integ
            + kd * de_y
            + khead * obs.heading_error
            + (self._wheelbase / vx) * obs.yaw_rate_ref
        )
        return Command(steer=float(steer), gains=(kp, ki, kd, khead))
