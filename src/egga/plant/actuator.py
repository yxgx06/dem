from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


class TransportDelay:
    """Delays a scalar command by a fixed whole number of control steps."""

    def __init__(self, steps: int) -> None:
        if steps < 0:
            raise ValueError("delay steps must be non-negative")
        self._steps = steps
        self._buf: deque[float] = deque([0.0] * steps)

    def step(self, command: float) -> float:
        if self._steps == 0:
            return command
        out = self._buf.popleft()
        self._buf.append(command)
        return out


def steer_actuator(
    command: float,
    previous: float,
    dt: float,
    rate_max: float,
    angle_max: float,
) -> float:
    max_step = rate_max * dt
    stepped = previous + float(np.clip(command - previous, -max_step, max_step))
    return float(np.clip(stepped, -angle_max, angle_max))


class SteeringActuator:
    """Command -> transport delay (fixed + jitter) -> gain/bias -> first-order lag -> rate -> angle.

    The delay is fractional: the output interpolates linearly between stored command samples.
    With lag, jitter, gain loss and bias all off and an integer delay this equals Phase 0's
    TransportDelay followed by steer_actuator.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        dt: float,
        rate_max: float,
        angle_max: float,
        rng: np.random.Generator,
    ) -> None:
        self._dt = dt
        self._delay_s = float(cfg["delay_s"])
        self._jitter_s = float(cfg["jitter_s"])
        self._tau = float(cfg["lag_tau_s"])
        self._gain = float(cfg["gain"])
        self._bias = float(cfg["bias_rad"])
        self._rate_max = rate_max
        self._angle_max = angle_max
        self._rng = rng
        if self._delay_s < 0.0 or self._jitter_s < 0.0 or self._tau < 0.0:
            raise ValueError("delay, jitter and lag must be non-negative")
        depth = int(np.ceil((self._delay_s + self._jitter_s) / dt)) + 2
        self._history: deque[float] = deque([0.0] * depth, maxlen=depth)
        self._lagged = 0.0
        self._delta = 0.0

    @property
    def angle(self) -> float:
        return self._delta

    def step(self, command: float) -> float:
        self._history.appendleft(command)
        delay = self._delay_s
        if self._jitter_s > 0.0:
            delay += float(self._rng.uniform(0.0, self._jitter_s))
        pos = delay / self._dt
        n = int(np.floor(pos))
        frac = pos - n
        delayed = self._history[n] if frac == 0.0 else (
            (1.0 - frac) * self._history[n] + frac * self._history[n + 1]
        )
        target = self._gain * delayed + self._bias
        if self._tau > 0.0:
            a = float(np.exp(-self._dt / self._tau))
            self._lagged = a * self._lagged + (1.0 - a) * target
            target = self._lagged
        self._delta = steer_actuator(target, self._delta, self._dt, self._rate_max, self._angle_max)
        return self._delta
