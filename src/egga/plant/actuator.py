from __future__ import annotations

from collections import deque

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
