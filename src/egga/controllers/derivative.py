from __future__ import annotations

import math


class FilteredDerivative:
    """Backward-difference derivative followed by a first-order low-pass.

    y_k = a * y_{k-1} + (1 - a) * (x_k - x_{k-1}) / dt with a = exp(-2 pi fc dt). The first
    sample returns 0. DC gain is 1, so a ramp input gives its true slope after the transient.
    """

    def __init__(self, cutoff_hz: float, dt: float) -> None:
        if cutoff_hz <= 0.0 or dt <= 0.0:
            raise ValueError("cutoff and dt must be positive")
        self._a = math.exp(-2.0 * math.pi * cutoff_hz * dt)
        self._dt = dt
        self._prev: float | None = None
        self._y = 0.0

    def reset(self) -> None:
        self._prev = None
        self._y = 0.0

    def update(self, x: float) -> float:
        if self._prev is None:
            self._prev = x
            return 0.0
        raw = (x - self._prev) / self._dt
        self._prev = x
        self._y = self._a * self._y + (1.0 - self._a) * raw
        return self._y
