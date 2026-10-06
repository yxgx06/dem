from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Measurement:
    value: float
    valid: bool


class Sensor:
    """Sample delay, bias, white noise, quantisation, then dropout.

    A dropout returns the last good value with valid=False.
    """

    def __init__(self, cfg: dict[str, Any], rng: np.random.Generator) -> None:
        self._noise = float(cfg["noise_std"])
        self._bias = float(cfg["bias"])
        self._quant = float(cfg["quant_step"])
        self._dropout = float(cfg["dropout_prob"])
        self._delay = int(cfg["delay_steps"])
        if self._noise < 0.0 or self._quant < 0.0 or not 0.0 <= self._dropout <= 1.0:
            raise ValueError("invalid sensor parameters")
        if self._delay < 0:
            raise ValueError("sensor delay must be non-negative")
        self._rng = rng
        self._buf: deque[float] = deque(maxlen=self._delay + 1)
        self._last = 0.0

    def measure(self, true_value: float) -> Measurement:
        self._buf.append(true_value)
        delayed = self._buf[0]
        value = delayed + self._bias
        if self._noise > 0.0:
            value += float(self._rng.normal(0.0, self._noise))
        if self._quant > 0.0:
            value = float(np.round(value / self._quant) * self._quant)
        if self._dropout > 0.0 and self._rng.random() < self._dropout:
            return Measurement(self._last, False)
        self._last = value
        return Measurement(value, True)
