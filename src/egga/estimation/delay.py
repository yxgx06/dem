from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
MAX_HELD_MISSES = 5


def _best_lag(u: FloatArray, d: FloatArray, max_lag: int) -> tuple[float, float]:
    """Lag (in samples, parabolic sub-sample) and peak correlation of d against delayed u."""
    du = np.diff(u)
    dd = np.diff(d)
    n = du.size
    corr = np.full(max_lag + 1, -1.0)
    for lag in range(max_lag + 1):
        a = du[: n - lag]
        b = dd[lag:]
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom > 1e-12:
            corr[lag] = float(a @ b) / denom
    k = int(np.argmax(corr))
    peak = float(corr[k])
    best = float(k)
    if 0 < k < max_lag:
        c0, c1, c2 = corr[k - 1], corr[k], corr[k + 1]
        curvature = c0 - 2.0 * c1 + c2
        if abs(curvature) > 1e-12:
            best += float(np.clip(0.5 * (c0 - c2) / curvature, -0.5, 0.5))
    return best, peak


class DelayBoundEstimator:
    """Upper bound on the steering latency from command-vs-measured-steering cross-correlation.

    The bound is tau_hat plus a margin, never a point value. When the command does not excite the
    steering the bound relaxes linearly to the prior maximum instead of holding the last value.
    """

    def __init__(self, cfg: dict[str, Any], tau_max: float, dt: float) -> None:
        self._dt = dt
        self._tau_max = tau_max
        self._window = int(round(float(cfg["window_s"]) / dt))
        self._every = max(1, int(round(float(cfg["update_every_s"]) / dt)))
        self._max_lag = int(round(float(cfg["max_lag_s"]) / dt))
        self._min_exc = float(cfg["min_excitation_rad"])
        self._min_corr = float(cfg["min_corr"])
        self._margin_steps = float(cfg["margin_steps"])
        self._margin_frac = float(cfg["margin_frac"])
        self._widen = float(cfg["widen_time_s"])
        self.reset()

    def reset(self) -> None:
        self._u: deque[float] = deque(maxlen=self._window)
        self._d: deque[float] = deque(maxlen=self._window)
        self._count = 0
        self._misses = 0
        self._good_bound = self._tau_max
        self._good_t: float | None = None
        self._tau_hat = float("nan")
        self._peak = 0.0
        self._excited = False

    def update(self, t: float, command: float, steer_meas: float | None) -> None:
        if steer_meas is None:
            self._misses += 1
            if self._misses > MAX_HELD_MISSES or not self._d:
                self._u.clear()  # a long gap breaks the alignment of the two signals
                self._d.clear()
                return
            steer_meas = self._d[-1]  # short gap: hold the last measurement
        else:
            self._misses = 0
        self._u.append(command)
        self._d.append(steer_meas)
        self._count += 1
        if len(self._u) < self._window or self._count % self._every:
            return
        u = np.asarray(self._u, dtype=np.float64)
        d = np.asarray(self._d, dtype=np.float64)
        self._excited = float(np.std(u)) >= self._min_exc
        if not self._excited:
            return
        lag, peak = _best_lag(u, d, self._max_lag)
        self._peak = peak
        if peak < self._min_corr:
            return
        self._tau_hat = lag * self._dt
        margin = self._margin_steps * self._dt + self._margin_frac * self._tau_hat
        self._good_bound = float(np.clip(self._tau_hat + margin, 0.0, self._tau_max))
        self._good_t = t

    def bound(self, t: float) -> tuple[float, bool]:
        """(tau_bar, informed). Not informed -> the prior maximum."""
        if self._good_t is None:
            return self._tau_max, False
        age = max(t - self._good_t, 0.0)
        frac = min(age / self._widen, 1.0)
        return self._good_bound + (self._tau_max - self._good_bound) * frac, frac < 1.0

    @property
    def tau_hat(self) -> float:
        return self._tau_hat
