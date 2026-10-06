from __future__ import annotations

import math
from typing import Any

from egga.estimation.delay import DelayBoundEstimator
from egga.estimation.friction import FrictionEKF, NominalVehicle
from egga.estimation.types import (
    STATUS_INVALID,
    STATUS_LOW_EXCITATION,
    STATUS_OK,
    STATUS_STALE,
    Estimate,
    Measurements,
)


def _finite(*values: float) -> bool:
    return all(math.isfinite(v) for v in values)


class EstimatorSuite:
    """Friction interval, steering-delay bound and mass bounds from vehicle signals only.

    Invalid input (NaN/inf, time going backwards) or a measurement gap longer than
    staleness.max_gap_s returns the widest bounds immediately and re-initialises the filters;
    the last good value is never returned. Shorter gaps propagate the model with growing
    uncertainty.
    """

    def __init__(
        self, vehicle_cfg: dict[str, Any], cfg: dict[str, Any], dt: float
    ) -> None:
        prior = cfg["prior"]
        self._dt = dt
        self._prior = prior
        self.mu_min = float(prior["mu_min"])
        self._mu_min = self.mu_min
        self._mu_max = float(prior["mu_max"])
        self._tau_max = float(prior["tau_max_s"])
        self._mass_lo = float(prior["mass_scale_min"])
        self._mass_hi = float(prior["mass_scale_max"])
        self._max_gap = float(cfg["staleness"]["max_gap_s"])
        nominal = NominalVehicle.from_config(vehicle_cfg)
        self._friction = FrictionEKF(nominal, cfg["friction"], prior, dt)
        self._delay = DelayBoundEstimator(cfg["delay"], self._tau_max, dt)
        self._mu_prior_sigma = self._friction.mu_prior_sigma
        self.k_safe = float(cfg["friction"]["k_safe"])
        self.reset()

    def reset(self) -> None:
        self._friction.reset_to_prior()
        self._delay.reset()
        self._last_t: float | None = None
        self._last_valid_t: float | None = None
        self._last_steer: float | None = None

    def widest(self, t: float, status: str) -> Estimate:
        return Estimate(
            mu_lo=self._mu_min,
            mu_hi=self._mu_max,
            tau_bar=self._tau_max,
            mass_lo=self._mass_lo,
            mass_hi=self._mass_hi,
            quality=0.0,
            status=status,
            timestamp=t if math.isfinite(t) else float("nan"),
        )

    def update(self, m: Measurements) -> Estimate:
        if not _finite(m.t, m.vx, m.steer_cmd) or m.vx <= 0.0:
            self._invalidate()
            return self.widest(m.t, STATUS_INVALID)
        if self._last_t is not None and m.t <= self._last_t:
            self._invalidate()
            return self.widest(m.t, STATUS_INVALID)
        self._last_t = m.t

        yaw = m.yaw_rate if m.yaw_rate_valid and _finite(m.yaw_rate) else None
        acc = m.lateral_accel if m.lateral_accel_valid and _finite(m.lateral_accel) else None
        steer = m.steer_meas if m.steer_meas_valid and _finite(m.steer_meas) else None
        bad_value = (m.yaw_rate_valid and yaw is None) or (m.lateral_accel_valid and acc is None)
        bad_value = bad_value or (m.steer_meas_valid and steer is None)
        if bad_value:
            self._invalidate()
            return self.widest(m.t, STATUS_INVALID)

        if yaw is not None and acc is not None and steer is not None:
            self._last_valid_t = m.t
        if self._last_valid_t is None:
            self._last_valid_t = m.t
        if m.t - self._last_valid_t > self._max_gap:
            self._friction.reset_to_prior()
            self._delay.reset()
            return self.widest(m.t, STATUS_STALE)

        if steer is not None:
            self._last_steer = steer
        delta = steer if steer is not None else (
            self._last_steer if self._last_steer is not None else m.steer_cmd
        )
        self._friction.step(delta, m.vx, yaw, acc)
        self._delay.update(m.t, m.steer_cmd, steer)

        mu_lo, mu_hi, sigma = self._friction.interval()
        tau_bar, tau_informed = self._delay.bound(m.t)
        mass_lo, mass_hi, _ = self._friction.mass_interval()
        q_mu = 1.0 - min(1.0, sigma / self._mu_prior_sigma)
        q_tau = 1.0 - tau_bar / self._tau_max if tau_informed else 0.0
        quality = float(min(q_mu, q_tau))
        status = STATUS_OK if (q_mu > 0.2 and tau_informed) else STATUS_LOW_EXCITATION
        return Estimate(
            mu_lo=mu_lo,
            mu_hi=mu_hi,
            tau_bar=tau_bar,
            mass_lo=mass_lo,
            mass_hi=mass_hi,
            quality=max(0.0, quality),
            status=status,
            timestamp=m.t,
        )

    def _invalidate(self) -> None:
        self._friction.reset_to_prior()
        self._delay.reset()
        self._last_valid_t = None
        self._last_steer = None
