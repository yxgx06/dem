from __future__ import annotations

import math
from dataclasses import dataclass

STATUS_OK = "ok"
STATUS_LOW_EXCITATION = "low_excitation"
STATUS_STALE = "stale"
STATUS_INVALID = "invalid"


@dataclass(frozen=True)
class Measurements:
    """Everything an estimator may see. No plant internals (true mu, delay, mass) appear here."""

    t: float
    vx: float
    yaw_rate: float
    yaw_rate_valid: bool
    lateral_accel: float
    lateral_accel_valid: bool
    steer_meas: float
    steer_meas_valid: bool
    steer_cmd: float


@dataclass(frozen=True)
class Estimate:
    """Interval estimates with a quality score. Widest bounds mean 'know nothing'."""

    mu_lo: float
    mu_hi: float
    tau_bar: float
    mass_lo: float
    mass_hi: float
    quality: float
    status: str
    timestamp: float

    @property
    def mu_hat(self) -> float:
        return 0.5 * (self.mu_lo + self.mu_hi)

    @property
    def mu_sigma(self) -> float:
        """Standard deviation of a uniform distribution over [mu_lo, mu_hi]."""
        return (self.mu_hi - self.mu_lo) / math.sqrt(12.0)

    def mu_safe(self, k: float, mu_min: float) -> float:
        """max(mu_min, mu_hat - k * sigma). With k = sqrt(3) this is the interval's lower bound."""
        return max(mu_min, self.mu_hat - k * self.mu_sigma)
