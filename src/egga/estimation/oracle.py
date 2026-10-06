from __future__ import annotations

from egga.estimation.types import STATUS_OK, Estimate


class OracleEstimator:
    """ABLATION ONLY: returns the true values as (near-)point intervals.

    This is the only estimator that may be handed plant truth. Every result produced with it must
    be labelled oracle in its output.
    """

    label = "oracle"

    def __init__(self, tolerance: float = 1e-6) -> None:
        self._tol = tolerance

    def estimate(self, t: float, mu: float, tau: float, mass_scale: float) -> Estimate:
        return Estimate(
            mu_lo=mu - self._tol,
            mu_hi=mu + self._tol,
            tau_bar=tau,
            mass_lo=mass_scale - self._tol,
            mass_hi=mass_scale + self._tol,
            quality=1.0,
            status=STATUS_OK,
            timestamp=t,
        )
