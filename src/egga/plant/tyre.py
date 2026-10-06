from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TyreParams:
    model: str
    shape_c: float
    curvature_e: float
    load_sensitivity: float
    min_force_cap: float


def pacejka_force(
    alpha: float, fz: float, mu: float, k_c: float, fz_nom: float, p: TyreParams
) -> float:
    """Lateral force of one wheel (Pacejka magic formula).

    Cornering stiffness is k_c * fz (so it scales with load and is independent of mu). The peak is
    mu * (1 - load_sensitivity * (fz / fz_nom - 1)) * fz, floored at half the minimum force cap.
    """
    if fz <= 0.0:
        return 0.0
    relative_load = fz / fz_nom - 1.0
    coeff = max(mu * (1.0 - p.load_sensitivity * relative_load), 1e-3)
    peak = max(coeff * fz, 0.5 * p.min_force_cap)
    b = k_c * fz / (p.shape_c * peak)
    x = b * alpha
    return float(peak * np.sin(p.shape_c * np.arctan(x - p.curvature_e * (x - np.arctan(x)))))


def linear_clip_force(alpha: float, c_alpha: float, fz_axle: float, mu: float, cap: float) -> float:
    """Phase 0 tyre: linear in slip, clipped at max(mu * Fz, cap)."""
    limit = max(mu * fz_axle, cap)
    return float(np.clip(c_alpha * alpha, -limit, limit))
