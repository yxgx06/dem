from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
MU_STATE_FLOOR = 0.05
MU_STATE_CEIL = 1.5
_EPS = 1e-6
_N = 4  # state: [lateral velocity, yaw rate, mu, theta = nominal mass / mass]


@dataclass(frozen=True)
class NominalVehicle:
    """Datasheet vehicle the estimators may assume (not the plant's true, scaled values)."""

    mass: float
    yaw_inertia: float
    lf: float
    lr: float
    cf: float
    cr: float
    gravity: float

    @property
    def wheelbase(self) -> float:
        return self.lf + self.lr

    @classmethod
    def from_config(cls, vehicle: dict[str, Any]) -> NominalVehicle:
        return cls(
            mass=float(vehicle["mass_kg"]),
            yaw_inertia=float(vehicle["yaw_inertia_kgm2"]),
            lf=float(vehicle["lf_m"]),
            lr=float(vehicle["lr_m"]),
            cf=float(vehicle["cf_n_per_rad"]),
            cr=float(vehicle["cr_n_per_rad"]),
            gravity=float(vehicle["gravity_mps2"]),
        )


def _axle_force(alpha: float, fz: float, mu: float, c_alpha: float) -> float:
    """Saturating tanh tyre: linear at small slip, peak mu * fz (not the plant's Pacejka)."""
    cap = max(mu, MU_STATE_FLOOR) * fz
    return cap * math.tanh(c_alpha * alpha / cap)


class FrictionEKF:
    """EKF on [lateral velocity, yaw rate, mu, theta] with measurements [yaw rate, lateral accel].

    theta = nominal mass / actual mass (inertia assumed to scale with mass). Friction and mass
    are estimated jointly because assuming the nominal mass makes a heavy vehicle look slippery.
    mu and theta are random walks; their variances shrink only when the measurements depend on
    them and grow back toward the prior otherwise.
    """

    def __init__(
        self, vehicle: NominalVehicle, cfg: dict[str, Any], prior: dict[str, Any], dt: float
    ) -> None:
        self._v = vehicle
        self._dt = dt
        self._mu_min = float(prior["mu_min"])
        self._mu_max = float(prior["mu_max"])
        self._mass_min = float(prior["mass_scale_min"])
        self._mass_max = float(prior["mass_scale_max"])
        self._theta_min = 1.0 / float(prior["mass_scale_max"])
        self._theta_max = 1.0 / float(prior["mass_scale_min"])
        self._mu0 = float(cfg["initial_mu"])
        self._theta0 = 1.0 / float(cfg["initial_mass_scale"])
        self._p0_mu = ((self._mu_max - self._mu_min) ** 2) / 12.0
        self._p0_theta = ((self._theta_max - self._theta_min) ** 2) / 12.0
        self._vy0 = float(cfg["vy0_var"])
        self._r0 = float(cfg["r0_var"])
        self._q = np.diag(
            [
                float(cfg["q_vy"]) ** 2 * dt,
                float(cfg["q_r"]) ** 2 * dt,
                float(cfg["q_mu_per_sqrt_s"]) ** 2 * dt,
                float(cfg["q_theta_per_sqrt_s"]) ** 2 * dt,
            ]
        )
        self._r_full = np.array([float(cfg["r_yaw_rate"]) ** 2, float(cfg["r_ay"]) ** 2])
        self._z = float(cfg["z"])
        self._z_mass = float(cfg["z_mass"])
        self._cal = float(cfg["mu_scale_cal"])
        self._steer_std = float(cfg["steer_noise_std"])
        self._x: FloatArray = np.zeros(_N)
        self._p: FloatArray = np.eye(_N)
        self._started = False
        self.reset_to_prior()

    def reset_to_prior(self) -> None:
        self._x = np.array([0.0, 0.0, self._mu0, self._theta0])
        self._p = np.diag([self._vy0, self._r0, self._p0_mu, self._p0_theta])
        self._started = False

    def _dynamics(
        self, vy: float, r: float, mu: float, theta: float, delta: float, vx: float
    ) -> tuple[float, float, float]:
        v = self._v
        mass = v.mass / theta
        inertia = v.yaw_inertia / theta
        fz_f = mass * v.gravity * v.lr / v.wheelbase
        fz_r = mass * v.gravity * v.lf / v.wheelbase
        alpha_f = delta - math.atan2(vy + v.lf * r, vx)
        alpha_r = -math.atan2(vy - v.lr * r, vx)
        fyf = _axle_force(alpha_f, fz_f, mu, v.cf)
        fyr = _axle_force(alpha_r, fz_r, mu, v.cr)
        ay = (fyf * math.cos(delta) + fyr) / mass
        d_r = (v.lf * fyf * math.cos(delta) - v.lr * fyr) / inertia
        return ay - vx * r, d_r, ay

    def _step_model(self, x: FloatArray, delta: float, vx: float) -> FloatArray:
        d_vy, d_r, _ = self._dynamics(
            float(x[0]), float(x[1]), float(x[2]), float(x[3]), delta, vx
        )
        return np.array([x[0] + self._dt * d_vy, x[1] + self._dt * d_r, x[2], x[3]])

    def _ay(self, x: FloatArray, delta: float, vx: float) -> float:
        return self._dynamics(float(x[0]), float(x[1]), float(x[2]), float(x[3]), delta, vx)[2]

    def step(
        self, delta: float, vx: float, yaw_rate: float | None, lateral_accel: float | None
    ) -> None:
        """One predict step and an update with whichever measurements are valid (or none)."""
        if not self._started and yaw_rate is not None:
            self._x[1] = yaw_rate
            self._started = True
        x_prior = self._step_model(self._x, delta, vx)
        jac = np.zeros((_N, _N))
        for j in range(_N):
            bump = np.zeros(_N)
            bump[j] = _EPS
            jac[:, j] = (self._step_model(self._x + bump, delta, vx) - x_prior) / _EPS
        d_step = (self._step_model(self._x, delta + _EPS, vx) - x_prior) / _EPS
        p = jac @ self._p @ jac.T + self._q + self._steer_std**2 * np.outer(d_step, d_step)

        rows: list[int] = []
        residual: list[float] = []
        if yaw_rate is not None:
            rows.append(0)
            residual.append(yaw_rate - float(x_prior[1]))
        if lateral_accel is not None:
            rows.append(1)
            residual.append(lateral_accel - self._ay(x_prior, delta, vx))
        if not rows:
            self._x, self._p = x_prior, p
            self._clip_state()
            return

        h_full = np.zeros((2, _N))
        h_full[0, 1] = 1.0
        ay0 = self._ay(x_prior, delta, vx)
        for j in range(_N):
            bump = np.zeros(_N)
            bump[j] = _EPS
            h_full[1, j] = (self._ay(x_prior + bump, delta, vx) - ay0) / _EPS
        h = h_full[rows]
        d_ay = (self._ay(x_prior, delta + _EPS, vx) - ay0) / _EPS
        h_delta = np.array([0.0, d_ay])[rows]
        r_cov = np.diag(self._r_full[rows]) + self._steer_std**2 * np.outer(h_delta, h_delta)
        s = h @ p @ h.T + r_cov
        gain = p @ h.T @ np.linalg.inv(s)
        self._x = x_prior + gain @ np.asarray(residual)
        ident = np.eye(_N)
        self._p = (ident - gain @ h) @ p @ (ident - gain @ h).T + gain @ r_cov @ gain.T
        self._p = 0.5 * (self._p + self._p.T)
        self._clip_state()

    def _clip_state(self) -> None:
        self._x[2] = float(np.clip(self._x[2], MU_STATE_FLOOR, MU_STATE_CEIL))
        self._x[3] = float(np.clip(self._x[3], self._theta_min, self._theta_max))
        self._p[2, 2] = float(min(self._p[2, 2], self._p0_mu))
        self._p[3, 3] = float(min(self._p[3, 3], self._p0_theta))

    @property
    def mu_prior_sigma(self) -> float:
        return math.sqrt(self._p0_mu)

    @property
    def theta_prior_sigma(self) -> float:
        return math.sqrt(self._p0_theta)

    def interval(self) -> tuple[float, float, float]:
        """(mu_lo, mu_hi, sigma) on the nominal-mu scale, clipped to the prior range."""
        mu = self._cal * float(self._x[2])
        sigma = self._cal * math.sqrt(max(float(self._p[2, 2]), 0.0))
        lo = float(np.clip(mu - self._z * sigma, self._mu_min, self._mu_max))
        hi = float(np.clip(mu + self._z * sigma, self._mu_min, self._mu_max))
        return lo, max(hi, lo), sigma

    def mass_interval(self) -> tuple[float, float, float]:
        """(mass_scale_lo, mass_scale_hi, sigma_theta) from the theta interval."""
        theta = float(self._x[3])
        sigma = math.sqrt(max(float(self._p[3, 3]), 0.0))
        th_lo = float(np.clip(theta - self._z_mass * sigma, self._theta_min, self._theta_max))
        th_hi = float(np.clip(theta + self._z_mass * sigma, self._theta_min, self._theta_max))
        lo = max(1.0 / th_hi, self._mass_min)
        hi = min(1.0 / th_lo, self._mass_max)
        return lo, max(hi, lo), sigma
