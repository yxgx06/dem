from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import expm, solve_discrete_are

from egga.controllers.base import Command, Observation, clip
from egga.controllers.derivative import FilteredDerivative


@dataclass(frozen=True)
class DesignVehicle:
    """Vehicle parameters the LQR design is allowed to know (not the plant's internals)."""

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

    @property
    def understeer_coeff(self) -> float:
        """K_us in delta = L * kappa + K_us * a_y (rad per m/s^2)."""
        return self.mass / self.wheelbase * (self.lr / self.cf - self.lf / self.cr)

    @classmethod
    def from_config(
        cls, vehicle: dict[str, Any], mass_scale: float, inertia_follows_mass: bool
    ) -> DesignVehicle:
        return cls(
            mass=float(vehicle["mass_kg"]) * mass_scale,
            yaw_inertia=float(vehicle["yaw_inertia_kgm2"])
            * (mass_scale if inertia_follows_mass else 1.0),
            lf=float(vehicle["lf_m"]),
            lr=float(vehicle["lr_m"]),
            cf=float(vehicle["cf_n_per_rad"]),
            cr=float(vehicle["cr_n_per_rad"]),
            gravity=float(vehicle["gravity_mps2"]),
        )


def error_model(v: DesignVehicle, speed: float) -> tuple[np.ndarray, np.ndarray]:
    """Continuous lateral error model, state [e, e_dot, psi_e, psi_e_dot]."""
    m, iz, lf, lr, cf, cr = v.mass, v.yaw_inertia, v.lf, v.lr, v.cf, v.cr
    a = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -(cf + cr) / (m * speed), (cf + cr) / m, (-cf * lf + cr * lr) / (m * speed)],
            [0.0, 0.0, 0.0, 1.0],
            [
                0.0,
                -(cf * lf - cr * lr) / (iz * speed),
                (cf * lf - cr * lr) / iz,
                -(cf * lf**2 + cr * lr**2) / (iz * speed),
            ],
        ]
    )
    b = np.array([[0.0], [cf / m], [0.0], [cf * lf / iz]])
    return a, b


def discretize(a: np.ndarray, b: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    n = a.shape[0]
    m = np.zeros((n + 1, n + 1))
    m[:n, :n] = a
    m[:n, n:] = b
    e = expm(m * dt)
    return e[:n, :n], e[:n, n:]


def lqr_gain(
    v: DesignVehicle, speed: float, q_diag: list[float], r: float, dt: float, delay_steps: int = 0
) -> np.ndarray:
    """Discrete LQR gain for the error model; delay_steps > 0 augments with the input buffer.

    With delay the returned gain acts on [x (4); u_{k-1} .. u_{k-N}] (length 4 + N).
    """
    ad, bd = discretize(*error_model(v, speed), dt)
    if delay_steps == 0:
        p = solve_discrete_are(ad, bd, np.diag(q_diag), np.array([[r]]))
        return np.linalg.solve(r + bd.T @ p @ bd, bd.T @ p @ ad)[0]
    n = 4 + delay_steps
    az = np.zeros((n, n))
    bz = np.zeros((n, 1))
    az[:4, :4] = ad
    az[:4, n - 1 : n] = bd  # oldest buffered input drives the plant
    bz[4, 0] = 1.0
    for i in range(1, delay_steps):
        az[4 + i, 4 + i - 1] = 1.0
    qz = np.zeros((n, n))
    qz[:4, :4] = np.diag(q_diag)
    p = solve_discrete_are(az, bz, qz, np.array([[r]]))
    return np.linalg.solve(r + bz.T @ p @ bz, bz.T @ p @ az)[0]


class LQRController:
    """Speed-scheduled LQR on the lateral error model with steering feedforward and an a_y limit.

    Gains are precomputed on a speed grid and interpolated. Sign convention matches the repo: the
    state is [-e, -e_dot, -heading_error, yaw_rate - yaw_rate_ref] with e = reference minus
    vehicle in the body frame. Source of the structure: legacy/lib3.py (audit port).
    """

    def __init__(
        self,
        name: str,
        design: DesignVehicle,
        cfg: dict[str, Any],
        dt: float,
        derivative_cutoff_hz: float | None = None,
        delay_design_s: float = 0.0,
        nominal_speed: float = 10.0,
    ) -> None:
        self.name = name
        self._design = design
        self._dt = dt
        self._nominal_speed = nominal_speed
        self._ay_fraction = float(cfg["ay_limit_fraction"])
        self._delay_steps = int(round(delay_design_s / dt))
        self._deriv = (
            FilteredDerivative(derivative_cutoff_hz, dt) if derivative_cutoff_hz else None
        )
        lo, hi = float(cfg["design_speed_min_mps"]), float(cfg["design_speed_max_mps"])
        step = float(cfg["design_speed_step_mps"])
        self._grid = np.arange(lo, hi + step / 2.0, step)
        speeds = [float(s) for s in cfg["q_speeds_mps"]]
        q_table = np.asarray(cfg["q_diag"], dtype=float)
        r = float(cfg["r"])
        gains = []
        for speed in self._grid:
            q = [float(np.interp(speed, speeds, q_table[:, i])) for i in range(4)]
            gains.append(lqr_gain(design, float(speed), q, r, dt, self._delay_steps))
        self._gains = np.asarray(gains)
        self._buffer = np.zeros(self._delay_steps)
        self._last_steer = 0.0
        self._delta_max = 0.5

    def reset(self) -> None:
        self._buffer[:] = 0.0
        self._last_steer = 0.0
        if self._deriv:
            self._deriv.reset()

    def _gain(self, speed: float) -> np.ndarray:
        s = clip(speed, float(self._grid[0]), float(self._grid[-1]))
        return np.array(
            [np.interp(s, self._grid, self._gains[:, i]) for i in range(self._gains.shape[1])]
        )

    def steer_limit(self, speed: float, mu_belief: float) -> float:
        """Steering angle that produces a_y,max = fraction * mu * g in steady state."""
        ay_max = self._ay_fraction * mu_belief * self._design.gravity
        kappa = ay_max / speed**2
        return self._design.wheelbase * kappa + self._design.understeer_coeff * ay_max

    def command(self, obs: Observation) -> Command:
        speed = obs.vx if obs.vx > 0.0 else self._nominal_speed
        de_y = self._deriv.update(obs.e_y) if self._deriv else obs.de_y
        k = self._gain(speed)
        x = np.array(
            [-obs.e_y, -de_y, -obs.heading_error, obs.yaw_rate - obs.yaw_rate_ref]
        )
        d = self._design
        kappa = obs.yaw_rate_ref / speed
        kv = d.understeer_coeff
        k_psi = k[2]
        ff = (
            d.wheelbase * kappa
            + kv * speed**2 * kappa
            - k_psi * (d.lr * kappa - d.lf * d.mass * speed**2 * kappa / (d.cr * d.wheelbase))
        )
        if self._delay_steps:
            z = np.concatenate([x, self._buffer])
            fb = -float(k @ z)
        else:
            fb = -float(k @ x)
        limit = min(self.steer_limit(speed, obs.mu_belief), self._delta_max)
        steer = clip(fb + ff, -limit, limit)
        if self._delay_steps:
            # The buffer holds the feedback part only: the design is in error coordinates, where
            # the feedforward cancels the reference. Buffering the total would feed it back.
            self._buffer = np.roll(self._buffer, 1)
            self._buffer[0] = steer - ff
        self._last_steer = steer
        return Command(steer=steer, gains=(float(k[0]), 0.0, float(k[1]), float(k[2])))
