from __future__ import annotations

import time
from typing import Any

import numpy as np
import osqp
import scipy.sparse as sp
from scipy.linalg import expm

from egga.controllers.base import Command, Observation, clip
from egga.controllers.derivative import FilteredDerivative
from egga.controllers.lqr import DesignVehicle, error_model


def _disturbance_vector(v: DesignVehicle, speed: float) -> np.ndarray:
    """Effect of the desired yaw rate on the error states (Rajamani lateral error model)."""
    m, iz, lf, lr, cf, cr = v.mass, v.yaw_inertia, v.lf, v.lr, v.cf, v.cr
    return np.array(
        [
            [0.0],
            [-(cf * lf - cr * lr) / (m * speed) - speed],
            [0.0],
            [-(cf * lf**2 + cr * lr**2) / (iz * speed)],
        ]
    )


class _QP:
    """Condensed prediction matrices and the OSQP problem for one design speed."""

    def __init__(self, design: DesignVehicle, speed: float, cfg: dict[str, Any], rate_max: float):
        self.speed = speed
        self.np_ = int(cfg["np"])
        self.nc = int(cfg["nc"])
        ts = float(cfg["ts_s"])
        a, b = error_model(design, speed)
        dvec = _disturbance_vector(design, speed)
        aug = np.zeros((6, 6))
        aug[:4, :4] = a
        aug[:4, 4:5] = b
        aug[:4, 5:6] = dvec
        e = expm(aug * ts)
        ad, bd, dd = e[:4, :4], e[:4, 4:5], e[:4, 5:6]
        c = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        n, nc = self.np_, self.nc
        sx = np.zeros((2 * n, 4))
        su = np.zeros((2 * n, nc))
        sw = np.zeros((2 * n, n))
        powers = [np.eye(4)]
        for _ in range(n):
            powers.append(ad @ powers[-1])
        for i in range(1, n + 1):
            sx[2 * (i - 1) : 2 * i] = c @ powers[i]
            for j in range(i):  # input / disturbance applied at step j affects x_i
                effect = powers[i - 1 - j]
                col = min(j, nc - 1)
                su[2 * (i - 1) : 2 * i, col] += (c @ effect @ bd).ravel()
                sw[2 * (i - 1) : 2 * i, j] = (c @ effect @ dd).ravel()
        qb = np.kron(np.eye(n), np.diag([float(cfg["q_lat"]), float(cfg["q_head"])]))
        r_u, r_du = float(cfg["r_u"]), float(cfg["r_du"])
        dm = np.eye(nc) - np.eye(nc, k=-1)
        self.sx, self.su, self.sw, self.qb, self.dm = sx, su, sw, qb, dm
        self.r_u, self.r_du = r_u, r_du
        hess = 2.0 * (su.T @ qb @ su + r_u * np.eye(nc) + r_du * dm.T @ dm)
        hess = 0.5 * (hess + hess.T) + 1e-6 * np.eye(nc)
        cons = sp.csc_matrix(np.vstack([np.eye(nc), dm]))
        self.step = rate_max * ts
        lo = np.concatenate([-np.ones(nc), -self.step * np.ones(nc)])
        hi = -lo
        self.prob = osqp.OSQP()
        osqp_cfg = cfg["osqp"]
        self.prob.setup(
            sp.csc_matrix(hess),
            np.zeros(nc),
            cons,
            lo,
            hi,
            verbose=False,
            eps_abs=float(osqp_cfg["eps_abs"]),
            eps_rel=float(osqp_cfg["eps_rel"]),
            max_iter=int(osqp_cfg["max_iter"]),
            polishing=False,
        )


class MPCController:
    """Linear MPC on the lateral error model: condensed QP, Np=15, Nc=5, solved with OSQP.

    Tracks zero lateral and heading error given a preview of the reference yaw rate (planner
    output), penalises steering relative to the steady-state feedforward and steering rate, with
    box limits |delta| <= min(delta_max, delta(a_y,max = fraction * mu_belief * g)) and rate limits.
    The QP is solved every ts_s and held in between. The model is the nominal bicycle with the
    design mass; delay, lag and tyre nonlinearity are unmodelled.
    """

    def __init__(
        self,
        design: DesignVehicle,
        cfg: dict[str, Any],
        dt: float,
        delta_max: float,
        rate_max: float,
        derivative_cutoff_hz: float | None = None,
        nominal_speed: float = 10.0,
    ) -> None:
        self.name = "b6_mpc"
        self._design = design
        self._cfg = cfg
        self._dt = dt
        self._delta_max = delta_max
        self._rate_max = rate_max
        self._nominal_speed = nominal_speed
        self._stride = max(1, int(round(float(cfg["ts_s"]) / dt)))
        self._ay_fraction = float(cfg["ay_limit_fraction"])
        self._rebuild_delta = float(cfg["rebuild_speed_delta_mps"])
        self._deriv = (
            FilteredDerivative(derivative_cutoff_hz, dt) if derivative_cutoff_hz else None
        )
        self.preview_spacing_steps = self._stride
        self.preview_count = int(cfg["np"])
        self._qp: _QP | None = None
        self.reset()

    def reset(self) -> None:
        self._count = 0
        self._held = 0.0
        self._prev = 0.0
        self.solve_failures = 0
        self.solve_times: list[float] = []
        if self._deriv:
            self._deriv.reset()
        self._qp = None

    def _ensure_qp(self, speed: float) -> _QP:
        if self._qp is None or abs(self._qp.speed - speed) > self._rebuild_delta:
            self._qp = _QP(self._design, speed, self._cfg, self._rate_max)
        return self._qp

    def steer_limit(self, speed: float, mu_belief: float) -> float:
        d = self._design
        ay_max = self._ay_fraction * mu_belief * d.gravity
        angle = d.wheelbase * ay_max / speed**2 + d.understeer_coeff * ay_max
        return min(angle, self._delta_max)

    def command(self, obs: Observation) -> Command:
        speed = obs.vx if obs.vx > 0.0 else self._nominal_speed
        de_y = self._deriv.update(obs.e_y) if self._deriv else obs.de_y
        tick = self._count
        self._count += 1
        if tick % self._stride != 0:
            return Command(steer=self._held, gains=(0.0, 0.0, 0.0, 0.0))

        t0 = time.perf_counter()
        qp = self._ensure_qp(speed)
        d = self._design
        x0 = np.array([-obs.e_y, -de_y, -obs.heading_error, obs.yaw_rate - obs.yaw_rate_ref])
        preview = np.asarray(obs.r_ref_preview, dtype=float)
        if preview.size == 0:
            preview = np.full(qp.np_, obs.yaw_rate_ref)
        if preview.size < qp.np_:
            preview = np.concatenate([preview, np.full(qp.np_ - preview.size, preview[-1])])
        preview = preview[: qp.np_]
        u_ff = (d.wheelbase / speed + d.understeer_coeff * speed) * preview[: qp.nc]
        pred0 = qp.sx @ x0 + qp.sw @ preview
        q_vec = 2.0 * (qp.su.T @ qp.qb @ pred0) - 2.0 * qp.r_u * u_ff
        q_vec[0] -= 2.0 * qp.r_du * self._prev
        limit = self.steer_limit(speed, obs.mu_belief)
        nc = qp.nc
        lo = np.concatenate([-limit * np.ones(nc), -qp.step * np.ones(nc)])
        hi = -lo
        lo[nc] += self._prev
        hi[nc] += self._prev
        qp.prob.update(q=q_vec, l=lo, u=hi)
        res = qp.prob.solve(raise_error=False)
        if res.info.status in ("solved", "solved inaccurate") and np.all(np.isfinite(res.x)):
            self._held = float(clip(res.x[0], -limit, limit))
        else:
            self.solve_failures += 1
        self._prev = self._held
        elapsed = time.perf_counter() - t0
        self.solve_times.append(elapsed)
        return Command(steer=self._held, gains=(0.0, 0.0, 0.0, 0.0), solve_time_s=elapsed)
