from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ArticulatedParams:
    """Kinematic and dynamic parameters for Class 8 tractor-semitrailer."""

    # Tractor parameters
    m1: float = 8000.0  # Tractor mass (kg)
    iz1: float = 35000.0  # Tractor yaw inertia (kg*m^2)
    lf: float = 1.4  # Front axle to CG (m)
    lr: float = 2.4  # Rear axle to CG (m)
    d1: float = 0.3  # Hitch offset behind rear axle (m)
    cf1: float = 180000.0  # Front cornering stiffness (N/rad)
    cr1: float = 320000.0  # Rear cornering stiffness (N/rad)

    # Trailer parameters
    m2: float = 24000.0  # Trailer mass (kg laden)
    iz2: float = 120000.0  # Trailer yaw inertia (kg*m^2)
    l2: float = 8.5  # Kingpin to trailer axle distance (m)
    cr2: float = 360000.0  # Trailer axle cornering stiffness (N/rad)
    cg_height2: float = 2.2  # Trailer CG height for rollover analysis (m)
    track_width2: float = 2.4  # Trailer track width (m)

    gravity: float = 9.80665
    tau_air: float = 0.25  # Pneumatic brake transport latency (s)
    max_brake_pressure_bar: float = 6.5  # Full pneumatic brake line supply (bar)

    @property
    def wheelbase(self) -> float:
        """Tractor wheelbase (m): lf + lr."""
        return self.lf + self.lr

    @property
    def understeer_gradient(self) -> float:
        """Tractor neutral/understeer gradient Kus (rad / (m/s^2)):

        Kus = (m1 * lr / (L * Cf1)) - (m1 * lf / (L * Cr1))
        """
        w_f = self.m1 * self.lr / self.wheelbase
        w_r = self.m1 * self.lf / self.wheelbase
        return (w_f / self.cf1) - (w_r / self.cr1)


@dataclass(frozen=True)
class ArticulatedState:
    """State vector of the tractor-semitrailer vehicle."""

    x: float = 0.0
    y: float = 0.0
    psi1: float = 0.0  # Tractor yaw heading (rad)
    vx: float = 10.0  # Forward speed (m/s)
    vy1: float = 0.0  # Tractor lateral velocity at CG (m/s)
    r1: float = 0.0  # Tractor yaw rate (rad/s)
    theta_a: float = 0.0  # Articulation angle (rad): psi1 - psi2
    r2: float = 0.0  # Trailer yaw rate (rad/s)


@dataclass(frozen=True)
class ArticulatedStepResult:
    """Outputs from stepping the articulated vehicle plant."""

    state: ArticulatedState
    ay1: float  # Tractor lateral acceleration (m/s^2)
    ay2: float  # Trailer lateral acceleration (m/s^2)
    theta_a: float  # Hitch articulation angle (rad)
    theta_a_dot: float  # Articulation rate (rad/s)
    ltr: float  # Trailer Load Transfer Ratio (rollover metric)


class ArticulatedVehicle:
    """Dynamic tractor-semitrailer plant with hitch kinematics and trailer roll dynamics."""

    def __init__(self, params: ArticulatedParams | None = None) -> None:
        self.p = params or ArticulatedParams()
        self.state = ArticulatedState()

    def reset(self, vx: float = 15.0, theta_a0: float = 0.0) -> ArticulatedState:
        self.state = ArticulatedState(vx=max(vx, 1.0), theta_a=theta_a0)
        return self.state

    def step(self, delta: float, ax: float, dt: float, mu: float = 0.8) -> ArticulatedStepResult:
        p = self.p
        s = self.state

        vx = max(s.vx + ax * dt, 0.5)
        vy1 = s.vy1
        r1 = s.r1
        theta_a = s.theta_a

        # Kinematic articulation rate at fifth wheel
        cos_th = math.cos(theta_a)
        sin_th = math.sin(theta_a)
        v_hitch_lat = vy1 - p.d1 * r1
        r2 = (vx * sin_th - v_hitch_lat * cos_th) / p.l2
        theta_a_dot = r1 - r2

        # Tire slip angles
        alpha_f = delta - (vy1 + p.lf * r1) / vx
        alpha_r = -(vy1 - p.lr * r1) / vx
        alpha_t = -(v_hitch_lat * cos_th - vx * sin_th - p.l2 * r2) / vx

        # Normal loads
        fz1_f = 0.5 * (p.m1 * p.lf / (p.lf + p.lr)) * p.gravity
        fz1_r = 0.5 * (p.m1 * p.lr / (p.lf + p.lr) + 0.3 * p.m2) * p.gravity
        fz2 = 0.7 * p.m2 * p.gravity
        fz_tot = fz1_f + fz1_r + fz2

        # Combined slip: Kamm's friction circle lateral capacity reduction
        fx_tot = (p.m1 + p.m2) * abs(ax)
        fx1_f = (fz1_f / fz_tot) * fx_tot
        fx1_r = (fz1_r / fz_tot) * fx_tot
        fx2 = (fz2 / fz_tot) * fx_tot

        # Lateral forces with coupled friction saturation
        fy_max_f = math.sqrt(max(0.0, (mu * fz1_f) ** 2 - fx1_f**2))
        fy_max_r = math.sqrt(max(0.0, (mu * fz1_r) ** 2 - fx1_r**2))
        fy_max_t = math.sqrt(max(0.0, (mu * fz2) ** 2 - fx2**2))

        fy1_f = min(max(p.cf1 * alpha_f, -fy_max_f), fy_max_f)
        fy1_r = min(max(p.cr1 * alpha_r, -fy_max_r), fy_max_r)
        fy2 = min(max(p.cr2 * alpha_t, -fy_max_t), fy_max_t)

        # Equations of lateral motion
        # m1 * ay1 = Fy_f + Fy_r + F_hitch_y
        m_tot = p.m1 + p.m2
        ay1 = (fy1_f + fy1_r + fy2 * cos_th) / m_tot
        ay2 = ay1 - p.d1 * (r1 * r1) - p.l2 * (r2 * r2) * cos_th

        # Yaw acceleration
        r1_dot = (p.lf * fy1_f - p.lr * fy1_r - p.d1 * fy2 * cos_th) / p.iz1

        # Integration
        vy1_new = vy1 + (ay1 - vx * r1) * dt
        r1_new = r1 + r1_dot * dt
        theta_a_new = theta_a + theta_a_dot * dt
        psi1_new = s.psi1 + r1 * dt
        x_new = s.x + (vx * math.cos(s.psi1) - vy1 * math.sin(s.psi1)) * dt
        y_new = s.y + (vx * math.sin(s.psi1) + vy1 * math.cos(s.psi1)) * dt

        # Load Transfer Ratio (LTR) on trailer axle
        ltr = min(1.0, (2.0 * p.cg_height2 / p.track_width2) * (abs(ay2) / p.gravity))

        self.state = ArticulatedState(
            x=x_new,
            y=y_new,
            psi1=psi1_new,
            vx=vx,
            vy1=vy1_new,
            r1=r1_new,
            theta_a=theta_a_new,
            r2=r2,
        )

        return ArticulatedStepResult(
            state=self.state,
            ay1=ay1,
            ay2=ay2,
            theta_a=theta_a_new,
            theta_a_dot=theta_a_dot,
            ltr=ltr,
        )
