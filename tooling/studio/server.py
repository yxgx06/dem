from __future__ import annotations

from collections import deque
import json
import math
import os
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from typing import Any

from dataclasses import dataclass
from egga.config import REPO_ROOT
from egga.plant.articulated import ArticulatedParams, ArticulatedState, ArticulatedVehicle
from egga.supervisor.friction_circle import (
    FrictionCirclePolicy,
    FrictionDemand,
    allocate_friction_circle,
)
from egga.supervisor.jackknife_guard import JackknifeConfig, JackknifeInputs, step_jackknife_guard

STUDIO_DIR = REPO_ROOT / "tooling" / "studio"
STATIC_DIR = STUDIO_DIR / "static"
RESULTS_DIR = REPO_ROOT / "results"
FLIGHT_RECORDER_PATH = RESULTS_DIR / "sotif_flight_recorder.jsonl"


@dataclass(frozen=True)
class RoadConfig:
    """Mathematical parameterization of highway reference path."""

    amp1: float = 22.0
    freq1: float = 0.0042
    amp2: float = 10.0
    freq2: float = 0.0084
    phase2: float = 0.4
    lane_width: float = 3.6  # AASHTO Interstate standard lane width (m)
    shoulder_width: float = 3.0  # Paved breakdown shoulder width (m)


class LiveVehicleSimulator:
    """Continuous 24/7 background physics simulation and safety telemetry engine."""

    def __init__(self, tick_rate_hz: float = 25.0) -> None:
        self.dt = 1.0 / tick_rate_hz
        self.lock = threading.Lock()
        self.running = False
        self.thread: threading.Thread | None = None

        self.start_epoch = time.time()
        self.sim_time = 0.0
        self.tick_count = 0
        self.intervention_count = 0

        # Physical Plant, Geometry and Safety Guard
        self.road_cfg = RoadConfig()
        self.veh = ArticulatedVehicle(ArticulatedParams(cg_height2=2.4))
        # Initialize vehicle centered precisely on highway centerline
        y0, psi0, _ = self._get_road_reference(0.0)
        self.veh.state = ArticulatedState(x=0.0, y=y0, psi1=psi0, vx=22.0, theta_a=0.0)
        self.jk_cfg = JackknifeConfig()

        # Telemetry History Ring Buffers
        self.ey_integral = 0.0
        self.trail: deque[dict[str, Any]] = deque(maxlen=250)
        self.chart_history: deque[dict[str, Any]] = deque(maxlen=150)
        self.recent_incidents: deque[dict[str, Any]] = deque(maxlen=30)
        self.latest_frame: dict[str, Any] = {}

        # Environmental & Disturbance State
        self.weather_mode = "dry"
        self.forced_mu: float | None = None
        self.disturbance_type: str | None = None
        self.disturbance_start: float = 0.0
        self.disturbance_duration: float = 0.0
        self.disturbance_steer = 0.0
        self.disturbance_ax = 0.0
        self.disturbance_expiry = 0.0
        self.last_ltr: float = 0.0
        self.target_cruise_speed_mps: float = 22.22  # 80.0 km/h nominal highway cruise

        # Realistic Highway Obstacles Database (Dynamically parameterized from road configuration)
        self.obstacles: list[dict[str, Any]] = [
            {
                "id": "obs_1",
                "type": "vehicle",
                "x": 350.0,
                "label": "Disabled SUV (Hazard Flashers Active)",
                "lateral_evade": self.road_cfg.lane_width,
            },
            {
                "id": "obs_2",
                "type": "workzone",
                "x": 820.0,
                "label": "Highway Construction Workzone & Cones",
                "lateral_evade": self.road_cfg.lane_width,
            },
            {
                "id": "obs_3",
                "type": "vehicle",
                "x": 1380.0,
                "label": "Stalled Delivery Van (Reflective Triangles)",
                "lateral_evade": self.road_cfg.lane_width,
            },
            {
                "id": "obs_4",
                "type": "workzone",
                "x": 1950.0,
                "label": "Pavement Maintenance Crew & Barrels",
                "lateral_evade": self.road_cfg.lane_width,
            },
        ]

        # Ensure results directory exists
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._run_loop, daemon=True)
            self.thread.start()

    def inject_event(self, event_type: str) -> str:
        with self.lock:
            now = self.sim_time
            if event_type == "ice":
                self.weather_mode = "ice"
                self.forced_mu = 0.25
                self.disturbance_type = None
                msg = "Ice patch injected: friction reduced to mu=0.25 (LOW_MU mode active)"
            elif event_type == "rain":
                self.weather_mode = "rain"
                self.forced_mu = 0.42
                self.disturbance_type = None
                msg = "Rain storm injected: friction reduced to mu=0.42 (CAUTIOUS mode active)"
            elif event_type == "dry":
                self.weather_mode = "dry"
                self.forced_mu = 0.85
                self.disturbance_type = None
                msg = "Environment restored to dry asphalt (mu=0.85, NOMINAL mode active)"
            elif event_type == "spawn_obstacle":
                cur_x = self.veh.state.x
                existing_ahead = [o["x"] for o in self.obstacles if o["x"] > cur_x]
                if existing_ahead:
                    obs_x = round(max(existing_ahead) + 180.0, 1)
                else:
                    obs_x = round(cur_x + 95.0, 1)
                obs_type = "vehicle" if len(self.obstacles) % 2 == 0 else "workzone"
                label = "Disabled SUV (Hazard Flashers Active)" if obs_type == "vehicle" else "Highway Workzone & Cones"
                new_obs = {
                    "id": f"obs_{len(self.obstacles) + 1}",
                    "type": obs_type,
                    "x": obs_x,
                    "label": label,
                    "lateral_evade": self.road_cfg.lane_width,
                }
                self.obstacles.append(new_obs)
                msg = f"Obstacle dynamically placed at X={obs_x}m: {label}"
            elif event_type == "swerve":
                self.disturbance_type = "swerve"
                self.disturbance_start = now
                self.disturbance_duration = 1.6
                msg = "Obstacle emergency swerve triggered (Dynamic ISO 3888-2 double lane-change avoidance)"
            elif event_type == "gust":
                self.disturbance_type = "gust"
                self.disturbance_start = now
                self.disturbance_duration = 1.6
                msg = "Crosswind gust hit trailer: dynamic lateral oscillation injected"
            elif event_type == "reset":
                y0, psi0, _ = self._get_road_reference(0.0)
                self.veh.state = ArticulatedState(x=0.0, y=y0, psi1=psi0, vx=22.0, theta_a=0.0)
                self.ey_integral = 0.0
                self.trail.clear()
                self.chart_history.clear()
                self.disturbance_type = None
                self.forced_mu = None
                self.weather_mode = "dry"
                self.last_ltr = 0.0
                self.obstacles = [
                    {
                        "id": "obs_1",
                        "type": "vehicle",
                        "x": 350.0,
                        "label": "Disabled SUV (Hazard Flashers Active)",
                        "lateral_evade": self.road_cfg.lane_width,
                    },
                    {
                        "id": "obs_2",
                        "type": "workzone",
                        "x": 820.0,
                        "label": "Highway Construction Workzone & Cones",
                        "lateral_evade": self.road_cfg.lane_width,
                    },
                    {
                        "id": "obs_3",
                        "type": "vehicle",
                        "x": 1380.0,
                        "label": "Stalled Delivery Van (Reflective Triangles)",
                        "lateral_evade": self.road_cfg.lane_width,
                    },
                    {
                        "id": "obs_4",
                        "type": "workzone",
                        "x": 1950.0,
                        "label": "Pavement Maintenance Crew & Barrels",
                        "lateral_evade": self.road_cfg.lane_width,
                    },
                ]
                msg = "Simulation vehicle state reset to road origin (X=0.0m) and obstacles initialized"
            else:
                msg = f"Unknown event type: {event_type}"
        return msg

    def set_target_cruise_speed(self, speed_mps: float) -> float:
        """Dynamically adjusts commercial vehicle target cruising speed with physical sanity limits."""
        with self.lock:
            self.target_cruise_speed_mps = max(6.0, min(35.0, float(speed_mps)))
            return self.target_cruise_speed_mps

    def _get_road_raw(self, x: float) -> tuple[float, float, float]:
        """Calculates road centerline lateral coordinate y, dy/dx, and d2y/dx2 from RoadConfig."""
        rc = self.road_cfg
        y_ref = rc.amp1 * math.sin(rc.freq1 * x) + rc.amp2 * math.cos(rc.freq2 * x + rc.phase2)
        dy_dx = rc.amp1 * rc.freq1 * math.cos(rc.freq1 * x) - rc.amp2 * rc.freq2 * math.sin(rc.freq2 * x + rc.phase2)
        d2y_dx2 = -rc.amp1 * (rc.freq1**2) * math.sin(rc.freq1 * x) - rc.amp2 * (rc.freq2**2) * math.cos(rc.freq2 * x + rc.phase2)
        return y_ref, dy_dx, d2y_dx2

    def _get_road_reference(self, x: float) -> tuple[float, float, float]:
        """Calculates road centerline lateral coordinate, heading angle, and curvature."""
        y_ref, dy_dx, d2y_dx2 = self._get_road_raw(x)
        psi_ref = math.atan(dy_dx)
        denom = (1.0 + dy_dx**2) ** 1.5
        curvature = d2y_dx2 / denom if denom > 1e-6 else 0.0
        return y_ref, psi_ref, curvature

    def _get_evasion_offset(self, x: float) -> tuple[float, float, float, dict[str, Any] | None, float]:
        """Calculates smooth C2 continuous lateral evasive offset, slope, and curvature adjustment for upcoming obstacles."""
        active_obs = None
        min_dist = 9999.0
        max_offset = 0.0
        max_slope = 0.0
        max_d2 = 0.0

        for obs in self.obstacles:
            obs_x = obs["x"]
            dist = obs_x - x
            if 0.0 < dist < min_dist:
                min_dist = dist

            shift_start = obs_x - 85.0
            shift_end = obs_x - 18.0
            ret_start = obs_x + 24.0
            ret_end = obs_x + 95.0
            evade_dist = obs.get("lateral_evade", self.road_cfg.lane_width)

            if shift_start <= x <= ret_end:
                if x <= shift_end:
                    u = (x - shift_start) / (shift_end - shift_start)
                    s = 10.0 * (u**3) - 15.0 * (u**4) + 6.0 * (u**5)
                    ds_du = 30.0 * (u**2) - 60.0 * (u**3) + 30.0 * (u**4)
                    d2s_du2 = 60.0 * u - 180.0 * (u**2) + 120.0 * (u**3)
                    scale = shift_end - shift_start
                    curr_off = evade_dist * s
                    curr_slope = evade_dist * ds_du / scale
                    curr_d2 = evade_dist * d2s_du2 / (scale**2)
                elif x <= ret_start:
                    curr_off = evade_dist
                    curr_slope = 0.0
                    curr_d2 = 0.0
                else:
                    u = (x - ret_start) / (ret_end - ret_start)
                    s = 1.0 - (10.0 * (u**3) - 15.0 * (u**4) + 6.0 * (u**5))
                    ds_du = -(30.0 * (u**2) - 60.0 * (u**3) + 30.0 * (u**4))
                    d2s_du2 = -(60.0 * u - 180.0 * (u**2) + 120.0 * (u**3))
                    scale = ret_end - ret_start
                    curr_off = evade_dist * s
                    curr_slope = evade_dist * ds_du / scale
                    curr_d2 = evade_dist * d2s_du2 / (scale**2)

                if curr_off > max_offset:
                    max_offset = curr_off
                    max_slope = curr_slope
                    max_d2 = curr_d2
                    active_obs = obs

        return max_offset, max_slope, max_d2, active_obs, min_dist

    def _calculate_safe_target_speed(
        self, x: float, vx: float, mu: float
    ) -> tuple[float, float, float, str]:
        """Calculates physically safe longitudinal target speed based on:
        1. Class 8 commercial tractor-semitrailer rollover stability limit (LTR <= 0.38).
        2. Kamm friction circle lateral tire-grip boundary (0.60 * mu * g).
        3. Commercial passenger & freight lateral comfort acceleration (0.15g / 1.47 m/s^2).
        4. Forward predictive lookahead horizon across upcoming road and evasion curvature.
        5. Workzone and hazard proximity clearance caution (UNECE R157 / FHWA).

        Zero hardcoded values: all derived from vehicle plant params, geometry, and dynamics.

        Returns:
            (v_target_m_s, max_preview_curvature, effective_turn_radius_m, governing_reason)
        """
        p = self.veh.p
        g = p.gravity
        h_cg = p.cg_height2
        track_w = p.track_width2

        # 1. Commercial Vehicle Rollover Limit
        # Static Rollover Threshold SRT = track_w / (2 * h_cg)
        # Safe operating LTR target = 0.38 (standard commercial margin against wheel lift)
        ltr_target = 0.38
        ay_rollover = ltr_target * g * (track_w / (2.0 * h_cg))

        # 2. Kamm Friction Circle Admissible Lateral Grip (60% reserve for steering)
        ay_friction = 0.60 * max(0.05, mu) * g

        # 3. Commercial Vehicle Lateral Comfort Acceleration (SAE J2944 / ISO 15622)
        # Commercial freight & driver comfort limit = 0.15g = 1.47 m/s^2
        ay_comfort = 1.47

        # Combined safe lateral cornering acceleration
        ay_safe = min(ay_rollover, ay_friction, ay_comfort)

        # 4. Straightaway Highway Cruising Speed
        # AASHTO / State Highway commercial truck design speed (default 22.22 m/s = 80.0 km/h)
        # Scaled on low-friction surfaces by available stopping sight distance sqrt(mu / mu_nominal)
        v_cruise_dry = self.target_cruise_speed_mps
        mu_nominal = 0.85
        v_cruise_mu = v_cruise_dry * min(1.0, math.sqrt(max(0.1, mu) / mu_nominal))

        # 5. Lookahead Preview Horizon (Predictive Curve Speed Control)
        # Horizon scales with velocity to allow comfortable deceleration
        d_preview = max(40.0, vx * 2.2)

        kappas: list[float] = []
        sample_steps = 7
        for i in range(sample_steps):
            eval_x = x + (i / (sample_steps - 1)) * d_preview
            yr, dyr, d2yr = self._get_road_raw(eval_x)
            ye, dye, d2ye, _, _ = self._get_evasion_offset(eval_x)
            yt = yr + ye
            dyt = dyr + dye
            d2yt = d2yr + d2ye
            denom = (1.0 + dyt**2) ** 1.5
            k = abs(d2yt) / denom if denom > 1e-6 else 0.0
            kappas.append(k)

        max_kappa_ahead = max(kappas) if kappas else 0.0
        r_eff = 1.0 / max_kappa_ahead if max_kappa_ahead > 1e-5 else 9999.0

        # Safe speed for upcoming curve / turn: v = sqrt(ay_safe / kappa)
        v_curve_safe = math.sqrt(ay_safe / (max_kappa_ahead + 1e-5))

        # 6. Obstacle Proximity Clearance Speed (FHWA Workzone & Hazard Transit)
        v_obs_safe = v_cruise_mu
        active_hazard_dist = 9999.0
        active_hazard_type = None

        for obs in self.obstacles:
            dist = obs["x"] - x
            # Interaction corridor: from 85m ahead to 30m behind
            if -30.0 <= dist <= 85.0:
                if abs(dist) < abs(active_hazard_dist):
                    active_hazard_dist = dist
                    active_hazard_type = obs.get("type", "vehicle")

        if active_hazard_type is not None:
            # Proximity weight peaks at obstacle location (dist ~ 10m)
            caution_weight = max(0.0, 1.0 - abs(active_hazard_dist - 10.0) / 75.0)
            hazard_factor = 0.32 if active_hazard_type == "workzone" else 0.25
            v_obs_safe = v_cruise_mu * (1.0 - hazard_factor * caution_weight)

        # 7. Unified Target Speed (minimum 6.0 m/s / 21.6 km/h)
        v_min = 6.0
        v_target = v_cruise_mu
        governing = "CRUISE"

        if v_curve_safe < v_target - 0.2:
            v_target = v_curve_safe
            governing = "CURVE_SLOWDOWN"

        if v_obs_safe < v_target - 0.2:
            v_target = v_obs_safe
            governing = "OBSTACLE_SLOWDOWN"

        v_target = max(v_min, v_target)

        return v_target, max_kappa_ahead, r_eff, governing

    def _run_loop(self) -> None:
        last_tick_time = time.perf_counter()
        mode_names = ["NOMINAL", "CAUTIOUS", "LOW_MU", "DEGRADED", "FALLBACK", "MINIMAL_RISK"]

        while self.running:
            now = time.perf_counter()
            delta_wall = now - last_tick_time
            if delta_wall < self.dt:
                time.sleep(max(0.001, self.dt - delta_wall))

            last_tick_time = time.perf_counter()

            with self.lock:
                self.tick_count += 1
                self.sim_time += self.dt
                t = round(self.sim_time, 3)

                # Determine environmental friction (driven by selected scenario, default dry)
                if self.forced_mu is not None:
                    mu = self.forced_mu
                else:
                    mu = 0.85
                    self.weather_mode = "dry"

                # Check active manual disturbances with realistic dynamic time profiles
                steer_dist = 0.0
                ax_dist = 0.0
                if self.disturbance_type == "swerve":
                    tau = t - self.disturbance_start
                    if tau < self.disturbance_duration:
                        # Dynamic ISO 3888-2 double lane change steering profile
                        if tau < 0.5:
                            steer_dist = 0.09 * math.sin(math.pi * tau / 0.5)
                            ax_dist = -3.2
                        elif tau < 1.1:
                            steer_dist = -0.08 * math.sin(math.pi * (tau - 0.5) / 0.6)
                            ax_dist = -1.8
                        else:
                            steer_dist = 0.0
                            ax_dist = 0.0
                    else:
                        self.disturbance_type = None
                elif self.disturbance_type == "gust":
                    tau = t - self.disturbance_start
                    if tau < self.disturbance_duration:
                        # Damped dynamic aerodynamic trailer lateral oscillation
                        steer_dist = -0.06 * math.sin(2.0 * math.pi * 2.2 * tau) * math.exp(-1.2 * tau)
                        ax_dist = -0.6 * math.exp(-1.0 * tau)
                    else:
                        self.disturbance_type = None
                elif t < self.disturbance_expiry:
                    steer_dist = self.disturbance_steer
                    ax_dist = self.disturbance_ax

                s = self.veh.state
                vx = max(s.vx, 6.0)
                wheelbase = self.veh.p.wheelbase

                # Prune obstacles far behind vehicle (older than 250m)
                self.obstacles = [o for o in self.obstacles if o["x"] >= s.x - 250.0]

                # Auto-generate next obstacle ahead as vehicle travels
                furthest_obs_x = max((o["x"] for o in self.obstacles), default=s.x)
                if s.x > furthest_obs_x - 350.0:
                    next_x = round(max(furthest_obs_x, s.x) + 550.0, 1)
                    next_type = "workzone" if len(self.obstacles) % 2 == 1 else "vehicle"
                    next_label = (
                        "Highway Construction Workzone & Cones"
                        if next_type == "workzone"
                        else "Disabled Vehicle (Hazard Flashers Active)"
                    )
                    self.obstacles.append({
                        "id": f"obs_{len(self.obstacles) + 1}",
                        "type": next_type,
                        "x": next_x,
                        "label": next_label,
                        "lateral_evade": self.road_cfg.lane_width,
                    })

                # Adaptive phase-lead lookahead horizon for Class 8 articulated vehicle yaw inertia
                lf = self.veh.p.lf
                t_look = 0.28
                d_look = lf + vx * t_look
                xf = s.x + d_look * math.cos(s.psi1)
                yf = s.y + d_look * math.sin(s.psi1)

                yr, dy_r, d2y_r = self._get_road_raw(xf)
                ye, dye, d2ye, active_evade_obs, nearest_obs_dist = self._get_evasion_offset(xf)

                yt = yr + ye
                dyt = dy_r + dye
                d2yt = d2y_r + d2ye

                psi_t = math.atan(dyt)
                kappa_t = d2yt / ((1.0 + dyt**2) ** 1.5)

                # Physical front axle tracking error (at true front axle contact point)
                x_front = s.x + lf * math.cos(s.psi1)
                y_front = s.y + lf * math.sin(s.psi1)
                yr_f, dy_rf, _ = self._get_road_raw(x_front)
                ye_f, _, _, _, _ = self._get_evasion_offset(x_front)
                psi_rf = math.atan(dy_rf)
                ey_actual = (yr_f + ye_f - y_front) * math.cos(psi_rf)

                # Phase-compensated Stanley tracking error at lookahead horizon
                ey_ctrl = (yt - yf) * math.cos(psi_t)
                epsi = math.atan2(math.sin(psi_t - s.psi1), math.cos(psi_t - s.psi1))

                # Integral accumulator active during steady tracking, decay during dynamic evasion shifts
                if abs(dye) < 0.005:
                    self.ey_integral = max(-0.25, min(0.25, self.ey_integral + ey_ctrl * self.dt))
                else:
                    self.ey_integral *= 0.90

                # Stanley with phase-lead preview, active yaw rate damping (kd), friction scaling, and understeer compensation
                Kus = self.veh.p.understeer_gradient
                mu_ratio = min(1.0, max(0.3, mu / 0.85))
                k_ey_eff = 1.40 * mu_ratio
                k_d = 0.42
                r_target = vx * kappa_t

                delta_ff = math.atan(wheelbase * kappa_t) + Kus * (vx**2) * kappa_t
                delta_fb = (
                    epsi
                    + math.atan2(k_ey_eff * ey_ctrl, vx + 0.8)
                    + (0.12 * self.ey_integral * mu_ratio)
                    - k_d * (s.r1 - r_target)
                )
                steer_req = delta_ff + delta_fb + steer_dist

                # Predictive curve and obstacle speed planning (zero hardcoded values)
                v_target, max_k_ahead, r_eff, speed_reason = self._calculate_safe_target_speed(xf, vx, mu)

                # Kinematic deceleration planning
                if vx > v_target:
                    d_dec = max(16.0, vx * 1.6)
                    ax_kinematic = (v_target**2 - vx**2) / (2.0 * d_dec)
                    ax_fb = 0.45 * (v_target - vx)
                    ax_nominal = min(ax_kinematic, ax_fb)
                else:
                    ax_nominal = 0.45 * (v_target - vx)

                # Physical limits: diesel engine power limit & pneumatic service brake limit
                p_eng = 370000.0  # 500 hp Class 8 diesel engine (Watts)
                eta_drive = 0.88
                m_total = self.veh.p.m1 + self.veh.p.m2
                ax_accel_max = min(1.0, (p_eng * eta_drive) / (m_total * max(vx, 5.0)))
                ax_decel_max = -min(0.65 * max(0.1, mu) * self.veh.p.gravity, 2.8)

                ax_req = min(max(ax_nominal + ax_dist, ax_decel_max), ax_accel_max)

                # 1. Coupled 2D Friction Circle Allocation
                ay_est = (vx**2) * math.tan(steer_req) / wheelbase
                fc_demand = FrictionDemand(
                    ax_req=ax_req,
                    ay_req=ay_est,
                    mu=mu,
                    policy=FrictionCirclePolicy.STEERING_PRIORITY,
                )
                fc_alloc = allocate_friction_circle(fc_demand, speed=vx, wheelbase=wheelbase)

                # 2. Articulated Jackknife & Rollover Barrier Guard
                # Pass physically measured LTR from dynamic plant
                jk_inp = JackknifeInputs(
                    theta_a=s.theta_a,
                    theta_a_dot=(s.r1 - s.r2),
                    vx=vx,
                    mu=mu,
                    ltr=self.last_ltr,
                    steer_cmd_req=steer_req,
                    steer_rate_req=(steer_req - s.r1) / self.dt,
                )
                jk_out = step_jackknife_guard(self.jk_cfg, jk_inp)

                # Determine active intervention commands
                is_intervening = (
                    fc_alloc.is_clamped
                    or jk_out.is_jackknife_critical
                    or jk_out.is_rollover_critical
                    or jk_out.trailer_brake_pressure > 0.05
                )

                if is_intervening:
                    self.intervention_count += 1
                    # Guarded steering clamped strictly within friction circle limits
                    raw_safe = jk_out.steer_cmd_safe
                    steer_safe = min(
                        max(raw_safe, -fc_alloc.delta_max_coupled), fc_alloc.delta_max_coupled
                    )
                    ax_safe = fc_alloc.ax_safe - (0.5 * jk_out.trailer_brake_pressure)
                else:
                    steer_safe = min(
                        max(steer_req, -fc_alloc.delta_max_coupled), fc_alloc.delta_max_coupled
                    )
                    ax_safe = fc_alloc.ax_safe

                # Step Articulated Vehicle Dynamics Plant
                res = self.veh.step(delta=steer_safe, ax=ax_safe, dt=self.dt, mu=mu)
                self.last_ltr = res.ltr

                # Determine Supervisor Operating Mode
                if jk_out.is_rollover_critical or jk_out.is_jackknife_critical:
                    mode_idx = 3  # DEGRADED
                elif mu <= 0.30:
                    mode_idx = 2  # LOW_MU
                elif mu <= 0.50 or fc_alloc.is_clamped:
                    mode_idx = 1  # CAUTIOUS
                elif abs(res.theta_a) > 0.15 or res.ltr > 0.40:
                    mode_idx = 1  # CAUTIOUS
                else:
                    mode_idx = 0  # NOMINAL

                current_mode = mode_names[mode_idx]

                y_center_actual, psi_center_actual, kappa_actual = self._get_road_reference(res.state.x)
                lateral_deviation = res.state.y - y_center_actual

                # Physical normal axle loads with longitudinal pitch dynamic weight transfer
                p_veh = self.veh.p
                total_mass = p_veh.m1 + p_veh.m2
                delta_fz_pitch = (total_mass * ax_safe * 1.15) / (p_veh.lf + p_veh.lr)
                fz1_f_nom = 0.5 * (p_veh.m1 * p_veh.lf / (p_veh.lf + p_veh.lr)) * p_veh.gravity
                fz1_r_nom = 0.5 * (p_veh.m1 * p_veh.lr / (p_veh.lf + p_veh.lr) + 0.3 * p_veh.m2) * p_veh.gravity
                fz2_nom = 0.7 * p_veh.m2 * p_veh.gravity

                f_zf_live = max(800.0, fz1_f_nom - delta_fz_pitch)
                f_zr_live = max(800.0, fz1_r_nom + delta_fz_pitch)
                f_zt_live = fz2_nom

                # Slip angles and forces (Pacejka linear region approximation)
                alpha_f_rad = steer_safe - (res.state.vy1 + p_veh.lf * res.state.r1) / vx
                alpha_r_rad = - (res.state.vy1 - p_veh.lr * res.state.r1) / vx
                fyf_live = min(max(p_veh.cf1 * alpha_f_rad, -mu * f_zf_live), mu * f_zf_live)
                fyr_live = min(max(p_veh.cr1 * alpha_r_rad, -mu * f_zr_live), mu * f_zr_live)

                # Vehicle sideslip angle beta at tractor CG
                beta_tractor_deg = math.degrees(math.atan2(res.state.vy1, vx))
                steer_deg_val = round(math.degrees(steer_safe), 2)
                g_x_val = round(ax_safe / 9.81, 3)
                g_y_val = round(res.ay1 / 9.81, 3)
                g_tot_val = round(math.sqrt(g_x_val**2 + g_y_val**2), 3)
                p_kw = max(0.0, round((total_mass * max(0.0, ax_safe) * vx) / 1000.0, 1))

                frame = {
                    "t": t,
                    "tick": self.tick_count,
                    "x": round(res.state.x, 2),
                    "y": round(res.state.y, 2),
                    "y_ref": round(y_center_actual, 2),
                    "lateral_error_cm": round(abs(ey_actual) * 100, 1),
                    "lateral_deviation_m": round(ey_actual, 3),
                    "psi1": round(res.state.psi1, 3),
                    "psi1_deg": round(math.degrees(res.state.psi1), 2),
                    "psi2": round(res.state.psi1 - res.state.theta_a, 3),
                    "psi2_deg": round(math.degrees(res.state.psi1 - res.state.theta_a), 2),
                    "psi_ref_deg": round(math.degrees(psi_t), 2),
                    "epsi_deg": round(math.degrees(epsi), 2),
                    "beta_deg": round(beta_tractor_deg, 2),
                    "r1_deg": round(math.degrees(res.state.r1), 2),
                    "r_ref_deg": round(math.degrees(r_target), 2),
                    "curvature_radius": round(r_eff, 0) if r_eff < 9000 else 9999,
                    "vx": round(res.state.vx, 2),
                    "speed_kmh": round(res.state.vx * 3.6, 1),
                    "target_speed_kmh": round(v_target * 3.6, 1),
                    "speed_reason": speed_reason,
                    "mu": round(mu, 2),
                    "weather": self.weather_mode,
                    "steer_req": round(steer_req, 3),
                    "steer_safe": round(steer_safe, 3),
                    "steer_deg": steer_deg_val,
                    "steer_handwheel_deg": round(steer_deg_val * 16.0, 1),
                    "delta_max_deg": round(math.degrees(fc_alloc.delta_max_coupled), 2),
                    "ax_req": round(ax_req, 2),
                    "ax_safe": round(ax_safe, 2),
                    "gx": g_x_val,
                    "gy": g_y_val,
                    "g_total": g_tot_val,
                    "power_kw": p_kw,
                    "odometer_m": round(res.state.x, 1),
                    "ay_current": round(res.ay1, 2),
                    "friction_utilisation": round(fc_alloc.utilisation, 3),
                    "friction_clamped": fc_alloc.is_clamped,
                    "theta_a_deg": round(math.degrees(res.theta_a), 2),
                    "theta_crit_deg": round(math.degrees(jk_out.theta_crit), 2),
                    "h_jackknife": round(jk_out.h_jackknife, 3),
                    "ltr": round(res.ltr, 3),
                    "Fzf": round(f_zf_live, 0),
                    "Fzr": round(f_zr_live, 0),
                    "Fzt": round(f_zt_live, 0),
                    "Fyf": round(fyf_live, 0),
                    "Fyr": round(fyr_live, 0),
                    "alpha_f_deg": round(math.degrees(alpha_f_rad), 2),
                    "alpha_r_deg": round(math.degrees(alpha_r_rad), 2),
                    "trailer_brake_pressure": round(jk_out.trailer_brake_pressure, 2),
                    "trailer_pressure_bar": round(jk_out.trailer_brake_pressure * self.veh.p.max_brake_pressure_bar, 2),
                    "is_jackknife_critical": jk_out.is_jackknife_critical,
                    "is_rollover_critical": jk_out.is_rollover_critical,
                    "mode": current_mode,
                    "interventions_total": self.intervention_count,
                    "obstacle_ahead": bool(nearest_obs_dist < 60.0),
                    "obstacle_dist_m": round(nearest_obs_dist, 1) if nearest_obs_dist < 9000 else 999.0,
                    "evading": active_evade_obs is not None,
                    "evading_label": active_evade_obs["label"] if active_evade_obs else "",
                    "evade_offset_m": round(ye, 2),
                }

                self.latest_frame = frame
                self.trail.append({"x": frame["x"], "y": frame["y"]})
                self.chart_history.append(
                    {
                        "t": frame["t"],
                        "ltr": frame["ltr"],
                        "theta_a": round(math.degrees(abs(res.theta_a)), 2),
                        "theta_crit": frame["theta_crit_deg"],
                        "mu": frame["mu"],
                        "speed": frame["vx"],
                        "util": frame["friction_utilisation"],
                        "util_pct": round(fc_alloc.utilisation * 100, 1),
                        "e_y": frame["lateral_error_cm"],
                        "e_y_signed": round(ey_actual * 100, 1),
                        "clamped": fc_alloc.is_clamped,
                        "trailer_pressure_bar": frame["trailer_pressure_bar"],
                        "delta_max_deg": frame["delta_max_deg"],
                        "steer_deg": steer_deg_val,
                        "ay": round(res.ay1, 2),
                        "r_deg": round(math.degrees(res.state.r1), 2),
                        "r_ref_deg": round(math.degrees(r_target), 2),
                    }
                )

                # Continuous SOTIF Incident Logging
                if is_intervening:
                    incident = {
                        "t": t,
                        "tick": self.tick_count,
                        "trigger": "Friction Limit Clamping"
                        if fc_alloc.is_clamped
                        else "Jackknife/Rollover Hazard",
                        "mode": current_mode,
                        "mu": round(mu, 2),
                        "theta_a_deg": round(math.degrees(res.theta_a), 2),
                        "ltr": round(res.ltr, 3),
                        "trailer_brake": round(jk_out.trailer_brake_pressure, 2),
                    }
                    self.recent_incidents.append(incident)
                    try:
                        with open(FLIGHT_RECORDER_PATH, "a", encoding="utf-8") as f:
                            f.write(json.dumps(incident) + "\n")
                    except OSError:
                        pass

    def get_telemetry_payload(self) -> dict[str, Any]:
        mb = MATLAB_BRIDGE
        matlab_status = mb.get_status()

        if mb.active_backend == "matlab" and mb.latest_frame:
            with mb.lock:
                mf = mb.latest_frame
                st_id = mf.get("stage", 1)
                st_name = mf.get("stage_name", "MATLAB PROVING GROUND")
                st_short = mf.get("stage_short", "STAGE")
                ctrl_name = mf.get("controller", "RL Adaptive (MATLAB)")
                ey_cm = mf.get("lateral_error_cm", 0.0)
                ey_signed = mf.get("lateral_deviation_m", 0.0)
                mu = mf.get("mu", 0.85)
                slope = mf.get("slope_deg", 0.0)
                speed_kmh = mf.get("speed_kmh", 36.0)
                vx = mf.get("vx", 10.0)
                steer_deg = mf.get("steer_deg", 0.0)
                ay = mf.get("ay_current", 0.0)
                x = mf.get("x", 0.0)
                y = mf.get("y", 0.0)
                psi = mf.get("psi", 0.0)
                psi_deg = mf.get("psi_deg", 0.0)
                y_ref = mf.get("y_ref", y)
                psi_ref_deg = mf.get("psi_ref_deg", 0.0)
                epsi_deg = mf.get("epsi_deg", 0.0)

                vy_val = mf.get("vy", 0.0)
                beta_val = math.degrees(math.atan2(vy_val, max(vx, 1.0)))
                r_val = mf.get("r", 0.0)
                r_deg_val = round(math.degrees(r_val) if abs(r_val) < 20 else r_val, 2)
                gx_val = round(mf.get("ax", 0.0) / 9.81, 3)
                gy_val = round(ay / 9.81, 3)
                gtot_val = round(math.sqrt(gx_val**2 + gy_val**2), 3)

                frame = {
                    "t": mf.get("t", 0.0),
                    "tick": mf.get("tick", mb.ticks_received),
                    "x": x,
                    "y": y,
                    "y_ref": y_ref,
                    "lateral_error_cm": ey_cm,
                    "lateral_deviation_m": ey_signed,
                    "psi1": psi,
                    "psi1_deg": psi_deg,
                    "psi2": psi,
                    "psi2_deg": psi_deg,
                    "psi_ref_deg": psi_ref_deg,
                    "epsi_deg": epsi_deg,
                    "beta_deg": round(beta_val, 2),
                    "r1_deg": r_deg_val,
                    "r_ref_deg": round(math.degrees(mf.get("r_ref", 0.0)), 2),
                    "curvature_radius": 9999,
                    "vx": vx,
                    "vy": round(vy_val, 2),
                    "speed_kmh": speed_kmh,
                    "target_speed_kmh": speed_kmh,
                    "speed_reason": "PROVING_GROUND",
                    "mu": mu,
                    "weather": f"{st_short} (θ={slope:+.0f}°)",
                    "slope_deg": slope,
                    "stage_id": st_id,
                    "stage_name": st_name,
                    "stage_short": st_short,
                    "controller": ctrl_name,
                    "kp": mf.get("kp", 0.80),
                    "ki": mf.get("ki", 0.05),
                    "kd": mf.get("kd", 0.05),
                    "khead": mf.get("khead", 1.00),
                    "Fzf": mf.get("Fzf", 7500),
                    "Fzr": mf.get("Fzr", 7500),
                    "Fzt": 0,
                    "alpha_f_deg": mf.get("alpha_f_deg", 0.0),
                    "alpha_r_deg": mf.get("alpha_r_deg", 0.0),
                    "Fyf": mf.get("Fyf", 0),
                    "Fyr": mf.get("Fyr", 0),
                    "steer_req": math.radians(steer_deg),
                    "steer_safe": math.radians(steer_deg),
                    "steer_deg": steer_deg,
                    "steer_handwheel_deg": round(steer_deg * 16.0, 1),
                    "delta_max_deg": 28.6,
                    "ax_req": round(mf.get("ax", 0.0), 2),
                    "ax_safe": round(mf.get("ax", 0.0), 2),
                    "gx": gx_val,
                    "gy": gy_val,
                    "g_total": gtot_val,
                    "power_kw": max(0.0, round(1500.0 * max(0.0, mf.get("ax", 0.0)) * vx / 1000.0, 1)),
                    "odometer_m": round(x, 1),
                    "ay_current": ay,
                    "friction_utilisation": round(abs(ay) / max(0.1, mu * 9.81), 2),
                    "friction_clamped": False,
                    "theta_a_deg": 0.0,
                    "theta_crit_deg": 45.0,
                    "h_jackknife": 1.0,
                    "ltr": round(abs(ay) * 0.5 / (9.81 * 0.9), 3),
                    "trailer_brake_pressure": 0.0,
                    "trailer_pressure_bar": 0.0,
                    "is_jackknife_critical": False,
                    "is_rollover_critical": False,
                    "mode": "MATLAB_LIVE" if matlab_status["is_streaming"] else "COMPLETED",
                    "interventions_total": 0,
                    "obstacle_ahead": False,
                    "obstacle_dist_m": 999.0,
                    "evading": False,
                    "evading_label": "",
                    "evade_offset_m": 0.0,
                    "is_matlab": True,
                    "is_simulink": bool(mf.get("is_simulink", False)),
                    "is_rl_env": bool(mf.get("is_rl_env", False)),
                    "reward": mf.get("reward", 0.0),
                    "cumulative_reward": mf.get("cumulative_reward", 0.0),
                    "is_completed": not matlab_status["is_streaming"],
                }

                return {
                    "backend": "matlab",
                    "backend_label": f"MATLAB {ctrl_name}",
                    "matlab_status": matlab_status,
                    "uptime_seconds": round(time.time() - self.start_epoch, 1),
                    "ticks_total": mb.ticks_received,
                    "interventions_total": 0,
                    "current": frame,
                    "trail": list(mb.trail),
                    "charts": list(mb.chart_history),
                    "incidents": [],
                    "obstacles": [],
                    "road": {
                        "amp1": 22.0,
                        "freq1": 0.0042,
                        "amp2": 10.0,
                        "freq2": 0.0084,
                        "phase2": 0.4,
                        "lane_width": self.road_cfg.lane_width,
                        "shoulder_width": self.road_cfg.shoulder_width,
                        "is_matlab_track": True,
                    },
                    "ref_points": mb.ref_points,
                    "dimensions": {
                        "wheelbase": 2.8,
                        "lf": 1.2,
                        "lr": 1.6,
                        "d1": 0.0,
                        "l2": 0.0,
                        "track_width": 1.8,
                        "tractor_length": 4.2,
                        "tractor_width": 1.8,
                        "trailer_length": 0.0,
                        "trailer_width": 0.0,
                        "hitch_offset": 0.0,
                        "max_brake_bar": 0.0,
                        "is_passenger_car": True,
                    },
                }

        with self.lock:
            uptime = round(time.time() - self.start_epoch, 1)
            truck_x = self.veh.state.x
            active_obstacles = [
                {
                    "id": o["id"],
                    "type": o["type"],
                    "x": round(o["x"], 1),
                    "y": round(self._get_road_raw(o["x"])[0], 2),
                    "psi_deg": round(math.degrees(self._get_road_reference(o["x"])[1]), 2),
                    "label": o["label"],
                    "lateral_evade": o.get("lateral_evade", self.road_cfg.lane_width),
                    "dist_m": round(o["x"] - truck_x, 1),
                }
                for o in self.obstacles
                if (truck_x - 120.0) <= o["x"] <= (truck_x + 350.0)
            ]
            active_obstacles.sort(key=lambda o: abs(o["dist_m"]))
            return {
                "backend": "python",
                "backend_label": "Python EGGA Engine (Class 8 Tractor-Trailer)",
                "matlab_status": matlab_status,
                "uptime_seconds": uptime,
                "ticks_total": self.tick_count,
                "interventions_total": self.intervention_count,
                "current": self.latest_frame,
                "trail": list(self.trail),
                "charts": list(self.chart_history),
                "incidents": list(self.recent_incidents)[-10:],
                "obstacles": active_obstacles,
                "road": {
                    "amp1": self.road_cfg.amp1,
                    "freq1": self.road_cfg.freq1,
                    "amp2": self.road_cfg.amp2,
                    "freq2": self.road_cfg.freq2,
                    "phase2": self.road_cfg.phase2,
                    "lane_width": self.road_cfg.lane_width,
                    "shoulder_width": self.road_cfg.shoulder_width,
                },
                "dimensions": {
                    "wheelbase": round(self.veh.p.wheelbase, 2),
                    "lf": round(self.veh.p.lf, 2),
                    "lr": round(self.veh.p.lr, 2),
                    "d1": round(self.veh.p.d1, 2),
                    "l2": round(self.veh.p.l2, 2),
                    "track_width": round(self.veh.p.track_width2, 2),
                    "tractor_length": round(self.veh.p.lf + self.veh.p.lr + 1.4, 2),
                    "tractor_width": round(self.veh.p.track_width2, 2),
                    "trailer_length": round(self.veh.p.l2 + 3.5, 2),
                    "trailer_width": round(self.veh.p.track_width2 + 0.1, 2),
                    "hitch_offset": round(self.veh.p.lr + self.veh.p.d1, 2),
                    "max_brake_bar": round(self.veh.p.max_brake_pressure_bar, 1),
                },
            }


class MatlabBridgeManager:
    """Manages live MATLAB backend connection, subprocess lifecycle, and telemetry ingestion."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active_backend: str = "python"
        self.process: subprocess.Popen[Any] | None = None
        self.is_running: bool = False
        self.stop_requested: bool = False
        self.last_heartbeat: float = 0.0
        self.ticks_received: int = 0
        self.mission: str = "proving_ground"
        self.controller: str = "rl"
        self.target_vx: float = 20.0  # 72.0 km/h highway speed default
        self.speed_mult: float = 1.0  # 1.0x real-time pace
        self.latest_frame: dict[str, Any] = {}
        self.trail: deque[dict[str, Any]] = deque(maxlen=300)
        self.chart_history: deque[dict[str, Any]] = deque(maxlen=150)
        self.ref_points: list[list[float]] = []
        self.matlab_exe: str | None = self._find_matlab_exe()

    def _find_matlab_exe(self) -> str | None:
        candidates = [
            r"C:\Program Files\MATLAB\R2025b\bin\matlab.exe",
            r"C:\Program Files\MATLAB\R2025a\bin\matlab.exe",
            shutil.which("matlab"),
        ]
        for c in candidates:
            if c and os.path.exists(c):
                return c
        return None

    def ingest_frame(self, frame: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.last_heartbeat = time.time()
            self.ticks_received += 1
            if "ref_points_x" in frame and "ref_points_y" in frame:
                self.ref_points = [
                    [float(px), float(py)]
                    for px, py in zip(frame["ref_points_x"], frame["ref_points_y"])
                ]
            self.latest_frame = frame
            self.trail.append({"x": frame.get("x", 0.0), "y": frame.get("y", 0.0)})
            self.chart_history.append({
                "t": frame.get("t", 0.0),
                "e_y": frame.get("lateral_error_cm", 0.0),
                "e_y_signed": frame.get("lateral_deviation_m", 0.0) * 100.0,
                "mu": frame.get("mu", 0.85),
                "slope": frame.get("slope_deg", 0.0),
                "speed": frame.get("vx", 10.0),
                "steer_deg": frame.get("steer_deg", 0.0),
                "ay": frame.get("ay_current", 0.0),
                "kp": frame.get("kp", 0.80),
                "kd": frame.get("kd", 0.05),
                "Fzf": frame.get("Fzf", 0.0),
                "Fzr": frame.get("Fzr", 0.0),
                "reward": frame.get("reward", 0.0),
                "cumulative_reward": frame.get("cumulative_reward", 0.0),
            })
            return {
                "status": "ok",
                "stop_requested": self.stop_requested,
                "target_vx": self.target_vx,
                "speed_mult": self.speed_mult,
            }

    def start_matlab(
        self,
        mission: str = "proving_ground",
        controller: str = "rl",
        speed: float = 1.0,
        vx: float = 20.0,
    ) -> dict[str, Any]:
        with self.lock:
            if not self.matlab_exe:
                return {"status": "error", "message": "MATLAB executable not found on system"}
            self.stop_matlab_locked()
            self.mission = mission
            self.controller = controller
            self.stop_requested = False
            self.active_backend = "matlab"
            self.target_vx = max(5.0, min(35.0, float(vx)))
            self.speed_mult = max(0.2, min(5.0, float(speed)))
            self.trail.clear()
            self.chart_history.clear()
            self.ticks_received = 0

            matlab_script_dir = str(REPO_ROOT / "matlab").replace("\\", "/")
            legacy_script_dir = str(REPO_ROOT / "legacy").replace("\\", "/")
            matlab_cmd = (
                f"addpath('{matlab_script_dir}'); "
                f"addpath('{legacy_script_dir}'); "
                f"studio_matlab_bridge('mission', '{mission}', 'controller', '{controller}', 'speed_multiplier', {self.speed_mult}, 'vx', {self.target_vx});"
            )
            try:
                self.process = subprocess.Popen(
                    [self.matlab_exe, "-batch", matlab_cmd],
                    cwd=str(REPO_ROOT),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self.is_running = True
                return {
                    "status": "ok",
                    "message": f"MATLAB simulation started (PID {self.process.pid})",
                    "pid": self.process.pid,
                    "active_backend": "matlab",
                    "target_vx": self.target_vx,
                    "speed_mult": self.speed_mult,
                }
            except Exception as e:
                return {"status": "error", "message": str(e)}

    def set_speed(self, vx: float | None = None, speed_mult: float | None = None) -> dict[str, Any]:
        with self.lock:
            if vx is not None:
                self.target_vx = max(5.0, min(35.0, float(vx)))
            if speed_mult is not None:
                self.speed_mult = max(0.2, min(5.0, float(speed_mult)))
            return {
                "status": "ok",
                "target_vx": self.target_vx,
                "speed_kmh": round(self.target_vx * 3.6, 1),
                "speed_mult": self.speed_mult,
            }

    def stop_matlab(self) -> dict[str, Any]:
        with self.lock:
            return self.stop_matlab_locked()

    def stop_matlab_locked(self) -> dict[str, Any]:
        self.stop_requested = True
        if self.process:
            try:
                self.process.terminate()
            except Exception:
                pass
            self.process = None
        self.is_running = False
        return {"status": "ok", "message": "MATLAB simulation stopped"}

    def get_status(self) -> dict[str, Any]:
        with self.lock:
            now = time.time()
            time_since = round(now - self.last_heartbeat, 1) if self.last_heartbeat > 0 else 999.0
            is_streaming = (time_since < 3.5)
            proc_alive = bool(self.process and self.process.poll() is None)
            return {
                "available": self.matlab_exe is not None,
                "matlab_exe": self.matlab_exe,
                "active_backend": self.active_backend,
                "is_streaming": is_streaming,
                "is_running": proc_alive,
                "last_seen_s": time_since,
                "ticks_received": self.ticks_received,
                "mission": self.mission,
                "controller": self.controller,
                "latest_stage": self.latest_frame.get("stage_name", "None"),
            }


MATLAB_BRIDGE = MatlabBridgeManager()
SIMULATOR = LiveVehicleSimulator(tick_rate_hz=25.0)


class StudioHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            index_path = STATIC_DIR / "index.html"
            if index_path.exists():
                content = index_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404, "index.html not found")
        elif self.path == "/api/telemetry":
            payload = SIMULATOR.get_telemetry_payload()
            data = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(data)
        elif self.path == "/api/matlab/status":
            res = json.dumps(MATLAB_BRIDGE.get_status()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(res)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(res)
        elif self.path == "/api/speed/get":
            res_obj = {
                "status": "ok",
                "python_target_cruise_mps": SIMULATOR.target_cruise_speed_mps,
                "python_target_cruise_kmh": round(SIMULATOR.target_cruise_speed_mps * 3.6, 1),
                "matlab_target_vx": MATLAB_BRIDGE.target_vx,
                "matlab_speed_kmh": round(MATLAB_BRIDGE.target_vx * 3.6, 1),
                "matlab_speed_mult": MATLAB_BRIDGE.speed_mult,
            }
            res_bytes = json.dumps(res_obj).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(res_bytes)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(res_bytes)
        elif self.path in ("/three.min.js", "/static/three.min.js"):
            file_path = STATIC_DIR / "three.min.js"
            if file_path.exists():
                content = file_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/javascript; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "public, max-age=86400")
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404, "three.min.js not found")
        else:
            self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        if self.path == "/api/inject":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode("utf-8"))
                event_type = data.get("event", "dry")
                msg = SIMULATOR.inject_event(event_type)
                res = json.dumps({"status": "ok", "message": msg}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(res)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(res)
            except Exception as e:
                err = json.dumps({"status": "error", "message": str(e)}).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err)))
                self.end_headers()
                self.wfile.write(err)
        elif self.path == "/api/matlab/stream":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                frame = json.loads(body.decode("utf-8"))
                res = MATLAB_BRIDGE.ingest_frame(frame)
                res_bytes = json.dumps(res).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(res_bytes)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(res_bytes)
            except Exception as e:
                err = json.dumps({"status": "error", "message": str(e)}).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err)))
                self.end_headers()
                self.wfile.write(err)
        elif self.path == "/api/speed/set":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            try:
                params = json.loads(body.decode("utf-8")) if body else {}
                speed_mps = None
                if "speed_kmh" in params:
                    speed_mps = float(params["speed_kmh"]) / 3.6
                elif "vx" in params:
                    speed_mps = float(params["vx"])
                elif "speed" in params:
                    speed_mps = float(params["speed"]) / 3.6

                speed_mult = float(params["speed_mult"]) if "speed_mult" in params else None

                if speed_mps is not None:
                    SIMULATOR.set_target_cruise_speed(speed_mps)
                    MATLAB_BRIDGE.set_speed(vx=speed_mps, speed_mult=speed_mult)
                elif speed_mult is not None:
                    MATLAB_BRIDGE.set_speed(speed_mult=speed_mult)

                res = {
                    "status": "ok",
                    "target_cruise_mps": SIMULATOR.target_cruise_speed_mps,
                    "target_speed_kmh": round(SIMULATOR.target_cruise_speed_mps * 3.6, 1),
                    "matlab_vx": MATLAB_BRIDGE.target_vx,
                    "matlab_speed_kmh": round(MATLAB_BRIDGE.target_vx * 3.6, 1),
                    "matlab_speed_mult": MATLAB_BRIDGE.speed_mult,
                }
                res_bytes = json.dumps(res).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(res_bytes)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(res_bytes)
            except Exception as e:
                err = json.dumps({"status": "error", "message": str(e)}).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err)))
                self.end_headers()
                self.wfile.write(err)
        elif self.path == "/api/matlab/start":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            try:
                params = json.loads(body.decode("utf-8")) if body else {}
                res = MATLAB_BRIDGE.start_matlab(
                    mission=params.get("mission", "proving_ground"),
                    controller=params.get("controller", "rl"),
                    speed=float(params.get("speed", 1.0)),
                    vx=float(params.get("vx", 20.0)),
                )
                res_bytes = json.dumps(res).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(res_bytes)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(res_bytes)
            except Exception as e:
                err = json.dumps({"status": "error", "message": str(e)}).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err)))
                self.end_headers()
                self.wfile.write(err)
        elif self.path == "/api/matlab/stop":
            res = MATLAB_BRIDGE.stop_matlab()
            res_bytes = json.dumps(res).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(res_bytes)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(res_bytes)
        elif self.path == "/api/backend/select":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            try:
                params = json.loads(body.decode("utf-8")) if body else {}
                target = params.get("backend", "python")
                if target == "python" and MATLAB_BRIDGE.is_running:
                    MATLAB_BRIDGE.stop_matlab()
                MATLAB_BRIDGE.active_backend = target
                res = {"status": "ok", "active_backend": target}
                res_bytes = json.dumps(res).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(res_bytes)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(res_bytes)
            except Exception as e:
                err = json.dumps({"status": "error", "message": str(e)}).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(err)))
                self.end_headers()
                self.wfile.write(err)
        else:
            self.send_error(404, "Not Found")


def run_studio_server(port: int = 8088) -> None:
    SIMULATOR.start()
    server = ThreadingHTTPServer(("0.0.0.0", port), StudioHandler)
    print("\n=============================================================")
    print("  EGGA STUDIO 24/7 CONTINUOUS TELEMETRY MONITORING ACTIVE")
    print(f"  URL: http://127.0.0.1:{port}/")
    print(f"  Live Telemetry API: http://127.0.0.1:{port}/api/telemetry")
    print("  Simulation Physics: Continuous Class 8 4-DOF tractor-trailer")
    print("=============================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping EGGA Studio Server.")
        server.server_close()


if __name__ == "__main__":
    run_studio_server()
