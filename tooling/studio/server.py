from __future__ import annotations

from collections import deque
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

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

        # Physical Plant and Safety Guard
        self.veh = ArticulatedVehicle(ArticulatedParams(cg_height2=2.4))
        # Initialize vehicle centered precisely on highway centerline
        y0, psi0, _ = self._get_road_reference(0.0)
        self.veh.state = ArticulatedState(x=0.0, y=y0, psi1=psi0, vx=22.0, theta_a=0.0)
        self.jk_cfg = JackknifeConfig()

        # Telemetry History Ring Buffers
        self.trail: deque[dict[str, Any]] = deque(maxlen=200)
        self.chart_history: deque[dict[str, Any]] = deque(maxlen=150)
        self.recent_incidents: deque[dict[str, Any]] = deque(maxlen=30)
        self.latest_frame: dict[str, Any] = {}

        # Environmental & Disturbance State
        self.weather_mode = "dry"
        self.forced_mu: float | None = None
        self.disturbance_steer = 0.0
        self.disturbance_ax = 0.0
        self.disturbance_expiry = 0.0

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
                msg = "Ice patch injected: friction reduced to mu=0.25"
            elif event_type == "rain":
                self.weather_mode = "rain"
                self.forced_mu = 0.42
                msg = "Rain storm injected: friction reduced to mu=0.42"
            elif event_type == "dry":
                self.weather_mode = "dry"
                self.forced_mu = 0.85
                msg = "Environment restored to dry asphalt (mu=0.85)"
            elif event_type == "swerve":
                self.disturbance_steer = 0.08
                self.disturbance_ax = -3.2
                self.disturbance_expiry = now + 1.0
                msg = "Obstacle emergency swerve triggered (-3.2 m/s^2 brake, 0.08 rad steer)"
            elif event_type == "gust":
                self.disturbance_steer = -0.05
                self.disturbance_expiry = now + 0.8
                msg = "Crosswind gust hit trailer: dynamic lateral oscillation injected"
            else:
                msg = f"Unknown event type: {event_type}"
        return msg

    @staticmethod
    def _get_road_reference(x: float) -> tuple[float, float, float]:
        """Calculates road centerline lateral coordinate, heading angle, and curvature."""
        # Realistic highway curvature with gentle high-speed interstate radii (R > 400m)
        y_ref = 12.0 * math.sin(0.003 * x) + 5.0 * math.cos(0.006 * x)
        dy_dx = 12.0 * 0.003 * math.cos(0.003 * x) - 5.0 * 0.006 * math.sin(0.006 * x)
        d2y_dx2 = -12.0 * (0.003**2) * math.sin(0.003 * x) - 5.0 * (0.006**2) * math.cos(0.006 * x)

        psi_ref = math.atan(dy_dx)
        denom = (1.0 + dy_dx**2) ** 1.5
        curvature = d2y_dx2 / denom if denom > 1e-6 else 0.0
        return y_ref, psi_ref, curvature

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

                # Determine dynamic environmental friction
                if self.forced_mu is not None:
                    mu = self.forced_mu
                else:
                    cycle = (t % 60.0)
                    if 25.0 <= cycle < 35.0:
                        mu = 0.38
                        self.weather_mode = "rain"
                    elif 45.0 <= cycle < 52.0:
                        mu = 0.28
                        self.weather_mode = "ice"
                    else:
                        mu = 0.85
                        self.weather_mode = "dry"

                # Check active manual disturbances
                if t < self.disturbance_expiry:
                    steer_dist = self.disturbance_steer
                    ax_dist = self.disturbance_ax
                else:
                    steer_dist = 0.0
                    ax_dist = 0.0

                s = self.veh.state
                vx = max(s.vx, 8.0)
                wheelbase = 3.8

                # Measure tracking error at the tractor front axle (Stanley formulation)
                lf = 1.4
                xf = s.x + lf * math.cos(s.psi1)
                yf = s.y + lf * math.sin(s.psi1)

                yr, psi_r, kappa_r = self._get_road_reference(xf)
                ey = yr - yf
                epsi = math.atan2(math.sin(psi_r - s.psi1), math.cos(psi_r - s.psi1))

                # Stanley path tracking: feedforward curvature + cross-track + heading alignment
                delta_ff = math.atan(wheelbase * kappa_r)
                delta_fb = epsi + math.atan2(1.2 * ey, vx)
                steer_req = delta_ff + 0.8 * delta_fb + steer_dist

                # Longitudinal cruise control
                target_speed = 22.0 if mu > 0.6 else 16.0
                ax_nominal = 0.6 * (target_speed - vx)
                ax_req = min(max(ax_nominal + ax_dist, -4.5), 1.5)

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
                jk_inp = JackknifeInputs(
                    theta_a=s.theta_a,
                    theta_a_dot=(s.r1 - s.r2),
                    vx=vx,
                    mu=mu,
                    ltr=abs(s.theta_a * 1.4),
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

                # Determine Supervisor Operating Mode
                if jk_out.is_rollover_critical or jk_out.is_jackknife_critical:
                    mode_idx = 3  # DEGRADED
                elif mu < 0.40 or fc_alloc.is_clamped:
                    mode_idx = 2  # LOW_MU
                elif abs(res.theta_a) > 0.15 or res.ltr > 0.50:
                    mode_idx = 1  # CAUTIOUS
                else:
                    mode_idx = 0  # NOMINAL

                current_mode = mode_names[mode_idx]

                y_center_actual, _, _ = self._get_road_reference(res.state.x)
                lateral_deviation = res.state.y - y_center_actual

                frame = {
                    "t": t,
                    "tick": self.tick_count,
                    "x": round(res.state.x, 2),
                    "y": round(res.state.y, 2),
                    "y_ref": round(y_center_actual, 2),
                    "lateral_error_cm": round(abs(lateral_deviation) * 100, 1),
                    "psi1": round(res.state.psi1, 3),
                    "psi2": round(res.state.psi1 - res.state.theta_a, 3),
                    "vx": round(res.state.vx, 2),
                    "mu": round(mu, 2),
                    "weather": self.weather_mode,
                    "steer_req": round(steer_req, 3),
                    "steer_safe": round(steer_safe, 3),
                    "ax_req": round(ax_req, 2),
                    "ax_safe": round(ax_safe, 2),
                    "ay_current": round(res.ay1, 2),
                    "friction_utilisation": round(fc_alloc.utilisation, 3),
                    "friction_clamped": fc_alloc.is_clamped,
                    "theta_a_deg": round(math.degrees(res.theta_a), 2),
                    "theta_crit_deg": round(math.degrees(jk_out.theta_crit), 2),
                    "h_jackknife": round(jk_out.h_jackknife, 3),
                    "ltr": round(res.ltr, 3),
                    "trailer_brake_pressure": round(jk_out.trailer_brake_pressure, 2),
                    "is_jackknife_critical": jk_out.is_jackknife_critical,
                    "is_rollover_critical": jk_out.is_rollover_critical,
                    "mode": current_mode,
                    "interventions_total": self.intervention_count,
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
        with self.lock:
            uptime = round(time.time() - self.start_epoch, 1)
            return {
                "uptime_seconds": uptime,
                "ticks_total": self.tick_count,
                "interventions_total": self.intervention_count,
                "current": self.latest_frame,
                "trail": list(self.trail),
                "charts": list(self.chart_history),
                "incidents": list(self.recent_incidents)[-10:],
            }


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
        else:
            self.send_error(404, "Not Found")


def run_studio_server(port: int = 8088) -> None:
    SIMULATOR.start()
    server = HTTPServer(("0.0.0.0", port), StudioHandler)
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
