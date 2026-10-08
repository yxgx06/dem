from __future__ import annotations

import json
import math
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from egga.config import REPO_ROOT
from egga.plant.articulated import ArticulatedParams, ArticulatedVehicle
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


def generate_simulation_frames() -> list[dict[str, Any]]:
    """Simulates a high-speed highway scenario with an evasive swerve and split-mu patch

    demonstrating friction circle allocation, jackknife prevention, and supervisor transitions.
    """
    veh = ArticulatedVehicle(ArticulatedParams(cg_height2=2.4))
    veh.reset(vx=20.0, theta_a0=0.0)
    jk_cfg = JackknifeConfig()

    frames: list[dict[str, Any]] = []
    dt = 0.02
    t = 0.0

    mode_names = ["NOMINAL", "CAUTIOUS", "LOW_MU", "DEGRADED", "FALLBACK", "MINIMAL_RISK"]

    incidents: list[dict[str, Any]] = []

    for step_i in range(350):
        t = round(step_i * dt, 3)

        # Environmental friction profile: sudden slick patch between t=2.0s and t=4.5s
        if 2.0 <= t <= 4.5:
            mu = 0.30
        else:
            mu = 0.85

        # Steering request: emergency avoidance double-lane-change maneuver
        if 1.5 <= t < 2.5:
            steer_req = 0.08
            ax_req = -3.5  # Heavy emergency braking
        elif 2.5 <= t < 3.5:
            steer_req = -0.08
            ax_req = -2.0
        elif 3.5 <= t < 4.5:
            steer_req = 0.03
            ax_req = 0.0
        else:
            steer_req = 0.0
            ax_req = 0.5  # Re-accelerate

        # Step plant
        res = veh.step(delta=steer_req, ax=ax_req, dt=dt, mu=mu)

        # 1. Friction circle allocation
        v_curr = res.state.vx
        ay_curr = res.ay1
        fc_demand = FrictionDemand(
            ax_req=ax_req,
            ay_req=ay_curr,
            mu=mu,
            policy=FrictionCirclePolicy.STEERING_PRIORITY,
        )
        fc_alloc = allocate_friction_circle(fc_demand, speed=v_curr, wheelbase=3.8)

        # 2. Articulated Jackknife Guard
        jk_inp = JackknifeInputs(
            theta_a=res.theta_a,
            theta_a_dot=res.theta_a_dot,
            vx=v_curr,
            mu=mu,
            ltr=res.ltr,
            steer_cmd_req=steer_req,
            steer_rate_req=(steer_req / dt),
        )
        jk_out = step_jackknife_guard(jk_cfg, jk_inp)

        # Determine supervisor mode
        if jk_out.is_rollover_critical:
            mode_idx = 5  # MINIMAL_RISK
        elif jk_out.is_jackknife_critical:
            mode_idx = 3  # DEGRADED
        elif mu < 0.4:
            mode_idx = 2  # LOW_MU
        elif fc_alloc.is_clamped:
            mode_idx = 1  # CAUTIOUS
        else:
            mode_idx = 0  # NOMINAL

        frame = {
            "t": t,
            "x": round(res.state.x, 2),
            "y": round(res.state.y, 2),
            "psi1": round(res.state.psi1, 3),
            "vx": round(v_curr, 2),
            "mu": round(mu, 2),
            "ax_req": round(ax_req, 2),
            "ax_safe": round(fc_alloc.ax_safe, 2),
            "ay_safe": round(fc_alloc.ay_safe, 2),
            "friction_utilisation": round(fc_alloc.utilisation, 3),
            "friction_clamped": fc_alloc.is_clamped,
            "theta_a_deg": round(math.degrees(res.theta_a), 2),
            "theta_crit_deg": round(math.degrees(jk_out.theta_crit), 2),
            "ltr": round(res.ltr, 3),
            "trailer_brake_pressure": round(jk_out.trailer_brake_pressure, 2),
            "is_jackknife_critical": jk_out.is_jackknife_critical,
            "is_rollover_critical": jk_out.is_rollover_critical,
            "mode": mode_names[mode_idx],
            "steer_cmd_safe": round(jk_out.steer_cmd_safe, 3),
        }
        frames.append(frame)

        # Log SOTIF incidents
        if fc_alloc.is_clamped or jk_out.is_jackknife_critical or jk_out.is_rollover_critical:
            incidents.append(
                {
                    "t": t,
                    "trigger": "Low Friction + Evasive Maneuver"
                    if mu < 0.4
                    else "High Dynamic Demand",
                    "mode": mode_names[mode_idx],
                    "fc_clamped": fc_alloc.is_clamped,
                    "jk_crit": jk_out.is_jackknife_critical,
                    "trailer_brake": round(jk_out.trailer_brake_pressure, 2),
                }
            )

    # Persist flight recorder log
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(FLIGHT_RECORDER_PATH, "w", encoding="utf-8") as f:
        for inc in incidents:
            f.write(json.dumps(inc) + "\n")

    return frames


class StudioHandler(BaseHTTPRequestHandler):
    cached_frames: list[dict[str, Any]] = []

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
            if not StudioHandler.cached_frames:
                StudioHandler.cached_frames = generate_simulation_frames()
            data = json.dumps(StudioHandler.cached_frames).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404, "Not Found")


def run_studio_server(port: int = 8088) -> None:
    server = HTTPServer(("127.0.0.1", port), StudioHandler)
    print(f"\n>>> EGGA Studio Telemetry Server running at: http://127.0.0.1:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping EGGA Studio Server.")
        server.server_close()


if __name__ == "__main__":
    generate_simulation_frames()
    run_studio_server()
