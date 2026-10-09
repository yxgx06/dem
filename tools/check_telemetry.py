from __future__ import annotations

import json
import urllib.request


def check() -> None:
    req = urllib.request.urlopen("http://127.0.0.1:8088/api/telemetry")
    data = json.loads(req.read().decode("utf-8"))
    c = data["current"]

    print("=============================================================")
    print("           EGGA STUDIO TELEMETRY HEALTH AUDIT               ")
    print("=============================================================")
    print(f"  Server Status:        HTTP {req.status} OK (Port 8088)")
    print(f"  Uptime:               {data['uptime_seconds']} seconds")
    print(f"  Total Ticks:          {data['ticks_total']:,}")
    print(f"  Vehicle Position:     X = {c['x']:.1f} m, Y = {c['y']:.1f} m")
    print(f"  Road Centerline Y:    {c['y_ref']:.1f} m")
    print(f"  Forward Speed:        {c['vx']:.1f} m/s ({c['vx']*3.6:.1f} km/h)")
    print(f"  Road Friction:        mu = {c['mu']:.2f} ({c['weather'].upper()})")
    print(f"  Supervisor Mode:      {c['mode']}")
    kamm_str = f"{c['friction_utilisation']*100:.1f}% (Clamped: {c['friction_clamped']})"
    print(f"  Kamm Utilization:     {kamm_str}")
    print(f"  Trailer Rollover LTR: {c['ltr']:.3f} (Limit: 1.000)")
    hitch_str = f"{c['theta_a_deg']:.2f} deg (Critical: {c['theta_crit_deg']:.2f} deg)"
    print(f"  Hitch Articulation:   {hitch_str}")
    print(f"  Trailer Drag Tension: {c['trailer_brake_pressure']:.2f}")
    print(f"  Trail Buffer Size:    {len(data['trail'])} points")
    print(f"  Chart History Size:   {len(data['charts'])} points")
    print(f"  Total Interventions:  {data['interventions_total']}")
    print("=============================================================")


if __name__ == "__main__":
    check()
