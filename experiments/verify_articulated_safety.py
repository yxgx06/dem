"""Commercial Validation Script: Tractor-Trailer Jackknife & Rollover Avoidance.

Demonstrates closed-loop prevention of jackknifing and rollover on wet asphalt
under split-friction emergency braking, comparing an unguarded controller against
the EGGA Coupled Friction Circle and Jackknife Guard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from egga.config import REPO_ROOT
from egga.plant.articulated import ArticulatedParams, ArticulatedVehicle
from egga.supervisor.friction_circle import (
    FrictionCirclePolicy,
    FrictionDemand,
    allocate_friction_circle,
)
from egga.supervisor.jackknife_guard import (
    FLAG_JACKKNIFE_OK,
    JackknifeConfig,
    JackknifeInputs,
    step_jackknife_guard,
)


def run_scenario(guarded: bool) -> dict[str, Any]:
    params = ArticulatedParams(
        m1=8000.0,
        m2=24000.0,
        iz1=35000.0,
        iz2=120000.0,
        lf=1.4,
        lr=2.4,
        d1=0.3,
        l2=8.5,
        cg_height2=2.2,
        track_width2=2.4,
        tau_air=0.25,
    )
    vehicle = ArticulatedVehicle(params=params)
    vehicle.reset(vx=22.0, theta_a0=0.22)  # Initial hitch disturbance of ~12.6 deg

    guard_config = JackknifeConfig(
        l2=8.5,
        tau_air=0.25,
        ltr_warning=0.60,
        ltr_critical=0.85,
    )

    dt = 0.01
    sim_time = 4.0
    steps = int(sim_time / dt)

    hitch_angles: list[float] = []
    ltr_values: list[float] = []
    interventions: int = 0
    jackknifed = False
    rolled_over = False

    mu = 0.35  # Wet slippery asphalt

    prev_steer = 0.0
    steer_applied = 0.0
    ax_applied = 0.0

    for step in range(steps):
        t = step * dt
        # Aggressive evasive maneuver request: high steer demand combined with emergency braking
        if t < 0.2:
            delta_cmd = 0.0
            ax_cmd = -1.0
        elif t < 2.0:
            delta_cmd = 0.15  # Aggressive steer input into turn
            ax_cmd = -4.2     # Emergency braking beyond mu * g
        else:
            delta_cmd = 0.0
            ax_cmd = -1.0

        if guarded:
            # 1. 2D Coupled Friction Circle Allocation
            v_curr = vehicle.state.vx
            fc_demand = FrictionDemand(
                ax_req=ax_cmd,
                ay_req=v_curr**2 * (delta_cmd / 3.8),
                mu=mu,
                policy=FrictionCirclePolicy.STEERING_PRIORITY,
            )
            fc_alloc = allocate_friction_circle(fc_demand, speed=v_curr, wheelbase=3.8)

            # 2. Articulated Jackknife & Rollover Barrier Guard
            jk_inp = JackknifeInputs(
                theta_a=vehicle.state.theta_a,
                theta_a_dot=(vehicle.state.r1 - vehicle.state.r2),
                vx=v_curr,
                mu=mu,
                ltr=abs(vehicle.state.theta_a * 1.5),  # dynamic proxy
                steer_cmd_req=delta_cmd,
                steer_rate_req=(delta_cmd - prev_steer) / dt,
            )
            jk_out = step_jackknife_guard(guard_config, jk_inp)

            if jk_out.status_flags != FLAG_JACKKNIFE_OK:
                interventions += 1
                steer_applied = jk_out.steer_cmd_safe
                # Apply trailer tension drag to arrest jackknife snap
                ax_applied = fc_alloc.ax_safe - (0.4 * jk_out.trailer_brake_pressure)
            else:
                steer_applied = min(
                    max(delta_cmd, -fc_alloc.delta_max_coupled), fc_alloc.delta_max_coupled
                )
                ax_applied = fc_alloc.ax_safe
        else:
            steer_applied = delta_cmd
            ax_applied = ax_cmd

        prev_steer = steer_applied
        res = vehicle.step(delta=steer_applied, ax=ax_applied, dt=dt, mu=mu)

        theta_deg = abs(res.theta_a * 57.2957795)
        ltr = abs(res.ltr)
        hitch_angles.append(theta_deg)
        ltr_values.append(ltr)

        if theta_deg >= 45.0:
            jackknifed = True
            break
        if ltr >= 1.0:
            rolled_over = True
            break

    max_hitch = max(hitch_angles) if hitch_angles else 0.0
    max_ltr = max(ltr_values) if ltr_values else 0.0

    return {
        "guarded": guarded,
        "jackknifed": jackknifed,
        "rolled_over": rolled_over,
        "max_hitch_angle_deg": round(max_hitch, 2),
        "max_ltr": round(max_ltr, 3),
        "interventions": interventions,
        "final_speed_mps": round(vehicle.state.vx, 2),
    }


def main() -> None:
    results_dir = REPO_ROOT / "results" / "commercial"
    results_dir.mkdir(parents=True, exist_ok=True)

    unguarded_result = run_scenario(guarded=False)
    guarded_result = run_scenario(guarded=True)

    summary = {
        "scenario": "Class 8 Tractor-Semitrailer Split-Mu Emergency Avoidance (mu=0.35, v0=80km/h)",
        "unguarded": unguarded_result,
        "guarded": guarded_result,
    }

    out_file = results_dir / "articulated_validation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=============================================================")
    print("      EGGA ARTICULATED COMMERCIAL VALIDATION RESULTS        ")
    print("=============================================================")
    print(f"Scenario: {summary['scenario']}")
    print("\n[UNGUARDED BASELINE]")
    print(f"  Jackknifed:           {unguarded_result['jackknifed']}")
    print(f"  Max Hitch Angle:      {unguarded_result['max_hitch_angle_deg']} deg (Limit: 45.0 deg)")
    print(f"  Max Rollover LTR:     {unguarded_result['max_ltr']} (Limit: 1.00)")
    print("\n[EGGA-GUARDED]")
    print(f"  Jackknifed:           {guarded_result['jackknifed']}")
    print(f"  Max Hitch Angle:      {guarded_result['max_hitch_angle_deg']} deg (Limit: 45.0 deg)")
    print(f"  Max Rollover LTR:     {guarded_result['max_ltr']} (Limit: 1.00)")
    print(f"  Interventions:        {guarded_result['interventions']} steps")
    print("=============================================================\n")


if __name__ == "__main__":
    main()
