# EGGA: Envelope-Guarded Gain Adaptation (research prototype)

Goal: let a learned steering-gain scheduler run safely when tyre-road friction and steering delay
are known only as intervals, by keeping it inside a pre-verified envelope. The standing rules are
in [CLAUDE.md](CLAUDE.md).

## Status

- **Phases 1–10 Complete**: Full engineering lifecycle established, from mathematical proofs and robust estimation to ANSI C99 embedded codegen, ISO 26262/21448 safety cases, and automated CI reproduction. See the [Final Engineering Synthesis Report](docs/FINAL_SYNTHESIS_REPORT.md).
- **Commercial Expansion (Heavy Trucking)**: Class 8 articulated tractor-trailer dynamics with coupled 2D friction circle allocation (Kamm's circle) and anti-jackknife/anti-rollover guard. See the [Commercial Validation Report](docs/commercial/commercial_validation_report.md).
- **EGGA Live Telemetry & 3D Studio**: Real-time interactive Three.js 3D simulation, cockpit/chase/drone camera views, dynamic hazard injection, and MATLAB telemetry streaming. See [Studio Guide](tooling/studio/README.md).

## What exists in this repository

- **Core Supervisor & Safe RL**: Pre-verified Lyapunov viability envelope (`src/egga/supervisor/`), dual-rate robust EKF estimation (`src/egga/estimation/`), 92-parameter safe PPO actor (`src/egga/policy/`), and C99 runtime (`src/c/`).
- **Commercial Vehicle Plant**: 4-DOF articulated tractor-trailer plant (`src/egga/plant/articulated.py`), coupled friction circle governor, and pneumatic brake latency models.
- **Tooling & Integration**:
  - **EGGA Studio**: 3D interactive telemetry monitoring suite (`tooling/studio/`).
  - **MATLAB Bridge**: Real-time simulation streaming bridge (`matlab/studio_matlab_bridge.m`) and R2025b cross-checks (`matlab/crosscheck.m`, `legacy/BicyclePathTracking1_R2025b.slx`).
  - **Simulink S-Function**: ANSI C99 MEX S-Function wrapper (`tooling/simulink/`).
  - **ROS 2 Node**: ASIL D runtime node package (`tooling/ros2/`).
- **Reports & Safety Evidence**:
  - Synthesis & Milestones: [docs/FINAL_SYNTHESIS_REPORT.md](docs/FINAL_SYNTHESIS_REPORT.md), [docs/PROGRESS.md](docs/PROGRESS.md)
  - Envelope Design & Assumptions: [docs/ENVELOPE_DESIGN.md](docs/ENVELOPE_DESIGN.md), [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md)
  - Commercial Heavy Vehicle Safety: [docs/commercial/commercial_validation_report.md](docs/commercial/commercial_validation_report.md)
  - Hazard Analysis & Risk Assessment: [docs/safety/HARA.md](docs/safety/HARA.md)

## Run

```bash
# Run pytest verification test suite
uv run pytest -q

# Launch EGGA Live Telemetry & 3D Studio
python tooling/studio/server.py --port 8088

# Stream live MATLAB simulation into Studio
matlab -batch "studio_matlab_bridge('mission', 'proving_ground', 'controller', 'rl')"
```

