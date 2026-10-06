# Decisions

## D1. Use the MATLAB script, not lib3.py, as the numerical reference (Phase 0)
`legacy/lib3.py` integrates heading with the updated yaw rate, while
`legacy/mission_proving_ground_rl.m` uses the old one. The egga port follows MATLAB. Its nominal
max/RMS error matches a headless MATLAB R2025b run (see the cross-check table in
`docs/phase0_discrepancies.md`; enforced by `tests/test_mission_and_controllers.py`).
Rejected: refactoring lib3.py in place (global state, deviates from MATLAB).

## D2. Delay mapping differs from lib3.py by one step
`lib3.py` delays the command by (d+1) steps for a buffer argument d. egga delays by exactly
round(delay_s / dt) steps. Verified with `tools/check_lib3_delay_offset.py`: lib3 patched to
MATLAB's heading order and run with N steps reproduces egga at (N+1) x 10 ms. The audit's "100 ms"
case is therefore 110 ms in lib3 terms. Three cells in the discrepancy table differ for this
reason; the qualitative ordering (RL fails first, everything fails at 120 ms) is unchanged.
No threshold was loosened to hide this.

## D3. Divergence criterion
A run is DIVERGED when |e_y| exceeds 1.5 m or is non-finite (`configs/stress.yaml`), matching
`legacy/t_lqr.py`. The run stops at that point and later samples are NaN.

## D4. RL slew limit applied once
MATLAB's RL controller applies a slew limit inside the controller and the mission applies it again.
The egga port applies the plant-side actuator limit only (controllers output a raw command). The
nominal MATLAB golden comparison passes, so the second limit does not change the nominal result.
Not verified for delay or noise cases (no MATLAB reference for those).

## D5. Python version
The system Python is 3.10; the project requires >= 3.11, so uv provides 3.12.
