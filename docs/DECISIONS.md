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

## D6. Phase 1 plant is a separate implementation with an exact-equivalence mode
`egga.plant.vehicle` shares no code with `egga.plant.bicycle` (Phase 0). With the `linear_equiv`
case (linear_clip tyre, no load transfer, ideal actuator and sensors, finite-difference
derivative) closed-loop results are bit-identical to Phase 0 (`tests/test_closed_loop.py`,
`docs/phase1_report.md`). Rejected: a tolerance-based equivalence, since exact equality is stricter.

## D7. Tyre and transfer modelling choices
Pacejka cornering stiffness is `k_c * Fz` per wheel (independent of mu), peak is
`mu * Fz * (1 - load_sensitivity * (Fz/Fz_nom - 1))`. Load sensitivity (default 0.10) is what makes
lateral load transfer reduce total axle force at saturation; with it set to 0 transfer has no
effect on the peak. Both values are assumptions, not fitted to a real tyre. Lateral transfer uses
the previous step's ay (quasi-static). Split mu scales wheel friction but adds no yaw-moment
disturbance beyond the force difference, so it is a coarse known-failure probe only.

## D8. Step-response tolerance is integrator error, shown by convergence
Euler at 100 Hz gives about 2% peak transient error against the exact linear step response. The
test requires < 3% at dt = 10 ms, error to shrink >= 1.7x at dt = 5 ms, and final value within
1e-3. This is not a loosened threshold hiding a model error; the steady state matches to ~1e-6.

## D9. Derivative filter alone does not make the noise test architecture-only
`docs/phase1_report.md` shows pd_ff and rl_handtyped still diverge under 2 cm white lateral-error
noise for derivative cutoffs from 10 Hz down to 0.5 Hz (PID survives at <= 2 Hz). The proportional
path through the 0.6 rad/s rate limit is also noise-sensitive. Default cutoff stays 5 Hz (an
unfitted assumption). Noise-robust state estimation is therefore deferred to Phase 3 and the
noise result is NOT claimed as filter-fixed.

## D10. Scenario sets: slice-disjoint, feasible-by-construction, frozen and hashed
`configs/scenario_sets.yaml` cuts every parameter range into 12 slices and gives each set its own
slice indices (train 8, val 2, test 2), so no two sets share a value of any parameter. Road
geometry is derived from (speed, mu_min, demand_ratio); seeds use separate blocks. Sets are written
to `experiments/scenario_sets/` with SHA-256 hashes; loaders verify the hash and the test set
requires `unlock_test=True` (a test forbids that string anywhere in `src/` or `tools/` except
`sets.py` until Phase 7). History: a first version had demand up to 1.1 and a 10-slice split, which
left train with no near-limit demand and made half the test set infeasible; a second version with
demand up to 1.0 made 50-75% of runs diverge for every controller above ratio 0.75 (Pacejka peak is
about 0.9 mu g and belief errors lower the friction-aware limit). The demand range was cut to
[0.3, 0.85] and the sets were re-frozen BEFORE any test-set use. The test set has never been run.

## D11. Baselines are tuned on TRAIN only, by a coordinate search with a fixed budget
`python -m egga.eval.tuning` loads only `load_scenarios("train")` (a test enforces this), uses
every 4th train scenario (12 per evaluation) and a log-scale coordinate search (<= 40 evaluations
per controller). Tuned values go to `configs/baselines_tuned.yaml`, which records the train-set
hash in its header (also tested). B2 (the original hand-typed RL with oracle mu) is deliberately
untuned. The objective is rms + 0.25 max (cm) + 100 per divergence.

## D12. LQR/MPC steering limit uses the friction BELIEF
Both clip steering to the angle for a_y,max = 0.85 * mu_belief * g. When belief under-estimates
friction, the limit falls below what the road demands and the controller diverges even though it
would track with the true value. This is a measured weakness of friction-aware baselines under
belief error (it is the situation EGGA targets), not a bug; `ay_limit_fraction` is not tuned.

## D13. B7 LQR variants and the audit reproduction
Three variants share one tuned Q/R: true-mass design, nominal-mass design, delay-augmented
(design delay 0.10 s, mass known). The delay buffer stores the feedback part only (buffering the
total steering fed the feedforward back through the buffer gains and shrank it). The audit's LQR
numbers reproduce on the Phase 0 simulator with its original Q/R (0.93 vs 0.87; 2.95 vs 3.0;
8.78 vs 8.8; survives 200 ms). The delay-augmented variant is designed for 0.10 s but is NOT better
than plain LQR on the Phase 1 scenarios (higher error even at matching delay); the linear design is
verified (stable, lower cost under true delay in a clean linear simulation, tested), so the loss
comes from nonlinear effects (rate limit, mismatch) and is reported as measured.

## D14. MPC sample time and preview
B6 runs the QP every 0.05 s (Np=15, Nc=5, horizon 0.75 s) and holds in between; the legacy MATLAB
MPC used 0.01 s, a 0.15 s horizon. It receives a preview of the reference yaw rate (planner output)
and uses the design mass; delay, lag and tyre nonlinearity are unmodelled. Solve times are HOST
Python+OSQP wall time, not ECU numbers.
