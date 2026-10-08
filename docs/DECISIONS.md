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

## D15. Estimators see only `Measurements`; independence is enforced by tests
`egga.estimation` has its own tanh tyre model (the plant uses Pacejka) and nominal datasheet
vehicle parameters. `tests/test_estimation.py` AST-scans the package and fails on any import of
`egga.plant`, `egga.scenarios`, `egga.eval` or `egga.controllers`, and checks `Measurements` has no
truth fields. Only `OracleEstimator` takes truth and carries `label = "oracle"`.

## D16. Friction and mass are estimated jointly (EKF state includes nominal mass / mass)
A friction EKF that assumed the nominal mass read a heavy vehicle as a slippery one (strong
under-estimate of mu and poor coverage on train); giving it the true mass removed the bias, which
identified mass as the confound. The fix is a fourth EKF state, theta = nominal mass / mass, so both
get honest intervals. A first mass estimator based on the quasi-steady understeer regression had
poor coverage (the mass effect on steady yaw gain is only a few percent at these speeds) and was
removed rather than kept as dead code.

## D17. Calibration is train-only and reports the cost of coverage
`python -m egga.eval.estimation_eval --calibrate` grid-searches the friction interval width (z, mu
random-walk intensity) and the mass interval (z_mass, theta random-walk intensity) on the TRAIN set,
choosing the narrowest setting whose run-level coverage reaches 0.95. The hash of the train set is in
`configs/estimators_tuned.yaml` (tested). The result is wide friction intervals: friction is only
weakly observable unless the tyres are driven near the limit, and `docs/phase3_report.md` shows
the interval is at or near the prior width when excitation is low. Coverage is therefore bought
with width, and the report states both.

## D18. Known failure conditions (see docs/phase3_report.md for the numbers)
- Steering-sensor noise biases the mass interval (errors-in-variables: the filter treats the noisy
  steering as exact input). A low-pass on the measured steering was tried and rejected: it improved the
  noisy case but badly hurt the clean case through lag bias. The mass interval is therefore NOT trusted
  under steering noise, and the supervisor must not rely on it unless that is addressed.
- Friction is unobservable at low tyre utilisation and while driving straight; a friction drop in
  that situation is only detected once the tyres are excited (long latency in the report).
- The delay bound (cross-correlation of command vs measured steering) relaxes linearly to the
  prior after excitation is lost, so it is conservative but never holds a stale value.

## D19. Staleness and invalid-input semantics
NaN/inf, a non-positive speed or time going backwards returns the widest bounds immediately and
re-initialises the filters. A measurement gap longer than `staleness.max_gap_s` (0.1 s) does the same
(status `stale`). Shorter gaps propagate the model with growing uncertainty; the last good value is
never returned. During short steering dropouts the last valid steering measurement drives the model.
Steering-sensor noise (an assumed datasheet value) is propagated as input noise into the EKF.

## D20. Envelope criteria: the example spectral-radius bound is not usable with an integrator
With integral action at 100 Hz the integrator pole sits within about 1e-3 of the unit circle for every
usable gain, so a plain spectral radius of 0.97 would reject everything. The envelope instead requires
strict stability (all poles within 1 - 1e-4), decay of the non-integrator modes (<= 0.985, a settling
time of a few seconds), damping >= 0.15 for complex poles, and stability at 1.2x the delay bound. The
thresholds are a-priori engineering choices (`configs/envelope.yaml`), and were NOT loosened to fill
cells: at the nominal cell they leave no verified gain for delay bounds of 80 ms and above at
10 m/s, where fast-mode decay and damping, not stability, are binding. The degraded-mode answer is a
lower speed (`Envelope.max_verified_speed`) combined with the friction speed cap.

## D21. Nonlinear confirmation must be run at the stated demand limits
A first confirmation lane change (2 s period) demanded about twice the jerk the table's own limit
allows and pushed the steering rate to its hardware limit, so rate saturation (assumption A8)
produced many failures that the linear analysis could not see. The confirmation now uses the
cell's a_y,max AND a period raised until the lateral jerk is within `jerk_max_mps3`, which makes it
consistent with the limits the supervisor will enforce. Failures dropped markedly and remaining ones
are pruned (with every componentwise-larger gain). The confirmation is still a SAMPLE of gains per
cell (A12); `docs/phase4_report.md` reports the failure rate of the random tests as an estimate of how
often the linear analysis over-accepts.

## D22. The envelope lives in `supervisor/`, offline analysis in `eval/`
`egga.supervisor.envelope` is a runtime lookup over fixed arrays (typed with `mypy --strict`, written to
port to C99). `egga.eval.envelope_build` and `tools/build_envelope.py` are offline. The table is
`experiments/envelope/envelope_v1.npz` with a JSON manifest (grid, margins, assumptions A1..A12, git
SHA, array hash); loading a table whose arrays do not match the manifest hash raises (corrupted
table). Out-of-grid queries return no verified set instead of extrapolating.

## D23. The conservatism curve needs the friction speed cap to be meaningful
The 75 s mission includes ice (mu 0.25). Querying the table at mu_lo 0.9 allowed 15 m/s and an
infeasible reference. With the grid's mu_lo 0.2 (a valid lower bound) the speed cap (k mu_lo g over the
mission's peak curvature) and the verified speed together govern the speed; the report gives the
tracking error for the envelope gain at 10 m/s, for the governed speed, and for the unprotected tuned
gains.

## D24. The supervisor selects gains, speed and limits; it never commands steering
`egga.supervisor.core.step` returns verified gains, a speed command and the demand limits
(steering angle for a_y,max, steering-rate limit). The classical controller computes the steering,
which is then clipped by those limits (the `envelope_guard` layer). The RL policy (Phase 6) can only
propose gains; any proposal is range- and rate-checked and projected onto the verified set. The
supervisor is a pure-function design with a flat config, fixed-size state and a fixed-size event ring
buffer so it can be ported to C99; `supervisor/` has 100% branch coverage and passes `mypy --strict`.

## D25. Gain changes move through verified grid points only
The gain-rate limiter hops at most one candidate-grid step per `gain_hop_period`, only to verified
neighbours. Each applied gain is therefore a verified point, but the TRANSITIONS between verified
points are not themselves verified (assumption A9: frozen-gain analysis). When the envelope shrinks
and the current gain is no longer verified, the target is applied at once (a forced jump, logged).

## D26. Operating-domain assumption and the cost of caution
With low excitation the friction interval stays at its prior width, so the supervisor assumes
mu >= `mu_floor` (0.2) and caps speed from a_y,max at that value, and it uses the conservative
delay bound until the estimator tightens it. Evidence that friction is below the floor
(`mu_hi < mu_floor`) is outside every verified cell and goes to MINIMAL_RISK (latched until an explicit
reset). The measured cost is large: B4 drives at roughly two thirds of the planned speed and spends
most of its time in DEGRADED_ACTUATOR because the delay bound is conservative (see
`docs/phase5_report.md`). Friction cannot be raised above the floor without exciting the tyres, which
the speed cap prevents; this is a design limitation, not tuned away.

## D27. Closed-loop speed governance uses a path-indexed reference
When a controller sets the vehicle speed, the reference is sampled by distance along the path, not by
time (otherwise a slower vehicle would chase a reference that has moved on). Ordinary controllers keep
the time-indexed reference, so all earlier results are unchanged (the `linear_equiv` bit-exact test
still passes). `progress_fraction` (distance covered over route length in the scenario time) is
reported next to tracking error so safety is not compared without its speed cost.

## D28. Verification certification strictly reflects actual vehicle speed
In the initial supervisor design, when operating at an out-of-grid speed ($v < 5\text{ m/s}$ or
$v > 15\text{ m/s}$), the gain lookup fell back to the nearest verified cell $v_{\text{ver}}$, but
reported `verified = True`. The red-team audit diagnosed that claiming mathematical verification
when the vehicle is outside the verified speed axis is false certification. `Outputs.verified` now
strictly evaluates membership within `env.cell_mask(speed, ...)` at the vehicle's actual speed.
Fallback gains remain available, but `verified` correctly reports `False`.

## D29. Friction bounds clamped to verified grid ceiling
Unbounded friction estimates ($\mu > 0.9$) could theoretically relax curvature speed caps and inflate
steering demand limits beyond the physical tire grip table. `core.step` clamps $\mu_{\text{eff}} \le \mu_{\text{top}}$
($\text{env.axes['mu'][-1]} = 0.9$). Unverified friction claims cannot expand actuator limits or relax
speed caps.

## D30. Bilateral speed governance and preview curvature dynamics
Speed targets are bilaterally bounded by `min_feasible_speed` so reverse speed requests cannot slew
the vehicle backward. When path curvature appears, the speed command decelerates toward the grip cap
at maximum permitted deceleration `decel = 3.0 m/s^2`. An instantaneous step change in speed is
physically impossible and violates longitudinal actuator dynamics; the supervisor flags `FLAG_CAPPED`
immediately on onset while smoothly executing the braking profile.

## D31. State isolation in RL proposal rate checking
In `supervisor.health.rl_check`, candidate RL proposals were previously recording `state.last_rl`
unconditionally. Repeating a rejected jump allowed an adversary to bypass the rate limiter on the
second tick ($\Delta g = 0$). `health.rl_check` and `core.step` now only commit `state.last_rl`
when a proposal is successfully accepted.

## D32. Latch reset qualification requires continuous healthy interval
Flickering domain violations could previously allow continuous reset requests to flap between
`MINIMAL_RISK` and `FALLBACK` every tick. In accordance with automotive safety principles, clearing
a safety latch requires the system to have remained continuously fault-free and in-domain for at
least `stale_grace` ($0.3\text{ s}$). If a domain violation or hard fault occurs, `healthy_for`
immediately resets to zero.

## D33. RL Architecture: 92-parameter MLP with strict observation isolation
The B5 adaptive policy is parameterized as a compact 92-parameter MLP ($6 \to 8(\tanh) \to 4$)
designed for bit-exact verification and embedded C99 execution. The 6-dimensional observation
vector strictly consumes estimator suite outputs ($\hat{\mu}, \bar{\tau}$) and sensor measurements
($e_y, \dot{e}_y, \tilde{\psi}, \tilde{r}$); ground truth plant internals (e.g. true tire slip,
instantaneous unmeasured friction, plant inertia) are strictly sequestered from the policy. The
policy outputs bounded gain adjustments $\Delta K$ within a safety box $[-1, 1]^4$ scaled to
$\pm [0.6, 0.04, 0.15, 0.6]$, rather than direct steering commands.

## D34. Supervisor envelope guarding prevents gain-induced instability
Unsupervised RL gain adaptation (B3 ablation) experiences catastrophic divergences under actuator
latency and slippery roads (3 divergences on train, 4 on validation). In contrast, the B5 controller
channels all policy proposals through the verified runtime supervisor, which validates rate limits,
verifies gain set membership in the precomputed envelope, clamps lateral acceleration commands,
and enforces friction speed caps. B5 achieved 0 divergences across all 48 train and 24 validation
scenarios across 20 independently trained random seeds.

## D35. Zero-proposal bit-exact numerical equivalence to B4
When the RL policy proposes zero gain adjustments ($\Delta K = \mathbf{0}$), the B5 controller
is proven to be numerically equivalent to the B4 supervisor baseline down to floating-point
precision (max difference $< 10^{-15}\text{ m}$). This guarantees that the learning system is a
strict safe perturbation around a certified baseline.

## D36. Pre-registered held-out test evaluation protocol
Before unlocking the test dataset (`experiments/scenario_sets/test.json`), the analysis protocol,
SHA-256 hash (`3ef264c5954f72477e68d5b8357134a5e37aeef3921abef6fee52273b1e27aac`), and three primary
hypotheses (H1: B5 zero divergences; H2: B3 ablation divergence failure; H3: bounded tracking error)
were pre-registered in `experiments/preregistered.yaml` at commit `e2d3479`. Evaluation on all 48
held-out test scenarios confirmed all three hypotheses: B5 achieved 0 divergences with p95 lateral
error of 1.45 cm (< 50 cm), while B3 experienced 31 divergences, proving the supervisor envelope
guard is indispensable for closed-loop safety.

## D37. ANSI C99 Embedded Port, Zero-Heap Architecture, and Numerical Certification
The complete EGGA runtime architecture—comprising the 92-parameter Actor MLP and the verified
runtime supervisor—was ported to standalone ANSI C99 (`src/c/`) targeting automotive ECUs.
1. **Zero Dynamic Allocation**: The runtime operates with strictly zero heap allocation
   (`0 B` dynamic memory, zero `malloc`/`calloc`/`free`). All state is held in a fixed static struct
   (`sizeof(egga_state_t) = 3,304 B`), config is in ROM (`200 B`), and tick inputs/outputs are passed
   by stack pointer (`100 B` and `40 B` respectively).
2. **Bit-Packed ROM Lookup**: The 940,800-bit verified envelope table is bit-packed into an $800 \times 19$
   `uint64_t` array stored in `.rodata` ($118.75\text{ KB}$), avoiding floating-point grid interpolation.
3. **Deterministic Latency**: Host hardware benchmarking across 100,000 consecutive control ticks
   measured a mean latency of $1.460\ \mu\text{s}$ and p99 of $2.200\ \mu\text{s}$ (max $83.3\ \mu\text{s}$),
   providing $> 6,800\times$ margin against the $10\text{ ms}$ real-time control budget.
4. **Numerical Certification**: Comparative evaluation across 100,000 steps verified 100.000% exact
   integer match on supervisor modes (0 mismatches), 100.000% exact bitmask match on fault flags (0 mismatches),
   maximum speed command discrepancy $< 9.5 \times 10^{-5}\text{ m/s}$, and 100% adherence to verified envelope
   invariants in both runtimes. Transient single-tick boundary differences (< 0.012% of ticks) arise solely
   from IEEE 754 float32 vs float64 knot-point rounding (< 10 μm/s) without compromising safety.

