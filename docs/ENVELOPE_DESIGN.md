# Envelope design (Phase 4)

Status: design document for the verified-gain envelope. Illustrative engineering artefact; the claim
is "stable under assumptions A1..A12" (see `ASSUMPTIONS.md`), nothing more.

## What the envelope is

A table over a grid of (speed, mu_lo, tau_bar, mass_scale) that stores, for each cell, which
steering-PID gains `[Kp, Ki, Kd, Khead]` from a candidate grid are *verified*, plus the demand limits
for that cell (a_y,max, steering-rate limit) and a reference-feasibility speed cap. The runtime
supervisor (Phase 5) may only apply gains that the table marks verified.

## Linear model and criteria

For each cell and each candidate gain we build the discrete closed loop at the control rate:
the 4-state lateral error model (lateral error, its rate, heading error, its rate; Rajamani
form, zero-order-hold discretisation) driven by the steering command through an integer-step delay
buffer sized to `tau_bar`, with the controller's own states (integral, previous error, filtered
derivative state). The controller algebra is identical to `PIDController` (a test steps both and
compares). Cornering stiffness is analysed at two corners (`analysis.stiffness_scales`) and the
cell mass scale scales mass and yaw inertia together.

A gain is accepted in a cell when, at every stiffness corner:

1. **Stability margin:** every pole satisfies `|lambda| <= 1 - stability_margin`.
2. **Fast-mode decay:** every pole except the integrator mode (the largest real pole) satisfies
   `|lambda| <= fast_rho_max`.
3. **Damping:** complex poles with `|lambda| > damping_check_radius` have damping ratio
   `>= min_damping`.
4. **Delay margin:** the loop is still stable (criterion 1) with the delay raised to
   `delay_margin_factor * tau_bar` (and at least `delay_margin_min_extra_steps` more steps).

*Why not a single spectral radius such as 0.97?* With integral action at 100 Hz the integrator pole
sits within about 1e-3 of the unit circle for every usable gain, so a plain spectral-radius bound
would reject all gains. The integrator mode is therefore checked only for strict stability, and the
margin is demanded on the remaining, fast dynamics (criteria 2 and 3) and on the delay (criterion 4).
All thresholds are in `configs/envelope.yaml`; they are engineering choices, not derived constants.

## Nonlinear confirmation

Linear analysis is not trusted alone. For every cell with a non-empty accepted set we simulate the
Phase 1 plant (Pacejka tyres, load transfer, actuator) for a sample of `tests_per_cell` gains: the one
nearest the reference gains, the largest-margin gain, the highest-Kp, highest-Kd and highest-Khead
accepted gains, and seeded random accepted gains. This is a SAMPLE, not an exhaustive check: accepted
gains that were not tested rely on the linear analysis (assumption A12), and the report gives the
failure rate of the random tests as an estimate of how often the linear analysis over-accepts. The manoeuvre is a double lane change whose peak required
lateral acceleration equals the cell's demand limit and whose period is raised until the lateral
jerk stays within `jerk_max_mps3` (so the steering rate stays within the cell's rate limit), with friction equal to `mu_lo`, mass equal to the
cell mass, and true delay equal to `tau_bar` (the worst case the cell covers). A gain passes if the
cross-track error stays below `max_abs_ey_m` and settles below `settle_abs_ey_m` in the last
`settle_window_s`. A failing gain is removed together with every gain that dominates it
componentwise (a conservative prune: it only shrinks the verified set). The pre- and post-confirmation
cell counts are reported.

## Demand limits and speed cap

`a_y,max = k * mu_lo * g`. The steering-rate limit follows from a jerk limit through the
steady-state steering-to-acceleration map, capped by the hardware limit. The reference-feasibility
speed cap lowers speed when the path's required lateral acceleration `v^2 * kappa` exceeds
`a_y,max`; the constraint is never relaxed.

## Lookup semantics (conservative interpolation)

A query `(v, mu_lo, tau_bar, mass_hi)` is mapped to its bracketing cell on every axis; the returned
set is the **intersection (AND) of all bracketing cells**, so the worst neighbour decides. Outside the
grid there is no extrapolation: speed outside the range, `mu_lo` below the lowest value, `tau_bar`
above the highest and `mass` above the highest return "no verified set" (the supervisor must use the
fallback). `mu_lo` above the grid maximum, `tau_bar` below zero and mass below the minimum are clamped
to the nearest cell. Gains are returned only at candidate-grid points inside the set.

## Outputs

`experiments/envelope/envelope_v1.npz` (arrays) and `envelope_v1.json` (manifest: grid, margins,
assumptions, git SHA, array hash). The hash is computed over the arrays and the analysis parameters, so
a rebuild from the same inputs produces the same hash.

## What this does not show

Stability while the gains or parameters change (A9), behaviour under sensor faults (A7), split-mu or
wind (A10), and manoeuvres other than the confirmation lane change (A12). Gain switching between
verified points needs its own argument, which Phase 5 handles by moving only through the verified set.
