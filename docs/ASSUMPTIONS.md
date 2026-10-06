# Assumptions register (envelope)

Status: illustrative engineering artefact, not a certified safety case. Every envelope claim holds
only under these assumptions. The statements are the single source in `configs/envelope.yaml`
(`assumptions:`) and are copied into the envelope manifest; the IDs here must match (tested).

| ID | Assumption | How it is checked | Status |
|---|---|---|---|
| A1 | Linear-tyre 2-DOF bicycle valid while tyre utilisation stays below `ay_fraction_k` | Demand limit `a_y,max = k mu_lo g`; nonlinear confirmation at the limit | Partly checked |
| A2 | True friction >= `mu_lo` | Estimator coverage (Phase 3); supervisor uses a lower bound | Depends on estimator |
| A3 | Total steering latency <= `tau_bar`, constant | Delay-bound estimator (Phase 3); confirmation uses true delay = `tau_bar` | Partly checked |
| A4 | Mass scale <= cell mass | Mass interval is NOT trusted under steering noise (Phase 3); prior upper bound used otherwise | Unverified in noise |
| A5 | Speed constant inside a cell | Not checked at run time yet | Unverified |
| A6 | Effective stiffness within the stiffness corners | Two corners in the linear analysis; Pacejka plant in confirmation | Partly checked |
| A7 | Unbiased sensors, small noise, no dropout | Not covered by the table | Unverified |
| A8 | Rate and angle limits do not bind in the verified regime | Demand limits; confirmation runs the real actuator limits | Partly checked |
| A9 | Frozen-parameter, frozen-gain analysis | Not covered | Known limitation |
| A10 | Flat road, uniform friction, no wind, no split-mu | Not covered | Known limitation |
| A11 | Controller is the stated PID with the stated derivative filter | Test compares the analysed algebra with `PIDController` | Checked |
| A12 | Confirmation covers one lane-change manoeuvre | Stated scope | Known limitation |

The envelope's claim is "stable under A1..A12", nothing more. In particular it does not claim
robustness to sensor faults, parameter switching, or conditions outside the grid.
