# Progress

## Phase 2 (honest baselines): done
- B0 PID+FF, B1 PD+FF, B2 oracle RL, B6 OSQP MPC (Np=15, Nc=5), B7 LQR x3 (true mass, nominal mass, delay-augmented).
- Scenario sets train/val/test frozen and hashed (`experiments/scenario_sets/`); test locked until Phase 7.
- Baselines tuned on train only; evaluated on train and val: `docs/phase2_report.md`.
- LQR audit claims reproduced; B6 solve time measured (HOST).
- Findings: friction-aware limits fail under under-estimated belief (D12); delay-augmented LQR no better (D13).

## Phase 1 (plant fidelity): done
- New plant: Pacejka per-wheel tyre, load transfer, variable speed, actuator (lag, fractional delay, jitter, gain, bias), sensors, friction profiles incl. split-mu, mass/stiffness scaling, crosswind; filtered derivative in every controller.
- Gate 1: equivalence with Phase 0 exact; every effect has a test; `docs/phase1_report.md` generated (plant limits, case table, noise-filter sweep).
- Finding: derivative filter alone does not fix 2 cm noise (D9).

## Done (Phase 0)
- Repo restructured; original files in `legacy/` with a README of what each really does.
- Python port: mission, plant, actuator, controllers (PID, PD+FF, RL with both weight sets).
- Tests: plant analytics, determinism, mission checks, MATLAB golden match.
- Stress suite and generated tables; audit comparison; MATLAB cross-check run in R2025b.
- Synthetic Figure 6 moved to `legacy/synthetic_do_not_cite/` and flagged.

## Measured
See `docs/phase0_table.md` and `docs/phase0_discrepancies.md` (generated).

## Open
- Gate 0 review by the user; commit pending until approved.
- Trained-weights actor differs from the hand-typed one; interpretation belongs in Phase 6.
- Delay modelling is whole-step only; jitter and sub-step delay arrive in Phase 1.
- Phase 1 not started (scope agreed: stop at Gate 0).
