# Progress

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
