# EGGA: Envelope-Guarded Gain Adaptation (research prototype)

Goal: let a learned steering-gain scheduler run safely when tyre-road friction and steering delay
are known only as intervals, by keeping it inside a pre-verified envelope. The standing rules are
in [CLAUDE.md](CLAUDE.md).

## Status

Phases 0-2 of 10 done (baseline repair, plant fidelity, honest baselines). Nothing in this repo has been verified for stability, safety or
production use. Wording about stability is limited to "verified under assumptions A1..An" once
Phase 4 exists.

What exists now:

- A Python port of the 75 s five-stage mission, the 2-DOF bicycle plant, and four named
  controllers: `pid`, `pd_ff` (the legacy "MPC"), `rl_handtyped`, `rl_trained`.
- A stress suite with logged results in `results/phase0/` and generated tables:
  [docs/phase0_table.md](docs/phase0_table.md) and
  [docs/phase0_discrepancies.md](docs/phase0_discrepancies.md), and the Phase 1 plant report
  [docs/phase1_report.md](docs/phase1_report.md), and the Phase 2 baselines
  [docs/phase2_report.md](docs/phase2_report.md).
- A MATLAB cross-check (`matlab/crosscheck.m`) whose golden output is compared by the tests.

Every number is generated from logs; none is typed into this README. The original prototype is in
[legacy/](legacy/README.md), including what each file really does.

## Run

```bash
uv sync --extra dev
uv run pytest -q                  # tests (plant analytics, determinism, MATLAB golden)
uv run python -m egga.eval.stress # writes results/phase0/
uv run python -m egga.eval.report # regenerates docs/phase0_*.md
```

`make` targets mirror these (`make test`, `make reproduce`).

## Known limits (Phase 0)

- Plant: 2-DOF, linear tyres clipped at mu*Fz, constant speed, no actuator lag, no sensor model
  beyond optional white noise on cross-track error.
- Friction and slope are ground-truth inputs to the controllers; there is no estimator yet.
- The "RL" scheduler is the original hand-typed or exported network plus hand rules; it is not
  retrained here. Training is Phase 6.
- Python vs MATLAB agreement is shown for the nominal mission only. MATLAB has no delay or noise
  model, so those cases have no MATLAB reference.
