.PHONY: envelope envelope-check phase4 test lint typecheck stress report phase1 freeze-sets tune baselines calibrate-estimators estimators matlab-golden check-lib3 reproduce

test:
	uv run pytest -q

lint:
	uv run ruff check src tests tools

typecheck:
	uv run mypy --strict src/egga/estimation src/egga/supervisor

envelope:
	uv run python tools/build_envelope.py

envelope-check:
	uv run python tools/build_envelope.py --check

phase4:
	uv run python -m egga.eval.phase4

stress:
	uv run python -m egga.eval.stress

report:
	uv run python -m egga.eval.report

phase1:
	uv run python -m egga.eval.phase1

freeze-sets:
	uv run python -c "from egga.scenarios.sets import freeze; print(freeze())"

tune:
	uv run python -m egga.eval.tuning

baselines:
	uv run python -m egga.eval.baselines

calibrate-estimators:
	uv run python -m egga.eval.estimation_eval --calibrate

estimators:
	uv run python -m egga.eval.estimation_eval

matlab-golden:
	cd matlab && matlab -batch "crosscheck('../results/phase0/matlab_golden.json')"

check-lib3:
	uv run python tools/check_lib3_delay_offset.py

# Phase 0 reproduce: regenerates the stress logs and the generated tables (MATLAB golden is
# regenerated separately with `make matlab-golden` because it needs a MATLAB licence).
reproduce: stress report phase1 baselines estimators phase4 test
