.PHONY: test lint stress report matlab-golden check-lib3 reproduce

test:
	uv run pytest -q

lint:
	uv run ruff check src tests tools

stress:
	uv run python -m egga.eval.stress

report:
	uv run python -m egga.eval.report

matlab-golden:
	cd matlab && matlab -batch "crosscheck('../results/phase0/matlab_golden.json')"

check-lib3:
	uv run python tools/check_lib3_delay_offset.py

# Phase 0 reproduce: regenerates the stress logs and the generated tables (MATLAB golden is
# regenerated separately with `make matlab-golden` because it needs a MATLAB licence).
reproduce: stress report test
