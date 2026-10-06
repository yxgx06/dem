"""Explain the delay discrepancy between the audit port (legacy/lib3.py) and the egga port.

Patches lib3's heading update to MATLAB's order (psi integrates the OLD yaw rate), then prints
lib3 results for N buffer steps. Compare with docs/phase0_table.md: lib3 with N steps matches egga
at (N+1) x 10 ms, i.e. lib3's buffer holds one extra step. MATLAB itself has no delay model.
"""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np

LEGACY = Path(__file__).resolve().parents[1] / "legacy"
OLD = "Vy+=dVy*dt;r+=dr*dt;psi+=r*dt"
NEW = "psi+=r*dt;Vy+=dVy*dt;r+=dr*dt"


def load_patched() -> types.ModuleType:
    source = (LEGACY / "lib3.py").read_text(encoding="utf-8")
    if OLD not in source:
        raise RuntimeError("lib3.py heading update line not found")
    module = types.ModuleType("lib3_matlab_order")
    exec(compile(source.replace(OLD, NEW), "lib3_matlab_order", "exec"), module.__dict__)  # noqa: S102
    return module


def main() -> None:
    lib = load_patched()
    print("steps | pid | mpc | rl   (max e_y cm, DIV = diverged)")
    for steps in (6, 8, 9, 10):
        cells = []
        for name in ("pid", "mpc", "rl"):
            ey = lib.run(name, delay=steps)
            bad = (not np.all(np.isfinite(ey))) or np.max(np.abs(ey)) > 1.5
            cells.append("DIV" if bad else f"{np.max(np.abs(ey)) * 100:.2f}")
        print(f"{steps:5d} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
