from __future__ import annotations

import copy
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from egga.config import REPO_ROOT, config_hash, load_baselines, load_config, set_baseline_override
from egga.controllers.lqr import DesignVehicle, discretize, error_model
from egga.eval.closed_loop import load_plant_config, run_closed_loop
from egga.eval.provenance import provenance
from egga.supervisor.envelope import ARRAY_KEYS, array_hash

ENVELOPE_DIR = REPO_ROOT / "experiments" / "envelope"
INTEGRATOR_IMAG_TOL = 1e-9


def deep_update(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def envelope_config(override: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = load_config("envelope.yaml")
    return deep_update(cfg, override) if override else cfg


# ---------------------------------------------------------------- linear model
class LinearLoop:
    """Closed loop z+ = (A_open + b c(g)^T) z for the PID with heading term, delay N steps.

    State z = [e1, e1_dot, e2, e2_dot, I, E, F, b_1 .. b_N] with e = -e1, heading error = -e2,
    I the integrator before this step's update, E the previous e, F the previous filtered
    derivative and b_i the delayed command buffer. The gain row c(g) = g @ R is linear in the gains.
    """

    def __init__(
        self,
        design: DesignVehicle,
        speed: float,
        delay_steps: int,
        dt: float,
        cutoff_hz: float,
    ) -> None:
        a_d, b_d = discretize(*error_model(design, speed), dt)
        n = 7 + delay_steps
        a_f = math.exp(-2.0 * math.pi * cutoff_hz * dt)
        g = (1.0 - a_f) / dt
        a_open = np.zeros((n, n))
        a_open[:4, :4] = a_d
        a_open[4, 4] = 1.0
        a_open[4, 0] = -dt
        a_open[5, 0] = -1.0
        a_open[6, 6] = a_f
        a_open[6, 0] = -g
        a_open[6, 5] = -g
        rows = np.zeros((4, n))  # rows for Kp, Ki, Kd, Khead
        rows[0, 0] = -1.0
        rows[1, 0] = -dt
        rows[1, 4] = 1.0
        rows[2, 0] = -g
        rows[2, 5] = -g
        rows[2, 6] = a_f
        rows[3, 2] = -1.0
        b_col = np.zeros(n)
        if delay_steps == 0:
            b_col[:4] = b_d[:, 0]
        else:
            a_open[:4, 7 + delay_steps - 1] += b_d[:, 0]
            b_col[7] = 1.0
            for i in range(1, delay_steps):
                a_open[7 + i, 7 + i - 1] = 1.0
        self.a_open, self.b_col, self.rows, self.n = a_open, b_col, rows, n

    def matrices(self, gains: NDArray[np.float64]) -> NDArray[np.float64]:
        """Stack of closed-loop matrices, shape (G, n, n), for gains of shape (G, 4)."""
        c = gains @ self.rows
        return self.a_open[None, :, :] + self.b_col[None, :, None] * c[:, None, :]


def design_vehicle(vehicle: dict[str, Any], mass: float, stiffness: float) -> DesignVehicle:
    base = DesignVehicle.from_config(vehicle, mass, True)
    return DesignVehicle(
        base.mass,
        base.yaw_inertia,
        base.lf,
        base.lr,
        base.cf * stiffness,
        base.cr * stiffness,
        base.gravity,
    )


def pole_metrics(
    eig: NDArray[np.complex128], damping_radius: float
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """(max modulus, max modulus excluding the integrator mode, min damping) per gain."""
    mod = np.abs(eig)
    real_mask = np.abs(eig.imag) < INTEGRATOR_IMAG_TOL
    score = np.where(real_mask, eig.real, -np.inf)
    drop = np.argmax(score, axis=1)
    has_real = np.isfinite(score[np.arange(len(eig)), drop])
    fast_mod = mod.copy()
    fast_mod[np.arange(len(eig))[has_real], drop[has_real]] = 0.0
    log_mod = np.log(np.maximum(mod, 1e-300))
    arg = np.angle(eig)
    zeta = -log_mod / np.sqrt(log_mod**2 + arg**2 + 1e-300)
    relevant = (~real_mask) & (mod > damping_radius)
    zeta = np.where(relevant, zeta, np.inf)
    return mod.max(axis=1), fast_mod.max(axis=1), zeta.min(axis=1)


def gain_grid(cfg: dict[str, Any]) -> NDArray[np.float64]:
    axes = [np.asarray(cfg["gains"][k], dtype=np.float64) for k in ("kp", "ki", "kd", "khead")]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=1)


def linear_assessment(
    cfg: dict[str, Any],
    vehicle: dict[str, Any],
    speed: float,
    tau: float,
    mass: float,
    gains: NDArray[np.float64],
) -> tuple[NDArray[np.bool_], NDArray[np.float32]]:
    """Accepted-gain mask and worst-case fast spectral radius over the stiffness corners."""
    dt = float(cfg["dt_s"])
    an = cfg["analysis"]
    cutoff = float(cfg["controller"]["derivative_cutoff_hz"])
    n_delay = int(round(tau / dt))
    n_margin = max(
        int(math.ceil(float(an["delay_margin_factor"]) * tau / dt - 1e-9)),
        n_delay + int(an["delay_margin_min_extra_steps"]),
    )
    ok = np.ones(len(gains), dtype=bool)
    worst = np.zeros(len(gains))
    limit = 1.0 - float(an["stability_margin"])
    for stiffness in an["stiffness_scales"]:
        design = design_vehicle(vehicle, mass, float(stiffness))
        loop = LinearLoop(design, speed, n_delay, dt, cutoff)
        rho, fast, zeta = pole_metrics(
            np.linalg.eigvals(loop.matrices(gains)), float(an["damping_check_radius"])
        )
        ok &= (rho <= limit) & (fast <= float(an["fast_rho_max"]))
        ok &= zeta >= float(an["min_damping"])
        worst = np.maximum(worst, fast)
        margin_loop = LinearLoop(design, speed, n_margin, dt, cutoff)
        rho_m, _, _ = pole_metrics(
            np.linalg.eigvals(margin_loop.matrices(gains)), float(an["damping_check_radius"])
        )
        ok &= rho_m <= limit
    return ok, np.where(ok, worst, np.inf).astype(np.float32)


# ---------------------------------------------------------------- nonlinear confirmation
def _lane_change_amplitude(
    cfg: dict[str, Any], vehicle: dict[str, Any], speed: float, mu_lo: float
) -> float:
    lim = cfg["limits"]
    g = float(vehicle["gravity_mps2"])
    wheelbase = float(vehicle["lf_m"]) + float(vehicle["lr_m"])
    tan_delta = float(lim["ay_fraction_k"]) * mu_lo * g * wheelbase / speed**2
    return min(math.atan(tan_delta), float(cfg["nonlinear"]["max_lc_amp_rad"]))


def lane_change_period(cfg: dict[str, Any], vehicle: dict[str, Any], mu_lo: float) -> float:
    """Period of each lane-change half-wave so the lateral jerk stays within the jerk limit.

    A sinusoidal lateral acceleration with amplitude a and period P has peak jerk 2 pi a / P, so
    P >= 2 pi a_y,max / jerk_max. The configured period is a lower bound.
    """
    lim = cfg["limits"]
    ay_max = float(lim["ay_fraction_k"]) * mu_lo * float(vehicle["gravity_mps2"])
    return max(
        float(cfg["nonlinear"]["lc_period_s"]), 2.0 * math.pi * ay_max / float(lim["jerk_max_mps3"])
    )


def simulate_gain(
    args: tuple[dict[str, Any], float, float, float, float, tuple[float, ...]],
) -> bool:
    """True when the nonlinear lane-change at the demand limit passes for this gain."""
    cfg, speed, mu_lo, tau, mass, gain = args
    vehicle = load_config("vehicle.yaml")
    nl = cfg["nonlinear"]
    amp = _lane_change_amplitude(cfg, vehicle, speed, mu_lo)
    period = lane_change_period(cfg, vehicle, mu_lo)
    duration = float(nl["lc_start_s"]) + 2.0 * period + float(nl["tail_s"])
    mission = load_config("mission.yaml")
    mission.update(
        {
            "t_final_s": duration,
            "vx_mps": speed,
            "stage_bounds_s": [0.0, 1e6, 2e6, 3e6, 4e6, 5e6],
            "reference": {
                "sine_amp_rad": 0.0,
                "sine_period_s": 10.0,
                "lc_start_s": float(nl["lc_start_s"]),
                "lc_amp_rad": amp,
                "lc_period_s": period,
                "lc_segments_s": [0.0, period, 2.0 * period],
            },
        }
    )
    plant = load_plant_config(
        {
            "speed": {"constant_mps": speed},
            "mass": {"mass_scale": mass},
            "actuator": {"delay_s": tau},
            "friction": {"profile": {"type": "constant", "mu": mu_lo}},
        }
    )
    base = load_baselines()["pid_ff"]
    override = {
        "pid_ff": {
            "kp": gain[0],
            "ki": gain[1],
            "kd": gain[2],
            "khead": gain[3],
            "integ_limit": float(base["integ_limit"]),
        }
    }
    set_baseline_override(override)
    try:
        run = run_closed_loop("b0_pid_ff", plant, mission_cfg=mission)
    finally:
        set_baseline_override(None)
    if run.diverged_at_s is not None:
        return False
    ey = np.abs(run.ey[np.isfinite(run.ey)])
    window = int(round(float(nl["settle_window_s"]) / float(mission["dt_s"])))
    return bool(
        ey.max() <= float(nl["max_abs_ey_m"]) and ey[-window:].max() <= float(nl["settle_abs_ey_m"])
    )


def _argmax_gain(mask: NDArray[np.bool_], dims: tuple[int, ...]) -> tuple[int, ...]:
    """Index of the verified gain that is lexicographically largest along `dims`."""
    idx = np.argwhere(mask)
    order = np.lexsort(tuple(idx[:, d] for d in reversed(dims)))
    return tuple(int(v) for v in idx[order[-1]])


PICK_TYPES = ("nominal", "best_margin", "max_kp", "max_kd", "max_khead", "random")


def select_test_gains(
    mask: NDArray[np.bool_],
    rho: NDArray[np.float32],
    reference: NDArray[np.float64],
    gain_axes: list[NDArray[np.float64]],
    count: int,
    rng: np.random.Generator,
) -> list[tuple[str, tuple[int, ...]]]:
    """Nominal, largest-margin, highest Kp / Kd / Khead, then seeded random accepted gains."""
    idx = np.argwhere(mask)
    spans = np.array([float(a[-1] - a[0]) or 1.0 for a in gain_axes])
    values = np.stack([gain_axes[d][idx[:, d]] for d in range(4)], axis=1)
    nominal = tuple(
        int(v) for v in idx[np.argmin((((values - reference) / spans) ** 2).sum(axis=1))]
    )
    best_margin = tuple(int(v) for v in idx[np.argmin(rho[mask])])
    picks: list[tuple[str, tuple[int, ...]]] = [
        ("nominal", nominal),
        ("best_margin", best_margin),
        ("max_kp", _argmax_gain(mask, (0, 2, 3, 1))),
        ("max_kd", _argmax_gain(mask, (2, 0, 3, 1))),
        ("max_khead", _argmax_gain(mask, (3, 0, 2, 1))),
    ]
    for _ in range(max(count - len(picks), 1)):
        picks.append(("random", tuple(int(v) for v in idx[int(rng.integers(len(idx)))])))
    unique: list[tuple[str, tuple[int, ...]]] = []
    seen: set[tuple[int, ...]] = set()
    for kind, pick in picks[: max(count, 1)]:
        if pick not in seen:
            seen.add(pick)
            unique.append((kind, pick))
    return unique


# ---------------------------------------------------------------- build
def demand_limits(
    cfg: dict[str, Any], vehicle: dict[str, Any]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    lim = cfg["limits"]
    g = float(vehicle["gravity_mps2"])
    mus = np.asarray(cfg["grid"]["mu_lo"], dtype=np.float64)
    ay_max = float(lim["ay_fraction_k"]) * mus * g
    speeds = cfg["grid"]["speed_mps"]
    masses = cfg["grid"]["mass_scale"]
    rate = np.zeros((len(speeds), len(masses)))
    for i, v in enumerate(speeds):
        for j, m in enumerate(masses):
            d = DesignVehicle.from_config(vehicle, float(m), True)
            steer_per_jerk = d.wheelbase / float(v) ** 2 + d.understeer_coeff
            rate[i, j] = min(
                float(lim["steer_rate_hw_max_rps"]), steer_per_jerk * float(lim["jerk_max_mps3"])
            )
    return ay_max, rate


def build_envelope(
    override: dict[str, Any] | None = None, workers: int | None = None, verbose: bool = False
) -> tuple[dict[str, NDArray[np.generic]], dict[str, Any]]:
    cfg = envelope_config(override)
    vehicle = load_config("vehicle.yaml")
    speeds = [float(v) for v in cfg["grid"]["speed_mps"]]
    mus = [float(v) for v in cfg["grid"]["mu_lo"]]
    taus = [float(v) for v in cfg["grid"]["tau_bar_s"]]
    masses = [float(v) for v in cfg["grid"]["mass_scale"]]
    gain_axes = [np.asarray(cfg["gains"][k], dtype=np.float64) for k in ("kp", "ki", "kd", "khead")]
    gains = gain_grid(cfg)
    gshape = tuple(len(a) for a in gain_axes)
    shape = (len(speeds), len(mus), len(taus), len(masses)) + gshape

    linear = np.zeros(shape, dtype=bool)
    rho = np.full(shape, np.inf, dtype=np.float32)
    for iv, v in enumerate(speeds):
        for it, tau in enumerate(taus):
            for im, m in enumerate(masses):
                ok, fast = linear_assessment(cfg, vehicle, v, tau, m, gains)
                linear[iv, :, it, im] = ok.reshape(gshape)[None]
                rho[iv, :, it, im] = fast.reshape(gshape)[None]
        if verbose:
            print(f"linear analysis: speed {v} done", flush=True)

    reference = np.array(
        [float(load_baselines()["pid_ff"][k]) for k in ("kp", "ki", "kd", "khead")]
    )
    verified = linear.copy()
    tasks: list[tuple[dict[str, Any], float, float, float, float, tuple[float, ...]]] = []
    owners: list[tuple[int, int, int, int, str, tuple[int, ...]]] = []
    for cell in np.ndindex(len(speeds), len(mus), len(taus), len(masses)):
        mask = linear[cell]
        if not mask.any():
            continue
        iv, imu, it, im = cell
        rng = np.random.default_rng([int(cfg["nonlinear"]["seed"]), iv, imu, it, im])
        for kind, pick in select_test_gains(
            mask, rho[cell], reference, gain_axes, int(cfg["nonlinear"]["tests_per_cell"]), rng
        ):
            gain = tuple(float(gain_axes[d][pick[d]]) for d in range(4))
            tasks.append((cfg, speeds[iv], mus[imu], taus[it], masses[im], gain))
            owners.append((iv, imu, it, im, kind, pick))
    n_workers = workers if workers is not None else max(1, min(8, (os.cpu_count() or 2) - 1))
    if n_workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(simulate_gain, tasks, chunksize=8))
    else:
        results = [simulate_gain(t) for t in tasks]
    failures = 0
    by_type: dict[str, dict[str, int]] = {k: {"tests": 0, "failures": 0} for k in PICK_TYPES}
    for (iv, imu, it, im, kind, pick), passed in zip(owners, results, strict=True):
        by_type[kind]["tests"] += 1
        if not passed:
            by_type[kind]["failures"] += 1
            failures += 1
            sl = (iv, imu, it, im) + tuple(slice(i, None) for i in pick)
            verified[sl] = False  # prune the failing gain and every gain dominating it

    ay_max, rate = demand_limits(cfg, vehicle)
    params = np.array(
        [
            float(cfg["limits"]["ay_fraction_k"]),
            float(vehicle["gravity_mps2"]),
            float(cfg["limits"]["min_feasible_speed_mps"]),
        ]
    )
    data: dict[str, NDArray[np.generic]] = {
        "speed_axis": np.asarray(speeds),
        "mu_axis": np.asarray(mus),
        "tau_axis": np.asarray(taus),
        "mass_axis": np.asarray(masses),
        "kp_axis": gain_axes[0],
        "ki_axis": gain_axes[1],
        "kd_axis": gain_axes[2],
        "khead_axis": gain_axes[3],
        "verified": verified,
        "linear_accepted": linear,
        "rho_fast": rho,
        "reference_gain": reference,
        "ay_max": ay_max,
        "rate_limit": rate,
        "params": params,
    }
    cells = int(np.prod(shape[:4]))
    stable_after = int(verified.reshape(cells, -1).any(axis=1).sum())
    stable_linear = int(linear.reshape(cells, -1).any(axis=1).sum())
    meta = provenance()
    manifest: dict[str, Any] = {
        "version": int(cfg["version"]),
        "hash": array_hash(data),
        "array_keys": list(ARRAY_KEYS),
        "git_sha": meta["git_sha"],
        "git_dirty": meta["git_dirty"],
        "config_hash": config_hash(
            ["envelope.yaml", "vehicle.yaml", "baselines.yaml", "baselines_tuned.yaml"]
        ),
        "grid": cfg["grid"],
        "gains": cfg["gains"],
        "analysis": cfg["analysis"],
        "limits": cfg["limits"],
        "nonlinear": cfg["nonlinear"],
        "assumptions": cfg["assumptions"],
        "counts": {
            "cells": cells,
            "cells_with_linear_set": stable_linear,
            "cells_with_verified_set": stable_after,
            "candidate_gains": int(gains.shape[0]),
            "mean_verified_gains_per_cell": float(verified.reshape(cells, -1).sum(axis=1).mean()),
            "nonlinear_tests": len(tasks),
            "nonlinear_failures": failures,
            "nonlinear_by_pick_type": by_type,
        },
    }
    return data, manifest


def write_envelope(
    data: dict[str, NDArray[np.generic]], manifest: dict[str, Any], out_dir: Path = ENVELOPE_DIR
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"envelope_v{manifest['version']}"
    npz = out_dir / f"{stem}.npz"
    meta = out_dir / f"{stem}.json"
    np.savez_compressed(npz, **data)
    meta.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    return npz, meta
