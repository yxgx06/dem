from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]
Gain = tuple[float, float, float, float]

# Per-axis policy outside the grid: None = no verified set, "clamp" = use the nearest cell.
# speed: neither side; mu_lo: higher is safer; tau_bar: lower is safer; mass: lower is safer.
AXIS_POLICY: dict[str, tuple[str | None, str | None]] = {
    "speed": (None, None),
    "mu": (None, "clamp"),
    "tau": ("clamp", None),
    "mass": ("clamp", None),
}
AXIS_ORDER = ("speed", "mu", "tau", "mass")
GAIN_AXES = ("kp", "ki", "kd", "khead")
ARRAY_KEYS = (
    "speed_axis",
    "mu_axis",
    "tau_axis",
    "mass_axis",
    "kp_axis",
    "ki_axis",
    "kd_axis",
    "khead_axis",
    "verified",
    "linear_accepted",
    "rho_fast",
    "reference_gain",
    "ay_max",
    "rate_limit",
    "params",
)


def array_hash(data: dict[str, NDArray[np.generic]]) -> str:
    """SHA-256 over the named arrays (dtype, shape and bytes), in a fixed key order."""
    digest = hashlib.sha256()
    for key in ARRAY_KEYS:
        arr = np.ascontiguousarray(data[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(arr.dtype).encode("utf-8"))
        digest.update(str(arr.shape).encode("utf-8"))
        digest.update(arr.tobytes())
    return digest.hexdigest()


class EnvelopeError(Exception):
    """Raised when an envelope file is missing, malformed or fails its hash check."""


class Envelope:
    """Verified-gain envelope: lookup, projection, demand limits and speed cap.

    `params` holds [ay_fraction_k, gravity, min_feasible_speed_mps].
    """

    def __init__(self, data: dict[str, NDArray[np.generic]]) -> None:
        self.axes: dict[str, FloatArray] = {
            "speed": np.asarray(data["speed_axis"], dtype=np.float64),
            "mu": np.asarray(data["mu_axis"], dtype=np.float64),
            "tau": np.asarray(data["tau_axis"], dtype=np.float64),
            "mass": np.asarray(data["mass_axis"], dtype=np.float64),
        }
        self.gain_axes: tuple[FloatArray, ...] = tuple(
            np.asarray(data[f"{name}_axis"], dtype=np.float64) for name in GAIN_AXES
        )
        self.verified: BoolArray = np.asarray(data["verified"], dtype=np.bool_)
        self.linear_accepted: BoolArray = np.asarray(data["linear_accepted"], dtype=np.bool_)
        self.rho_fast: NDArray[np.float32] = np.asarray(data["rho_fast"], dtype=np.float32)
        self.reference_gain: FloatArray = np.asarray(data["reference_gain"], dtype=np.float64)
        self._ay_max: FloatArray = np.asarray(data["ay_max"], dtype=np.float64)
        self._rate_limit: FloatArray = np.asarray(data["rate_limit"], dtype=np.float64)
        params = np.asarray(data["params"], dtype=np.float64)
        self.ay_fraction_k = float(params[0])
        self.gravity = float(params[1])
        self.min_feasible_speed = float(params[2])
        expected = tuple(len(self.axes[a]) for a in AXIS_ORDER) + tuple(
            len(a) for a in self.gain_axes
        )
        if self.verified.shape != expected:
            raise EnvelopeError(
                f"verified mask has shape {self.verified.shape}, expected {expected}"
            )

    @classmethod
    def load(cls, npz_path: Path, manifest_path: Path) -> Envelope:
        if not npz_path.exists() or not manifest_path.exists():
            raise EnvelopeError("envelope file or manifest missing")
        with np.load(npz_path) as loaded:
            data = {key: loaded[key] for key in loaded.files}
        missing = [k for k in ARRAY_KEYS if k not in data]
        if missing:
            raise EnvelopeError(f"envelope is missing arrays: {missing}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if array_hash(data) != manifest.get("hash"):
            raise EnvelopeError("envelope hash does not match its manifest (corrupted table)")
        return cls(data)

    # ------------------------------------------------------------------ lookup
    def _bracket(self, axis_name: str, value: float) -> tuple[int, int] | None:
        axis = self.axes[axis_name]
        low_policy, high_policy = AXIS_POLICY[axis_name]
        if not math.isfinite(value):
            return None
        n = len(axis)
        if value < axis[0]:
            return (0, 0) if low_policy == "clamp" else None
        if value > axis[-1]:
            return (n - 1, n - 1) if high_policy == "clamp" else None
        upper = int(np.searchsorted(axis, value, side="left"))
        if axis[upper] == value:
            return upper, upper
        return upper - 1, upper

    def cell_mask(
        self, speed: float, mu_lo: float, tau_bar: float, mass_hi: float
    ) -> BoolArray | None:
        """Gains verified in every cell bracketing the query, or None outside the grid."""
        brackets = []
        for name, value in zip(AXIS_ORDER, (speed, mu_lo, tau_bar, mass_hi), strict=True):
            bracket = self._bracket(name, value)
            if bracket is None:
                return None
            brackets.append(sorted({bracket[0], bracket[1]}))
        mask = np.ones(self.verified.shape[4:], dtype=np.bool_)
        for idx in itertools.product(*brackets):
            mask = mask & self.verified[idx]
        return mask

    def gain_at(self, index: tuple[int, int, int, int]) -> Gain:
        return (
            float(self.gain_axes[0][index[0]]),
            float(self.gain_axes[1][index[1]]),
            float(self.gain_axes[2][index[2]]),
            float(self.gain_axes[3][index[3]]),
        )

    def gain_distance(self, index: tuple[int, int, int, int], target: Gain) -> float:
        """Squared normalised distance between a candidate-grid gain and a target gain."""
        total = 0.0
        for dim in range(4):
            axis = self.gain_axes[dim]
            span = float(axis[-1] - axis[0]) or 1.0
            total += ((float(axis[index[dim]]) - target[dim]) / span) ** 2
        return total

    def project_index(
        self, mask: BoolArray, target: Gain
    ) -> tuple[int, int, int, int] | None:
        """Index of the nearest verified candidate gain (normalised distance), or None if empty."""
        best: tuple[float, tuple[int, int, int, int]] | None = None
        for row in np.argwhere(mask):
            index = (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
            dist = self.gain_distance(index, target)
            if best is None or dist < best[0]:
                best = (dist, index)
        return None if best is None else best[1]

    def project(self, mask: BoolArray, target: Gain) -> Gain | None:
        """Nearest verified candidate gain (normalised Euclidean distance), or None if empty."""
        index = self.project_index(mask, target)
        return None if index is None else self.gain_at(index)

    def nominal(self, mask: BoolArray) -> Gain | None:
        ref = self.reference_gain
        return self.project(mask, (float(ref[0]), float(ref[1]), float(ref[2]), float(ref[3])))

    def contains(self, mask: BoolArray, gain: Gain) -> bool:
        """True when `gain` is exactly a candidate-grid point marked verified in `mask`."""
        index = []
        for dim in range(4):
            axis = self.gain_axes[dim]
            hits = np.flatnonzero(np.isclose(axis, gain[dim], rtol=0.0, atol=1e-12))
            if hits.size == 0:
                return False
            index.append(int(hits[0]))
        return bool(mask[tuple(index)])

    def max_verified_speed(self, mu_lo: float, tau_bar: float, mass_hi: float) -> float | None:
        """Highest grid speed whose bracketing cells share a verified gain, or None.

        This is the degraded-mode action when no gain is verified at the current speed: lower the
        speed to this value (never above), and go to minimal risk if it is None.
        """
        for speed in sorted((float(s) for s in self.axes["speed"]), reverse=True):
            mask = self.cell_mask(speed, mu_lo, tau_bar, mass_hi)
            if mask is not None and bool(mask.any()):
                return speed
        return None

    # ------------------------------------------------------------------ demand limits
    def ay_max(self, mu_lo: float) -> float:
        """Lateral-acceleration demand limit k * mu_lo * g (continuous in mu_lo)."""
        return self.ay_fraction_k * max(mu_lo, 0.0) * self.gravity

    def rate_limit(self, speed: float, mass: float) -> float | None:
        """Steering-rate limit, the minimum over the bracketing cells; None outside the grid."""
        bs = self._bracket("speed", speed)
        bm = self._bracket("mass", mass)
        if bs is None or bm is None:
            return None
        values = [float(self._rate_limit[i, j]) for i in {bs[0], bs[1]} for j in {bm[0], bm[1]}]
        return min(values)

    def speed_cap(self, kappa_max: float, mu_lo: float) -> float | None:
        """Highest speed whose required lateral acceleration v^2 * kappa stays within a_y,max.

        Returns None when even the lowest verified speed is infeasible (no relaxation of the
        constraint: the caller must go to the fallback / minimal-risk mode).
        """
        v_max = float(self.axes["speed"][-1])
        if not math.isfinite(kappa_max) or not math.isfinite(mu_lo):
            return None
        kappa = abs(kappa_max)
        cap = v_max if kappa < 1e-9 else min(v_max, math.sqrt(self.ay_max(mu_lo) / kappa))
        return cap if cap >= self.min_feasible_speed else None
