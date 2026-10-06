from __future__ import annotations

import itertools

from egga.supervisor.envelope import BoolArray, Envelope, Gain

Index = tuple[int, int, int, int]
_OFFSETS = tuple(itertools.product((-1, 0, 1), repeat=4))


def _index_distance(a: Index, b: Index) -> int:
    return sum((x - y) * (x - y) for x, y in zip(a, b, strict=True))


def next_index(
    env: Envelope,
    mask: BoolArray,
    current: Index | None,
    target: Index,
    may_hop: bool,
) -> tuple[Index, bool]:
    """Next applied gain index and whether it was a forced jump.

    Every returned index is verified in `mask`. If the current gain is not verified (first call
    or the envelope shrank) the target is applied at once (forced). Otherwise the gain moves at
    most one grid step per hop (through verified neighbours only) toward the target.
    """
    if current is None or not bool(mask[current]):
        return target, True
    if not may_hop:
        return current, False
    best = current
    best_distance = _index_distance(current, target)
    shape = mask.shape
    for offset in _OFFSETS:
        cand = (
            current[0] + offset[0],
            current[1] + offset[1],
            current[2] + offset[2],
            current[3] + offset[3],
        )
        if any(c < 0 or c >= n for c, n in zip(cand, shape, strict=True)) or not bool(mask[cand]):
            continue
        dist = _index_distance(cand, target)
        if dist < best_distance:
            best, best_distance = cand, dist
    return best, False


def gain_bounds(env: Envelope) -> tuple[Gain, Gain]:
    lower = tuple(float(axis[0]) for axis in env.gain_axes)
    upper = tuple(float(axis[-1]) for axis in env.gain_axes)
    return (lower[0], lower[1], lower[2], lower[3]), (upper[0], upper[1], upper[2], upper[3])
