from __future__ import annotations

import numpy as np


def residual_basis(
    *,
    candidate_count: int,
    current_step: float,
    ratio_step: float,
    thermal_headroom: np.ndarray | None = None,
) -> np.ndarray:
    """Return structured residuals in lower raw-action space.

    Current dimensions use a step converted from physical fraction to the
    tanh-affine raw interval. Rho/tau dimensions use their own step. The basis
    is centered at the policy output and contains load-shift directions rather
    than a fixed global action grid.
    """
    k = max(1, int(candidate_count))
    current_delta = float(current_step) * 2.0
    ratio_delta = float(ratio_step) * 2.0

    rows: list[np.ndarray] = []

    def add(cur: tuple[float, float, float], rho: float = 0.0, tau: float = 0.0) -> None:
        rows.append(
            np.asarray(
                [
                    cur[0] * current_delta,
                    cur[1] * current_delta,
                    cur[2] * current_delta,
                    rho * ratio_delta,
                    tau * ratio_delta,
                ],
                dtype=np.float32,
            )
        )

    add((0.0, 0.0, 0.0))
    add((1.0, 1.0, 1.0))
    add((-1.0, -1.0, -1.0))
    add((-1.0, 0.5, 0.5))
    add((1.0, -0.5, -0.5))
    add((0.0, 1.0, -1.0))
    add((0.0, -1.0, 1.0))
    add((-1.0, 0.0, 0.0))
    add((0.0, -1.0, 0.0))
    add((0.0, 0.0, -1.0))
    add((1.0, 0.0, 0.0))
    add((0.0, 1.0, 0.0))
    add((0.0, 0.0, 1.0))
    add((0.0, 0.0, 0.0), 1.0, 0.0)
    add((0.0, 0.0, 0.0), -1.0, 0.0)
    add((0.0, 0.0, 0.0), 0.0, 1.0)
    add((0.0, 0.0, 0.0), 0.0, -1.0)
    add((0.0, 0.0, 0.0), 1.0, 1.0)
    add((0.0, 0.0, 0.0), 1.0, -1.0)
    add((0.0, 0.0, 0.0), -1.0, 1.0)
    add((0.0, 0.0, 0.0), -1.0, -1.0)

    if thermal_headroom is not None:
        headroom = np.asarray(thermal_headroom, dtype=np.float32).reshape(-1)[:3]
        if headroom.size == 3 and np.all(np.isfinite(headroom)):
            cool = int(np.argmax(headroom))
            hot = int(np.argmin(headroom))
            shift = np.zeros(3, dtype=np.float32)
            shift[cool] += 1.0
            shift[hot] -= 1.0
            add(tuple(shift.tolist()))
            add(tuple((-shift).tolist()))
            reduce_hot = np.zeros(3, dtype=np.float32)
            reduce_hot[hot] = -1.0
            add(tuple(reduce_hot.tolist()))

    if len(rows) < k:
        base = list(rows[1:])
        scale = 0.5
        while len(rows) < k and base:
            for row in base:
                rows.append((scale * row).astype(np.float32))
                if len(rows) >= k:
                    break
            scale *= 0.5

    return np.stack(rows[:k], axis=0).astype(np.float32)
