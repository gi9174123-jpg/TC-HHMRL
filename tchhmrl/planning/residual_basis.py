from __future__ import annotations

import numpy as np


def residual_basis(
    *,
    candidate_count: int,
    total_current_raw_step: float,
    allocation_logit_raw_step: float,
    ratio_raw_step: float,
    mode: int | None = None,
    active_source_mask: np.ndarray | None = None,
    thermal_headroom: np.ndarray | None = None,
) -> np.ndarray:
    """Return residuals in structured lower latent-action space.

    The five dimensions are
    [total_current_raw, allocation_logit_ld1, allocation_logit_ld2, rho_raw, tau_raw].
    The LED anchor allocation logit is fixed at zero in the decoder, so source
    transfer residuals act through the two LD logits rather than direct I0/I1/I2
    current coordinates.
    """
    k = max(1, int(candidate_count))
    total_delta = float(total_current_raw_step)
    alloc_delta = float(allocation_logit_raw_step)
    ratio_delta = float(ratio_raw_step)
    mode_i = None if mode is None else int(np.clip(int(mode), 0, 2))
    active = np.ones(3, dtype=np.float32)
    if active_source_mask is not None:
        active_arr = np.asarray(active_source_mask, dtype=np.float32).reshape(-1)
        active[: min(3, active_arr.size)] = active_arr[:3]
    ld1_active = bool(active[1] > 0.0)
    ld2_active = bool(active[2] > 0.0)
    rho_valid = mode_i is None or mode_i in {0, 2}
    tau_valid = mode_i is None or mode_i in {1, 2}

    rows: list[np.ndarray] = []

    def add(total: float = 0.0, ld1: float = 0.0, ld2: float = 0.0, rho: float = 0.0, tau: float = 0.0) -> None:
        rows.append(
            np.asarray(
                [
                    total * total_delta,
                    ld1 * alloc_delta,
                    ld2 * alloc_delta,
                    rho * ratio_delta,
                    tau * ratio_delta,
                ],
                dtype=np.float32,
            )
        )

    add()
    pairs: list[tuple[tuple[float, float, float, float, float], tuple[float, float, float, float, float]]] = []

    def pair(total: float = 0.0, ld1: float = 0.0, ld2: float = 0.0, rho: float = 0.0, tau: float = 0.0) -> None:
        pairs.append(((total, ld1, ld2, rho, tau), (-total, -ld1, -ld2, -rho, -tau)))

    pair(total=1.0)
    if ld1_active:
        pair(ld1=1.0)
    if ld2_active:
        pair(ld2=1.0)
    if ld1_active and ld2_active:
        pair(ld1=1.0, ld2=-1.0)
        pair(ld1=1.0, ld2=1.0)
    if rho_valid:
        pair(rho=1.0)
    if tau_valid:
        pair(tau=1.0)
    if rho_valid and tau_valid:
        pair(rho=1.0, tau=-1.0)
        pair(rho=1.0, tau=1.0)
    if ld1_active or ld2_active:
        pair(total=1.0, ld1=1.0 if ld1_active else 0.0, ld2=1.0 if ld2_active else 0.0)

    pair_slots = max(0, (k - 2) // 2)
    pair_idx = 0
    scale = 1.0
    while len(rows) < 1 + 2 * pair_slots and pairs:
        pos, neg = pairs[pair_idx % len(pairs)]
        add(*(scale * np.asarray(pos, dtype=np.float32)))
        if len(rows) >= 1 + 2 * pair_slots:
            break
        add(*(scale * np.asarray(neg, dtype=np.float32)))
        pair_idx += 1
        if pair_idx % len(pairs) == 0:
            scale *= 0.5

    if thermal_headroom is not None:
        headroom = np.asarray(thermal_headroom, dtype=np.float32).reshape(-1)[:3]
        if headroom.size == 3 and np.all(np.isfinite(headroom)):
            hot = int(np.argmin(headroom))
            cool = int(np.argmax(headroom))

            def source_shift(src: int, sign: float) -> tuple[float, float]:
                if src == 1:
                    return sign, 0.0
                if src == 2:
                    return 0.0, sign
                return -sign, -sign

            cool_ld1, cool_ld2 = source_shift(cool, 1.0)
            hot_ld1, hot_ld2 = source_shift(hot, -1.0)
            guided_ld1 = cool_ld1 + hot_ld1 if ld1_active else 0.0
            guided_ld2 = cool_ld2 + hot_ld2 if ld2_active else 0.0
            add(total=-1.0 if float(np.min(headroom)) < 1.0 else 0.0, ld1=guided_ld1, ld2=guided_ld2)

    if len(rows) < k:
        add(total=-1.0)

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
