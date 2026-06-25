from __future__ import annotations

import numpy as np


def residual_basis(
    *,
    candidate_count: int,
    current_step: float,
    ratio_step: float,
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
    current_delta = float(current_step) * 2.0
    ratio_delta = float(ratio_step) * 2.0

    rows: list[np.ndarray] = []

    def add(total: float = 0.0, ld1: float = 0.0, ld2: float = 0.0, rho: float = 0.0, tau: float = 0.0) -> None:
        rows.append(
            np.asarray(
                [
                    total * current_delta,
                    ld1 * current_delta,
                    ld2 * current_delta,
                    rho * ratio_delta,
                    tau * ratio_delta,
                ],
                dtype=np.float32,
            )
        )

    add()
    add(total=1.0)
    add(total=-1.0)
    add(ld1=1.0)
    add(ld1=-1.0)
    add(ld2=1.0)
    add(ld2=-1.0)
    add(ld1=1.0, ld2=1.0)      # anchor -> LDs
    add(ld1=-1.0, ld2=-1.0)    # LDs -> anchor
    add(ld1=1.0, ld2=-1.0)     # LD2 -> LD1
    add(ld1=-1.0, ld2=1.0)     # LD1 -> LD2
    add(total=1.0, ld1=1.0, ld2=1.0)
    add(total=-1.0, ld1=-1.0, ld2=-1.0)
    add(rho=1.0)
    add(rho=-1.0)
    add(tau=1.0)
    add(tau=-1.0)
    add(rho=1.0, tau=1.0)
    add(rho=1.0, tau=-1.0)
    add(rho=-1.0, tau=1.0)
    add(rho=-1.0, tau=-1.0)

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
            add(ld1=cool_ld1 + hot_ld1, ld2=cool_ld2 + hot_ld2)
            add(ld1=-(cool_ld1 + hot_ld1), ld2=-(cool_ld2 + hot_ld2))
            add(total=-1.0, ld1=hot_ld1, ld2=hot_ld2)

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
