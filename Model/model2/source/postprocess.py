"""Causal score-state transforms for the absorbing break target.

Once a structural break occurs the label remains one.  A row model can still
flicker when a short suffix window stops containing the onset.  The decayed
maximum below preserves strong prior evidence while allowing isolated false
alarms to expire.  It is mathematically causal and can be reproduced with one
scalar of state during streaming inference.

No transform is enabled merely because it looks plausible.  ``evaluate_grid``
exists so a penalty can be selected out of fold; raw predictions are always
included as the ``None`` candidate.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

from tsauc import tsauc


DEFAULT_PENALTIES: tuple[Optional[float], ...] = (
    None,
    0.0,
    0.01,
    0.025,
    0.05,
    0.10,
    0.20,
    0.50,
)


def logit(probability: np.ndarray, epsilon: float = 1e-7) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float64), epsilon, 1.0 - epsilon)
    return np.log(p) - np.log1p(-p)


def causal_decayed_max(
    prediction: Iterable[float],
    series_position: Iterable[int],
    penalty: Optional[float],
) -> np.ndarray:
    """Apply ``state_t = max(raw_t, state_(t-1) - penalty)`` per series.

    ``prediction`` is converted to log odds first.  ``penalty=None`` returns
    raw log odds (no post-processing); zero is a permanent running maximum.
    Rows for each series must be contiguous and chronologically ordered, as
    they are in the competition files and feature-cache metadata.
    """

    raw = logit(np.asarray(prediction, dtype=np.float64))
    owner = np.asarray(series_position)
    if raw.shape != owner.shape or raw.ndim != 1:
        raise ValueError("prediction and series_position must be same-length 1-D")
    if penalty is None:
        return raw
    if penalty < 0 or not np.isfinite(penalty):
        raise ValueError("penalty must be finite and non-negative, or None")
    if raw.size == 0:
        return raw
    if np.any(owner[1:] < owner[:-1]):
        raise ValueError("series_position must be nondecreasing")

    output = np.empty_like(raw)
    boundaries = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1], True])
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        age = np.arange(right - left, dtype=np.float64)
        tilted = raw[left:right] + penalty * age
        output[left:right] = np.maximum.accumulate(tilted) - penalty * age
    return output


def evaluate_grid(
    target: np.ndarray,
    prediction: np.ndarray,
    time_online: np.ndarray,
    series_position: np.ndarray,
    penalties: Sequence[Optional[float]] = DEFAULT_PENALTIES,
) -> dict[str, float]:
    """Score raw and causal-persistence choices on fixed held-out rows."""

    scores: dict[str, float] = {}
    for penalty in penalties:
        transformed = causal_decayed_max(prediction, series_position, penalty)
        name = "raw" if penalty is None else f"decayed_max_{penalty:g}"
        scores[name] = tsauc(target, transformed, time_online)
    return scores


class StreamingDecayedMax:
    """One-scalar streaming equivalent of :func:`causal_decayed_max`."""

    def __init__(self, penalty: Optional[float]) -> None:
        if penalty is not None and (penalty < 0 or not np.isfinite(penalty)):
            raise ValueError("penalty must be finite and non-negative, or None")
        self.penalty = penalty
        self.state = -np.inf

    def step(self, probability: float) -> float:
        raw = float(logit(np.asarray([probability]))[0])
        if self.penalty is None:
            return raw
        self.state = max(raw, self.state - self.penalty)
        return self.state

