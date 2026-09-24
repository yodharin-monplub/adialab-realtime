"""Causal score-state transforms for the absorbing break target.

Once a structural break occurs the label remains one.  A row model can still
flicker when a short suffix window stops containing the onset.  The two
decayed maxima below preserve strong prior evidence while allowing isolated
false alarms to expire.  Both are mathematically causal and can be reproduced
with one scalar of state during streaming inference.

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


# Multiplicative persistence is much more sensitive near one on long series,
# so the default grid is intentionally dense there.  ``1.0`` is the strict
# running maximum requested by the absorbing-target hypothesis; ``0.5`` has a
# one-step half-life.  Raw predictions are represented separately by ``None``.
DEFAULT_PROBABILITY_DECAY_FACTORS: tuple[float, ...] = (
    1.0,
    0.9999,
    0.999,
    0.995,
    0.99,
    0.98,
    0.95,
    0.90,
    0.85,
    0.80,
    0.70,
    0.60,
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


def causal_probability_decayed_max(
    prediction: Iterable[float],
    series_position: Iterable[int],
    decay_factor: Optional[float],
) -> np.ndarray:
    """Apply ``state_t = max(raw_t, factor * state_(t-1))`` per series.

    Unlike :func:`causal_decayed_max`, this recurrence stays in probability
    space.  ``decay_factor=None`` is the exact raw baseline, while ``1.0`` is
    a permanent running maximum.  Rows must be contiguous and chronological
    within each nondecreasing ``series_position`` block.
    """

    raw = np.asarray(prediction, dtype=np.float64)
    owner = np.asarray(series_position)
    if raw.shape != owner.shape or raw.ndim != 1:
        raise ValueError("prediction and series_position must be same-length 1-D")
    if not np.all(np.isfinite(raw)) or np.any((raw < 0.0) | (raw > 1.0)):
        raise ValueError("prediction must contain finite probabilities in [0, 1]")
    if decay_factor is None:
        return raw.copy()
    if (
        not np.isfinite(decay_factor)
        or decay_factor < 0.0
        or decay_factor > 1.0
    ):
        raise ValueError("decay_factor must be finite and in [0, 1], or None")
    if raw.size == 0:
        return raw.copy()
    if np.any(owner[1:] < owner[:-1]):
        raise ValueError("series_position must be nondecreasing")

    if decay_factor == 0.0:
        return raw.copy()

    output = np.empty_like(raw)
    log_factor = float(np.log(decay_factor))
    boundaries = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1], True])
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        values = raw[left:right]
        age = np.arange(right - left, dtype=np.float64)
        log_values = np.full(values.shape, -np.inf, dtype=np.float64)
        positive = values > 0.0
        log_values[positive] = np.log(values[positive])
        tilted = log_values - log_factor * age
        output[left:right] = np.exp(
            np.maximum.accumulate(tilted) + log_factor * age
        )
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


class StreamingProbabilityDecayedMax:
    """Streaming equivalent of :func:`causal_probability_decayed_max`."""

    def __init__(self, decay_factor: Optional[float]) -> None:
        if decay_factor is not None and (
            not np.isfinite(decay_factor)
            or decay_factor < 0.0
            or decay_factor > 1.0
        ):
            raise ValueError("decay_factor must be finite and in [0, 1], or None")
        self.decay_factor = decay_factor
        self.state = 0.0

    def step(self, probability: float) -> float:
        raw = float(probability)
        if not np.isfinite(raw) or not 0.0 <= raw <= 1.0:
            raise ValueError("probability must be finite and in [0, 1]")
        if self.decay_factor is None:
            return raw
        self.state = max(raw, self.decay_factor * self.state)
        return self.state
