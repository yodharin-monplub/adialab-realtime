"""Fisher, Cauchy, and signed-Stouffer aggregation across causal windows.

This bank deliberately contains only low-capacity reductions of the calibrated
`d61` distribution scores and `d64` trend/dependence scores.  It tests the old
winner's multi-window p-value aggregation idea without fitting aggregation
weights or reading labels.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

import detectors61 as d61
import detectors64 as d64


AGGREGATIONS = ("fisher", "cauchy", "signed_stouffer")
GROUPS = tuple(f"d61_{metric}" for metric in d61.METRICS) + tuple(
    f"d64_{metric}" for metric in d64.METRICS
)


def feature_names() -> list[str]:
    names = [f"{group}_{method}" for group in GROUPS for method in AGGREGATIONS]
    names.extend(f"all_evidence_{method}" for method in AGGREGATIONS)
    return names


N_FEATURES = len(feature_names())


def _combine(score: np.ndarray) -> np.ndarray:
    values = np.asarray(score, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    available = np.abs(values) > 1e-12
    count = available.sum(axis=1)
    probability = np.clip(2.0 * ndtr(-np.abs(values)), 1e-12, 1.0)
    fisher_sum = np.sum(np.where(available, -2.0 * np.log(probability), 0.0), axis=1)
    fisher = np.divide(
        fisher_sum - 2.0 * count,
        np.sqrt(np.maximum(4.0 * count, 1.0)),
        out=np.zeros(values.shape[0], dtype=np.float64),
        where=count > 0,
    )
    cauchy_terms = np.tan(np.pi * (0.5 - probability))
    cauchy = np.divide(
        np.sum(np.where(available, cauchy_terms, 0.0), axis=1),
        np.maximum(count, 1),
    )
    stouffer = np.divide(
        np.sum(np.where(available, values, 0.0), axis=1),
        np.sqrt(np.maximum(count, 1)),
    )
    output = np.column_stack((fisher, cauchy, stouffer))
    output[count == 0] = 0.0
    return np.clip(
        np.nan_to_num(output, nan=0.0, posinf=12.0, neginf=-12.0),
        -12.0,
        12.0,
    )


def aggregate_features(score61: np.ndarray, score64: np.ndarray) -> np.ndarray:
    """Aggregate already-computed d61 and d64 matrices without recomputation."""

    first = np.asarray(score61, dtype=np.float64)
    second = np.asarray(score64, dtype=np.float64)
    if first.ndim == 1:
        first = first[None, :]
    if second.ndim == 1:
        second = second[None, :]
    if first.shape[0] != second.shape[0]:
        raise ValueError("d61 and d64 aggregation rows differ")
    base61 = first[:, : len(d61.WINDOWS) * len(d61.METRICS)].reshape(
        first.shape[0], len(d61.WINDOWS), len(d61.METRICS)
    )
    base64 = second[:, : len(d64.WINDOWS) * len(d64.METRICS)].reshape(
        second.shape[0], len(d64.WINDOWS), len(d64.METRICS)
    )
    blocks = []
    for metric in range(len(d61.METRICS)):
        blocks.append(_combine(base61[:, :, metric]))
    for metric in range(len(d64.METRICS)):
        blocks.append(_combine(base64[:, :, metric]))
    all_evidence = np.column_stack((base61.reshape(first.shape[0], -1), base64.reshape(first.shape[0], -1)))
    blocks.append(_combine(all_evidence))
    result = np.column_stack(blocks)
    if result.shape[1] != N_FEATURES:
        raise AssertionError("detectors65 aggregation contract changed")
    return result


def compute_features_offline(
    historical: np.ndarray, online: np.ndarray
) -> np.ndarray:
    """Aggregate independently computed causal d61/d64 score matrices."""

    return aggregate_features(
        d61.compute_features_offline(historical, online),
        d64.compute_features_offline(historical, online),
    )


class StreamDetector65:
    """Streaming aggregation over independent d61 and d64 states."""

    def __init__(self, historical: np.ndarray):
        self.first = d61.StreamDetector61(historical)
        self.second = d64.StreamDetector64(historical)

    def step(self, value: float) -> np.ndarray:
        return aggregate_features(self.first.step(value), self.second.step(value))[0]


if __name__ == "__main__":
    rng = np.random.default_rng(6501)
    history = rng.normal(size=900)
    online = rng.normal(size=150)
    online[77:] *= 1.7
    offline = compute_features_offline(history, online)
    detector = StreamDetector65(history)
    streamed = np.vstack([detector.step(value) for value in online])
    error = float(np.max(np.abs(offline - streamed)))
    assert error < 1e-9 and np.all(np.isfinite(offline))
    print(f"OK: detectors65 stream == offline; {N_FEATURES} features; error={error:.3g}")
