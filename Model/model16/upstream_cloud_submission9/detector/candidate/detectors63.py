"""Symbolic sample-entropy and BDS-style nonlinear-dependence evidence.

Values are discretized by quantiles frozen from clean history.  Rolling state,
transition, triple, and ordinal-pattern frequencies provide inexpensive causal
analogues of sample entropy and the BDS independence statistic.  The same
statistics are calibrated on history-only pseudo-suffixes.
"""

from __future__ import annotations

import numpy as np

from sequential_utils import (
    aggregate_maxima,
    entropy,
    fit_location_scale,
    last_maxima,
    rolling_sum,
    standardize,
)


WINDOWS = (32, 64, 128, 256)
N_BINS = 8
METRICS = (
    "state_entropy",
    "transition_mi",
    "bds_gap",
    "symbolic_sampen",
    "permutation_entropy",
)
MAX_WINDOW = max(WINDOWS)


def feature_names() -> list[str]:
    names = [f"{metric}_w{window}" for window in WINDOWS for metric in METRICS]
    for metric in METRICS:
        names.extend((f"{metric}_maxabs", f"{metric}_signedmax"))
    return names


N_FEATURES = len(feature_names())


def _event_codes(bins: np.ndarray, prefix: np.ndarray) -> tuple[np.ndarray, ...]:
    current = np.asarray(bins, dtype=np.int16)
    context = np.asarray(prefix, dtype=np.int16)
    if context.size < 2:
        raise ValueError("symbolic detector requires two causal context bins")
    extended = np.concatenate((context[-2:], current))
    prior2 = extended[:-2]
    prior1 = extended[1:-1]
    now = extended[2:]
    pair = prior1 * N_BINS + now
    triple = (prior2 * N_BINS + prior1) * N_BINS + now
    ordinal = ((prior1 > prior2).astype(np.int16) << 1) + (
        now > prior1
    ).astype(np.int16)
    return current, pair, triple, ordinal


def _rolling_counts(codes: np.ndarray, categories: int, window: int) -> np.ndarray:
    one_hot = np.eye(categories, dtype=np.float64)[np.asarray(codes, dtype=np.int64)]
    return rolling_sum(one_hot, window)


def _statistics(
    state_count: np.ndarray,
    pair_count: np.ndarray,
    triple_count: np.ndarray,
    ordinal_count: np.ndarray,
    window: int,
) -> np.ndarray:
    p1 = np.asarray(state_count, dtype=np.float64) / window
    p2 = np.asarray(pair_count, dtype=np.float64) / window
    p3 = np.asarray(triple_count, dtype=np.float64) / window
    po = np.asarray(ordinal_count, dtype=np.float64) / window
    joint = p2.reshape((-1, N_BINS, N_BINS))
    left = joint.sum(axis=2, keepdims=True)
    right = joint.sum(axis=1, keepdims=True)
    independent = left * right
    positive = joint > 0
    ratio = np.ones_like(joint)
    ratio[positive] = joint[positive] / np.maximum(independent[positive], 1e-12)
    mutual_information = np.sum(
        np.where(positive, joint * np.log(np.maximum(ratio, 1e-12)), 0.0),
        axis=(1, 2),
    )
    collision1 = np.sum(p1 * p1, axis=1)
    collision2 = np.sum(p2 * p2, axis=1)
    collision3 = np.sum(p3 * p3, axis=1)
    bds_gap = collision2 - collision1 * collision1
    sample_entropy = -np.log(
        np.maximum(collision3, 1e-12) / np.maximum(collision2, 1e-12)
    )
    return np.column_stack(
        (
            entropy(p1) / np.log(N_BINS),
            mutual_information,
            bds_gap,
            sample_entropy,
            entropy(po) / np.log(4.0),
        )
    )


def _raw_for_window(codes: tuple[np.ndarray, ...], window: int) -> np.ndarray:
    state, pair, triple, ordinal = codes
    if state.size < window:
        return np.empty((0, len(METRICS)), dtype=np.float64)
    return _statistics(
        _rolling_counts(state, N_BINS, window),
        _rolling_counts(pair, N_BINS**2, window),
        _rolling_counts(triple, N_BINS**3, window),
        _rolling_counts(ordinal, 4, window),
        window,
    )


class HistoryNull63:
    """Frozen quantizer and history-only nonlinear null calibration."""

    def __init__(self, historical: np.ndarray):
        history = np.asarray(historical, dtype=np.float64)
        if history.ndim != 1 or history.size < MAX_WINDOW + 8:
            raise ValueError("detectors63 needs at least 264 historical values")
        self.edges = np.quantile(history, np.linspace(0.0, 1.0, N_BINS + 1))
        historical_bins = self.bin_array(history)
        prefix = np.repeat(historical_bins[0], 2)
        codes = _event_codes(historical_bins, prefix)
        self.calibration = {
            window: fit_location_scale(_raw_for_window(codes, window))
            for window in WINDOWS
        }
        self.trailing_bins = historical_bins[-2:].copy()

    def bin_array(self, values: np.ndarray) -> np.ndarray:
        return np.searchsorted(
            self.edges[1:-1], np.asarray(values, dtype=np.float64), side="right"
        ).astype(np.int16)

    def score(self, window: int, raw: np.ndarray) -> np.ndarray:
        center, scale = self.calibration[window]
        return standardize(raw, center, scale)


def compute_features_offline(
    historical: np.ndarray, online: np.ndarray
) -> np.ndarray:
    """Vectorized symbolic features for each available online endpoint."""

    null = HistoryNull63(historical)
    values = np.asarray(online, dtype=np.float64)
    if values.size == 0:
        return np.empty((0, N_FEATURES), dtype=np.float64)
    codes = _event_codes(null.bin_array(values), null.trailing_bins)
    columns: list[np.ndarray] = []
    by_metric = {metric: [] for metric in METRICS}
    for window in WINDOWS:
        score = np.zeros((values.size, len(METRICS)), dtype=np.float64)
        raw = _raw_for_window(codes, window)
        if raw.shape[0]:
            score[window - 1 :] = null.score(window, raw)
        for index, metric in enumerate(METRICS):
            columns.append(score[:, index])
            by_metric[metric].append(score[:, index])
    columns.extend(aggregate_maxima(by_metric, METRICS))
    result = np.column_stack(columns)
    if result.shape != (values.size, N_FEATURES):
        raise AssertionError("detectors63 emitted an invalid feature shape")
    return result


class StreamDetector63:
    """Incremental rolling-count implementation of the symbolic statistics."""

    def __init__(self, historical: np.ndarray):
        self.null = HistoryNull63(historical)
        self.previous = self.null.trailing_bins.tolist()
        self.time = 0
        self.rings = {
            "state": np.zeros(MAX_WINDOW, dtype=np.int16),
            "pair": np.zeros(MAX_WINDOW, dtype=np.int16),
            "triple": np.zeros(MAX_WINDOW, dtype=np.int16),
            "ordinal": np.zeros(MAX_WINDOW, dtype=np.int16),
        }
        sizes = {"state": N_BINS, "pair": N_BINS**2, "triple": N_BINS**3, "ordinal": 4}
        self.counts = {
            window: {
                name: np.zeros(size, dtype=np.float64) for name, size in sizes.items()
            }
            for window in WINDOWS
        }

    def step(self, value: float) -> np.ndarray:
        current = int(self.null.bin_array(np.asarray([value]))[0])
        prior2, prior1 = self.previous[-2:]
        events = {
            "state": current,
            "pair": prior1 * N_BINS + current,
            "triple": (prior2 * N_BINS + prior1) * N_BINS + current,
            "ordinal": (int(prior1 > prior2) << 1) + int(current > prior1),
        }
        time = self.time
        count = time + 1
        position = time % MAX_WINDOW
        output: list[float] = []
        by_metric = {metric: [] for metric in METRICS}
        for window in WINDOWS:
            for name, code in events.items():
                self.counts[window][name][code] += 1.0
                if time >= window:
                    old = int(self.rings[name][(time - window) % MAX_WINDOW])
                    self.counts[window][name][old] -= 1.0
            if count < window:
                score = np.zeros(len(METRICS), dtype=np.float64)
            else:
                counts = self.counts[window]
                raw = _statistics(
                    counts["state"][None, :],
                    counts["pair"][None, :],
                    counts["triple"][None, :],
                    counts["ordinal"][None, :],
                    window,
                )[0]
                score = self.null.score(window, raw)
            output.extend(float(item) for item in score)
            for metric, item in zip(METRICS, score, strict=True):
                by_metric[metric].append(float(item))
        for name, code in events.items():
            self.rings[name][position] = code
        self.previous.append(current)
        del self.previous[:-2]
        output.extend(last_maxima(by_metric, METRICS))
        self.time = count
        result = np.asarray(output, dtype=np.float64)
        if result.size != N_FEATURES:
            raise AssertionError("detectors63 streaming contract changed")
        return result


if __name__ == "__main__":
    rng = np.random.default_rng(6301)
    history = rng.normal(size=900)
    online = rng.normal(size=180)
    for index in range(92, online.size):
        online[index] += 0.72 * online[index - 1]
    offline = compute_features_offline(history, online)
    detector = StreamDetector63(history)
    streamed = np.vstack([detector.step(value) for value in online])
    error = float(np.max(np.abs(offline - streamed)))
    assert error < 1e-9 and np.all(np.isfinite(offline))
    print(f"OK: detectors63 stream == offline; {N_FEATURES} features; error={error:.3g}")
