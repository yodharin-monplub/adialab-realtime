"""History-calibrated rank, Anderson--Darling, and energy evidence.

The original winning features compared two complete samples.  This real-time
rewrite compares each *observed* online suffix with a reference distribution
frozen from clean history.  Anderson--Darling and one-dimensional energy
distance are evaluated on a fixed historical-quantile grid, which keeps both
offline extraction and streaming inference causal and inexpensive.
"""

from __future__ import annotations

import numpy as np

from sequential_utils import (
    aggregate_maxima,
    fit_location_scale,
    frozen_midrank,
    last_maxima,
    rolling_sum,
    standardize,
)


WINDOWS = (16, 32, 64, 128, 256)
PROBABILITIES = np.asarray(
    (0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975, 0.99),
    dtype=np.float64,
)
METRICS = ("wilcoxon", "anderson_darling", "energy", "tail_balance")
MAX_WINDOW = max(WINDOWS)


def feature_names() -> list[str]:
    names = [f"{metric}_w{window}" for window in WINDOWS for metric in METRICS]
    for metric in METRICS:
        names.extend((f"{metric}_maxabs", f"{metric}_signedmax"))
    return names


N_FEATURES = len(feature_names())


def _raw_statistics(
    rank_sum: np.ndarray,
    cdf_sum: np.ndarray,
    window: int,
    grid_width: np.ndarray,
) -> np.ndarray:
    rank_mean = np.asarray(rank_sum, dtype=np.float64) / window
    cdf = np.asarray(cdf_sum, dtype=np.float64) / window
    deviation = cdf - PROBABILITIES
    denominator = np.maximum(PROBABILITIES * (1.0 - PROBABILITIES), 1e-6)
    wilcoxon = np.sqrt(12.0 * window) * (rank_mean - 0.5)
    anderson = window * np.mean(deviation * deviation / denominator, axis=-1)
    energy = 2.0 * window * np.sum(
        deviation * deviation * grid_width, axis=-1
    )
    tail_balance = np.sqrt(window) * np.mean(
        deviation / np.sqrt(denominator), axis=-1
    )
    return np.column_stack((wilcoxon, anderson, energy, tail_balance))


class HistoryNull61:
    """Frozen empirical reference and pseudo-suffix calibrations."""

    def __init__(self, historical: np.ndarray):
        history = np.asarray(historical, dtype=np.float64)
        if history.ndim != 1 or history.size < MAX_WINDOW + 8:
            raise ValueError("detectors61 needs at least 264 historical values")
        self.sorted_history = np.sort(history)
        self.thresholds = np.quantile(history, PROBABILITIES)
        reference_scale = max(float(np.std(history)), 1e-12)
        self.grid_width = np.clip(
            np.gradient(self.thresholds),
            1e-4 * reference_scale,
            5.0 * reference_scale,
        ) / reference_scale
        ranks, indicators = self.transform_array(history)
        self.calibration: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for window in WINDOWS:
            raw = _raw_statistics(
                rolling_sum(ranks, window),
                rolling_sum(indicators, window),
                window,
                self.grid_width,
            )
            self.calibration[window] = fit_location_scale(raw)

    def transform_array(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        query = np.asarray(values, dtype=np.float64)
        ranks = frozen_midrank(self.sorted_history, query)
        indicators = query[:, None] <= self.thresholds[None, :]
        return ranks, indicators.astype(np.float64)

    def transform_one(self, value: float) -> tuple[float, np.ndarray]:
        rank, indicators = self.transform_array(np.asarray([value]))
        return float(rank[0]), indicators[0]

    def score(self, window: int, raw: np.ndarray) -> np.ndarray:
        center, scale = self.calibration[window]
        return standardize(raw, center, scale)


def compute_features_offline(
    historical: np.ndarray, online: np.ndarray
) -> np.ndarray:
    """Return causal features for every online prefix."""

    null = HistoryNull61(historical)
    values = np.asarray(online, dtype=np.float64)
    if values.size == 0:
        return np.empty((0, N_FEATURES), dtype=np.float64)
    ranks, indicators = null.transform_array(values)
    columns: list[np.ndarray] = []
    by_metric = {metric: [] for metric in METRICS}
    for window in WINDOWS:
        score = np.zeros((values.size, len(METRICS)), dtype=np.float64)
        if values.size >= window:
            raw = _raw_statistics(
                rolling_sum(ranks, window),
                rolling_sum(indicators, window),
                window,
                null.grid_width,
            )
            score[window - 1 :] = null.score(window, raw)
        for index, metric in enumerate(METRICS):
            columns.append(score[:, index])
            by_metric[metric].append(score[:, index])
    columns.extend(aggregate_maxima(by_metric, METRICS))
    result = np.column_stack(columns)
    if result.shape != (values.size, N_FEATURES):
        raise AssertionError("detectors61 emitted an invalid feature shape")
    return result


class StreamDetector61:
    """O(quantile-grid × windows) streaming twin of the offline extractor."""

    def __init__(self, historical: np.ndarray):
        self.null = HistoryNull61(historical)
        self.time = 0
        self.rank_ring = np.zeros(MAX_WINDOW, dtype=np.float64)
        self.cdf_ring = np.zeros((MAX_WINDOW, PROBABILITIES.size), dtype=np.float64)
        self.rank_sums = {window: 0.0 for window in WINDOWS}
        self.cdf_sums = {
            window: np.zeros(PROBABILITIES.size, dtype=np.float64)
            for window in WINDOWS
        }

    def step(self, value: float) -> np.ndarray:
        time = self.time
        count = time + 1
        position = time % MAX_WINDOW
        rank, indicators = self.null.transform_one(float(value))
        by_metric = {metric: [] for metric in METRICS}
        output: list[float] = []
        for window in WINDOWS:
            self.rank_sums[window] += rank
            self.cdf_sums[window] += indicators
            if time >= window:
                old = (time - window) % MAX_WINDOW
                self.rank_sums[window] -= self.rank_ring[old]
                self.cdf_sums[window] -= self.cdf_ring[old]
            if count < window:
                scores = np.zeros(len(METRICS), dtype=np.float64)
            else:
                raw = _raw_statistics(
                    np.asarray([self.rank_sums[window]]),
                    self.cdf_sums[window][None, :],
                    window,
                    self.null.grid_width,
                )[0]
                scores = self.null.score(window, raw)
            output.extend(float(score) for score in scores)
            for metric, score in zip(METRICS, scores, strict=True):
                by_metric[metric].append(float(score))
        self.rank_ring[position] = rank
        self.cdf_ring[position] = indicators
        output.extend(last_maxima(by_metric, METRICS))
        self.time = count
        result = np.asarray(output, dtype=np.float64)
        if result.size != N_FEATURES:
            raise AssertionError("detectors61 streaming contract changed")
        return result


if __name__ == "__main__":
    rng = np.random.default_rng(6101)
    history = rng.standard_t(6, 900)
    online = rng.standard_t(6, 160)
    online[83:] = 1.5 * online[83:] + 0.4
    offline = compute_features_offline(history, online)
    detector = StreamDetector61(history)
    streamed = np.vstack([detector.step(value) for value in online])
    error = float(np.max(np.abs(offline - streamed)))
    assert error < 1e-9 and np.all(np.isfinite(offline))
    print(f"OK: detectors61 stream == offline; {N_FEATURES} features; error={error:.3g}")
