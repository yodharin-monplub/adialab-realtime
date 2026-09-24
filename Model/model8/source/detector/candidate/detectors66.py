"""IQR-winsorized change evidence and causal stationary Symlet-2 energy.

Three historical IQR fences produce robust rolling location, scale, and
outlier-rate features.  A causal undecimated Symlet-2 filter bank adds local
frequency evidence without using symmetric/future padding.  Every channel is
standardized on clean-history pseudo-suffixes.
"""

from __future__ import annotations

import numpy as np

from sequential_utils import (
    aggregate_maxima,
    fit_location_scale,
    last_maxima,
    rolling_sum,
    standardize,
)


IQR_MULTIPLIERS = (1.0, 1.75, 2.0)
ROBUST_WINDOWS = (16, 64, 256)
WAVELET_WINDOWS = (32, 128, 512)
WAVELET_LEVELS = 3
MAX_WINDOW = max(max(ROBUST_WINDOWS), max(WAVELET_WINDOWS))
SYM2_LOW = np.asarray(
    (-0.12940952255092145, 0.22414386804185735,
     0.83651630373746899, 0.48296291314469025),
    dtype=np.float64,
)
SYM2_HIGH = ((-1.0) ** np.arange(SYM2_LOW.size)) * SYM2_LOW[::-1]


def _tag(value: float) -> str:
    return str(value).replace(".", "p")


ROBUST_METRICS = tuple(
    f"{name}_m{_tag(multiplier)}"
    for multiplier in IQR_MULTIPLIERS
    for name in ("winsor_mean", "winsor_logvar", "outlier_rate")
)
WAVELET_METRICS = tuple(f"sym2_detail_l{level}" for level in range(1, WAVELET_LEVELS + 1))


def feature_names() -> list[str]:
    names = [
        f"{metric}_w{window}"
        for window in ROBUST_WINDOWS
        for metric in ROBUST_METRICS
    ]
    names.extend(
        f"{metric}_w{window}"
        for window in WAVELET_WINDOWS
        for metric in WAVELET_METRICS
    )
    for metric in (*ROBUST_METRICS, *WAVELET_METRICS):
        names.extend((f"{metric}_maxabs", f"{metric}_signedmax"))
    return names


N_FEATURES = len(feature_names())


def _causal_filter(values: np.ndarray, coefficients: np.ndarray, dilation: int) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    output = np.zeros_like(source)
    for index, coefficient in enumerate(coefficients):
        shift = index * dilation
        if shift == 0:
            output += coefficient * source
        elif shift < source.size:
            output[shift:] += coefficient * source[:-shift]
    return output


def _wavelet(values: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    approximation = np.asarray(values, dtype=np.float64)
    details: list[np.ndarray] = []
    inputs: list[np.ndarray] = []
    for level in range(WAVELET_LEVELS):
        inputs.append(approximation)
        dilation = 1 << level
        details.append(_causal_filter(approximation, SYM2_HIGH, dilation))
        approximation = _causal_filter(approximation, SYM2_LOW, dilation)
    return details, inputs


def _robust_raw(sums: np.ndarray, window: int) -> np.ndarray:
    mean = sums[:, 0::3] / window
    second = sums[:, 1::3] / window
    outlier = sums[:, 2::3] / window
    log_variance = np.log(np.maximum(second - mean * mean, 1e-12))
    columns = []
    for index in range(len(IQR_MULTIPLIERS)):
        columns.extend((mean[:, index], log_variance[:, index], outlier[:, index]))
    return np.column_stack(columns)


class HistoryNull66:
    """Frozen IQR fences, robust scale, and history-only calibrations."""

    def __init__(self, historical: np.ndarray):
        history = np.asarray(historical, dtype=np.float64)
        if history.ndim != 1 or history.size < MAX_WINDOW + 8:
            raise ValueError("detectors66 needs at least 520 historical values")
        q1, self.median, q3 = np.quantile(history, (0.25, 0.50, 0.75))
        iqr = max(float(q3 - q1), 1e-12)
        self.scale = max(iqr / 1.349, 0.1 * float(history.std()), 1e-12)
        multipliers = np.asarray(IQR_MULTIPLIERS)
        self.lower = q1 - multipliers * iqr
        self.upper = q3 + multipliers * iqr
        channels = self.transform_array(history)
        self.robust_calibration = {}
        for window in ROBUST_WINDOWS:
            raw = _robust_raw(rolling_sum(channels, window), window)
            self.robust_calibration[window] = fit_location_scale(raw)

        historical_z = (history - self.median) / self.scale
        details, inputs = _wavelet(historical_z)
        self.wavelet_tails = []
        for level, values in enumerate(inputs):
            keep = (SYM2_LOW.size - 1) * (1 << level) + 1
            self.wavelet_tails.append(values[-keep:].tolist())
        self.wavelet_calibration = {}
        for window in WAVELET_WINDOWS:
            raw = np.column_stack(
                [
                    np.log(np.maximum(rolling_sum(detail * detail, window) / window, 1e-12))
                    for detail in details
                ]
            )
            self.wavelet_calibration[window] = fit_location_scale(raw)

    def transform_array(self, values: np.ndarray) -> np.ndarray:
        query = np.asarray(values, dtype=np.float64)
        channels = []
        for lower, upper in zip(self.lower, self.upper, strict=True):
            clipped = np.clip(query, lower, upper)
            z = (clipped - self.median) / self.scale
            channels.extend((z, z * z, (query < lower) | (query > upper)))
        return np.column_stack(channels).astype(np.float64)

    def score_robust(self, window: int, raw: np.ndarray) -> np.ndarray:
        center, scale = self.robust_calibration[window]
        return standardize(raw, center, scale)

    def score_wavelet(self, window: int, raw: np.ndarray) -> np.ndarray:
        center, scale = self.wavelet_calibration[window]
        return standardize(raw, center, scale)


def compute_features_offline(
    historical: np.ndarray, online: np.ndarray
) -> np.ndarray:
    """Vectorized causal robust and Symlet features."""

    null = HistoryNull66(historical)
    values = np.asarray(online, dtype=np.float64)
    if values.size == 0:
        return np.empty((0, N_FEATURES), dtype=np.float64)
    channels = null.transform_array(values)
    columns: list[np.ndarray] = []
    by_metric = {
        metric: [] for metric in (*ROBUST_METRICS, *WAVELET_METRICS)
    }
    for window in ROBUST_WINDOWS:
        score = np.zeros((values.size, len(ROBUST_METRICS)), dtype=np.float64)
        sums = rolling_sum(channels, window)
        if sums.shape[0]:
            score[window - 1 :] = null.score_robust(window, _robust_raw(sums, window))
        for index, metric in enumerate(ROBUST_METRICS):
            columns.append(score[:, index])
            by_metric[metric].append(score[:, index])

    historical_z = (np.asarray(historical, dtype=np.float64) - null.median) / null.scale
    online_z = (values - null.median) / null.scale
    details, _ = _wavelet(np.concatenate((historical_z, online_z)))
    for window in WAVELET_WINDOWS:
        score = np.zeros((values.size, WAVELET_LEVELS), dtype=np.float64)
        raw_columns = []
        for detail in details:
            online_detail = detail[-values.size :]
            raw_columns.append(
                np.log(
                    np.maximum(
                        rolling_sum(online_detail * online_detail, window) / window,
                        1e-12,
                    )
                )
            )
        if raw_columns[0].size:
            raw = np.column_stack(raw_columns)
            score[window - 1 :] = null.score_wavelet(window, raw)
        for level, metric in enumerate(WAVELET_METRICS):
            columns.append(score[:, level])
            by_metric[metric].append(score[:, level])
    columns.extend(
        aggregate_maxima(by_metric, (*ROBUST_METRICS, *WAVELET_METRICS))
    )
    result = np.column_stack(columns)
    if result.shape != (values.size, N_FEATURES):
        raise AssertionError("detectors66 emitted an invalid feature shape")
    return result


class StreamDetector66:
    """Incremental robust moments and causal stationary Symlet filters."""

    def __init__(self, historical: np.ndarray):
        self.null = HistoryNull66(historical)
        self.time = 0
        self.robust_ring = np.zeros((MAX_WINDOW, 3 * len(IQR_MULTIPLIERS)))
        self.robust_sums = {
            window: np.zeros(3 * len(IQR_MULTIPLIERS)) for window in ROBUST_WINDOWS
        }
        self.wavelet_sequences = [list(values) for values in self.null.wavelet_tails]
        self.wavelet_ring = np.zeros((MAX_WINDOW, WAVELET_LEVELS))
        self.wavelet_sums = {
            window: np.zeros(WAVELET_LEVELS) for window in WAVELET_WINDOWS
        }

    def _wavelet_step(self, z: float) -> np.ndarray:
        input_value = z
        detail = np.empty(WAVELET_LEVELS, dtype=np.float64)
        for level, sequence in enumerate(self.wavelet_sequences):
            sequence.append(input_value)
            dilation = 1 << level
            keep = (SYM2_LOW.size - 1) * dilation + 1
            if len(sequence) > keep:
                del sequence[:-keep]
            low = high = 0.0
            for index in range(SYM2_LOW.size):
                source = sequence[-1 - index * dilation]
                low += SYM2_LOW[index] * source
                high += SYM2_HIGH[index] * source
            detail[level] = high
            input_value = low
        return detail

    def step(self, value: float) -> np.ndarray:
        time = self.time
        count = time + 1
        position = time % MAX_WINDOW
        channels = self.null.transform_array(np.asarray([value]))[0]
        for window, total in self.robust_sums.items():
            total += channels
            if time >= window:
                total -= self.robust_ring[(time - window) % MAX_WINDOW]
        self.robust_ring[position] = channels

        z = (float(value) - self.null.median) / self.null.scale
        wave_square = self._wavelet_step(z) ** 2
        for window, total in self.wavelet_sums.items():
            total += wave_square
            if time >= window:
                total -= self.wavelet_ring[(time - window) % MAX_WINDOW]
        self.wavelet_ring[position] = wave_square

        output: list[float] = []
        by_metric = {
            metric: [] for metric in (*ROBUST_METRICS, *WAVELET_METRICS)
        }
        for window in ROBUST_WINDOWS:
            if count < window:
                score = np.zeros(len(ROBUST_METRICS), dtype=np.float64)
            else:
                raw = _robust_raw(self.robust_sums[window][None, :], window)[0]
                score = self.null.score_robust(window, raw)
            output.extend(float(item) for item in score)
            for metric, item in zip(ROBUST_METRICS, score, strict=True):
                by_metric[metric].append(float(item))
        for window in WAVELET_WINDOWS:
            if count < window:
                score = np.zeros(WAVELET_LEVELS, dtype=np.float64)
            else:
                raw = np.log(
                    np.maximum(self.wavelet_sums[window] / window, 1e-12)
                )
                score = self.null.score_wavelet(window, raw)
            output.extend(float(item) for item in score)
            for metric, item in zip(WAVELET_METRICS, score, strict=True):
                by_metric[metric].append(float(item))
        output.extend(
            last_maxima(by_metric, (*ROBUST_METRICS, *WAVELET_METRICS))
        )
        self.time = count
        result = np.asarray(output, dtype=np.float64)
        if result.size != N_FEATURES:
            raise AssertionError("detectors66 streaming contract changed")
        return result


if __name__ == "__main__":
    rng = np.random.default_rng(6601)
    history = rng.standard_t(5, 900)
    online = rng.standard_t(5, 180)
    online[93:] = 1.8 * online[93:] - 0.2
    offline = compute_features_offline(history, online)
    detector = StreamDetector66(history)
    streamed = np.vstack([detector.step(value) for value in online])
    error = float(np.max(np.abs(offline - streamed)))
    assert error < 1e-9 and np.all(np.isfinite(offline))
    print(f"OK: detectors66 stream == offline; {N_FEATURES} features; error={error:.3g}")
