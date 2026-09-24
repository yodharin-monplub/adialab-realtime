"""Causal adaptations of previously untried winning-solution features.

The old ADIA challenge compared two complete segments. That is illegal in the
real-time task, so this bank rewrites promising ideas as history-calibrated
sequential features: robust scale/location tests, histogram distances, tail
expected shortfall, in-band runs, Hurst roughness, and causal db4 energies.
Every calibration comes only from pseudo-suffixes of the clean history.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtri


TEST_WINDOWS = (16, 32, 64, 128, 256, 512)
SHORT_WINDOWS = (16, 64, 256)
HURST_WINDOWS = (64, 128, 256, 512)
WAVELET_WINDOWS = (32, 128, 512)
HURST_LAGS = np.asarray((1, 2, 4, 8), dtype=np.int64)
TAIL_ALPHA = (0.01, 0.025, 0.05, 0.10)
IN_BANDS = (0.10, 0.25, 0.50, 1.00, 2.00)
ROBUST_METRICS = ("fvar", "levene", "abswelch", "fligner", "mood")
DISTANCE_METRICS = ("hellinger", "canberra", "braycurtis")
N_BINS = 16
MAX_WINDOW = max(TEST_WINDOWS)
WAVELET_LEVELS = 4
DB4_LOW = np.asarray(
    (-0.010597401785069032, 0.03288301166688520,
     0.030841381835560764, -0.18703481171909309,
     -0.027983769416859854, 0.6308807679298587,
     0.7148465705529154, 0.2303778133088964),
    dtype=np.float64,
)
DB4_HIGH = ((-1.0) ** np.arange(DB4_LOW.size)) * DB4_LOW[::-1]


def _tag(value: float) -> str:
    return str(value).replace(".", "p")


def feature_names() -> list[str]:
    names: list[str] = []
    for window in TEST_WINDOWS:
        names += [f"{metric}_w{window}" for metric in ROBUST_METRICS]
    for window in SHORT_WINDOWS:
        names += [f"{metric}_w{window}" for metric in DISTANCE_METRICS]
    for window in SHORT_WINDOWS:
        for alpha in TAIL_ALPHA:
            names += [f"cvar_lower_a{_tag(alpha)}_w{window}",
                      f"cvar_upper_a{_tag(alpha)}_w{window}"]
    for window in SHORT_WINDOWS:
        names += [f"inband_b{_tag(bound)}_w{window}" for bound in IN_BANDS]
    names += [f"inband_run_b{_tag(bound)}" for bound in IN_BANDS]
    names += [f"hurst_w{window}" for window in HURST_WINDOWS]
    for window in WAVELET_WINDOWS:
        names += [f"db4_detail_l{level}_w{window}"
                  for level in range(1, WAVELET_LEVELS + 1)]
    for metric in ROBUST_METRICS:
        names += [f"{metric}_maxabs", f"{metric}_signedmax"]
    names += [f"{metric}_maxabs" for metric in DISTANCE_METRICS]
    for alpha in TAIL_ALPHA:
        names += [f"cvar_lower_a{_tag(alpha)}_maxabs",
                  f"cvar_upper_a{_tag(alpha)}_maxabs"]
    names += [f"inband_b{_tag(bound)}_maxabs" for bound in IN_BANDS]
    names += [f"db4_detail_l{level}_maxabs"
              for level in range(1, WAVELET_LEVELS + 1)]
    return names


N_FEATURES = len(feature_names())


def _rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[0] < window:
        return np.empty((0, *values.shape[1:]))
    cumulative = np.concatenate(
        (np.zeros((1, *values.shape[1:])), np.cumsum(values, axis=0)), axis=0
    )
    return cumulative[window:] - cumulative[:-window]


def _causal_filter(values: np.ndarray, coefficients: np.ndarray,
                   dilation: int) -> np.ndarray:
    output = np.zeros_like(values, dtype=np.float64)
    for index, coefficient in enumerate(coefficients):
        shift = index * dilation
        if shift == 0:
            output += coefficient * values
        elif shift < values.size:
            output[shift:] += coefficient * values[:-shift]
    return output


def _wavelet(values: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    approximation = np.asarray(values, dtype=np.float64)
    details, inputs = [], []
    for level in range(WAVELET_LEVELS):
        inputs.append(approximation)
        dilation = 1 << level
        details.append(_causal_filter(approximation, DB4_HIGH, dilation))
        approximation = _causal_filter(approximation, DB4_LOW, dilation)
    return details, inputs


def _hurst(values: np.ndarray, window: int) -> np.ndarray:
    if values.size < window:
        return np.empty(0)
    endpoints = np.arange(window - 1, values.size)
    energies = []
    for lag in HURST_LAGS:
        difference = np.zeros(values.size)
        difference[lag:] = (values[lag:] - values[:-lag]) ** 2
        cumulative = np.r_[0.0, np.cumsum(difference)]
        left = endpoints - window + 1 + lag
        energies.append(
            (cumulative[endpoints + 1] - cumulative[left]) / (window - lag)
        )
    log_lag = np.log(HURST_LAGS.astype(float))
    centered = log_lag - log_lag.mean()
    return 0.5 * (np.log(np.maximum(np.column_stack(energies), 1e-12))
                  @ centered) / float(centered @ centered)


def _distances(counts: np.ndarray, reference: np.ndarray) -> np.ndarray:
    probability = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    denominator = probability + reference
    hellinger = np.sqrt(np.sum(
        (np.sqrt(probability) - np.sqrt(reference)) ** 2, axis=1
    )) / np.sqrt(2.0)
    canberra = np.sum(np.divide(
        np.abs(probability - reference), denominator,
        out=np.zeros_like(probability), where=denominator > 1e-12,
    ), axis=1)
    bray = np.sum(np.abs(probability - reference), axis=1) / np.maximum(
        np.sum(denominator, axis=1), 1e-12
    )
    return np.column_stack((hellinger, canberra, bray))


def _runs(indicator: np.ndarray) -> np.ndarray:
    result = np.empty(indicator.size)
    current = 0
    for index, inside in enumerate(indicator):
        current = current + 1 if inside else 0
        result[index] = np.log1p(current)
    return result


class HistoryNull60:
    """Transform one clean history and fit only within-history calibrations."""

    def __init__(self, historical: np.ndarray):
        history = np.asarray(historical, dtype=np.float64)
        if history.ndim != 1 or history.size < MAX_WINDOW + 16:
            raise ValueError("detectors60 needs at least 528 history values")
        self.n = history.size
        self.mean = float(history.mean())
        self.scale = max(float(history.std()), 1e-12)
        z = (history - self.mean) / self.scale
        self.median = float(np.median(z))
        abs_median = np.abs(z - self.median)
        self.sorted_abs = np.sort(abs_median)
        left = np.searchsorted(self.sorted_abs, abs_median, side="left")
        right = np.searchsorted(self.sorted_abs, abs_median, side="right")
        rank = (left + right + 1.0) / (2.0 * (self.n + 1.0))
        fligner = ndtri(np.clip(0.5 + rank / 2.0, 1e-9, 1 - 1e-9))

        self.edges = np.quantile(history, np.linspace(0, 1, N_BINS + 1))
        bins = np.searchsorted(self.edges[1:-1], history, side="right")
        one_hot = np.eye(N_BINS)[bins]
        self.bin_probability = one_hot.mean(axis=0)
        self.lower = np.quantile(z, TAIL_ALPHA)
        self.upper = np.quantile(z, 1 - np.asarray(TAIL_ALPHA))
        tail = np.column_stack([
            channel for lower, upper in zip(self.lower, self.upper)
            for channel in (np.maximum(lower - z, 0), np.maximum(z - upper, 0))
        ])
        occupancy = np.column_stack([np.abs(z) <= bound for bound in IN_BANDS])
        self.calibration: dict[str, tuple[float, float]] = {}

        base = np.column_stack((z, z * z, abs_median, np.abs(z), fligner,
                                z > self.median))
        for window in TEST_WINDOWS:
            sums = _rolling_sum(base, window)
            mean = sums[:, 0] / window
            raw = np.column_stack((
                np.log(np.maximum(sums[:, 1] / window - mean * mean, 1e-12)),
                sums[:, 2:] / window,
            ))
            for index, metric in enumerate(ROBUST_METRICS):
                self._fit(f"{metric}_w{window}", raw[:, index])
        for window in SHORT_WINDOWS:
            raw = _distances(_rolling_sum(one_hot, window), self.bin_probability)
            for index, metric in enumerate(DISTANCE_METRICS):
                self._fit(f"{metric}_w{window}", raw[:, index])
            tail_mean = _rolling_sum(tail, window) / window
            for index, alpha in enumerate(TAIL_ALPHA):
                self._fit(f"cvar_lower_a{_tag(alpha)}_w{window}",
                          tail_mean[:, 2 * index])
                self._fit(f"cvar_upper_a{_tag(alpha)}_w{window}",
                          tail_mean[:, 2 * index + 1])
            occupancy_mean = _rolling_sum(occupancy, window) / window
            for index, bound in enumerate(IN_BANDS):
                self._fit(f"inband_b{_tag(bound)}_w{window}",
                          occupancy_mean[:, index])
        for index, bound in enumerate(IN_BANDS):
            self._fit(f"inband_run_b{_tag(bound)}", _runs(occupancy[:, index]))
        for window in HURST_WINDOWS:
            self._fit(f"hurst_w{window}", _hurst(z, window))

        details, inputs = _wavelet(z)
        self.wavelet_tails = []
        for level, values in enumerate(inputs):
            keep = 7 * (1 << level) + 1
            self.wavelet_tails.append(values[-keep:].tolist())
        for window in WAVELET_WINDOWS:
            for level, detail in enumerate(details, 1):
                raw = np.log(np.maximum(
                    _rolling_sum(detail * detail, window) / window, 1e-12
                ))
                self._fit(f"db4_detail_l{level}_w{window}", raw)

    def _fit(self, name: str, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        center = float(values.mean()) if values.size else 0.0
        scale = float(values.std()) if values.size else 1.0
        self.calibration[name] = center, scale if scale > 1e-7 else 1.0

    def score(self, name: str, value: float) -> float:
        center, scale = self.calibration[name]
        return float(np.clip((value - center) / scale, -12, 12))

    def score_array(self, name: str, values: np.ndarray) -> np.ndarray:
        center, scale = self.calibration[name]
        return np.clip((np.asarray(values) - center) / scale, -12, 12)

    def transform(self, value: float) -> tuple[np.ndarray, int]:
        z = (float(value) - self.mean) / self.scale
        absolute = abs(z - self.median)
        left = self.sorted_abs.searchsorted(absolute, side="left")
        right = self.sorted_abs.searchsorted(absolute, side="right")
        rank = (left + right + 1.0) / (2.0 * (self.n + 1.0))
        fligner = ndtri(np.clip(0.5 + rank / 2.0, 1e-9, 1 - 1e-9))
        tail = [channel for lower, upper in zip(self.lower, self.upper)
                for channel in (max(lower - z, 0), max(z - upper, 0))]
        occupancy = [float(abs(z) <= bound) for bound in IN_BANDS]
        channels = np.asarray([z, z * z, absolute, abs(z), fligner,
                               float(z > self.median), *tail, *occupancy])
        bin_index = int(np.searchsorted(self.edges[1:-1], value, side="right"))
        return channels, bin_index


class StreamDetector60:
    """Incremental detector; each call consumes exactly one online value."""

    def __init__(self, historical: np.ndarray):
        self.null = HistoryNull60(historical)
        self.t = 0
        self.channel_count = 6 + 2 * len(TAIL_ALPHA) + len(IN_BANDS)
        self.ring = np.zeros((MAX_WINDOW, self.channel_count))
        self.sums = {window: np.zeros(self.channel_count)
                     for window in TEST_WINDOWS}
        self.bin_ring = np.zeros(MAX_WINDOW, dtype=np.int16)
        self.bin_counts = {window: np.zeros(N_BINS) for window in SHORT_WINDOWS}
        self.run = np.zeros(len(IN_BANDS), dtype=np.int64)
        self.z_values: list[float] = []
        self.diff_cumulative = {int(lag): [0.0] for lag in HURST_LAGS}
        self.wavelet_sequences = [list(values) for values in self.null.wavelet_tails]
        self.wave_ring = np.zeros((MAX_WINDOW, WAVELET_LEVELS))
        self.wave_sums = {window: np.zeros(WAVELET_LEVELS)
                          for window in WAVELET_WINDOWS}

    def _wavelet_step(self, z: float) -> np.ndarray:
        input_value = z
        detail = np.empty(WAVELET_LEVELS)
        for level, sequence in enumerate(self.wavelet_sequences):
            sequence.append(input_value)
            dilation = 1 << level
            keep = 7 * dilation + 1
            if len(sequence) > keep:
                del sequence[:-keep]
            low = high = 0.0
            for index in range(DB4_LOW.size):
                source = sequence[-1 - index * dilation]
                low += DB4_LOW[index] * source
                high += DB4_HIGH[index] * source
            detail[level] = high
            input_value = low
        return detail

    def step(self, value: float) -> np.ndarray:
        t, n, position = self.t, self.t + 1, self.t % MAX_WINDOW
        channels, bin_index = self.null.transform(float(value))
        z = float(channels[0])
        for window, total in self.sums.items():
            total += channels
            if t >= window:
                total -= self.ring[(t - window) % MAX_WINDOW]
        self.ring[position] = channels
        for window, counts in self.bin_counts.items():
            counts[bin_index] += 1
            if t >= window:
                counts[int(self.bin_ring[(t - window) % MAX_WINDOW])] -= 1
        self.bin_ring[position] = bin_index
        inside = channels[-len(IN_BANDS):] > 0.5
        self.run = np.where(inside, self.run + 1, 0)

        self.z_values.append(z)
        for lag in HURST_LAGS:
            lag = int(lag)
            difference = (z - self.z_values[t - lag]) ** 2 if t >= lag else 0.0
            cumulative = self.diff_cumulative[lag]
            cumulative.append(cumulative[-1] + difference)
        wave_square = self._wavelet_step(z) ** 2
        for window, total in self.wave_sums.items():
            total += wave_square
            if t >= window:
                total -= self.wave_ring[(t - window) % MAX_WINDOW]
        self.wave_ring[position] = wave_square

        output: list[float] = []
        robust = {metric: [] for metric in ROBUST_METRICS}
        for window in TEST_WINDOWS:
            if n < window:
                scores = [0.0] * len(ROBUST_METRICS)
            else:
                means = self.sums[window] / window
                raw = (np.log(max(means[1] - means[0] ** 2, 1e-12)),
                       means[2], means[3], means[4], means[5])
                scores = [self.null.score(f"{metric}_w{window}", raw_value)
                          for metric, raw_value in zip(ROBUST_METRICS, raw)]
            output += scores
            for metric, score in zip(ROBUST_METRICS, scores):
                robust[metric].append(score)

        distance = {metric: [] for metric in DISTANCE_METRICS}
        for window in SHORT_WINDOWS:
            raw = (_distances(self.bin_counts[window][None],
                              self.null.bin_probability)[0]
                   if n >= window else np.zeros(len(DISTANCE_METRICS)))
            scores = [self.null.score(f"{metric}_w{window}", raw_value)
                      if n >= window else 0.0
                      for metric, raw_value in zip(DISTANCE_METRICS, raw)]
            output += scores
            for metric, score in zip(DISTANCE_METRICS, scores):
                distance[metric].append(score)

        tails = {(side, index): [] for index in range(len(TAIL_ALPHA))
                 for side in ("lower", "upper")}
        occupancy = {index: [] for index in range(len(IN_BANDS))}
        tail_start = 6
        occupancy_start = tail_start + 2 * len(TAIL_ALPHA)
        for window in SHORT_WINDOWS:
            means = self.sums[window] / window if n >= window else None
            for index, alpha in enumerate(TAIL_ALPHA):
                for offset, side in enumerate(("lower", "upper")):
                    score = (self.null.score(
                        f"cvar_{side}_a{_tag(alpha)}_w{window}",
                        means[tail_start + 2 * index + offset])
                        if means is not None else 0.0)
                    output.append(score)
                    tails[(side, index)].append(score)
        for window in SHORT_WINDOWS:
            means = self.sums[window] / window if n >= window else None
            for index, bound in enumerate(IN_BANDS):
                score = (self.null.score(
                    f"inband_b{_tag(bound)}_w{window}",
                    means[occupancy_start + index])
                    if means is not None else 0.0)
                output.append(score)
                occupancy[index].append(score)
        for index, bound in enumerate(IN_BANDS):
            output.append(self.null.score(f"inband_run_b{_tag(bound)}",
                                          np.log1p(float(self.run[index]))))

        log_lag = np.log(HURST_LAGS.astype(float))
        centered = log_lag - log_lag.mean()
        for window in HURST_WINDOWS:
            if n < window:
                output.append(0.0)
                continue
            energies = []
            for lag in HURST_LAGS:
                lag = int(lag)
                cumulative = self.diff_cumulative[lag]
                left = t - window + 1 + lag
                energies.append((cumulative[t + 1] - cumulative[left]) /
                                (window - lag))
            value = 0.5 * float(np.log(np.maximum(energies, 1e-12)) @ centered)
            value /= float(centered @ centered)
            output.append(self.null.score(f"hurst_w{window}", value))

        wavelet = {level: [] for level in range(WAVELET_LEVELS)}
        for window in WAVELET_WINDOWS:
            for level in range(WAVELET_LEVELS):
                score = (self.null.score(
                    f"db4_detail_l{level + 1}_w{window}",
                    np.log(max(self.wave_sums[window][level] / window, 1e-12)))
                    if n >= window else 0.0)
                output.append(score)
                wavelet[level].append(score)

        for metric in ROBUST_METRICS:
            values = np.asarray(robust[metric])
            index = int(np.argmax(np.abs(values)))
            output += [float(abs(values[index])), float(values[index])]
        output += [float(np.max(np.abs(distance[metric])))
                   for metric in DISTANCE_METRICS]
        for index in range(len(TAIL_ALPHA)):
            for side in ("lower", "upper"):
                output.append(float(np.max(np.abs(tails[(side, index)]))))
        output += [float(np.max(np.abs(occupancy[index])))
                   for index in range(len(IN_BANDS))]
        output += [float(np.max(np.abs(wavelet[level])))
                   for level in range(WAVELET_LEVELS)]
        self.t = n
        result = np.asarray(output)
        if result.size != N_FEATURES:
            raise AssertionError(f"emitted {result.size}, expected {N_FEATURES}")
        return result


def compute_features_offline(historical: np.ndarray,
                             online: np.ndarray) -> np.ndarray:
    """Vectorized causal twin of :class:`StreamDetector60`."""

    null = HistoryNull60(historical)
    values = np.asarray(online, dtype=float)
    steps = values.size
    if not steps:
        return np.empty((0, N_FEATURES))
    z = (values - null.mean) / null.scale
    absolute = np.abs(z - null.median)
    left = np.searchsorted(null.sorted_abs, absolute, side="left")
    right = np.searchsorted(null.sorted_abs, absolute, side="right")
    rank = (left + right + 1.0) / (2.0 * (null.n + 1.0))
    fligner = ndtri(np.clip(0.5 + rank / 2.0, 1e-9, 1 - 1e-9))
    tail = np.column_stack([
        channel for lower, upper in zip(null.lower, null.upper)
        for channel in (np.maximum(lower - z, 0), np.maximum(z - upper, 0))
    ])
    occupancy_raw = np.column_stack([np.abs(z) <= bound for bound in IN_BANDS])
    channels = np.column_stack((z, z * z, absolute, np.abs(z), fligner,
                                z > null.median, tail, occupancy_raw))
    bins = np.searchsorted(null.edges[1:-1], values, side="right")
    one_hot = np.eye(N_BINS)[bins]

    def aligned(name: str, raw: np.ndarray, window: int) -> np.ndarray:
        output = np.zeros(steps)
        if raw.size:
            output[window - 1:] = null.score_array(name, raw)
        return output

    columns: list[np.ndarray] = []
    robust = {metric: [] for metric in ROBUST_METRICS}
    for window in TEST_WINDOWS:
        sums = _rolling_sum(channels[:, :6], window)
        if sums.size:
            mean = sums[:, 0] / window
            raw = np.column_stack((
                np.log(np.maximum(sums[:, 1] / window - mean * mean, 1e-12)),
                sums[:, 2:] / window,
            ))
        else:
            raw = np.empty((0, len(ROBUST_METRICS)))
        for index, metric in enumerate(ROBUST_METRICS):
            score = aligned(f"{metric}_w{window}", raw[:, index], window)
            columns.append(score)
            robust[metric].append(score)

    distance = {metric: [] for metric in DISTANCE_METRICS}
    for window in SHORT_WINDOWS:
        counts = _rolling_sum(one_hot, window)
        raw = (_distances(counts, null.bin_probability) if counts.size
               else np.empty((0, len(DISTANCE_METRICS))))
        for index, metric in enumerate(DISTANCE_METRICS):
            score = aligned(f"{metric}_w{window}", raw[:, index], window)
            columns.append(score)
            distance[metric].append(score)

    tails = {(side, index): [] for index in range(len(TAIL_ALPHA))
             for side in ("lower", "upper")}
    for window in SHORT_WINDOWS:
        raw = _rolling_sum(tail, window) / window
        for index, alpha in enumerate(TAIL_ALPHA):
            for offset, side in enumerate(("lower", "upper")):
                score = aligned(f"cvar_{side}_a{_tag(alpha)}_w{window}",
                                raw[:, 2 * index + offset], window)
                columns.append(score)
                tails[(side, index)].append(score)

    occupancy = {index: [] for index in range(len(IN_BANDS))}
    for window in SHORT_WINDOWS:
        raw = _rolling_sum(occupancy_raw, window) / window
        for index, bound in enumerate(IN_BANDS):
            score = aligned(f"inband_b{_tag(bound)}_w{window}", raw[:, index],
                            window)
            columns.append(score)
            occupancy[index].append(score)
    for index, bound in enumerate(IN_BANDS):
        columns.append(null.score_array(f"inband_run_b{_tag(bound)}",
                                        _runs(occupancy_raw[:, index])))
    for window in HURST_WINDOWS:
        columns.append(aligned(f"hurst_w{window}", _hurst(z, window), window))

    historical_z = (np.asarray(historical) - null.mean) / null.scale
    details, _ = _wavelet(np.concatenate((historical_z, z)))
    wavelet = {level: [] for level in range(WAVELET_LEVELS)}
    for window in WAVELET_WINDOWS:
        for level in range(WAVELET_LEVELS):
            online_detail = details[level][-steps:]
            raw = np.log(np.maximum(
                _rolling_sum(online_detail * online_detail, window) / window,
                1e-12,
            ))
            score = aligned(f"db4_detail_l{level + 1}_w{window}", raw, window)
            columns.append(score)
            wavelet[level].append(score)

    for metric in ROBUST_METRICS:
        matrix = np.column_stack(robust[metric])
        index = np.argmax(np.abs(matrix), axis=1)
        signed = matrix[np.arange(steps), index]
        columns += [np.abs(signed), signed]
    for metric in DISTANCE_METRICS:
        columns.append(np.max(np.abs(np.column_stack(distance[metric])), axis=1))
    for index in range(len(TAIL_ALPHA)):
        for side in ("lower", "upper"):
            columns.append(np.max(np.abs(np.column_stack(tails[(side, index)])),
                                  axis=1))
    for index in range(len(IN_BANDS)):
        columns.append(np.max(np.abs(np.column_stack(occupancy[index])), axis=1))
    for level in range(WAVELET_LEVELS):
        columns.append(np.max(np.abs(np.column_stack(wavelet[level])), axis=1))
    result = np.column_stack(columns)
    if result.shape != (steps, N_FEATURES):
        raise AssertionError(f"offline shape {result.shape}, expected {(steps, N_FEATURES)}")
    return result
