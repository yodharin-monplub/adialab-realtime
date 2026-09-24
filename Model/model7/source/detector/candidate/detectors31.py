"""Empirical-null suffix scores for learned marginal break templates.

Each observation is mapped through the series' break-free historical ECDF
and then into a fixed quantile bin.  Globally learned template vectors assign
a signed value to every bin.  For every template and suffix horizon, the
online projected sum is converted to a normal score using rolling sums from
that same series' history.  Thus global supervision supplies only directions
of plausible change; all score calibration remains series-local and legal.

Templates are fitted by the training/CV driver and passed explicitly.  Both
offline and streaming implementations use identical binning, rounding, and
empirical-rank conventions.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.special import ndtri


N_BINS = 24
WINDOWS = (4, 8, 16, 32, 64, 128, 256, 512)
N_PER_TEMPLATE = len(WINDOWS) + 2
ROUND_DECIMALS = 8


def feature_names(n_templates):
    names = []
    for template in range(int(n_templates)):
        names.extend(
            f"tmpl{template}_w{window}" for window in WINDOWS)
        names.extend((f"tmpl{template}_maxabs",
                      f"tmpl{template}_signedmax"))
    return names


def _normal_scores(reference, values):
    values = np.round(np.asarray(values, dtype=np.float64), ROUND_DECIMALS)
    left = np.searchsorted(reference, values, side="left")
    right = np.searchsorted(reference, values, side="right")
    probability = (left + right + 1.0) / (2.0 * (reference.size + 1.0))
    return ndtri(probability)


class TemplateSuffixNull:
    def __init__(self, x_hist, templates):
        history = np.asarray(x_hist, dtype=np.float64)
        templates = np.asarray(templates, dtype=np.float64)
        if templates.ndim != 2 or templates.shape[1] != N_BINS:
            raise ValueError(
                f"templates must have shape (k, {N_BINS}), got "
                f"{templates.shape}")
        self.templates = templates
        self.n_templates = templates.shape[0]
        self.sorted_hist = np.sort(history)
        self.n_hist = history.size
        hist_bins = self.bins(history)
        projected = templates[:, hist_bins].T
        self.center = projected.mean(axis=0)
        projected = projected - self.center[None, :]
        cumulative = np.vstack((
            np.zeros((1, self.n_templates), dtype=np.float64),
            np.cumsum(projected, axis=0),
        ))
        self.reference = {}
        for window in WINDOWS:
            sums = cumulative[window:] - cumulative[:-window]
            for template in range(self.n_templates):
                self.reference[(template, window)] = np.sort(np.round(
                    sums[:, template], ROUND_DECIMALS))

    def bins(self, values):
        values = np.asarray(values, dtype=np.float64)
        left = np.searchsorted(self.sorted_hist, values, side="left")
        right = np.searchsorted(self.sorted_hist, values, side="right")
        probability = (left + right) / (2.0 * self.n_hist)
        return np.minimum((probability * N_BINS).astype(np.int64),
                          N_BINS - 1)

    def scalar_bin(self, value):
        left = self.sorted_hist.searchsorted(float(value), side="left")
        right = self.sorted_hist.searchsorted(float(value), side="right")
        probability = (left + right) / (2.0 * self.n_hist)
        return min(int(probability * N_BINS), N_BINS - 1)

    def probability(self, template, window, value):
        reference = self.reference[(template, window)]
        rounded = round(float(value), ROUND_DECIMALS)
        left = reference.searchsorted(rounded, side="left")
        right = reference.searchsorted(rounded, side="right")
        return float((left + right + 1.0) /
                     (2.0 * (reference.size + 1.0)))


def extra_features(x_hist, x_online, templates):
    calibration = TemplateSuffixNull(x_hist, templates)
    online = np.asarray(x_online, dtype=np.float64)
    steps = online.size
    bins = calibration.bins(online)
    projected = (calibration.templates[:, bins].T
                 - calibration.center[None, :])
    cumulative = np.vstack((
        np.zeros((1, calibration.n_templates), dtype=np.float64),
        np.cumsum(projected, axis=0),
    ))
    output = np.zeros(
        (steps, calibration.n_templates * N_PER_TEMPLATE),
        dtype=np.float64)
    for template in range(calibration.n_templates):
        template_scores = []
        column = template * N_PER_TEMPLATE
        for window_index, window in enumerate(WINDOWS):
            scores = np.zeros(steps, dtype=np.float64)
            if steps >= window:
                sums = cumulative[window:, template] - cumulative[
                    :-window, template]
                scores[window - 1:] = _normal_scores(
                    calibration.reference[(template, window)], sums)
            output[:, column + window_index] = scores
            template_scores.append(scores)
        matrix = np.column_stack(template_scores)
        best = np.argmax(np.abs(matrix), axis=1)
        output[:, column + len(WINDOWS)] = np.max(np.abs(matrix), axis=1)
        output[:, column + len(WINDOWS) + 1] = matrix[
            np.arange(steps), best]
    return output


class StreamDetector31:
    def __init__(self, x_hist, templates):
        self.calibration = TemplateSuffixNull(x_hist, templates)
        self.t = 0
        self.ring = np.zeros(
            (max(WINDOWS), self.calibration.n_templates),
            dtype=np.float64)
        self.window_sum = np.zeros(
            (self.calibration.n_templates, len(WINDOWS)),
            dtype=np.float64)

    def step(self, x):
        calibration = self.calibration
        current_bin = calibration.scalar_bin(x)
        increment = (calibration.templates[:, current_bin]
                     - calibration.center)
        n = self.t + 1
        position = self.t % max(WINDOWS)
        probabilities = np.full(
            (calibration.n_templates, len(WINDOWS)), 0.5,
            dtype=np.float64)
        for window_index, window in enumerate(WINDOWS):
            self.window_sum[:, window_index] += increment
            if self.t >= window:
                self.window_sum[:, window_index] -= self.ring[
                    (self.t - window) % max(WINDOWS)]
            if n >= window:
                for template in range(calibration.n_templates):
                    probabilities[template, window_index] = (
                        calibration.probability(
                            template, window,
                            self.window_sum[template, window_index]))
        score_matrix = ndtri(probabilities)
        output = np.empty(
            calibration.n_templates * N_PER_TEMPLATE, dtype=np.float64)
        for template in range(calibration.n_templates):
            scores = score_matrix[template]
            column = template * N_PER_TEMPLATE
            output[column:column + len(WINDOWS)] = scores
            best = int(np.argmax(np.abs(scores)))
            output[column + len(WINDOWS)] = float(np.max(np.abs(scores)))
            output[column + len(WINDOWS) + 1] = scores[best]
        self.ring[position] = increment
        self.t = n
        return output


if __name__ == "__main__":
    rng = np.random.default_rng(31)
    templates = rng.normal(size=(8, N_BINS))
    templates -= templates.mean(axis=1, keepdims=True)
    templates /= np.maximum(
        np.linalg.norm(templates, axis=1, keepdims=True), 1e-12)
    for trial in range(10):
        n_hist = int(rng.integers(1000, 3500))
        steps = int(rng.integers(10, 800))
        coefficient = rng.uniform(-0.8, 0.8)
        noise = rng.standard_t(rng.uniform(3.0, 15.0), n_hist + steps)
        values = np.empty(n_hist + steps)
        values[0] = noise[0]
        for index in range(1, values.size):
            values[index] = coefficient * values[index - 1] + noise[index]
        history, online = values[:n_hist], values[n_hist:].copy()
        if trial % 2 == 0:
            change = steps // 2
            online[change:] = np.sign(online[change:]) * np.sqrt(
                np.abs(online[change:]))
        offline = extra_features(history, online, templates)
        detector = StreamDetector31(history, templates)
        streamed = np.vstack([detector.step(value) for value in online])
        error = float(np.max(np.abs(offline - streamed)))
        assert offline.shape == (steps, len(feature_names(len(templates))))
        assert error < 1e-9, (trial, error)
    detector = StreamDetector31(rng.normal(size=3000), templates)
    started = time.perf_counter()
    for value in rng.normal(size=1000):
        detector.step(value)
    elapsed = (time.perf_counter() - started) / 1000
    print(f"OK: detectors31 stream == offline; "
          f"{len(feature_names(len(templates)))} features; "
          f"{elapsed * 1e6:.0f} us/step")
