"""Empirical-null calibration of detectors3 global CUSUM scans.

detectors3 normalizes its prefix, draw-up, and draw-down scans by each
series' one-step historical null standard deviation.  That does not fully
standardize the scan extrema when increments are serially dependent or
heavy-tailed.  Here, overlapping break-free history blocks provide the null
distribution of those three scan statistics at geometric horizons.

At online time n, the causal global scans are mapped through the largest
calibrated horizon no longer than n (capped at 512).  Each aspect emits the
empirical-normal up and down scores plus maximum and signed-maximum scan
evidence.  Prefix sums are omitted because detectors22 already empirically
calibrates that geometry.  The history is the only calibration source.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from scipy.special import ndtri

import detectors3 as d3
import detectors15 as d15


WINDOWS = d3.WINDOWS
N_ASPECTS = len(d3.ASPECTS)
N_PER_ASPECT = 4
N_EXTRA = N_ASPECTS * N_PER_ASPECT


def feature_names():
    names = []
    for aspect in d3.ASPECTS:
        names.extend((
            f"empcusum_{aspect}_up",
            f"empcusum_{aspect}_down",
            f"empcusum_{aspect}_max",
            f"empcusum_{aspect}_signedmax",
        ))
    return names


def _normal_score(reference, value):
    rounded = np.round(np.asarray(value, dtype=np.float64), 8)
    left = np.searchsorted(reference, rounded, side="left")
    right = np.searchsorted(reference, rounded, side="right")
    probability = (left + right + 1.0) / (
        2.0 * (reference.size + 1.0)
    )
    return ndtri(probability)


def _block_scans(increments, window):
    """Prefix endpoint and Page draw-up/down for all overlapping blocks."""
    values = np.asarray(increments, dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    size = int(window) + 1
    count = cumulative.size - int(window)
    # With this origin, output[j] filters cumulative[j:j + size].
    origin = -(size // 2)
    minimum = minimum_filter1d(
        cumulative, size=size, origin=origin,
        mode="constant", cval=np.inf,
    )[:count]
    maximum = maximum_filter1d(
        cumulative, size=size, origin=origin,
        mode="constant", cval=-np.inf,
    )[:count]
    start = cumulative[:count]
    end = cumulative[window:window + count]
    root = np.sqrt(float(window))
    prefix = (end - start) / root
    up = (end - minimum) / root
    down = (maximum - end) / root
    return prefix, up, down


class EmpiricalCusumNull:
    def __init__(self, x_hist):
        hs, hist_inc, _ = d15._hist_and_online_increments(x_hist)
        self.reference = {}
        for aspect_index, aspect in enumerate(d3.ASPECTS):
            increments = (
                np.asarray(hist_inc[aspect], dtype=np.float64) / hs.s[aspect]
            )
            for window_index, window in enumerate(WINDOWS):
                _, up, down = _block_scans(increments, window)
                for statistic, values in enumerate((up, down)):
                    self.reference[(aspect_index, window_index, statistic)] = (
                        np.sort(np.round(values, 8))
                    )

    @staticmethod
    def window_index(n):
        return max(
            0,
            min(
                len(WINDOWS) - 1,
                int(np.searchsorted(WINDOWS, int(n), side="right") - 1),
            ),
        )

    def transform(self, n, scan_matrix):
        output = np.zeros(N_EXTRA, dtype=np.float64)
        if n < WINDOWS[0]:
            return output
        window_index = self.window_index(n)
        probabilities = np.empty((N_ASPECTS, 2), dtype=np.float64)
        for aspect_index in range(N_ASPECTS):
            for statistic in range(2):
                reference = self.reference[
                    (aspect_index, window_index, statistic)
                ]
                rounded = round(
                    float(scan_matrix[aspect_index, statistic]), 8
                )
                left = reference.searchsorted(rounded, side="left")
                right = reference.searchsorted(rounded, side="right")
                probabilities[aspect_index, statistic] = (
                    (left + right + 1.0) /
                    (2.0 * (reference.size + 1.0))
                )
        score_matrix = ndtri(probabilities)
        for aspect_index in range(N_ASPECTS):
            scores = score_matrix[aspect_index]
            up, down = scores
            column = aspect_index * N_PER_ASPECT
            output[column:column + 2] = scores
            output[column + 2] = max(up, down)
            output[column + 3] = up if up >= down else -down
        return output


def _scan_columns(base_features):
    base = np.asarray(base_features, dtype=np.float64)
    block = 3 + len(d3.WINDOWS)
    columns = []
    for aspect_index in range(N_ASPECTS):
        start = aspect_index * block
        columns.extend((start + 1, start + 2))
    return base[:, columns].reshape(base.shape[0], N_ASPECTS, 2)


def extra_features(x_hist, x_online):
    calibration = EmpiricalCusumNull(x_hist)
    base = d3.compute_features_offline(x_hist, x_online)
    scans = _scan_columns(base)
    return np.vstack([
        calibration.transform(index + 1, scans[index])
        for index in range(scans.shape[0])
    ])


class StreamDetector29:
    def __init__(self, x_hist, inner=None):
        self.inner = d3.StreamDetector3(x_hist) if inner is None else inner
        self.owns_inner = inner is None
        self.calibration = EmpiricalCusumNull(x_hist)

    def _from_current_inner(self, base=None):
        if base is None:
            block = 3 + len(d3.WINDOWS)
            scans = np.empty((N_ASPECTS, 2), dtype=np.float64)
            root = np.sqrt(float(self.inner.t))
            for aspect_index, aspect in enumerate(d3.ASPECTS):
                scale = self.inner.hs.s[aspect] * root
                current = self.inner.C[aspect]
                scans[aspect_index, 0] = (
                    current - self.inner.Cmin[aspect]
                ) / scale
                scans[aspect_index, 1] = (
                    self.inner.Cmax[aspect] - current
                ) / scale
        else:
            block = 3 + len(d3.WINDOWS)
            scans = np.vstack([
                base[index * block + 1:index * block + 3]
                for index in range(N_ASPECTS)
            ])
        return self.calibration.transform(self.inner.t, scans)

    def step(self, x):
        if not self.owns_inner:
            raise RuntimeError("use step_from_inner with a shared detector")
        base = self.inner.step(x)
        return self._from_current_inner(base)

    def step_from_inner(self):
        return self._from_current_inner()


if __name__ == "__main__":
    rng = np.random.default_rng(29)
    for trial in range(8):
        n_hist = int(rng.integers(1000, 3000))
        steps = int(rng.integers(10, 700))
        ar = rng.uniform(-0.85, 0.85)
        noise = rng.standard_t(rng.uniform(3.0, 12.0), n_hist + steps)
        values = np.empty(n_hist + steps, dtype=np.float64)
        values[0] = noise[0]
        for index in range(1, values.size):
            values[index] = ar * values[index - 1] + noise[index]
        history, online = values[:n_hist], values[n_hist:].copy()
        if trial % 2 == 0 and steps > 20:
            tau = int(rng.integers(5, steps - 5))
            online[tau:] += rng.uniform(-1.0, 1.0)
            online[tau:] *= rng.uniform(1.1, 1.8)
        offline = extra_features(history, online)
        detector = StreamDetector29(history)
        streamed = np.vstack([detector.step(value) for value in online])
        shared_inner = d3.StreamDetector3(history)
        shared = StreamDetector29(history, inner=shared_inner)
        shared_rows = []
        for value in online:
            shared_inner.step(value)
            shared_rows.append(shared.step_from_inner())
        shared_streamed = np.vstack(shared_rows)
        error = float(np.max(np.abs(offline - streamed)))
        shared_error = float(np.max(np.abs(offline - shared_streamed)))
        assert offline.shape == (steps, N_EXTRA)
        assert error < 1e-9, (trial, error)
        assert shared_error < 1e-9, (trial, shared_error)
    detector = StreamDetector29(rng.normal(size=3000))
    started = time.perf_counter()
    for value in rng.normal(size=1000):
        detector.step(value)
    elapsed = (time.perf_counter() - started) / 1000
    assert len(feature_names()) == N_EXTRA
    print(
        f"OK: detectors29 stream == offline; {N_EXTRA} features; "
        f"{elapsed * 1e6:.0f} us/step"
    )
