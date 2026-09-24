"""Causal GARCH forecast-error and volatility-parameter score bank.

The accepted detectors measure unconditional scale and simple volatility
clustering, but do not ask whether a series' *conditional variance law* has
changed.  This bank runs three fixed, complementary ARCH/GARCH filters on
the AR(5)-whitened residual.  It monitors standardized volatility surprises,
tail exceedances, and score-like interactions for ARCH, persistence, and
asymmetric leverage changes.  Every channel is centered and scaled on that
series' guaranteed break-free history before causal scan accumulation.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.signal import lfilter

import detectors3 as d3


FILTERS = ((0.20, 0.00), (0.10, 0.80), (0.03, 0.95))
RAW_NAMES = ("q", "absu", "tail", "alpha_score", "beta_score",
             "leverage_score")
WINDOWS = (8, 32, 128, 512)
N_PER_RAW = 3 + len(WINDOWS)
N_ASPECTS = len(FILTERS) * len(RAW_NAMES)
N_FEATURES = N_ASPECTS * N_PER_RAW
MAX_WINDOW = max(WINDOWS)


def _filter_tag(index, alpha, beta):
    return f"g{index + 1}_a{alpha:g}_b{beta:g}"


def feature_names():
    names = []
    for filter_index, (alpha, beta) in enumerate(FILTERS):
        tag = _filter_tag(filter_index, alpha, beta)
        for raw_name in RAW_NAMES:
            base = f"{tag}_{raw_name}"
            names.extend((f"{base}_prefix", f"{base}_up", f"{base}_down"))
            names.extend(f"{base}_w{window}" for window in WINDOWS)
    return names


def _variance_path(e2, alpha, beta, omega, initial):
    e2 = np.asarray(e2, dtype=np.float64)
    variance = np.empty(e2.size, dtype=np.float64)
    if e2.size == 0:
        return variance
    variance[0] = min(max(float(initial), 1e-4), 100.0)
    if e2.size > 1:
        forcing = omega + alpha * e2[:-1]
        variance[1:] = lfilter(
            [1.0], [1.0, -beta], forcing,
            zi=np.asarray([beta * variance[0]], dtype=np.float64))[0]
        np.clip(variance[1:], 1e-4, 100.0, out=variance[1:])
    return variance


def _variance_continue(e2, alpha, beta, omega, previous_e2,
                       previous_variance):
    """Vector continuation whose first value is the next online forecast."""
    e2 = np.asarray(e2, dtype=np.float64)
    if e2.size == 0:
        return np.empty(0, dtype=np.float64)
    lagged_e2 = np.concatenate(([float(previous_e2)], e2[:-1]))
    forcing = omega + alpha * lagged_e2
    variance = lfilter(
        [1.0], [1.0, -beta], forcing,
        zi=np.asarray([beta * float(previous_variance)], dtype=np.float64))[0]
    np.clip(variance, 1e-4, 100.0, out=variance)
    return variance


def _raw_matrix(e, e2, variance, constants):
    prev_e = np.concatenate(([constants["mean_e"]], e[:-1]))
    prev_e2 = np.concatenate(([constants["mean_e2"]], e2[:-1]))
    prev_variance = np.concatenate(
        ([constants["mean_variance"]], variance[:-1]))
    ratio = np.minimum(e2 / np.maximum(variance, 1e-4), 25.0)
    absu = np.minimum(np.abs(e) / np.sqrt(np.maximum(variance, 1e-4)), 5.0)
    centered_ratio = ratio - constants["mean_ratio"]
    negative_energy = (prev_e < 0.0).astype(np.float64) * prev_e2
    return np.column_stack((
        ratio,
        absu,
        (absu > constants["tail_threshold"]).astype(np.float64),
        centered_ratio * (prev_e2 - constants["mean_prev_e2"]),
        centered_ratio * (prev_variance - constants["mean_prev_variance"]),
        centered_ratio * (negative_energy - constants["mean_negative_energy"]),
    ))


class Hist37:
    def __init__(self, x_hist):
        history = np.asarray(x_hist, dtype=np.float64)
        self.hs = d3.Hist3(history)
        z = (history - self.hs.mu0) / self.hs.sd0
        residual = d3._ar_residuals(z, self.hs.phi, np.zeros(d3.AR_P))
        e = np.clip(residual / self.hs.resid_sd, -10.0, 10.0)
        e2 = np.minimum(e * e, 100.0)
        warmup = min(max(d3.AR_P, 16), max(e.size - 1, 0))
        valid = slice(warmup, None)
        mean_e2 = max(float(np.mean(e2[valid])), 1e-4)
        mean_e = float(np.mean(e[valid]))

        raw_blocks = []
        self.filter_constants = []
        self.last_variance = np.empty(len(FILTERS), dtype=np.float64)
        for filter_index, (alpha, beta) in enumerate(FILTERS):
            omega = max(1.0 - alpha - beta, 0.005) * mean_e2
            variance = _variance_path(
                e2, alpha, beta, omega, mean_e2)
            ratio = np.minimum(e2 / np.maximum(variance, 1e-4), 25.0)
            absu = np.minimum(
                np.abs(e) / np.sqrt(np.maximum(variance, 1e-4)), 5.0)
            prev_e = np.concatenate(([mean_e], e[:-1]))
            prev_e2 = np.concatenate(([mean_e2], e2[:-1]))
            mean_variance = float(np.mean(variance[valid]))
            prev_variance = np.concatenate(
                ([mean_variance], variance[:-1]))
            negative_energy = (prev_e < 0.0).astype(np.float64) * prev_e2
            constants = {
                "alpha": float(alpha),
                "beta": float(beta),
                "omega": float(omega),
                "mean_e": mean_e,
                "mean_e2": mean_e2,
                "mean_variance": mean_variance,
                "mean_ratio": float(np.mean(ratio[valid])),
                "tail_threshold": float(np.quantile(absu[valid], 0.95)),
                "mean_prev_e2": float(np.mean(prev_e2[valid])),
                "mean_prev_variance": float(np.mean(prev_variance[valid])),
                "mean_negative_energy": float(np.mean(negative_energy[valid])),
            }
            raw_blocks.append(_raw_matrix(e, e2, variance, constants))
            self.filter_constants.append(constants)
            self.last_variance[filter_index] = variance[-1]

        raw = np.column_stack(raw_blocks)
        self.raw_mean = raw[valid].mean(axis=0)
        raw_sd = raw[valid].std(axis=0)
        self.raw_sd = np.maximum(raw_sd, 1e-9)
        self.last_e = float(e[-1])
        self.last_e2 = float(e2[-1])

    def online_increments(self, x_online):
        online = np.asarray(x_online, dtype=np.float64)
        z = (online - self.hs.mu0) / self.hs.sd0
        residual = d3._ar_residuals(z, self.hs.phi, self.hs.zh_tail)
        e = np.clip(residual / self.hs.resid_sd, -10.0, 10.0)
        e2 = np.minimum(e * e, 100.0)
        blocks = []
        for filter_index, constants in enumerate(self.filter_constants):
            alpha = constants["alpha"]
            beta = constants["beta"]
            omega = constants["omega"]
            variance = _variance_continue(
                e2, alpha, beta, omega, self.last_e2,
                self.last_variance[filter_index])

            prev_e = np.concatenate(([self.last_e], e[:-1]))
            prev_e2 = np.concatenate(([self.last_e2], e2[:-1]))
            prev_variance = np.concatenate(
                ([self.last_variance[filter_index]], variance[:-1]))
            ratio = np.minimum(e2 / np.maximum(variance, 1e-4), 25.0)
            absu = np.minimum(
                np.abs(e) / np.sqrt(np.maximum(variance, 1e-4)), 5.0)
            centered_ratio = ratio - constants["mean_ratio"]
            negative_energy = (prev_e < 0.0).astype(np.float64) * prev_e2
            blocks.append(np.column_stack((
                ratio,
                absu,
                (absu > constants["tail_threshold"]).astype(np.float64),
                centered_ratio * (prev_e2 - constants["mean_prev_e2"]),
                centered_ratio * (
                    prev_variance - constants["mean_prev_variance"]),
                centered_ratio * (
                    negative_energy - constants["mean_negative_energy"]),
            )))
        raw = np.column_stack(blocks)
        return (raw - self.raw_mean) / self.raw_sd


def _scan_offline(increments):
    length = increments.shape[0]
    if length == 0:
        return np.empty((0, N_FEATURES), dtype=np.float64)
    n = np.arange(1, length + 1, dtype=np.float64)
    cumulative = np.cumsum(increments, axis=0)
    padded = np.vstack((np.zeros((1, N_ASPECTS)), cumulative))
    minimum = np.minimum.accumulate(np.minimum(cumulative, 0.0), axis=0)
    maximum = np.maximum.accumulate(np.maximum(cumulative, 0.0), axis=0)
    scan = np.empty((length, N_ASPECTS, N_PER_RAW), dtype=np.float64)
    root_n = np.sqrt(n)[:, None]
    scan[:, :, 0] = cumulative / root_n
    scan[:, :, 1] = (cumulative - minimum) / root_n
    scan[:, :, 2] = (maximum - cumulative) / root_n
    index = np.arange(1, length + 1, dtype=np.int64)
    for column, window in enumerate(WINDOWS, start=3):
        start = np.maximum(0, index - int(window))
        sums = padded[index] - padded[start]
        scan[:, :, column] = sums / np.sqrt(
            np.minimum(n, float(window)))[:, None]
    return scan.reshape(length, N_FEATURES)


def compute_features_offline(x_hist, x_online):
    hist = Hist37(x_hist)
    return _scan_offline(hist.online_increments(x_online))


class StreamDetector37:
    def __init__(self, x_hist, inner=None):
        self.inner = d3.StreamDetector3(x_hist) if inner is None else inner
        self.owns_inner = inner is None
        self.hist = Hist37(x_hist)
        self.t = 0
        self.previous_e = self.hist.last_e
        self.previous_e2 = self.hist.last_e2
        self.previous_variance = self.hist.last_variance.copy()
        self.cumulative = np.zeros(N_ASPECTS, dtype=np.float64)
        self.minimum = np.zeros(N_ASPECTS, dtype=np.float64)
        self.maximum = np.zeros(N_ASPECTS, dtype=np.float64)
        self.ring = np.zeros((N_ASPECTS, MAX_WINDOW), dtype=np.float64)
        self.window_sum = np.zeros((N_ASPECTS, len(WINDOWS)),
                                   dtype=np.float64)

    def _from_current_inner(self):
        e = float(np.clip(
            self.inner.e_prev / self.inner.hs.resid_sd, -10.0, 10.0))
        e2 = min(e * e, 100.0)
        blocks = []
        new_variance = np.empty(len(FILTERS), dtype=np.float64)
        for filter_index, constants in enumerate(
                self.hist.filter_constants):
            variance = min(max(
                constants["omega"] +
                constants["alpha"] * self.previous_e2 +
                constants["beta"] * self.previous_variance[filter_index],
                1e-4), 100.0)
            new_variance[filter_index] = variance
            ratio = min(e2 / variance, 25.0)
            absu = min(abs(e) / np.sqrt(variance), 5.0)
            centered_ratio = ratio - constants["mean_ratio"]
            negative_energy = (
                self.previous_e2 if self.previous_e < 0.0 else 0.0)
            blocks.extend((
                ratio,
                absu,
                float(absu > constants["tail_threshold"]),
                centered_ratio * (
                    self.previous_e2 - constants["mean_prev_e2"]),
                centered_ratio * (
                    self.previous_variance[filter_index] -
                    constants["mean_prev_variance"]),
                centered_ratio * (
                    negative_energy - constants["mean_negative_energy"]),
            ))
        raw = np.asarray(blocks, dtype=np.float64)
        increment = (raw - self.hist.raw_mean) / self.hist.raw_sd

        t = self.t
        n = t + 1
        self.cumulative += increment
        self.minimum = np.minimum(
            self.minimum, np.minimum(self.cumulative, 0.0))
        self.maximum = np.maximum(
            self.maximum, np.maximum(self.cumulative, 0.0))
        scan = np.empty((N_ASPECTS, N_PER_RAW), dtype=np.float64)
        root_n = np.sqrt(float(n))
        scan[:, 0] = self.cumulative / root_n
        scan[:, 1] = (self.cumulative - self.minimum) / root_n
        scan[:, 2] = (self.maximum - self.cumulative) / root_n
        ring_position = t % MAX_WINDOW
        for column, window in enumerate(WINDOWS, start=3):
            window_index = column - 3
            self.window_sum[:, window_index] += increment
            if t >= int(window):
                self.window_sum[:, window_index] -= self.ring[
                    :, (t - int(window)) % MAX_WINDOW]
            scan[:, column] = (
                self.window_sum[:, window_index] /
                np.sqrt(float(min(n, int(window)))))

        self.ring[:, ring_position] = increment
        self.previous_e = e
        self.previous_e2 = e2
        self.previous_variance = new_variance
        self.t = n
        return scan.reshape(N_FEATURES)

    def step(self, value):
        if not self.owns_inner:
            raise RuntimeError("use step_from_inner with a shared detector")
        self.inner.step(value)
        return self._from_current_inner()

    def step_from_inner(self):
        return self._from_current_inner()


if __name__ == "__main__":
    rng = np.random.default_rng(20260813)
    worst = 0.0
    for trial in range(10):
        n_hist = int(rng.integers(1000, 3000))
        length = int(rng.integers(10, 700))
        ar = float(rng.uniform(-0.8, 0.8))
        noise = rng.standard_t(float(rng.uniform(3.5, 15.0)),
                               n_hist + length)
        values = np.empty_like(noise)
        values[0] = noise[0]
        variance = 1.0
        for index in range(1, values.size):
            variance = 0.05 + 0.10 * values[index - 1] ** 2 + 0.85 * variance
            values[index] = ar * values[index - 1] + np.sqrt(
                max(variance, 1e-4)) * noise[index] / np.sqrt(1.4)
        history = values[:n_hist].copy()
        online = values[n_hist:].copy()
        if trial % 2 == 0 and length > 30:
            tau = int(rng.integers(5, length - 5))
            for index in range(tau, length):
                online[index] *= 1.0 + 0.35 * (
                    abs(online[index - 1]) > np.median(np.abs(history)))
        offline = compute_features_offline(history, online)
        detector = StreamDetector37(history)
        streamed = np.vstack([detector.step(value) for value in online])
        error = float(np.max(np.abs(offline - streamed)))
        worst = max(worst, error)
        assert offline.shape == (length, N_FEATURES)
        assert error < 1e-9, (trial, error)
    assert len(feature_names()) == N_FEATURES

    detector = StreamDetector37(rng.normal(size=3000))
    sample = rng.normal(size=1000)
    started = time.perf_counter()
    for value in sample:
        detector.step(value)
    elapsed = time.perf_counter() - started
    print(f"detectors37 OK: {N_FEATURES} features, worst equality "
          f"{worst:.3e}, {elapsed * 1e6 / len(sample):.0f} us/step")
