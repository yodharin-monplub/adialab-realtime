"""Supervised break-template projections with causal scan statistics.

Templates are fitted outside this module from training-series post-break
effects.  For a new series, each multivariate projection is normalized by
its own null standard deviation measured on that series' guaranteed
break-free history, then accumulated with prefix/CUSUM/suffix scans.
"""

from __future__ import annotations

import time

import numpy as np

import channels18
import detectors3 as d3


WINDOWS = d3.WINDOWS
N_PER_TEMPLATE = 3 + len(WINDOWS)


def feature_names(n_templates):
    """Names for a fitted template count; the output width is model-dependent."""

    names = []
    for template in range(int(n_templates)):
        names.extend(
            (
                f"projection{template}_prefix",
                f"projection{template}_up",
                f"projection{template}_down",
            )
        )
        names.extend(
            f"projection{template}_w{window}" for window in WINDOWS
        )
    return names


def _history_matrix(x_hist):
    return channels18.compute_history_channels(x_hist)


def projection_scale(x_hist, templates):
    projected = _history_matrix(x_hist) @ templates.T
    scale = np.std(projected, axis=0)
    return np.maximum(scale, 0.10)


def compute_features_from_channels(x_hist, online_channels, templates):
    templates = np.asarray(templates, dtype=np.float64)
    g = np.asarray(online_channels, dtype=np.float64)[:, :len(d3.ASPECTS)]
    projected = (g @ templates.T) / projection_scale(x_hist, templates)
    T, n_templates = projected.shape
    n = np.arange(1, T + 1, dtype=np.float64)
    indices = np.arange(1, T + 1)
    cols = []
    for j in range(n_templates):
        cumulative = np.cumsum(projected[:, j])
        padded = np.concatenate(([0.0], cumulative))
        run_min = np.minimum.accumulate(np.minimum(cumulative, 0.0))
        run_max = np.maximum.accumulate(np.maximum(cumulative, 0.0))
        cols.extend((
            cumulative / np.sqrt(n),
            (cumulative - run_min) / np.sqrt(n),
            (run_max - cumulative) / np.sqrt(n),
        ))
        for window in WINDOWS:
            start = np.maximum(0, indices - window)
            effective = np.minimum(n, window)
            total = padded[1:] - padded[start]
            cols.append(total / np.sqrt(effective))
    return np.column_stack(cols)


def compute_features_offline(x_hist, x_online, templates):
    channels = channels18.compute_channels_offline(x_hist, x_online)
    return compute_features_from_channels(x_hist, channels, templates)


class StreamDetector19:
    def __init__(self, x_hist, templates):
        self.inner = d3.StreamDetector3(x_hist)
        self.templates = np.asarray(templates, dtype=np.float64)
        self.scale = projection_scale(x_hist, self.templates)
        self.n_templates = self.templates.shape[0]
        self.t = 0
        self.cumulative = np.zeros(self.n_templates, dtype=np.float64)
        self.running_min = np.zeros(self.n_templates, dtype=np.float64)
        self.running_max = np.zeros(self.n_templates, dtype=np.float64)
        self.ring = np.zeros((max(WINDOWS), self.n_templates), dtype=np.float64)
        self.window_sum = np.zeros((len(WINDOWS), self.n_templates),
                                   dtype=np.float64)

    def step(self, x):
        self.inner.step(x)
        t = self.t
        n = t + 1
        pos = t % self.inner._wmax
        g = np.asarray([
            np.clip(self.inner._ring[aspect][pos] / self.inner.hs.s[aspect],
                    -channels18.CLIP, channels18.CLIP)
            for aspect in d3.ASPECTS
        ])
        projected = (self.templates @ g) / self.scale
        self.cumulative += projected
        self.running_min = np.minimum(self.running_min,
                                      np.minimum(self.cumulative, 0.0))
        self.running_max = np.maximum(self.running_max,
                                      np.maximum(self.cumulative, 0.0))

        out = np.empty(self.n_templates * N_PER_TEMPLATE, dtype=np.float64)
        root_n = np.sqrt(float(n))
        for j in range(self.n_templates):
            offset = j * N_PER_TEMPLATE
            out[offset] = self.cumulative[j] / root_n
            out[offset + 1] = ((self.cumulative[j] - self.running_min[j]) /
                               root_n)
            out[offset + 2] = ((self.running_max[j] - self.cumulative[j]) /
                               root_n)
            for k, window in enumerate(WINDOWS):
                self.window_sum[k, j] += projected[j]
                if t >= window:
                    self.window_sum[k, j] -= self.ring[(t-window) %
                                                       self.ring.shape[0], j]
                out[offset + 3 + k] = (self.window_sum[k, j] /
                                       np.sqrt(float(min(n, window))))
        self.ring[t % self.ring.shape[0]] = projected
        self.t = n
        return out


if __name__ == "__main__":
    rng = np.random.default_rng(19)
    templates = rng.normal(size=(12, len(d3.ASPECTS)))
    templates /= np.linalg.norm(templates, axis=1, keepdims=True)
    for trial in range(8):
        n_hist = int(rng.integers(1000, 3000))
        T = int(rng.integers(3, 700))
        ar = rng.uniform(-0.85, 0.85)
        eps = rng.standard_t(rng.uniform(3.0, 12.0), n_hist + T)
        x = np.empty(n_hist + T)
        x[0] = eps[0]
        for i in range(1, x.size):
            x[i] = ar * x[i - 1] + eps[i]
        hist, online = x[:n_hist], x[n_hist:].copy()
        if trial % 2 == 0 and T > 10:
            tau = int(rng.integers(1, T - 1))
            online[tau:] += rng.normal() * 0.7
        offline = compute_features_offline(hist, online, templates)
        stream = StreamDetector19(hist, templates)
        streamed = np.vstack([stream.step(value) for value in online])
        error = float(np.max(np.abs(offline - streamed)))
        assert offline.shape == (T, templates.shape[0] * N_PER_TEMPLATE)
        assert error < 1e-9, (trial, error)
    assert len(feature_names(templates.shape[0])) == templates.shape[0] * N_PER_TEMPLATE
    detector = StreamDetector19(rng.normal(size=3000), templates)
    started = time.perf_counter()
    for value in rng.normal(size=1000):
        detector.step(value)
    elapsed = (time.perf_counter() - started) / 1000
    print(f"OK: detectors19 stream == offline; "
          f"{templates.shape[0] * N_PER_TEMPLATE} features; "
          f"{elapsed*1e6:.0f} us/step")
