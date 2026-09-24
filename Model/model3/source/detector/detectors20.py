"""History-recalibrated learned emission score with causal accumulators.

An external deterministic model maps the 23 normalized detectors3 aspect
increments at one observation to a scalar post-break emission margin.  The
margin is centered and scaled on each series' guaranteed break-free history,
then accumulated with CUSUM/suffix and EWMA geometries.  The emission model
never sees online future values.
"""

from __future__ import annotations

import time

import numpy as np

import channels18
import detectors3 as d3
import detectors15 as d15
import detectors19 as d19


WINDOWS = d3.WINDOWS
ALPHAS = d15.ALPHAS
N_SCAN = 3 + len(WINDOWS)
N_EMA = len(ALPHAS) * 2
N_FEATURES = N_SCAN + N_EMA


def feature_names():
    names = ["emit_prefix", "emit_up", "emit_down"]
    names.extend(f"emit_w{window}" for window in WINDOWS)
    for alpha in ALPHAS:
        tag = f"a{alpha:.7f}".rstrip("0").rstrip(".")
        names.extend((f"emit_ema_{tag}", f"emit_absema_{tag}"))
    return names


def _predict_margin(model, matrix):
    values = model.predict(np.asarray(matrix, dtype=np.float64), raw_score=True)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError("emission model produced non-finite margins")
    return values


class EmissionCalibration:
    def __init__(self, x_hist, model):
        history = d19._history_matrix(x_hist)
        raw = _predict_margin(model, history)
        self.mu = float(np.mean(raw))
        sd = float(np.std(raw))
        self.sd = sd if np.isfinite(sd) and sd > 1e-8 else 1.0
        standardized = (raw - self.mu) / self.sd
        self.var1, self.ratio = d15._steady_ratios(standardized)

    def transform(self, raw):
        return float(np.clip((float(raw) - self.mu) / self.sd,
                             -channels18.CLIP, channels18.CLIP))

    def ema_scale(self, n):
        n = int(n)
        r = 1.0 - ALPHAS
        independent = (self.var1 * ALPHAS / (2.0 - ALPHAS) *
                       (1.0 - r ** (2 * n)))
        ramp = 1.0 + (self.ratio - 1.0) * (
            1.0 - r ** max(n - 1, 0))
        return np.sqrt(np.maximum(independent * ramp, 1e-18))

    def ema_matrix(self, length):
        n = np.arange(1, int(length) + 1, dtype=np.float64)
        r = 1.0 - ALPHAS
        independent = (self.var1 * ALPHAS[None, :] /
                       (2.0 - ALPHAS[None, :]) *
                       (1.0 - r[None, :] ** (2 * n[:, None])))
        ramp = 1.0 + (self.ratio[None, :] - 1.0) * (
            1.0 - r[None, :] ** np.maximum(n[:, None] - 1, 0))
        return np.sqrt(np.maximum(independent * ramp, 1e-18))


def _features_from_values(values, calibration):
    values = np.asarray(values, dtype=np.float64)
    T = values.size
    n = np.arange(1, T + 1, dtype=np.float64)
    cumulative = np.cumsum(values)
    padded = np.concatenate(([0.0], cumulative))
    run_min = np.minimum.accumulate(np.minimum(cumulative, 0.0))
    run_max = np.maximum.accumulate(np.maximum(cumulative, 0.0))
    columns = [
        cumulative / np.sqrt(n),
        (cumulative - run_min) / np.sqrt(n),
        (run_max - cumulative) / np.sqrt(n),
    ]
    index = np.arange(1, T + 1)
    for window in WINDOWS:
        start = np.maximum(0, index - window)
        total = padded[1:] - padded[start]
        columns.append(total / np.sqrt(np.minimum(n, window)))

    state = np.zeros(len(ALPHAS), dtype=np.float64)
    states = np.empty((T, len(ALPHAS)), dtype=np.float64)
    for t, value in enumerate(values):
        state = (1.0 - ALPHAS) * state + ALPHAS * value
        states[t] = state
    z = states / calibration.ema_matrix(T)
    columns.extend(np.stack((z, np.abs(z)), axis=-1).reshape(T, N_EMA).T)
    return np.column_stack(columns)


def compute_features_from_channels(x_hist, online_channels, model):
    calibration = EmissionCalibration(x_hist, model)
    matrix = np.asarray(online_channels, dtype=np.float64)[:, :len(d3.ASPECTS)]
    raw = _predict_margin(model, matrix)
    values = np.clip((raw - calibration.mu) / calibration.sd,
                     -channels18.CLIP, channels18.CLIP)
    return _features_from_values(values, calibration)


def compute_features_offline(x_hist, x_online, model):
    channels = channels18.compute_channels_offline(x_hist, x_online)
    return compute_features_from_channels(x_hist, channels, model)


class StreamDetector20:
    def __init__(self, x_hist, model):
        self.inner = d3.StreamDetector3(x_hist)
        self.model = model
        self.calibration = EmissionCalibration(x_hist, model)
        self.t = 0
        self.cumulative = 0.0
        self.running_min = 0.0
        self.running_max = 0.0
        self.ring = np.zeros(max(WINDOWS), dtype=np.float64)
        self.window_sum = np.zeros(len(WINDOWS), dtype=np.float64)
        self.ema = np.zeros(len(ALPHAS), dtype=np.float64)

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
        raw = _predict_margin(self.model, g[None, :])[0]
        value = self.calibration.transform(raw)
        self.cumulative += value
        self.running_min = min(self.running_min, self.cumulative, 0.0)
        self.running_max = max(self.running_max, self.cumulative, 0.0)
        root_n = np.sqrt(float(n))
        out = np.empty(N_FEATURES, dtype=np.float64)
        out[0] = self.cumulative / root_n
        out[1] = (self.cumulative - self.running_min) / root_n
        out[2] = (self.running_max - self.cumulative) / root_n
        for k, window in enumerate(WINDOWS):
            self.window_sum[k] += value
            if t >= window:
                self.window_sum[k] -= self.ring[(t-window) % self.ring.size]
            out[3 + k] = self.window_sum[k] / np.sqrt(float(min(n, window)))
        self.ring[t % self.ring.size] = value

        self.ema = (1.0 - ALPHAS) * self.ema + ALPHAS * value
        z = self.ema / self.calibration.ema_scale(n)
        out[N_SCAN:] = np.stack((z, np.abs(z)), axis=-1).ravel()
        self.t = n
        return out


class _DummyEmission:
    def predict(self, matrix, raw_score=False):
        x = np.asarray(matrix, dtype=np.float64)
        return (0.31 * x[:, 0] - 0.17 * x[:, 1] +
                0.13 * np.abs(x[:, 3]) + 0.07 * x[:, 8] * x[:, 11])


if __name__ == "__main__":
    rng = np.random.default_rng(20)
    model = _DummyEmission()
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
            online[tau:] *= rng.uniform(1.2, 1.8)
        offline = compute_features_offline(hist, online, model)
        stream = StreamDetector20(hist, model)
        streamed = np.vstack([stream.step(value) for value in online])
        error = float(np.max(np.abs(offline - streamed)))
        assert offline.shape == (T, N_FEATURES)
        assert error < 1e-9, (trial, error)
    detector = StreamDetector20(rng.normal(size=3000), model)
    started = time.perf_counter()
    for value in rng.normal(size=1000):
        detector.step(value)
    elapsed = (time.perf_counter() - started) / 1000
    assert len(feature_names()) == N_FEATURES
    print(f"OK: detectors20 stream == offline; {N_FEATURES} features; "
          f"{elapsed*1e6:.0f} us/step")
