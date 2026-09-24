"""Historical-reference features with equivalent batch and online updates."""
from collections import deque

import numpy as np

WINDOWS = (8, 16, 32, 64, 128, 256)
CHANNELS = ("level", "difference", "ar_residual", "cdf")
STATISTICS = ("mean_shift", "abs_mean_shift", "log_scale", "abs_log_scale", "tail_shift", "lag_shift")
FEATURE_NAMES = (
    ["log_history_length", "history_rho", "log_difference_scale", "history_tail", "history_trend", "log_online_step"]
    + [f"current_{c}" for c in CHANNELS]
    + [f"w{w}_{c}_{s}" for w in WINDOWS for c in CHANNELS for s in STATISTICS]
)


def _moments(channels, previous):
    return np.stack((channels, channels**2, (np.abs(channels) > 2).astype(float), channels * previous), axis=-1)


class Reference:
    def __init__(self, historical):
        h = np.asarray(historical, dtype=np.float64).reshape(-1)
        if h.size < 3 or not np.all(np.isfinite(h)):
            raise ValueError("history must contain at least three finite values")
        self.location = float(np.median(h))
        self.scale = max(float(np.std(h)), 1e-10)
        self.diff_location = float(np.mean(np.diff(h)))
        self.diff_scale = max(float(np.std(np.diff(h))), self.scale * 1e-5, 1e-10)
        centered = h - self.location
        self.rho = float(np.clip(np.dot(centered[:-1], centered[1:]) / max(np.dot(centered[:-1], centered[:-1]), 1e-20), -.995, .995))
        residual = centered[1:] - self.rho * centered[:-1]
        self.res_location = float(np.mean(residual))
        self.res_scale = max(float(np.std(residual)), self.scale * 1e-5, 1e-10)
        self.sorted_history = np.sort(h)
        self.historical = h
        self.history_channels = self.channels(h, np.r_[h[0], h[:-1]])
        self.history_moments = _moments(self.history_channels, np.vstack((self.history_channels[0], self.history_channels[:-1])))
        self.reference = self.history_moments.mean(axis=0)
        self.context = np.array([np.log1p(h.size), self.rho, np.log(self.diff_scale / self.scale), np.mean(np.abs(centered) > 2 * self.scale), (np.mean(h[-max(1,h.size//4):]) - np.mean(h[:max(1,h.size//4)])) / self.scale])

    def channels(self, x, previous):
        x = np.asarray(x)
        previous = np.asarray(previous)
        values = np.stack(((x - self.location) / self.scale,
                           (x - previous - self.diff_location) / self.diff_scale,
                           (x - self.location - self.rho * (previous - self.location) - self.res_location) / self.res_scale,
                           (np.searchsorted(self.sorted_history, x, side="right") / self.sorted_history.size - .5) * np.sqrt(12)), axis=-1)
        return np.clip(values, -12, 12)

    def assemble(self, average, channels, step):
        # average shape: (..., window, channel, moment).
        delta = average[..., 0] - self.reference[:, 0]
        variance = np.maximum(average[..., 1] - average[..., 0]**2, 1e-6)
        reference_var = np.maximum(self.reference[:, 1] - self.reference[:, 0]**2, 1e-6)
        log_scale = .5 * np.log(variance / reference_var)
        tail = average[..., 2] - self.reference[:, 2]
        lag = (average[..., 3] - average[..., 0]**2) / variance
        reference_lag = (self.reference[:, 3] - self.reference[:, 0]**2) / reference_var
        stats = np.stack((delta, np.abs(delta), log_scale, np.abs(log_scale), tail, np.clip(lag - reference_lag, -10, 10)), axis=-1)
        leading = channels.shape[:-1]
        context = np.broadcast_to(self.context, (*leading, 5))
        step_feature = np.broadcast_to(np.log1p(step), leading)[..., None]
        result = np.concatenate((context, step_feature, channels, stats.reshape(*leading, -1)), axis=-1)
        return np.clip(result, -100, 100).astype(np.float32)


def compute_features(historical, online):
    """Vectorized computation; every row depends only on its online prefix."""
    reference = Reference(historical)
    online = np.asarray(online, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(online)):
        raise ValueError("online values must be finite")
    if not online.size:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    channels = reference.channels(online, np.r_[reference.historical[-1], online[:-1]])
    moments = _moments(channels, np.vstack((reference.history_channels[-1], channels[:-1])))
    history = reference.history_moments[-max(WINDOWS):]
    cumulative = np.concatenate((np.zeros((1, 4, 4)), np.cumsum(np.concatenate((history, moments)), axis=0)))
    right = history.shape[0] + np.arange(1, online.size + 1)
    averages = []
    for window in WINDOWS:
        left = np.maximum(0, right - window)
        averages.append((cumulative[right] - cumulative[left]) / (right-left)[:, None, None])
    return reference.assemble(np.stack(averages, axis=1), channels, np.arange(1, online.size+1))


class StreamingFeatures:
    """Bounded-memory feature state, reset independently for each series."""
    def __init__(self, historical):
        self.reference = Reference(historical)
        self.buffers = [deque(self.reference.history_moments[-w:], maxlen=w) for w in WINDOWS]
        self.sums = np.stack([np.sum(buffer, axis=0) for buffer in self.buffers])
        self.previous = float(self.reference.historical[-1])
        self.previous_channel = self.reference.history_channels[-1]
        self.step = 0

    def update(self, value):
        if not np.isfinite(value):
            raise ValueError("online value must be finite")
        channel = self.reference.channels(value, self.previous)
        moment = _moments(channel, self.previous_channel)
        for i, (window, buffer) in enumerate(zip(WINDOWS, self.buffers)):
            if len(buffer) == window:
                self.sums[i] -= buffer[0]
            buffer.append(moment)
            self.sums[i] += moment
        self.previous, self.previous_channel = float(value), channel
        self.step += 1
        average = self.sums / np.array([len(b) for b in self.buffers])[:, None, None]
        return self.reference.assemble(average, channel, self.step)
