"""O(1)-per-row implementation of Model73's frozen whitening features."""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter
from scipy.special import ndtri

from model73_whitening import (AR_ORDERS, CHANNELS, CONTEXT, EWMA_LAMBDA,
    FEATURE_NAMES, WINDOWS, _empirical_normal, _fit_ar, _raw_channels,
    _residuals, _robust_scale)


class StreamingWhitening:
    """Incrementally reproduce ``whitening_features(...)[-1]`` exactly."""

    def __init__(self, history):
        h = np.asarray(history, dtype=np.float64)
        if h.ndim != 1 or len(h) < 64 or not np.isfinite(h).all():
            raise ValueError("Invalid history for Model87 whitening")
        center, scale = _robust_scale(h)
        hz = np.clip((h - center) / scale, -20, 20)
        order, intercept, phi = _fit_ar(hz)
        eh, _ = _residuals(hz, np.empty(0, dtype=np.float64),
                           order, intercept, phi)
        clipped = np.clip(np.asarray(eh, dtype=np.float64), -20, 20)
        variance0 = max(float(np.mean(clipped * clipped)), 1e-4)
        forcing = (1 - EWMA_LAMBDA) * np.r_[
            variance0, clipped[:-1] * clipped[:-1]]
        historical_variance = lfilter(
            [1.], [1., -EWMA_LAMBDA], forcing,
            zi=np.asarray([EWMA_LAMBDA * variance0]))[0]
        rh = clipped / np.sqrt(np.clip(historical_variance, 1e-4, 100.))
        zh = _empirical_normal(rh, rh)
        historical = _raw_channels(zh, np.r_[0., zh[:-1]])
        reference = historical[min(16, len(historical) - 1):]

        self.center = center
        self.scale = scale
        self.order = order
        self.intercept = intercept
        self.phi = np.asarray(phi, dtype=np.float64)
        self.ar_values = list(hz[-order:]) if order else []
        self.previous_residual = float(clipped[-1])
        self.previous_variance = float(historical_variance[-1])
        self.sorted_reference = np.sort(rh)
        self.previous_normal = float(zh[-1])
        self.channel_center = np.mean(reference, axis=0)
        self.channel_scale = np.maximum(np.std(reference, axis=0), .15)
        self.prefix = [np.zeros(len(CHANNELS), dtype=np.float64)]

        phi1 = float(phi[0]) if order else 0.
        squared = np.clip(eh * eh, 0, 100)
        vol_cluster = (0. if len(squared) < 3 else
            float(np.corrcoef(squared[1:], squared[:-1])[0, 1]))
        if not np.isfinite(vol_cluster):
            vol_cluster = 0.
        residual_scale = max(float(np.std(eh)), 1e-8)
        self.context = np.asarray((order / 12., phi1,
            np.log(residual_scale / max(float(np.std(hz)), 1e-8)),
            np.clip(vol_cluster, -1, 1),
            np.log(max(float(np.mean(
                np.clip(eh / residual_scale, -10, 10) ** 4)), 1.))),
            dtype=np.float64)
        if self.context.shape != (len(CONTEXT),) or not np.isfinite(self.context).all():
            raise ValueError("Invalid Model87 whitening context")

    def step(self, value):
        raw = float(value)
        if not np.isfinite(raw):
            raise ValueError("Nonfinite Model87 online value")
        normalized = float(np.clip((raw - self.center) / self.scale, -20, 20))
        residual = normalized - self.intercept
        if self.order:
            predictors = np.asarray(self.ar_values[-self.order:][::-1],
                                    dtype=np.float64)
            residual -= float(predictors @ self.phi)
            self.ar_values.append(normalized)
            if len(self.ar_values) > self.order:
                del self.ar_values[0]
        clipped_residual = float(np.clip(residual, -20, 20))
        variance = (EWMA_LAMBDA * self.previous_variance +
                    (1 - EWMA_LAMBDA) * self.previous_residual ** 2)
        standardized = clipped_residual / np.sqrt(np.clip(variance, 1e-4, 100.))
        self.previous_residual = clipped_residual
        self.previous_variance = variance

        left = np.searchsorted(self.sorted_reference, standardized, side="left")
        right = np.searchsorted(self.sorted_reference, standardized, side="right")
        probability = (left + right + 1.) / (2. * (len(self.sorted_reference) + 1.))
        normal = float(np.clip(ndtri(probability), -3.75, 3.75))
        channels = np.asarray((normal, normal * normal, abs(normal),
            normal * self.previous_normal, float(normal > 2.5),
            float(normal < -2.5), abs(normal - self.previous_normal)),
            dtype=np.float64)
        self.previous_normal = normal
        adjusted = np.clip(
            (channels - self.channel_center) / self.channel_scale, -8, 8)
        self.prefix.append(self.prefix[-1] + adjusted)
        end = len(self.prefix) - 1
        blocks = []
        for window in WINDOWS:
            start = 0 if window == 0 else max(0, end - window)
            count = end - start
            blocks.append(np.clip(
                (self.prefix[end] - self.prefix[start]) / np.sqrt(count),
                -20, 20))
        maximum = np.max(np.abs(np.stack(blocks)), axis=0)
        row = np.ascontiguousarray(
            np.concatenate((*blocks, maximum, self.context)), dtype=np.float32)
        if row.shape != (len(FEATURE_NAMES),) or not np.isfinite(row).all():
            raise ValueError("Invalid Model87 whitening row")
        return row
