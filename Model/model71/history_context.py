"""History-only process fingerprints for causal, series-conditioned detection."""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = (
    "hist_context_log_std", "hist_context_mean_over_std",
    "hist_context_abs_mean_over_std", "hist_context_median_over_mad",
    "hist_context_skew", "hist_context_kurtosis", "hist_context_iqr_over_std",
    "hist_context_acf1", "hist_context_acf2", "hist_context_acf4",
    "hist_context_acf8", "hist_context_diff_over_level",
    "hist_context_absdiff_acf1", "hist_context_tail3_rate",
    "hist_context_linear_trend", "hist_context_halfmean_shift",
    "hist_context_low_frequency_power", "hist_context_spectral_entropy",
)


def _correlation(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ac = a - a.mean()
    bc = b - b.mean()
    return float(np.mean(ac * bc) / np.sqrt(max(np.mean(ac * ac) * np.mean(bc * bc), 1e-12)))


def history_context_features(history, online):
    """Repeat a clean-history fingerprint at every online prefix.

    It never reads online values; repeating the vector preserves the existing
    training/inference row contract and gives exact prefix invariance.
    """
    h = np.asarray(history, dtype=np.float64)
    o = np.asarray(online, dtype=np.float64)
    if h.ndim != 1 or o.ndim != 1 or len(h) < 64 or not np.isfinite(h).all() or not np.isfinite(o).all():
        raise ValueError("Invalid series for Model71 context")
    if not len(o):
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    mean = float(np.mean(h))
    sd = max(float(np.std(h)), 1e-9)
    median = float(np.median(h))
    mad = max(float(np.median(np.abs(h - median))) * 1.4826, 1e-9)
    z = np.clip((h - mean) / sd, -20.0, 20.0)
    centered = z - z.mean()
    zvar = max(float(np.mean(centered * centered)), 1e-9)
    skew = float(np.mean(centered**3) / zvar**1.5)
    kurt = float(np.mean(centered**4) / zvar**2)
    quartiles = np.quantile(h, (0.25, 0.75))
    acf = [_correlation(z[lag:], z[:-lag]) for lag in (1, 2, 4, 8)]
    diff = np.diff(z)
    absdiff = np.abs(diff)
    time = np.linspace(-1.0, 1.0, len(h))
    spectrum = np.abs(np.fft.rfft(centered))**2
    spectrum = spectrum[1:]
    spectral_total = max(float(np.sum(spectrum)), 1e-12)
    frequency_count = len(spectrum)
    low_end = max(1, frequency_count // 32)
    probabilities = spectrum / spectral_total
    entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-15)))) / max(np.log(frequency_count), 1.0)
    half = len(h) // 2
    context = np.asarray((
        np.log(sd), mean / sd, abs(mean) / sd, median / mad,
        skew, kurt, (quartiles[1] - quartiles[0]) / sd,
        *acf, np.std(diff), _correlation(absdiff[1:], absdiff[:-1]),
        np.mean(np.abs(z) > 3.0), _correlation(z, time),
        (np.mean(z[half:]) - np.mean(z[:half])),
        np.sum(spectrum[:low_end]) / spectral_total, entropy,
    ), dtype=np.float64)
    if len(context) != len(FEATURE_NAMES):
        raise ValueError("Model71 context feature count mismatch")
    context = np.clip(context, -30.0, 30.0).astype(np.float32)
    if not np.isfinite(context).all():
        raise ValueError("Nonfinite Model71 context")
    return np.repeat(context[None, :], len(o), axis=0)
