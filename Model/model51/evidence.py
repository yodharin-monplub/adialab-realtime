"""Causal AR-residual rank whitening and integrated shift Bayes factors."""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
from scipy.special import logsumexp, ndtri

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model20.features import detector_module

WINDOWS = (16, 32, 64, 128, 256, 512)
PRIOR_VARIANCES = (0.05, 0.5)


def empirical_normal(reference, values):
    sorted_ref = np.sort(np.asarray(reference, dtype=np.float64))
    values = np.asarray(values, dtype=np.float64)
    left = np.searchsorted(sorted_ref, values, side="left")
    right = np.searchsorted(sorted_ref, values, side="right")
    probability = (left + right + 1.0) / (2.0 * (len(sorted_ref) + 1.0))
    return np.clip(ndtri(probability), -3.75, 3.75)


def whitened_residuals(history, online):
    d3 = detector_module().d3
    h = np.asarray(history, dtype=np.float64)
    o = np.asarray(online, dtype=np.float64)
    if len(h) < 64 or not np.isfinite(h).all() or not np.isfinite(o).all():
        raise ValueError("Invalid Model51 series input")
    reference = d3.Hist3(h)
    hz = (h - reference.mu0) / reference.sd0
    oz = (o - reference.mu0) / reference.sd0
    p = len(reference.phi)
    he = d3._ar_residuals(hz, reference.phi, np.zeros(p))[p:]
    oe = d3._ar_residuals(oz, reference.phi, reference.zh_tail)
    if len(he) < 32:
        raise ValueError("Insufficient residual history")
    return empirical_normal(he, he), empirical_normal(he, oe)


def channels(history_normal, online_normal):
    h = np.asarray(history_normal, dtype=np.float64)
    o = np.asarray(online_normal, dtype=np.float64)
    h_dep = np.r_[0.0, h[1:] * h[:-1]]
    o_dep = np.empty_like(o)
    if len(o):
        o_dep[0] = o[0] * h[-1]
        o_dep[1:] = o[1:] * o[:-1]
    history = np.column_stack((h, h * h, np.abs(h), h_dep))
    online = np.column_stack((o, o * o, np.abs(o), o_dep))
    location = history.mean(axis=0)
    scale = history.std(axis=0)
    scale = np.maximum(scale, .05)
    return np.clip((history - location) / scale, -8, 8), np.clip((online - location) / scale, -8, 8)


def integrated_log_bayes(channels):
    values = np.asarray(channels, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all():
        raise ValueError("Invalid Model51 innovation channels")
    steps = len(values)
    if steps == 0:
        return np.empty(0, dtype=np.float64)
    cumulative = np.vstack((np.zeros((1, 4)), np.cumsum(values, axis=0)))
    combined = np.full((steps, 4), -np.inf, dtype=np.float64)
    count = np.zeros(steps, dtype=np.int32)
    for window in WINDOWS:
        if steps < window:
            continue
        sums = cumulative[window:] - cumulative[:-window]
        for prior_variance in PRIOR_VARIANCES:
            shrink = prior_variance / (1.0 + window * prior_variance)
            penalty = math.log1p(window * prior_variance)
            log_bayes = .5 * (shrink * sums * sums - penalty)
            combined[window - 1:] = np.logaddexp(combined[window - 1:], log_bayes)
            count[window - 1:] += 1
    result = np.zeros(steps, dtype=np.float64)
    valid = count > 0
    result[valid] = logsumexp(combined[valid] - np.log(count[valid])[:, None], axis=1) - math.log(4)
    return result


def evidence_score(history, online):
    h, o = whitened_residuals(history, online)
    hist_channels, online_channels = channels(h, o)
    reference = integrated_log_bayes(hist_channels)
    online_score = integrated_log_bayes(online_channels)
    burn = max(WINDOWS) if len(reference) >= max(WINDOWS) + 32 else min(
        len(reference), max(128, len(reference) // 4))
    quiet = reference[burn - 1:]
    if len(quiet) < 32:
        quiet = reference[min(WINDOWS) - 1:]
    center = float(np.median(quiet))
    scale = float(1.4826 * np.median(np.abs(quiet - center)))
    scale = max(scale, .25)
    result = np.clip((online_score - center) / scale, -5, 15)
    result[:min(WINDOWS) - 1] = 0
    if len(result) != len(online) or not np.isfinite(result).all():
        raise ValueError("Nonfinite or misaligned Model51 evidence")
    return result.astype(np.float32)


def score_task(task):
    position, history, online = task
    return position, evidence_score(history, online)
