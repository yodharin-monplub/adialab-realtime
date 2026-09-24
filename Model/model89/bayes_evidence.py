"""Causal, history-whitened Bayes evidence for scale and AR(1) breaks."""
from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln, logsumexp

from models.model57.whitening import (
    _empirical_normal,
    _fit_ar,
    _residuals,
    _robust_scale,
    _variance_paths,
)


# Ages are representatives of a uniform prior over every candidate tau.  The
# interval ending at node m receives mass m - previous_m; a final partial cell
# is added at the currently observed age so the weights always sum to n.
AGE_NODES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128,
             192, 256, 384, 512, 768)
SCALE_SHAPES = (100.0, 30.0, 12.0, 6.0, 3.0)
DEPENDENCE_MAGNITUDES = (0.06, 0.12, 0.20, 0.35)
DEPENDENCE_RHOS = tuple(
    sign * magnitude
    for magnitude in DEPENDENCE_MAGNITUDES
    for sign in (-1.0, 1.0)
)
FEATURE_NAMES = ("white_bayes_scale_tau_uniform", "white_bayes_dependence_tau_uniform")


def whitened_innovations(history, online):
    """Return reference and causal online Rosenblatt-normal innovations."""
    h = np.asarray(history, dtype=np.float64)
    o = np.asarray(online, dtype=np.float64)
    if (h.ndim != 1 or o.ndim != 1 or len(h) < 64 or
            not np.isfinite(h).all() or not np.isfinite(o).all()):
        raise ValueError("Invalid Model89 series input")
    center, scale = _robust_scale(h)
    hz = np.clip((h - center) / scale, -20.0, 20.0)
    oz = np.clip((o - center) / scale, -20.0, 20.0)
    order, intercept, phi = _fit_ar(hz)
    historical_residual, online_residual = _residuals(
        hz, oz, order, intercept, phi)
    historical_normalized, online_normalized = _variance_paths(
        historical_residual, online_residual)
    history_white = _empirical_normal(historical_normalized, historical_normalized)
    online_white = _empirical_normal(historical_normalized, online_normalized)
    return (np.asarray(history_white, dtype=np.float64),
            np.asarray(online_white, dtype=np.float64))


def _logmeanexp(values, axis=0):
    return logsumexp(values, axis=axis) - math.log(values.shape[axis])


def _scale_log_bayes(sum_squares, count):
    """IG(a,a-1) variance alternative versus fixed unit variance."""
    s = np.asarray(sum_squares, dtype=np.float64)
    n = np.asarray(count, dtype=np.float64)
    alternatives = []
    for shape in SCALE_SHAPES:
        scale = shape - 1.0
        alternatives.append(
            shape * math.log(scale) - gammaln(shape)
            + gammaln(shape + 0.5 * n)
            - (shape + 0.5 * n) * np.log(scale + 0.5 * s)
            + 0.5 * s
        )
    return _logmeanexp(np.stack(alternatives), axis=0)


def _dependence_log_bayes(sum_y2, sum_x2, sum_xy, count):
    """Mixture of stationary Gaussian AR(1) alternatives versus iid N(0,1)."""
    y2 = np.asarray(sum_y2, dtype=np.float64)
    x2 = np.asarray(sum_x2, dtype=np.float64)
    xy = np.asarray(sum_xy, dtype=np.float64)
    n = np.asarray(count, dtype=np.float64)
    alternatives = []
    for rho in DEPENDENCE_RHOS:
        variance = 1.0 - rho * rho
        residual = y2 - 2.0 * rho * xy + rho * rho * x2
        alternatives.append(
            -0.5 * n * math.log(variance)
            -0.5 * residual / variance
            +0.5 * y2
        )
    return _logmeanexp(np.stack(alternatives), axis=0)


def _suffix(prefix, window):
    return prefix[window:] - prefix[:-window]


def integrated_evidence(history_white, online_white):
    """Return two log Bayes mixtures with exactly uniform candidate-tau mass."""
    h = np.asarray(history_white, dtype=np.float64)
    z = np.asarray(online_white, dtype=np.float64)
    if h.ndim != 1 or z.ndim != 1 or not len(h):
        raise ValueError("Invalid whitened Model89 inputs")
    steps = len(z)
    if not steps:
        return np.empty((0, 2), dtype=np.float32)

    previous = np.r_[h[-1], z[:-1]]
    prefix_y2 = np.r_[0.0, np.cumsum(z * z)]
    prefix_x2 = np.r_[0.0, np.cumsum(previous * previous)]
    prefix_xy = np.r_[0.0, np.cumsum(z * previous)]
    scale_mix = np.full(steps, -np.inf, dtype=np.float64)
    dependence_mix = np.full(steps, -np.inf, dtype=np.float64)

    prior_mass = np.zeros(steps, dtype=np.int64)
    prior_node = 0
    for node in AGE_NODES:
        if node > steps:
            break
        rows = slice(node - 1, None)
        mass = node - prior_node
        count = float(node)
        scale_bf = _scale_log_bayes(_suffix(prefix_y2, node), count)
        dependence_bf = _dependence_log_bayes(
            _suffix(prefix_y2, node), _suffix(prefix_x2, node),
            _suffix(prefix_xy, node), count)
        scale_mix[rows] = np.logaddexp(scale_mix[rows], scale_bf + math.log(mass))
        dependence_mix[rows] = np.logaddexp(
            dependence_mix[rows], dependence_bf + math.log(mass))
        prior_mass[rows] += mass
        prior_node = node

    # At ages between grid nodes (and beyond 768), this final cell represents
    # every as-yet-uncovered tau using the complete observed online prefix.
    ages = np.arange(1, steps + 1, dtype=np.int64)
    completed = np.searchsorted(np.asarray(AGE_NODES), ages, side="right") - 1
    last = np.where(completed >= 0, np.asarray(AGE_NODES)[np.maximum(completed, 0)], 0)
    remainder = ages - last
    partial = remainder > 0
    if np.any(partial):
        count = ages[partial].astype(np.float64)
        scale_bf = _scale_log_bayes(prefix_y2[1:][partial], count)
        dependence_bf = _dependence_log_bayes(
            prefix_y2[1:][partial], prefix_x2[1:][partial],
            prefix_xy[1:][partial], count)
        scale_mix[partial] = np.logaddexp(
            scale_mix[partial], scale_bf + np.log(remainder[partial]))
        dependence_mix[partial] = np.logaddexp(
            dependence_mix[partial], dependence_bf + np.log(remainder[partial]))
        prior_mass[partial] += remainder[partial]

    if not np.array_equal(prior_mass, ages):
        raise AssertionError("Model89 candidate-tau prior mass does not sum to age")
    result = np.column_stack((scale_mix, dependence_mix)) - np.log(ages)[:, None]
    result = np.ascontiguousarray(np.clip(result, -30.0, 30.0), dtype=np.float32)
    if result.shape != (steps, len(FEATURE_NAMES)) or not np.isfinite(result).all():
        raise ValueError("Invalid Model89 evidence features")
    return result


def bayes_evidence_features(history, online):
    historical, current = whitened_innovations(history, online)
    return integrated_evidence(historical, current)


def prefix_parity(history, online):
    full = bayes_evidence_features(history, online)
    for cut in (1, 2, 3, 4, 6, 8, 16, 64, 128, 256, 768):
        if cut <= len(online):
            np.testing.assert_array_equal(
                bayes_evidence_features(history, online[:cut])[-1], full[cut - 1])
    return full
