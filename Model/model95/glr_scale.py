"""Causal maximum generalized likelihood ratio for a variance change."""
from __future__ import annotations

import numpy as np

from models.model89.bayes_evidence import whitened_innovations


AGE_NODES = (4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192,
             256, 384, 512, 768)
FEATURE_NAMES = ("white_scale_glr_max_tau",)


def _variance_glr(sum_squares, count):
    """Gaussian max_v log p(x|v)/p(x|v=1), constrained only by v>0."""
    total = np.asarray(sum_squares, dtype=np.float64)
    n = np.asarray(count, dtype=np.float64)
    ratio = np.clip(total / n, 1e-8, 1e8)
    return 0.5 * n * (ratio - 1.0 - np.log(ratio))


def glr_scale_features(history, online):
    _, z = whitened_innovations(history, online)
    steps = len(z)
    result = np.zeros(steps, dtype=np.float64)
    if not steps:
        return result.reshape(0, 1).astype(np.float32)
    prefix = np.r_[0.0, np.cumsum(z * z)]
    for node in AGE_NODES:
        if node > steps:
            break
        sums = prefix[node:] - prefix[:-node]
        result[node - 1:] = np.maximum(result[node - 1:], _variance_glr(sums, node))

    # The current full prefix is the exact final partial grid cell and also
    # covers candidate ages beyond the largest fixed node.
    ages = np.arange(1, steps + 1)
    eligible = ages >= AGE_NODES[0]
    result[eligible] = np.maximum(
        result[eligible], _variance_glr(prefix[1:][eligible], ages[eligible]))
    matrix = np.ascontiguousarray(np.clip(result, 0.0, 100.0)[:, None], dtype=np.float32)
    if matrix.shape != (steps, 1) or not np.isfinite(matrix).all():
        raise ValueError("Invalid Model95 scale GLR")
    return matrix


def prefix_parity(history, online):
    full = glr_scale_features(history, online)
    for cut in (1, 4, 6, 8, 16, 64, 128, 256, 768):
        if cut <= len(online):
            np.testing.assert_array_equal(
                glr_scale_features(history, online[:cut])[-1], full[cut - 1])
    return full
