"""Causal maximum generalized likelihood ratio for dependence change."""
from __future__ import annotations

import numpy as np

from models.model89.bayes_evidence import whitened_innovations


AGE_NODES = (6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192,
             256, 384, 512, 768)
FEATURE_NAMES = ("white_dependence_glr_max_tau",)


def _correlation_glr(sum_x2, sum_y2, sum_xy, count):
    """Bivariate-normal correlation GLR against rho=0, with zero means."""
    denominator = np.sqrt(np.maximum(sum_x2 * sum_y2, 1e-16))
    rho = np.clip(sum_xy / denominator, -0.99, 0.99)
    return -0.5 * np.asarray(count, dtype=np.float64) * np.log1p(-(rho * rho))


def glr_dependence_features(history, online):
    _, z = whitened_innovations(history, online)
    steps = len(z)
    result = np.zeros(steps, dtype=np.float64)
    if steps < 2:
        return result.reshape(steps, 1).astype(np.float32)
    x, y = z[:-1], z[1:]
    px2 = np.r_[0.0, np.cumsum(x * x)]
    py2 = np.r_[0.0, np.cumsum(y * y)]
    pxy = np.r_[0.0, np.cumsum(x * y)]

    for node in AGE_NODES:
        if node > steps:
            break
        pair_count = node - 1
        sx2 = px2[pair_count:] - px2[:-pair_count]
        sy2 = py2[pair_count:] - py2[:-pair_count]
        sxy = pxy[pair_count:] - pxy[:-pair_count]
        result[node - 1:] = np.maximum(
            result[node - 1:], _correlation_glr(sx2, sy2, sxy, pair_count))

    # Exact full-prefix suffix is the final partial grid cell and covers ages
    # beyond the largest fixed node.
    ages = np.arange(1, steps + 1)
    eligible = ages >= AGE_NODES[0]
    count = ages[eligible] - 1
    result[eligible] = np.maximum(result[eligible], _correlation_glr(
        px2[ages[eligible] - 1], py2[ages[eligible] - 1],
        pxy[ages[eligible] - 1], count))
    matrix = np.ascontiguousarray(np.clip(result, 0.0, 100.0)[:, None], dtype=np.float32)
    if matrix.shape != (steps, 1) or not np.isfinite(matrix).all():
        raise ValueError("Invalid Model100 dependence GLR")
    return matrix


def prefix_parity(history, online):
    full = glr_dependence_features(history, online)
    for cut in (1, 6, 8, 12, 16, 64, 128, 256, 768):
        if cut <= len(online):
            np.testing.assert_array_equal(
                glr_dependence_features(history, online[:cut])[-1], full[cut - 1])
    return full
