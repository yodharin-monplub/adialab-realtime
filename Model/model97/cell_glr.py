"""Causal maximum multinomial GLR for distributional cell changes."""
from __future__ import annotations

import numpy as np
from scipy.special import ndtri

from models.model89.bayes_evidence import whitened_innovations


AGE_NODES = (4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192,
             256, 384, 512, 768)
N_CELLS = 8
EDGES = ndtri(np.arange(1, N_CELLS, dtype=np.float64) / N_CELLS)
FEATURE_NAMES = ("white_cell8_glr_max_tau",)


def _cell_glr(counts):
    c = np.asarray(counts, dtype=np.float64)
    n = c.sum(axis=1, keepdims=True)
    expected = n / N_CELLS
    term = np.zeros_like(c)
    positive = c > 0.0
    term[positive] = c[positive] * np.log(c[positive] / np.broadcast_to(expected, c.shape)[positive])
    return term.sum(axis=1)


def cell_glr_features(history, online):
    _, z = whitened_innovations(history, online)
    steps = len(z)
    if not steps:
        return np.empty((0, 1), dtype=np.float32)
    cell = np.searchsorted(EDGES, z, side="right")
    one_hot = np.eye(N_CELLS, dtype=np.int32)[cell]
    prefix = np.vstack((np.zeros((1, N_CELLS), dtype=np.int32),
                        np.cumsum(one_hot, axis=0, dtype=np.int32)))
    result = np.zeros(steps, dtype=np.float64)
    for node in AGE_NODES:
        if node > steps:
            break
        counts = prefix[node:] - prefix[:-node]
        result[node - 1:] = np.maximum(result[node - 1:], _cell_glr(counts))
    ages = np.arange(1, steps + 1)
    eligible = ages >= AGE_NODES[0]
    result[eligible] = np.maximum(result[eligible], _cell_glr(prefix[1:][eligible]))
    matrix = np.ascontiguousarray(np.clip(result, 0.0, 100.0)[:, None], dtype=np.float32)
    if matrix.shape != (steps, 1) or not np.isfinite(matrix).all():
        raise ValueError("Invalid Model97 cell GLR")
    return matrix


def prefix_parity(history, online):
    full = cell_glr_features(history, online)
    for cut in (1, 4, 6, 8, 16, 64, 128, 256, 768):
        if cut <= len(online):
            np.testing.assert_array_equal(
                cell_glr_features(history, online[:cut])[-1], full[cut - 1])
    return full
