"""Fitting-fold-only, TSAUC-weighted Fisher selection on late online rows."""
from __future__ import annotations

import numpy as np

from models.model20.run_model20 import reference_weights


def late_fisher_names(layout, real_cache, parent, fold, start=160, stop=640,
                      count=256):
    union = parent["contract"]["union_names"]
    fit = np.asarray(parent["plans"][fold]["fit_rows"], dtype=np.int64)
    if len(fit) != 400_000 or real_cache.shape != (layout.row_count, len(union)):
        raise ValueError("Unexpected Model20 training rows or feature cache")
    fitting = (layout.row_fold != fold) & (layout.row_fold != (fold + 1) % 5)
    late = (layout.time_online >= start) & (layout.time_online < stop)
    if not fitting[fit].all():
        raise ValueError("Feature ranking would use held or stopping fold")
    rows = fit[late[fit]]
    y, t = layout.target[rows], layout.time_online[rows]
    reference = fitting & late
    weights = reference_weights(y, t, layout.target[reference],
        layout.time_online[reference]).astype(np.float64)
    if len(rows) < 50_000 or len(np.unique(y)) != 2:
        raise ValueError("Insufficient late training examples")
    w0, w1 = weights * (y == 0), weights * (y == 1)
    n0, n1 = w0.sum(), w1.sum()
    if min(n0, n1) <= 0:
        raise ValueError("No weighted late-time positive-negative pairs")
    matrix = np.asarray(real_cache[rows], dtype=np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("Nonfinite Model20 detector feature")
    score = np.zeros(len(union), dtype=np.float64)
    for left in range(0, len(union), 64):
        right = min(left + 64, len(union))
        block = matrix[:, left:right].astype(np.float64)
        mean0 = np.einsum("ij,i->j", block, w0) / n0
        mean1 = np.einsum("ij,i->j", block, w1) / n1
        var0 = np.einsum("ij,i->j", (block - mean0) ** 2, w0) / n0
        var1 = np.einsum("ij,i->j", (block - mean1) ** 2, w1) / n1
        score[left:right] = np.abs(mean1 - mean0) / np.sqrt(var0 + var1 + 1e-8)
    if not np.isfinite(score).all() or count > len(score):
        raise ValueError("Invalid late Fisher ranking")
    selected = np.lexsort((np.arange(len(score)), -score))[:count]
    names = [union[int(index)] for index in selected]
    diagnostics = dict(late_rows=len(rows), positive_rows=int(y.sum()),
        selected_score_min=float(score[selected[-1]]),
        selected_score_max=float(score[selected[0]]),
        overlap_original=len(set(names) & set(parent["contract"]["names"][fold])))
    return names, diagnostics
