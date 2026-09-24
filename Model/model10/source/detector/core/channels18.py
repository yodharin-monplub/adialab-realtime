"""Normalized per-observation channels shared by detectors 19 and 20.

The original experimental files import ``channels18`` but the helper was not
retained with the detector sources.  This module reconstructs the contract
from their streaming implementations: one column per :mod:`detectors3`
aspect, divided by that series' history-only null scale and clipped before a
learned projection consumes it.

Both functions are causal.  Calibration uses only ``x_hist`` and every row of
``compute_channels_offline`` depends only on the corresponding online value
and earlier online values (for lagged aspects).
"""

from __future__ import annotations

import numpy as np

import detectors3 as d3
import detectors15 as d15


CLIP = 12.0
N_CHANNELS = len(d3.ASPECTS)


def _stack_normalized(increments, scales, *, align_tail=False):
    """Return channels in the stable ``d3.ASPECTS`` order."""

    arrays = {
        aspect: np.asarray(increments[aspect], dtype=np.float64)
        for aspect in d3.ASPECTS
    }
    if align_tail:
        # AR residual calibration drops its five warm-up observations, whereas
        # the other history channels retain the complete history.  Keep their
        # common, most-recent interval exactly as detectors19 originally did.
        length = min(len(values) for values in arrays.values())
        arrays = {aspect: values[-length:] for aspect, values in arrays.items()}
    matrix = np.column_stack(
        [arrays[aspect] / scales[aspect] for aspect in d3.ASPECTS]
    )
    return np.clip(matrix, -CLIP, CLIP)


def compute_history_channels(x_hist):
    """History innovations used only to calibrate a projection's null scale."""

    hs, history_increments, _ = d15._hist_and_online_increments(x_hist)
    return _stack_normalized(history_increments, hs.s, align_tail=True)


def compute_channels_offline(x_hist, x_online):
    """Compute the normalized d3 innovation channel for each online row."""

    hs, _, online_increments = d15._hist_and_online_increments(x_hist, x_online)
    return _stack_normalized(online_increments, hs.s)
