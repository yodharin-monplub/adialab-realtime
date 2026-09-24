"""Model110: Model109 distillation with a stronger look-ahead teacher.

The teacher additionally sees the top-60 causal features (Model109 fold-0
hard-student gain order) evaluated at future rows t+32, t+128 and the final
online row.  Student rows, weights, settings and the nested protocol are
identical to Model109, so Model109's hard arm is the direct baseline.

Usage: python models/model110/run_model110.py [fold ...]
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import models.model109.run_model109 as base  # noqa: E402

HERE = Path(__file__).resolve().parent
TOP = 60
HORIZONS = (32, 128, None)  # None = final online row


class LookaheadData(base.Data):
    def __init__(self):
        super().__init__()
        order = np.load(ROOT / "models/model109/runs/fold0_hard_gain_order.npy")
        self.top = np.sort(order[:TOP])
        pos = self.layout.series_position
        starts = np.flatnonzero(np.r_[True, pos[1:] != pos[:-1]])
        lengths = np.diff(np.r_[starts, self.layout.row_count])
        self.row_start = np.repeat(starts, lengths)
        self.row_last = np.repeat(starts + lengths - 1, lengths)

    def top_causal(self, rows: np.ndarray) -> np.ndarray:
        order = np.argsort(rows)
        full = self.causal(rows)  # causal() sorts internally
        out = np.empty((len(rows), TOP), dtype=np.float32)
        out[order] = full[:, self.top]
        return out

    def privileged(self, rows: np.ndarray) -> np.ndarray:
        rows = np.sort(rows)
        parts = [np.asarray(self.teacher[rows])]
        for horizon in HORIZONS:
            future = self.row_last[rows] if horizon is None else \
                np.minimum(rows + horizon, self.row_last[rows])
            parts.append(self.top_causal(future))
        return np.hstack(parts)


def main():
    base.RUNS = HERE / "runs"
    base.RUNS.mkdir(parents=True, exist_ok=True)
    base.ARMS = {"mix": 0.5, "mix25": 0.25}
    folds = [int(a) for a in sys.argv[1:]] or [0, 1, 2, 3, 4]
    data = LookaheadData()
    for k in folds:
        base.run_fold(data, k)


if __name__ == "__main__":
    main()
