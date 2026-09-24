"""Model145: convert the 2025 Structural Break Challenge data (allowed external data, see readme)
into this year's real-time format.

2025 format: X_train.parquet, MultiIndex (id, time), columns value, period (0 = before the known
boundary, 1 = after); y_train.parquet: boolean per id = "a break happened AT the boundary".
2026 format: period 1 = history (break-free, 1,000-5,000 points), period 2 = online (10-1,000
points, at most one break), label tau = index of the first post-break online point (-1 = none).

Conversion of one 2025 series (before B, after A):
  online length n drawn from the empirical online-length distribution of the 2026 training set;
  number of post-boundary points m drawn so that the break position tau = n - m follows the 2026
  tau/length distribution (series with y = False get tau = -1 but are cut the same way, so that the
  cut itself carries no label information);
  history = B[:-(n - m)], online = B[-(n - m):] + A[:m].  Series whose history would be shorter
  than 1,000 points or whose segments are too short are dropped.
The values are re-standardized exactly like the 2026 data is delivered (checked in `stats`).

Usage: python models/model145/convert_2025.py        -> data/external_2025/realtime_X.parquet, realtime_y.parquet
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/external_2025"
SEED = 145


def empirical_2026():
    """Online lengths and relative break positions of this year's training series."""
    frame = pd.read_parquet(ROOT / "train/X_train.parquet", columns=["period"])
    ids = frame.index.get_level_values(0).to_numpy(); period = frame["period"].to_numpy()
    online = pd.Series(period == 2).groupby(ids).sum()
    tau = pd.read_parquet(ROOT / "train/y_train_index.parquet")["tau_index"].reindex(online.index)
    rel = (tau[tau >= 0] / online[tau >= 0]).to_numpy()
    return online.to_numpy(), rel


def main():
    x = pd.read_parquet(SRC / "X_train.parquet"); y = pd.read_parquet(SRC / "y_train.parquet").iloc[:, 0]
    lengths, rel = empirical_2026()
    rng = np.random.default_rng(SEED)
    rows, labels, dropped = [], {}, 0
    for sid, g in x.groupby(level=0, sort=False):
        v = g["value"].to_numpy(np.float64); p = g["period"].to_numpy()
        before, after = v[p == 0], v[p == 1]
        n = int(rng.choice(lengths)); r = float(rng.choice(rel))
        pre = max(1, int(round(r * n))); post = n - pre
        if post < 1 or post > len(after) or len(before) - pre < 1000:
            dropped += 1; continue
        hist, online = before[:-pre], np.r_[before[-pre:], after[:post]]
        new_id = 10_000_000 + int(sid)
        rows.append(pd.DataFrame({"id": new_id, "time": np.arange(len(hist) + n), "value": np.r_[hist, online],
                                  "period": np.r_[np.ones(len(hist), np.int8), np.full(n, 2, np.int8)]}))
        labels[new_id] = pre if bool(y.loc[sid]) else -1
    out = pd.concat(rows).set_index(["id", "time"])
    out.to_parquet(SRC / "realtime_X.parquet")
    pd.DataFrame({"tau_index": pd.Series(labels)}).to_parquet(SRC / "realtime_y.parquet")
    print(f"converted {len(labels)} series ({dropped} dropped), break share {np.mean([t >= 0 for t in labels.values()]):.3f}")


if __name__ == "__main__":
    if not (SRC / "X_train.parquet").exists():
        sys.exit(f"missing {SRC / 'X_train.parquet'} - the user has to join the 2025 competition / open benchmark first")
    main()
