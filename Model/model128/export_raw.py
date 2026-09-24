"""Model128: raw-value sequence inputs (no engineered features) for a GRU student.

Per online row: v (history-standardized value, as stored), v^2, |v|, sign(v),
the previous value v[t-1] (0 at t=0) and log1p(t) -> 6 inputs, float16.
Soft targets are copied from models/model124/data/sequences.npz (fold-clean
Model111 teacher targets) so the row order is identical.
Output: models/model124/data/sequences_raw.npz + sequences_raw_contract.json
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.model10.sources import load_development_layout  # noqa: E402

OUT = ROOT / "models/model124/data"


def main():
    L = load_development_layout(ROOT)
    frame = pd.read_parquet(ROOT / "train/X_train.parquet")
    ids = frame.index.get_level_values(0).to_numpy()
    values = frame["value"].to_numpy().astype(np.float32)
    period = frame["period"].to_numpy()
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    ends = np.r_[starts[1:], len(ids)]
    bounds = {int(ids[s]): (s, e) for s, e in zip(starts, ends)}
    v = np.empty(L.row_count, dtype=np.float32)
    prev = np.empty(L.row_count, dtype=np.float32)
    pos = L.series_position
    row_start = np.flatnonzero(np.r_[True, pos[1:] != pos[:-1]])
    row_end = np.r_[row_start[1:], L.row_count]
    for a, b in zip(row_start, row_end):
        s, e = bounds[int(L.series_ids[pos[a]])]
        online = values[s:e][period[s:e] == 2]
        if len(online) != b - a:
            raise ValueError("layout mismatch")
        v[a:b] = online
        prev[a:b] = np.r_[0.0, online[:-1]]
    t = L.time_online.astype(np.float32)
    X = np.column_stack([v, v * v, np.abs(v), np.sign(v), prev, np.log1p(t)]).astype(np.float32)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X = np.clip((X - mu) / sd, -10, 10).astype(np.float16)
    base = np.load(OUT / "sequences.npz")
    np.savez(OUT / "sequences_raw.npz", X=X, soft=base["soft"], target=base["target"],
             time_online=base["time_online"], row_fold=base["row_fold"],
             series_position=base["series_position"], series_ids=base["series_ids"])
    (OUT / "sequences_raw_contract.json").write_text(json.dumps(
        {"names": ["v", "v2", "abs_v", "sign_v", "prev_v", "log1p_t"], "mu": mu.tolist(), "sd": sd.tolist(),
         "clip": 10.0}, indent=2))
    print("saved", OUT / "sequences_raw.npz", X.shape, flush=True)


if __name__ == "__main__":
    main()
