"""Model143: export the RAW series for an end-to-end sequence model (no engineered features).

Per online row (development layout order): z = (x - history mean) / history sd, |z|, dz (first
difference, seeded with the last history point) and log1p(t) -> X (n, 4) float16, clipped to +/-20.
Per series: the last 512 history points, z-standardized, left-padded (H (series, 512) float16,
hist_len (series,)). Soft targets, labels, times, folds and series positions are copied from
Model124's export (models/model124/data/sequences.npz) so the folds and teachers are identical.
Output: models/model143/data/raw.npz (~120 MB).
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.model10.sources import load_development_layout  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "data"
HIST = 512
CLIP = 20.0


def main():
    OUT.mkdir(exist_ok=True)
    L = load_development_layout(ROOT)
    frame = pd.read_parquet(ROOT / "train/X_train.parquet")
    ids = frame.index.get_level_values(0).to_numpy(); values = frame["value"].to_numpy(np.float64); period = frame["period"].to_numpy()
    s0 = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]]); bounds = {int(ids[s]): (s, e) for s, e in zip(s0, np.r_[s0[1:], len(ids)])}
    pos = L.series_position
    starts = np.flatnonzero(np.r_[True, pos[1:] != pos[:-1]]); ends = np.r_[starts[1:], L.row_count]
    X = np.zeros((L.row_count, 4), np.float32)
    H = np.zeros((len(starts), HIST), np.float32); hist_len = np.zeros(len(starts), np.int16)
    t0 = time.time()
    for i, (a, b) in enumerate(zip(starts, ends)):
        s, e = bounds[int(L.series_ids[pos[a]])]; v, p = values[s:e], period[s:e]
        h, o = v[p == 1], v[p == 2]
        assert len(o) == b - a, (len(o), b - a)
        hm, hsd = h.mean(), h.std() + 1e-9
        zh = (h - hm) / hsd; zo = (o - hm) / hsd
        X[a:b, 0] = zo; X[a:b, 1] = np.abs(zo); X[a:b, 2] = np.diff(np.concatenate(([zh[-1]], zo)))
        X[a:b, 3] = np.log1p(L.time_online[a:b])
        tail = zh[-HIST:]; H[i, HIST - len(tail):] = tail; hist_len[i] = len(tail)
        if i % 1000 == 0:
            print(f"{i} series ({time.time() - t0:.0f}s)", flush=True)
    np.clip(X, -CLIP, CLIP, out=X); np.clip(H, -CLIP, CLIP, out=H)
    np.nan_to_num(X, copy=False); np.nan_to_num(H, copy=False)
    m124 = np.load(ROOT / "models/model124/data/sequences.npz")
    assert np.array_equal(m124["series_position"], pos.astype(np.int32))
    np.savez(OUT / "raw.npz", X=X.astype(np.float16), H=H.astype(np.float16), hist_len=hist_len,
             soft=m124["soft"], target=m124["target"], time_online=m124["time_online"],
             row_fold=m124["row_fold"], series_position=m124["series_position"], series_ids=m124["series_ids"])
    print("saved", OUT / "raw.npz", X.shape, H.shape, f"hist_len min {hist_len.min()} median {np.median(hist_len):.0f}", flush=True)


if __name__ == "__main__":
    main()
