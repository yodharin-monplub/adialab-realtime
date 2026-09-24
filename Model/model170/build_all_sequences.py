"""Model170: sequence data for an ALL-SERIES neural student (development + holdout series).

The tree student already trains on all 10,000 labelled series (Model154); the neural part still sees only
the development ones.  This builds the same sequence file for dev + holdout:
  inputs   the 102 GRU inputs, built for the holdout rows exactly as the bundle does
           (models/model153/offline_gru.gru_inputs, parity 5e-7) and taken as-is for the dev rows
  targets  0.5*y + 0.5*teacher.  Dev rows: Model161's oracle soft targets (softB, cross-fitted).
           Holdout rows: the dev-only oracle teacher (models/model154/runs/hold_teacherB.txt) applied to
           EVERY holdout row - out-of-sample for those series, since it never saw them.
Output: models/model170/data/sequences_all.npz  (X, soft, target, time_online, row_fold, series_position)
        row_fold is -1 for holdout rows so any fold model can train on them.
Usage: python models/model170/build_all_sequences.py
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(1, str(ROOT / "models/model153")); sys.path.insert(2, str(ROOT / "models/model152"))

import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"


def holdout_teacher_features():
    """25 Model109 features for EVERY holdout row (hold_rows.npz has them only for the sampled rows)."""
    cache = OUT / "holdout_teacher_feat.npy"
    if cache.exists():
        return np.load(cache)
    import pandas as pd
    from models.model109.teacher_features import series_features as tf109
    ref = np.load(ROOT / "models/model123/holdout_predictions.npz")
    frame = pd.read_parquet(ROOT / "train/X_train.parquet")
    starts = np.flatnonzero(ref["step"] == 0); ends = np.r_[starts[1:], len(ref["step"])]
    X = np.empty((len(ref["step"]), 25), np.float32)
    for sid, a, b in zip(ref["ids"], starts, ends):
        s = frame.loc[int(sid)]
        X[a:b] = tf109(s.loc[s.period == 1, "value"].to_numpy(np.float64), s.loc[s.period == 2, "value"].to_numpy(np.float64))
    OUT.mkdir(parents=True, exist_ok=True); np.save(cache, X)
    return X


def main():
    import json
    import lightgbm as lgb
    import offline_gru as og
    import offline_holdout as oh
    from tsauc import tsauc
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "sequences_all.npz"
    if out.exists():
        print("exists", out); return
    t0 = time.time()
    dev = np.load(ROOT / "models/model124/data/sequences.npz")
    softB = np.load(ROOT / "models/model161/data/softB.npy")            # (5, dev rows)
    ref = np.load(ROOT / "models/model123/holdout_predictions.npz")
    # holdout GRU inputs, standardized exactly like the bundle
    b = og.load_bundle()
    Xh = og.gru_inputs(b).astype(np.float16)
    print(f"holdout inputs {Xh.shape} ({time.time() - t0:.0f}s)", flush=True)
    # holdout soft targets from the dev-only oracle teacher
    protocol = json.loads((ROOT / "models/model84/runs/e1e98a6e8af2df25/protocol.json").read_text())
    union = protocol["contract"]["union_names"]
    col0 = np.asarray([union.index(n) for n in protocol["contract"]["names"][0]])
    XH = np.load(oh.H / "X.npy", mmap_mode="r")
    feats = np.hstack([np.asarray(XH[:, :500])[:, col0], holdout_teacher_features(),
                       np.load(ROOT / "models/model157/data/holdout.npy")])
    booster = lgb.Booster(model_file=str(ROOT / "models/model154/runs/hold_teacherB.txt"))
    prob = np.empty(len(feats), np.float64)
    for lo in range(0, len(feats), 200_000):
        prob[lo:lo + 200_000] = booster.predict(feats[lo:lo + 200_000])
    print(f"holdout teacher TS-AUC on ALL holdout rows {tsauc(ref['target'], prob, ref['step']):.5f} "
          f"(out-of-sample) ({time.time() - t0:.0f}s)", flush=True)
    yh = ref["target"].astype(np.float32)
    soft_h = (0.5 * yh + 0.5 * prob).astype(np.float16)
    # series positions: keep dev ones, give the holdout series new ids after them
    pos_dev = dev["series_position"].astype(np.int32)
    starts = np.flatnonzero(ref["step"] == 0)
    pos_h = np.repeat(np.arange(len(starts), dtype=np.int32) + pos_dev.max() + 1,
                      np.diff(np.r_[starts, len(ref["step"])]))
    X = np.concatenate([dev["X"], Xh]); Xh = None
    soft = np.concatenate([softB, np.repeat(soft_h[None, :], 5, axis=0)], axis=1)
    np.savez(out, X=X, soft=soft,
             target=np.r_[dev["target"], ref["target"].astype(np.int8)].astype(np.int8),
             time_online=np.r_[dev["time_online"], ref["step"].astype(np.int16)].astype(np.int16),
             row_fold=np.r_[dev["row_fold"], np.full(len(ref["step"]), -1, np.int8)].astype(np.int8),
             series_position=np.r_[pos_dev, pos_h])
    print(f"saved {out.name}: {X.shape} rows ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
