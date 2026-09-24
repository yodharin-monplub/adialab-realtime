"""Model154 step 1: turn the 1,667 sealed-holdout series into TRAINING material for the final refit
(dev + holdout = all 10,000 labelled series).  Only used for the last bundle, after every model choice was
made on the holdout; nothing is evaluated on the holdout any more once this is used.

Outputs (models/model154/data/):
  hold_rows.npz      rows = sampled holdout rows (Model20 sampled_mask, same 20% rule as the dev rows),
                     X_union (500 = 439 detector union + 61 whitening, from Model152's streaming pass - bit-exact
                     with the offline dev cache), split (27), target, time_online, sid, teacher_feat (25, Model109)
  recuts_hold{tag}.npz   Model147 re-cuts of the holdout series (same rules; seeds 147/148/149 -> "", _s148, _s149)
                     X_union (500), split (27), target, time_online, sid, shift, tau
Usage: python models/model154/prepare_holdout.py rows | recuts SEED TAG [--workers 4]
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_n] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
H = ROOT / "models/model152/holdout"


def holdout_series():
    """[(sid, history, online, tau)] in the holdout order of models/model123/holdout_predictions.npz."""
    import pandas as pd
    ref = np.load(ROOT / "models/model123/holdout_predictions.npz")
    frame = pd.read_parquet(ROOT / "train/X_train.parquet")
    starts = np.flatnonzero(ref["step"] == 0); ends = np.r_[starts[1:], len(ref["step"])]
    out = []
    for sid, a, b in zip(ref["ids"], starts, ends):
        s = frame.loc[int(sid)]
        h = s.loc[s.period == 1, "value"].to_numpy(np.float64); o = s.loc[s.period == 2, "value"].to_numpy(np.float64)
        assert len(o) == b - a
        y = ref["target"][a:b]
        tau = int(np.argmax(y)) if y.any() else -1
        out.append((int(sid), h, o, tau, a, b))
    return out, ref


def rows():
    import models.model111.run_model111  # noqa: F401  (installs the Linux stubs the legacy runners import)
    from models.model20.run_model20 import sampled_mask
    from models.model109.teacher_features import series_features
    series, ref = holdout_series()
    sid_row = np.concatenate([np.full(b - a, sid, np.int64) for sid, _, _, _, a, b in series])
    keep = np.flatnonzero(sampled_mask(sid_row, ref["step"]))
    X = np.load(H / "X.npy", mmap_mode="r")
    tf = np.concatenate([series_features(h, o) for _, h, o, _, _, _ in series])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "hold_rows.npz", rows=keep, X_union=np.asarray(X[keep][:, :500]), split=np.asarray(X[keep][:, 500:]),
             target=ref["target"][keep].astype(np.int8), time_online=ref["step"][keep].astype(np.int32),
             sid=sid_row[keep], teacher_feat=tf[keep])
    print(f"holdout rows: {len(keep):,} of {len(sid_row):,} sampled; positives {ref['target'][keep].mean():.3f}", flush=True)


def features(args):
    history, online, keep, det_names = args
    from models.model147.build_recuts import features_for
    union = features_for((history, online, keep, det_names))
    split = features_for((history, online, keep, None))
    return union, split


def recuts(seed, tag, workers):
    import json
    union = json.loads((ROOT / "models/model84/runs/e1e98a6e8af2df25/protocol.json").read_text())["contract"]["union_names"]
    det_names = [n for n in union if not n.startswith("white_")]
    series, _ = holdout_series()
    rng = np.random.default_rng(seed + 1000)
    tasks, meta = [], []
    for sid, h, o, tau, _, _ in series:
        n = len(o)
        if 0 <= tau < n:
            if tau < 24:
                continue
            s = int(rng.integers(8, tau - 8 + 1)); new_tau = tau - s
        else:
            if n < 32:
                continue
            s = int(rng.integers(8, n - 16 + 1)); new_tau = -1
        history, online = np.r_[h, o[:s]], o[s:]
        m = len(online)
        keep = np.sort(rng.choice(m, min(50, m), replace=False))
        target = (np.arange(m) >= new_tau).astype(np.int8) if new_tau >= 0 else np.zeros(m, np.int8)
        tasks.append((history.astype(np.float32), online.astype(np.float32), keep, det_names))
        meta.append((sid, s, new_tau, keep, target[keep]))
    total = sum(len(m[3]) for m in meta)
    print(f"{len(tasks)} holdout re-cuts, {total:,} rows", flush=True)
    XU = np.empty((total, 500), np.float32); XS = np.empty((total, 27), np.float32); at = 0; t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        for i, (u, sp) in enumerate(ex.map(features, tasks, chunksize=4)):
            XU[at:at + len(u)] = u; XS[at:at + len(u)] = sp; at += len(u)
            if (i + 1) % 250 == 0:
                print(f"  {i + 1}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
    np.nan_to_num(XU, copy=False, nan=0.0, posinf=1e6, neginf=-1e6); np.nan_to_num(XS, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
    rep = lambda j, dt: np.concatenate([np.full(len(m[3]), m[j], dt) for m in meta])  # noqa: E731
    np.savez(OUT / f"recuts_hold{tag}.npz", X_union=XU, split=XS, target=np.concatenate([m[4] for m in meta]),
             time_online=np.concatenate([m[3] for m in meta]).astype(np.int32), sid=rep(0, np.int64), shift=rep(1, np.int32),
             tau=rep(2, np.int32))
    print(f"saved recuts_hold{tag}.npz {XU.shape} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    a = sys.argv[1:]
    if a[0] == "rows":
        rows()
    else:
        w = int(a[a.index("--workers") + 1]) if "--workers" in a else 4
        recuts(int(a[1]), "" if a[2] == "base" else a[2], w)
