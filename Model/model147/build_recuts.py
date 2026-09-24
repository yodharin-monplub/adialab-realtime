"""Model147: "move the start" - re-cut this year's own series (user's question 2026-09-20, option A).

The first s online points of a series are clean as long as s is before the break, so they can be
handed to the history.  The result is a new, fully REAL series (nothing glued, same generator):
    history' = history + online[:s]      online' = online[s:]      tau' = tau - s   (or no break)
Every window / prefix / step feature changes because "step t" now starts s points later.
  break series  : s uniform in [8, tau - 8]          (needs tau >= 24)
  no-break      : s uniform in [8, n - 16]           (needs n >= 32)
One re-cut per development series (the 1,667 holdout series are never touched).  ROWS sampled online
steps per re-cut get the 500 union columns (439 detector + 61 whitening).  A re-cut inherits the fold of
its parent, and the runner only uses re-cuts whose parent is in the fitting folds of the held fold.
Output: models/model147/data/recuts.npz  (X_union, target, time_online, parent, fold, tau, shift)
Usage: python models/model147/build_recuts.py [--rows 50] [--workers 6] [--limit N]
"""
from __future__ import annotations

import argparse
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

HERE = Path(__file__).resolve().parent
OUT = HERE / "data"


def features_for(args):
    history, online, keep, det_names = args
    if det_names is None:                     # --split: only the 27 whitened split-test columns of Model144
        os.environ["SP_WHITEN"] = "1"
        from models.model144.split_tests import series_features
        return series_features(np.asarray(history, np.float64), np.asarray(online, np.float64))[keep]
    from models.model57.whitening import whitening_features
    from models.model20.features import selected_features
    det = np.asarray(selected_features(np.asarray(history, np.float32), np.asarray(online, np.float32), det_names), np.float32)
    white = np.asarray(whitening_features(np.asarray(history, np.float64), np.asarray(online, np.float64)), np.float32)
    return np.hstack([det[keep], white[keep]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=50)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=147)
    ap.add_argument("--tag", default="", help="output suffix for additional re-cut sets, e.g. _s148 with --seed 148")
    ap.add_argument("--flip", action="store_true", help="mirror every re-cut series around its history mean (sign-flip augmentation)")
    ap.add_argument("--split", action="store_true", help="same re-cuts (same seed), only the Model144 split-test columns -> recuts_split.npy")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    import models.model111.run_model111 as m111  # installs Linux stubs
    _, protocol, layout, real, pool, _ = m111.load_everything()
    union = protocol["contract"]["union_names"]
    det_names = [n for n in union if not n.startswith("white_")]
    assert union == det_names + [n for n in union if n.startswith("white_")]
    rng = np.random.default_rng(a.seed)
    tau_all, fold_all = layout.tau_index, layout.inner_series_fold
    tasks, meta = [], []
    for p in sorted(pool):
        if fold_all[p] < 0:
            continue                                     # sealed holdout series: never used
        h, o = np.asarray(pool[p][0], np.float64), np.asarray(pool[p][1], np.float64)
        n, tau = len(o), int(tau_all[p])
        if 0 <= tau < n:
            if tau < 24:
                continue
            s = int(rng.integers(8, tau - 8 + 1)); new_tau = tau - s
        else:
            if n < 32:
                continue
            s = int(rng.integers(8, n - 16 + 1)); new_tau = -1
        history, online = np.r_[h, o[:s]], o[s:]
        if a.flip:                              # Model150: the same real series mirrored around its history mean
            centre = 2.0 * float(h.mean())
            history, online = centre - history, centre - online
        m = len(online)
        keep = np.sort(rng.choice(m, min(a.rows, m), replace=False))
        target = (np.arange(m) >= new_tau).astype(np.int8) if new_tau >= 0 else np.zeros(m, np.int8)
        tasks.append((history.astype(np.float32), online.astype(np.float32), keep, None if a.split else det_names))
        meta.append((p, int(fold_all[p]), new_tau, s, keep, target[keep]))
        if a.limit and len(tasks) >= a.limit:
            break
    total = sum(len(m[4]) for m in meta)
    print(f"{len(tasks)} re-cuts, {total:,} rows, break share {np.mean([m[2] >= 0 for m in meta]):.3f}, "
          f"positive rows {np.mean(np.concatenate([m[5] for m in meta])):.3f}", flush=True)
    X = np.empty((total, 27 if a.split else len(union)), np.float32); at = 0; t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as ex:
        for i, f in enumerate(ex.map(features_for, tasks, chunksize=8)):
            X[at:at + len(f)] = f; at += len(f)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
    if a.split:
        saved = np.load(OUT / f"recuts{a.tag}.npz")
        assert np.array_equal(saved["parent"], np.concatenate([np.full(len(m[4]), m[0], np.int32) for m in meta]))
        assert np.array_equal(saved["time_online"], np.concatenate([m[4] for m in meta]).astype(np.int32)), "re-cuts differ from recuts.npz"
        np.save(OUT / f"recuts{a.tag}_split.npy", X); print(f"saved split columns {X.shape} in {time.time() - t0:.0f}s", flush=True)
        return
    rep = lambda j, dt: np.concatenate([np.full(len(m[4]), m[j], dt) for m in meta])  # noqa: E731
    np.savez(OUT / ("recuts_smoke.npz" if a.limit else f"recuts{a.tag}.npz"), X_union=X,
             target=np.concatenate([m[5] for m in meta]), time_online=np.concatenate([m[4] for m in meta]).astype(np.int32),
             parent=rep(0, np.int32), fold=rep(1, np.int8), tau=rep(2, np.int32), shift=rep(3, np.int32))
    print(f"saved {X.shape} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
