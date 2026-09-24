"""Model134: moving-block bootstrap replicas of the real development series.

For each real development series (donor): blocks of length B resampled with
replacement from the pre-break part (history + online[:tau]) form a new
history (same length) and a new pre-break online segment (same tau); blocks
from the post-break part (online[tau:]) form the post-break segment (same
length).  Break-free series are bootstrapped whole.  Labels: y_t = 1 for
t >= tau.  ROWS_PER_SERIES online rows are sampled per replica and given the
full causal features: the union of the five fold contracts' detector columns
(computed by the frozen feature pipeline) plus the 61 whitening columns.

Output: models/model134/data/boot.npz
  X (rows, n_union + 61) float32, names, target, time_online, donor_fold,
  donor_position, tau
Usage: python models/model134/build_bootstrap.py [--workers 6] [--rows 80] [--seed 134]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "data"
BLOCK = 48


def block_bootstrap(x, length, rng):
    """Moving-block bootstrap of x to the requested length."""
    x = np.asarray(x, np.float64)
    n = len(x)
    if n == 0 or length == 0:
        return np.zeros(length)
    b = min(BLOCK, max(1, n // 2))
    out = np.empty(length)
    at = 0
    while at < length:
        s = rng.integers(0, n - b + 1)
        take = min(b, length - at)
        out[at:at + take] = x[s:s + take]
        at += take
    return out


def make_replica(history, online, tau, rng):
    history = np.asarray(history, np.float64); online = np.asarray(online, np.float64)
    if tau < 0:
        pre = np.r_[history, online]
        new_hist = block_bootstrap(pre, len(history), rng)
        new_online = block_bootstrap(pre, len(online), rng)
    else:
        pre = np.r_[history, online[:tau]]
        post = online[tau:]
        new_hist = block_bootstrap(pre, len(history), rng)
        new_online = np.r_[block_bootstrap(pre, tau, rng), block_bootstrap(post, len(online) - tau, rng)]
    # re-standardize by the new history, as the competition does
    mu, sd = new_hist.mean(), new_hist.std() + 1e-9
    return (new_hist - mu) / sd, (new_online - mu) / sd


def features_for(args):
    history, online, keep, names = args
    from models.model57.whitening import whitening_features
    from models.model20.features import selected_features
    det = np.asarray(selected_features(np.asarray(history, np.float32), np.asarray(online, np.float32), names), np.float32)
    white = np.asarray(whitening_features(np.asarray(history, np.float64), np.asarray(online, np.float64)), np.float32)
    return np.hstack([det[keep], white[keep]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--rows", type=int, default=80)
    parser.add_argument("--seed", type=int, default=134)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    import models.model111.run_model111 as m111  # installs Linux stubs
    from models.model20.features import load_real_pool
    _, protocol, layout, real, pool, _ = m111.load_everything()
    union = protocol["contract"]["union_names"]
    fold_names = protocol["contract"]["names"]
    det_names = sorted({n for names in fold_names for n in names[:-61]}, key=union.index)
    from models.model57.whitening import FEATURE_NAMES
    all_names = det_names + list(FEATURE_NAMES)
    print(f"detector union {len(det_names)} + whitening {len(FEATURE_NAMES)} columns", flush=True)
    rng = np.random.default_rng(args.seed)
    positions = np.flatnonzero(layout.inner_series_fold >= 0)
    tasks, meta = [], []
    for p in positions:
        h, o = pool[int(p)]
        tau = int(layout.tau_index[p])
        nh, no = make_replica(h, o, tau, rng)
        keep = np.sort(rng.choice(len(no), min(args.rows, len(no)), replace=False))
        tasks.append((nh, no, keep, det_names))
        y = np.zeros(len(no), np.int8) if tau < 0 else (np.arange(len(no)) >= tau).astype(np.int8)
        meta.append((y[keep], keep, int(layout.inner_series_fold[p]), int(p), tau))
    total = sum(len(m[1]) for m in meta)
    print(f"{len(tasks)} replicas, {total:,} sampled rows", flush=True)
    X = np.empty((total, len(all_names)), np.float32)
    t0 = time.time(); at = 0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
        for i, f in enumerate(ex.map(features_for, tasks, chunksize=8)):
            X[at:at + len(f)] = f; at += len(f)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(tasks)} replicas ({time.time() - t0:.0f}s)", flush=True)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
    np.savez(OUT / "boot.npz", X=X, names=np.asarray(all_names), target=np.concatenate([m[0] for m in meta]),
             time_online=np.concatenate([m[1] for m in meta]).astype(np.int32),
             donor_fold=np.concatenate([np.full(len(m[1]), m[2], np.int8) for m in meta]),
             donor_position=np.concatenate([np.full(len(m[1]), m[3], np.int32) for m in meta]),
             tau=np.concatenate([np.full(len(m[1]), m[4], np.int32) for m in meta]))
    print(f"saved {X.shape} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
