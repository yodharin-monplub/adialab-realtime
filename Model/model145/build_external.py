"""Model145: turn the 2025 Structural Break Challenge series (allowed external data) into real-time
training rows with the student's causal columns.

Source: data/ADIA Lab Structural Break Challenge/{X_train,y_train}.parquet (identical to the Open
Benchmark set; 10,001 series, boundary known, 29% breaks, no series shared with this year's data).
Cut of one 2025 series (before B, after A, label y):
  draw (online length n, break position tau) from this year's BREAK series (empirical joint);
  pre = tau points come from the end of B, post = n - tau from the start of A; the history is the
  rest of B and must keep >= 1,000 points (up to 20 draws, else the series is dropped).
  y = True  -> tau_index = pre (a real break at a known step);  y = False -> no break (-1).
  The cut is drawn the same way for both labels, so the cut itself carries no label information.
Rows: ROWS sampled online steps per series; columns in the order of the protocol's union
(detector columns via models.model20.features.selected_features + 61 whitening columns) plus the 27
whitened split-test columns of Model144 (kept in a separate array).
Output: models/model145/data/external.npz  (X_union, X_split, target, time_online, series, tau)
Usage: python models/model145/build_external.py [--rows 50] [--workers 6] [--limit N]
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

for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_n] = "1"
os.environ["SP_WHITEN"] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "data"
SRC = ROOT / "data/ADIA Lab Structural Break Challenge"
MIN_HISTORY = 1000


def features_for(args):
    history, online, keep, det_names = args
    from models.model57.whitening import whitening_features
    from models.model20.features import selected_features
    from models.model144.split_tests import series_features
    det = np.asarray(selected_features(np.asarray(history, np.float32), np.asarray(online, np.float32), det_names), np.float32)
    white = np.asarray(whitening_features(np.asarray(history, np.float64), np.asarray(online, np.float64)), np.float32)
    split = series_features(np.asarray(history, np.float64), np.asarray(online, np.float64))
    return np.hstack([det[keep], white[keep]]), split[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=50)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=145)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    protocol = json.loads((ROOT / "models/model84/runs/e1e98a6e8af2df25/protocol.json").read_text())
    union = protocol["contract"]["union_names"]
    det_names = [n for n in union if not n.startswith("white_")]
    assert union == det_names + [n for n in union if n.startswith("white_")], "union must be detector block then whitening block"

    rt = pd.read_parquet(ROOT / "train/X_train.parquet", columns=["period"])
    rid = rt.index.get_level_values(0).to_numpy()
    n_online = pd.Series(rt["period"].to_numpy() == 2).groupby(rid).sum()
    tau_rt = pd.read_parquet(ROOT / "train/y_train_index.parquet")["tau_index"].reindex(n_online.index)
    pairs = np.column_stack((n_online[tau_rt >= 0].to_numpy(), tau_rt[tau_rt >= 0].to_numpy())).astype(int)
    del rt

    x = pd.read_parquet(SRC / "X_train.parquet"); y = pd.read_parquet(SRC / "y_train.parquet").iloc[:, 0]
    ids = x.index.get_level_values(0).to_numpy(); v = x["value"].to_numpy(np.float64); p = x["period"].to_numpy()
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]]); ends = np.r_[starts[1:], len(ids)]
    rng = np.random.default_rng(a.seed)
    tasks, meta, dropped = [], [], 0
    for s, e in zip(starts, ends):
        sid = int(ids[s]); vv, pp = v[s:e], p[s:e]
        before, after = vv[pp == 0], vv[pp == 1]
        for _ in range(20):
            n, pre = pairs[rng.integers(len(pairs))]
            if pre <= len(before) - MIN_HISTORY and 1 <= n - pre <= len(after) and pre >= 1:
                break
        else:
            dropped += 1; continue
        history, online = before[:-pre], np.r_[before[-pre:], after[:n - pre]]
        if not (np.isfinite(history).all() and np.isfinite(online).all()) or history.std() <= 0:
            dropped += 1; continue
        keep = np.sort(rng.choice(n, min(a.rows, n), replace=False))
        tau = int(pre) if bool(y.loc[sid]) else -1
        target = (np.arange(n) >= tau).astype(np.int8) if tau >= 0 else np.zeros(n, np.int8)
        tasks.append((history.astype(np.float32), online.astype(np.float32), keep, det_names))
        meta.append((sid, tau, keep, target[keep]))
        if a.limit and len(tasks) >= a.limit:
            break
    total = sum(len(m[2]) for m in meta)
    print(f"{len(tasks)} series converted ({dropped} dropped), {total:,} rows, break share {np.mean([m[1] >= 0 for m in meta]):.3f}, "
          f"positive rows {np.mean(np.concatenate([m[3] for m in meta])):.3f}", flush=True)
    XU = np.empty((total, len(union)), np.float32); XS = np.empty((total, 27), np.float32); at = 0; t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn")) as ex:
        for i, (fu, fs) in enumerate(ex.map(features_for, tasks, chunksize=8)):
            XU[at:at + len(fu)] = fu; XS[at:at + len(fs)] = fs; at += len(fu)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
    np.nan_to_num(XU, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
    np.savez(OUT / ("external_smoke.npz" if a.limit else "external.npz"), X_union=XU, X_split=XS,
             target=np.concatenate([m[3] for m in meta]), time_online=np.concatenate([m[2] for m in meta]).astype(np.int32),
             series=np.concatenate([np.full(len(m[2]), m[0], np.int32) for m in meta]),
             tau=np.concatenate([np.full(len(m[2]), m[1], np.int32) for m in meta]))
    print(f"saved {XU.shape} + {XS.shape} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
