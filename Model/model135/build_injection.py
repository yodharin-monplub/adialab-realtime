"""Model135: parametric break injection into real break-free series.

For each break-free development series (donor, fold-tagged) two replicas are
made: history unchanged; online segment = the donor's real online values with
a break injected at a random tau (uniform in [2, L-3]) of one random type:
  mean    : + delta,            |delta| ~ lognormal fitted to real |mean shift| (median .064 sd, p90 .21)
  scale   : * exp(g),           |g|     ~ lognormal fitted to real |log sd ratio| (median .11, p90 .40)
  ar      : AR(1) refilter,     |dphi|  ~ lognormal fitted to real |ac1 change| (median .075, p90 .21)
  shape   : sign(x)|x|^p with p ~ U(0.6, 1.6), rescaled to keep the variance
Signs are random. Labels y_t = 1 for t >= tau.  ROWS_PER_SERIES online rows
per replica receive the full causal features (detector union + whitening) via
the frozen pipeline, exactly as Model134.
Output: models/model135/data/inject.npz
Usage: python models/model135/build_injection.py [--workers 6] [--rows 80] [--seed 135]
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
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
TYPES = ("mean", "scale", "ar", "shape")


def lognormal_from_quantiles(median, p90):
    mu = np.log(median)
    sigma = (np.log(p90) - mu) / 1.2816
    return mu, sigma


MAG = {"mean": lognormal_from_quantiles(0.064, 0.21), "scale": lognormal_from_quantiles(0.109, 0.40),
       "ar": lognormal_from_quantiles(0.075, 0.21)}


def inject(online, tau, kind, rng):
    x = np.asarray(online, np.float64).copy()
    post = x[tau:]
    sign = rng.choice([-1.0, 1.0])
    if kind == "mean":
        mu, s = MAG["mean"]; post = post + sign * rng.lognormal(mu, s)
    elif kind == "scale":
        mu, s = MAG["scale"]; post = post * np.exp(sign * rng.lognormal(mu, s))
    elif kind == "ar":
        mu, s = MAG["ar"]; dphi = float(np.clip(sign * rng.lognormal(mu, s), -0.6, 0.6))
        # add serial dependence of +-dphi on top of the existing process, keep variance
        y = np.empty_like(post); prev = x[tau - 1] if tau > 0 else 0.0
        for i, v in enumerate(post):
            y[i] = dphi * prev + v * np.sqrt(max(1.0 - dphi * dphi, 0.05)); prev = y[i]
        post = y
    else:  # shape
        p = rng.uniform(0.6, 1.6); sd = post.std() + 1e-9
        z = np.sign(post) * np.abs(post / sd) ** p
        post = post.mean() + (z - z.mean()) / (z.std() + 1e-9) * sd
    x[tau:] = post
    return x


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
    parser.add_argument("--seed", type=int, default=135)
    parser.add_argument("--replicas", type=int, default=2)
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    import models.model111.run_model111 as m111  # installs Linux stubs
    _, protocol, layout, real, pool, _ = m111.load_everything()
    union = protocol["contract"]["union_names"]
    det_names = sorted({n for names in protocol["contract"]["names"] for n in names[:-61]}, key=union.index)
    from models.model57.whitening import FEATURE_NAMES
    all_names = det_names + list(FEATURE_NAMES)
    rng = np.random.default_rng(args.seed)
    donors = [int(p) for p in np.flatnonzero(layout.inner_series_fold >= 0) if layout.tau_index[p] < 0]
    tasks, meta = [], []
    for p in donors:
        h, o = pool[p]
        if len(o) < 8:
            continue
        for _ in range(args.replicas):
            tau = int(rng.integers(2, len(o) - 2)); kind = TYPES[rng.integers(len(TYPES))]
            no = inject(o, tau, kind, rng)
            keep = np.sort(rng.choice(len(no), min(args.rows, len(no)), replace=False))
            tasks.append((np.asarray(h, np.float64), no, keep, det_names))
            y = (np.arange(len(no)) >= tau).astype(np.int8)
            meta.append((y[keep], keep, int(layout.inner_series_fold[p]), p, tau, TYPES.index(kind)))
    total = sum(len(m[1]) for m in meta)
    print(f"{len(donors)} break-free donors -> {len(tasks)} injected series, {total:,} rows", flush=True)
    X = np.empty((total, len(all_names)), np.float32); t0 = time.time(); at = 0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as ex:
        for i, f in enumerate(ex.map(features_for, tasks, chunksize=8)):
            X[at:at + len(f)] = f; at += len(f)
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(tasks)} series ({time.time() - t0:.0f}s)", flush=True)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
    np.savez(OUT / "inject.npz", X=X, names=np.asarray(all_names), target=np.concatenate([m[0] for m in meta]),
             time_online=np.concatenate([m[1] for m in meta]).astype(np.int32),
             donor_fold=np.concatenate([np.full(len(m[1]), m[2], np.int8) for m in meta]),
             donor_position=np.concatenate([np.full(len(m[1]), m[3], np.int32) for m in meta]),
             tau=np.concatenate([np.full(len(m[1]), m[4], np.int32) for m in meta]),
             kind=np.concatenate([np.full(len(m[1]), m[5], np.int8) for m in meta]))
    print(f"saved {X.shape} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
