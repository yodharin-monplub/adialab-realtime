"""Model165: EXTRA splice sets (more glued series per fold), the splice analogue of Model163's extra re-cut
sets.  Every student so far sees the same 1,000 spliced series per fold (the frozen Model20 plan); pooling
several re-cut sets gave the last big gain once the rows got oracle-teacher soft targets, so more splice
variety may pay too (the 2025 "8x pairing" test failed with HARD labels).

Per fold k: new pairs are drawn with models/model18/augmentation.select_pairs from the fitting donors only
(never fold k or the stack-reserved fold k+1), assembled exactly like the deployed splices (aligned=True),
and ROWS sampled steps get:
  500 union columns (439 detector + 61 whitening), 27 whitened split tests, 32 teacher inputs
  (25 Model109 + 7 Model157 oracle features) so the rows can get soft targets.
Output: models/model165/data/splice2_fold{k}{tag}.npz
Usage: python models/model165/build_splices2.py [--count 3000] [--rows 50] [--seed 165] [--tag ""] [--workers 6] [--folds 0,1,2,3,4]
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
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"


def features_for(args):
    history, online, keep = args
    os.environ["SP_WHITEN"] = "1"
    from models.model20.features import selected_features
    from models.model57.whitening import whitening_features
    from models.model109.teacher_features import series_features as tf109
    from models.model144.split_tests import series_features as split_features
    from models.model157.oracle_bayes import series_oracle
    h32, o32 = np.asarray(history, np.float32), np.asarray(online, np.float32)
    h64, o64 = h32.astype(np.float64), o32.astype(np.float64)
    det = np.asarray(selected_features(h32, o32, features_for.det_names), np.float32)
    white = np.asarray(whitening_features(h64, o64), np.float32)
    union = np.hstack([det[keep], white[keep]])
    split = np.asarray(split_features(h64, o64), np.float32)[keep]
    teach = np.hstack([tf109(h64, o64), series_oracle(h64, o64)])[keep]
    return union, split, teach.astype(np.float32)


def init(det_names):
    features_for.det_names = det_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=3000); ap.add_argument("--rows", type=int, default=50)
    ap.add_argument("--seed", type=int, default=165); ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--folds", default="0,1,2,3,4")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    import models.model111.run_model111 as m111
    from models.model18.augmentation import assemble, select_pairs
    _, protocol, layout, real, pool, _ = m111.load_everything()
    union_names = protocol["contract"]["union_names"]
    det_names = [n for n in union_names if not n.startswith("white_")]
    tau, sfold = layout.tau_index, layout.inner_series_fold
    for k in [int(f) for f in a.folds.split(",")]:
        out = OUT / f"splice2_fold{k}{a.tag}.npz"
        if out.exists():
            print("exists", out.name, flush=True); continue
        donors = {p: pool[p] for p in pool if sfold[p] >= 0 and sfold[p] not in (k, (k + 1) % 5)}
        pairs, stats = select_pairs(donors, tau, sfold, k, a.seed + k, count=a.count, reuse=5)
        rng = np.random.default_rng(a.seed + 100 + k)
        tasks, meta = [], []
        for pair in pairs:
            h, o, y, steps = assemble(donors, pair, aligned=True)
            keep = np.sort(rng.choice(len(o), min(a.rows, len(o)), replace=False))
            tasks.append((h, o, keep)); meta.append((y[keep].astype(np.int8), keep.astype(np.int32)))
        total = sum(len(m[1]) for m in meta)
        print(f"fold {k}{a.tag}: {len(tasks)} splices ({stats['accepted']} accepted), {total:,} rows", flush=True)
        XU = np.empty((total, len(union_names)), np.float32); XS = np.empty((total, 27), np.float32)
        XT = np.empty((total, 32), np.float32); at = 0; t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn"),
                                 initializer=init, initargs=(det_names,)) as ex:
            for i, (u, s, tt) in enumerate(ex.map(features_for, tasks, chunksize=4)):
                XU[at:at + len(u)] = u; XS[at:at + len(u)] = s; XT[at:at + len(u)] = tt; at += len(u)
                if (i + 1) % 500 == 0:
                    print(f"  {i + 1}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
        assert at == total
        for arr in (XU, XS, XT):
            np.nan_to_num(arr, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
        np.savez(out, X_union=XU, X_split=XS, teacher=XT,
                 target=np.concatenate([m[0] for m in meta]), time_online=np.concatenate([m[1] for m in meta]),
                 stats=json.dumps(stats))
        print(f"saved {out.name} {XU.shape} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
