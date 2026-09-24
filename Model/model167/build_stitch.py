"""Model167: "stitch" rows = the Model147 re-cut trick applied to the SPLICED series.

A splice is history(A) + pre-break(A) + a real post-break tail of B, with the break at tau_a.  A re-cut moves
the first s online points into the history, so the break lands at a different step.  Combining both gives
glued series whose break position differs from the deployed splices - a new synthetic type rather than more
of the same (extra splice sets alone did not help, Model165).
Per fold the deployed plan's pairs are re-assembled (aligned, exactly like the splices), re-cut once with
s uniform in [8, tau-8], and ROWS sampled steps get the 500 union columns, the 27 split tests and the 32
teacher inputs (25 Model109 + 7 Model157 oracle) so the rows can carry soft targets.
Output: models/model167/data/stitch_fold{k}.npz
Usage: python models/model167/build_stitch.py [--rows 50] [--seed 167] [--workers 6] [--folds 0,1,2,3,4]
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

from models.model165.build_splices2 import features_for, init  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=50); ap.add_argument("--seed", type=int, default=167)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--folds", default="0,1,2,3,4")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    import models.model111.run_model111 as m111
    from models.model18.augmentation import assemble
    _, protocol, layout, real, pool, _ = m111.load_everything()
    union_names = protocol["contract"]["union_names"]
    det_names = [n for n in union_names if not n.startswith("white_")]
    for k in [int(f) for f in a.folds.split(",")]:
        out = OUT / f"stitch_fold{k}.npz"
        if out.exists():
            print("exists", out.name, flush=True); continue
        rng = np.random.default_rng(a.seed + k)
        tasks, meta = [], []
        for pair in protocol["plans"][k]["pairs"]:
            h, o, _, _ = assemble(pool, pair, aligned=True)
            h = np.asarray(h, np.float64); o = np.asarray(o, np.float64)
            tau = int(pair["tau_a"])
            if tau < 24 or len(o) - tau < 8:
                continue
            s = int(rng.integers(8, tau - 8 + 1))
            history, online, new_tau = np.r_[h, o[:s]], o[s:], tau - s
            keep = np.sort(rng.choice(len(online), min(a.rows, len(online)), replace=False))
            target = (np.arange(len(online)) >= new_tau).astype(np.int8)
            tasks.append((history.astype(np.float32), online.astype(np.float32), keep))
            meta.append((target[keep], keep.astype(np.int32)))
        total = sum(len(m[1]) for m in meta)
        print(f"fold {k}: {len(tasks)} stitched series, {total:,} rows, positives {np.concatenate([m[0] for m in meta]).mean():.3f}", flush=True)
        XU = np.empty((total, len(union_names)), np.float32); XS = np.empty((total, 27), np.float32)
        XT = np.empty((total, 32), np.float32); at = 0; t0 = time.time()
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn"),
                                 initializer=init, initargs=(det_names,)) as ex:
            for i, (u, s_, tt) in enumerate(ex.map(features_for, tasks, chunksize=4)):
                XU[at:at + len(u)] = u; XS[at:at + len(u)] = s_; XT[at:at + len(u)] = tt; at += len(u)
                if (i + 1) % 250 == 0:
                    print(f"  {i + 1}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
        assert at == total
        for arr in (XU, XS, XT):
            np.nan_to_num(arr, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
        np.savez(out, X_union=XU, X_split=XS, teacher=XT,
                 target=np.concatenate([m[0] for m in meta]), time_online=np.concatenate([m[1] for m in meta]))
        print(f"saved {out.name} {XU.shape} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
