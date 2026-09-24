"""Model155: all 500 union columns (439 detector + 61 whitening) for the splice rows of fold plan k, so a
wide student sees real values instead of NaN on splices (the NaN pilot made splices recognisable and lost
-0.012 on fold 1).  Same tasks/rows as models/model20 synthetic_tasks (aligned), same feature code as the
Model147 re-cuts.  Parity: the fold's own 317 columns must equal the cached Model20/57 splice features.
Output: models/model155/data/splice_union_fold{k}.npy  (float32, rows aligned with synthetic_block)
Usage: python models/model155/build_splice_union.py FOLD [--workers 6]
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


def work(args):
    from models.model147.build_recuts import features_for
    return features_for(args)


def main():
    fold = int(sys.argv[1]); workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 6
    out = OUT / f"splice_union_fold{fold}.npy"
    if out.exists():
        print("exists", out); return
    import models.model111.run_model111 as m111
    from models.model20.run_model20 import synthetic_tasks
    from models.model137.run_model137 import synthetic_block
    _, protocol, layout, real, pool, _ = m111.load_everything()
    union = protocol["contract"]["union_names"]
    det_names = [n for n in union if not n.startswith("white_")]
    tasks, sy, _ = synthetic_tasks(pool, protocol["plans"][fold], layout, True)
    jobs = [(np.asarray(h, np.float32), np.asarray(o, np.float32), keep, det_names) for h, o, keep in tasks]
    X = np.empty((len(sy), len(union)), np.float32); at = 0; t0 = time.time()
    print(f"fold {fold}: {len(jobs)} splices, {len(sy):,} rows", flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        for i, f in enumerate(ex.map(work, jobs, chunksize=2)):
            X[at:at + len(f)] = f; at += len(f)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    assert at == len(sy)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
    cached, _, _ = synthetic_block(protocol, layout, pool, fold, protocol["contract"]["names"][fold])
    cols = np.asarray([union.index(n) for n in protocol["contract"]["names"][fold]])
    ref = np.nan_to_num(np.asarray(cached, np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
    diff = np.abs(X[:, cols] - ref); rel = diff / (np.abs(ref) + 1e-3)
    print(f"parity vs cached splice features (fold's 317 cols): max abs {diff.max():.3e}, max rel {rel.max():.3e}, "
          f"share rel>1e-4 {np.mean(rel > 1e-4):.2e}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(out, X)
    print(f"saved {out.name} {X.shape} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
