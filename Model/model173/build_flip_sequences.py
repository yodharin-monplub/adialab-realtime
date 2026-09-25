"""Model173: MIRRORED REAL series as sequences for the neural student.

Model172 showed that re-cut synthetic series HURT the nets (-0.0018, 2/5 folds) even though they help the
tree student: a re-cut series has its history truncated and its break moved, which distorts the time/state
prior a recurrent net builds up.  Mirroring does not touch either.  For a real series with history mean m,
    x -> 2*m - x
is still a real series with the SAME history length, the SAME break time and the same label, but every
signed feature (up/down CUSUM, signed t, tail_up/tail_down, ...) sees the opposite direction.  Model150
found the mirrored re-cuts were the best single TREE student ever measured here (+0.0088 dev, 5/5), so the
transform carries real information.

The sequence machinery (102 contract inputs, oracle-teacher soft targets) is Model172's, imported as-is;
only the task list differs.  row_fold of a mirrored series is its parent's fold, so fold k's net never
trains on the mirror of a series it is judged on (nor on the stack-reserved fold's mirrors).

  python models/model173/build_flip_sequences.py [--workers 5] [--limit 0]
Output: models/model173/data/seq_flip.npz  (same keys as Model172's files)
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
sys.path.insert(1, str(ROOT / "models/model172"))

import numpy as np  # noqa: E402
from build_synth_sequences import CONTRACT, init, work  # noqa: E402  (identical inputs and targets)

OUT = Path(__file__).resolve().parent / "data"


def flip_tasks(pool, layout, limit):
    """Every development series, mirrored around its own history mean.

    models/model20/features.load_real_pool returns a dict keyed by LAYOUT POSITION and holds only the
    8,333 series with inner_series_fold >= 0 (the sealed-holdout series are not in it), so iterate its
    own keys rather than a range.
    """
    tasks, meta = [], []
    positions = sorted(pool)
    for p in (positions[:limit] if limit else positions):
        h = np.asarray(pool[p][0], np.float64)
        o = np.asarray(pool[p][1], np.float64)
        centre = 2.0 * h.mean()
        hh, oo = centre - h, centre - o
        tau = int(layout.tau_index[p])
        target = (np.arange(len(oo)) >= tau).astype(np.int8) if tau >= 0 else np.zeros(len(oo), np.int8)
        fold = int(layout.inner_series_fold[p])          # -1 for the sealed-holdout series
        teacher = fold if fold >= 0 else 0                  # teacher j never saw fold j; holdout series
        tasks.append((hh.astype(np.float32), oo.astype(np.float32), target, teacher))
        meta.append((fold, target))                         # row_fold -1 -> every fold's net may use it
    return tasks, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "seq_flip.npz"
    if out.exists():
        print("exists", out.name)
        return
    import models.model111.run_model111 as m111
    _, protocol, layout, _real, pool, _ = m111.load_everything()
    union_names = protocol["contract"]["union_names"]
    det_names = [n for n in union_names if not n.startswith("white_")]
    gru_cols = np.asarray(CONTRACT["detector_union_indices"])
    student_cols = np.asarray([union_names.index(n) for n in protocol["contract"]["names"][0]])
    t0 = time.time()
    tasks, meta = flip_tasks(pool, layout, a.limit)
    print(f"flip: {len(tasks)} series, {sum(len(m[1]) for m in meta):,} steps ({time.time() - t0:.0f}s)", flush=True)
    Xs, softs, pos, folds_out = [], [], [], []
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn"),
                             initializer=init, initargs=(det_names, gru_cols, student_cols)) as ex:
        for i, (X, soft) in enumerate(ex.map(work, tasks, chunksize=4)):
            Xs.append(X); softs.append(soft)
            pos.append(np.full(len(X), i, np.int32)); folds_out.append(np.full(len(X), meta[i][0], np.int8))
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
    np.savez(out, X=np.concatenate(Xs), soft=np.concatenate(softs),
             target=np.concatenate([m[1] for m in meta]).astype(np.int8),
             time_online=np.concatenate([np.arange(len(m[1]), dtype=np.int16) for m in meta]),
             series_position=np.concatenate(pos), row_fold=np.concatenate(folds_out))
    print(f"saved {out.name}: {sum(len(x) for x in Xs):,} rows in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
