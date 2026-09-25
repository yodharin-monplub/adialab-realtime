"""Model172: SYNTHETIC sequences for the neural student.

The tree student draws 40% of its training weight from synthetic series (glued splices and re-cut real
series); the neural student has only ever seen real ones, and Model170 showed the network is starved of
series.  This builds full per-step sequences for the synthetic series in the GRU's own input space:
  inputs   the 102 contract inputs (61 whitening + 40 detector columns + log1p_t), standardized and
           clipped exactly like models/model124/export_data.py did
  targets  0.5*y + 0.5*teacher, where the teacher is the Model157 oracle teacher of the fold that the
           parent series does NOT belong to (full_teacherB{j}), applied at EVERY step
The feature code computes all steps internally anyway (Model147 subsampled afterwards), so a full
sequence costs the same CPU as the 50 sampled rows did.
  python models/model172/build_synth_sequences.py recuts [--tag ""] [--workers 6] [--limit 0]
  python models/model172/build_synth_sequences.py splices [--folds 0,1,2,3,4] [--workers 6]
Output: models/model172/data/seq_{kind}{tag}.npz  (X float16, soft float16, target, time_online,
        series_position, row_fold)
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
CONTRACT = json.loads((ROOT / "models/model124/data/input_contract.json").read_text())
PROTOCOL = json.loads((ROOT / "models/model84/runs/e1e98a6e8af2df25/protocol.json").read_text())


def init(det_names, gru_cols, student_cols):
    import lightgbm as lgb
    from models.model20.features import selected_features
    from models.model57.whitening import whitening_features
    from models.model109.teacher_features import series_features as tf109
    from models.model157.oracle_bayes import series_oracle
    work.det_names = det_names
    work.gru_cols = gru_cols          # 40 detector columns, indices into the 439 detector block
    work.student_cols = student_cols  # fold-0 student columns, indices into the 500-column union
    work.fns = (selected_features, whitening_features, tf109, series_oracle)
    work.teachers = {j: lgb.Booster(model_file=str(ROOT / f"models/model152/runs/full_teacherB{j}.txt")) for j in range(5)}
    work.mu = np.asarray(CONTRACT["mu"], np.float32); work.sd = np.asarray(CONTRACT["sd"], np.float32)


def work(args):
    """-> (inputs 102 float16 standardized, soft float16) for every step of one synthetic series."""
    history, online, target, teacher_fold = args
    selected_features, whitening_features, tf109, series_oracle = work.fns
    h32, o32 = np.asarray(history, np.float32), np.asarray(online, np.float32)
    h64, o64 = h32.astype(np.float64), o32.astype(np.float64)
    det = np.asarray(selected_features(h32, o32, work.det_names), np.float32)      # (T, 439)
    white = np.asarray(whitening_features(h64, o64), np.float32)                   # (T, 61)
    T = len(o32)
    X = np.empty((T, 102), np.float32)
    X[:, :61] = white
    X[:, 61:101] = det[:, work.gru_cols]
    X[:, 101] = np.log1p(np.arange(T, dtype=np.float32))
    np.nan_to_num(X, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
    X = np.clip((X - work.mu) / work.sd, -CONTRACT["clip"], CONTRACT["clip"]).astype(np.float16)
    union = np.hstack([det, white])                                               # detector block then whitening block
    teacher_in = np.hstack([union[:, work.student_cols], tf109(h64, o64), series_oracle(h64, o64)])
    np.nan_to_num(teacher_in, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
    prob = work.teachers[teacher_fold].predict(teacher_in)
    soft = (0.5 * target.astype(np.float32) + 0.5 * prob).astype(np.float16)
    return X, soft


def recut_tasks(tag, pool, layout, limit):
    """Re-cut series: same parents / shifts as the Model147 set, but every step is kept."""
    rec = np.load(ROOT / f"models/model147/data/recuts{tag}.npz")
    parent, shift, fold = rec["parent"], rec["shift"], rec["fold"]
    starts = np.flatnonzero(np.r_[True, (parent[1:] != parent[:-1]) | (shift[1:] != shift[:-1])])
    tasks, meta = [], []
    for a in starts[:limit] if limit else starts:
        p, s, f = int(parent[a]), int(shift[a]), int(fold[a])
        h, o = np.asarray(pool[p][0], np.float64), np.asarray(pool[p][1], np.float64)
        hh, oo = np.r_[h, o[:s]], o[s:]
        tau = int(layout.tau_index[p]) - s
        target = (np.arange(len(oo)) >= tau).astype(np.int8) if tau >= 0 else np.zeros(len(oo), np.int8)
        tasks.append((hh.astype(np.float32), oo.astype(np.float32), target, f))   # teacher f never saw fold f
        meta.append((f, target))
    return tasks, meta


def splice_tasks(folds, pool, limit):
    """Glued series of the deployed plans; the teacher of fold v never saw plan v's donors."""
    from models.model18.augmentation import assemble
    tasks, meta = [], []
    for v in folds:
        pairs = PROTOCOL["plans"][v]["pairs"]
        for pair in (pairs[:limit] if limit else pairs):
            h, o, y, _ = assemble(pool, pair, aligned=True)
            tasks.append((np.asarray(h, np.float32), np.asarray(o, np.float32), y.astype(np.int8), v))
            meta.append((v, y.astype(np.int8)))
    return tasks, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["recuts", "splices"])
    ap.add_argument("--tag", default=""); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--folds", default="0,1,2,3,4"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"seq_{a.kind}{a.tag}.npz"
    if out.exists():
        print("exists", out.name); return
    import models.model111.run_model111 as m111
    _, protocol, layout, real, pool, _ = m111.load_everything()
    union_names = protocol["contract"]["union_names"]
    det_names = [n for n in union_names if not n.startswith("white_")]
    gru_cols = np.asarray(CONTRACT["detector_union_indices"])          # indices into the detector block
    student_cols = np.asarray([union_names.index(n) for n in protocol["contract"]["names"][0]])
    t0 = time.time()
    tasks, meta = (recut_tasks(a.tag, pool, layout, a.limit) if a.kind == "recuts"
                   else splice_tasks([int(f) for f in a.folds.split(",")], pool, a.limit))
    print(f"{a.kind}{a.tag}: {len(tasks)} series, {sum(len(m[1]) for m in meta):,} steps ({time.time() - t0:.0f}s)", flush=True)
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
