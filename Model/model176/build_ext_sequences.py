"""Model176: the 2025 challenge's ~10,000 REAL labelled series as sequences for the neural student.

Why this and nothing else: the nets are the one component that has responded to more REAL series
(Model170: +0.0069, 5/5, from the 1,667 holdout series), and Model172/173 showed they reject every kind of
DERIVED series (re-cuts, splices, mirrors). The 2025 Structural Break Challenge data - allowed as external
data by the rules (models/model145/readme.txt) - is the only other source of real labelled series, and it
has never been given to the nets. It failed in Model145 only as TREE rows, in an older recipe; the risk it
carries here is the same one (a different generator: the student scored just 0.56 on those rows).

Conversion is models/model145/convert_2025.py's, reproduced exactly (same seed 145, same draws from this
year's online-length and break-position distributions), so the series set is the one Model145 measured.
Sequences are built with Model172's machinery (102 contract inputs, 0.5*y + 0.5*oracle-teacher soft
targets). No external series is in any development fold, so every fold's net may train on all of them
(row_fold = -1) with no leak; teacher j = id % 5 (none of them ever saw these series).
  python models/model176/build_ext_sequences.py [--workers 5] [--limit 0]
Output: models/model176/data/seq_ext.npz
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
sys.path.insert(1, str(ROOT / "models/model172"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from build_synth_sequences import CONTRACT, init, work  # noqa: E402

RAW = ROOT / "Data/ADIA Lab Structural Break Challenge"
OUT = Path(__file__).resolve().parent / "data"
SEED = 145                                            # models/model145/convert_2025.py's seed


def convert(limit):
    """models/model145/convert_2025.main, in memory, draw for draw."""
    from models.model145.convert_2025 import empirical_2026
    x = pd.read_parquet(RAW / "X_train.parquet"); y = pd.read_parquet(RAW / "y_train.parquet").iloc[:, 0]
    lengths, rel = empirical_2026()
    rng = np.random.default_rng(SEED)
    tasks, meta, dropped = [], [], 0
    for sid, g in x.groupby(level=0, sort=False):
        v = g["value"].to_numpy(np.float64); p = g["period"].to_numpy()
        before, after = v[p == 0], v[p == 1]
        n = int(rng.choice(lengths)); r = float(rng.choice(rel))
        pre = max(1, int(round(r * n))); post = n - pre
        if post < 1 or post > len(after) or len(before) - pre < 1000:
            dropped += 1
            continue
        hist, online = before[:-pre], np.r_[before[-pre:], after[:post]]
        tau = pre if bool(y.loc[sid]) else -1
        target = (np.arange(n) >= tau).astype(np.int8) if tau >= 0 else np.zeros(n, np.int8)
        tasks.append((hist.astype(np.float32), online.astype(np.float32), target, int(sid) % 5))
        meta.append((-1, target))
        if limit and len(tasks) >= limit:
            break
    print(f"converted {len(tasks)} series ({dropped} dropped), break share "
          f"{np.mean([m[1].any() for m in meta]):.3f}", flush=True)
    return tasks, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / ("seq_ext.npz" if not a.limit else f"seq_ext_smoke{a.limit}.npz")
    if out.exists():
        print("exists", out.name)
        return
    import models.model111.run_model111 as m111
    _, protocol, _layout, _real, _pool, _ = m111.load_everything()
    union_names = protocol["contract"]["union_names"]
    det_names = [n for n in union_names if not n.startswith("white_")]
    gru_cols = np.asarray(CONTRACT["detector_union_indices"])
    student_cols = np.asarray([union_names.index(n) for n in protocol["contract"]["names"][0]])
    t0 = time.time()
    tasks, meta = convert(a.limit)
    print(f"ext: {len(tasks)} series, {sum(len(m[1]) for m in meta):,} steps ({time.time() - t0:.0f}s)", flush=True)
    Xs, softs, pos = [], [], []
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("spawn"),
                             initializer=init, initargs=(det_names, gru_cols, student_cols)) as ex:
        for i, (X, soft) in enumerate(ex.map(work, tasks, chunksize=4)):
            Xs.append(X); softs.append(soft); pos.append(np.full(len(X), i, np.int32))
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(tasks)} ({time.time() - t0:.0f}s)", flush=True)
    rows = sum(len(x) for x in Xs)
    np.savez(out, X=np.concatenate(Xs), soft=np.concatenate(softs),
             target=np.concatenate([m[1] for m in meta]).astype(np.int8),
             time_online=np.concatenate([np.arange(len(m[1]), dtype=np.int16) for m in meta]),
             series_position=np.concatenate(pos), row_fold=np.full(rows, -1, np.int8))
    print(f"saved {out.name}: {rows:,} rows in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
