"""Model129 stage 2a: full-sequence GRU inputs for the synthetic splice series.

For each fold plan's 1,000 spliced (history, online) tasks, compute the 61
Model57 whitening columns and the 40 detector columns of the Model124 input
contract for EVERY online step, standardize with the contract constants and
store float16, plus the sequence starts and the index of each kept training
row (the rows the tree student actually uses) inside the full array.

Output: models/model129/data/syn_fold{k}.npz  (X, starts, keep_index, n_tasks)
Usage: python models/model129/build_synthetic_sequences.py [--workers 4] [fold ...]
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
CONTRACT = json.loads((ROOT / "models/model124/data/input_contract.json").read_text())
DETECTOR_NAMES = CONTRACT["names"][61:101]
MU = np.asarray(CONTRACT["mu"], np.float32)
SD = np.asarray(CONTRACT["sd"], np.float32)
CLIP = float(CONTRACT["clip"])


def task_features(args):
    """(history, online) -> standardized float16 (len(online), 102)."""
    history, online = args
    from models.model57.whitening import whitening_features
    from models.model20.features import selected_features
    white = np.asarray(whitening_features(np.asarray(history, np.float64), np.asarray(online, np.float64)), np.float32)
    det = np.asarray(selected_features(np.asarray(history, np.float32), np.asarray(online, np.float32), DETECTOR_NAMES), np.float32)
    t = np.log1p(np.arange(len(online), dtype=np.float32))[:, None]
    raw = np.hstack([white, det, t])
    np.nan_to_num(raw, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
    return np.clip((raw - MU) / SD, -CLIP, CLIP).astype(np.float16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    import models.model111.run_model111 as m111  # installs Linux stubs
    from models.model20.run_model20 import synthetic_tasks
    _, protocol, layout, real, pool, _ = m111.load_everything()
    for fold in args.folds:
        path = OUT / f"syn_fold{fold}.npz"
        if path.exists():
            print(f"fold {fold}: exists", flush=True)
            continue
        tasks, sy, st = synthetic_tasks(pool, protocol["plans"][fold], layout, True)
        lengths = np.asarray([len(o) for _, o, _ in tasks])
        starts = np.r_[0, np.cumsum(lengths)[:-1]]
        keep_index = np.concatenate([starts[i] + np.asarray(keep) for i, (_, _, keep) in enumerate(tasks)])
        assert len(keep_index) == len(sy)
        t0 = time.time()
        # timing probe on three tasks in-process
        for i in range(3):
            task_features((tasks[i][0], tasks[i][1]))
        print(f"fold {fold}: {len(tasks)} tasks, {lengths.sum():,} rows; probe {(time.time() - t0) / 3:.2f} s/task", flush=True)
        X = np.empty((int(lengths.sum()), 102), dtype=np.float16)
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool_:
            for i, feats in enumerate(pool_.map(task_features, [(h, o) for h, o, _ in tasks], chunksize=8)):
                X[starts[i]:starts[i] + lengths[i]] = feats
                if (i + 1) % 200 == 0:
                    print(f"  {i + 1}/{len(tasks)} tasks ({time.time() - t0:.0f}s)", flush=True)
        # sanity: kept-row whitening columns must match the cached synthetic whitening exactly (after standardization)
        white_cache = m111.m84.white_source.cached_white(m111.m84.PARENT57 / f"aligned_25_fold{fold}_white.npy",
                                                          m111.m84.CONFIG["model57_protocol_sha256"], len(sy))
        ref = np.clip((np.asarray(white_cache, np.float32) - MU[:61]) / SD[:61], -CLIP, CLIP).astype(np.float16)
        err = float(np.max(np.abs(X[keep_index, :61].astype(np.float32) - ref.astype(np.float32))))
        np.savez(path, X=X, starts=starts, lengths=lengths, keep_index=keep_index, n_tasks=len(tasks))
        print(f"fold {fold}: saved {X.shape}, kept-row whitening max diff vs cache {err:.4g} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
