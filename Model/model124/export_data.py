"""Export sequence-model training data for Model124 (GPU student).

Per online row: 61 Model57 whitened columns + the 40 highest-gain detector
columns (Model109 fold-0 hard-student gain order) + log1p(t) = 102 inputs,
standardized with label-free global constants (saved for serving), float16.
Soft targets per held fold k: 0.5*y + 0.5*teacher for rows of folds
j not in {k, k+1}, scored by Model111's saved teacher fold{k}_teacher{j};
NaN elsewhere (rows the tree student never trains on either).

Output: models/model124/data/sequences.npz (~1 GB)
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import models.model111.run_model111 as m111  # noqa: E402  (installs Linux stubs)
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402

from models.model10.sources import load_development_layout  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "data"
TOP_DETECTORS = int(sys.argv[1]) if len(sys.argv) > 1 else 40   # e.g. 200 for Model126
OUT_NAME = sys.argv[2] if len(sys.argv) > 2 else "sequences"    # output stem
CHUNK = 250_000


def main():
    OUT.mkdir(exist_ok=True)
    L = load_development_layout(ROOT)
    real = np.load(ROOT / "models/model20/runs/82abe027caea808d/real_features.npy", mmap_mode="r")
    white = np.load(ROOT / "models/model57/runs/dfab0ede1c3206a4/real_white.npy", mmap_mode="r")
    teacher = np.load(ROOT / "models/model109/cache/teacher_features.npy", mmap_mode="r")
    union = json.loads((ROOT / "models/model20/runs/82abe027caea808d/real_features.cache.json").read_text())["contract"]["names"]
    white_names = json.loads((ROOT / "models/model57/runs/dfab0ede1c3206a4/real_white.json").read_text())["names"]
    order = np.load(ROOT / "models/model109/runs/fold0_hard_gain_order.npy")
    detector_cols = [int(i) for i in order if i < 439][:TOP_DETECTORS]
    input_names = list(white_names) + [union[i] for i in detector_cols] + ["log1p_t"]
    n = L.row_count
    X = np.empty((n, len(input_names)), dtype=np.float32)
    t0 = time.time()
    nd = len(detector_cols)
    for a in range(0, n, CHUNK):
        b = min(n, a + CHUNK)
        X[a:b, :61] = white[a:b]
        X[a:b, 61:61 + nd] = real[a:b][:, detector_cols]
        X[a:b, 61 + nd] = np.log1p(L.time_online[a:b])
    np.nan_to_num(X, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X = np.clip((X - mu) / sd, -10, 10).astype(np.float16)
    print(f"inputs {X.shape} in {time.time() - t0:.0f}s", flush=True)

    protocol = json.loads((ROOT / "models/model84/runs/e1e98a6e8af2df25/protocol.json").read_text())
    soft = np.full((5, n), np.nan, dtype=np.float16)
    for k in range(5):
        names = protocol["contract"]["names"][k]
        columns = np.asarray([union.index(nm) for nm in names[:-61]])  # 256 detector columns
        for j in range(5):
            if j in (k, (k + 1) % 5):
                continue
            booster = lgb.Booster(model_file=str(ROOT / f"models/model111/runs/fold{k}_teacher{j}.txt"))
            rows = np.flatnonzero(L.row_fold == j)
            t1 = time.time()
            for a in range(0, len(rows), CHUNK):
                r = rows[a:a + CHUNK]
                x = np.hstack([np.asarray(real[r][:, columns]), np.asarray(white[r]), np.asarray(teacher[r])])
                prob = booster.predict(x, num_threads=4)
                soft[k, r] = 0.5 * L.target[r] + 0.5 * prob
            print(f"soft targets k={k} j={j}: {len(rows):,} rows ({time.time() - t1:.0f}s)", flush=True)
    np.savez(OUT / f"{OUT_NAME}.npz", X=X, soft=soft, target=L.target.astype(np.int8),
             time_online=L.time_online.astype(np.int16), row_fold=L.row_fold.astype(np.int8),
             series_position=L.series_position.astype(np.int32), series_ids=L.series_ids)
    (OUT / f"{'input_contract' if OUT_NAME == 'sequences' else OUT_NAME + '_contract'}.json").write_text(json.dumps(
        {"names": input_names, "detector_union_indices": detector_cols, "mu": mu.tolist(), "sd": sd.tolist(),
         "clip": 10.0}, indent=2))
    print("saved", OUT / f"{OUT_NAME}.npz", flush=True)


if __name__ == "__main__":
    main()
