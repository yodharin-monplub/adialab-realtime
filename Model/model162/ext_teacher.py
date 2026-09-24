"""Model162 step 1: teacher inputs (25 Model109 + 7 oracle features) for the Model145 external rows
(2025 challenge series cut into the real-time format).  The cuts are REPLAYED with Model145's exact random
draws (same seed / order, no feature computation) and checked against external.npz (series, step, tau).
Output: models/model162/data/external_teacher.npy  (rows aligned with models/model145/data/external.npz)
Usage: python models/model162/ext_teacher.py
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from models.model109.teacher_features import series_features as tf109  # noqa: E402
from models.model157.oracle_bayes import series_oracle  # noqa: E402

SRC = ROOT / "data/ADIA Lab Structural Break Challenge"
OUT = Path(__file__).resolve().parent / "data"
MIN_HISTORY, ROWS, SEED = 1000, 50, 145


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rt = pd.read_parquet(ROOT / "train/X_train.parquet", columns=["period"])
    rid = rt.index.get_level_values(0).to_numpy()
    n_online = pd.Series(rt["period"].to_numpy() == 2).groupby(rid).sum()
    tau_rt = pd.read_parquet(ROOT / "train/y_train_index.parquet")["tau_index"].reindex(n_online.index)
    pairs = np.column_stack((n_online[tau_rt >= 0].to_numpy(), tau_rt[tau_rt >= 0].to_numpy())).astype(int)
    x = pd.read_parquet(SRC / "X_train.parquet"); y = pd.read_parquet(SRC / "y_train.parquet").iloc[:, 0]
    ids = x.index.get_level_values(0).to_numpy(); v = x["value"].to_numpy(np.float64); p = x["period"].to_numpy()
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]]); ends = np.r_[starts[1:], len(ids)]
    rng = np.random.default_rng(SEED)
    rows, sids, steps, taus = [], [], [], []
    t0 = time.time()
    for s, e in zip(starts, ends):
        sid = int(ids[s]); vv, pp = v[s:e], p[s:e]
        before, after = vv[pp == 0], vv[pp == 1]
        for _ in range(20):
            n, pre = pairs[rng.integers(len(pairs))]
            if pre <= len(before) - MIN_HISTORY and 1 <= n - pre <= len(after) and pre >= 1:
                break
        else:
            continue
        history, online = before[:-pre], np.r_[before[-pre:], after[:n - pre]]
        if not (np.isfinite(history).all() and np.isfinite(online).all()) or history.std() <= 0:
            continue
        keep = np.sort(rng.choice(n, min(ROWS, n), replace=False))
        tau = int(pre) if bool(y.loc[sid]) else -1
        h32, o32 = history.astype(np.float32).astype(np.float64), online.astype(np.float32).astype(np.float64)
        rows.append(np.hstack([tf109(h32, o32), series_oracle(h32, o32)])[keep])
        sids.append(np.full(len(keep), sid)); steps.append(keep); taus.append(np.full(len(keep), tau))
    X = np.concatenate(rows).astype(np.float32)
    ref = np.load(ROOT / "models/model145/data/external.npz")
    assert np.array_equal(np.concatenate(sids), ref["series"]) and np.array_equal(np.concatenate(steps), ref["time_online"]) \
        and np.array_equal(np.concatenate(taus), ref["tau"]), "replayed cuts differ from external.npz"
    np.save(OUT / "external_teacher.npy", X)
    print(f"saved external_teacher {X.shape} ({time.time() - t0:.0f}s); replay verified", flush=True)


if __name__ == "__main__":
    main()
