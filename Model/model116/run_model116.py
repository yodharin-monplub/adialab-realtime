"""Model116: learned causal temporal stacker over deployable OOF score trajectories.

Inputs per online row (all causal, per series):
  t, log1p(t); chain z-score c and distilled z-score d (Submission30 parts);
  EMAs of c and d (half-lives 2, 8, 32, 128); running max of c and d over the
  last 16 / 64 / 256 steps and since start; running mean since start; 16-step
  change; steps since the running max; and a few history descriptors from the
  Model57 whitening cache.
Target: hard label with time-pair weights (tsauc.time_pair_weights).
Cross-fitting: stacker for held fold k is trained on OOF rows of the other four
folds only (the OOF scores themselves were produced without fold k's labels
for fold k rows).  Score compared with Submission30 (c + 0.5 d).

Usage: python models/model116/run_model116.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import lightgbm as lgb
import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.model10.sources import load_development_layout  # noqa: E402
from tsauc import time_pair_weights, tsauc  # noqa: E402

HERE = Path(__file__).resolve().parent
HALF_LIVES = (2.0, 8.0, 32.0, 128.0)
WINDOWS = (16, 64, 256)
HIST_COLUMNS = ("white_hist_vol_cluster", "white_hist_log_kurtosis", "white_hist_residual_ratio",
                "white_hist_phi1", "white_hist_ar_order")
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=2000,
              subsample=0.7, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, max_bin=127)
TRAIN_ROWS = 1_500_000


@njit(cache=True)
def trajectory_features(x, pos, half_lives, windows):
    n = len(x)
    nh, nw = len(half_lives), len(windows)
    out = np.empty((n, nh + nw + 5), dtype=np.float32)
    ema = np.zeros(nh)
    alpha = np.empty(nh)
    for i in range(nh):
        alpha[i] = 1.0 - 0.5 ** (1.0 / half_lives[i])
    start = 0
    run_sum = 0.0
    run_max = -1e30
    arg_max = 0
    for i in range(n):
        if i == 0 or pos[i] != pos[i - 1]:
            start = i
            run_sum = 0.0
            run_max = -1e30
            for h in range(nh):
                ema[h] = x[i]
        k = i - start
        for h in range(nh):
            ema[h] = alpha[h] * x[i] + (1 - alpha[h]) * ema[h]
            out[i, h] = ema[h]
        for w in range(nw):
            lo = max(start, i - windows[w] + 1)
            m = x[lo]
            for j in range(lo + 1, i + 1):
                if x[j] > m:
                    m = x[j]
            out[i, nh + w] = m
        run_sum += x[i]
        if x[i] >= run_max:
            run_max = x[i]
            arg_max = k
        out[i, nh + nw] = run_max
        out[i, nh + nw + 1] = run_sum / (k + 1)
        out[i, nh + nw + 2] = x[i] - x[max(start, i - 16)]
        out[i, nh + nw + 3] = k - arg_max
        out[i, nh + nw + 4] = x[i]
    return out


def build_features(L):
    c = json.loads((ROOT / "models/model112/constants.json").read_text())
    chain = np.load(ROOT / "models/model112/chain_m73_all.npy").astype(np.float64)
    zc = (chain - c["mu_chain"]) / c["sd_chain"]
    d = np.empty(L.row_count)
    for k in range(5):
        p = np.clip(np.load(ROOT / f"models/model111/runs/a050_fold{k}.npz")["prediction"].astype(np.float64),
                    1e-6, 1 - 1e-6)
        d[L.row_fold == k] = np.log(p) - np.log1p(-p)
    zd = (d - c["mu_dist"]) / c["sd_dist"]
    pos = L.series_position.astype(np.int64)
    hl = np.asarray(HALF_LIVES)
    wn = np.asarray(WINDOWS, dtype=np.int64)
    fc = trajectory_features(zc, pos, hl, wn)
    fd = trajectory_features(zd, pos, hl, wn)
    white_names = json.loads((ROOT / "models/model57/runs/dfab0ede1c3206a4/real_white.json").read_text())["names"]
    white = np.load(ROOT / "models/model57/runs/dfab0ede1c3206a4/real_white.npy", mmap_mode="r")
    hist = np.asarray(white[:, [white_names.index(n) for n in HIST_COLUMNS]])
    t = L.time_online.astype(np.float32)
    X = np.hstack([t[:, None], np.log1p(t)[:, None], fc, fd, (fc - fd), hist]).astype(np.float32)
    base = (zc + c["weight"] * zd).astype(np.float32)
    return X, base


def main():
    L = load_development_layout(ROOT)
    started = time.monotonic()
    X, base = build_features(L)
    print(f"features {X.shape} in {time.monotonic() - started:.0f}s", flush=True)
    pred = np.empty(L.row_count, dtype=np.float32)
    rows = {}
    for k in range(5):
        train = np.flatnonzero(L.row_fold != k)
        train = np.sort(np.random.default_rng(116 + k).choice(train, TRAIN_ROWS, replace=False))
        y, t = L.target[train], L.time_online[train]
        model = lgb.LGBMClassifier(objective="binary", random_state=k, n_jobs=6, verbosity=-1,
                                   force_col_wise=True, **PARAMS)
        model.fit(X[train], y, sample_weight=time_pair_weights(y, t))
        held = L.row_fold == k
        pred[held] = model.predict_proba(X[held])[:, 1]
        model.booster_.save_model(str(HERE / f"stacker_fold{k}.txt"))
        a = tsauc(L.target[held], base[held], L.time_online[held])
        b = tsauc(L.target[held], pred[held], L.time_online[held])
        rows[k] = {"submission30": a, "stacker": b}
        print(f"fold {k}: Submission30 {a:.6f}  stacker {b:.6f}  gain {b - a:+.6f}", flush=True)
    np.save(HERE / "stacker_oof.npy", pred)
    gains = [r["stacker"] - r["submission30"] for r in rows.values()]
    summary = {"folds": rows, "mean_submission30": float(np.mean([r["submission30"] for r in rows.values()])),
               "mean_stacker": float(np.mean([r["stacker"] for r in rows.values()])),
               "mean_gain": float(np.mean(gains)), "wins": int(np.sum(np.array(gains) > 0))}
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
