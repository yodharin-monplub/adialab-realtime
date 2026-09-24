"""Model133: causal "moments & transforms" bank on the online values (user suggestion).

Per online step t, for windows w in (16, 64, 256) and the whole prefix:
  m3, m4, m5   standardized central moments of the window minus the history's
  sqrt, log    mean sqrt|x| and mean log1p|x| of the window minus the history's
For the last 64 points: FFT band powers (low: bins 1-4, mid: 5-16, high: 17-32)
as fractions of total power, minus the history's (computed on its last 512 points).
Total 4*5 + 3 = 23 columns, float32, computed with numba from running power sums.

Outputs (models/model133/data): real.npy (rows in development layout order) and
syn_fold{k}.npy (kept synthetic rows in the fold plan order).
Usage: python models/model133/moment_features.py [real] [syn 0 1 2 3 4]
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import time

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
OUT = HERE / "data"
WINDOWS = (16, 64, 256)
NAMES = ([f"mom_{stat}_w{w}" for w in (*WINDOWS, 0) for stat in ("m3", "m4", "m5", "sqrt", "log")]
         + ["fft_low64", "fft_mid64", "fft_high64"])


@njit(cache=True)
def _moments(s1, s2, s3, s4, s5, n):
    """Standardized central moments 3..5 from raw power sums."""
    m = s1 / n
    c2 = s2 / n - m * m
    if c2 < 1e-12:
        return 0.0, 0.0, 0.0
    e3 = s3 / n; e2 = s2 / n; e4 = s4 / n; e5 = s5 / n
    c3 = e3 - 3 * m * e2 + 2 * m ** 3
    c4 = e4 - 4 * m * e3 + 6 * m * m * e2 - 3 * m ** 4
    c5 = e5 - 5 * m * e4 + 10 * m * m * e3 - 10 * m ** 3 * e2 + 4 * m ** 5
    sd = math.sqrt(c2)
    return c3 / sd ** 3, c4 / sd ** 4 - 3.0, c5 / sd ** 5


@njit(cache=True)
def _bands(x):
    """FFT band power fractions of a length-64 window via direct DFT (low/mid/high)."""
    n = 64
    p = np.zeros(33)
    for k in range(1, 33):
        re = 0.0; im = 0.0
        for j in range(n):
            a = 2.0 * math.pi * k * j / n
            re += x[j] * math.cos(a); im -= x[j] * math.sin(a)
        p[k] = re * re + im * im
    tot = p.sum() + 1e-12
    return p[1:5].sum() / tot, p[5:17].sum() / tot, p[17:33].sum() / tot


@njit(cache=True)
def series_features(h, o, windows):
    n = len(o)
    out = np.zeros((n, 23), dtype=np.float32)
    # history references
    hs = np.zeros(5)
    hsq = 0.0; hlog = 0.0
    for v in h:
        hs[0] += v; hs[1] += v * v; hs[2] += v ** 3; hs[3] += v ** 4; hs[4] += v ** 5
        hsq += math.sqrt(abs(v)); hlog += math.log1p(abs(v))
    nh = len(h)
    h3, h4, h5 = _moments(hs[0], hs[1], hs[2], hs[3], hs[4], nh)
    hsq /= nh; hlog /= nh
    hb = _bands(h[nh - 64:]) if nh >= 64 else (0.0, 0.0, 0.0)
    # cumulative sums over the online segment (index i+1 = sum of o[:i+1])
    c1 = np.zeros(n + 1); c2 = np.zeros(n + 1); c3 = np.zeros(n + 1); c4 = np.zeros(n + 1); c5 = np.zeros(n + 1)
    csq = np.zeros(n + 1); clog = np.zeros(n + 1)
    for i in range(n):
        v = o[i]
        c1[i + 1] = c1[i] + v; c2[i + 1] = c2[i] + v * v; c3[i + 1] = c3[i] + v ** 3
        c4[i + 1] = c4[i] + v ** 4; c5[i + 1] = c5[i] + v ** 5
        csq[i + 1] = csq[i] + math.sqrt(abs(v)); clog[i + 1] = clog[i] + math.log1p(abs(v))
    for t in range(n):
        col = 0
        for wi in range(4):
            w = windows[wi] if wi < 3 else t + 1
            lo = t + 1 - w
            if lo < 0:
                lo = 0
            m = t + 1 - lo
            if m >= 4:
                a3, a4, a5 = _moments(c1[t + 1] - c1[lo], c2[t + 1] - c2[lo], c3[t + 1] - c3[lo],
                                      c4[t + 1] - c4[lo], c5[t + 1] - c5[lo], m)
                out[t, col] = a3 - h3; out[t, col + 1] = a4 - h4; out[t, col + 2] = a5 - h5
                out[t, col + 3] = (csq[t + 1] - csq[lo]) / m - hsq
                out[t, col + 4] = (clog[t + 1] - clog[lo]) / m - hlog
            col += 5
        if t >= 63:
            b = _bands(o[t - 63:t + 1])
            out[t, 20] = b[0] - hb[0]; out[t, 21] = b[1] - hb[1]; out[t, 22] = b[2] - hb[2]
    return out


def build_real():
    import pandas as pd
    from models.model10.sources import load_development_layout
    L = load_development_layout(ROOT)
    frame = pd.read_parquet(ROOT / "train/X_train.parquet")
    ids = frame.index.get_level_values(0).to_numpy(); values = frame["value"].to_numpy().astype(np.float64)
    period = frame["period"].to_numpy()
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]]); ends = np.r_[starts[1:], len(ids)]
    bounds = {int(ids[s]): (s, e) for s, e in zip(starts, ends)}
    pos = L.series_position
    row_start = np.flatnonzero(np.r_[True, pos[1:] != pos[:-1]]); row_end = np.r_[row_start[1:], L.row_count]
    X = np.empty((L.row_count, 23), np.float32); w = np.asarray(WINDOWS, np.int64); t0 = time.time()
    for i, (a, b) in enumerate(zip(row_start, row_end)):
        s, e = bounds[int(L.series_ids[pos[a]])]
        v, p = values[s:e], period[s:e]
        X[a:b] = series_features(v[p == 1], v[p == 2], w)
        if i % 2000 == 0:
            print(f"real: {i} series ({time.time() - t0:.0f}s)", flush=True)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
    np.save(OUT / "real.npy", X); print("saved real", X.shape, flush=True)


def build_syn(folds):
    import models.model111.run_model111 as m111
    from models.model20.run_model20 import synthetic_tasks
    _, protocol, layout, real, pool, _ = m111.load_everything()
    w = np.asarray(WINDOWS, np.int64)
    for k in folds:
        tasks, sy, st = synthetic_tasks(pool, protocol["plans"][k], layout, True)
        X = np.empty((len(sy), 23), np.float32); at = 0; t0 = time.time()
        for h, o, keep in tasks:
            f = series_features(np.asarray(h, np.float64), np.asarray(o, np.float64), w)
            X[at:at + len(keep)] = f[np.asarray(keep)]; at += len(keep)
        assert at == len(sy)
        np.nan_to_num(X, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
        np.save(OUT / f"syn_fold{k}.npy", X); print(f"saved syn fold {k} {X.shape} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    args = sys.argv[1:] or ["real", "syn", "0", "1", "2", "3", "4"]
    if "real" in args:
        build_real()
    if "syn" in args:
        build_syn([int(a) for a in args if a.isdigit()])
