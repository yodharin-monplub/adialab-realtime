"""Stage 4 feature builder: fast (numba) versions of the stage-3 winning tests, for ALL rows.

Per online step t, history-referenced, causal:
  prefix change-point scans : cusum_sq, var_lr_scan, chow_f_scan, buishand_range, page_hinkley
  window 128 vs history     : levene (mean |x-mean| ratio, log), brown_forsythe (median-centred),
                              abs_log_var_ratio, wasserstein1, kuiper, ljung_box10 (window minus
                              history), kurtosis (window minus history)
  window 64                 : spectral entropy (rfft band entropy) minus the history's
= 13 columns (NAMES).  History reference = last 1,024 history points.

Outputs (models/model140/data): real.npy (development layout order) and
syn_fold{k}.npy (kept synthetic rows of each fold plan).
Usage: python models/model140/fast_tests.py [real] [syn 0 1 2 3 4]
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
# FT_TRANSFORM=identity|diff|abs : series transform applied before the tests (stage 2 winners);
# diff = first differences (online prefixed with the last history point), abs = |x - history mean|
TRANSFORM = __import__("os").environ.get("FT_TRANSFORM", "identity")
SUFFIX = "" if TRANSFORM == "identity" else f"_{TRANSFORM}"


def transformed(h, o):
    if TRANSFORM == "diff":
        return np.diff(h), np.diff(np.concatenate(([h[-1]], o)))
    if TRANSFORM == "abs":
        m = h.mean()
        return np.abs(h - m), np.abs(o - m)
    return h, o
W = 128
HIST = 1024
NAMES = ["ft_cusum_sq", "ft_var_lr_scan", "ft_chow_f_scan", "ft_buishand", "ft_page_hinkley",
         "ft_levene128", "ft_brown_forsythe128", "ft_abs_log_var_ratio128", "ft_wasserstein128", "ft_kuiper128",
         "ft_ljung_box128", "ft_kurtosis128", "ft_spec_entropy64"]


@njit(cache=True)
def _sorted_insert_stats(hs, w):
    """Wasserstein-1 and Kuiper between sorted history hs and window w (sorted inside)."""
    ws = np.sort(w)
    n, m = len(hs), len(ws)
    # Wasserstein-1 via quantile functions on a common grid of m points
    wd = 0.0
    for i in range(m):
        q = (i + 0.5) / m
        hq = hs[min(n - 1, int(q * n))]
        wd += abs(ws[i] - hq)
    wd /= m
    # Kuiper: max(Fw - Fh) + max(Fh - Fw) over the union
    dplus = -1.0; dminus = -1.0; i = 0; j = 0
    while i < n or j < m:
        if j >= m or (i < n and hs[i] <= ws[j]):
            i += 1
        else:
            j += 1
        fh = i / n; fw = j / m
        if fw - fh > dplus: dplus = fw - fh
        if fh - fw > dminus: dminus = fh - fw
    return wd, dplus + dminus


@njit(cache=True)
def _acf_sum(x, lags):
    n = len(x); m = x.mean(); d = 0.0
    for i in range(n): d += (x[i] - m) ** 2
    if d <= 0 or n < 4: return 0.0
    q = 0.0
    lags = min(lags, n - 2)
    for k in range(1, lags + 1):
        s = 0.0
        for i in range(k, n): s += (x[i] - m) * (x[i - k] - m)
        r = s / d; q += r * r / (n - k)
    return n * (n + 2) * q / n


@njit(cache=True)
def _kurt(x):
    n = len(x); m = x.mean(); s2 = 0.0; s4 = 0.0
    for i in range(n):
        d = x[i] - m; s2 += d * d; s4 += d ** 4
    if s2 <= 0: return 0.0
    return n * s4 / (s2 * s2) - 3.0


@njit(cache=True)
def series_features(h, o):
    n = len(o)
    out = np.zeros((n, 13), dtype=np.float32)
    hist = h[-HIST:] if len(h) > HIST else h
    hs = np.sort(hist)
    hm = hist.mean(); hv = hist.var(); hsd = math.sqrt(hv) + 1e-12
    hmad = np.mean(np.abs(hist - hm)) + 1e-12
    hmed = np.median(hist); hbf = np.mean(np.abs(hist - hmed)) + 1e-12
    hlb = _acf_sum(hist[-W:], 10) if len(hist) >= W else 0.0
    hk = _kurt(hist[-W:]) if len(hist) >= W else 0.0
    # cumulative sums over the online prefix
    c1 = np.zeros(n + 1); c2 = np.zeros(n + 1)
    for i in range(n):
        c1[i + 1] = c1[i] + o[i]; c2[i + 1] = c2[i] + o[i] * o[i]
    ph = 0.0; ph_min = 0.0; ph_max = 0.0
    for t in range(n):
        p = t + 1
        # --- prefix scans
        cs_max = 0.0; cs_min = 0.0; bu_max = -1e30; bu_min = 1e30
        mean_p = c1[p] / p
        for k in range(1, p + 1):
            v = (c2[k] - hv * k)
            if v > cs_max: cs_max = v
            if v < cs_min: cs_min = v
            b = c1[k] - k * mean_p
            if b > bu_max: bu_max = b
            if b < bu_min: bu_min = b
        out[t, 0] = max(cs_max, -cs_min) / (hv * math.sqrt(p) + 1e-12)
        out[t, 3] = (bu_max - bu_min) / (hsd * math.sqrt(p))
        ph += o[t] - hm - 0.05 * hsd
        if ph > ph_max: ph_max = ph
        if ph < ph_min: ph_min = ph
        out[t, 4] = (ph_max - ph_min) / hsd
        if p >= 8:
            best_v = 0.0; best_m = 0.0
            for k in range(4, p - 3):
                v1 = c2[k] / k; v2 = (c2[p] - c2[k]) / (p - k); v = c2[p] / p
                ll = p * math.log(v + 1e-12) - k * math.log(v1 + 1e-12) - (p - k) * math.log(v2 + 1e-12)
                if ll > best_v: best_v = ll
                m1 = c1[k] / k; m2 = (c1[p] - c1[k]) / (p - k)
                f = k * (p - k) / p * (m1 - m2) ** 2
                if f > best_m: best_m = f
            out[t, 1] = best_v
            out[t, 2] = best_m / (hv + 1e-12)
        # --- window 128 vs history
        lo = max(0, p - W)
        w = o[lo:p]
        if len(w) >= 8:
            wm = w.mean()
            out[t, 5] = math.log((np.mean(np.abs(w - wm)) + 1e-12) / hmad)
            out[t, 6] = math.log((np.mean(np.abs(w - np.median(w))) + 1e-12) / hbf)
            out[t, 7] = abs(math.log((w.var() + 1e-12) / (hv + 1e-12)))
            wd, ku = _sorted_insert_stats(hs, w)
            out[t, 8] = wd; out[t, 9] = ku
            out[t, 10] = _acf_sum(w, 10) - hlb
            out[t, 11] = _kurt(w) - hk
    return out


def spec_entropy_batch(o, hist):
    """Spectral entropy of the last 64 points at every step minus the history's (numpy, batched)."""
    n = len(o); out = np.zeros(n, np.float32)
    if n < 64:
        return out

    def ent(mat):
        p = np.abs(np.fft.rfft(mat - mat.mean(axis=1, keepdims=True), axis=1))[:, 1:] ** 2
        p = p / (p.sum(axis=1, keepdims=True) + 1e-12)
        return -(np.where(p > 0, p * np.log(p + 1e-30), 0.0)).sum(axis=1)
    href = ent(hist[-64:][None, :])[0] if len(hist) >= 64 else 0.0
    idx = np.arange(63, n)
    win = np.lib.stride_tricks.sliding_window_view(o, 64)
    out[idx] = ent(win) - href
    return out


def build_real():
    import pandas as pd
    from models.model10.sources import load_development_layout
    L = load_development_layout(ROOT)
    frame = pd.read_parquet(ROOT / "train/X_train.parquet")
    ids = frame.index.get_level_values(0).to_numpy(); values = frame["value"].to_numpy().astype(np.float64); period = frame["period"].to_numpy()
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]]); ends = np.r_[starts[1:], len(ids)]
    bounds = {int(ids[s]): (s, e) for s, e in zip(starts, ends)}
    pos = L.series_position; rs = np.flatnonzero(np.r_[True, pos[1:] != pos[:-1]]); re_ = np.r_[rs[1:], L.row_count]
    X = np.empty((L.row_count, 13), np.float32); t0 = time.time()
    for i, (a, b) in enumerate(zip(rs, re_)):
        s, e = bounds[int(L.series_ids[pos[a]])]; v, p = values[s:e], period[s:e]; h, o = transformed(v[p == 1], v[p == 2])
        f = series_features(h, o); f[:, 12] = spec_entropy_batch(o, h); X[a:b] = f
        if i % 1000 == 0: print(f"real{SUFFIX}: {i} series ({time.time() - t0:.0f}s)", flush=True)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
    np.save(OUT / f"real{SUFFIX}.npy", X); print(f"saved real{SUFFIX}", X.shape, flush=True)


def build_syn(folds):
    import models.model111.run_model111 as m111
    from models.model20.run_model20 import synthetic_tasks
    _, protocol, layout, real, pool, _ = m111.load_everything()
    for k in folds:
        tasks, sy, st = synthetic_tasks(pool, protocol["plans"][k], layout, True)
        X = np.empty((len(sy), 13), np.float32); at = 0; t0 = time.time()
        for h, o, keep in tasks:
            h, o = transformed(np.asarray(h, np.float64), np.asarray(o, np.float64))
            f = series_features(h, o); f[:, 12] = spec_entropy_batch(o, h)
            X[at:at + len(keep)] = f[np.asarray(keep)]; at += len(keep)
        np.nan_to_num(X, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
        np.save(OUT / f"syn{SUFFIX}_fold{k}.npy", X); print(f"saved syn{SUFFIX} fold {k} {X.shape} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    args = sys.argv[1:] or ["real", "syn", "0", "1", "2", "3", "4"]
    if "real" in args: build_real()
    if "syn" in args: build_syn([int(a) for a in args if a.isdigit()])
