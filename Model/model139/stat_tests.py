"""Stage 2+3 tooling: series transforms and a bank of statistical tests, one feature each.

Every test maps (window, history) -> one scalar computed causally at an online
step: `window` = the last W online values (W in WINDOWS, or the whole prefix)
and `history` = the last HIST history values (history-fitted transforms use
the full history).  Change-point tests use the whole online prefix.

  python stat_tests.py build  TRANSFORM   -> data/feat_<transform>.npz (rows x tests x windows)
  python stat_tests.py list
"""
from __future__ import annotations

import math
import os
import sys
import warnings
from pathlib import Path

for _n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_n] = "1"
import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
OUT = HERE / "data"
WINDOWS = (32, 128, 0)          # 0 = whole online prefix
HIST = 512                      # history points used by two-sample tests
ROWS = 60_000                   # sampled online rows (same rows for every test/transform)
SEED = 139

# ----------------------------------------------------------------------------- transforms
def t_identity(h, o): return h, o
def t_abs(h, o): return np.abs(h), np.abs(o)
def t_square(h, o): return h * h, o * o
def t_sqrt(h, o): return np.sign(h) * np.sqrt(np.abs(h)), np.sign(o) * np.sqrt(np.abs(o))
def t_log(h, o): return np.sign(h) * np.log1p(np.abs(h)), np.sign(o) * np.log1p(np.abs(o))
def t_diff(h, o):
    return np.diff(h), np.diff(np.r_[h[-1], o])
def _rank_to(h, o, ppf):
    hs = np.sort(h); n = len(hs)
    def f(x):
        u = (np.searchsorted(hs, x, side="right") + 0.5) / (n + 1.0)
        return ppf(np.clip(u, 0.5 / (n + 1), 1 - 0.5 / (n + 1)))
    return f(h), f(o)
def t_rankgauss(h, o): return _rank_to(h, o, stats.norm.ppf)
def t_ranklaplace(h, o): return _rank_to(h, o, stats.laplace.ppf)

TRANSFORMS = {"identity": t_identity, "abs": t_abs, "square": t_square, "sqrt": t_sqrt, "log": t_log,
              "diff": t_diff, "rankgauss": t_rankgauss, "ranklaplace": t_ranklaplace}

# ----------------------------------------------------------------------------- helpers
def _ac(x, k):
    x = x - x.mean(); d = float(x @ x)
    return float(x[k:] @ x[:-k]) / d if d > 0 and len(x) > k else 0.0

def _ljung_box(x, lags=10):
    n = len(x); q = 0.0
    for k in range(1, min(lags, n - 2) + 1):
        r = _ac(x, k); q += r * r / (n - k)
    return n * (n + 2) * q

def _box_pierce(x, lags=10):
    n = len(x); return n * sum(_ac(x, k) ** 2 for k in range(1, min(lags, n - 2) + 1))

def _runs_z(x):
    m = np.median(x); s = x > m; n1 = s.sum(); n2 = len(s) - n1
    if n1 == 0 or n2 == 0: return 0.0
    runs = 1 + np.sum(s[1:] != s[:-1]); mu = 1 + 2 * n1 * n2 / (n1 + n2)
    var = 2 * n1 * n2 * (2 * n1 * n2 - n1 - n2) / ((n1 + n2) ** 2 * (n1 + n2 - 1))
    return (runs - mu) / math.sqrt(var) if var > 0 else 0.0

def _turning_z(x):
    n = len(x)
    if n < 3: return 0.0
    d = np.diff(x); tp = np.sum(d[1:] * d[:-1] < 0); mu = 2 * (n - 2) / 3; var = (16 * n - 29) / 90
    return (tp - mu) / math.sqrt(var)

def _vr(x, q=4):
    n = len(x)
    if n < 4 * q: return 0.0
    v1 = np.var(np.diff(x)) + 1e-12; vq = np.var(x[q:] - x[:-q]) / q
    return vq / v1 - 1.0

def _kuiper(a, b):
    A = np.sort(a); B = np.sort(b); allv = np.sort(np.r_[A, B])
    fa = np.searchsorted(A, allv, side="right") / len(A); fb = np.searchsorted(B, allv, side="right") / len(B)
    return float((fa - fb).max() - (fa - fb).min())

def _chi2_cells(w, h, k=8):
    q = np.quantile(h, np.linspace(0, 1, k + 1)[1:-1]); obs = np.bincount(np.searchsorted(q, w), minlength=k)
    exp = len(w) / k; return float(((obs - exp) ** 2 / exp).sum())

def _spec(x, nb=32):
    p = np.abs(np.fft.rfft(x - x.mean()))[1:] ** 2
    if p.sum() <= 0: return np.zeros(nb)
    # resample to nb bands
    idx = np.minimum((np.arange(len(p)) * nb) // len(p), nb - 1)
    return np.bincount(idx, weights=p, minlength=nb) / p.sum()

def _sig_ent(p): p = p[p > 0]; return float(-(p * np.log(p)).sum())

# ----------------------------------------------------------------------------- tests
# kind: "two" = f(window, hist); "one" = f(window) (compared with the same on the last HIST history);
#       "prefix" = f(prefix, hist) change-point statistic on the whole online prefix (window ignored)
def _safe(fn):
    def g(*a):
        try:
            v = float(fn(*a))
            return v if math.isfinite(v) else 0.0
        except Exception:
            return 0.0
    return g

TESTS = {
    # location
    "welch_t":        ("two", _safe(lambda w, h: stats.ttest_ind(w, h, equal_var=False).statistic)),
    "student_t":      ("two", _safe(lambda w, h: stats.ttest_ind(w, h).statistic)),
    "mann_whitney":   ("two", _safe(lambda w, h: stats.mannwhitneyu(w, h).statistic / (len(w) * len(h)) - 0.5)),
    "brunner_munzel": ("two", _safe(lambda w, h: stats.brunnermunzel(w, h).statistic)),
    "mood_median":    ("two", _safe(lambda w, h: stats.median_test(w, h)[0])),
    "sign_test":      ("two", _safe(lambda w, h: (np.mean(w > np.median(h)) - 0.5) * math.sqrt(len(w)))),
    # scale
    "log_var_ratio":  ("two", _safe(lambda w, h: math.log((w.var() + 1e-12) / (h.var() + 1e-12)))),
    "levene":         ("two", _safe(lambda w, h: stats.levene(w, h, center="mean").statistic)),
    "brown_forsythe": ("two", _safe(lambda w, h: stats.levene(w, h, center="median").statistic)),
    "ansari_bradley": ("two", _safe(lambda w, h: stats.ansari(w, h).statistic / (len(w) * (len(w) + len(h)) / 4.0) - 1.0)),
    "mood_scale":     ("two", _safe(lambda w, h: stats.mood(w, h).statistic)),
    "fligner":        ("two", _safe(lambda w, h: stats.fligner(w, h).statistic)),
    "mad_ratio":      ("two", _safe(lambda w, h: math.log((np.median(np.abs(w - np.median(w))) + 1e-9) / (np.median(np.abs(h - np.median(h))) + 1e-9)))),
    # distribution
    "ks":             ("two", _safe(lambda w, h: stats.ks_2samp(w, h).statistic)),
    "anderson_k":     ("two", _safe(lambda w, h: stats.anderson_ksamp([w, h]).statistic)),
    "cramer_vm":      ("two", _safe(lambda w, h: stats.cramervonmises_2samp(w, h).statistic)),
    "kuiper":         ("two", _safe(_kuiper)),
    "epps_singleton": ("two", _safe(lambda w, h: stats.epps_singleton_2samp(w, h).statistic)),
    "chi2_cells8":    ("two", _safe(_chi2_cells)),
    "energy":         ("two", _safe(lambda w, h: stats.energy_distance(w, h))),
    "wasserstein":    ("two", _safe(lambda w, h: stats.wasserstein_distance(w, h))),
    # shape (window statistic minus the history's)
    "jarque_bera":    ("one", _safe(lambda x: stats.jarque_bera(x).statistic / len(x))),
    "skew_z":         ("one", _safe(lambda x: stats.skew(x))),
    "kurt_z":         ("one", _safe(lambda x: stats.kurtosis(x))),
    "shapiro_w":      ("one", _safe(lambda x: stats.shapiro(x[-4000:]).statistic if len(x) >= 8 else 0.0)),
    "lilliefors":     ("one", _safe(lambda x: stats.kstest((x - x.mean()) / (x.std() + 1e-12), "norm").statistic)),
    # dependence (window statistic minus the history's)
    "ac1":            ("one", _safe(lambda x: _ac(x, 1))),
    "ac2":            ("one", _safe(lambda x: _ac(x, 2))),
    "ac3":            ("one", _safe(lambda x: _ac(x, 3))),
    "ljung_box10":    ("one", _safe(lambda x: _ljung_box(x) / len(x))),
    "box_pierce10":   ("one", _safe(lambda x: _box_pierce(x) / len(x))),
    "durbin_watson":  ("one", _safe(lambda x: np.sum(np.diff(x) ** 2) / (np.sum((x - x.mean()) ** 2) + 1e-12))),
    "runs_z":         ("one", _safe(_runs_z)),
    "turning_points": ("one", _safe(_turning_z)),
    "variance_ratio4":("one", _safe(_vr)),
    "arch_lm":        ("one", _safe(lambda x: _ac(x * x, 1))),
    "abs_ac1":        ("one", _safe(lambda x: _ac(np.abs(x), 1))),
    # spectral (band vector distance / entropy diff vs history)
    "spec_low":       ("one", _safe(lambda x: _spec(x)[:4].sum())),
    "spec_high":      ("one", _safe(lambda x: _spec(x)[16:].sum())),
    "spec_entropy":   ("one", _safe(lambda x: _sig_ent(_spec(x)))),
    "spec_peak":      ("one", _safe(lambda x: float(np.argmax(_spec(x))))),
    # change-point statistics on the whole prefix (standardized by the history sd)
    "cusum_mean":     ("prefix", _safe(lambda p, h: np.max(np.abs(np.cumsum(p - h.mean()))) / (h.std() * math.sqrt(len(p)) + 1e-12))),
    "cusum_sq":       ("prefix", _safe(lambda p, h: np.max(np.abs(np.cumsum(p * p - h.var()))) / (h.var() * math.sqrt(len(p)) + 1e-12))),
    "page_hinkley":   ("prefix", _safe(lambda p, h: (lambda c: float(c.max() - c.min()))(np.cumsum(p - h.mean() - 0.05 * h.std())) / (h.std() + 1e-12))),
    "chow_f_scan":    ("prefix", _safe(lambda p, h: _mean_lr_scan(p, h))),
    "var_lr_scan":    ("prefix", _safe(lambda p, h: _var_lr_scan(p, h))),
    "pettitt":        ("prefix", _safe(lambda p, h: _pettitt(p))),
    "buishand_range": ("prefix", _safe(lambda p, h: (lambda c: float((c.max() - c.min())))(np.cumsum(p - p.mean())) / (h.std() * math.sqrt(len(p)) + 1e-12))),
}

def _mean_lr_scan(p, h):
    n = len(p)
    if n < 6: return 0.0
    c = np.cumsum(p); k = np.arange(2, n - 1); m1 = c[k - 1] / k; m2 = (c[-1] - c[k - 1]) / (n - k)
    return float(np.max(k * (n - k) / n * (m1 - m2) ** 2) / (h.var() + 1e-12))

def _var_lr_scan(p, h):
    n = len(p)
    if n < 8: return 0.0
    s = np.cumsum(p * p); k = np.arange(4, n - 3); v1 = s[k - 1] / k; v2 = (s[-1] - s[k - 1]) / (n - k); v = s[-1] / n
    ll = n * np.log(v + 1e-12) - k * np.log(v1 + 1e-12) - (n - k) * np.log(v2 + 1e-12)
    return float(np.max(ll))

def _pettitt(p):
    n = len(p)
    if n < 4: return 0.0
    r = stats.rankdata(p); u = 2 * np.cumsum(r)[:-1] - np.arange(1, n) * (n + 1)
    return float(np.max(np.abs(u)) / (n * math.sqrt(n)))

# ----------------------------------------------------------------------------- driver
def series_rows(h, o, steps):
    """Features for one series at the requested online steps: (len(steps), n_tests, n_windows)."""
    names = list(TESTS); out = np.zeros((len(steps), len(names), len(WINDOWS)), np.float32)
    hist = h[-HIST:]
    hist_one = {nm: TESTS[nm][1](hist) for nm in names if TESTS[nm][0] == "one"}
    for si, t in enumerate(steps):
        prefix = o[:t + 1]
        for wi, W in enumerate(WINDOWS):
            w = prefix if W == 0 else prefix[-W:]
            if len(w) < 4:
                continue
            for ti, nm in enumerate(names):
                kind, fn = TESTS[nm]
                if kind == "two": out[si, ti, wi] = fn(w, hist)
                elif kind == "one": out[si, ti, wi] = fn(w) - hist_one[nm]
                elif wi == 2: out[si, ti, wi] = fn(prefix, hist)
    return out

def _work(args):
    transform, h, o, steps = args
    h2, o2 = TRANSFORMS[transform](np.asarray(h, np.float64), np.asarray(o, np.float64))
    return series_rows(h2, o2, steps)

def build(transform, workers):
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp, time
    from models.model10.sources import load_development_layout
    L = load_development_layout(ROOT)
    frame = pd.read_parquet(ROOT / "train/X_train.parquet")
    ids = frame.index.get_level_values(0).to_numpy(); values = frame["value"].to_numpy().astype(np.float64); period = frame["period"].to_numpy()
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]]); ends = np.r_[starts[1:], len(ids)]
    bounds = {int(ids[s]): (s, e) for s, e in zip(starts, ends)}
    rng = np.random.default_rng(SEED); rows = np.sort(rng.choice(L.row_count, ROWS, replace=False))
    pos = L.series_position[rows]; t = L.time_online[rows]
    tasks, spans = [], []
    for p in np.unique(pos):
        sel = np.flatnonzero(pos == p); s, e = bounds[int(L.series_ids[p])]
        v, pr = values[s:e], period[s:e]
        tasks.append((transform, v[pr == 1], v[pr == 2], t[sel])); spans.append(sel)
    X = np.zeros((ROWS, len(TESTS), len(WINDOWS)), np.float32); t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        for i, (sel, f) in enumerate(zip(spans, ex.map(_work, tasks, chunksize=16))):
            X[sel] = f
            if (i + 1) % 500 == 0: print(f"{transform}: {i + 1}/{len(tasks)} series ({time.time() - t0:.0f}s)", flush=True)
    OUT.mkdir(exist_ok=True)
    np.savez(OUT / f"feat_{transform}.npz", X=X, rows=rows, tests=np.asarray(list(TESTS)), windows=np.asarray(WINDOWS),
             target=L.target[rows], time_online=t, series_position=pos, row_fold=L.row_fold[rows])
    print(f"{transform}: saved {X.shape} in {time.time() - t0:.0f}s", flush=True)

if __name__ == "__main__":
    if sys.argv[1] == "list":
        for nm, (kind, _) in TESTS.items(): print(f"{nm:16s} {kind}")
        print(len(TESTS), "tests;", list(TRANSFORMS))
    else:
        build(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4)
