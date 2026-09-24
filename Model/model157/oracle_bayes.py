"""Model157 step 1: Bayesian ORACLE features for the teacher (privileged: they see the whole online segment).

For a series with online normal scores u_0..u_(L-1) (history-only AR whitening + history-ECDF, Model156) and
every candidate break position k, the Bayes factor of "break at k" vs "no break" is computed with Model156's
calibrated amplitude priors, per channel (scale / dependence / mean) and for the 1/3-1/3-1/3 mixture of the
three break types.  With P(break) = 1/2 and a uniform prior over k:
    post(k) = 0.5/L * BF_k / Z,   P(no break) = 0.5 / Z
Features per online step t (7): P(tau <= t | all data) for the mixture and each channel, P(break | all data)
for the mixture, log posterior odds of a break, posterior mean of (t - tau) given tau <= t (mixture).
This is (up to the model) the Bayesian posterior of the label itself - the natural oracle for a soft target.
Usage: python models/model157/oracle_bayes.py check | real | holdout
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import gammaln, logsumexp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.model156.bayes_evidence import A_SCALE, R_DEP, S_MEAN, _ndtri, normal_scores  # noqa: E402

OUT = Path(__file__).resolve().parent / "data"
NAMES = ["ob_cdf", "ob_cdf_scale", "ob_cdf_dep", "ob_cdf_mean", "ob_pbreak", "ob_logodds", "ob_age"]


def _lme(a, axis):
    return logsumexp(a, axis=axis) - np.log(a.shape[axis])


def series_oracle(h, o):
    z, hs, ulast = normal_scores(h, o)
    H = len(hs)
    u = np.array([_ndtri((np.searchsorted(hs, x) + 0.5) / (H + 1.0)) for x in z])
    L = len(u)
    prev = np.r_[ulast, u[:-1]]
    rs = lambda v: np.cumsum(v[::-1])[::-1]  # noqa: E731  suffix sums: index k -> sum over i >= k
    S, T, Sxy, Syy = rs(u * u), rs(u), rs(u * prev), rs(prev * prev)
    m = np.arange(L, 0, -1, dtype=np.float64)                      # suffix length for break at k
    a = A_SCALE[:, None]; b = a - 1.0
    ls = _lme(a * np.log(b) - gammaln(a) + gammaln(a + m / 2) - (a + m / 2) * np.log(b + S / 2) + S / 2, 0)
    s2 = (S_MEAN ** 2)[:, None]
    lm = _lme(-0.5 * np.log1p(m * s2) + s2 * T * T / (2 * (1 + m * s2)), 0)
    r = R_DEP[:, None]; q = 1 - r * r
    ld = _lme(-0.5 * m * np.log(q) - (S - 2 * r * Sxy + r * r * Syy) / (2 * q) + S / 2, 0)
    lmix = _lme(np.vstack([ls, ld, lm]), 0)
    t = np.arange(L)
    feats = []
    for lbf in (lmix, ls, ld, lm):
        lpost_k = np.log(0.5 / L) + lbf                               # unnormalised log posterior of tau = k
        lz = np.logaddexp(logsumexp(lpost_k), np.log(0.5))
        post = np.exp(lpost_k - lz)
        feats.append((np.cumsum(post), float(post.sum()), post))
    cdf, pb, post = feats[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        age = np.where(cdf > 1e-12, (t * cdf - np.cumsum(t * post)) / np.maximum(cdf, 1e-12), 0.0)
    X = np.column_stack([cdf, feats[1][0], feats[2][0], feats[3][0], np.full(L, pb),
                         np.full(L, np.log(pb + 1e-12) - np.log(1 - pb + 1e-12)), age]).astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)


def check():
    import pandas as pd
    from tsauc import tsauc
    from models.model10.sources import load_development_layout
    L = load_development_layout(ROOT)
    frame = pd.read_parquet(ROOT / "train/X_train.parquet")
    pos = L.series_position; rs_ = np.flatnonzero(np.r_[True, pos[1:] != pos[:-1]]); re_ = np.r_[rs_[1:], L.row_count]
    rows = []; X = []; t0 = time.time()
    for a, b in list(zip(rs_, re_))[:400]:
        s = frame.loc[int(L.series_ids[pos[a]])]
        X.append(series_oracle(s.loc[s.period == 1, "value"].to_numpy(np.float64), s.loc[s.period == 2, "value"].to_numpy(np.float64)))
        rows.append(np.arange(a, b))
    X = np.concatenate(X); r = np.concatenate(rows)
    print(f"400 series in {time.time() - t0:.1f}s; TS-AUC per feature:",
          {n: round(tsauc(L.target[r], X[:, j], L.time_online[r]), 4) for j, n in enumerate(NAMES)})
    old = np.load(ROOT / "models/model109/cache/teacher_features.npy", mmap_mode="r")
    print("old privileged z_cdf / r_cdf:", round(tsauc(L.target[r], np.asarray(old[r][:, 7]), L.time_online[r]), 4),
          round(tsauc(L.target[r], np.asarray(old[r][:, 16]), L.time_online[r]), 4))


def build_real():
    import pandas as pd
    from models.model10.sources import load_development_layout
    L = load_development_layout(ROOT)
    frame = pd.read_parquet(ROOT / "train/X_train.parquet")
    ids = frame.index.get_level_values(0).to_numpy(); values = frame["value"].to_numpy(np.float64); period = frame["period"].to_numpy()
    s0 = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]]); bounds = {int(ids[s]): (s, e) for s, e in zip(s0, np.r_[s0[1:], len(ids)])}
    pos = L.series_position; rs_ = np.flatnonzero(np.r_[True, pos[1:] != pos[:-1]]); re_ = np.r_[rs_[1:], L.row_count]
    X = np.empty((L.row_count, len(NAMES)), np.float32); t0 = time.time()
    for i, (a, b) in enumerate(zip(rs_, re_)):
        s, e = bounds[int(L.series_ids[pos[a]])]; v, p = values[s:e], period[s:e]
        X[a:b] = series_oracle(v[p == 1], v[p == 2])
        if i % 2000 == 0: print(f"real: {i} series ({time.time() - t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True); np.save(OUT / "real.npy", X); print("saved real", X.shape, flush=True)
    from tsauc import tsauc
    print({n: round(tsauc(L.target, X[:, j], L.time_online), 4) for j, n in enumerate(NAMES)}, flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    {"check": check, "real": build_real}[sys.argv[1]]()
