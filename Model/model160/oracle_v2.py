"""Model160: more oracle channels for the teacher (privileged, whole online segment).

Model157's oracle posterior uses the AR-whitened normal-score stream u with scale / lag-1 dependence / mean
break alternatives.  Two extra views:
  raw  : the same Bayesian posterior on normal scores of the RAW level (history ECDF of the raw values, no AR
         whitening) - a break in the AR structure itself can be partly whitened away;
  dep2 : lag-2 dependence alternative on u.
Features (6): ob2_raw_cdf, ob2_raw_pbreak, ob2_raw_scale_cdf, ob2_dep2_cdf, ob2_dep2_pbreak, ob2_mix4_cdf
(mixture of scale/dep/mean/dep2 on u).
Usage: python models/model160/oracle_v2.py real | probe
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
NAMES = ["ob2_raw_cdf", "ob2_raw_pbreak", "ob2_raw_scale_cdf", "ob2_dep2_cdf", "ob2_dep2_pbreak", "ob2_mix4_cdf"]


def _lme(a):
    return logsumexp(a, axis=0) - np.log(a.shape[0])


def _rs(v):
    return np.cumsum(v[::-1])[::-1]


def channel_lbf(u, prev):
    """Per candidate break k: log BF of scale, dependence (vs prev), mean alternatives on suffix u[k:]."""
    m = np.arange(len(u), 0, -1, dtype=np.float64)
    S, T, Sxy, Syy = _rs(u * u), _rs(u), _rs(u * prev), _rs(prev * prev)
    a = A_SCALE[:, None]; b = a - 1.0
    ls = _lme(a * np.log(b) - gammaln(a) + gammaln(a + m / 2) - (a + m / 2) * np.log(b + S / 2) + S / 2)
    s2 = (S_MEAN ** 2)[:, None]
    lm = _lme(-0.5 * np.log1p(m * s2) + s2 * T * T / (2 * (1 + m * s2)))
    r = R_DEP[:, None]; q = 1 - r * r
    ld = _lme(-0.5 * m * np.log(q) - (S - 2 * r * Sxy + r * r * Syy) / (2 * q) + S / 2)
    return ls, ld, lm


def posterior(lbf):
    L = len(lbf)
    lp = np.log(0.5 / L) + lbf
    lz = np.logaddexp(logsumexp(lp), np.log(0.5))
    post = np.exp(lp - lz)
    return np.cumsum(post), float(post.sum())


def ecdf_scores(hs, x):
    return np.array([_ndtri((np.searchsorted(hs, v) + 0.5) / (len(hs) + 1.0)) for v in x])


def series_oracle2(h, o):
    h = np.asarray(h, np.float64); o = np.asarray(o, np.float64)
    L = len(o)
    # whitened stream (as Model157) + lag-2 dependence
    z, hs, ulast = normal_scores(h, o)
    u = ecdf_scores(hs, z)
    prev1 = np.r_[ulast, u[:-1]]
    ls, ld, lm = channel_lbf(u, prev1)
    prev2 = np.r_[0.0, 0.0, u[:-2]][:L] if L > 2 else np.zeros(L)
    _, ld2, _ = channel_lbf(u, prev2)
    # raw-level stream
    hr = np.sort(h[-1000:])
    w = ecdf_scores(hr, o)
    wl = float(_ndtri((np.searchsorted(hr, h[-1]) + 0.5) / (len(hr) + 1.0)))
    rs, rd, rm = channel_lbf(w, np.r_[wl, w[:-1]])
    raw_mix = _lme(np.vstack([rs, rd, rm]))
    c_raw, p_raw = posterior(raw_mix)
    c_rs, _ = posterior(rs)
    c_d2, p_d2 = posterior(ld2)
    c_m4, _ = posterior(_lme(np.vstack([ls, ld, lm, ld2])))
    X = np.column_stack([c_raw, np.full(L, p_raw), c_rs, c_d2, np.full(L, p_d2), c_m4]).astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)


def build_real():
    import pandas as pd
    from models.model10.sources import load_development_layout
    from tsauc import tsauc
    Ly = load_development_layout(ROOT)
    frame = pd.read_parquet(ROOT / "train/X_train.parquet")
    ids = frame.index.get_level_values(0).to_numpy(); values = frame["value"].to_numpy(np.float64); period = frame["period"].to_numpy()
    s0 = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]]); bounds = {int(ids[s]): (s, e) for s, e in zip(s0, np.r_[s0[1:], len(ids)])}
    pos = Ly.series_position; rs_ = np.flatnonzero(np.r_[True, pos[1:] != pos[:-1]]); re_ = np.r_[rs_[1:], Ly.row_count]
    X = np.empty((Ly.row_count, len(NAMES)), np.float32); t0 = time.time()
    for i, (a, b) in enumerate(zip(rs_, re_)):
        s, e = bounds[int(Ly.series_ids[pos[a]])]; v, p = values[s:e], period[s:e]
        X[a:b] = series_oracle2(v[p == 1], v[p == 2])
        if i % 2000 == 0: print(f"real: {i} series ({time.time() - t0:.0f}s)", flush=True)
    OUT.mkdir(parents=True, exist_ok=True); np.save(OUT / "real.npy", X)
    print({n: round(tsauc(Ly.target, X[:, j], Ly.time_online), 4) for j, n in enumerate(NAMES)}, flush=True)


def probe():
    """Same teacher slot as models/model157/teacher_probe.py (held fold 0, j=2): +these 6 features."""
    import models.model147.run_model147 as m147
    import lightgbm as lgb
    from models.model20.run_model20 import fitting_rows
    from tsauc import time_pair_weights, tsauc
    _, protocol, layout, real, pool, old = m147.m111.load_everything()
    new = np.load(ROOT / "models/model157/data/real.npy", mmap_mode="r"); v2 = np.load(OUT / "real.npy", mmap_mode="r")
    fold, j = 0, 2
    fit, _ = fitting_rows(layout, fold, protocol["contract"]["recipes"][fold]["seed"])
    union = protocol["contract"]["union_names"]
    columns = np.asarray([union.index(n) for n in protocol["contract"]["names"][fold]])
    tr = fit[layout.row_fold[fit] == j]
    pool_rows = np.flatnonzero(~np.isin(layout.row_fold, [fold, (fold + 1) % 5, j]))
    rows = np.sort(np.random.default_rng(7000 + 10 * fold + j).choice(pool_rows, 450_000, replace=False))
    for name, extra in (("oracle7", False), ("oracle7+v2", True)):
        parts = lambda r: [real[r][:, columns], np.asarray(old[r]), np.asarray(new[r])] + ([np.asarray(v2[r])] if extra else [])  # noqa: E731
        y, t = layout.target[rows], layout.time_online[rows]
        model = lgb.LGBMClassifier(objective="binary", random_state=31 + j, n_jobs=8, verbosity=-1, force_col_wise=True,
                                   **m147.m111.TEACHER_PARAMS)
        model.fit(np.hstack(parts(rows)), y, sample_weight=time_pair_weights(y, t))
        p = model.predict_proba(np.hstack(parts(tr)))[:, 1]
        print(f"{name}: teacher TS-AUC {tsauc(layout.target[tr], p, layout.time_online[tr]):.5f}", flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    {"real": build_real, "probe": probe}[sys.argv[1]]()
