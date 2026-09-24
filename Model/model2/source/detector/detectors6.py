"""
detectors6.py -- detectors3 + robust distributional / dependence prefix
statistics, all incrementally streamable.

Motivated by the 2025 winners' feature bank (validated at ROC 0.706 on the
known-boundary task vs 0.625 for detectors3): the missing signal lives in
ROBUST two-sample statistics. Everything here is computed against fixed
hist-derived references so each statistic is O(1)/O(B) per step:

Binned-CDF block (B = 32 bins at hist quantiles), for z and dz streams:
    ks_bin      max_b |F_prefix(b) - F_hist(b)|
    cvm_bin     mean_b (F_prefix - F_hist)^2      (Cramér-von Mises-ish)
    w1_bin      mean_b |F_prefix - F_hist|        (binned Wasserstein-1)
    js_bin      Jensen-Shannon divergence of bin masses
  plus for z only, bin-quantile deltas at 10/25/50/75/90 and an
  IDR log-ratio (q90-q10 of prefix vs hist).

Trend block (on winsorized z):  OLS slope t-stat of prefix-vs-time, and
  log variance ratio of detrended prefix vs raw prefix.

Ordinal block: permutation-entropy (m=3) delta and alternation-rate delta
  of the prefix vs hist baselines.

Robust-scale block: log-ratio of prefix mean|z - med0| vs hist,
  interquartile-based scale delta from bins.

Rolling 128-window binned-CDF block for z: ks/w1/js on the last 128
  points (ring-buffer bin counts), catching mid-stream distribution
  shifts without prefix dilution.

Layout: detectors3's 236 features come first (verbatim), then this file's
extras.  Offline == streaming verified in __main__.
"""

from __future__ import annotations

import numpy as np

import detectors3 as d3

N_BINS = 32
ROLL_W = 128
QIDX = {"q10": 0.10, "q25": 0.25, "q50": 0.50, "q75": 0.75, "q90": 0.90}

EXTRA_NAMES = (
    ["z_ks", "z_cvm", "z_w1", "z_js",
     "dz_ks", "dz_cvm", "dz_w1", "dz_js"]
    + [f"z_{q}_delta" for q in QIDX]
    + ["z_idr_logratio", "z_mad_logratio",
       "slope_tstat", "detrend_logvr",
       "perm_ent_delta", "altern_delta",
       "roll_ks", "roll_w1", "roll_js"]
)

N_FEATURES = d3.N_FEATURES + len(EXTRA_NAMES)


def feature_names():
    return d3.feature_names() + list(EXTRA_NAMES)


def _safe_log(x, eps=1e-9):
    return np.log(np.maximum(x, eps))


class HistRef:
    """Fixed references from the hist segment for the extra blocks."""

    def __init__(self, x_hist):
        h = np.asarray(x_hist, dtype=np.float64)
        self.med0 = float(np.median(h))
        mad = float(np.median(np.abs(h - self.med0))) * 1.4826
        sd = float(h.std())
        self.scale0 = mad if mad > 1e-9 else (sd if sd > 1e-9 else 1e-9)
        zr = (h - self.med0) / self.scale0
        self.clip_lo = float(np.quantile(zr, 0.01))
        self.clip_hi = float(np.quantile(zr, 0.99))

        # z bins at hist quantiles (open-ended outer bins)
        qs = np.linspace(0.0, 1.0, N_BINS + 1)[1:-1]
        self.z_edges = np.unique(np.quantile(zr, qs))
        zb = np.searchsorted(self.z_edges, zr, side="right")
        nb = self.z_edges.size + 1
        self.nzb = nb
        cnt = np.bincount(zb, minlength=nb).astype(np.float64)
        self.z_hist_pmf = cnt / cnt.sum()
        self.z_hist_cdf = np.cumsum(self.z_hist_pmf)
        # representative value per bin (median of hist points in the bin)
        reps = np.zeros(nb)
        for b in range(nb):
            pts = zr[zb == b]
            reps[b] = float(np.median(pts)) if pts.size else 0.0
        self.z_reps = reps

        dz = np.diff(zr)
        if dz.size < 4:
            dz = np.array([-1.0, 0.0, 1.0])
        self.dz_med0 = float(np.median(dz))
        dmad = float(np.median(np.abs(dz - self.dz_med0))) * 1.4826
        self.dz_scale0 = dmad if dmad > 1e-9 else 1e-9
        dzr = (dz - self.dz_med0) / self.dz_scale0
        self.dz_edges = np.unique(np.quantile(dzr, qs))
        dzb = np.searchsorted(self.dz_edges, dzr, side="right")
        ndb = self.dz_edges.size + 1
        self.ndzb = ndb
        dcnt = np.bincount(dzb, minlength=ndb).astype(np.float64)
        self.dz_hist_pmf = dcnt / dcnt.sum()
        self.dz_hist_cdf = np.cumsum(self.dz_hist_pmf)

        # hist quantiles of zr for the quantile-delta features
        self.z_q0 = {k: float(np.quantile(zr, v)) for k, v in QIDX.items()}
        idr = self.z_q0["q90"] - self.z_q0["q10"]
        self.idr0 = idr if abs(idr) > 1e-9 else 1e-9
        self.absdev0 = float(np.mean(np.abs(zr)))
        if self.absdev0 < 1e-9:
            self.absdev0 = 1e-9

        # ordinal (m=3) pattern distribution + alternation rate of hist
        self.pe0, self.alt0 = _ordinal_stats_arr(zr)

    def zr(self, x):
        return (np.asarray(x, dtype=np.float64) - self.med0) / self.scale0


def _pattern_index(a, b, c):
    """Ordinal pattern of a triple -> 0..5 (ties resolved by order)."""
    if a <= b:
        if b <= c:
            return 0
        return 1 if a <= c else 2
    if b <= c:
        return 3 if a <= c else 4
    return 5


def _ordinal_stats_arr(z):
    """(perm-entropy_m3, alternation rate) of an array."""
    n = z.size
    if n < 4:
        return 0.0, 0.5
    cnt = np.zeros(6)
    for i in range(n - 2):
        cnt[_pattern_index(z[i], z[i + 1], z[i + 2])] += 1
    p = cnt / cnt.sum()
    pe = float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(6.0))
    s = np.sign(np.diff(z))
    alt = float((s[1:] * s[:-1] < 0).mean()) if s.size > 1 else 0.5
    return pe, alt


def _dist_feats(cnt, total, hist_pmf, hist_cdf):
    """ks, cvm, w1, js from bin counts vs the hist reference."""
    if total <= 0:
        return 0.0, 0.0, 0.0, 0.0
    p = cnt / total
    F = np.cumsum(p)
    d = F - hist_cdf
    ks = float(np.max(np.abs(d)))
    cvm = float(np.mean(d * d))
    w1 = float(np.mean(np.abs(d)))
    m = 0.5 * (p + hist_pmf)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(p > 0, p * (_safe_log(p) - _safe_log(m)), 0.0)
        t2 = np.where(hist_pmf > 0,
                      hist_pmf * (_safe_log(hist_pmf) - _safe_log(m)), 0.0)
    js = float(0.5 * (t1.sum() + t2.sum()))
    return ks, cvm, w1, js


def _bin_quantile(F, reps, q):
    """Approximate quantile from binned CDF using bin representatives."""
    b = int(np.searchsorted(F, q, side="left"))
    b = min(b, reps.size - 1)
    return float(reps[b])


def _dist_feats_matrix(CDF, PMF, hist_pmf, hist_cdf):
    """Vectorized ks/cvm/w1/js for a (T, B) stack of per-step CDFs/PMFs."""
    d = CDF - hist_cdf[None, :]
    ks = np.max(np.abs(d), axis=1)
    cvm = np.mean(d * d, axis=1)
    w1 = np.mean(np.abs(d), axis=1)
    m = 0.5 * (PMF + hist_pmf[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(PMF > 0, PMF * (_safe_log(PMF) - _safe_log(m)), 0.0)
        t2 = np.where(hist_pmf[None, :] > 0,
                      hist_pmf[None, :] * (_safe_log(hist_pmf[None, :])
                                           - _safe_log(m)), 0.0)
    js = 0.5 * (t1.sum(axis=1) + t2.sum(axis=1))
    return ks, cvm, w1, js


def _bin_quantiles_matrix(CDF, reps, qs):
    """(T, len(qs)) approximate quantiles from per-step binned CDFs."""
    out = np.empty((CDF.shape[0], len(qs)))
    for j, q in enumerate(qs):
        idx = np.minimum((CDF < q).sum(axis=1), reps.size - 1)
        out[:, j] = reps[idx]
    return out


def _extra_offline(hr: HistRef, x_online):
    """Vectorized twin of the streaming extra block."""
    z = hr.zr(x_online)
    T = z.size
    n = np.arange(1, T + 1, dtype=np.float64)
    zc = np.clip(z, hr.clip_lo, hr.clip_hi)

    zb = np.searchsorted(hr.z_edges, z, side="right")
    dz_full = np.empty(T)
    dz_full[0] = 0.0
    dz_full[1:] = np.diff(z)
    dzr = (dz_full - hr.dz_med0) / hr.dz_scale0
    dzb = np.searchsorted(hr.dz_edges, dzr, side="right")

    out = np.empty((T, len(EXTRA_NAMES)))

    # ---- cumulative one-hot bin counts (T, B) ----
    Zoh = np.zeros((T, hr.nzb))
    Zoh[np.arange(T), zb] = 1.0
    Zcnt = np.cumsum(Zoh, axis=0)
    ZPMF = Zcnt / n[:, None]
    ZCDF = np.cumsum(ZPMF, axis=1)

    Doh = np.zeros((T, hr.ndzb))
    if T > 1:
        Doh[np.arange(1, T), dzb[1:]] = 1.0
    Dcnt = np.cumsum(Doh, axis=0)
    nd = np.maximum(n - 1, 1.0)
    DPMF = Dcnt / nd[:, None]
    DCDF = np.cumsum(DPMF, axis=1)

    ks, cvm, w1, js = _dist_feats_matrix(ZCDF, ZPMF, hr.z_hist_pmf,
                                         hr.z_hist_cdf)
    out[:, 0], out[:, 1], out[:, 2], out[:, 3] = ks, cvm, w1, js
    ks, cvm, w1, js = _dist_feats_matrix(DCDF, DPMF, hr.dz_hist_pmf,
                                         hr.dz_hist_cdf)
    out[:, 4], out[:, 5], out[:, 6], out[:, 7] = ks, cvm, w1, js
    out[0, 4:8] = 0.0
    i = 8

    Q = _bin_quantiles_matrix(ZCDF, hr.z_reps, list(QIDX.values()))
    for k, qname in enumerate(QIDX):
        out[:, i + k] = Q[:, k] - hr.z_q0[qname]
    i += len(QIDX)

    Qw = _bin_quantiles_matrix(ZCDF, hr.z_reps, [0.90, 0.10])
    out[:, i] = _safe_log(np.maximum(Qw[:, 0] - Qw[:, 1], 1e-9)) \
        - _safe_log(hr.idr0)
    out[:, i + 1] = _safe_log(np.cumsum(np.abs(z)) / n) - _safe_log(hr.absdev0)
    i += 2

    # ---- slope t-stat + detrended log variance ratio (on zc) ----
    t_idx = np.arange(T, dtype=np.float64)
    s_y = np.cumsum(zc)
    s_ty = np.cumsum(t_idx * zc)
    s_y2 = np.cumsum(zc * zc)
    tbar = (n - 1) / 2.0
    sxx = n * (n * n - 1) / 12.0
    sxy = s_ty - tbar * s_y
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(sxx > 0, sxy / np.maximum(sxx, 1e-12), 0.0)
        resid_var = np.maximum(s_y2 - s_y * s_y / n - slope * sxy, 1e-12) / n
        se = np.sqrt(resid_var / np.maximum(sxx, 1e-12)) + 1e-12
        tstat = np.clip(slope / se, -50, 50)
        var_raw = np.maximum(s_y2 / n - (s_y / n) ** 2, 1e-12)
        var_det = np.maximum(var_raw - slope * sxy / n, 1e-12)
        logvr = _safe_log(var_det) - _safe_log(var_raw)
    tstat[n < 3] = 0.0
    logvr[n < 3] = 0.0
    out[:, i] = tstat
    out[:, i + 1] = logvr
    i += 2

    # ---- ordinal patterns + alternation (small loop, O(T) scalar ops) ----
    pe_col = np.zeros(T)
    alt_col = np.zeros(T)
    pat_cnt = np.zeros(6)
    alt_flips = alt_tot = 0
    prev_sign = 0.0
    logs = np.log(6.0)
    for t in range(T):
        if t >= 2:
            pat_cnt[_pattern_index(z[t - 2], z[t - 1], z[t])] += 1
        if t >= 1:
            sg = np.sign(z[t] - z[t - 1])
            if prev_sign != 0.0 and sg != 0.0:
                alt_tot += 1
                if sg * prev_sign < 0:
                    alt_flips += 1
            if sg != 0.0:
                prev_sign = sg
        tot = pat_cnt.sum()
        if tot >= 5:
            pp = pat_cnt / tot
            pe_col[t] = -(pp[pp > 0] * np.log(pp[pp > 0])).sum() / logs \
                - hr.pe0
        if alt_tot >= 5:
            alt_col[t] = alt_flips / alt_tot - hr.alt0
    out[:, i] = pe_col
    out[:, i + 1] = alt_col
    i += 2

    # ---- rolling 128-window binned distances ----
    Roll = Zcnt.copy()
    if T > ROLL_W:
        Roll[ROLL_W:] -= Zcnt[:-ROLL_W]
    mroll = np.minimum(n, ROLL_W)
    RPMF = Roll / mroll[:, None]
    RCDF = np.cumsum(RPMF, axis=1)
    ks, cvm, w1, js = _dist_feats_matrix(RCDF, RPMF, hr.z_hist_pmf,
                                         hr.z_hist_cdf)
    out[:, i] = ks
    out[:, i + 1] = w1
    out[:, i + 2] = js
    return out


def compute_features_offline(x_hist, x_online):
    F3 = d3.compute_features_offline(x_hist, x_online)
    hr = HistRef(x_hist)
    E = _extra_offline(hr, x_online)
    return np.column_stack([F3, E])


class StreamDetector6:
    """Streaming twin: wraps StreamDetector3 and mirrors _extra_offline's
    state machine step by step (the offline loop above IS the streaming
    algorithm, so this just re-uses it incrementally)."""

    def __init__(self, x_hist):
        self._inner = d3.StreamDetector3(x_hist)
        self.hr = HistRef(x_hist)
        hr = self.hr
        self.t = 0
        self.zcnt = np.zeros(hr.nzb)
        self.dcnt = np.zeros(hr.ndzb)
        self.rollbuf = np.zeros(ROLL_W, dtype=np.int64)
        self.rollcnt = np.zeros(hr.nzb)
        self.s_y = self.s_ty = self.s_y2 = 0.0
        self.pat_cnt = np.zeros(6)
        self.alt_flips = 0
        self.alt_tot = 0
        self.z1 = self.z2 = None
        self.prev_sign = 0.0
        self.absdev_sum = 0.0
        self.z_prev_raw = None

    def step(self, x):
        base = self._inner.step(x)
        hr = self.hr
        t = self.t
        n = t + 1
        z = (float(x) - hr.med0) / hr.scale0
        zc = min(max(z, hr.clip_lo), hr.clip_hi)
        zb = int(np.searchsorted(hr.z_edges, z, side="right"))

        out = np.empty(N_FEATURES)
        out[:d3.N_FEATURES] = base
        E = out[d3.N_FEATURES:]

        self.zcnt[zb] += 1
        if t >= 1:
            dzr = ((z - self.z_prev_raw) - hr.dz_med0) / hr.dz_scale0
            dzb = int(np.searchsorted(hr.dz_edges, dzr, side="right"))
            self.dcnt[dzb] += 1
        pos = t % ROLL_W
        if t >= ROLL_W:
            self.rollcnt[self.rollbuf[pos]] -= 1
        self.rollbuf[pos] = zb
        self.rollcnt[zb] += 1

        self.s_y += zc
        self.s_ty += t * zc
        self.s_y2 += zc * zc
        self.absdev_sum += abs(z)

        if self.z2 is not None:
            self.pat_cnt[_pattern_index(self.z2, self.z1, z)] += 1
        if self.z1 is not None:
            sg = np.sign(z - self.z1)
            if self.prev_sign != 0.0 and sg != 0.0:
                self.alt_tot += 1
                if sg * self.prev_sign < 0:
                    self.alt_flips += 1
            if sg != 0.0:
                self.prev_sign = sg
        self.z2, self.z1 = self.z1, z

        E[0:4] = _dist_feats(self.zcnt, n, hr.z_hist_pmf, hr.z_hist_cdf)
        E[4:8] = _dist_feats(self.dcnt, t, hr.dz_hist_pmf, hr.dz_hist_cdf) \
            if t >= 1 else (0.0, 0.0, 0.0, 0.0)
        i = 8

        p = self.zcnt / n
        F = np.cumsum(p)
        for k, (qname, qv) in enumerate(QIDX.items()):
            E[i + k] = _bin_quantile(F, hr.z_reps, qv) - hr.z_q0[qname]
        i += len(QIDX)
        q90 = _bin_quantile(F, hr.z_reps, 0.90)
        q10 = _bin_quantile(F, hr.z_reps, 0.10)
        E[i] = _safe_log(max(q90 - q10, 1e-9)) - _safe_log(hr.idr0)
        E[i + 1] = _safe_log(self.absdev_sum / n) - _safe_log(hr.absdev0)
        i += 2

        if n >= 3:
            tbar = (n - 1) / 2.0
            sxx = n * (n * n - 1) / 12.0
            sxy = self.s_ty - tbar * self.s_y
            slope = sxy / sxx
            resid_var = max(self.s_y2 - self.s_y * self.s_y / n
                            - slope * sxy, 1e-12) / n
            se = np.sqrt(resid_var / sxx) + 1e-12
            E[i] = np.clip(slope / se, -50, 50)
            var_raw = max(self.s_y2 / n - (self.s_y / n) ** 2, 1e-12)
            var_det = max(var_raw - slope * sxy / n, 1e-12)
            E[i + 1] = _safe_log(var_det) - _safe_log(var_raw)
        else:
            E[i] = E[i + 1] = 0.0
        i += 2

        tot = self.pat_cnt.sum()
        if tot >= 5:
            pp = self.pat_cnt / tot
            E[i] = float(-(pp[pp > 0] * np.log(pp[pp > 0])).sum()
                         / np.log(6.0)) - hr.pe0
        else:
            E[i] = 0.0
        E[i + 1] = (self.alt_flips / self.alt_tot - hr.alt0) \
            if self.alt_tot >= 5 else 0.0
        i += 2

        m = min(n, ROLL_W)
        ks, cvm, w1, js = _dist_feats(self.rollcnt, m,
                                      hr.z_hist_pmf, hr.z_hist_cdf)
        E[i] = ks
        E[i + 1] = w1
        E[i + 2] = js

        self.z_prev_raw = z
        self.t = n
        return out


if __name__ == "__main__":
    import time
    rng = np.random.default_rng(0)
    for trial in range(8):
        n_hist = int(rng.integers(60, 500))
        T = int(rng.integers(5, 700))
        a = rng.uniform(-0.6, 0.6)
        eps = rng.normal(size=n_hist + T)
        xs = np.empty(n_hist + T)
        xs[0] = eps[0]
        for k in range(1, n_hist + T):
            xs[k] = a * xs[k - 1] + eps[k]
        x_hist, x_online = xs[:n_hist].copy(), xs[n_hist:].copy()
        if trial % 2 == 0 and T > 10:
            tau = int(rng.integers(1, T - 1))
            if trial % 4 == 0:
                x_online[tau:] += 1.0
            else:
                x_online[tau:] *= 1.8
        F_off = compute_features_offline(x_hist, x_online)
        det = StreamDetector6(x_hist)
        F_str = np.vstack([det.step(v) for v in x_online])
        assert F_off.shape == (T, N_FEATURES), (F_off.shape, N_FEATURES)
        err = np.max(np.abs(F_off - F_str))
        assert err < 1e-9, (trial, err)
    assert len(feature_names()) == N_FEATURES
    det = StreamDetector6(rng.normal(size=3000))
    xs = rng.normal(size=1000)
    t0 = time.perf_counter()
    for v in xs:
        det.step(v)
    dt = (time.perf_counter() - t0) / 1000
    print(f"OK: stream == offline; {N_FEATURES} features; {dt*1e6:.0f} us/step")
