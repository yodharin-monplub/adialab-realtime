"""
detectors7.py -- detectors6 + the remaining high-value blocks from the
2025 winners' bank (union test showed +7 ROC points of independent
signal; this ports the streamable majority of it).

Additions on top of detectors6 (all O(1) or O(lags) per step):

Context (static per series): med0_raw, log_scale0, hist mean-med gap,
    hist skew, n_hist raw.
Boundary-local block, w in {32, 128}: once >= w online points exist,
    median/log-MAD/mean/log-SD/slope jumps between online[0:w] and
    hist[-w:]. Static after step w; partial (>= 8 pts) before.
Prefix ACF block: lag-1..8 autocorrelations of the online prefix
    (hist-standardized products), Ljung-Box statistic, and the short-lag
    L1 distance to the hist ACF.
Decayed crossing rate: EWMA (halflife 64) of the median-crossing
    indicator, as a logit delta vs the hist crossing rate.
Mean-vs-median: prefix mean minus binned prefix median.

Layout: detectors6's 258 features first, then these extras.
"""

from __future__ import annotations

import numpy as np

import detectors6 as d6
import detectors3 as d3

BW = (32, 128)
ACF_LAGS = 8
CR_ALPHA = 1.0 - 0.5 ** (1.0 / 64.0)      # halflife 64

EXTRA7 = (
    ["ctx_med0", "ctx_logscale0", "ctx_meanmed", "ctx_skew", "ctx_nhist"]
    + [f"b{w}_{k}" for w in BW
       for k in ("med_jump", "logmad_jump", "mean_jump", "logsd_jump",
                 "slope_jump")]
    + [f"acf{k}_delta" for k in range(1, ACF_LAGS + 1)]
    + ["lbq_delta", "shortlag_l1", "cr_decay_logit_delta", "mean_vs_med"]
)

N_FEATURES = d6.N_FEATURES + len(EXTRA7)


def feature_names():
    return d6.feature_names() + list(EXTRA7)


def _safe_log(x, eps=1e-9):
    return float(np.log(max(x, eps)))


def _seg_stats(seg):
    """(median, log MAD, mean, log SD, slope) of a 1-D segment."""
    med = float(np.median(seg))
    madv = float(np.median(np.abs(seg - med))) * 1.4826
    mean = float(seg.mean())
    sd = float(seg.std())
    n = seg.size
    if n >= 3:
        t = np.arange(n, dtype=np.float64)
        tc = t - t.mean()
        slope = float((tc * (seg - mean)).sum() / max((tc * tc).sum(), 1e-12))
    else:
        slope = 0.0
    return med, _safe_log(madv), mean, _safe_log(sd), slope


class Hist7:
    def __init__(self, x_hist):
        h = np.asarray(x_hist, dtype=np.float64)
        hr = d6.HistRef(x_hist)
        self.hr = hr
        zr = hr.zr(h)
        self.ctx_med0 = float(np.median(h))
        self.ctx_logscale0 = _safe_log(hr.scale0)
        self.ctx_meanmed = float(h.mean() - np.median(h)) / hr.scale0
        m2 = float((zr ** 2).mean())
        m3 = float((zr ** 3).mean())
        self.ctx_skew = m3 / max(m2 ** 1.5, 1e-9)
        self.ctx_nhist = float(h.size)

        self.tail_stats = {w: _seg_stats(zr[-min(w, zr.size):]) for w in BW}

        # hist ACF at lags 1..8 (on zr, hist-standardized products)
        var = max(float((zr * zr).mean()), 1e-9)
        self.acf0 = np.array([
            (float((zr[k:] * zr[:-k]).mean()) / var) if zr.size > k else 0.0
            for k in range(1, ACF_LAGS + 1)])
        self.var0 = var

        med = float(np.median(zr))
        above = zr > med
        self.cr0 = float((above[1:] != above[:-1]).mean()) \
            if zr.size > 1 else 0.5
        self.cr0 = min(max(self.cr0, 1e-3), 1 - 1e-3)
        self.med_zr = med


def _logit(p):
    p = min(max(p, 1e-3), 1 - 1e-3)
    return float(np.log(p / (1 - p)))


class StreamDetector7:
    def __init__(self, x_hist):
        self._inner = d6.StreamDetector6(x_hist)
        self.h7 = Hist7(x_hist)
        self.t = 0
        self._head = {w: [] for w in BW}     # first w online zr values
        self._bcache = {w: None for w in BW}
        # prefix ACF state on zr
        self._zlags = []
        self._sum_prod = np.zeros(ACF_LAGS)
        self._sum_z = 0.0
        self._sum_z2 = 0.0
        # decayed crossing rate
        self._cr_ewma = self.h7.cr0
        self._above_prev = None

    def step(self, x):
        base = self._inner.step(x)
        h7 = self.h7
        hr = h7.hr
        t = self.t
        n = t + 1
        zr = (float(x) - hr.med0) / hr.scale0

        out = np.empty(N_FEATURES)
        out[:d6.N_FEATURES] = base
        E = out[d6.N_FEATURES:]
        i = 0
        E[0] = h7.ctx_med0
        E[1] = h7.ctx_logscale0
        E[2] = h7.ctx_meanmed
        E[3] = h7.ctx_skew
        E[4] = h7.ctx_nhist
        i = 5

        for w in BW:
            head = self._head[w]
            if len(head) < w:
                head.append(zr)
                self._bcache[w] = None
            if self._bcache[w] is None:
                if len(head) >= 8 or len(head) == w:
                    hs_ = h7.tail_stats[w]
                    os_ = _seg_stats(np.asarray(head))
                    vals = (os_[0] - hs_[0], os_[1] - hs_[1],
                            os_[2] - hs_[2], os_[3] - hs_[3],
                            os_[4] - hs_[4])
                else:
                    vals = (0.0, 0.0, 0.0, 0.0, 0.0)
                if len(head) == w:
                    self._bcache[w] = vals   # frozen from now on
            else:
                vals = self._bcache[w]
            E[i:i + 5] = vals
            i += 5

        # prefix ACF vs hist
        zl = self._zlags
        for k in range(1, min(ACF_LAGS, t) + 1):
            self._sum_prod[k - 1] += zr * zl[-k]
        self._sum_z += zr
        self._sum_z2 += zr * zr
        var_p = max(self._sum_z2 / n - (self._sum_z / n) ** 2, 1e-9)
        acf = np.zeros(ACF_LAGS)
        for k in range(1, ACF_LAGS + 1):
            m = n - k
            if m >= 3:
                acf[k - 1] = (self._sum_prod[k - 1] / m) / var_p
        delta = acf - h7.acf0
        E[i:i + ACF_LAGS] = delta
        i += ACF_LAGS
        E[i] = n * float((acf * acf).sum())          # LBQ-style
        E[i + 1] = float(np.abs(delta).sum())        # shortlag L1
        i += 2

        above = zr > h7.med_zr
        if self._above_prev is not None:
            flip = 1.0 if above != self._above_prev else 0.0
            self._cr_ewma = (1 - CR_ALPHA) * self._cr_ewma + CR_ALPHA * flip
        self._above_prev = above
        E[i] = _logit(self._cr_ewma) - _logit(h7.cr0)
        i += 1

        # mean vs binned median of prefix (reuse inner's bin counts)
        inner6 = self._inner
        F = np.cumsum(inner6.zcnt / n)
        med_pref = d6._bin_quantile(F, hr.z_reps, 0.5)
        E[i] = self._sum_z / n - med_pref

        zl.append(zr)
        if len(zl) > ACF_LAGS:
            zl.pop(0)
        self.t = n
        return out


def compute_features_offline(x_hist, x_online):
    """Offline = run the stream (the extra blocks are cheap scalar state;
    d6's base features inside are computed via its own fast streaming
    path... for training-scale runs use compute_features_offline_fast)."""
    det = StreamDetector7(x_hist)
    return np.vstack([det.step(v) for v in np.asarray(x_online, float)])


def compute_features_offline_fast(x_hist, x_online):
    """Vectorized offline: d6 fast path + vectorized extras."""
    F6 = d6.compute_features_offline(x_hist, x_online)
    h7 = Hist7(x_hist)
    hr = h7.hr
    z = hr.zr(x_online)
    T = z.size
    n = np.arange(1, T + 1, dtype=np.float64)

    E = np.zeros((T, len(EXTRA7)))
    E[:, 0] = h7.ctx_med0
    E[:, 1] = h7.ctx_logscale0
    E[:, 2] = h7.ctx_meanmed
    E[:, 3] = h7.ctx_skew
    E[:, 4] = h7.ctx_nhist
    i = 5

    for w in BW:
        hs_ = h7.tail_stats[w]
        for t in range(min(T, w)):
            m = t + 1
            if m >= 8:
                os_ = _seg_stats(z[:m])
                E[t, i:i + 5] = (os_[0] - hs_[0], os_[1] - hs_[1],
                                 os_[2] - hs_[2], os_[3] - hs_[3],
                                 os_[4] - hs_[4])
        if T > w:
            os_ = _seg_stats(z[:w])
            E[w:, i:i + 5] = (os_[0] - hs_[0], os_[1] - hs_[1],
                              os_[2] - hs_[2], os_[3] - hs_[3],
                              os_[4] - hs_[4])
        i += 5

    # prefix ACF block, vectorized over t
    s_z = np.cumsum(z)
    s_z2 = np.cumsum(z * z)
    var_p = np.maximum(s_z2 / n - (s_z / n) ** 2, 1e-9)
    acf_cols = np.zeros((T, ACF_LAGS))
    for k in range(1, ACF_LAGS + 1):
        prod = np.zeros(T)
        if T > k:
            prod[k:] = np.cumsum(z[k:] * z[:-k])
        m = n - k
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(m >= 3, (prod / np.maximum(m, 1)) / var_p, 0.0)
        acf_cols[:, k - 1] = a
    delta = acf_cols - h7.acf0[None, :]
    E[:, i:i + ACF_LAGS] = delta
    i += ACF_LAGS
    E[:, i] = n * (acf_cols * acf_cols).sum(axis=1)
    E[:, i + 1] = np.abs(delta).sum(axis=1)
    i += 2

    above = z > h7.med_zr
    cr = np.empty(T)
    ew = h7.cr0
    prev = None
    for t in range(T):
        if prev is not None:
            flip = 1.0 if above[t] != prev else 0.0
            ew = (1 - CR_ALPHA) * ew + CR_ALPHA * flip
        prev = above[t]
        cr[t] = ew
    crc = np.clip(cr, 1e-3, 1 - 1e-3)
    cr0c = min(max(h7.cr0, 1e-3), 1 - 1e-3)
    E[:, i] = np.log(crc / (1 - crc)) - np.log(cr0c / (1 - cr0c))
    i += 1

    # mean vs binned median (recompute the cumulative z-bin CDF)
    zb = np.searchsorted(hr.z_edges, z, side="right")
    Zoh = np.zeros((T, hr.nzb))
    Zoh[np.arange(T), zb] = 1.0
    ZCDF = np.cumsum(np.cumsum(Zoh, axis=0) / n[:, None], axis=1)
    idx = np.minimum((ZCDF < 0.5).sum(axis=1), hr.z_reps.size - 1)
    E[:, i] = s_z / n - hr.z_reps[idx]

    return np.column_stack([F6, E])


if __name__ == "__main__":
    import time
    rng = np.random.default_rng(0)
    for trial in range(6):
        n_hist = int(rng.integers(60, 500))
        T = int(rng.integers(5, 600))
        a = rng.uniform(-0.6, 0.6)
        eps = rng.normal(size=n_hist + T)
        xs = np.empty(n_hist + T)
        xs[0] = eps[0]
        for k in range(1, n_hist + T):
            xs[k] = a * xs[k - 1] + eps[k]
        x_hist, x_online = xs[:n_hist].copy(), xs[n_hist:].copy()
        if trial % 2 == 0 and T > 10:
            tau = int(rng.integers(1, T - 1))
            x_online[tau:] *= 1.8
        F_slow = compute_features_offline(x_hist, x_online)
        F_fast = compute_features_offline_fast(x_hist, x_online)
        assert F_slow.shape == (T, N_FEATURES)
        err = np.max(np.abs(F_slow - F_fast))
        assert err < 1e-8, (trial, err)
    assert len(feature_names()) == N_FEATURES
    det = StreamDetector7(rng.normal(size=3000))
    xs = rng.normal(size=1000)
    t0 = time.perf_counter()
    for v in xs:
        det.step(v)
    dt = (time.perf_counter() - t0) / 1000
    print(f"OK: stream == offline(fast); {N_FEATURES} features; "
          f"{dt*1e6:.0f} us/step")
