"""
detectors2.py -- calibrated multi-aspect change detection features.

WHY v2
------
Diagnostics on the training data (diagnose.py) showed the breaks are subtle
and heterogeneous: median post-break mean shift is 0.05 hist-sd, median
sd-ratio 0.985, median KS 0.08, and lag-1 autocorrelation changes are as
large as everything else. Two consequences drive this design:

1.  MANY ASPECTS. Each online point contributes an "increment" to several
    aspect processes: location, scale, distribution shape (Legendre
    polynomials of the hist-CDF rank), tails, and dependence (multi-lag
    autocorrelation, volatility clustering, sign agreement). A break in ANY
    of these shows up as a drifting mean in the corresponding process.

2.  PER-SERIES H0 CALIBRATION. The metric is a cross-sectional AUC, so
    statistics must be comparable ACROSS series. The hist segment is
    guaranteed break-free: we measure each increment's standard deviation
    on the hist segment and standardize by it. A series whose values are
    autocorrelated (so its CUSUM wanders more under H0) is calibrated by
    its own hist behavior instead of polluting the ranking.

FEATURES per aspect a (increment g_a, hist-estimated sd s_a):
    prefix_t = C_t / (s_a * sqrt(n))                 standardized prefix mean
    up_t     = (C_t - min_{s<=t} C_s) / (s_a * sqrt(n))
    down_t   = (max_{s<=t} C_s - C_t) / (s_a * sqrt(n))
    for w in WINDOWS: win_w = (C_t - C_{t-w}) / (s_a * sqrt(w_eff))
Plus context: log1p(n), log(n_hist), and hist descriptors (sd of z incr
aspects is 1 by construction; we expose hist lag-1 autocorr, kurtosis).

Both an offline vectorized path and an exactly-matching streaming path are
provided, unit-tested for equality in __main__.
"""

from __future__ import annotations

import numpy as np

WINDOWS = (8, 16, 32, 64, 128, 256, 512)

ASPECTS = (
    "z",        # location
    "z2",       # scale (z^2 - 1)
    "absz",     # robust scale (|z| - E0)
    "L1",       # Legendre P1 of u: 2u-1               location (rank)
    "L2",       # Legendre P2: 6u^2-6u+1               dispersion (rank)
    "L3",       # Legendre P3: 20u^3-30u^2+12u-1       skew (rank)
    "L4",       # Legendre P4: 70u^4-140u^3+90u^2-20u+1  kurtosis (rank)
    "tail_up",  # 1[u>0.95] - E0
    "tail_dn",  # 1[u<0.05] - E0
    "ac1",      # z_t z_{t-1} - r1_0
    "ac2",      # z_t z_{t-2} - r2_0
    "ac3",      # z_t z_{t-3} - r3_0
    "vc1",      # |z_t||z_{t-1}| - m1_0   volatility clustering
    "sgn1",     # 1[z_t>0 == z_{t-1}>0] - p1_0  robust dependence
    "rk1",      # (u_t-.5)(u_{t-1}-.5) - q1_0   rank autocorrelation
)

_LAGS = {"ac1": 1, "ac2": 2, "ac3": 3, "vc1": 1, "sgn1": 1, "rk1": 1}

N_CONTEXT = 4       # log_n, log_hist, hist_r1, hist_kurt


def feature_names():
    names = []
    for a in ASPECTS:
        names.append(f"{a}_prefix")
        names.append(f"{a}_up")
        names.append(f"{a}_down")
        for w in WINDOWS:
            names.append(f"{a}_w{w}")
    names += ["log_n", "log_hist", "hist_r1", "hist_kurt"]
    return names


N_FEATURES = len(ASPECTS) * (3 + len(WINDOWS)) + N_CONTEXT


class Hist2:
    """Hist-segment statistics: baselines and per-aspect H0 sds."""

    def __init__(self, x_hist):
        h = np.asarray(x_hist, dtype=np.float64)
        self.n_hist = h.size
        self.mu0 = float(h.mean())
        sd = float(h.std())
        self.sd0 = sd if sd > 1e-12 else 1e-12
        self.sorted_hist = np.sort(h)
        zh = (h - self.mu0) / self.sd0
        uh = self._cdf(h)

        self.abs_mean = float(np.abs(zh).mean())
        self.tail_up0 = float((uh > 0.95).mean())
        self.tail_dn0 = float((uh < 0.05).mean())
        self.r1_0 = _lagmean(zh * 1.0, zh, 1)
        self.r2_0 = _lagmean(zh, zh, 2)
        self.r3_0 = _lagmean(zh, zh, 3)
        self.m1_0 = _lagmean(np.abs(zh), np.abs(zh), 1)
        self.p1_0 = _lagmean((zh > 0).astype(np.float64) * 2 - 1,
                             (zh > 0).astype(np.float64) * 2 - 1, 1)
        self.p1_0 = (self.p1_0 + 1) / 2       # back to agreement probability
        self.q1_0 = _lagmean(uh - 0.5, uh - 0.5, 1)
        self.hist_r1 = self.r1_0
        m2 = float((zh ** 2).mean())
        m4 = float((zh ** 4).mean())
        self.hist_kurt = m4 / (m2 * m2 + 1e-12)
        self.log_hist = float(np.log(max(self.n_hist, 1)))

        # H0 sd of each aspect increment, measured on the hist segment
        inc_h = _aspect_increments(self, zh, uh, warm=True)
        self.s = {}
        for a in ASPECTS:
            v = float(np.std(inc_h[a]))
            self.s[a] = v if v > 1e-9 else 1e-9

    def _cdf(self, x):
        x = np.asarray(x, dtype=np.float64)
        lo = np.searchsorted(self.sorted_hist, x, side="left")
        hi = np.searchsorted(self.sorted_hist, x, side="right")
        return (lo + hi) / (2.0 * self.n_hist)


def _lagmean(a, b, k):
    if len(a) <= k:
        return 0.0
    return float((a[k:] * b[:-k]).mean())


def _aspect_increments(hs, z, u, warm=False, z_prev=None, u_prev=None):
    """Increment arrays for every aspect given z, u sequences.

    warm=True (hist calibration): lagged products use the true lagged value
    and drop nothing (arrays are shortened consistently is avoided -- we pad
    with the aspect's H0 mean so sd estimates are unaffected materially).
    For the online path, the first `lag` points of a lagged aspect use 0.0
    increments (no reliable lag value yet), matching the streaming version.
    """
    T = z.size
    out = {}
    out["z"] = z.copy()
    out["z2"] = z * z - 1.0
    out["absz"] = np.abs(z) - hs.abs_mean
    out["L1"] = 2 * u - 1
    out["L2"] = 6 * u * u - 6 * u + 1
    out["L3"] = 20 * u ** 3 - 30 * u * u + 12 * u - 1
    out["L4"] = 70 * u ** 4 - 140 * u ** 3 + 90 * u * u - 20 * u + 1
    out["tail_up"] = (u > 0.95).astype(np.float64) - hs.tail_up0
    out["tail_dn"] = (u < 0.05).astype(np.float64) - hs.tail_dn0

    def lagged(vals, lag, base):
        prev = np.empty(T)
        if z_prev is not None and not warm:
            # online: seed from the tail of hist? No -- keep it simple and
            # identical to streaming: zero increments for the first `lag`.
            pass
        r = np.zeros(T)
        if T > lag:
            r[lag:] = vals[lag:] * vals[:-lag] if base is None else base
        return r

    zp = z
    for name, lag in (("ac1", 1), ("ac2", 2), ("ac3", 3)):
        r = np.zeros(T)
        if T > lag:
            r[lag:] = z[lag:] * z[:-lag] - getattr(hs, f"r{lag}_0")
        out[name] = r
    r = np.zeros(T)
    if T > 1:
        r[1:] = np.abs(z[1:]) * np.abs(z[:-1]) - hs.m1_0
    out["vc1"] = r
    r = np.zeros(T)
    if T > 1:
        agree = ((z[1:] > 0) == (z[:-1] > 0)).astype(np.float64)
        r[1:] = agree - hs.p1_0
    out["sgn1"] = r
    r = np.zeros(T)
    if T > 1:
        r[1:] = (u[1:] - 0.5) * (u[:-1] - 0.5) - hs.q1_0
    out["rk1"] = r
    return out


def compute_features_offline(x_hist, x_online):
    hs = Hist2(x_hist)
    x = np.asarray(x_online, dtype=np.float64)
    z = (x - hs.mu0) / hs.sd0
    u = hs._cdf(x)
    inc = _aspect_increments(hs, z, u)

    T = z.size
    n = np.arange(1, T + 1, dtype=np.float64)
    sqrt_n = np.sqrt(n)

    cols = []
    for a in ASPECTS:
        g = inc[a]
        s = hs.s[a]
        C = np.cumsum(g)
        Cpad = np.concatenate([[0.0], C])
        run_min = np.minimum.accumulate(np.minimum(C, 0.0))
        run_max = np.maximum.accumulate(np.maximum(C, 0.0))
        cols.append(C / (s * sqrt_n))
        cols.append((C - run_min) / (s * sqrt_n))
        cols.append((run_max - C) / (s * sqrt_n))
        for w in WINDOWS:
            w_eff = np.minimum(n, w)
            start = np.maximum(0, np.arange(1, T + 1) - w)
            m = (Cpad[1:] - Cpad[start])
            cols.append(m / (s * np.sqrt(w_eff)))

    cols.append(np.log1p(n))
    cols.append(np.full(T, hs.log_hist))
    cols.append(np.full(T, hs.hist_r1))
    cols.append(np.full(T, hs.hist_kurt))
    return np.column_stack(cols)


class StreamDetector2:
    """Streaming twin of compute_features_offline. One instance per series."""

    def __init__(self, x_hist):
        hs = Hist2(x_hist)
        self.hs = hs
        self.t = 0
        self.z_hist = []                    # last 3 z values
        self.u_prev = 0.5
        self.C = {a: 0.0 for a in ASPECTS}
        self.Cmin = {a: 0.0 for a in ASPECTS}
        self.Cmax = {a: 0.0 for a in ASPECTS}
        wmax = max(WINDOWS)
        self._ring = {a: np.zeros(wmax) for a in ASPECTS}
        self._wsum = {a: {w: 0.0 for w in WINDOWS} for a in ASPECTS}

    def step(self, x):
        hs = self.hs
        x = float(x)
        z = (x - hs.mu0) / hs.sd0
        lo = np.searchsorted(hs.sorted_hist, x, side="left")
        hi = np.searchsorted(hs.sorted_hist, x, side="right")
        u = (lo + hi) / (2.0 * hs.n_hist)

        zh = self.z_hist
        t = self.t
        inc = {
            "z": z,
            "z2": z * z - 1.0,
            "absz": abs(z) - hs.abs_mean,
            "L1": 2 * u - 1,
            "L2": 6 * u * u - 6 * u + 1,
            "L3": 20 * u ** 3 - 30 * u * u + 12 * u - 1,
            "L4": 70 * u ** 4 - 140 * u ** 3 + 90 * u * u - 20 * u + 1,
            "tail_up": (1.0 if u > 0.95 else 0.0) - hs.tail_up0,
            "tail_dn": (1.0 if u < 0.05 else 0.0) - hs.tail_dn0,
            "ac1": (z * zh[-1] - hs.r1_0) if t >= 1 else 0.0,
            "ac2": (z * zh[-2] - hs.r2_0) if t >= 2 else 0.0,
            "ac3": (z * zh[-3] - hs.r3_0) if t >= 3 else 0.0,
            "vc1": (abs(z) * abs(zh[-1]) - hs.m1_0) if t >= 1 else 0.0,
            "sgn1": (((z > 0) == (zh[-1] > 0)) - hs.p1_0) if t >= 1 else 0.0,
            "rk1": ((u - 0.5) * (self.u_prev - 0.5) - hs.q1_0) if t >= 1 else 0.0,
        }

        n = t + 1
        sqrt_n = np.sqrt(float(n))
        out = np.empty(N_FEATURES)
        i = 0
        wmax = max(WINDOWS)
        pos = t % wmax
        for a in ASPECTS:
            g = inc[a]
            s = self.hs.s[a]
            C = self.C[a] + g
            self.C[a] = C
            self.Cmin[a] = min(self.Cmin[a], min(C, 0.0))
            self.Cmax[a] = max(self.Cmax[a], max(C, 0.0))
            out[i] = C / (s * sqrt_n)
            out[i + 1] = (C - self.Cmin[a]) / (s * sqrt_n)
            out[i + 2] = (self.Cmax[a] - C) / (s * sqrt_n)
            i += 3
            ring = self._ring[a]
            for w in WINDOWS:
                self._wsum[a][w] += g
                drop = t - w
                if drop >= 0:
                    self._wsum[a][w] -= ring[drop % wmax]
                w_eff = min(n, w)
                out[i] = self._wsum[a][w] / (s * np.sqrt(float(w_eff)))
                i += 1
            ring[pos] = g

        out[i] = np.log1p(float(n))
        out[i + 1] = hs.log_hist
        out[i + 2] = hs.hist_r1
        out[i + 3] = hs.hist_kurt

        self.z_hist.append(z)
        if len(self.z_hist) > 3:
            self.z_hist.pop(0)
        self.u_prev = u
        self.t = n
        return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for trial in range(8):
        n_hist = int(rng.integers(50, 400))
        T = int(rng.integers(5, 700))
        x_hist = rng.normal(size=n_hist)
        x_online = rng.normal(size=T)
        if trial % 2 == 0 and T > 10:
            tau = int(rng.integers(1, T - 1))
            if trial % 4 == 0:
                x_online[tau:] += 1.0
            else:
                x_online[tau:] *= 1.8
        F_off = compute_features_offline(x_hist, x_online)
        det = StreamDetector2(x_hist)
        F_str = np.vstack([det.step(v) for v in x_online])
        assert F_off.shape == (T, N_FEATURES), (F_off.shape, N_FEATURES)
        err = np.max(np.abs(F_off - F_str))
        assert err < 1e-9, (trial, err)
    assert len(feature_names()) == N_FEATURES
    print(f"OK: stream == offline; {N_FEATURES} features")
