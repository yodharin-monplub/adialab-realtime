"""
detectors3.py -- detectors2 + AR-whitened residual aspects + quantile
crossings + longer dependence lags.

New vs detectors2:

*  AR(5) fit on the hist z-series (regularized Yule-Walker). Online
   residuals e_t = z_t - sum(phi_k z_{t-k}) turn DEPENDENCE breaks into
   visible mean/variance breaks of e. Aspects: e, e^2, |e|, e_t*e_{t-1}.
   The first 5 online steps use hist-tail values as lag context so the
   residual is defined from step 0.
*  Quantile-crossing aspects: 1[u > q] - (1-q) for q in {0.25, 0.5, 0.75}
   (threshold-crossing rates were top features in the 2025 challenge).
*  ac5: lag-5 autocorrelation aspect.
*  Context adds hist AR R^2 and residual sd.

Same feature layout per aspect as detectors2 (prefix, up, down + geometric
windows), same per-series H0 calibration, offline == streaming exactly.
"""

from __future__ import annotations

import numpy as np

WINDOWS = (8, 16, 32, 64, 128, 256, 512)

ASPECTS = (
    "z", "z2", "absz",
    "L1", "L2", "L3", "L4",
    "tail_up", "tail_dn",
    "q25", "q50", "q75",
    "ac1", "ac2", "ac3", "ac5",
    "vc1", "sgn1", "rk1",
    "e", "e2", "abse", "eac1",
)

AR_P = 5
N_CONTEXT = 6      # log_n, log_hist, hist_r1, hist_kurt, ar_r2, resid_sd


def feature_names():
    names = []
    for a in ASPECTS:
        names.append(f"{a}_prefix")
        names.append(f"{a}_up")
        names.append(f"{a}_down")
        for w in WINDOWS:
            names.append(f"{a}_w{w}")
    names += ["log_n", "log_hist", "hist_r1", "hist_kurt", "ar_r2", "resid_sd"]
    return names


N_FEATURES = len(ASPECTS) * (3 + len(WINDOWS)) + N_CONTEXT


def _yule_walker(z, p=AR_P, ridge=1e-3):
    """AR(p) coefficients from the autocovariance of z, ridge-regularized."""
    n = z.size
    if n < 10 * p:
        return np.zeros(p)
    zc = z - z.mean()
    acov = np.array([float((zc[k:] * zc[:-k] if k else zc * zc).mean())
                     for k in range(p + 1)])
    R = np.empty((p, p))
    for i in range(p):
        for j in range(p):
            R[i, j] = acov[abs(i - j)]
    R += np.eye(p) * ridge * acov[0]
    try:
        phi = np.linalg.solve(R, acov[1:])
    except np.linalg.LinAlgError:
        phi = np.zeros(p)
    # clamp wild fits (near-unit-root hists can explode the solve)
    if not np.all(np.isfinite(phi)) or np.abs(phi).sum() > 2.0:
        phi = np.zeros(p)
    return phi


class Hist3:
    def __init__(self, x_hist):
        h = np.asarray(x_hist, dtype=np.float64)
        self.n_hist = h.size
        self.mu0 = float(h.mean())
        sd = float(h.std())
        self.sd0 = sd if sd > 1e-12 else 1e-12
        self.sorted_hist = np.sort(h)
        zh = (h - self.mu0) / self.sd0
        self.zh_tail = zh[-AR_P:] if zh.size >= AR_P else \
            np.concatenate([np.zeros(AR_P - zh.size), zh])
        uh = self._cdf(h)

        self.abs_mean = float(np.abs(zh).mean())
        self.tail_up0 = float((uh > 0.95).mean())
        self.tail_dn0 = float((uh < 0.05).mean())
        self.q25_0 = float((uh > 0.25).mean())
        self.q50_0 = float((uh > 0.50).mean())
        self.q75_0 = float((uh > 0.75).mean())
        self.r1_0 = _lagmean(zh, zh, 1)
        self.r2_0 = _lagmean(zh, zh, 2)
        self.r3_0 = _lagmean(zh, zh, 3)
        self.r5_0 = _lagmean(zh, zh, 5)
        self.m1_0 = _lagmean(np.abs(zh), np.abs(zh), 1)
        p1 = _lagmean((zh > 0).astype(np.float64) * 2 - 1,
                      (zh > 0).astype(np.float64) * 2 - 1, 1)
        self.p1_0 = (p1 + 1) / 2
        self.q1_0 = _lagmean(uh - 0.5, uh - 0.5, 1)
        self.hist_r1 = self.r1_0
        m2 = float((zh ** 2).mean())
        m4 = float((zh ** 4).mean())
        self.hist_kurt = m4 / (m2 * m2 + 1e-12)
        self.log_hist = float(np.log(max(self.n_hist, 1)))

        # ---- AR(5) on hist z ----
        self.phi = _yule_walker(zh)
        eh = _ar_residuals(zh, self.phi, np.zeros(AR_P))
        eh = eh[AR_P:] if eh.size > AR_P else eh   # drop warmup-contaminated
        if eh.size < 10:
            eh = zh
        self.e_mean = float(eh.mean())
        ve = float(eh.var())
        self.e_var = ve if ve > 1e-12 else 1e-12
        self.resid_sd = float(np.sqrt(self.e_var))
        self.abse_mean = float(np.abs(eh).mean())
        self.e2_mean = float((eh ** 2).mean())
        self.eac1_0 = _lagmean(eh, eh, 1)
        self.ar_r2 = max(0.0, 1.0 - self.e_var / max(m2, 1e-12))

        inc_h = _aspect_increments(self, zh, uh, eh_pair=(eh, None))
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


def _ar_residuals(z, phi, z_context):
    """e_t = z_t - sum_k phi_k * z_{t-k}; z_context supplies the AR_P values
    preceding z[0] (most recent last)."""
    p = phi.size
    zfull = np.concatenate([z_context[-p:], z])
    e = z.copy().astype(np.float64)
    for k in range(1, p + 1):
        e -= phi[k - 1] * zfull[p - k:p - k + z.size]
    return e


def _aspect_increments(hs, z, u, eh_pair=None, e=None):
    """Increment arrays for all aspects. For hist calibration pass
    eh_pair=(eh, None) so residual aspects use hist residuals."""
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
    out["q25"] = (u > 0.25).astype(np.float64) - hs.q25_0
    out["q50"] = (u > 0.50).astype(np.float64) - hs.q50_0
    out["q75"] = (u > 0.75).astype(np.float64) - hs.q75_0

    for name, lag in (("ac1", 1), ("ac2", 2), ("ac3", 3), ("ac5", 5)):
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

    # ---- AR residual aspects ----
    if eh_pair is not None:
        e_arr = eh_pair[0]
    else:
        e_arr = e
    out["e"] = e_arr - hs.e_mean if eh_pair is None else e_arr - e_arr.mean()
    out["e2"] = e_arr ** 2 - hs.e2_mean if eh_pair is None else \
        e_arr ** 2 - (e_arr ** 2).mean()
    out["abse"] = np.abs(e_arr) - hs.abse_mean if eh_pair is None else \
        np.abs(e_arr) - np.abs(e_arr).mean()
    r = np.zeros(e_arr.size)
    if e_arr.size > 1:
        r[1:] = e_arr[1:] * e_arr[:-1] - hs.eac1_0
    out["eac1"] = r
    return out


def compute_features_offline(x_hist, x_online):
    hs = Hist3(x_hist)
    x = np.asarray(x_online, dtype=np.float64)
    z = (x - hs.mu0) / hs.sd0
    u = hs._cdf(x)
    e = _ar_residuals(z, hs.phi, hs.zh_tail)
    inc = _aspect_increments(hs, z, u, e=e)

    T = z.size
    n = np.arange(1, T + 1, dtype=np.float64)
    sqrt_n = np.sqrt(n)

    cols = []
    idx = np.arange(1, T + 1)
    start_by_w = {w: np.maximum(0, idx - w) for w in WINDOWS}
    w_eff_by_w = {w: np.minimum(n, w) for w in WINDOWS}
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
            m = (Cpad[1:] - Cpad[start_by_w[w]])
            cols.append(m / (s * np.sqrt(w_eff_by_w[w])))

    cols.append(np.log1p(n))
    cols.append(np.full(T, hs.log_hist))
    cols.append(np.full(T, hs.hist_r1))
    cols.append(np.full(T, hs.hist_kurt))
    cols.append(np.full(T, hs.ar_r2))
    cols.append(np.full(T, hs.resid_sd))
    return np.column_stack(cols)


class StreamDetector3:
    def __init__(self, x_hist):
        hs = Hist3(x_hist)
        self.hs = hs
        self.t = 0
        self.z_lags = list(hs.zh_tail[::-1])   # z_lags[0] = z_{t-1}
        self.z_lags_own = []                   # online-only lags for ac feats
        self.u_prev = 0.5
        self.e_prev = 0.0
        self.C = {a: 0.0 for a in ASPECTS}
        self.Cmin = {a: 0.0 for a in ASPECTS}
        self.Cmax = {a: 0.0 for a in ASPECTS}
        wmax = max(WINDOWS)
        self._wmax = wmax
        self._ring = {a: np.zeros(wmax) for a in ASPECTS}
        self._wsum = {a: {w: 0.0 for w in WINDOWS} for a in ASPECTS}

    def step(self, x):
        hs = self.hs
        x = float(x)
        z = (x - hs.mu0) / hs.sd0
        lo = np.searchsorted(hs.sorted_hist, x, side="left")
        hi = np.searchsorted(hs.sorted_hist, x, side="right")
        u = (lo + hi) / (2.0 * hs.n_hist)

        # AR residual, using hist tail as pre-online context
        e = z
        for k in range(AR_P):
            e -= hs.phi[k] * self.z_lags[k]

        t = self.t
        zo = self.z_lags_own
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
            "q25": (1.0 if u > 0.25 else 0.0) - hs.q25_0,
            "q50": (1.0 if u > 0.50 else 0.0) - hs.q50_0,
            "q75": (1.0 if u > 0.75 else 0.0) - hs.q75_0,
            "ac1": (z * zo[-1] - hs.r1_0) if t >= 1 else 0.0,
            "ac2": (z * zo[-2] - hs.r2_0) if t >= 2 else 0.0,
            "ac3": (z * zo[-3] - hs.r3_0) if t >= 3 else 0.0,
            "ac5": (z * zo[-5] - hs.r5_0) if t >= 5 else 0.0,
            "vc1": (abs(z) * abs(zo[-1]) - hs.m1_0) if t >= 1 else 0.0,
            "sgn1": (((z > 0) == (zo[-1] > 0)) - hs.p1_0) if t >= 1 else 0.0,
            "rk1": ((u - 0.5) * (self.u_prev - 0.5) - hs.q1_0) if t >= 1 else 0.0,
            "e": e - hs.e_mean,
            "e2": e * e - hs.e2_mean,
            "abse": abs(e) - hs.abse_mean,
            "eac1": (e * self.e_prev - hs.eac1_0) if t >= 1 else 0.0,
        }

        n = t + 1
        sqrt_n = np.sqrt(float(n))
        out = np.empty(N_FEATURES)
        i = 0
        wmax = self._wmax
        pos = t % wmax
        for a in ASPECTS:
            g = inc[a]
            s = hs.s[a]
            C = self.C[a] + g
            self.C[a] = C
            self.Cmin[a] = min(self.Cmin[a], min(C, 0.0))
            self.Cmax[a] = max(self.Cmax[a], max(C, 0.0))
            out[i] = C / (s * sqrt_n)
            out[i + 1] = (C - self.Cmin[a]) / (s * sqrt_n)
            out[i + 2] = (self.Cmax[a] - C) / (s * sqrt_n)
            i += 3
            ring = self._ring[a]
            ws = self._wsum[a]
            for w in WINDOWS:
                ws[w] += g
                drop = t - w
                if drop >= 0:
                    ws[w] -= ring[drop % wmax]
                w_eff = min(n, w)
                out[i] = ws[w] / (s * np.sqrt(float(w_eff)))
                i += 1
            ring[pos] = g

        out[i] = np.log1p(float(n))
        out[i + 1] = hs.log_hist
        out[i + 2] = hs.hist_r1
        out[i + 3] = hs.hist_kurt
        out[i + 4] = hs.ar_r2
        out[i + 5] = hs.resid_sd

        # roll lag state
        self.z_lags = [z] + self.z_lags[:AR_P - 1]
        zo.append(z)
        if len(zo) > 5:
            zo.pop(0)
        self.u_prev = u
        self.e_prev = e
        self.t = n
        return out


if __name__ == "__main__":
    import time
    rng = np.random.default_rng(0)
    for trial in range(10):
        n_hist = int(rng.integers(60, 500))
        T = int(rng.integers(5, 800))
        # AR(1)-correlated hist so phi is non-trivial
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
        det = StreamDetector3(x_hist)
        F_str = np.vstack([det.step(v) for v in x_online])
        assert F_off.shape == (T, N_FEATURES), (F_off.shape, N_FEATURES)
        err = np.max(np.abs(F_off - F_str))
        assert err < 1e-9, (trial, err)
    assert len(feature_names()) == N_FEATURES

    det = StreamDetector3(rng.normal(size=3000))
    xs = rng.normal(size=1000)
    t0 = time.perf_counter()
    for v in xs:
        det.step(v)
    dt = (time.perf_counter() - t0) / 1000
    print(f"OK: stream == offline; {N_FEATURES} features; "
          f"{dt * 1e6:.0f} us/step")
