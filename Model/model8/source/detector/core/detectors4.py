"""
detectors4.py -- detectors3 + multi-scale difference energies + extreme
tails.

New aspects vs detectors3 (which had 23):

*  d1e, d2e, d4e, d8e : (z_t - z_{t-k})^2 - E0 for k = 1, 2, 4, 8.
   Squared multi-scale differences are streaming band-power proxies -- the
   FFT/bandpower features of the 2025 winners, made incremental. The lag
   context crosses the hist/online boundary (hist tail), so they are
   defined from the first online step; E0 is measured on the hist segment.
*  absd1 : |z_t - z_{t-1}| - E0, a robust version of d1e.
*  x99_up, x99_dn : 1[u > 0.99] - E0, 1[u < 0.01] - E0, extreme tails.

30 aspects x (prefix, up, down + 7 windows) + 6 context = 306 features.
"""

from __future__ import annotations

import numpy as np

WINDOWS = (8, 16, 32, 64, 128, 256, 512)

ASPECTS = (
    "z", "z2", "absz",
    "L1", "L2", "L3", "L4",
    "tail_up", "tail_dn", "x99_up", "x99_dn",
    "q25", "q50", "q75",
    "ac1", "ac2", "ac3", "ac5",
    "vc1", "sgn1", "rk1",
    "e", "e2", "abse", "eac1",
    "d1e", "d2e", "d4e", "d8e", "absd1",
)

AR_P = 5
DIFF_LAGS = (1, 2, 4, 8)
MAX_LAG = 8
N_CONTEXT = 6


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
    if not np.all(np.isfinite(phi)) or np.abs(phi).sum() > 2.0:
        phi = np.zeros(p)
    return phi


def _lagmean(a, b, k):
    if len(a) <= k:
        return 0.0
    return float((a[k:] * b[:-k]).mean())


def _ar_residuals(z, phi, z_context):
    p = phi.size
    zfull = np.concatenate([z_context[-p:], z])
    e = z.copy().astype(np.float64)
    for k in range(1, p + 1):
        e -= phi[k - 1] * zfull[p - k:p - k + z.size]
    return e


def _diffs(z, k, z_context):
    """z_t - z_{t-k}, with hist-tail context so it's defined from t=0."""
    zfull = np.concatenate([z_context[-k:], z])
    return z - zfull[:z.size]


class Hist4:
    def __init__(self, x_hist):
        h = np.asarray(x_hist, dtype=np.float64)
        self.n_hist = h.size
        self.mu0 = float(h.mean())
        sd = float(h.std())
        self.sd0 = sd if sd > 1e-12 else 1e-12
        self.sorted_hist = np.sort(h)
        zh = (h - self.mu0) / self.sd0
        self.zh_tail = zh[-MAX_LAG:] if zh.size >= MAX_LAG else \
            np.concatenate([np.zeros(MAX_LAG - zh.size), zh])
        uh = self._cdf(h)

        self.abs_mean = float(np.abs(zh).mean())
        self.tail_up0 = float((uh > 0.95).mean())
        self.tail_dn0 = float((uh < 0.05).mean())
        self.x99_up0 = float((uh > 0.99).mean())
        self.x99_dn0 = float((uh < 0.01).mean())
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

        self.phi = _yule_walker(zh)
        eh = _ar_residuals(zh, self.phi, np.zeros(AR_P))
        eh = eh[AR_P:] if eh.size > AR_P else eh
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

        # hist means of the diff energies
        self.dke0 = {}
        for k in DIFF_LAGS:
            if zh.size > k:
                dk = zh[k:] - zh[:-k]
                self.dke0[k] = float((dk ** 2).mean())
            else:
                self.dke0[k] = 2.0
        if zh.size > 1:
            self.absd1_0 = float(np.abs(zh[1:] - zh[:-1]).mean())
        else:
            self.absd1_0 = 1.0

        inc_h = _aspect_increments_hist(self, zh, uh, eh)
        self.s = {}
        for a in ASPECTS:
            v = float(np.std(inc_h[a]))
            self.s[a] = v if v > 1e-9 else 1e-9

    def _cdf(self, x):
        x = np.asarray(x, dtype=np.float64)
        lo = np.searchsorted(self.sorted_hist, x, side="left")
        hi = np.searchsorted(self.sorted_hist, x, side="right")
        return (lo + hi) / (2.0 * self.n_hist)


def _common_increments(hs, z, u, out):
    T = z.size
    out["z"] = z.copy()
    out["z2"] = z * z - 1.0
    out["absz"] = np.abs(z) - hs.abs_mean
    out["L1"] = 2 * u - 1
    out["L2"] = 6 * u * u - 6 * u + 1
    out["L3"] = 20 * u ** 3 - 30 * u * u + 12 * u - 1
    out["L4"] = 70 * u ** 4 - 140 * u ** 3 + 90 * u * u - 20 * u + 1
    out["tail_up"] = (u > 0.95).astype(np.float64) - hs.tail_up0
    out["tail_dn"] = (u < 0.05).astype(np.float64) - hs.tail_dn0
    out["x99_up"] = (u > 0.99).astype(np.float64) - hs.x99_up0
    out["x99_dn"] = (u < 0.01).astype(np.float64) - hs.x99_dn0
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
    return out


def _aspect_increments_hist(hs, zh, uh, eh):
    """Hist-side increments used ONLY for calibration sds."""
    out = {}
    _common_increments(hs, zh, uh, out)
    out["e"] = eh - eh.mean()
    out["e2"] = eh ** 2 - (eh ** 2).mean()
    out["abse"] = np.abs(eh) - np.abs(eh).mean()
    r = np.zeros(eh.size)
    if eh.size > 1:
        r[1:] = eh[1:] * eh[:-1] - hs.eac1_0
    out["eac1"] = r
    for k in DIFF_LAGS:
        r = np.zeros(zh.size)
        if zh.size > k:
            r[k:] = (zh[k:] - zh[:-k]) ** 2 - hs.dke0[k]
        out[f"d{k}e"] = r
    r = np.zeros(zh.size)
    if zh.size > 1:
        r[1:] = np.abs(zh[1:] - zh[:-1]) - hs.absd1_0
    out["absd1"] = r
    return out


def _aspect_increments_online(hs, z, u, e):
    out = {}
    _common_increments(hs, z, u, out)
    out["e"] = e - hs.e_mean
    out["e2"] = e ** 2 - hs.e2_mean
    out["abse"] = np.abs(e) - hs.abse_mean
    r = np.zeros(e.size)
    if e.size > 1:
        r[1:] = e[1:] * e[:-1] - hs.eac1_0
    out["eac1"] = r
    for k in DIFF_LAGS:
        dk = _diffs(z, k, hs.zh_tail)
        out[f"d{k}e"] = dk ** 2 - hs.dke0[k]
    d1 = _diffs(z, 1, hs.zh_tail)
    out["absd1"] = np.abs(d1) - hs.absd1_0
    return out


def compute_features_offline(x_hist, x_online):
    hs = Hist4(x_hist)
    x = np.asarray(x_online, dtype=np.float64)
    z = (x - hs.mu0) / hs.sd0
    u = hs._cdf(x)
    e = _ar_residuals(z, hs.phi, hs.zh_tail[-AR_P:])
    inc = _aspect_increments_online(hs, z, u, e)

    T = z.size
    n = np.arange(1, T + 1, dtype=np.float64)
    sqrt_n = np.sqrt(n)
    idx = np.arange(1, T + 1)
    start_by_w = {w: np.maximum(0, idx - w) for w in WINDOWS}
    w_eff_by_w = {w: np.minimum(n, w) for w in WINDOWS}

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
            m = (Cpad[1:] - Cpad[start_by_w[w]])
            cols.append(m / (s * np.sqrt(w_eff_by_w[w])))

    cols.append(np.log1p(n))
    cols.append(np.full(T, hs.log_hist))
    cols.append(np.full(T, hs.hist_r1))
    cols.append(np.full(T, hs.hist_kurt))
    cols.append(np.full(T, hs.ar_r2))
    cols.append(np.full(T, hs.resid_sd))
    return np.column_stack(cols)


class StreamDetector4:
    def __init__(self, x_hist):
        hs = Hist4(x_hist)
        self.hs = hs
        self.t = 0
        # z_lags[0] = z_{t-1}, ..., up to MAX_LAG; seeded with hist tail
        self.z_lags = list(hs.zh_tail[::-1])
        self.z_online = []          # online-only lags (ac/vc/sgn aspects)
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

        e = z
        for k in range(AR_P):
            e -= hs.phi[k] * self.z_lags[k]

        t = self.t
        zo = self.z_online
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
            "x99_up": (1.0 if u > 0.99 else 0.0) - hs.x99_up0,
            "x99_dn": (1.0 if u < 0.01 else 0.0) - hs.x99_dn0,
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
        for k in DIFF_LAGS:
            d = z - self.z_lags[k - 1]
            inc[f"d{k}e"] = d * d - hs.dke0[k]
        d1 = z - self.z_lags[0]
        inc["absd1"] = abs(d1) - hs.absd1_0

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

        self.z_lags = [z] + self.z_lags[:MAX_LAG - 1]
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
        det = StreamDetector4(x_hist)
        F_str = np.vstack([det.step(v) for v in x_online])
        assert F_off.shape == (T, N_FEATURES), (F_off.shape, N_FEATURES)
        err = np.max(np.abs(F_off - F_str))
        assert err < 1e-9, (trial, err)
    assert len(feature_names()) == N_FEATURES

    det = StreamDetector4(rng.normal(size=3000))
    xs = rng.normal(size=1000)
    t0 = time.perf_counter()
    for v in xs:
        det.step(v)
    dt = (time.perf_counter() - t0) / 1000
    print(f"OK: stream == offline; {N_FEATURES} features; "
          f"{dt * 1e6:.0f} us/step")
