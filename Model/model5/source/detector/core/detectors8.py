"""
detectors8.py -- detectors3 + RICH ROBUST STATISTICS ON SUFFIX WINDOWS.

THE LESSON THAT PRODUCED THIS FILE
----------------------------------
detectors7 ported the 2025 winners' "boundary-local" block: rich robust
statistics (median / MAD / mean / SD / slope jumps) comparing the FIRST w
online points against the hist tail. On the 2025 proxy task -- where the
break sits exactly at the boundary -- that lifted ROC from 0.625 to 0.698,
nearly matching their own bank. On the REAL task it *dropped* TS-AUC from
0.6036 to 0.5797, and those blocks dominated feature importance: they
answer "did a break happen at the very start", which is the wrong question
here and crowds out the features that answer the right one.

The statistics were not the problem; the window placement was. Here the
same family is computed on SUFFIX windows -- the last w observations,
ending at the current step -- which is precisely the "has the regime
changed recently" question, scanned across w so an unknown break age is
covered by whichever window best isolates the post-break segment.

Per window w in WINDOWS_S, versus the hist reference:
    med_jump      binned median of the window minus hist median
    logmad_jump   log robust scale ratio (binned IQR-based)
    mean_jump     standardized mean shift, scaled by sqrt(w_eff)
    logsd_jump    log SD ratio
    slope_t       within-window OLS slope t-statistic
    ks / js       binned CDF distances to the hist distribution
    acf1_delta    within-window lag-1 autocorrelation minus hist
    cross_delta   within-window median-crossing rate minus hist
Plus, across the window grid: max and argmax (log2 w) of |mean_jump|,
logsd_jump, ks -- the scan summaries.

Base = detectors3's 236 columns, verbatim and first.
"""

from __future__ import annotations

import numpy as np

import detectors3 as d3
import detectors6 as d6

WINDOWS_S = (16, 32, 64, 128, 256, 512)
PER_W = ("med_jump", "logmad_jump", "mean_jump", "logsd_jump",
         "slope_t", "ks", "js", "acf1_delta", "cross_delta")
SCAN_OF = ("mean_jump", "logsd_jump", "ks")

EXTRA8 = (
    [f"s{w}_{k}" for w in WINDOWS_S for k in PER_W]
    + [f"scan_{k}_{s}" for k in SCAN_OF for s in ("max", "argw")]
)

N_FEATURES = d3.N_FEATURES + len(EXTRA8)


def feature_names():
    return d3.feature_names() + list(EXTRA8)


class Hist8:
    def __init__(self, x_hist):
        h = np.asarray(x_hist, dtype=np.float64)
        self.hr = d6.HistRef(x_hist)          # bins, med0, scale0, reps
        hr = self.hr
        zr = hr.zr(h)
        self.med0 = float(np.median(zr))
        q75 = float(np.quantile(zr, 0.75))
        q25 = float(np.quantile(zr, 0.25))
        self.iqr0 = max(q75 - q25, 1e-9)
        self.mean0 = float(zr.mean())
        sd = float(zr.std())
        self.sd0 = sd if sd > 1e-9 else 1e-9
        var = max(float(((zr - self.mean0) ** 2).mean()), 1e-9)
        self.acf1_0 = (float(((zr[1:] - self.mean0) * (zr[:-1] - self.mean0))
                             .mean()) / var) if zr.size > 1 else 0.0
        above = zr > self.med0
        self.cross0 = float((above[1:] != above[:-1]).mean()) \
            if zr.size > 1 else 0.5


def _binned_q(CDF, reps, q):
    idx = np.minimum((CDF < q).sum(axis=1), reps.size - 1)
    return reps[idx]


def compute_features_offline(x_hist, x_online):
    F3 = d3.compute_features_offline(x_hist, x_online)
    h8 = Hist8(x_hist)
    hr = h8.hr
    z = hr.zr(x_online)
    T = z.size
    n = np.arange(1, T + 1, dtype=np.float64)
    t_idx = np.arange(T, dtype=np.float64)

    # cumulative sums for windowed moments
    c_z = np.concatenate([[0.0], np.cumsum(z)])
    c_z2 = np.concatenate([[0.0], np.cumsum(z * z)])
    c_tz = np.concatenate([[0.0], np.cumsum(t_idx * z)])
    prod = np.zeros(T)
    if T > 1:
        prod[1:] = z[1:] * z[:-1]
    c_pr = np.concatenate([[0.0], np.cumsum(prod)])
    above = z > h8.med0
    cross = np.zeros(T)
    if T > 1:
        cross[1:] = (above[1:] != above[:-1]).astype(np.float64)
    c_cr = np.concatenate([[0.0], np.cumsum(cross)])

    # cumulative bin counts
    zb = np.searchsorted(hr.z_edges, z, side="right")
    Zoh = np.zeros((T, hr.nzb))
    Zoh[np.arange(T), zb] = 1.0
    Zcum = np.cumsum(Zoh, axis=0)

    E = np.zeros((T, len(EXTRA8)))
    col = 0
    scan_vals = {k: [] for k in SCAN_OF}
    for w in WINDOWS_S:
        start = np.maximum(0, np.arange(1, T + 1) - w)
        w_eff = np.minimum(n, w)

        s1 = c_z[1:] - c_z[start]
        s2 = c_z2[1:] - c_z2[start]
        stz = c_tz[1:] - c_tz[start]
        mean = s1 / w_eff
        var = np.maximum(s2 / w_eff - mean * mean, 1e-12)
        sd = np.sqrt(var)

        Wc = Zcum - np.vstack([np.zeros((min(w, T), hr.nzb)),
                               Zcum[:max(T - w, 0)]])
        WPMF = Wc / w_eff[:, None]
        WCDF = np.cumsum(WPMF, axis=1)

        med = _binned_q(WCDF, hr.z_reps, 0.5)
        q75 = _binned_q(WCDF, hr.z_reps, 0.75)
        q25 = _binned_q(WCDF, hr.z_reps, 0.25)

        d = WCDF - hr.z_hist_cdf[None, :]
        ks = np.max(np.abs(d), axis=1)
        m = 0.5 * (WPMF + hr.z_hist_pmf[None, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = np.where(WPMF > 0,
                          WPMF * (np.log(np.maximum(WPMF, 1e-12))
                                  - np.log(np.maximum(m, 1e-12))), 0.0)
            t2 = np.where(hr.z_hist_pmf[None, :] > 0,
                          hr.z_hist_pmf[None, :]
                          * (np.log(np.maximum(hr.z_hist_pmf[None, :], 1e-12))
                             - np.log(np.maximum(m, 1e-12))), 0.0)
        js = 0.5 * (t1.sum(axis=1) + t2.sum(axis=1))

        # within-window slope t-stat (local time origin)
        tbar = (w_eff - 1) / 2.0
        sxx = w_eff * (w_eff * w_eff - 1) / 12.0
        t0_ = np.maximum(np.arange(1, T + 1) - w_eff, 0)
        sxy = (stz - t0_ * s1) - tbar * s1
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = np.where(sxx > 0, sxy / np.maximum(sxx, 1e-12), 0.0)
            rv = np.maximum(s2 - s1 * s1 / w_eff - slope * sxy, 1e-12) / w_eff
            se = np.sqrt(rv / np.maximum(sxx, 1e-12)) + 1e-12
            slope_t = np.clip(slope / se, -50, 50)
        slope_t[w_eff < 4] = 0.0

        pr = c_pr[1:] - c_pr[start]
        mw = np.maximum(w_eff - 1, 1)
        acf1 = (pr / mw - mean * mean) / np.maximum(var, 1e-12)
        acf1 = np.clip(acf1, -2, 2) - h8.acf1_0
        acf1[w_eff < 4] = 0.0

        crs = (c_cr[1:] - c_cr[start]) / mw - h8.cross0
        crs[w_eff < 4] = 0.0

        mean_jump = (mean - h8.mean0) / h8.sd0 * np.sqrt(w_eff)
        logsd_jump = np.log(np.maximum(sd, 1e-9)) - np.log(h8.sd0)
        logmad_jump = np.log(np.maximum(q75 - q25, 1e-9)) - np.log(h8.iqr0)

        vals = {
            "med_jump": med - h8.med0,
            "logmad_jump": logmad_jump,
            "mean_jump": mean_jump,
            "logsd_jump": logsd_jump,
            "slope_t": slope_t,
            "ks": ks,
            "js": js,
            "acf1_delta": acf1,
            "cross_delta": crs,
        }
        for k in PER_W:
            E[:, col] = vals[k]
            col += 1
        for k in SCAN_OF:
            scan_vals[k].append(np.abs(vals[k]))

    logw = np.log2(np.asarray(WINDOWS_S, dtype=np.float64))
    for k in SCAN_OF:
        M = np.column_stack(scan_vals[k])
        E[:, col] = M.max(axis=1)
        E[:, col + 1] = logw[M.argmax(axis=1)]
        col += 2

    return np.column_stack([F3, E])


class StreamDetector8:
    """Streaming twin: ring buffers of bin counts and running sums per
    suffix window."""

    def __init__(self, x_hist, inner=None):
        self._inner = d3.StreamDetector3(x_hist) if inner is None else inner
        self.owns_inner = inner is None
        self.h8 = Hist8(x_hist)
        hr = self.h8.hr
        self.t = 0
        wmax = max(WINDOWS_S)
        self._wmax = wmax
        self._zring = np.zeros(wmax)
        self._bring = np.zeros(wmax, dtype=np.int64)
        self._prring = np.zeros(wmax)
        self._crring = np.zeros(wmax)
        self._cnt = {w: np.zeros(hr.nzb) for w in WINDOWS_S}
        self._s1 = {w: 0.0 for w in WINDOWS_S}
        self._s2 = {w: 0.0 for w in WINDOWS_S}
        self._stz = {w: 0.0 for w in WINDOWS_S}
        self._pr = {w: 0.0 for w in WINDOWS_S}
        self._cr = {w: 0.0 for w in WINDOWS_S}
        self._z_prev = None
        self._above_prev = None

    def _extra_from_current_inner(self, x):
        h8 = self.h8
        hr = h8.hr
        t = self.t
        n = t + 1
        z = (float(x) - hr.med0) / hr.scale0
        zb = int(np.searchsorted(hr.z_edges, z, side="right"))
        pr = 0.0 if self._z_prev is None else z * self._z_prev
        above = z > h8.med0
        cr = 0.0 if self._above_prev is None else \
            (1.0 if above != self._above_prev else 0.0)

        E = np.empty(len(EXTRA8), dtype=np.float64)
        col = 0
        wmax = self._wmax
        pos = t % wmax
        scan = {k: [] for k in SCAN_OF}

        for w in WINDOWS_S:
            drop = t - w
            self._cnt[w][zb] += 1
            self._s1[w] += z
            self._s2[w] += z * z
            self._stz[w] += t * z
            self._pr[w] += pr
            self._cr[w] += cr
            if drop >= 0:
                dpos = drop % wmax
                zo = self._zring[dpos]
                self._cnt[w][self._bring[dpos]] -= 1
                self._s1[w] -= zo
                self._s2[w] -= zo * zo
                self._stz[w] -= drop * zo
                self._pr[w] -= self._prring[dpos]
                self._cr[w] -= self._crring[dpos]
            w_eff = float(min(n, w))

            mean = self._s1[w] / w_eff
            var = max(self._s2[w] / w_eff - mean * mean, 1e-12)
            sd = np.sqrt(var)
            CDF = np.cumsum(self._cnt[w] / w_eff)[None, :]
            med = float(_binned_q(CDF, hr.z_reps, 0.5)[0])
            q75 = float(_binned_q(CDF, hr.z_reps, 0.75)[0])
            q25 = float(_binned_q(CDF, hr.z_reps, 0.25)[0])
            d = CDF[0] - hr.z_hist_cdf
            ks = float(np.max(np.abs(d)))
            pmf = self._cnt[w] / w_eff
            m = 0.5 * (pmf + hr.z_hist_pmf)
            with np.errstate(divide="ignore", invalid="ignore"):
                t1 = np.where(pmf > 0, pmf * (np.log(np.maximum(pmf, 1e-12))
                                              - np.log(np.maximum(m, 1e-12))),
                              0.0)
                t2 = np.where(hr.z_hist_pmf > 0,
                              hr.z_hist_pmf
                              * (np.log(np.maximum(hr.z_hist_pmf, 1e-12))
                                 - np.log(np.maximum(m, 1e-12))), 0.0)
            js = float(0.5 * (t1.sum() + t2.sum()))

            tbar = (w_eff - 1) / 2.0
            sxx = w_eff * (w_eff * w_eff - 1) / 12.0
            t0_ = max(n - w_eff, 0.0)
            sxy = (self._stz[w] - t0_ * self._s1[w]) - tbar * self._s1[w]
            if w_eff >= 4 and sxx > 0:
                slope = sxy / max(sxx, 1e-12)
                rv = max(self._s2[w] - self._s1[w] ** 2 / w_eff
                         - slope * sxy, 1e-12) / w_eff
                se = np.sqrt(rv / max(sxx, 1e-12)) + 1e-12
                slope_t = float(np.clip(slope / se, -50, 50))
            else:
                slope_t = 0.0

            mw = max(w_eff - 1, 1.0)
            if w_eff >= 4:
                acf1 = float(np.clip((self._pr[w] / mw - mean * mean)
                                     / max(var, 1e-12), -2, 2)) - h8.acf1_0
                crs = self._cr[w] / mw - h8.cross0
            else:
                acf1 = 0.0
                crs = 0.0

            mean_jump = (mean - h8.mean0) / h8.sd0 * np.sqrt(w_eff)
            logsd_jump = float(np.log(max(sd, 1e-9)) - np.log(h8.sd0))
            logmad_jump = float(np.log(max(q75 - q25, 1e-9))
                                - np.log(h8.iqr0))

            vals = {
                "med_jump": med - h8.med0,
                "logmad_jump": logmad_jump,
                "mean_jump": mean_jump,
                "logsd_jump": logsd_jump,
                "slope_t": slope_t,
                "ks": ks,
                "js": js,
                "acf1_delta": acf1,
                "cross_delta": crs,
            }
            for k in PER_W:
                E[col] = vals[k]
                col += 1
            for k in SCAN_OF:
                scan[k].append(abs(vals[k]))

        logw = np.log2(np.asarray(WINDOWS_S, dtype=np.float64))
        for k in SCAN_OF:
            arr = np.asarray(scan[k])
            j = int(arr.argmax())
            E[col] = float(arr[j])
            E[col + 1] = float(logw[j])
            col += 2

        self._zring[pos] = z
        self._bring[pos] = zb
        self._prring[pos] = pr
        self._crring[pos] = cr
        self._z_prev = z
        self._above_prev = above
        self.t = n
        return E

    def step(self, x):
        if not self.owns_inner:
            raise RuntimeError("use step_from_inner with a shared detector")
        base = self._inner.step(x)
        return np.concatenate((base, self._extra_from_current_inner(x)))

    def step_from_inner(self, x):
        return self._extra_from_current_inner(x)


if __name__ == "__main__":
    import time
    rng = np.random.default_rng(0)
    for trial in range(6):
        n_hist = int(rng.integers(80, 400))
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
            x_online[tau:] *= 1.8
        F_off = compute_features_offline(x_hist, x_online)
        det = StreamDetector8(x_hist)
        F_str = np.vstack([det.step(v) for v in x_online])
        assert F_off.shape == (T, N_FEATURES), (F_off.shape, N_FEATURES)
        err = np.max(np.abs(F_off - F_str))
        assert err < 1e-8, (trial, err)
    assert len(feature_names()) == N_FEATURES
    det = StreamDetector8(rng.normal(size=3000))
    xs = rng.normal(size=1000)
    t0 = time.perf_counter()
    for v in xs:
        det.step(v)
    dt = (time.perf_counter() - t0) / 1000
    print(f"OK: stream == offline; {N_FEATURES} features; {dt*1e6:.0f} us/step")
