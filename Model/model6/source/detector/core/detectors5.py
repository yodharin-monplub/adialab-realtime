"""
detectors5.py -- detectors3 + Gaussian GLR scan features.

For each suffix window w (the same geometric grid as detectors3) and for
both the raw z stream and the AR-whitened residual stream e/resid_sd:

    m = mean(window),  v = var(window)          (in hist-calibrated units)
    GLR_w = w_eff * (v + m^2 - ln(max(v, 1e-6)) - 1)

which is 2x the log generalized-likelihood ratio of "this segment has its
own Gaussian parameters" vs "it still follows the hist Gaussian". This is
the classic joint mean+variance scan statistic; the max over the window
grid approximates the full GLR scan over an unknown break age.

Features added (18): GLR_w for w in WINDOWS + max + log2(argmax age), for
z and for e.   Total = 236 + 18 = 254.

Everything else (aspects, calibration, layout) is inherited verbatim from
detectors3; offline == streaming is re-verified here including the new
block.
"""

from __future__ import annotations

import numpy as np

import detectors3 as d3

WINDOWS = d3.WINDOWS
ASPECTS = d3.ASPECTS
AR_P = d3.AR_P

N_GLR = 2 * (len(WINDOWS) + 2)
N_FEATURES = d3.N_FEATURES + N_GLR


def feature_names():
    names = d3.feature_names()
    for src in ("z", "e"):
        for w in WINDOWS:
            names.append(f"glr_{src}_w{w}")
        names.append(f"glr_{src}_max")
        names.append(f"glr_{src}_argage")
    return names


def _glr_block(sig, n, start_by_w, w_eff_by_w):
    """Offline GLR features for one signal (already H0-standardized).
    Returns list of columns: per-window GLR, max, log2 argmax age."""
    T = sig.size
    c1 = np.concatenate([[0.0], np.cumsum(sig)])
    c2 = np.concatenate([[0.0], np.cumsum(sig * sig)])
    cols = []
    glrs = []
    for w in WINDOWS:
        w_eff = w_eff_by_w[w]
        s1 = c1[1:] - c1[start_by_w[w]]
        s2 = c2[1:] - c2[start_by_w[w]]
        m = s1 / w_eff
        v = np.maximum(s2 / w_eff - m * m, 1e-6)
        glr = w_eff * (v + m * m - np.log(v) - 1.0)
        cols.append(glr)
        glrs.append(glr)
    G = np.column_stack(glrs)
    cols.append(G.max(axis=1))
    cols.append(np.log2(np.asarray(WINDOWS, dtype=np.float64))[G.argmax(axis=1)])
    return cols


def compute_features_offline(x_hist, x_online):
    F3 = d3.compute_features_offline(x_hist, x_online)

    hs = d3.Hist3(x_hist)
    x = np.asarray(x_online, dtype=np.float64)
    z = (x - hs.mu0) / hs.sd0
    e = d3._ar_residuals(z, hs.phi, hs.zh_tail) / hs.resid_sd

    T = z.size
    n = np.arange(1, T + 1, dtype=np.float64)
    idx = np.arange(1, T + 1)
    start_by_w = {w: np.maximum(0, idx - w) for w in WINDOWS}
    w_eff_by_w = {w: np.minimum(n, w) for w in WINDOWS}

    cols = _glr_block(z, n, start_by_w, w_eff_by_w)
    cols += _glr_block(e, n, start_by_w, w_eff_by_w)
    return np.column_stack([F3] + cols)


class StreamDetector5:
    def __init__(self, x_hist, inner=None):
        self._inner = d3.StreamDetector3(x_hist) if inner is None else inner
        self.owns_inner = inner is None
        hs = self._inner.hs
        self.hs = hs
        wmax = max(WINDOWS)
        self._wmax = wmax
        self._rings = {s: np.zeros(wmax) for s in ("z", "e")}
        self._s1 = {s: {w: 0.0 for w in WINDOWS} for s in ("z", "e")}
        self._s2 = {s: {w: 0.0 for w in WINDOWS} for s in ("z", "e")}
        self._log2w = {w: float(np.log2(w)) for w in WINDOWS}

    def _extra_from_current_inner(self, x):
        inner = self._inner
        hs = self.hs
        t = inner.t - 1

        z = (float(x) - hs.mu0) / hs.sd0
        # inner.e_prev holds this step's residual after the shared step.
        e = inner.e_prev / hs.resid_sd

        out = np.empty(N_GLR)
        i = 0
        n = t + 1
        pos = t % self._wmax
        for name, sig in (("z", z), ("e", e)):
            ring = self._rings[name]
            best = -np.inf
            best_logw = 0.0
            for w in WINDOWS:
                self._s1[name][w] += sig
                self._s2[name][w] += sig * sig
                drop = t - w
                if drop >= 0:
                    old = ring[drop % self._wmax]
                    self._s1[name][w] -= old
                    self._s2[name][w] -= old * old
                w_eff = min(n, w)
                m = self._s1[name][w] / w_eff
                v = max(self._s2[name][w] / w_eff - m * m, 1e-6)
                glr = w_eff * (v + m * m - np.log(v) - 1.0)
                out[i] = glr
                i += 1
                if glr > best:
                    best = glr
                    best_logw = self._log2w[w]
            out[i] = best
            out[i + 1] = best_logw
            i += 2
            ring[pos] = sig
        return out

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
            x_online[tau:] *= 1.8
        F_off = compute_features_offline(x_hist, x_online)
        det = StreamDetector5(x_hist)
        F_str = np.vstack([det.step(v) for v in x_online])
        assert F_off.shape == (T, N_FEATURES), (F_off.shape, N_FEATURES)
        err = np.max(np.abs(F_off - F_str))
        assert err < 1e-8, (trial, err)
    assert len(feature_names()) == N_FEATURES
    det = StreamDetector5(rng.normal(size=3000))
    xs = rng.normal(size=1000)
    t0 = time.perf_counter()
    for v in xs:
        det.step(v)
    dt = (time.perf_counter() - t0) / 1000
    print(f"OK: stream == offline; {N_FEATURES} features; {dt*1e6:.0f} us/step")
