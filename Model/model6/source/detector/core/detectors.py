"""
detectors.py -- per-step online change detection features.
===========================================================

Two implementations of the SAME math:

  compute_features_offline(x_hist, x_online) -> (T, F) ndarray
      Vectorized over the whole online segment at once. Used for training
      and for fast local evaluation.

  StreamDetector(x_hist).step(x) -> (F,) ndarray
      Incremental O(1)-per-point version used by the submission's infer
      generator. Produces values identical to the offline version at every
      prefix (unit-tested in __main__).

DESIGN
------
The historical segment is break-free, so everything is measured relative to
it. Per online point x_t:

  z_t = (x_t - mu0) / sd0                    mean/scale-standardized
  u_t = F0(x_t)                              hist empirical CDF (rank) value

Cumulative processes (from the online start), each tracked with running
min/max so both one-sided CUSUMs are available in O(1):

  S  = cumsum(z)               mean shift
  Q  = cumsum(z^2 - 1)         variance shift (hist-standardized)
  A  = cumsum(|z| - E0|z|)     robust scale shift (E0|z| measured on hist z)
  U  = cumsum(u - 0.5)         median / location shift, rank-based
  V  = cumsum(|u - .5| - v0)   dispersion shift, rank-based (v0 from hist)
  W  = cumsum(tail - w0)       tail-frequency shift (u<0.05 or u>0.95)
  R  = cumsum(z_t*z_{t-1}-r0)  first-lag dependence shift (r0 from hist)

For each cumulative process C the features at step t (0-based, n = t+1):
  up_t   = (C_t - min_{s<=t} C_s) / sqrt(n)
  down_t = (max_{s<=t} C_s - C_t) / sqrt(n)
CUSUM normalized by sqrt(n) so magnitudes are comparable across steps and
series. The max of up/down over both is the classic two-sided CUSUM scan.

Plus rolling suffix windows w in WINDOWS (last min(n, w) points):
  |mean(z)| * sqrt(w_eff), (mean(z^2)-1)*sqrt(w_eff/2), |mean(u)-.5|*sqrt(12*w_eff)
i.e. approximately N(0,1)-scaled shift statistics on the most recent window,
which react fast right after a break.

And context features: log1p(n), log(len hist).

All arithmetic is float64 numpy in both paths, accumulated in the same
order, so the two implementations agree exactly (not just approximately).
"""

from __future__ import annotations

import numpy as np

WINDOWS = (10, 25, 50, 100, 250)

# cumulative processes get 2 features each (up/down); windows get 3 each
_CUM_NAMES = ("S", "Q", "A", "U", "V", "W", "R")


def feature_names():
    names = []
    for c in _CUM_NAMES:
        names += [f"{c}_up", f"{c}_down"]
    for w in WINDOWS:
        names += [f"win{w}_mean", f"win{w}_var", f"win{w}_rank"]
    names += ["log_n", "log_hist"]
    return names


N_FEATURES = len(feature_names())


class HistStats:
    """Everything about the historical segment the detectors need."""

    __slots__ = ("mu0", "sd0", "sorted_hist", "n_hist", "abs_mean",
                 "v0", "w0", "r0", "log_hist")

    def __init__(self, x_hist):
        h = np.asarray(x_hist, dtype=np.float64)
        self.n_hist = h.size
        self.mu0 = float(h.mean())
        sd = float(h.std())
        self.sd0 = sd if sd > 1e-12 else 1e-12
        self.sorted_hist = np.sort(h)
        zh = (h - self.mu0) / self.sd0
        self.abs_mean = float(np.abs(zh).mean())
        # hist ranks against itself -> the H0 values of the rank statistics
        uh = self._cdf(h)
        self.v0 = float(np.abs(uh - 0.5).mean())
        self.w0 = float(((uh < 0.05) | (uh > 0.95)).mean())
        if self.n_hist > 1:
            self.r0 = float((zh[1:] * zh[:-1]).mean())
        else:
            self.r0 = 0.0
        self.log_hist = float(np.log(max(self.n_hist, 1)))

    def _cdf(self, x):
        """Empirical CDF of the hist segment, mid-rank convention."""
        x = np.asarray(x, dtype=np.float64)
        lo = np.searchsorted(self.sorted_hist, x, side="left")
        hi = np.searchsorted(self.sorted_hist, x, side="right")
        return (lo + hi) / (2.0 * self.n_hist)


def _increments(hs: HistStats, x_online):
    """The per-point increments of every cumulative process, vectorized.
    Returns dict name -> (T,) float64 array, plus z and u themselves."""
    x = np.asarray(x_online, dtype=np.float64)
    z = (x - hs.mu0) / hs.sd0
    u = hs._cdf(x)
    zprev = np.empty_like(z)
    zprev[0] = 0.0          # no previous point: contributes r0-neutral 0 - r0
    zprev[1:] = z[:-1]
    inc = {
        "S": z,
        "Q": z * z - 1.0,
        "A": np.abs(z) - hs.abs_mean,
        "U": u - 0.5,
        "V": np.abs(u - 0.5) - hs.v0,
        "W": ((u < 0.05) | (u > 0.95)).astype(np.float64) - hs.w0,
        "R": z * zprev - hs.r0,
    }
    return inc, z, u


def compute_features_offline(x_hist, x_online):
    """(T, N_FEATURES) float64: the feature vector the stream sees at each
    online step t (after observing x_online[t])."""
    hs = HistStats(x_hist)
    inc, z, u = _increments(hs, x_online)
    T = z.size
    n = np.arange(1, T + 1, dtype=np.float64)
    sqrt_n = np.sqrt(n)

    cols = []
    for name in _CUM_NAMES:
        C = np.cumsum(inc[name])
        run_min = np.minimum.accumulate(np.minimum(C, 0.0))
        run_max = np.maximum.accumulate(np.maximum(C, 0.0))
        cols.append((C - run_min) / sqrt_n)
        cols.append((run_max - C) / sqrt_n)

    cz = np.concatenate([[0.0], np.cumsum(z)])
    cz2 = np.concatenate([[0.0], np.cumsum(z * z)])
    cu = np.concatenate([[0.0], np.cumsum(u)])
    for w in WINDOWS:
        w_eff = np.minimum(n, w)
        start = np.maximum(0, np.arange(1, T + 1) - w)
        mz = (cz[1:] - cz[start]) / w_eff
        mz2 = (cz2[1:] - cz2[start]) / w_eff
        mu_ = (cu[1:] - cu[start]) / w_eff
        cols.append(np.abs(mz) * np.sqrt(w_eff))
        cols.append((mz2 - 1.0) * np.sqrt(w_eff / 2.0))
        cols.append(np.abs(mu_ - 0.5) * np.sqrt(12.0 * w_eff))

    cols.append(np.log1p(n))
    cols.append(np.full(T, hs.log_hist))
    return np.column_stack(cols)


class StreamDetector:
    """Incremental version of compute_features_offline. One instance per
    series; call .step(x) per online point, get the same row the offline
    matrix has at that prefix."""

    def __init__(self, x_hist):
        self.hs = HistStats(x_hist)
        self.t = 0
        self.z_prev = 0.0
        self.C = {k: 0.0 for k in _CUM_NAMES}
        self.Cmin = {k: 0.0 for k in _CUM_NAMES}
        self.Cmax = {k: 0.0 for k in _CUM_NAMES}
        wmax = max(WINDOWS)
        self._buf_z = np.zeros(wmax)      # ring buffers of z and u
        self._buf_u = np.zeros(wmax)
        self._sum_z = {w: 0.0 for w in WINDOWS}
        self._sum_z2 = {w: 0.0 for w in WINDOWS}
        self._sum_u = {w: 0.0 for w in WINDOWS}

    def step(self, x):
        hs = self.hs
        x = float(x)
        z = (x - hs.mu0) / hs.sd0
        lo = np.searchsorted(hs.sorted_hist, x, side="left")
        hi = np.searchsorted(hs.sorted_hist, x, side="right")
        u = (lo + hi) / (2.0 * hs.n_hist)

        inc = {
            "S": z,
            "Q": z * z - 1.0,
            "A": abs(z) - hs.abs_mean,
            "U": u - 0.5,
            "V": abs(u - 0.5) - hs.v0,
            "W": (1.0 if (u < 0.05 or u > 0.95) else 0.0) - hs.w0,
            "R": z * self.z_prev - hs.r0,
        }
        t = self.t
        n = t + 1
        sqrt_n = np.sqrt(float(n))

        out = np.empty(N_FEATURES)
        i = 0
        for name in _CUM_NAMES:
            C = self.C[name] + inc[name]
            self.C[name] = C
            self.Cmin[name] = min(self.Cmin[name], min(C, 0.0))
            self.Cmax[name] = max(self.Cmax[name], max(C, 0.0))
            out[i] = (C - self.Cmin[name]) / sqrt_n
            out[i + 1] = (self.Cmax[name] - C) / sqrt_n
            i += 2

        wmax = self._buf_z.size
        pos = t % wmax
        for w in WINDOWS:
            drop = t - w                     # index leaving THIS window
            self._sum_z[w] += z
            self._sum_z2[w] += z * z
            self._sum_u[w] += u
            if drop >= 0:
                zd = self._buf_z[drop % wmax]
                ud = self._buf_u[drop % wmax]
                self._sum_z[w] -= zd
                self._sum_z2[w] -= zd * zd
                self._sum_u[w] -= ud
            w_eff = min(n, w)
            mz = self._sum_z[w] / w_eff
            mz2 = self._sum_z2[w] / w_eff
            mu_ = self._sum_u[w] / w_eff
            out[i] = abs(mz) * np.sqrt(float(w_eff))
            out[i + 1] = (mz2 - 1.0) * np.sqrt(w_eff / 2.0)
            out[i + 2] = abs(mu_ - 0.5) * np.sqrt(12.0 * w_eff)
            i += 3
        self._buf_z[pos] = z
        self._buf_u[pos] = u

        out[i] = np.log1p(float(n))
        out[i + 1] = hs.log_hist
        self.z_prev = z
        self.t = n
        return out


# ---------------------------------------------------------------------------
# Self-test: stream == offline, exactly.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for trial in range(8):
        n_hist = int(rng.integers(50, 400))
        T = int(rng.integers(5, 300))
        x_hist = rng.normal(size=n_hist)
        x_online = rng.normal(size=T)
        if trial % 2 == 0 and T > 10:       # inject a break
            tau = int(rng.integers(1, T - 1))
            kind = trial % 4
            if kind == 0:
                x_online[tau:] += 1.5
            else:
                x_online[tau:] *= 2.5
        F_off = compute_features_offline(x_hist, x_online)
        det = StreamDetector(x_hist)
        F_str = np.vstack([det.step(x) for x in x_online])
        assert F_off.shape == (T, N_FEATURES)
        err = np.max(np.abs(F_off - F_str))
        assert err < 1e-10, (trial, err)
    assert len(feature_names()) == N_FEATURES
    print(f"OK: stream == offline on all trials; {N_FEATURES} features")
