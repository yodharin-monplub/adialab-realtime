"""Omnibus energies over normalized detectors3 aspect channels.

Tree models must otherwise learn many high-order interactions to recognize a
break that moves several individually weak channels at once.  This module
groups related, already history-normalized detectors3 statistics and emits
three explicit cross-channel summaries at every scan/window scale:

* mean squared evidence;
* maximum absolute evidence;
* mean absolute evidence.

The transform is stateless once a detectors3 row exists.  The standalone
stream class is retained to verify the independent offline and incremental
paths exactly; a package can reuse an existing detectors3 row directly.
"""

from __future__ import annotations

import time

import numpy as np

import detectors3 as d3


BLOCK = 3 + len(d3.WINDOWS)
GROUPS = {
    "shape": ("L1", "L2", "L3", "L4"),
    "tail": ("tail_up", "tail_dn", "q25", "q50", "q75"),
    "moments": ("z", "z2", "absz", "e", "e2", "abse"),
    "depend": ("ac1", "ac2", "ac3", "ac5", "vc1", "sgn1", "rk1",
               "eac1"),
    "resid": ("e", "e2", "abse", "eac1"),
    "all": d3.ASPECTS,
}
_ASPECT_INDEX = {name: index for index, name in enumerate(d3.ASPECTS)}
_GROUP_INDEX = tuple(
    np.asarray([_ASPECT_INDEX[name] for name in aspects], dtype=np.intp)
    for aspects in GROUPS.values()
)
STAT_NAMES = ("prefix", "up", "down") + tuple(f"w{w}" for w in d3.WINDOWS)
SUMMARY_NAMES = ("meansq", "maxabs", "meanabs")
N_EXTRA = len(GROUPS) * BLOCK * len(SUMMARY_NAMES)
N_FEATURES = d3.N_FEATURES + N_EXTRA


def feature_names():
    return d3.feature_names() + [
        f"omni_{group}_{stat}_{summary}"
        for group in GROUPS
        for stat in STAT_NAMES
        for summary in SUMMARY_NAMES
    ]


def _group_columns(aspects, stat):
    return [_ASPECT_INDEX[name] * BLOCK + stat for name in aspects]


def extra_from_base(base):
    F = np.asarray(base, dtype=np.float64)
    one_row = F.ndim == 1
    if one_row:
        matrix = F[: len(d3.ASPECTS) * BLOCK].reshape(len(d3.ASPECTS), BLOCK)
        out = np.empty(N_EXTRA, dtype=np.float64)
        offset = 0
        for columns in _GROUP_INDEX:
            values = matrix[columns]
            absolute = np.abs(values)
            summaries = np.column_stack(
                (
                    np.mean(values * values, axis=0),
                    np.max(absolute, axis=0),
                    np.mean(absolute, axis=0),
                )
            )
            width = summaries.size
            out[offset : offset + width] = summaries.ravel()
            offset += width
        assert offset == N_EXTRA
        return out
    out = np.empty((F.shape[0], N_EXTRA), dtype=np.float64)
    k = 0
    for aspects in GROUPS.values():
        for stat in range(BLOCK):
            values = F[:, _group_columns(aspects, stat)]
            absolute = np.abs(values)
            out[:, k] = np.mean(values * values, axis=1)
            out[:, k + 1] = np.max(absolute, axis=1)
            out[:, k + 2] = np.mean(absolute, axis=1)
            k += 3
    assert k == N_EXTRA
    return out


def compute_features_offline(x_hist, x_online):
    base = d3.compute_features_offline(x_hist, x_online)
    return np.column_stack([base, extra_from_base(base)])


class StreamDetector16:
    def __init__(self, x_hist):
        self.inner = d3.StreamDetector3(x_hist)

    def step(self, x):
        base = self.inner.step(x)
        return np.concatenate([base, extra_from_base(base)])


if __name__ == "__main__":
    rng = np.random.default_rng(16)
    for trial in range(10):
        n_hist = int(rng.integers(1000, 3000))
        T = int(rng.integers(5, 700))
        ar = rng.uniform(-0.8, 0.8)
        eps = rng.standard_t(df=rng.uniform(3.0, 12.0), size=n_hist + T)
        x = np.empty(n_hist + T)
        x[0] = eps[0]
        for i in range(1, x.size):
            x[i] = ar * x[i - 1] + eps[i]
        hist, online = x[:n_hist].copy(), x[n_hist:].copy()
        if trial % 2 == 0 and T > 10:
            tau = int(rng.integers(1, T - 1))
            online[tau:] += rng.uniform(0.3, 1.2)
        offline = compute_features_offline(hist, online)
        stream = StreamDetector16(hist)
        streamed = np.vstack([stream.step(value) for value in online])
        assert offline.shape == (T, N_FEATURES)
        error = float(np.max(np.abs(offline - streamed)))
        assert error < 1e-9, (trial, error)
    assert len(feature_names()) == N_FEATURES
    detector = StreamDetector16(rng.normal(size=3000))
    online = rng.normal(size=1000)
    t0 = time.perf_counter()
    for value in online:
        detector.step(value)
    elapsed = (time.perf_counter() - t0) / len(online)
    print(f"OK: stream == offline; {N_FEATURES} features; "
          f"{elapsed * 1e6:.0f} us/step")
