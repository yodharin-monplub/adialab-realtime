"""Privileged (future-seeing) teacher features for Model109.

These features look at the *entire* online segment of a series, so they can
never be used at inference.  They exist only to train a teacher whose
cross-fitted probabilities become low-variance soft targets for a causal
student.  By the tower property, E[P(y_t | full series) | past] equals
P(y_t | past), so the student still estimates the correct causal posterior.

Output rows follow the Model 5 development layout used by the feature caches.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model10.sources import load_development_layout  # noqa: E402

AR_ORDER = 10
MIN_SEGMENT = 5
NAMES = [
    "L", "t", "t_frac",
    # innovation mean/scale channel
    "z_max", "z_full", "z_tau_hat", "z_age", "z_cdf", "z_cdf_max", "z_seg",
    "z_post_mean", "z_post_lsd", "z_post_ac1",
    # raw mean/scale channel
    "r_max", "r_tau_hat", "r_age", "r_cdf", "r_seg",
    # dependence channel
    "d_max", "d_age", "d_cdf", "d_seg",
    # series summaries
    "hist_ac1", "hist_kurt", "hist_sqac",
]


def ar_innovations(history: np.ndarray, online: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit AR(p) on history only; return standardized history and online innovations."""
    p = AR_ORDER
    h = history.astype(np.float64)
    design = np.column_stack([h[p - k:len(h) - k] for k in range(1, p + 1)] + [np.ones(len(h) - p)])
    coef, *_ = np.linalg.lstsq(design, h[p:], rcond=None)
    rh = h[p:] - design @ coef
    scale = rh.std() or 1.0
    stream = np.r_[h[-p:], online.astype(np.float64)]
    lagged = np.column_stack([stream[p - k:len(stream) - k] for k in range(1, p + 1)] + [np.ones(len(online))])
    ro = online - lagged @ coef
    return rh / scale, ro / scale


def suffix_gaussian_llr(x: np.ndarray) -> np.ndarray:
    """LLR of N(m, v) on x[k:] against N(0, 1), for every start k."""
    n = np.arange(len(x), 0, -1, dtype=np.float64)
    s1 = np.cumsum(x[::-1])[::-1]
    s2 = np.cumsum((x * x)[::-1])[::-1]
    mean = s1 / n
    var = np.maximum(s2 / n - mean * mean, 1e-6)
    llr = 0.5 * (s2 - n - n * np.log(var))
    llr[n < MIN_SEGMENT] = 0.0
    return llr


def suffix_dependence_llr(x: np.ndarray) -> np.ndarray:
    """LLR of lag-one correlation on x[k:] (length len(x)); zero where too short."""
    out = np.zeros(len(x))
    if len(x) < MIN_SEGMENT + 1:
        return out
    prod = x[1:] * x[:-1]
    sq = x[1:] * x[1:]
    sp = np.cumsum(prod[::-1])[::-1]
    ss = np.cumsum(sq[::-1])[::-1]
    m = np.arange(len(prod), 0, -1, dtype=np.float64)
    rho = np.clip(sp / np.maximum(ss, 1e-9), -0.99, 0.99)
    llr = -0.5 * m * np.log1p(-rho * rho)
    llr[m < MIN_SEGMENT] = 0.0
    out[:-1] = llr
    return out


def posterior_cdf(llr: np.ndarray) -> np.ndarray:
    """P(tau <= t | break, full data) under a uniform prior on tau."""
    w = np.exp(np.clip(llr - llr.max(), -50, 0))
    return np.cumsum(w) / w.sum()


def prefix_segment_llr(x: np.ndarray, start: int) -> np.ndarray:
    """For each t, Gaussian LLR of x[start:t+1] (zero before start)."""
    out = np.zeros(len(x))
    seg = x[start:]
    n = np.arange(1, len(seg) + 1, dtype=np.float64)
    s1 = np.cumsum(seg)
    s2 = np.cumsum(seg * seg)
    mean = s1 / n
    var = np.maximum(s2 / n - mean * mean, 1e-6)
    llr = 0.5 * (s2 - n - n * np.log(var))
    llr[n < MIN_SEGMENT] = 0.0
    out[start:] = llr
    return out


def acf1(a: np.ndarray) -> float:
    a = a - a.mean()
    d = float(a @ a)
    return float(a[1:] @ a[:-1]) / d if d > 0 else 0.0


def series_features(history: np.ndarray, online: np.ndarray) -> np.ndarray:
    L = len(online)
    t = np.arange(L, dtype=np.float64)
    zh, z = ar_innovations(history, online)
    zl = suffix_gaussian_llr(z)
    zk = int(np.argmax(zl))
    zc = posterior_cdf(zl)
    rl = suffix_gaussian_llr(online.astype(np.float64))
    rk = int(np.argmax(rl))
    rc = posterior_cdf(rl)
    dl = suffix_dependence_llr(z)
    dk = int(np.argmax(dl))
    dc = posterior_cdf(dl)
    post = z[zk:]
    kurt = float(((zh - zh.mean()) ** 4).mean() / (zh.var() ** 2 + 1e-12) - 3.0)
    cols = {
        "L": np.full(L, L), "t": t, "t_frac": (t + 1) / L,
        "z_max": np.full(L, zl.max()), "z_full": np.full(L, zl[0]),
        "z_tau_hat": np.full(L, zk), "z_age": t - zk, "z_cdf": zc,
        "z_cdf_max": zc * zl.max(), "z_seg": prefix_segment_llr(z, zk),
        "z_post_mean": np.full(L, post.mean()),
        "z_post_lsd": np.full(L, np.log(post.std() + 1e-6)),
        "z_post_ac1": np.full(L, acf1(post) if len(post) > 2 else 0.0),
        "r_max": np.full(L, rl.max()), "r_tau_hat": np.full(L, rk), "r_age": t - rk,
        "r_cdf": rc, "r_seg": prefix_segment_llr(online.astype(np.float64), rk),
        "d_max": np.full(L, dl.max()), "d_age": t - dk, "d_cdf": dc,
        "d_seg": np.where(t >= dk, dl.max(), 0.0),
        "hist_ac1": np.full(L, acf1(history)), "hist_kurt": np.full(L, kurt),
        "hist_sqac": np.full(L, acf1(history * history)),
    }
    return np.column_stack([cols[name] for name in NAMES]).astype(np.float32)


def build(out_path: Path) -> np.ndarray:
    layout = load_development_layout(ROOT)
    frame = pd.read_parquet(ROOT / "train/X_train.parquet")
    ids = frame.index.get_level_values(0).to_numpy()
    values = frame["value"].to_numpy()
    period = frame["period"].to_numpy()
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    ends = np.r_[starts[1:], len(ids)]
    bounds = {int(ids[s]): (s, e) for s, e in zip(starts, ends)}
    result = np.empty((layout.row_count, len(NAMES)), dtype=np.float32)
    position = layout.series_position
    row_start = np.flatnonzero(np.r_[True, position[1:] != position[:-1]])
    row_end = np.r_[row_start[1:], layout.row_count]
    for count, (a, b) in enumerate(zip(row_start, row_end)):
        sid = int(layout.series_ids[position[a]])
        s, e = bounds[sid]
        v, p = values[s:e], period[s:e]
        history, online = v[p == 1], v[p == 2]
        if len(online) != b - a or not np.array_equal(layout.time_online[a:b], np.arange(b - a)):
            raise ValueError(f"layout mismatch for series {sid}")
        result[a:b] = series_features(history, online)
        if count % 1000 == 0:
            print(f"teacher features: {count} series", flush=True)
    if not np.isfinite(result).all():
        raise ValueError("non-finite teacher feature")
    np.save(out_path, result)
    return result


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "cache/teacher_features.npy"
    target.parent.mkdir(parents=True, exist_ok=True)
    build(target)
    print("saved", target)
