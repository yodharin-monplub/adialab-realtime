"""Break down the outer-holdout result by online-time bucket and series half.

Usage: python models/model123/holdout_buckets.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tsauc import tsauc, tsauc_details  # noqa: E402

HERE = Path(__file__).resolve().parent
BUCKETS = [0, 32, 64, 128, 256, 512, 1000]


def main():
    d = np.load(HERE / "holdout_predictions.npz")
    chain, final, y, t = d["chain"], d["final"], d["target"], d["step"]
    ids = d["ids"]
    print(f"rows {len(y):,}  series {len(ids)}  positive rate {y.mean():.4f}")
    print(f"overall  sub28 {tsauc(y, chain, t):.6f}  sub30 {tsauc(y, final, t):.6f}")
    rows = {}
    for lo, hi in zip(BUCKETS[:-1], BUCKETS[1:]):
        m = (t >= lo) & (t < hi)
        a, b = tsauc_details(y[m], chain[m], t[m]), tsauc_details(y[m], final[m], t[m])
        rows[f"[{lo},{hi})"] = {"weight": a.total_pair_weight, "sub28": a.score, "sub30": b.score}
        print(f"t[{lo:>3},{hi:>4})  weight {a.total_pair_weight:>13,d}  sub28 {a.score:.4f}  sub30 {b.score:.4f}  "
              f"gain {b.score - a.score:+.4f}")
    # series-ID halves (same hash as the research runners)
    series_index = np.repeat(np.arange(len(ids)), np.diff(np.r_[np.flatnonzero(t == 0), len(t)]))
    halves = (ids.astype(np.uint64) * np.uint64(0x9E3779B1)) & np.uint64(1)
    for h in (0, 1):
        m = halves[series_index] == h
        rows[f"half{h}"] = {"sub28": tsauc(y[m], chain[m], t[m]), "sub30": tsauc(y[m], final[m], t[m])}
        print(f"ID half {h}: sub28 {rows[f'half{h}']['sub28']:.6f}  sub30 {rows[f'half{h}']['sub30']:.6f}")
    (HERE / "holdout_buckets.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
