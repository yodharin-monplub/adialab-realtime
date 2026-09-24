"""Inspect the three datasets the user downloaded into data/ (2025 challenge, open benchmark, real-time)."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DIRS = {"2025": ROOT / "data/ADIA Lab Structural Break Challenge",
        "open": ROOT / "data/ADIA Lab Structural Break Open Benchmark Challenge",
        "rt": ROOT / "data/ADIA Lab Structural Break Challenge Real-Time Edition"}


def fingerprint(values, k=24):
    """Hash of k consecutive rounded values - identical sub-sequences give identical hashes."""
    return hash(np.round(values[:k], 6).tobytes())


frames = {}
for name, d in DIRS.items():
    x = pd.read_parquet(d / "X_train.parquet"); y = pd.read_parquet(d / "y_train.parquet")
    frames[name] = x
    ids = x.index.get_level_values(0)
    n = ids.nunique(); lengths = x.groupby(level=0).size()
    print(f"\n== {name}: {d.name}")
    print("   X columns", list(x.columns), "index", x.index.names, "| rows", len(x), "| series", n)
    print("   length min/median/max", int(lengths.min()), int(lengths.median()), int(lengths.max()))
    print("   period values", sorted(x['period'].unique().tolist()), "| value mean/sd", round(float(x['value'].mean()), 4), round(float(x['value'].std()), 4))
    print("   y columns", list(y.columns), "shape", y.shape, "| head", y.iloc[:3, 0].tolist(), "| mean", round(float(y.iloc[:, 0].astype(float).mean()), 3))
    if name != "rt":
        after = x[x["period"] == x["period"].max()].groupby(level=0).size()
        before = x[x["period"] == x["period"].min()].groupby(level=0).size()
        print("   before-boundary length min/median/max", int(before.min()), int(before.median()), int(before.max()),
              "| after", int(after.min()), int(after.median()), int(after.max()))

# overlap: same first values of the series?
prints = {}
for name, x in frames.items():
    v = x["value"].to_numpy(); ids = x.index.get_level_values(0).to_numpy()
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    prints[name] = {fingerprint(v[s:s + 24]) for s in starts}
for a, b in (("2025", "open"), ("2025", "rt"), ("open", "rt")):
    common = len(prints[a] & prints[b])
    print(f"\nseries starting with identical values: {a} vs {b}: {common} of {len(prints[a])} / {len(prints[b])}")
