"""Where does Model84 final beat the raw student? TS-AUC by online-time bucket."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.model10.sources import load_development_layout  # noqa: E402
from tsauc import tsauc_details  # noqa: E402

BUCKETS = [0, 8, 32, 64, 128, 256, 512, 1000]

L = load_development_layout(ROOT)
m84 = np.load(ROOT / "models/model109/cache/model84_final_oof.npy")
m67 = np.load(ROOT / "models/model109/cache/model67_baseline_oof.npy")
for k in (0, 1):
    held = L.row_fold == k
    y, t = L.target[held], L.time_online[held]
    stu = np.load(ROOT / f"models/model109/runs/fold{k}_mix.npy")
    hard = np.load(ROOT / f"models/model109/runs/fold{k}_hard.npy")
    print(f"fold {k}")
    for lo, hi in zip(BUCKETS[:-1], BUCKETS[1:]):
        m = (t >= lo) & (t < hi)
        r = [tsauc_details(y[m], v[held][m] if v.shape[0] != y.shape[0] else v[m], t[m])
             for v in (m84, m67)]
        s = [tsauc_details(y[m], v[m], t[m]) for v in (hard, stu)]
        print(f"  t[{lo},{hi}) weight {r[0].total_pair_weight:>12,d}  m84 {r[0].score:.4f} m67 {r[1].score:.4f}"
              f"  hard {s[0].score:.4f} mix {s[1].score:.4f}")
