"""Sanity check: the CURRENT student (trained on this year's data only) scored on the converted 2025 rows.
Similar TS-AUC to this year's held folds => similar kind of breaks; very different => different generator."""
import json
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tsauc import tsauc  # noqa: E402

ext = np.load(ROOT / "models/model145/data/external.npz")
protocol = json.loads((ROOT / "models/model84/runs/e1e98a6e8af2df25/protocol.json").read_text())
union = protocol["contract"]["union_names"]
y, t = ext["target"], ext["time_online"]
for fold in (0, 1):
    cols = np.asarray([union.index(n) for n in protocol["contract"]["names"][fold]])
    model = lgb.Booster(model_file=str(ROOT / f"models/model111/runs/a050_fold{fold}.txt"))
    p = model.predict(ext["X_union"][:, cols], num_threads=2)
    print(f"current student fold-{fold} model on external rows: TS-AUC {tsauc(y, p, t):.4f}  (rows {len(y):,}, positives {y.mean():.3f})")
    for lo, hi in ((0, 64), (64, 256), (256, 2000)):
        b = (t >= lo) & (t < hi)
        print(f"    steps {lo:4d}-{hi:<5d} {tsauc(y[b], p[b], t[b]):.4f}")
