"""Series-level bootstrap of the holdout gain: candidate (SRE40) vs Submission31, 1,667 sealed series."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tsauc import tsauc  # noqa: E402

ARM = sys.argv[1] if len(sys.argv) > 1 else "SRE40"
ref = np.load(ROOT / "models/model123/holdout_predictions.npz")
y, t = ref["target"], ref["step"]
cand = np.load(ROOT / f"models/model147/holdout_predictions_{ARM}.npz")["final"]
s31 = np.load(ROOT / "models/model124/holdout_predictions31.npz")["final"]
starts = np.flatnonzero(t == 0); ends = np.r_[starts[1:], len(t)]
print(f"point estimate: candidate {tsauc(y, cand, t):.6f}  Sub31 {tsauc(y, s31, t):.6f}  gain {tsauc(y, cand, t) - tsauc(y, s31, t):+.6f}")
half = len(starts) // 2
for name, sel in (("first half of the series", np.arange(half)), ("second half", np.arange(half, len(starts)))):
    rows = np.concatenate([np.arange(starts[i], ends[i]) for i in sel])
    print(f"  {name}: gain {tsauc(y[rows], cand[rows], t[rows]) - tsauc(y[rows], s31[rows], t[rows]):+.5f}")
for lo, hi in ((0, 64), (64, 256), (256, 512), (512, 10 ** 6)):
    b = (t >= lo) & (t < hi)
    print(f"  steps {lo:4d}-{hi if hi < 10 ** 6 else 'end':>4}: gain {tsauc(y[b], cand[b], t[b]) - tsauc(y[b], s31[b], t[b]):+.5f}")
rng = np.random.default_rng(0); gains = []
for _ in range(200):
    pick = rng.integers(0, len(starts), len(starts))
    rows = np.concatenate([np.arange(starts[i], ends[i]) for i in pick])
    gains.append(tsauc(y[rows], cand[rows], t[rows]) - tsauc(y[rows], s31[rows], t[rows]))
g = np.array(gains)
print(f"bootstrap (200 resamples of series): mean {g.mean():+.5f}  sd {g.std():.5f}  "
      f"95% CI [{np.percentile(g, 2.5):+.5f}, {np.percentile(g, 97.5):+.5f}]  P(gain > 0) = {np.mean(g > 0):.2f}")
