"""Series-level bootstrap of an offline holdout candidate (holdout_student_{TAG}.npz) vs BAG3 = Submission33.
Usage: python models/model152/bootstrap152.py TAG
"""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tsauc import tsauc  # noqa: E402

TAG = sys.argv[1]
ref = np.load(ROOT / "models/model123/holdout_predictions.npz")
y, t = ref["target"], ref["step"]
saved = np.load(Path(__file__).resolve().parent / f"holdout_student_{TAG}.npz")
assert len(saved["rows"]) == len(y), "candidate must cover the whole holdout"
cand = saved["final"]
base = np.load(ROOT / "models/model147/holdout_predictions_BAG3.npz")["final"]
starts = np.flatnonzero(t == 0); ends = np.r_[starts[1:], len(t)]
gain = lambda r: tsauc(y[r], cand[r], t[r]) - tsauc(y[r], base[r], t[r])  # noqa: E731
print(f"{TAG}: candidate {tsauc(y, cand, t):.6f}  BAG3/Sub33 {tsauc(y, base, t):.6f}  gain {gain(slice(None)):+.6f}")
half = len(starts) // 2
for name, sel in (("first half", np.arange(half)), ("second half", np.arange(half, len(starts)))):
    print(f"  {name}: {gain(np.concatenate([np.arange(starts[i], ends[i]) for i in sel])):+.5f}")
for lo, hi in ((0, 64), (64, 256), (256, 512), (512, 10 ** 6)):
    print(f"  steps {lo}-{hi if hi < 10 ** 6 else 'end'}: {gain(np.flatnonzero((t >= lo) & (t < hi))):+.5f}")
rng = np.random.default_rng(0)
g = np.array([gain(np.concatenate([np.arange(starts[i], ends[i]) for i in rng.integers(0, len(starts), len(starts))]))
              for _ in range(200)])
print(f"  bootstrap (200 series resamples): mean {g.mean():+.5f} sd {g.std():.5f} "
      f"95% [{np.percentile(g, 2.5):+.5f}, {np.percentile(g, 97.5):+.5f}]  P(gain>0) {np.mean(g > 0):.2f}")
