"""Split seq_splices.npz into one file per PLAN - the fold-safety fix.

LEAK FOUND 2026-09-25: models/model18/augmentation.select_pairs drops a donor only when its series is in
the plan's OWN held fold, so plan v's spliced series may contain real post-break tails of fold k for every
k != v.  train_gpu2.py keeps a synthetic series when its row_fold is neither the held fold k nor (k+1)%5,
which for splices tagged by PLAN index keeps plans v != k - i.e. exactly the plans that may carry fold-k
donors.  Fold 0 then scored 0.6551 (+0.017 over the baseline), the signature of a leak, not a gain.

The deployed tree student never had this problem: for fold k it uses plan k only.  This writes
seq_splices_p{v}.npz holding plan v's series with row_fold = -1 ("always keep"), so training fold k with
ONLY seq_splices_p{k}.npz reproduces the student's convention and cannot see fold-k donors.
  python models/model172/split_splices.py
"""
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE / "data/seq_splices.npz"


def main():
    d = np.load(SRC)
    X, soft, target = d["X"], d["soft"], d["target"]
    time_online, pos, fold = d["time_online"], d["series_position"], d["row_fold"]
    for v in range(5):
        out = HERE / f"data/seq_splices_p{v}.npz"
        if out.exists():
            print("exists", out.name)
            continue
        rows = fold == v
        p = pos[rows]
        _, compact = np.unique(p, return_inverse=True)          # renumber series from 0
        np.savez(out, X=X[rows], soft=soft[rows], target=target[rows], time_online=time_online[rows],
                 series_position=compact.astype(np.int32),
                 row_fold=np.full(rows.sum(), -1, np.int8))     # -1 = never excluded; safety is the plan
        print(f"{out.name}: {rows.sum():,} rows, {len(np.unique(p)):,} series", flush=True)


if __name__ == "__main__":
    main()
