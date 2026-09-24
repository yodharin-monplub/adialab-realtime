"""Stage 5, seeds: TS-AUC of the k-seed logit bag (a050 + seed1..4) per fold vs the single seed.

Also writes runs/bag5_fold{k}.npz (probability of the 5-seed bag) so select_blend.py / multi_blend
can use it as a component.  Usage: python models/model141/bag_eval.py
"""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.model10.sources import load_development_layout  # noqa: E402
from tsauc import tsauc  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def logit(p):
    p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def main():
    L = load_development_layout(ROOT)
    print("seeds  " + "  ".join(f"fold{k}" for k in range(5)) + "   mean   gain")
    rows = []
    for n in range(1, 6):
        scores = []
        for k in range(5):
            held = L.row_fold == k
            files = [ROOT / f"models/model111/runs/a050_fold{k}.npz"] + [RUNS / f"seed{s}_fold{k}.npz" for s in range(1, n)]
            if not all(f.exists() for f in files):
                scores.append(np.nan); continue
            z = np.mean([logit(np.load(f)["prediction"]) for f in files], axis=0)
            if n == 5:
                np.savez(RUNS / f"bag5_fold{k}.npz", prediction=(1 / (1 + np.exp(-z))).astype(np.float32),
                         held_rows=np.flatnonzero(held))
            scores.append(tsauc(L.target[held], z, L.time_online[held]))
        scores = np.asarray(scores); rows.append(scores)
        print(f"{n:5d}  " + "  ".join(f"{s:.4f}" for s in scores) + f"  {np.nanmean(scores):.4f}  {np.nanmean(scores) - np.nanmean(rows[0]):+.4f}")
    text = "\n".join(f"{n + 1} seeds: folds " + " ".join(f"{s:.4f}" for s in r) + f" mean {np.nanmean(r):.4f}" for n, r in enumerate(rows))
    (HERE / "score.txt").write_text("Seed bag of the Model111 recipe (logit mean), held-fold TS-AUC\n" + text + "\n")


if __name__ == "__main__":
    main()
