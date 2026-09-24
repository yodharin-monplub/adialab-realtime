"""Model168: what is a BIGGER GRU ensemble worth?  (decides whether renting a GPU to train more GRUs pays)

Model153 showed the GRU part is limited by seed variance, not by data or targets: one network scores
0.60-0.62 on the holdout while the mean of five scores 0.6198.  Here the 15 deployed GRUs are scored
individually on the sealed holdout (from Model152's saved inputs) and the blend is measured with 1, 3, 5, 10
and 15 of them, at the deployed student weight and at 1.5 (Submission40).  The slope from 5 -> 15 says what
another 15-30 networks could add.
Usage: python models/model168/gru_count.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(1, str(ROOT / "models/model153")); sys.path.insert(2, str(ROOT / "models/model152"))

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
CONFIGS = [("", 2), ("h96l3_", 3), ("h128_", 2)]   # the deployed three configs x 5 folds


def main():
    import offline_gru as og
    import offline_holdout as oh
    from tsauc import tsauc
    cache = HERE / "data/gru_logits.npy"
    b = og.load_bundle(); starts, ends = og.series_bounds()
    if cache.exists():
        L = np.load(cache)
    else:
        Z = og.gru_inputs(b)
        rows = []
        for tag, layers in CONFIGS:
            for k in range(5):
                rows.append(og.torch_logits(ROOT / f"models/model124/remote_runs/{tag}fold{k}_weights.npz", layers, Z, starts, ends))
                print("scored", tag or "h192_", k, flush=True)
        L = np.asarray(rows, np.float32)
        (HERE / "data").mkdir(parents=True, exist_ok=True); np.save(cache, L)
    ref = np.load(ROOT / "models/model123/holdout_predictions.npz"); y, t = ref["target"], ref["step"]
    bc = oh.bundle_constants(); const = json.loads((ROOT / "models/model112/constants_m147_bag3.json").read_text())
    chain = np.load(oh.H / "chain.npy")
    st = np.load(ROOT / "models/model152/holdout_student_AP3.npz")   # in-sample student, only for the weight scale
    dist = st["dist"]
    zc = (chain - bc["mu_chain"]) / bc["sd_chain"]; zs = (dist - const["mu_dist"]) / const["sd_dist"]
    rng = np.random.default_rng(168)
    print("single GRUs:", np.round([tsauc(y, L[i], t) for i in range(len(L))], 4).tolist(), flush=True)
    for n in (1, 3, 5, 10, 15):
        raws, finals = [], []
        for _ in range(5 if n < 15 else 1):
            pick = rng.choice(len(L), n, replace=False)
            g = L[pick].mean(0)
            raws.append(tsauc(y, g, t))
            zg = (g - bc["mu_gru"]) / bc["sd_gru"]
            finals.append(tsauc(y, 1 / (1 + np.exp(-(zc + 1.5 * zs + bc["weight_gru"] * zg))), t))
        print(f"{n:2d} GRUs: raw {np.mean(raws):.6f}  final(ws=1.5, in-sample student) {np.mean(finals):.6f}", flush=True)


if __name__ == "__main__":
    main()
