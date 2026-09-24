"""Model169: score the GPU-trained h256 nets on the sealed holdout and test them in the blend.

The deployed GRU part scores 0.6191 raw on the holdout (mean of its 15 nets, Model168).  Here the 15 new
nets (h256, 3 seeds x 5 folds, oracle-teacher targets, trained on the rented RTX 3090) are scored the same
way, and the blend is computed with an HONEST student: the dev-only SRP40 logits saved by Model164
(models/model164/holdout_SRP40.npz), at the deployed student weight 1.5.
Usage: python models/model169/score_gpu.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(1, str(ROOT / "models/model153")); sys.path.insert(2, str(ROOT / "models/model152"))

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    import offline_gru as og
    import offline_holdout as oh
    from tsauc import tsauc
    b = og.load_bundle(); starts, ends = og.series_bounds()
    cache = HERE / "data/h256_logits.npy"
    files = sorted((HERE / "runs/gpu").glob("h256s*_fold*_weights.npz"))
    if cache.exists():
        L = np.load(cache)
    else:
        Z = og.gru_inputs(b)
        L = np.asarray([og.torch_logits(f, 2, Z, starts, ends) for f in files], np.float32)
        (HERE / "data").mkdir(parents=True, exist_ok=True); np.save(cache, L)
    ref = np.load(ROOT / "models/model123/holdout_predictions.npz"); y, t = ref["target"], ref["step"]
    print("new nets, single:", np.round([tsauc(y, L[i], t) for i in range(len(L))], 4).tolist(), flush=True)
    new = L.mean(0)
    old = np.load(oh.H / "gru.npy")
    print(f"GRU component raw: deployed {tsauc(y, old, t):.6f} | new h256 x{len(L)} {tsauc(y, new, t):.6f} | "
          f"both averaged {tsauc(y, 0.5 * (old + np.load(ROOT / 'models/model168/data/gru_logits.npy').mean(0) * 0 + new), t):.6f}", flush=True)
    bc = oh.bundle_constants(); const = json.loads((ROOT / "models/model112/constants_m147.json").read_text())
    chain = np.load(oh.H / "chain.npy")
    dist = np.load(ROOT / "models/model164/holdout_SRP40.npz")["dist"]      # dev-only student (honest)
    zc = (chain - bc["mu_chain"]) / bc["sd_chain"]; zs = (dist - const["mu_dist"]) / const["sd_dist"]
    for name, g in (("deployed GRUs", old), ("new h256 nets", new), ("both", 0.5 * (old + new))):
        zg = (g - g.mean()) / g.std() * (old.std() / 1.0) if False else (g - bc["mu_gru"]) / bc["sd_gru"]
        for ws in (1.5,):
            f = 1 / (1 + np.exp(-(zc + ws * zs + bc["weight_gru"] * zg)))
            print(f"  blend with {name:<14} (ws={ws}): {tsauc(y, f, t):.6f}", flush=True)


if __name__ == "__main__":
    main()
