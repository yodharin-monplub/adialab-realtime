"""Score the synthetic full sequences of fold k with the fold-k GRU ensemble (CPU torch).

The fold-k GRUs (h192, h96l3, h128) never saw fold k, so their outputs on
fold k's synthetic plan are fold-clean for the held-fold evaluation.
Output: models/model129/data/syn_gru_fold{k}.npy  (mean logit per full row)
Usage: python models/model129/score_synthetic.py [fold ...]
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models/model124"))
from train_gru import GRUStudent  # noqa: E402

HERE = Path(__file__).resolve().parent
CONFIGS = [("", 2), ("h96l3_", 3), ("h128_", 2)]


def load_model(path, layers):
    w = dict(np.load(path))
    model = GRUStudent(w["gru.weight_ih_l0"].shape[1], w["gru.weight_hh_l0"].shape[1], layers)
    model.load_state_dict({k: torch.from_numpy(v) for k, v in w.items()})
    return model.eval()


def main(folds):
    torch.set_num_threads(8)
    for k in folds:
        out = HERE / "data" / f"syn_gru_fold{k}.npy"
        if out.exists():
            continue
        syn = np.load(HERE / "data" / f"syn_fold{k}.npz")
        X = syn["X"].astype(np.float32); starts = syn["starts"]; lengths = syn["lengths"]
        models = [load_model(ROOT / f"models/model124/remote_runs/{tag}fold{k}_weights.npz", layers) for tag, layers in CONFIGS]
        logit = np.zeros(len(X), np.float32)
        t0 = time.time()
        order = np.argsort(lengths)
        with torch.no_grad():
            for b in range(0, len(order), 64):
                idx = order[b:b + 64]
                T = int(lengths[idx].max())
                xb = np.zeros((len(idx), T, X.shape[1]), np.float32)
                for i, s in enumerate(idx):
                    xb[i, :lengths[s]] = X[starts[s]:starts[s] + lengths[s]]
                xb = torch.from_numpy(xb)
                o = np.mean([m(xb).numpy() for m in models], axis=0)
                for i, s in enumerate(idx):
                    logit[starts[s]:starts[s] + lengths[s]] = o[i, :lengths[s]]
        np.save(out, logit)
        print(f"fold {k}: scored {len(X):,} synthetic rows ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [0, 1, 2, 3, 4])
