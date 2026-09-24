"""Stage 5, ensembling: top-level blend grid over the OOF components produced by the protocol.

chain (Submission28 OOF) + sum_i w_i * z(logit(component_i)), w_i in {0, .25, .5},
pilot folds 0-1 select, folds 2-4 confirm (models/model112/multi_blend.py machinery).
Components: bag5 (5-seed LightGBM student), gru (mean logit of the Model124 GRU sets),
E (stage-4 test features), cfg03 (stage-5 best hyperparameters), splx8k (8x pairing data).
Usage: python models/model141/ensemble.py [names...]
"""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import models.model112.multi_blend as mb  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
GRU = ROOT / "models/model124/remote_runs"


def gru_component():
    for k in range(5):
        out = RUNS / f"gru_fold{k}.npz"
        if out.exists():
            continue
        files = sorted(GRU.glob(f"*fold{k}_pred.npz"))
        logits, held = [], None
        for f in files:
            d = np.load(f)
            held = d["held_rows"] if held is None else held
            assert np.array_equal(d["held_rows"], held)
            p = np.clip(d["prediction"].astype(np.float64), 1e-6, 1 - 1e-6)
            logits.append(np.log(p) - np.log1p(-p))
        z = np.mean(logits, axis=0)
        np.savez(out, prediction=(1 / (1 + np.exp(-z))).astype(np.float32), held_rows=held)
        print(f"gru fold {k}: {len(files)} sets averaged", flush=True)


def main(names):
    gru_component()
    mb.COMPONENTS.update({
        "bag5": lambda k: RUNS / f"bag5_fold{k}.npz",
        "gru": lambda k: RUNS / f"gru_fold{k}.npz",
        "E": lambda k: ROOT / f"models/model140/runs/E_fold{k}.npz",
        "ED": lambda k: ROOT / f"models/model140/runs/ED_fold{k}.npz",
        "cfg03": lambda k: RUNS / f"cfg03_fold{k}.npz",
        "splx8k": lambda k: ROOT / f"models/model137/runs/splx8k_fold{k}.npz",
        "raw143": lambda k: ROOT / f"models/model143/fold{k}_pred.npz",  # raw-series TCN+GRU (Model143)
    })
    mb.main(names or ["bag5", "gru", "E", "cfg03", "splx8k"])


if __name__ == "__main__":
    main(sys.argv[1:])
