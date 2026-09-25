"""Model172 copy of models/model171/weights_eval171.py: the blend weights with the nets that
trained on REAL + SYNTHETIC sequences.

blend = z(chain) + ws * z(student SRP40) + wg * z(GRU set).  Weights are picked on pilot folds 0-1 and
confirmed on folds 2-4, exactly as every earlier weight decision in this project.
  python models/model172/weights_eval172.py synR     (real + re-cut sequences)
  python models/model172/weights_eval172.py synRS    (real + re-cuts + splices)
  python models/model172/weights_eval172.py all      (Model170 nets, the deployed Submission42 set)
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.model10.sources import load_development_layout  # noqa: E402
from tsauc import tsauc  # noqa: E402

OLD = ROOT / "models/model124/remote_runs"
NEW = ROOT / "models/model148"
SET = sys.argv[1] if len(sys.argv) > 1 else "old"


def lg(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def main():
    L = load_development_layout(ROOT)
    c = json.loads((ROOT / "models/model112/constants.json").read_text())
    cb = json.loads((ROOT / "models/model112/constants_m147_bag3.json").read_text())
    chain = (np.load(ROOT / "models/model112/chain_m73_all.npy").astype(np.float64) - c["mu_chain"]) / c["sd_chain"]
    student = np.empty(L.row_count); gru = np.empty(L.row_count)
    for k in range(5):
        m = L.row_fold == k
        student[m] = (lg(np.load(ROOT / f"models/model164/runs/SRP40_fold{k}.npz")["prediction"]) - cb["mu_dist"]) / cb["sd_dist"]
        if SET in ("synR", "synRS"):
            files = [ROOT / f"models/model172/runs/gpu/{SET}_fold{k}_pred.npz"]
        elif SET == "all":
            files = [ROOT / f"models/model170/runs/gpu/all_fold{k}_pred.npz"]
        elif SET == "new":
            files = [ROOT / f"models/model169/runs/gpu/h256s0_fold{k}_pred.npz"]
        else:
            files = [OLD / f"fold{k}_pred.npz", OLD / f"h96l3_fold{k}_pred.npz", OLD / f"h128_fold{k}_pred.npz"]
        gru[m] = np.mean([lg(np.load(f)["prediction"]) for f in files], axis=0)
    gru = (gru - gru.mean()) / gru.std()
    masks = [L.row_fold == k for k in range(5)]

    def score(v):
        return np.array([tsauc(L.target[m], v[m], L.time_online[m]) for m in masks])

    base = score(chain + 0.5 * student + 0.5 * gru)
    print(f"GRU set '{SET}': reference ws=0.5 wg=0.5 folds {np.round(base, 5).tolist()} mean {base.mean():.5f}")
    table = {}
    for ws in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        for wg in (0.25, 0.5, 0.75, 1.0, 1.5):
            d = score(chain + ws * student + wg * gru) - base
            table[(ws, wg)] = d
            print(f"  ws={ws:<4} wg={wg:<4} pilot {d[:2].mean():+.5f}  confirm {d[2:].mean():+.5f}  all {d.mean():+.5f}  wins {(d > 0).sum()}/5", flush=True)
    best = max(table, key=lambda kv: table[kv][:2].mean())
    d = table[best]
    print(f"PILOT-SELECTED ws={best[0]} wg={best[1]}: pilot {d[:2].mean():+.5f}, confirmation (folds 2-4) {d[2:].mean():+.5f} "
          f"({(d[2:] > 0).sum()}/3 wins), all {d.mean():+.5f}")
    nochain = score(student + 0.5 * gru) - base
    print(f"for reference, WITHOUT the chain (student + 0.5 gru): {nochain.mean():+.5f}")


if __name__ == "__main__":
    main()


