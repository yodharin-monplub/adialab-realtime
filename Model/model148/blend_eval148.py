"""What is the split-input GRU worth inside the full blend?   (dev folds, OOF)

blend = z(chain) + 0.5 * z(student logit) + 0.5 * z(GRU ensemble logit)
  student : Model147 BAG3 (the Submission33 student)
  GRU sets compared (logit mean of the listed configs, per fold):
     old   = h192 (no tag), h96l3_, h128_            -> Submission31 / 33
     swap  = h192, h96l3s_ (Model148), h128_         -> same size, one member replaced
     add   = old + h96l3s_
Reports per fold and mean for every available fold.  Usage: python models/model148/blend_eval148.py
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


def lg(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def z(v):
    return (v - v.mean()) / v.std()


def main():
    L = load_development_layout(ROOT)
    c = json.loads((ROOT / "models/model112/constants.json").read_text())
    cb = json.loads((ROOT / "models/model112/constants_m147_bag3.json").read_text())
    chain = (np.load(ROOT / "models/model112/chain_m73_all.npy").astype(np.float64) - c["mu_chain"]) / c["sd_chain"]
    rows = []
    for k in range(5):
        new = NEW / f"h96l3s_fold{k}_pred.npz"
        if not new.exists():
            continue
        m = L.row_fold == k; y, t = L.target[m], L.time_online[m]
        student = (lg(np.load(ROOT / f"models/model147/runs/BAG3_fold{k}.npz")["prediction"]) - cb["mu_dist"]) / cb["sd_dist"]
        g = {tag: lg(np.load(OLD / f"{tag}fold{k}_pred.npz")["prediction"]) for tag in ("", "h96l3_", "h128_")}
        g["h96l3s_"] = lg(np.load(new)["prediction"])
        sets = {"old": ["", "h96l3_", "h128_"], "swap": ["", "h96l3s_", "h128_"], "add": ["", "h96l3_", "h128_", "h96l3s_"]}
        # upgraded sizes that exist so far replace their old counterparts ("swap_all"); "six" keeps old + new
        pairs = {"h96l3_": "h96l3s_", "h128_": "h128s_", "": "h192s_"}
        for new_tag in ("h128s_", "h192s_"):
            f = NEW / f"{new_tag}fold{k}_pred.npz"
            if f.exists():
                g[new_tag] = lg(np.load(f)["prediction"])
        sets["swap_all"] = [pairs[t] if pairs[t] in g else t for t in ("", "h96l3_", "h128_")]
        sets["six"] = ["", "h96l3_", "h128_"] + [v for v in pairs.values() if v in g]
        base = chain[m] + 0.5 * student
        r = {"fold": k, "raw_old_h96l3": tsauc(y, g["h96l3_"], t), "raw_new_h96l3s": tsauc(y, g["h96l3s_"], t), "no_gru": tsauc(y, base, t)}
        for name, tags in sets.items():
            r[name] = tsauc(y, base + 0.5 * z(np.mean([g[tag] for tag in tags], axis=0)), t)
        rows.append(r)
        print(f"fold {k}: raw GRU {r['raw_old_h96l3']:.4f} -> {r['raw_new_h96l3s']:.4f} | blend old {r['old']:.5f}  "
              f"swap {r['swap'] - r['old']:+.5f}  add {r['add'] - r['old']:+.5f}  swap_all {r['swap_all'] - r['old']:+.5f} "
              f"({'+'.join(sets['swap_all'])})  six {r['six'] - r['old']:+.5f}", flush=True)
    if rows:
        for name in ("swap", "add", "swap_all", "six"):
            d = np.array([r[name] - r["old"] for r in rows])
            print(f"{name}: mean {d.mean():+.5f}  wins {(d > 0).sum()}/{len(d)}")


if __name__ == "__main__":
    main()
