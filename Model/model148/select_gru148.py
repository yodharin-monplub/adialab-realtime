"""Constants of the Model148 GRU ensemble (three sizes, 129 inputs) for the top-level blend.

gru148 = per-row mean logit of the configs below (out-of-fold); mu / sd over all development rows
(label-free), weight 0.5 as in Submission31 (weights_eval.py: .5/.5 is still right).  Also reports the
dev value of the full blend against the Submission33 structure (old GRU set).
Writes models/model148/constants_gru148.json and runs/gru148_fold{k}.npz.
"""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.model10.sources import load_development_layout  # noqa: E402
from tsauc import tsauc  # noqa: E402

HERE = Path(__file__).resolve().parent
OLD = ROOT / "models/model124/remote_runs"
CONFIGS = [{"tag": "h192s_", "hidden": 192, "layers": 2}, {"tag": "h96l3s_", "hidden": 96, "layers": 3},
           {"tag": "h128s_", "hidden": 128, "layers": 2}]


def lg(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def main():
    L = load_development_layout(ROOT)
    (HERE / "runs").mkdir(exist_ok=True)
    c = json.loads((ROOT / "models/model112/constants.json").read_text())
    cb = json.loads((ROOT / "models/model112/constants_m147_bag3.json").read_text())
    chain = (np.load(ROOT / "models/model112/chain_m73_all.npy").astype(np.float64) - c["mu_chain"]) / c["sd_chain"]
    new = np.empty(L.row_count); old = np.empty(L.row_count); student = np.empty(L.row_count); raw = {}
    for k in range(5):
        m = L.row_fold == k
        parts = []
        for cfg in CONFIGS:
            d = np.load(HERE / f"{cfg['tag']}fold{k}_pred.npz"); assert np.array_equal(d["held_rows"], np.flatnonzero(m))
            parts.append(lg(d["prediction"])); raw.setdefault(cfg["tag"], []).append(float(d["score"]))
        new[m] = np.mean(parts, axis=0)
        old[m] = np.mean([lg(np.load(OLD / f"{t}fold{k}_pred.npz")["prediction"]) for t in ("", "h96l3_", "h128_")], axis=0)
        student[m] = (lg(np.load(ROOT / f"models/model147/runs/BAG3_fold{k}.npz")["prediction"]) - cb["mu_dist"]) / cb["sd_dist"]
        np.savez(HERE / f"runs/gru148_fold{k}.npz", prediction=(1 / (1 + np.exp(-new[m]))).astype(np.float32), held_rows=np.flatnonzero(m))
    mu, sd = float(new.mean()), float(new.std())
    zo = (old - old.mean()) / old.std(); zn = (new - mu) / sd
    rows = []
    for k in range(5):
        m = L.row_fold == k; y, t = L.target[m], L.time_online[m]
        base = chain[m] + 0.5 * student[m]
        rows.append((tsauc(y, old[m], t), tsauc(y, new[m], t), tsauc(y, base + 0.5 * zo[m], t), tsauc(y, base + 0.5 * zn[m], t)))
    r = np.array(rows); gain = r[:, 3] - r[:, 2]
    print("raw GRU ensemble  old", np.round(r[:, 0], 4).tolist(), "mean", round(r[:, 0].mean(), 4))
    print("raw GRU ensemble  new", np.round(r[:, 1], 4).tolist(), "mean", round(r[:, 1].mean(), 4))
    print("full blend        old", np.round(r[:, 2], 5).tolist(), "mean", round(r[:, 2].mean(), 5))
    print("full blend        new", np.round(r[:, 3], 5).tolist(), "mean", round(r[:, 3].mean(), 5))
    print(f"gain {np.round(gain, 5).tolist()} mean {gain.mean():+.5f} s.e. {gain.std(ddof=1) / 5 ** .5:.5f} wins {(gain > 0).sum()}/5")
    out = {"configs": CONFIGS, "mu_gru": mu, "sd_gru": sd, "weight_gru": 0.5, "raw_scores": raw,
           "blend_old": r[:, 2].tolist(), "blend_new": r[:, 3].tolist(), "gain_mean": float(gain.mean()), "wins": int((gain > 0).sum())}
    (HERE / "constants_gru148.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
