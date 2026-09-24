"""Select the Submission32 GRU-ensemble weight (4 configs incl. the whitening-only
Model127 GRU) on pilot folds 0-1; confirm on 2-4.  Writes constants_gru32.json.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.model10.sources import load_development_layout  # noqa: E402
from tsauc import tsauc  # noqa: E402

HERE = Path(__file__).resolve().parent
# (prediction pattern, weights pattern, layers, contract index: 0 = 102 inputs, 1 = 62 inputs)
CONFIGS = [
    ("models/model124/remote_runs/fold{k}_pred.npz", "models/model124/remote_runs/fold{k}_weights.npz", 2, 0),
    ("models/model124/remote_runs/h96l3_fold{k}_pred.npz", "models/model124/remote_runs/h96l3_fold{k}_weights.npz", 3, 0),
    ("models/model124/remote_runs/h128_fold{k}_pred.npz", "models/model124/remote_runs/h128_fold{k}_weights.npz", 2, 0),
    ("models/model127/runs/d0_fold{k}_pred.npz", "models/model127/runs/d0_fold{k}_weights.npz", 3, 1),
]
GRID = (0.25, 0.35, 0.5, 0.65, 0.8, 1.0)
PILOT, CONFIRM = (0, 1), (2, 3, 4)


def lg(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def main():
    L = load_development_layout(ROOT)
    c = json.loads((ROOT / "models/model112/constants.json").read_text())
    chain = (np.load(ROOT / "models/model112/chain_m73_all.npy").astype(np.float64) - c["mu_chain"]) / c["sd_chain"]
    dist = np.empty(L.row_count); gru = np.zeros(L.row_count)
    for k in range(5):
        m = L.row_fold == k
        dist[m] = lg(np.load(ROOT / f"models/model111/runs/a050_fold{k}.npz")["prediction"])
        for pred, _, _, _ in CONFIGS:
            g = np.load(ROOT / pred.format(k=k))
            assert np.array_equal(g["held_rows"], np.flatnonzero(m))
            gru[m] += lg(g["prediction"]) / len(CONFIGS)
    sub30 = chain + c["weight"] * (dist - c["mu_dist"]) / c["sd_dist"]
    mu_g, sd_g = float(gru.mean()), float(gru.std())
    zg = (gru - mu_g) / sd_g

    def fold_score(v, k):
        m = L.row_fold == k
        return tsauc(L.target[m], v[m], L.time_online[m])

    base = np.array([fold_score(sub30, k) for k in range(5)])
    raw = np.array([fold_score(zg, k) for k in range(5)])
    table = {w: np.array([fold_score(sub30 + w * zg, k) for k in range(5)]) for w in GRID}
    for w, s in table.items():
        print(f"w={w}: folds {np.round(s, 5).tolist()} mean {s.mean():.6f} gain {s.mean() - base.mean():+.6f}", flush=True)
    pilot_gain = {w: float((s - base)[list(PILOT)].mean()) for w, s in table.items()}
    weight = max(pilot_gain, key=pilot_gain.get)
    gains = table[weight] - base
    sub31 = json.loads((HERE / "constants_gru.json").read_text())
    summary = {
        "configs": [{"pred": p, "weights": w_, "layers": l, "contract": ci} for p, w_, l, ci in CONFIGS],
        "weight_gru": weight, "mu_gru": mu_g, "sd_gru": sd_g,
        "gru_raw_fold_scores": raw.tolist(), "gru_raw_mean": float(raw.mean()),
        "sub30_fold_scores": base.tolist(), "sub30_mean": float(base.mean()),
        "candidate_fold_scores": table[weight].tolist(), "candidate_mean": float(table[weight].mean()),
        "pilot_gain": float(gains[list(PILOT)].mean()), "confirmation_gain": float(gains[list(CONFIRM)].mean()),
        "confirmation_wins": int((gains[list(CONFIRM)] > 0).sum()),
        "submission31_candidate_mean": sub31["candidate_mean"],
        "gain_vs_submission31_dev": float(table[weight].mean() - sub31["candidate_mean"]),
        "pilot_gain_by_weight": pilot_gain,
    }
    (HERE / "constants_gru32.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("pilot_gain_by_weight", "configs")}, indent=2))


if __name__ == "__main__":
    main()
