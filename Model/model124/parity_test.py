"""Parity of the numpy GRU against PyTorch on real development series.

Usage: parity_test.py WEIGHTS_NPZ LAYERS [n_series]
Also checks that the exported per-step inputs can be rebuilt from the raw
feature caches with input_contract.json (the serving contract).
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models/model124"))
from gru_numpy import NumpyGRU  # noqa: E402
from train_gru import GRUStudent  # noqa: E402


def main(weights_path, layers, n_series=5):
    w = dict(np.load(weights_path))
    hidden = w["gru.weight_hh_l0"].shape[1]
    model = GRUStudent(w["gru.weight_ih_l0"].shape[1], hidden, layers)
    model.load_state_dict({k: torch.from_numpy(v) for k, v in w.items()})
    model.eval()
    npg = NumpyGRU(w, layers)
    d = np.load(ROOT / "models/model124/data/sequences.npz")
    X = d["X"]; pos = d["series_position"]
    starts = np.flatnonzero(np.r_[True, pos[1:] != pos[:-1]])
    ends = np.r_[starts[1:], len(pos)]
    contract = json.loads((ROOT / "models/model124/data/input_contract.json").read_text())
    real = np.load(ROOT / "models/model20/runs/82abe027caea808d/real_features.npy", mmap_mode="r")
    white = np.load(ROOT / "models/model57/runs/dfab0ede1c3206a4/real_white.npy", mmap_mode="r")
    t = d["time_online"]
    mu, sd = np.asarray(contract["mu"], np.float32), np.asarray(contract["sd"], np.float32)
    worst = 0.0; worst_in = 0.0
    for s in np.random.default_rng(0).choice(len(starts), n_series, replace=False):
        a, b = starts[s], ends[s]
        x = X[a:b].astype(np.float32)
        with torch.no_grad():
            ref = model(torch.from_numpy(x)[None])[0].numpy()
        out = npg.run(x)
        worst = max(worst, float(np.max(np.abs(out - ref))))
        # rebuild inputs from raw caches through the serving contract
        raw = np.hstack([np.asarray(white[a:b]), np.asarray(real[a:b][:, contract["detector_union_indices"]]),
                         np.log1p(t[a:b].astype(np.float32))[:, None]])
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
        rebuilt = np.clip((raw - mu) / sd, -contract["clip"], contract["clip"]).astype(np.float16).astype(np.float32)
        worst_in = max(worst_in, float(np.max(np.abs(rebuilt - x))))
    print(json.dumps({"series": int(n_series), "max_abs_logit_diff_numpy_vs_torch": worst,
                      "max_abs_input_rebuild_diff": worst_in, "hidden": int(hidden), "layers": int(layers)}))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 5)
