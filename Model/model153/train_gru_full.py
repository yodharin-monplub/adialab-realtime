"""Model153: full-development refit of the Submission31 GRU students (Model124 recipe).

Model152 showed the tree student is series-limited (refit on all dev series: holdout +0.0013).  Same idea
for the GRU (weight 0.5 in the blend): train each config on ALL 8,333 dev series instead of 3/5, with soft
targets 0.5*y + 0.5*teacher from Model152's cross-fitted 4-fold teachers (runs/full_teacher.npz).
No held fold -> the FINAL epoch is kept (OneCycle schedule ends at a small learning rate).
Configs (as deployed): h192 L2 16 ep lr 2e-3 b48 | h96 L3 24 ep lr 1e-3 b48 | h128 L2 24 ep lr 1e-3 b32.
Output: models/model153/runs/{config}_s{seed}_weights.npz (same keys as Model124 fold weights).
Usage: python models/model153/train_gru_full.py CONFIG [SEED]      CONFIG in h192 h96l3 h128
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from models.model124.train_gru import GRUStudent, batches, pad, series_index  # noqa: E402
from tsauc import time_pair_weights  # noqa: E402

CONFIGS = {"h192": dict(hidden=192, layers=2, epochs=16, lr=2e-3, batch=48),
           "h96l3": dict(hidden=96, layers=3, epochs=24, lr=1e-3, batch=48),
           "h128": dict(hidden=128, layers=2, epochs=24, lr=1e-3, batch=32)}
RUNS = Path(__file__).resolve().parent / "runs"


def main():
    name = sys.argv[1]; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    c = CONFIGS[name]
    out = RUNS / f"{name}_s{seed}_weights.npz"
    if out.exists():
        print("exists", out); return
    RUNS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(153 + seed)
    d = np.load(ROOT / "models/model124/data/sequences.npz")
    X = d["X"].astype(np.float32); y = d["target"].astype(np.float32)
    t = d["time_online"].astype(np.int64); fold = d["row_fold"]; pos = d["series_position"]
    ft = np.load(ROOT / "models/model152/runs/full_teacher.npz")
    assert np.array_equal(ft["rows"], np.arange(len(y))) and (fold >= 0).all()
    soft = (0.5 * y + 0.5 * ft["prob"]).astype(np.float32)
    starts, ends = series_index(pos)
    train_series = np.arange(len(starts))
    w = time_pair_weights(y.astype(np.int8), t).astype(np.float32)
    model = GRUStudent(X.shape[1], c["hidden"], c["layers"])
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=c["lr"], total_steps=c["epochs"] * (len(train_series) // c["batch"] + 1))
    rng = np.random.default_rng(1530 + seed)
    for epoch in range(c["epochs"]):
        model.train(); t0 = time.time(); total = 0.0
        for idx in batches(train_series, starts, ends, c["batch"], rng):
            xb, yb, wb = pad(X, soft, w, starts, ends, idx, "cpu")
            loss = (nn.functional.binary_cross_entropy_with_logits(model(xb), yb, reduction="none") * wb).sum() / wb.sum()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); total += float(loss)
        print(f"{name} s{seed} epoch {epoch}: loss {total:.3f} ({time.time() - t0:.0f}s)", flush=True)
    np.savez(out, **{n: p.detach().cpu().numpy() for n, p in model.state_dict().items()})
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
