"""Model161 step 2: the deployed GRU recipe (models/model124/train_gru.py, unchanged training loop and best-
held-epoch selection) trained on the oracle-teacher soft targets softB instead of the old teacher's.
Usage: python models/model161/train_gru_b.py FOLD [--hidden 96 --layers 3 --epochs 24 --lr 1e-3 --batch 48 --tag h96l3B_]
Writes models/model161/runs/{tag}fold{k}_weights.npz / _pred.npz.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from models.model124.train_gru import GRUStudent, batches, pad, series_index  # noqa: E402
from tsauc import time_pair_weights, tsauc  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fold", type=int)
    ap.add_argument("--hidden", type=int, default=96); ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=24); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=48); ap.add_argument("--tag", default="h96l3B_")
    a = ap.parse_args(); k = a.fold
    out = HERE / "runs" / f"{a.tag}fold{k}_weights.npz"
    if out.exists():
        print("exists", out); return
    (HERE / "runs").mkdir(exist_ok=True)
    d = np.load(ROOT / "models/model124/data/sequences.npz")
    X = d["X"].astype(np.float32); y = d["target"].astype(np.float32)
    soft = np.load(HERE / "data/softB.npy")[k].astype(np.float32)
    t = d["time_online"].astype(np.int64); fold = d["row_fold"]; pos = d["series_position"]
    starts, ends = series_index(pos); sfold = fold[starts]
    train_series = np.flatnonzero((sfold != k) & (sfold != (k + 1) % 5)); held_series = np.flatnonzero(sfold == k)
    train_rows = np.flatnonzero((fold != k) & (fold != (k + 1) % 5))
    assert np.isfinite(soft[train_rows]).all()
    w = np.zeros(len(y), np.float32); w[train_rows] = time_pair_weights(y[train_rows].astype(np.int8), t[train_rows])
    torch.manual_seed(124 + k)
    model = GRUStudent(X.shape[1], a.hidden, a.layers)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * (len(train_series) // a.batch + 1))
    rng = np.random.default_rng(124 + k); best = (-1.0, None, None)
    held_rows = np.flatnonzero(fold == k)
    for epoch in range(a.epochs):
        model.train(); t0 = time.time(); total = 0.0
        for idx in batches(train_series, starts, ends, a.batch, rng):
            xb, yb, wb = pad(X, soft, w, starts, ends, idx, "cpu")
            loss = (nn.functional.binary_cross_entropy_with_logits(model(xb), yb, reduction="none") * wb).sum() / wb.sum()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); total += float(loss.detach())
        model.eval(); preds = np.zeros(len(y), np.float32)
        with torch.no_grad():
            for idx in [held_series[i:i + 128] for i in range(0, len(held_series), 128)]:
                xb, _, _ = pad(X, soft, w, starts, ends, idx, "cpu")
                o = torch.sigmoid(model(xb)).numpy()
                for i, s in enumerate(idx):
                    preds[starts[s]:ends[s]] = o[i, :ends[s] - starts[s]]
        score = tsauc(y[held_rows].astype(np.int8), preds[held_rows], t[held_rows])
        print(f"fold {k} epoch {epoch}: loss {total:.3f} held TS-AUC {score:.6f} ({time.time() - t0:.0f}s)", flush=True)
        if score > best[0]:
            best = (score, preds[held_rows].copy(), {n: p.detach().numpy() for n, p in model.state_dict().items()})
    np.savez(HERE / "runs" / f"{a.tag}fold{k}_pred.npz", prediction=best[1], held_rows=held_rows, score=best[0])
    np.savez(out, **best[2])
    print(f"fold {k} best held TS-AUC {best[0]:.6f}", flush=True)


if __name__ == "__main__":
    main()
