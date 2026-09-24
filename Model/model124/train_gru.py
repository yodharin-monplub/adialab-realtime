"""Model124 remote trainer: causal GRU student on distilled targets (PyTorch, GPU).

python train_gru.py --data sequences.npz --fold 0 [--epochs 8] [--hidden 96]
Writes fold{k}_pred.npz (held-fold predictions) and fold{k}_weights.npz.
tsauc.py must sit next to this file.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from tsauc import time_pair_weights, tsauc


class GRUStudent(nn.Module):
    def __init__(self, inputs, hidden, layers):
        super().__init__()
        self.gru = nn.GRU(inputs, hidden, num_layers=layers, batch_first=True, dropout=0.1 if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

    def forward(self, x):
        h, _ = self.gru(x)
        return self.head(h).squeeze(-1)


def series_index(position):
    starts = np.flatnonzero(np.r_[True, position[1:] != position[:-1]])
    ends = np.r_[starts[1:], len(position)]
    return starts, ends


def batches(series, starts, ends, batch_size, rng):
    order = series[np.argsort(ends[series] - starts[series] + rng.uniform(0, 8, len(series)))]
    chunks = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    rng.shuffle(chunks)
    return chunks


def pad(X, y, w, starts, ends, idx, device):
    lengths = ends[idx] - starts[idx]
    T = int(lengths.max())
    xb = np.zeros((len(idx), T, X.shape[1]), np.float32)
    yb = np.zeros((len(idx), T), np.float32)
    wb = np.zeros((len(idx), T), np.float32)
    for i, s in enumerate(idx):
        a, b = starts[s], ends[s]
        xb[i, :b - a] = X[a:b]
        yb[i, :b - a] = y[a:b]
        wb[i, :b - a] = w[a:b]
    return (torch.from_numpy(xb).to(device), torch.from_numpy(yb).to(device), torch.from_numpy(wb).to(device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sequences.npz")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--tag", default="", help="output file prefix, e.g. h64_")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(args.data)
    X = d["X"]; soft = d["soft"][args.fold].astype(np.float32); y = d["target"].astype(np.float32)
    t = d["time_online"].astype(np.int64); fold = d["row_fold"]; pos = d["series_position"]
    starts, ends = series_index(pos)
    sfold = fold[starts]
    k = args.fold
    train_series = np.flatnonzero((sfold != k) & (sfold != (k + 1) % 5))
    held_series = np.flatnonzero(sfold == k)
    train_rows = np.flatnonzero((fold != k) & (fold != (k + 1) % 5))
    assert np.isfinite(soft[train_rows]).all()
    w = np.zeros(len(y), np.float32)
    w[train_rows] = time_pair_weights(y[train_rows].astype(np.int8), t[train_rows])
    model = GRUStudent(X.shape[1], args.hidden, args.layers).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * (len(train_series) // args.batch + 1))
    rng = np.random.default_rng(124 + k)
    Xf = X.astype(np.float32)
    best = (-1.0, None, None)
    for epoch in range(args.epochs):
        model.train(); t0 = time.time(); total = 0.0
        for idx in batches(train_series, starts, ends, args.batch, rng):
            xb, yb, wb = pad(Xf, soft, w, starts, ends, idx, dev)
            logits = model(xb)
            loss = (nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="none") * wb).sum() / wb.sum()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); total += float(loss)
        model.eval(); preds = np.zeros(len(y), np.float32)
        with torch.no_grad():
            for idx in [held_series[i:i + 128] for i in range(0, len(held_series), 128)]:
                xb, _, _ = pad(Xf, soft, w, starts, ends, idx, dev)
                out = torch.sigmoid(model(xb)).cpu().numpy()
                for i, s in enumerate(idx):
                    preds[starts[s]:ends[s]] = out[i, :ends[s] - starts[s]]
        held_rows = np.flatnonzero(fold == k)
        score = tsauc(y[held_rows].astype(np.int8), preds[held_rows], t[held_rows])
        print(f"fold {k} epoch {epoch}: loss {total:.3f} held TS-AUC {score:.6f} ({time.time() - t0:.0f}s)", flush=True)
        if score > best[0]:
            best = (score, preds[held_rows].copy(), {n: p.detach().cpu().numpy() for n, p in model.state_dict().items()})
    np.savez(f"{args.tag}fold{k}_pred.npz", prediction=best[1], held_rows=np.flatnonzero(fold == k), score=best[0])
    np.savez(f"{args.tag}fold{k}_weights.npz", **best[2])
    print(f"fold {k} best held TS-AUC {best[0]:.6f}", flush=True)


if __name__ == "__main__":
    main()
