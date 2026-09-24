"""Model131: bidirectional-GRU teacher (sees the whole online segment) for (k, j).

Trains on hard labels of series in folds outside {k, k+1, j} using the same
102 causal inputs as Model124 but a BIDIRECTIONAL GRU (future visible), and
writes sigmoid probabilities for the rows of fold j -> soft targets for the
Model111 student of held fold k.  Fold-k labels never reach it.
Runs next to train_gru.py, tsauc.py and sequences.npz.

python train_teacher_bigru.py --k 0 --j 2 [--hidden 96 --layers 2 --epochs 12 --lr 1e-3]
Writes teacher_k{k}_j{j}.npz: prob_j, rows_j, held_k_tsauc
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from train_gru import batches, pad, series_index
from tsauc import time_pair_weights, tsauc


class BiGRUTeacher(nn.Module):
    def __init__(self, inputs, hidden, layers):
        super().__init__()
        self.gru = nn.GRU(inputs, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                          dropout=0.1 if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x):
        h, _ = self.gru(x)
        return self.head(h).squeeze(-1)


def predict_rows(model, Xf, starts, ends, series, dev, n_rows):
    out = np.zeros(n_rows, np.float32)
    dummy = np.zeros(n_rows, np.float32)
    with torch.no_grad():
        for idx in [series[i:i + 128] for i in range(0, len(series), 128)]:
            xb, _, _ = pad(Xf, dummy, dummy, starts, ends, idx, dev)
            o = torch.sigmoid(model(xb)).cpu().numpy()
            for i, s in enumerate(idx):
                out[starts[s]:ends[s]] = o[i, :ends[s] - starts[s]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sequences.npz")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--j", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    k, j = args.k, args.j
    assert j not in (k, (k + 1) % 5)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(args.data)
    X = d["X"]; y = d["target"].astype(np.float32)
    t = d["time_online"].astype(np.int64); fold = d["row_fold"]; pos = d["series_position"]
    starts, ends = series_index(pos)
    sfold = fold[starts]
    excluded = [k, (k + 1) % 5, j]
    train_series = np.flatnonzero(~np.isin(sfold, excluded))
    train_rows = np.flatnonzero(~np.isin(fold, excluded))
    w = np.zeros(len(y), np.float32)
    w[train_rows] = time_pair_weights(y[train_rows].astype(np.int8), t[train_rows])
    Xf = X.astype(np.float32)
    model = BiGRUTeacher(X.shape[1], args.hidden, args.layers).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * (len(train_series) // args.batch + 1))
    rng = np.random.default_rng(1310 + 10 * k + j)
    j_series = np.flatnonzero(sfold == j); j_rows = np.flatnonzero(fold == j)
    best = (-1.0, None)
    for epoch in range(args.epochs):
        model.train(); t0 = time.time()
        for idx in batches(train_series, starts, ends, args.batch, rng):
            xb, yb, wb = pad(Xf, y, w, starts, ends, idx, dev)
            loss = (nn.functional.binary_cross_entropy_with_logits(model(xb), yb, reduction="none") * wb).sum() / wb.sum()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        model.eval()
        pj = predict_rows(model, Xf, starts, ends, j_series, dev, len(y))
        score = tsauc(y[j_rows].astype(np.int8), pj[j_rows], t[j_rows])   # teacher quality on its target fold
        if score > best[0]:
            best = (score, pj[j_rows].copy())
        print(f"teacher k{k} j{j} epoch {epoch}: fold-j TS-AUC {score:.5f} ({time.time() - t0:.0f}s)", flush=True)
    np.savez(f"teacher_k{k}_j{j}.npz", prob_j=best[1], rows_j=j_rows, fold_j_tsauc=best[0])
    print(f"teacher k{k} j{j}: best fold-j TS-AUC {best[0]:.5f}; saved", flush=True)


if __name__ == "__main__":
    main()
