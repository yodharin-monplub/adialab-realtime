"""Model129 stage 2b: nested GRU for (held fold k, training fold j).

Trains on series of folds outside {k, k+1, j} with the fold-k soft targets
(which never used fold k), then predicts: rows of fold j (student features),
rows of fold k (held evaluation) and the synthetic sequences of fold k's plan.
Runs next to train_gru.py, tsauc.py, sequences.npz and syn_fold{k}.npz.

python train_gru_nested.py --k 0 --j 2 [--hidden 96 --layers 3 --epochs 24 --lr 1e-3]
Writes nested_k{k}_j{j}.npz: pred_j, rows_j, pred_k, rows_k, pred_syn
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from train_gru import GRUStudent, batches, pad, series_index
from tsauc import time_pair_weights, tsauc


def predict_rows(model, Xf, starts, ends, series, dev, n_rows):
    out = np.zeros(n_rows, np.float32)
    dummy = np.zeros(n_rows, np.float32)
    with torch.no_grad():
        for idx in [series[i:i + 128] for i in range(0, len(series), 128)]:
            xb, _, _ = pad(Xf, dummy, dummy, starts, ends, idx, dev)
            o = model(xb).cpu().numpy()  # logits
            for i, s in enumerate(idx):
                out[starts[s]:ends[s]] = o[i, :ends[s] - starts[s]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sequences.npz")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--j", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    k, j = args.k, args.j
    assert j not in (k, (k + 1) % 5)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(args.data)
    X = d["X"]; soft = d["soft"][k].astype(np.float32); y = d["target"].astype(np.float32)
    t = d["time_online"].astype(np.int64); fold = d["row_fold"]; pos = d["series_position"]
    starts, ends = series_index(pos)
    sfold = fold[starts]
    excluded = {k, (k + 1) % 5, j}
    train_series = np.flatnonzero(~np.isin(sfold, list(excluded)))
    train_rows = np.flatnonzero(~np.isin(fold, list(excluded)))
    assert np.isfinite(soft[train_rows]).all()
    w = np.zeros(len(y), np.float32)
    w[train_rows] = time_pair_weights(y[train_rows].astype(np.int8), t[train_rows])
    Xf = X.astype(np.float32)
    model = GRUStudent(X.shape[1], args.hidden, args.layers).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * (len(train_series) // args.batch + 1))
    rng = np.random.default_rng(1290 + 10 * k + j)
    held_series = np.flatnonzero(sfold == k)
    held_rows = np.flatnonzero(fold == k)
    best = (-1.0, None)
    for epoch in range(args.epochs):
        model.train(); t0 = time.time()
        for idx in batches(train_series, starts, ends, args.batch, rng):
            xb, yb, wb = pad(Xf, soft, w, starts, ends, idx, dev)
            loss = (nn.functional.binary_cross_entropy_with_logits(model(xb), yb, reduction="none") * wb).sum() / wb.sum()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        model.eval()
        pk = predict_rows(model, Xf, starts, ends, held_series, dev, len(y))
        score = tsauc(y[held_rows].astype(np.int8), pk[held_rows], t[held_rows])
        if score > best[0]:
            best = (score, {n: p.detach().cpu().clone() for n, p in model.state_dict().items()})
        if epoch % 6 == 5 or epoch == args.epochs - 1:
            print(f"k{k} j{j} epoch {epoch}: held-k TS-AUC {score:.5f} ({time.time() - t0:.0f}s)", flush=True)
    model.load_state_dict(best[1]); model.eval()
    j_series = np.flatnonzero(sfold == j); j_rows = np.flatnonzero(fold == j)
    pred_all = predict_rows(model, Xf, starts, ends, np.concatenate([j_series, held_series]), dev, len(y))
    syn = np.load(f"syn_fold{k}.npz")
    Xs = syn["X"].astype(np.float32); s_starts = syn["starts"]; s_ends = s_starts + syn["lengths"]
    pred_syn = predict_rows(model, Xs, s_starts, s_ends, np.arange(len(s_starts)), dev, len(Xs))
    np.savez(f"nested_k{k}_j{j}.npz", pred_j=pred_all[j_rows], rows_j=j_rows, pred_k=pred_all[held_rows],
             rows_k=held_rows, pred_syn=pred_syn, best_held_k=best[0])
    print(f"k{k} j{j}: best held-k TS-AUC {best[0]:.5f}; saved nested_k{k}_j{j}.npz", flush=True)


if __name__ == "__main__":
    main()
