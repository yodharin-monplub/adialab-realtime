"""Model172 remote trainer: the neural student on REAL + SYNTHETIC sequences.

Same network and loop as models/model169/train_gpu.py (which is Model124's deployed recipe), but the
training set is several sequence files:
  --data real.npz,synth1.npz,...   the first file is the real one, the rest are synthetic
  --synth-share 0.4                synthetic rows carry this fraction of the total sample weight,
                                   mirroring the tree student's 40% synthetic mass
A synthetic series is excluded from a fold's training set when its row_fold is the held fold k or the
stack-reserved fold (k+1) % 5 - exactly the rule the tree student uses for splices and re-cuts.
Epoch selection is still on the held development fold k (real rows only), so the comparison with
Model169/170 stays honest.
  python train_gpu2.py --fold 0 --hidden 256 --epochs 24 --data sequences_all.npz,seq_recuts.npz,seq_splices.npz
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from tsauc import time_pair_weights, tsauc
from train_gpu import GRUStudent, batches, pad, series_index


def load_many(paths, soft_path):
    """-> X, soft(per held fold), target, time, row_fold, series_position, is_real"""
    Xs, softs, ys, ts, folds, pos, real = [], [], [], [], [], [], []
    offset = 0
    for i, p in enumerate(paths):
        d = np.load(p)
        X = d["X"]
        Xs.append(X)
        ys.append(d["target"].astype(np.int8)); ts.append(d["time_online"].astype(np.int64))
        folds.append(d["row_fold"].astype(np.int8))
        sp = d["series_position"].astype(np.int64)
        pos.append(sp - sp.min() + offset); offset = pos[-1].max() + 1
        real.append(np.full(len(X), i == 0, bool))
        if i == 0:
            softs.append(np.load(soft_path) if soft_path else d["soft"])     # (5, rows)
        else:
            softs.append(np.repeat(d["soft"][None, :], 5, axis=0))
        print(f"  {p}: {len(X):,} rows, {len(np.unique(sp)):,} series", flush=True)
    return (np.concatenate(Xs), np.concatenate(softs, axis=1), np.concatenate(ys), np.concatenate(ts),
            np.concatenate(folds), np.concatenate(pos), np.concatenate(real))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--hidden", type=int, default=256); ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=24); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=48); ap.add_argument("--tag", default="mix_")
    ap.add_argument("--data", required=True); ap.add_argument("--soft", default="soft_all.npy")
    ap.add_argument("--synth-share", type=float, default=0.4); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); k = a.fold
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", dev, torch.cuda.get_device_name(0) if dev.type == "cuda" else "", flush=True)
    X, soft_all, y, t, fold, pos, is_real = load_many(a.data.split(","), a.soft)
    X = X.astype(np.float32); soft = soft_all[k].astype(np.float32); yf = y.astype(np.float32)
    starts, ends = series_index(pos)
    sfold = fold[starts]; sreal = is_real[starts]
    keep_series = np.flatnonzero((sfold != k) & (sfold != (k + 1) % 5))           # holdout rows have fold -1
    keep_rows = np.zeros(len(y), bool)
    for s in keep_series:
        keep_rows[starts[s]:ends[s]] = True
    held_series = np.flatnonzero((sfold == k) & sreal)
    held_rows = np.flatnonzero((fold == k) & is_real)
    assert np.isfinite(soft[keep_rows]).all()
    # sample weights: real rows as deployed, synthetic scaled to carry `synth-share` of the total
    w = np.zeros(len(y), np.float32)
    for mask in (keep_rows & is_real, keep_rows & ~is_real):
        if mask.any():
            w[mask] = time_pair_weights(y[mask], t[mask])
    rw, sw = w[keep_rows & is_real].sum(), w[keep_rows & ~is_real].sum()
    if sw > 0 and a.synth_share > 0:
        w[keep_rows & ~is_real] *= (a.synth_share / (1 - a.synth_share)) * rw / sw
    print(f"fold {k}: {len(keep_series):,} training series ({sreal[keep_series].sum():,} real), "
          f"weight share synthetic {w[keep_rows & ~is_real].sum() / w[keep_rows].sum():.3f}", flush=True)
    torch.manual_seed(1000 + 17 * k + a.seed)
    model = GRUStudent(X.shape[1], a.hidden, a.layers).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * (len(keep_series) // a.batch + 1))
    rng = np.random.default_rng(1000 + 17 * k + a.seed); best = (-1.0, None, None)
    for epoch in range(a.epochs):
        model.train(); t0 = time.time(); total = 0.0
        for idx in batches(keep_series, starts, ends, a.batch, rng):
            xb, yb, wb = pad(X, soft, w, starts, ends, idx, dev)
            if float(wb.sum()) <= 0:
                continue
            loss = (nn.functional.binary_cross_entropy_with_logits(model(xb), yb, reduction="none") * wb).sum() / wb.sum()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); total += float(loss.detach())
        model.eval(); preds = np.zeros(len(y), np.float32)
        with torch.no_grad():
            for idx in [held_series[i:i + 128] for i in range(0, len(held_series), 128)]:
                xb, _, _ = pad(X, soft, w, starts, ends, idx, dev)
                o = torch.sigmoid(model(xb)).cpu().numpy()
                for i, s in enumerate(idx):
                    preds[starts[s]:ends[s]] = o[i, :ends[s] - starts[s]]
        score = tsauc(y[held_rows], preds[held_rows], t[held_rows])
        print(f"{a.tag}fold {k} epoch {epoch}: loss {total:.3f} held TS-AUC {score:.6f} ({time.time() - t0:.0f}s)", flush=True)
        if score > best[0]:
            best = (score, preds[held_rows].copy(), {n: p.detach().cpu().numpy() for n, p in model.state_dict().items()})
    np.savez(f"{a.tag}fold{k}_pred.npz", prediction=best[1], held_rows=held_rows, score=best[0])
    np.savez(f"{a.tag}fold{k}_weights.npz", **best[2])
    print(f"{a.tag}fold {k} best held TS-AUC {best[0]:.6f}", flush=True)


if __name__ == "__main__":
    main()
