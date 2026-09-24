"""Model143 trainer: end-to-end causal sequence model on the RAW series (PyTorch, GPU or CPU).

python train_tcn.py --data raw.npz --fold 0 [--epochs 12] [--channels 64] [--blocks 8] [--hidden 96]
History encoder: causal TCN over the last 512 standardized history points -> context vector.
Online model: [4 raw channels + context] -> dilated causal TCN (receptive field 2^blocks-1 steps)
-> GRU -> per-step logit.  Soft targets 0.5*y + 0.5*teacher (Model111 teachers), time-pair weights,
same folds as the tree student (fold k held, k+1 reserved).  Writes {tag}fold{k}_pred.npz /
{tag}fold{k}_weights.npz next to this file; tsauc.py must sit beside it.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from tsauc import time_pair_weights, tsauc


class CausalBlock(nn.Module):
    def __init__(self, ch, dilation, drop):
        super().__init__()
        self.pad = 2 * dilation
        self.conv1 = nn.Conv1d(ch, ch, 3, dilation=dilation)
        self.conv2 = nn.Conv1d(ch, ch, 3, dilation=dilation)
        # per-step normalization over channels only: GroupNorm(1, ch) on (B, C, T) would normalize
        # across the whole time axis and leak future statistics into every step (found 03:30)
        self.norm = nn.LayerNorm(ch)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()

    def forward(self, x):  # x (B, C, T)
        h = self.act(self.conv1(nn.functional.pad(x, (self.pad, 0))))
        h = self.drop(self.conv2(nn.functional.pad(h, (self.pad, 0))))
        return self.norm(self.act(x + h).transpose(1, 2)).transpose(1, 2)


class TCN(nn.Module):
    def __init__(self, inputs, ch, blocks, drop):
        super().__init__()
        self.inp = nn.Conv1d(inputs, ch, 1)
        self.blocks = nn.ModuleList([CausalBlock(ch, 2 ** i, drop) for i in range(blocks)])

    def forward(self, x):  # (B, T, inputs) -> (B, T, ch)
        h = self.inp(x.transpose(1, 2))
        for b in self.blocks:
            h = b(h)
        return h.transpose(1, 2)


class RawStudent(nn.Module):
    def __init__(self, ch, blocks, hidden, ctx, inputs=4, drop=0.1):
        super().__init__()
        self.hist = TCN(2, 32, 5, drop)
        self.ctx = nn.Linear(64, ctx)
        self.online = TCN(inputs + ctx, ch, blocks, drop)
        self.gru = nn.GRU(ch, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

    def forward(self, x, hist):  # x (B, T, 4), hist (B, 512, 2) [value, mask]
        e = self.hist(hist)                                   # (B, 512, 32)
        mask = hist[:, :, 1:2]
        pooled = (e * mask).sum(1) / mask.sum(1).clamp(min=1)  # masked mean
        c = self.ctx(torch.cat((pooled, e[:, -1]), 1))         # (B, ctx)
        z = torch.cat((x, c[:, None, :].expand(-1, x.shape[1], -1)), 2)
        h, _ = self.gru(self.online(z))
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


def pad(X, H, y, w, starts, ends, idx, device):
    lengths = ends[idx] - starts[idx]
    T = int(lengths.max())
    xb = np.zeros((len(idx), T, X.shape[1]), np.float32)
    yb = np.zeros((len(idx), T), np.float32)
    wb = np.zeros((len(idx), T), np.float32)
    for i, s in enumerate(idx):
        a, b = starts[s], ends[s]
        xb[i, :b - a] = X[a:b]; yb[i, :b - a] = y[a:b]; wb[i, :b - a] = w[a:b]
    hb = H[idx]
    return (torch.from_numpy(xb).to(device), torch.from_numpy(hb).to(device),
            torch.from_numpy(yb).to(device), torch.from_numpy(wb).to(device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="raw.npz")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--ctx", type=int, default=32)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--extra", default="", help="npz with X (rows, d) float16 to concatenate per step, e.g. Model124's sequences.npz")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(args.data)
    X = d["X"].astype(np.float32); soft = d["soft"][args.fold].astype(np.float32); y = d["target"].astype(np.float32)
    if args.extra:
        X = np.concatenate((X, np.load(args.extra)["X"].astype(np.float32)), axis=1)
        print(f"inputs per step: {X.shape[1]} (4 raw + {X.shape[1] - 4} extra)", flush=True)
    t = d["time_online"].astype(np.int64); fold = d["row_fold"]; pos = d["series_position"]
    Hv = d["H"].astype(np.float32); hl = d["hist_len"].astype(np.int64)
    mask = (np.arange(Hv.shape[1])[None, :] >= (Hv.shape[1] - hl)[:, None]).astype(np.float32)
    H = np.stack((Hv, mask), 2)                     # (series, 512, 2)
    starts, ends = series_index(pos)
    sfold = fold[starts]
    k = args.fold
    train_series = np.flatnonzero((sfold != k) & (sfold != (k + 1) % 5))
    held_series = np.flatnonzero(sfold == k)
    train_rows = np.flatnonzero((fold != k) & (fold != (k + 1) % 5))
    assert np.isfinite(soft[train_rows]).all()
    w = np.zeros(len(y), np.float32)
    w[train_rows] = time_pair_weights(y[train_rows].astype(np.int8), t[train_rows])
    model = RawStudent(args.channels, args.blocks, args.hidden, args.ctx, inputs=X.shape[1]).to(dev)
    print(f"parameters {sum(p.numel() for p in model.parameters()):,} device {dev}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * (len(train_series) // args.batch + 1))
    rng = np.random.default_rng(143 + k)
    best = (-1.0, None, None)
    for epoch in range(args.epochs):
        model.train(); t0 = time.time(); total = 0.0
        for idx in batches(train_series, starts, ends, args.batch, rng):
            xb, hb, yb, wb = pad(X, H, soft, w, starts, ends, idx, dev)
            logits = model(xb, hb)
            loss = (nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="none") * wb).sum() / wb.sum()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); total += float(loss)
        model.eval(); preds = np.zeros(len(y), np.float32)
        with torch.no_grad():
            for idx in [held_series[i:i + 64] for i in range(0, len(held_series), 64)]:
                xb, hb, _, _ = pad(X, H, soft, w, starts, ends, idx, dev)
                out = torch.sigmoid(model(xb, hb)).cpu().numpy()
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
