"""Deterministic training-only donors and mass-preserving augmentation weights."""
from collections import Counter

import numpy as np

from tsauc import time_pair_weights


def real_fit_rows(row_fold, fold, limit=500_000):
    eligible = np.flatnonzero(row_fold != fold)
    return np.sort(np.random.default_rng(20260906 + fold).choice(
        eligible, min(limit, len(eligible)), replace=False))


def pre_median(history, online, tau):
    return float(np.median(np.concatenate((history, online[:tau]))[-128:]))


def select_pairs(pool, tau, series_fold, held_fold, seed, count=1000, reuse=5):
    """Pool maps global series positions to (history, online), not feature rows.

    Reject nonfinite sources before drawing. B must have a genuine postbreak
    segment long enough to fill A's original horizon. A is never repeated.
    """
    rng = np.random.default_rng(seed)
    valid = []
    for position in sorted(pool):
        h, o = pool[position]
        if series_fold[position] < 0 or series_fold[position] == held_fold:
            raise ValueError("Donor pool includes validation or reserved series")
        if (len(h) >= 3 and 0 <= tau[position] < len(o)
                and np.isfinite(h).all() and np.isfinite(o).all()):
            valid.append(position)
    eligible = [p for p in valid if tau[p] >= 32 and len(pool[p][1]) - tau[p] >= 32]
    usage = Counter()
    pairs = []
    for a in rng.permutation(eligible):
        a = int(a)
        remaining = len(pool[a][1]) - int(tau[a])
        choices = [b for b in valid if b != a and usage[b] < reuse
                   and len(pool[b][1]) - tau[b] >= remaining]
        if not choices:
            continue
        b = int(rng.choice(choices))
        ah, ao = pool[a]
        bh, bo = pool[b]
        shift = pre_median(ah, ao, tau[a]) - pre_median(bh, bo, tau[b])
        # Validate the actual float32 assembly before accepting either arm.
        with np.errstate(over="ignore", invalid="ignore"):
            aligned = (bo[tau[b]:tau[b] + remaining] + shift).astype(np.float32)
            raw = bo[tau[b]:tau[b] + remaining].astype(np.float32)
            prefix = np.concatenate((ah, ao[:tau[a]])).astype(np.float32)
        if not all(np.isfinite(v).all() for v in (aligned, raw, prefix)):
            continue
        pairs.append(dict(a=a, b=b, tau_a=int(tau[a]), tau_b=int(tau[b]),
                          length=len(ao), remaining=remaining, offset=shift))
        usage[b] += 1
        if len(pairs) == count:
            break
    return pairs, dict(requested=count, accepted=len(pairs), shortfall=count-len(pairs),
                       eligible_a=len(eligible), finite_break_donors=len(valid),
                       max_b_reuse=max(usage.values(), default=0))


def assemble(pool, pair, aligned=False):
    ah, ao = pool[pair["a"]]
    _, bo = pool[pair["b"]]
    tau = pair["tau_a"]
    tail = bo[pair["tau_b"]:pair["tau_b"] + pair["remaining"]]
    if aligned:
        tail = tail + pair["offset"]
    online = np.concatenate((ao[:tau], tail)).astype(np.float32)
    history = np.asarray(ah, dtype=np.float32)
    if len(online) != pair["length"] or not np.isfinite(online).all() or not np.isfinite(history).all():
        raise ValueError("Invalid composite length or values")
    steps = np.arange(len(online), dtype=np.int32)
    return history, online, (steps >= tau).astype(np.uint8), steps


def augmented_weights(real_y, real_t, synthetic_y, synthetic_t, share):
    if not 0 < share < 1:
        raise ValueError("Synthetic share must lie strictly between zero and one")
    original = time_pair_weights(real_y, real_t)
    real_cell = 2 * np.asarray(real_t, dtype=np.int64) + real_y
    syn_cell = 2 * np.asarray(synthetic_t, dtype=np.int64) + synthetic_y
    size = int(max(real_cell.max(initial=0), syn_cell.max(initial=0))) + 1
    mass = np.bincount(real_cell, weights=original, minlength=size)
    counts = np.bincount(syn_cell, minlength=size)
    real = (original * np.where(counts[real_cell] > 0, 1-share, 1)).astype(np.float32)
    per_cell = np.divide(share * mass, counts, out=np.zeros(size), where=counts > 0)
    synthetic = per_cell[syn_cell].astype(np.float32)
    final = (np.bincount(real_cell, weights=real, minlength=size)
             + np.bincount(syn_cell, weights=synthetic, minlength=size))
    np.testing.assert_allclose(final, mass, rtol=2e-7, atol=1e-6)
    return real, synthetic, dict(nominal_share=share,
        actual_share=float(synthetic.sum(dtype=np.float64) / original.sum(dtype=np.float64)),
        original_total=float(original.sum(dtype=np.float64)),
        final_total=float(final.sum()), max_cell_error=float(np.max(np.abs(final-mass))),
        synthetic_zero_weight_rows=int(np.sum(synthetic == 0)))
