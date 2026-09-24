"""Family-matched, contrastive synthetic structural breaks.

Positive and negative composites share the same clean prefix and donor family.
The positive uses the donor's true post-break tail.  The negative uses a clean
historical donor segment.  Robust affine transport puts both tails in the
recipient's reference frame without erasing the positive donor's relative
post-break change.
"""
from collections import Counter

import numpy as np


CONTEXT_NAMES = (
    "ctx_log_history_length", "ctx_r1", "ctx_r2", "ctx_r5", "ctx_r12",
    "ctx_difference_scale", "ctx_abs_r1", "ctx_skew", "ctx_log_kurtosis",
    "ctx_tail_2_5", "ctx_sign_agreement", "ctx_trend", "ctx_low_frequency",
)


def _correlation(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3:
        return 0.0
    x = x-x.mean()
    y = y-y.mean()
    scale = np.sqrt(np.dot(x, x)*np.dot(y, y))
    return 0.0 if scale <= 1e-12 else float(np.dot(x, y)/scale)


def history_context(history):
    """Deployment-safe descriptors computed from clean history only."""
    raw = np.asarray(history, dtype=np.float64)
    if len(raw) < 32 or not np.isfinite(raw).all():
        raise ValueError("History must contain at least 32 finite observations")
    x = raw[-min(len(raw), 2048):]
    mean, sd = float(x.mean()), float(x.std())
    sd = max(sd, 1e-12)
    z = np.clip((x-mean)/sd, -20, 20)
    r = [_correlation(z[lag:], z[:-lag]) for lag in (1, 2, 5, 12)]
    difference_scale = float(np.std(np.diff(z))/np.sqrt(2.0))
    abs_r1 = _correlation(np.abs(z[1:]), np.abs(z[:-1]))
    skew = float(np.mean(z**3))
    kurtosis = max(float(np.mean(z**4)), 1.0)
    tail = float(np.mean(np.abs(z) > 2.5))
    sign_agreement = float(np.mean((z[1:] >= 0) == (z[:-1] >= 0)))
    trend = _correlation(z, np.linspace(-1, 1, len(z)))
    centered = z-z.mean()
    spectrum = np.abs(np.fft.rfft(centered))**2
    if len(spectrum) <= 2 or spectrum[1:].sum() <= 1e-12:
        low_frequency = 0.0
    else:
        cutoff = max(2, len(spectrum)//16)
        low_frequency = float(spectrum[1:cutoff].sum()/spectrum[1:].sum())
    result = np.asarray((np.log(len(raw)), *r, difference_scale, abs_r1,
        skew, np.log(kurtosis), tail, sign_agreement, trend, low_frequency), dtype=np.float32)
    if result.shape != (len(CONTEXT_NAMES),) or not np.isfinite(result).all():
        raise ValueError("Invalid history context")
    return result


def contexts(pool):
    return {int(position): history_context(values[0]) for position, values in pool.items()}


def _robust_location_scale(values):
    x = np.asarray(values, dtype=np.float64)
    location = float(np.median(x))
    q25, q75 = np.quantile(x, (.25, .75))
    scale = float((q75-q25)/1.3489795003921634)
    if scale <= 1e-8:
        scale = float(x.std())
    return location, max(scale, 1e-8)


def _transport(source_reference, target_reference):
    source_location, source_scale = _robust_location_scale(source_reference)
    target_location, target_scale = _robust_location_scale(target_reference)
    scale = target_scale/source_scale
    return float(scale), float(target_location-scale*source_location)


def select_matched_pairs(pool, tau, series_fold, held_fold, stopping_fold,
                         seed, count=1000, reuse=5, candidates=256, context=None):
    """Select nearest clean-history families without validation donors."""
    rng = np.random.default_rng(seed)
    allowed = [int(p) for p in sorted(pool)
        if series_fold[p] not in (-1, held_fold, stopping_fold)]
    valid = [p for p in allowed if 32 <= tau[p] < len(pool[p][1])-31]
    context = contexts({p: pool[p] for p in allowed}) if context is None else context
    matrix = np.stack([context[p] for p in valid]).astype(np.float64)
    median = np.median(matrix, axis=0)
    scale = np.quantile(matrix, .75, axis=0)-np.quantile(matrix, .25, axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    standardized = {p:(context[p]-median)/scale for p in valid}
    usage = Counter()
    pairs = []
    distances = []
    for a in rng.permutation(valid):
        a = int(a)
        ah, ao = pool[a]
        remaining = len(ao)-int(tau[a])
        possible = [b for b in valid if b != a and usage[b] < reuse
            and len(pool[b][1])-int(tau[b]) >= remaining
            and len(pool[b][0]) >= remaining]
        if not possible:
            continue
        if len(possible) > candidates:
            possible = np.asarray(possible)[rng.choice(len(possible), candidates, replace=False)].tolist()
        distance = np.asarray([np.mean((standardized[a]-standardized[b])**2) for b in possible])
        b = int(possible[int(np.argmin(distance))])
        bh, bo = pool[b]
        target_reference = np.concatenate((ah, ao[:int(tau[a])]))[-256:]
        donor_reference = np.concatenate((bh, bo[:int(tau[b])]))[-256:]
        positive_scale, positive_offset = _transport(donor_reference, target_reference)
        negative = bh[-remaining:]
        negative_scale, negative_offset = _transport(negative, target_reference)
        if not all(np.isfinite(v) and .1 <= abs(v) <= 10 for v in
                   (positive_scale, negative_scale)):
            continue
        pairs.append(dict(a=a, b=b, tau_a=int(tau[a]), tau_b=int(tau[b]),
            length=len(ao), remaining=remaining,
            positive_scale=positive_scale, positive_offset=positive_offset,
            negative_scale=negative_scale, negative_offset=negative_offset,
            match_distance=float(np.min(distance))))
        distances.append(float(np.min(distance)))
        usage[b] += 1
        if len(pairs) == count:
            break
    stats = dict(requested=count, accepted=len(pairs), shortfall=count-len(pairs),
        eligible=len(valid), max_b_reuse=max(usage.values(), default=0),
        median_match_distance=float(np.median(distances)) if distances else None,
        maximum_match_distance=float(np.max(distances)) if distances else None)
    return pairs, stats, context


def assemble(pool, pair, changed):
    ah, ao = pool[pair["a"]]
    bh, bo = pool[pair["b"]]
    tau = pair["tau_a"]
    if changed:
        tail = bo[pair["tau_b"]:pair["tau_b"]+pair["remaining"]]
        tail = pair["positive_scale"]*tail+pair["positive_offset"]
    else:
        tail = bh[-pair["remaining"]:]
        tail = pair["negative_scale"]*tail+pair["negative_offset"]
    online = np.concatenate((ao[:tau], tail)).astype(np.float32)
    history = np.asarray(ah, dtype=np.float32)
    steps = np.arange(pair["length"], dtype=np.int32)
    target = (steps >= tau).astype(np.uint8) if changed else np.zeros(len(steps), dtype=np.uint8)
    if len(online) != pair["length"] or not np.isfinite(online).all():
        raise ValueError("Invalid matched composite")
    return history, online, target, steps
