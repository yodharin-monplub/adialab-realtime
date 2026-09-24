"""Deterministic prebreak-regime matching for aligned splice donors."""
from __future__ import annotations

from collections import Counter

import numpy as np

from models.model18.augmentation import pre_median

DESCRIPTOR_NAMES = ("log_scale", "acf1", "log_diff_scale", "vol_cluster",
                    "log_kurtosis", "low_frequency_share")


def safe_correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 3 or np.std(left) < 1e-8 or np.std(right) < 1e-8:
        return 0.0
    value = float(np.corrcoef(left, right)[0, 1])
    return value if np.isfinite(value) else 0.0


def descriptor(pool, tau_index, position, window):
    history, online = pool[position]
    tau = int(tau_index[position])
    values = np.r_[history, online[:max(0, tau)]][-window:].astype(np.float64)
    center = float(np.median(values))
    scale = max(float(1.4826 * np.median(np.abs(values - center))), 1e-6)
    z = np.clip((values - center) / scale, -12, 12)
    difference = np.diff(z)
    squared = z * z
    power = np.abs(np.fft.rfft(z - np.mean(z))) ** 2
    low_end = max(2, len(power) // 8)
    low_share = float(power[1:low_end].sum() / max(power[1:].sum(), 1e-12))
    result = np.asarray((np.log(scale), safe_correlation(z[1:], z[:-1]),
        np.log(max(float(np.sqrt(np.mean(difference * difference))), 1e-6)),
        safe_correlation(squared[1:], squared[:-1]),
        np.log(max(float(np.mean(z ** 4)), 1e-6)), low_share), dtype=np.float64)
    if result.shape != (len(DESCRIPTOR_NAMES),) or not np.isfinite(result).all():
        raise ValueError("Invalid Model79 regime descriptor")
    return result


def match_plan(pool, tau_index, series_fold, original_plan, window, scale_floor, max_reuse):
    fold = original_plan["fold"]
    positions = sorted(pool)
    matrix = np.vstack([descriptor(pool, tau_index, position, window)
                        for position in positions])
    center = np.median(matrix, axis=0)
    scale = np.maximum(1.4826 * np.median(np.abs(matrix - center), axis=0), scale_floor)
    standardized = (matrix - center) / scale
    lookup = {position: index for index, position in enumerate(positions)}
    usage = Counter()
    pairs, old_distances, new_distances = [], [], []
    old_jumps, new_jumps, real_jumps = [], [], []
    for old in original_plan["pairs"]:
        a = int(old["a"])
        if a not in lookup or series_fold[a] in (-1, fold, (fold + 1) % 5):
            raise ValueError("Model79 A donor is outside the fitting pool")
        remaining = int(old["remaining"])
        za = standardized[lookup[a]]
        choices = [b for b in positions if b != a and usage[b] < max_reuse
                   and len(pool[b][1]) - int(tau_index[b]) >= remaining]
        if not choices:
            raise ValueError("Model79 has no eligible matched B donor")
        ah, ao = pool[a]
        tau_a = int(tau_index[a])
        b = None
        for candidate in sorted(choices, key=lambda p: (
                float(np.sum((standardized[lookup[p]] - za) ** 2)), p)):
            candidate_h, candidate_o = pool[candidate]
            candidate_tau = int(tau_index[candidate])
            candidate_shift = (pre_median(ah, ao, tau_a) -
                               pre_median(candidate_h, candidate_o, candidate_tau))
            with np.errstate(over="ignore", invalid="ignore"):
                candidate_tail = (candidate_o[candidate_tau:candidate_tau + remaining] +
                                  candidate_shift).astype(np.float32)
            if len(candidate_tail) == remaining and np.isfinite(candidate_tail).all():
                b, bh, bo, tau_b, shift = (candidate, candidate_h, candidate_o,
                                            candidate_tau, candidate_shift)
                break
        if b is None:
            raise ValueError("Model79 has no finite aligned B donor")
        usage[b] += 1
        distance = float(np.sqrt(np.sum((standardized[lookup[b]] - za) ** 2)))
        old_distance = float(np.sqrt(np.sum(
            (standardized[lookup[int(old["b"])]] - za) ** 2)))
        pair = dict(a=a, b=int(b), tau_a=tau_a, tau_b=tau_b,
            length=int(old["length"]), remaining=remaining, offset=float(shift))
        pairs.append(pair)
        old_distances.append(old_distance)
        new_distances.append(distance)
        prefix = np.r_[ah, ao[:tau_a]][-128:].astype(np.float64)
        seam_scale = max(float(1.4826 * np.median(np.abs(prefix - np.median(prefix)))), 1e-6)
        previous = float(ao[tau_a - 1] if tau_a else ah[-1])
        old_bh, old_bo = pool[int(old["b"])]
        old_tau = int(old["tau_b"])
        old_shift = pre_median(ah, ao, tau_a) - pre_median(old_bh, old_bo, old_tau)
        old_jumps.append(abs((float(old_bo[old_tau]) + old_shift - previous) / seam_scale))
        new_jumps.append(abs((float(bo[tau_b]) + shift - previous) / seam_scale))
        real_jumps.append(abs((float(ao[tau_a]) - previous) / seam_scale))
    if len(pairs) != len(original_plan["pairs"]):
        raise ValueError("Model79 changed pair count")
    quantiles = (0.5, 0.75, 0.9, 0.99)
    stats = dict(pair_count=len(pairs), unique_b=len(usage),
        max_b_reuse=max(usage.values(), default=0), descriptor_names=list(DESCRIPTOR_NAMES),
        old_distance_quantiles=np.quantile(old_distances, quantiles).tolist(),
        matched_distance_quantiles=np.quantile(new_distances, quantiles).tolist(),
        real_jump_quantiles=np.quantile(real_jumps, quantiles).tolist(),
        old_jump_quantiles=np.quantile(old_jumps, quantiles).tolist(),
        matched_jump_quantiles=np.quantile(new_jumps, quantiles).tolist())
    return pairs, stats
