"""Model177 - the float32 lattice gate (Submission43 = Submission42 + this file + a 30-line main.py).

About 1.25% of the series (125 of the 10,000 training series, 2 of the 100 local test series) are
real-world sensor readings whose raw values sit on a fixed quantisation grid.  The organisers z-score
each series on its history, which keeps the grid intact (step = quantum / std, offset = -mean / std) and
it survives the float32 storage whenever the step is at least ~8 float32 ulps.  The synthetic break that
was spliced into those series produces values that are NOT on the grid, so in the training set the first
off-grid online value falls exactly on the break step in 57 of 63 breaks (the rest within 5 steps) and no
break-free lattice series ever leaves the grid more than once.  The organisers confirmed on 2026-10-03 and
2026-10-09 that these series stay in the test sets and that using the artefact is allowed.

The gate wraps the deployed scorer: for a series whose HISTORY is on a lattice it multiplies the score by
0.01 while every online value is still on the grid (a break has almost surely not happened yet) and
returns 1.0 from the step at which the online values leave the grid.  Every other series is untouched.

Detection is done once per series from the history only (the online segment is never looked ahead at):
  1. the sorted unique history values give the candidate step = smallest gap, provided it is at least
     8 ulps above float32 noise;
  2. the step is refined progressively (gaps of <= 20 quanta, then <= 400, 8,000, 160,000, 4e6) by least
     squares, so that the phase stays coherent over ~1e5 lattice points - a single-gap estimate is ~0.5%
     off and scrambles the phase beyond ~100 points;
  3. base and step are re-fitted jointly, step / {1,2,3,4} is tried (the smallest gap may be 2 quanta),
     and the lattice is accepted when > 98% of the unique values are on it and that fraction beats the
     chance level (2 x tolerance) by at least 0.3.
A value is "on the grid" when its phase error is below max(0.01, 2 ulp / step); values whose own float32
spacing is too coarse for the test (tolerance > 0.3) are ignored rather than counted as off-grid.

Trigger rule (evaluated by scan177.py on the training set):
  * clean lattice (no off-grid value anywhere in the history): the first off-grid online value;
  * unclean lattice (a few off-grid history values, e.g. series 5643 / 6569): the first step at which at
    least two of the last three decidable online values are off-grid (a burst), because an isolated
    off-grid value happens at the history's background rate.
Pure numpy, deterministic, ~3 ms per series.
"""
from __future__ import annotations

from collections import deque

import numpy as np

ACCEPT_FRACTION = 0.98          # share of unique history values that must sit on the lattice
MARGIN_OVER_CHANCE = 0.3        # ... and must beat the chance pass rate by this much
MIN_UNIQUE = 50
MIN_STEP_ULPS = 8.0             # candidate steps closer than this to float32 noise are not testable
SCHEDULES = (((20, 400, 8000, 160000, 4e6), 0.15),
             ((2, 20, 400, 8000, 160000, 4e6), 0.30))


def _ulp(values):
    return np.spacing(np.abs(np.asarray(values, dtype=np.float32))).astype(np.float64)


def on_grid(values, step, base):
    """1.0 on the lattice, 0.0 off it, nan when float32 is too coarse at that magnitude to decide."""
    v = np.asarray(values, dtype=np.float64)
    k = (v - base) / step
    error = np.abs(k - np.round(k))
    tolerance = np.maximum(0.01, 2.0 * _ulp(v) / step)
    result = (error < tolerance).astype(np.float64)
    result[tolerance > 0.3] = np.nan
    return result


def fit_lattice(values):
    """Return {'step', 'base', 'fraction', 'chance', 'div', 'step_ulps'} for a lattice history, else None."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    u = np.unique(v)
    if len(u) < MIN_UNIQUE:
        return None
    gaps = np.diff(u)
    s0 = float(gaps.min())
    ulp_med = float(np.median(_ulp(u)))
    if s0 <= 0.0 or s0 < MIN_STEP_ULPS * ulp_med:
        return None
    for schedule, tol in SCHEDULES:
        s = s0
        for kmax in schedule:
            k = gaps / s
            kr = np.round(k)
            m = (kr >= 1) & (kr <= kmax) & (np.abs(k - kr) < tol)
            if m.sum() < 10:
                continue
            s = float(np.sum(gaps[m] * kr[m]) / np.sum(kr[m] ** 2))
        k = gaps / s
        kr = np.round(k)
        if np.sum((kr >= 1) & (np.abs(k - kr) < 0.3)) < 10:
            continue
        for div in (1, 2, 3, 4):
            sd = s / div
            n = np.round((u - u[0]) / sd)
            design = np.vstack([np.ones_like(n), n]).T
            base, step = np.linalg.lstsq(design, u, rcond=None)[0]
            if not np.isfinite(step) or step <= 0.0:
                continue
            flags = on_grid(u, step, base)
            decidable = np.isfinite(flags)
            if decidable.sum() < MIN_UNIQUE:
                continue
            fraction = float(np.mean(flags[decidable]))
            tolerance = np.maximum(0.01, 2.0 * _ulp(u[decidable]) / step)
            chance = float(np.mean(np.minimum(1.0, 2.0 * tolerance)))
            if fraction > ACCEPT_FRACTION and fraction - chance > MARGIN_OVER_CHANCE:
                return {'step': float(step), 'base': float(base), 'fraction': fraction, 'chance': chance,
                        'div': div, 'step_ulps': float(step / ulp_med)}
    return None


class LatticeGate:
    """Per-series state: fit on the history, observe each online value, adjust each score."""

    LOW_FACTOR = 0.01
    HIGH = 1.0
    BURST_WINDOW = 3
    BURST_COUNT = 2

    def __init__(self, historical, rule='auto'):
        history = np.asarray(historical, dtype=np.float64)
        self.fit = fit_lattice(history)
        self.active = self.fit is not None
        self.triggered = False
        self.rule = rule
        self.history_off = 0
        self.history_decidable = 0
        self.first_off_step = -1
        self.step_index = -1
        if self.active:
            flags = on_grid(history, self.fit['step'], self.fit['base'])
            decidable = np.isfinite(flags)
            self.history_decidable = int(decidable.sum())
            self.history_off = int(np.sum(flags[decidable] == 0.0))
            self.clean = self.history_off == 0
            self.recent = deque(maxlen=self.BURST_WINDOW)

    def observe(self, value):
        """Feed one online value (before scoring it).  Returns the on-grid flag (1, 0 or nan)."""
        self.step_index += 1
        if not self.active:
            return np.nan
        flag = float(on_grid(np.array([float(value)]), self.fit['step'], self.fit['base'])[0])
        if self.triggered:
            return flag
        if not np.isnan(flag):
            off = flag == 0.0
            self.recent.append(1 if off else 0)
            if off and self.first_off_step < 0:
                self.first_off_step = self.step_index
            if self.rule == 'first' or (self.rule == 'auto' and self.clean):
                fire = off
            else:
                fire = sum(self.recent) >= self.BURST_COUNT
            if fire:
                self.triggered = True
        return flag

    def adjust(self, score):
        if not self.active:
            return score
        if self.triggered:
            return self.HIGH
        return float(score) * self.LOW_FACTOR
