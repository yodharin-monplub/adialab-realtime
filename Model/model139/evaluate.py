"""Stage 3 evaluation: TS-AUC, series-bootstrap s.e., z and p for every (test, window, transform).

  python evaluate.py [transform ...]   -> data/results_<transform>.csv and a combined table
H0: the single feature has no ranking power (TS-AUC = 0.5).  Two-sided p under normality,
because a feature with TS-AUC < 0.5 is equally usable with its sign flipped; the signed
TS-AUC is reported too.  Flags at p < 0.01 / 0.05 / 0.10.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tsauc import tsauc  # noqa: E402

HERE = Path(__file__).resolve().parent
BOOT = 100
# signed statistics whose sign depends on the break direction: also score |statistic|
ABS_TESTS = {"welch_t", "student_t", "mann_whitney", "brunner_munzel", "sign_test", "log_var_ratio", "ansari_bradley",
             "mad_ratio", "skew_z", "kurt_z", "ac1", "ac2", "ac3", "durbin_watson", "runs_z", "turning_points",
             "variance_ratio4", "arch_lm", "abs_ac1", "spec_low", "spec_high", "spec_entropy", "spec_peak"}


def evaluate(transform):
    d = np.load(HERE / "data" / f"feat_{transform}.npz")
    X, y, t, pos = d["X"], d["target"], d["time_online"], d["series_position"]
    tests, windows = list(d["tests"]), list(d["windows"])
    series = np.unique(pos); idx_by_series = {s: np.flatnonzero(pos == s) for s in series}
    rng = np.random.default_rng(1390)
    boots = [np.concatenate([idx_by_series[s] for s in rng.choice(series, len(series), replace=True)]) for _ in range(BOOT)]
    rows = []
    for ti, name in enumerate(tests):
        for wi, W in enumerate(windows):
            f0 = X[:, ti, wi].astype(np.float64)
            if np.all(f0 == f0[0]):
                continue
            # signed statistic and its absolute value (two-sided detection); keep the stronger
            cand = [(f0, name), (np.abs(f0), name + "|abs")] if name in ABS_TESTS else [(f0, name)]
            f, name = max(cand, key=lambda c: abs(tsauc(y, c[0], t) - 0.5))
            auc = tsauc(y, f, t)
            bs = np.array([tsauc(y[b], f[b], t[b]) for b in boots])
            se = float(bs.std(ddof=1)) + 1e-9
            z = (auc - 0.5) / se; p = 2 * stats.norm.sf(abs(z))
            rows.append((transform, name, int(W), auc, se, z, p))
        print(f"{transform}: {name} done", flush=True)
    rows.sort(key=lambda r: -abs(r[5]))
    with (HERE / "data" / f"results_{transform}.csv").open("w") as fh:
        fh.write("transform,test,window,tsauc,se,z,p,sig01,sig05,sig10\n")
        for r in rows:
            fh.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.5f},{r[4]:.5f},{r[5]:.2f},{r[6]:.3g},{int(r[6] < .01)},{int(r[6] < .05)},{int(r[6] < .10)}\n")
    n01 = sum(r[6] < .01 for r in rows); n05 = sum(r[6] < .05 for r in rows); n10 = sum(r[6] < .10 for r in rows)
    print(f"{transform}: {len(rows)} features; significant at .01/.05/.10 = {n01}/{n05}/{n10}; best {rows[0][1]} w{rows[0][2]} auc {rows[0][3]:.4f} z {rows[0][5]:.1f}")
    return rows


if __name__ == "__main__":
    for tr in sys.argv[1:] or ["identity"]:
        evaluate(tr)
