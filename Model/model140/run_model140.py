"""Model140 (stage 4: FEATURES from the statistical tests + feature selection).

Same student as Model111 (distilled LightGBM, 40% splices), same folds; only the column set changes.
Arms (name -> columns):
  A      : the current 317 columns = Model111 a050 files (reference, not retrained)
  B      : 317 + all 13 test features (fast_tests.NAMES)
  C01/C05/C10 : 317 + the test features whose stage-3 one-test TS-AUC is significant at p<.01/.05/.10
  D100/D200   : gain top-k of (317 + 13) taken from the B booster of the same fold, refit on k columns
  E      : (317 + 13) with |rho| > 0.95 duplicates dropped (greedy, on the fitting rows)
  F      : forward addition of the 13, one at a time, by fold gain (13 fits per fold; run last)
Usage: python models/model140/run_model140.py ARM [ARM ...] [--folds 0 1] | report
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import models.model111.run_model111 as m111  # noqa: E402  (installs Linux stubs)
import lightgbm as lgb  # noqa: E402

from models.model137.run_model137 import synthetic_block  # noqa: E402
from models.model20.run_model20 import fitting_rows, reference_weights, splice_weights  # noqa: E402
from models.model140.fast_tests import NAMES  # noqa: E402
from tsauc import tsauc  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
DATA = HERE / "data"
THREADS = int(__import__("os").environ.get("LGBM_THREADS", "8"))
# FT_SETS: comma list of feature sets appended after the 317 (identity | diff | abs), 13 columns each
FT_SETS = __import__("os").environ.get("FT_SETS", "identity").split(",")
SUFFIXES = {s: ("" if s == "identity" else f"_{s}") for s in FT_SETS}
EXTRA_NAMES = [f"{n}{SUFFIXES[s]}" for s in FT_SETS for n in NAMES]
# stage-3 (Model139) test name and window of each fast feature, for the p-value arms
STAGE3 = {"ft_cusum_sq": ("cusum_sq", 0), "ft_var_lr_scan": ("var_lr_scan", 0), "ft_chow_f_scan": ("chow_f_scan", 0),
          "ft_buishand": ("buishand_range", 0), "ft_page_hinkley": ("page_hinkley", 0), "ft_levene128": ("levene", 128),
          "ft_brown_forsythe128": ("brown_forsythe", 128), "ft_abs_log_var_ratio128": ("log_var_ratio|abs", 128),
          "ft_wasserstein128": ("wasserstein", 128), "ft_kuiper128": ("kuiper", 128), "ft_ljung_box128": ("ljung_box10", 128),
          "ft_kurtosis128": ("kurt_z|abs", 128), "ft_spec_entropy64": ("spec_entropy", 128)}


def student(seed):
    return lgb.LGBMRegressor(objective="cross_entropy", random_state=seed, n_jobs=THREADS, verbosity=-1,
                             deterministic=True, force_col_wise=True, **m111.m84.CONFIG["arms"]["expressive"])


def significant(level):
    r = pd.read_csv(ROOT / "models/model139/data/results_identity.csv")
    keep = []
    for i, n in enumerate(NAMES):
        test, w = STAGE3[n]
        row = r[(r.test == test) & (r.window == w)]
        if len(row) and float(row.p.iloc[0]) < level:
            keep.append(i)
    return np.asarray(keep, dtype=int)


class Fold:
    """Fitting matrices of one fold with the 13 extra columns appended (317 + 13 = 330)."""

    def __init__(self, protocol, layout, real, pool, real13, fold, syn13=None, extra_names=None):
        """syn13 / extra_names let other models (e.g. Model144) pass their own extra column blocks."""
        self.fold = fold
        self.seed = protocol["contract"]["recipes"][fold]["seed"]
        fit, fitting = fitting_rows(layout, fold, self.seed)
        names, union = protocol["contract"]["names"][fold], protocol["contract"]["union_names"]
        columns = np.asarray([union.index(n) for n in names])
        y, t = layout.target[fit], layout.time_online[fit]
        w = reference_weights(y, t, layout.target[fitting], layout.time_online[fitting])
        sx, sy, st = synthetic_block(protocol, layout, pool, fold, names)
        rw, sw, _ = splice_weights(w, y, t, sy, st, m111.m84.CONFIG["synthetic_mass"])
        saved = np.load(ROOT / f"models/model111/runs/fold{fold}_teacher.npz"); assert np.array_equal(saved["rows"], fit)
        target = 0.5 * y.astype(np.float32) + 0.5 * saved["prob"]
        if syn13 is None:
            syn13 = np.concatenate([np.load(DATA / f"syn{SUFFIXES[s]}_fold{fold}.npy") for s in FT_SETS], axis=1)
        assert len(syn13) == len(sy) and syn13.shape[1] == real13.shape[1], (syn13.shape, real13.shape, len(sy))
        self.x = np.concatenate((np.concatenate((real[fit][:, columns], real13[fit]), axis=1),
                                 np.concatenate((sx, syn13), axis=1)))
        self.labels = np.r_[target, sy.astype(np.float32)]
        self.weights = np.r_[rw, sw]
        self.held = np.flatnonzero(layout.row_fold == fold)
        self.xh = np.concatenate((real[self.held][:, columns], real13[self.held]), axis=1)
        self.y_held, self.t_held = layout.target[self.held], layout.time_online[self.held]
        self.ref = tsauc(self.y_held, np.load(ROOT / f"models/model111/runs/a050_fold{fold}.npz")["prediction"], self.t_held)
        self.names = list(names) + list(extra_names if extra_names is not None else EXTRA_NAMES)

    def fit(self, arm, cols):
        out = RUNS / f"{arm}_fold{self.fold}.npz"
        if out.exists():
            return np.load(out)["prediction"]
        t0 = time.monotonic(); cols = np.asarray(cols)
        model = student(self.seed); model.fit(self.x[:, cols], self.labels, sample_weight=self.weights)
        pred = np.empty(len(self.held), dtype=np.float32)
        for left in range(0, len(self.held), 100_000):
            pred[left:left + 100_000] = model.predict(self.xh[left:left + 100_000][:, cols])
        model.booster_.save_model(str(RUNS / f"{arm}_fold{self.fold}.txt"))
        np.savez(out, prediction=pred, held_rows=self.held, columns=cols)
        s = tsauc(self.y_held, pred, self.t_held)
        print(f"{arm} fold {self.fold}: {s:.6f} vs A (Model111) {self.ref:.6f} ({s - self.ref:+.6f}) "
              f"{len(cols)} cols [{time.monotonic() - t0:.0f}s]", flush=True)
        return pred

    def top_gain(self, k):
        booster = lgb.Booster(model_file=str(RUNS / f"B_fold{self.fold}.txt"))
        gain = booster.feature_importance("gain")
        return np.sort(np.argsort(gain)[::-1][:k])

    def corr_prune(self, thr=0.95):
        rng = np.random.default_rng(self.fold)
        rows = rng.choice(len(self.x), 200_000, replace=False)
        c = np.corrcoef(self.x[rows].astype(np.float64), rowvar=False); c = np.nan_to_num(np.abs(c))
        keep = []
        for j in range(c.shape[0]):
            if all(c[j, i] <= thr for i in keep):
                keep.append(j)
        return np.asarray(keep)


def run(arms, folds):
    RUNS.mkdir(exist_ok=True)
    _, protocol, layout, real, pool, _ = m111.load_everything()
    real13 = np.concatenate([np.load(DATA / f"real{SUFFIXES[s]}.npy") for s in FT_SETS], axis=1)
    assert len(real13) == layout.row_count
    base = np.arange(317); extra = 317 + np.arange(len(EXTRA_NAMES))
    for fold in folds:
        F = Fold(protocol, layout, real, pool, real13, fold)
        for arm in arms:
            if arm == "B":
                F.fit("B", np.r_[base, extra])
            elif arm.startswith("C"):
                F.fit(arm, np.r_[base, 317 + significant({"C01": .01, "C05": .05, "C10": .10}[arm])])
            elif arm.startswith("D"):
                F.fit("B", np.r_[base, extra]); F.fit(arm, F.top_gain(int(arm[1:])))
            elif arm in ("E", "ED"):  # ED = E with FT_SETS=identity,diff,abs (stage-2 winners), pruned the same way
                cols = F.corr_prune(); print(f"{arm} fold {fold}: {len(cols)} columns kept", flush=True); F.fit(arm, cols)
            elif arm == "P":  # control for E: the same pruning applied to the 317 alone (no test features)
                cols = F.corr_prune(); cols = cols[cols < 317]
                print(f"P fold {fold}: {len(cols)} columns kept", flush=True); F.fit("P", cols)
            elif arm == "F":
                chosen, best = [], F.ref
                for step in range(13):
                    gains = {}
                    for j in range(13):
                        if j in chosen:
                            continue
                        pred = F.fit(f"F{step}_{NAMES[j]}", np.r_[base, 317 + np.asarray(chosen + [j])])
                        gains[j] = tsauc(F.y_held, pred, F.t_held)
                    j = max(gains, key=gains.get)
                    print(f"F fold {fold} step {step}: best add {NAMES[j]} -> {gains[j]:.6f} ({gains[j] - best:+.6f})", flush=True)
                    if gains[j] <= best:
                        break
                    chosen.append(j); best = gains[j]
                json.dump({"chosen": [NAMES[j] for j in chosen], "score": best}, open(RUNS / f"F_fold{fold}.json", "w"))
        del F


def report():
    _, _, layout, _, _, _ = m111.load_everything()
    arms = sorted({p.name.split("_fold")[0] for p in RUNS.glob("*_fold?.npz")})
    lines = []
    for arm in arms:
        gains = []
        for fold in range(5):
            p = RUNS / f"{arm}_fold{fold}.npz"
            if not p.exists():
                gains.append(np.nan); continue
            d = np.load(p); held = d["held_rows"]
            ref = np.load(ROOT / f"models/model111/runs/a050_fold{fold}.npz")["prediction"]
            gains.append(tsauc(layout.target[held], d["prediction"], layout.time_online[held])
                         - tsauc(layout.target[held], ref, layout.time_online[held]))
        g = np.asarray(gains); ok = g[np.isfinite(g)]
        se = ok.std(ddof=1) / np.sqrt(len(ok)) if len(ok) > 1 else np.nan
        lines.append(f"{arm:10s} mean {ok.mean():+.4f} s.e. {se:.4f} wins {(ok > 0).sum()}/{len(ok)}  folds " +
                     " ".join("   nan " if not np.isfinite(x) else f"{x:+.4f}" for x in g))
    text = "\n".join(lines); print(text); (HERE / "score.txt").write_text("gain vs A (Model111 a050), per fold\n" + text + "\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["report"]:
        report()
    else:
        folds = [int(a) for a in args if a.isdigit()] or [0, 1]
        run([a for a in args if not a.isdigit() and a != "--folds"], folds)
