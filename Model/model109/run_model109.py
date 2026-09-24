"""Model109: oracle-distilled LightGBM student (nested, series-fold clean).

For each outer held fold k:
  1. For every other fold j, a *teacher* is trained on folds not in {k, j}
     using the 500 causal features plus future-seeing privileged features
     (teacher_features.py).  It predicts fold-j rows of the student sample.
  2. Students are trained on the causal features only, with targets
       hard : y
       soft : teacher probability
       mix  : 0.5 * y + 0.5 * teacher
     and identical rows, weights and LightGBM settings.
  3. Every arm predicts the complete held fold k; TS-AUC is recorded.
Fold-k labels never reach a teacher or student used to predict fold k.

Usage: python models/model109/run_model109.py [fold ...]
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model10.sources import load_development_layout  # noqa: E402
from tsauc import time_pair_weights, tsauc  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
REAL = ROOT / "models/model20/runs/82abe027caea808d/real_features.npy"
WHITE = ROOT / "models/model57/runs/dfab0ede1c3206a4/real_white.npy"
TEACHER = HERE / "cache/teacher_features.npy"

THREADS = 8
STUDENT_ROWS = 500_000
TEACHER_ROWS = 450_000
TEACHER_PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=63, min_child_samples=300,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=3.0,
                      max_bin=127)
STUDENT_PARAMS = dict(n_estimators=1000, learning_rate=0.03, num_leaves=63, min_child_samples=300,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_alpha=0.2,
                      reg_lambda=3.0, max_bin=127)
ARMS = {"hard": 1.0, "mix": 0.5, "soft": 0.0}


class Data:
    def __init__(self):
        self.layout = load_development_layout(ROOT)
        t0 = time.time()
        self.real = np.load(REAL, mmap_mode="r")
        self.white = np.load(WHITE, mmap_mode="r")
        self.teacher = np.load(TEACHER, mmap_mode="r")
        n = self.layout.row_count
        assert self.real.shape[0] == n and self.white.shape[0] == n and self.teacher.shape[0] == n
        print(f"opened caches in {time.time() - t0:.1f}s", flush=True)

    def causal(self, rows: np.ndarray) -> np.ndarray:
        rows = np.sort(rows)
        out = np.empty((len(rows), self.real.shape[1] + self.white.shape[1]), dtype=np.float32)
        for a in range(0, len(rows), 200_000):
            r = rows[a:a + 200_000]
            out[a:a + len(r), :self.real.shape[1]] = self.real[r]
            out[a:a + len(r), self.real.shape[1]:] = self.white[r]
        return out

    def privileged(self, rows: np.ndarray) -> np.ndarray:
        return np.asarray(self.teacher[np.sort(rows)])


def sample_rows(candidates: np.ndarray, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if size >= len(candidates):
        return np.sort(candidates)
    return np.sort(rng.choice(candidates, size=size, replace=False))


def weights_for(layout, rows):
    return time_pair_weights(layout.target[rows], layout.time_online[rows])


def fit(params, x, y, w, seed, objective):
    model = lgb.LGBMClassifier(objective=objective, random_state=seed, n_jobs=THREADS,
                               verbosity=-1, force_col_wise=True, **params) \
        if objective == "binary" else \
        lgb.LGBMRegressor(objective=objective, random_state=seed, n_jobs=THREADS,
                          verbosity=-1, force_col_wise=True, **params)
    model.fit(x, y, sample_weight=w)
    return model


def predict(model, x):
    if isinstance(model, lgb.LGBMClassifier):
        return model.predict_proba(x)[:, 1].astype(np.float32)
    return model.predict(x).astype(np.float32)


def teacher_targets(data: Data, k: int, student_rows: np.ndarray) -> np.ndarray:
    path = RUNS / f"fold{k}_teacher.npz"
    if path.exists():
        saved = np.load(path)
        assert np.array_equal(saved["rows"], student_rows)
        return saved["prob"]
    L = data.layout
    prob = np.full(len(student_rows), np.nan, dtype=np.float32)
    scores = {}
    for j in range(5):
        if j == k:
            continue
        pool = np.flatnonzero((L.row_fold != k) & (L.row_fold != j))
        rows = sample_rows(pool, TEACHER_ROWS, 1000 * k + j)
        x = np.hstack([data.causal(rows), data.privileged(rows)])
        y = L.target[rows]
        t0 = time.time()
        model = fit(TEACHER_PARAMS, x, y, weights_for(L, rows), 17 + j, "binary")
        del x
        target_mask = L.row_fold[student_rows] == j
        target_rows = student_rows[target_mask]
        xt = np.hstack([data.causal(target_rows), data.privileged(target_rows)])
        prob[target_mask] = predict(model, xt)
        scores[j] = tsauc(L.target[target_rows], prob[target_mask], L.time_online[target_rows])
        print(f"fold {k} teacher {j}: tsauc on student rows {scores[j]:.5f} ({time.time() - t0:.0f}s)",
              flush=True)
        del xt, model
    assert np.isfinite(prob).all()
    np.savez(path, rows=student_rows, prob=prob)
    (RUNS / f"fold{k}_teacher.json").write_text(json.dumps(scores, indent=2))
    return prob


def run_fold(data: Data, k: int):
    L = data.layout
    student_rows = sample_rows(np.flatnonzero(L.row_fold != k), STUDENT_ROWS, 500 + k)
    soft = teacher_targets(data, k, student_rows)
    x = data.causal(student_rows)
    y = L.target[student_rows].astype(np.float32)
    w = weights_for(L, student_rows)
    held = np.flatnonzero(L.row_fold == k)
    xh = None
    results = {}
    for arm, alpha in ARMS.items():
        path = RUNS / f"fold{k}_{arm}.npy"
        if path.exists():
            pred = np.load(path)
        else:
            t0 = time.time()
            if alpha == 1.0:
                model = fit(STUDENT_PARAMS, x, y.astype(np.int8), w, 7, "binary")
            else:
                target = alpha * y + (1 - alpha) * soft
                model = fit(STUDENT_PARAMS, x, target, w, 7, "cross_entropy")
            if xh is None:
                xh = data.causal(held)
            pred = predict(model, xh)
            np.save(path, pred)
            model.booster_.save_model(str(RUNS / f"fold{k}_{arm}.txt"))
            print(f"fold {k} {arm}: fit+predict {time.time() - t0:.0f}s", flush=True)
        results[arm] = tsauc(L.target[held], pred, L.time_online[held])
        print(f"fold {k} {arm}: held TSAUC {results[arm]:.6f}", flush=True)
    (RUNS / f"fold{k}_scores.json").write_text(json.dumps(results, indent=2))
    return results


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    folds = [int(a) for a in sys.argv[1:]] or [0, 1, 2, 3, 4]
    data = Data()
    for k in folds:
        run_fold(data, k)


if __name__ == "__main__":
    main()
