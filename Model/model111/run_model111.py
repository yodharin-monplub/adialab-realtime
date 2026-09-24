"""Model111: Model84 expressive voter with oracle-distilled real-row targets.

Everything is frozen from Model84 (fit rows, 40% aligned synthetic splices,
317 fold contract columns, reference/splice weights, expressive LightGBM
settings, stack and causal postprocess). Only the real-row targets change:
    target = ALPHA * y + (1 - ALPHA) * teacher
Synthetic rows keep their hard labels. Teachers (Model109 privileged
features + the 317 fold columns) are trained on folds outside
{k, k+1, j} and predict fit rows of fold j, so neither held fold k nor the
stack-reserved fold k+1 reaches any teacher.

Usage: python models/model111/run_model111.py [fold ...]
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for _name in ("fcntl", "resource"):
    if _name not in sys.modules:
        _stub = types.ModuleType(_name)
        _stub.flock = lambda *a, **k: None
        _stub.LOCK_EX = _stub.LOCK_NB = 0
        _stub.getrusage = lambda *a, **k: types.SimpleNamespace(ru_maxrss=0)
        _stub.RUSAGE_SELF = 0
        sys.modules[_name] = _stub

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
from scipy.special import expit  # noqa: E402

import models.model84.run_model84 as m84  # noqa: E402
from models.model10.causal_transforms import apply_causal_transform  # noqa: E402
from models.model10.sources import load_development_layout  # noqa: E402
from models.model20.features import load_real_pool  # noqa: E402
from models.model20.report import ensemble_margin  # noqa: E402
from models.model20.run_model20 import fitting_rows, reference_weights, splice_weights, synthetic_tasks  # noqa: E402
from models.model21.run_model21 import read_cache  # noqa: E402
from tsauc import time_pair_weights, tsauc  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
TEACHER = ROOT / "models/model109/cache/teacher_features.npy"
ALPHA = float(json.loads((HERE / "config.json").read_text())["alpha"]) if (HERE / "config.json").exists() else 0.5
TAG = f"a{int(round(ALPHA * 100)):03d}"
THREADS = 8
TEACHER_ROWS = 450_000
TEACHER_PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=63, min_child_samples=300,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=3.0, max_bin=127)


def load_everything():
    directory, protocol, fingerprint = m84.frozen_study()
    layout = load_development_layout(ROOT)
    parent = json.loads((m84.PARENT20 / "protocol.json").read_text())
    base = read_cache(m84.PARENT20, "real_features", {"protocol_sha256": m84.CONFIG["model20_protocol_sha256"],
        "rows": layout.row_count, "names": parent["contract"]["union_names"],
        "tasks": int(np.sum(layout.inner_series_fold >= 0))})
    white = m84.white_source.cached_white(m84.PARENT57 / "real_white.npy",
        m84.CONFIG["model57_protocol_sha256"], layout.row_count)
    real = m84.white_source.CombinedFeatures(base, white)
    pool = load_real_pool(layout)
    teacher = np.load(TEACHER, mmap_mode="r")
    return directory, protocol, layout, real, pool, teacher


def teacher_probabilities(layout, real, teacher, columns, fold, fit):
    path = RUNS / f"fold{fold}_teacher.npz"
    if path.exists():
        saved = np.load(path)
        assert np.array_equal(saved["rows"], fit)
        return saved["prob"]
    prob = np.full(len(fit), np.nan, dtype=np.float32)
    report = {}
    for j in range(5):
        if j in (fold, (fold + 1) % 5):
            continue
        pool_rows = np.flatnonzero(~np.isin(layout.row_fold, [fold, (fold + 1) % 5, j]))
        rows = np.sort(np.random.default_rng(7000 + 10 * fold + j).choice(
            pool_rows, size=min(TEACHER_ROWS, len(pool_rows)), replace=False))
        x = np.hstack([real[rows][:, columns], np.asarray(teacher[rows])])
        y, t = layout.target[rows], layout.time_online[rows]
        started = time.monotonic()
        model = lgb.LGBMClassifier(objective="binary", random_state=31 + j, n_jobs=THREADS,
                                   verbosity=-1, force_col_wise=True, **TEACHER_PARAMS)
        model.fit(x, y, sample_weight=time_pair_weights(y, t))
        del x
        mask = layout.row_fold[fit] == j
        target_rows = fit[mask]
        xt = np.hstack([real[target_rows][:, columns], np.asarray(teacher[target_rows])])
        prob[mask] = model.predict_proba(xt)[:, 1]
        report[j] = tsauc(layout.target[target_rows], prob[mask], layout.time_online[target_rows])
        print(f"fold {fold} teacher {j}: {report[j]:.5f} on {mask.sum():,} fit rows "
              f"({time.monotonic() - started:.0f}s)", flush=True)
        model.booster_.save_model(str(RUNS / f"fold{fold}_teacher{j}.txt"))
    if not np.isfinite(prob).all():
        raise ValueError("teacher did not cover every fit row")
    np.savez(path, rows=fit, prob=prob)
    (RUNS / f"fold{fold}_teacher.json").write_text(json.dumps(report, indent=2))
    return prob


def run_fold(protocol, layout, real, pool, teacher, fold):
    out = RUNS / f"{TAG}_fold{fold}.npz"
    held = np.flatnonzero(layout.row_fold == fold)
    if out.exists():
        return np.load(out)["prediction"]
    plan = protocol["plans"][fold]
    seed = protocol["contract"]["recipes"][fold]["seed"]
    fit, fitting = fitting_rows(layout, fold, seed)
    np.testing.assert_array_equal(fit, plan["fit_rows"])
    names, union = protocol["contract"]["names"][fold], protocol["contract"]["union_names"]
    columns = np.asarray([union.index(name) for name in names])
    y, t = layout.target[fit], layout.time_online[fit]
    weights = reference_weights(y, t, layout.target[fitting], layout.time_online[fitting])
    tasks, sy, st = synthetic_tasks(pool, plan, layout, True)
    rw, sw, _ = splice_weights(weights, y, t, sy, st, m84.CONFIG["synthetic_mass"])
    sbase = read_cache(m84.PARENT20, f"aligned_25_fold{fold}_features", {
        "protocol_sha256": m84.CONFIG["model20_protocol_sha256"], "rows": len(sy),
        "names": json.loads((m84.PARENT20 / "protocol.json").read_text())["contract"]["names"][fold],
        "tasks": len(tasks)})
    swhite = m84.white_source.cached_white(m84.PARENT57 / f"aligned_25_fold{fold}_white.npy",
        m84.CONFIG["model57_protocol_sha256"], len(sy))
    sx = np.concatenate((np.asarray(sbase), np.asarray(swhite)), axis=1)
    assert sx.shape == (len(sy), 317)
    del tasks, sbase, swhite
    soft = teacher_probabilities(layout, real, teacher, columns, fold, fit)
    target = ALPHA * y.astype(np.float32) + (1 - ALPHA) * soft
    x = np.concatenate((real[fit][:, columns], sx))
    labels = np.r_[target, sy.astype(np.float32)]
    sample_weights = np.r_[rw, sw]
    model = lgb.LGBMRegressor(objective="cross_entropy", random_state=seed, n_jobs=THREADS,
                              verbosity=-1, deterministic=True, force_col_wise=True,
                              **m84.CONFIG["arms"]["expressive"])
    started = time.monotonic()
    model.fit(x, labels, sample_weight=sample_weights)
    del x
    prediction = np.empty(len(held), dtype=np.float32)
    for left in range(0, len(held), 50_000):
        rows = held[left:left + 50_000]
        prediction[left:left + len(rows)] = model.predict(real[rows][:, columns])
    model.booster_.save_model(str(RUNS / f"{TAG}_fold{fold}.txt"))
    np.savez(out, prediction=prediction, held_rows=held)
    score = tsauc(layout.target[held], prediction, layout.time_online[held])
    print(f"fold {fold} {TAG}: raw component {score:.6f} ({time.monotonic() - started:.0f}s)", flush=True)
    return prediction


def final_scores(layout, predictions):
    """Re-apply Model84's frozen stack/postprocess with our expressive voter."""
    parents = m84.baseline_source.parent_state()
    _, baseline, _ = m84.baseline_source.predictions(layout, parents)
    m26, m35, _ = m84.incumbent.margins(layout)
    anti_delta = m35.astype(np.float64) - m26.astype(np.float64)
    d26, _, s26 = m84.voter.frozen_study()
    probability = np.empty(layout.row_count, dtype=np.float32)
    for fold in range(5):
        held = np.flatnonzero(layout.row_fold == fold)
        probability[held] = m84.voter.prediction_checkpoint(d26, "depth_mix", fold, s26, held)
    for fold, value in predictions.items():
        probability[layout.row_fold == fold] = value
    replaced = ensemble_margin(probability, layout).astype(np.float64)
    adjusted = expit(replaced + anti_delta)
    signal = apply_causal_transform(adjusted, layout.series_position,
        m84.smoother.transform_spec(), layout.time_online).astype(np.float32)
    final = (0.5 * np.asarray(baseline, dtype=np.float32) + 0.5 * signal).astype(np.float32)
    reference = np.load(ROOT / "models/model109/cache/model84_final_oof.npy")
    rows = {}
    for fold in predictions:
        m = layout.row_fold == fold
        rows[fold] = {"model111": tsauc(layout.target[m], final[m], layout.time_online[m]),
                      "model84": tsauc(layout.target[m], reference[m], layout.time_online[m])}
    return final, rows


def main():
    RUNS.mkdir(parents=True, exist_ok=True)
    folds = [int(a) for a in sys.argv[1:]] or [0, 1, 2, 3, 4]
    _, protocol, layout, real, pool, teacher = load_everything()
    predictions = {}
    for fold in range(5):
        path = RUNS / f"{TAG}_fold{fold}.npz"
        if fold in folds:
            predictions[fold] = run_fold(protocol, layout, real, pool, teacher, fold)
        elif path.exists():
            predictions[fold] = np.load(path)["prediction"]
    final, rows = final_scores(layout, predictions)
    np.save(RUNS / f"{TAG}_final_partial.npy", final)
    (RUNS / f"{TAG}_final_scores.json").write_text(json.dumps(rows, indent=2))
    for fold, r in sorted(rows.items()):
        print(f"fold {fold}: Model111 final {r['model111']:.6f} vs Model84 {r['model84']:.6f} "
              f"gain {r['model111'] - r['model84']:+.6f}", flush=True)
    if rows:
        gains = [r["model111"] - r["model84"] for r in rows.values()]
        print(f"mean gain over {len(rows)} folds: {np.mean(gains):+.6f}", flush=True)


if __name__ == "__main__":
    main()
