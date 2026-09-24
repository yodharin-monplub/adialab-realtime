"""Resumable development-only CPU experiments for Models 11--13."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import ExtraTreesClassifier

from models.model10.sources import load_development_layout, load_model6_incumbent_postprocess, load_model9_pairwise_stack
from models.model11.config import BLEND_WEIGHTS, CONFIGS, SEED
from models.model11.features import FEATURE_NAMES, compute_features
from tsauc import fold_standard_error, time_pair_weights, tsauc

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "models/artifacts/compact_multiscale"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def cache_contract():
    return {"version": 1, "feature_names": FEATURE_NAMES,
            "features_sha256": sha256(Path(__file__).with_name("features.py")),
            "data_sha256": sha256(ROOT / "train/X_train.parquet"),
            "target_index_sha256": sha256(ROOT / "train/y_train_index.parquet"),
            "split_sha256": sha256(ROOT / "models/model5/artifacts/split_assignments.npz")}


def prepare():
    """Restore unlabeled row metadata and cache development features only."""
    CACHE.mkdir(parents=True, exist_ok=True)
    contract = cache_contract()
    manifest = CACHE / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text())["contract"] != contract:
            raise ValueError("Feature cache contract changed; use a new cache directory")
        matrix = np.load(CACHE / "features.npy", mmap_mode="r")
        layout = load_development_layout(ROOT)
        if matrix.shape != (layout.row_count, len(FEATURE_NAMES)):
            raise ValueError("Cached feature dimensions disagree with development layout")
        return matrix, layout
    frame = pd.read_parquet(ROOT / "train/X_train.parquet", columns=["value", "period"])
    if frame.index.names != ["id", "time"] or not frame.index.is_monotonic_increasing:
        raise ValueError("Training data must be sorted by (id, time)")
    ids = frame.index.get_level_values("id").to_numpy()
    values = frame.value.to_numpy(dtype=np.float64)
    periods = frame.period.to_numpy()
    boundaries = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, ids.size]
    series_ids = ids[boundaries[:-1]]
    with np.load(ROOT / "models/model5/artifacts/split_assignments.npz") as split:
        inner = split["inner_series_fold"]
    if series_ids.size != inner.size:
        raise ValueError("Downloaded training population differs from saved split; cannot reuse OOF")
    lengths, starts = [], []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        local = periods[left:right]
        online = np.flatnonzero(local == 2)
        if not online.size or np.any(local[:online[0]] != 1) or np.any(local[online[0]:] != 2):
            raise ValueError("Series is not historical-then-online")
        starts.append(int(left + online[0]))
        lengths.append(int(online.size))
    lengths = np.array(lengths, dtype=np.int32)
    metadata_path = ROOT / "models/artifacts/core_features.metadata.npz"
    metadata = dict(series_position=np.repeat(np.arange(len(lengths), dtype=np.int32), lengths),
                    time_online=np.concatenate([np.arange(n, dtype=np.int32) for n in lengths]),
                    online_lengths=lengths, series_ids=series_ids)
    if metadata_path.exists():
        with np.load(metadata_path) as old:
            if any(not np.array_equal(old[k], v) for k, v in metadata.items()):
                raise ValueError("Existing row metadata differs from downloaded training data")
    else:
        np.savez_compressed(metadata_path, **metadata)
    del metadata, frame, ids, periods
    layout = load_development_layout(ROOT)
    # Recompute the old scores before trusting positional artifact alignment.
    baseline = load_model6_incumbent_postprocess(ROOT, layout=layout)
    print(f"Restored baseline TSAUC={np.mean(baseline.validation.observed_fold_scores):.9f}; {layout.series_count} development series", flush=True)
    temporary = CACHE / "features.partial.npy"
    features = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(layout.row_count, len(FEATURE_NAMES)))
    offset = 0
    count = 0
    for position in np.flatnonzero(inner >= 0):
        left, right = boundaries[position:position+2]
        start = starts[position]
        block = compute_features(values[left:start], values[start:right])
        if not np.all(np.isfinite(block)):
            raise ValueError(f"Nonfinite features for development position {position}")
        features[offset:offset+len(block)] = block
        offset += len(block)
        count += 1
        if count % 500 == 0:
            print(f"Features: {count}/{layout.series_count} series", flush=True)
    assert offset == layout.row_count
    features.flush()
    del features
    temporary.replace(CACHE / "features.npy")
    write_json(manifest, {"contract": contract, "development_rows": offset, "development_series": layout.series_count,
                          "sealed_holdout_labels_loaded": False})
    return np.load(CACHE / "features.npy", mmap_mode="r"), layout


def sample_pairs(target, steps, count, seed):
    """Sample positive-negative pairs with official same-time pair weights."""
    rng = np.random.default_rng(seed)
    times = np.unique(steps)
    positives = [np.flatnonzero((steps == t) & (target == 1)) for t in times]
    negatives = [np.flatnonzero((steps == t) & (target == 0)) for t in times]
    weight = np.array([len(p)*len(n) for p,n in zip(positives, negatives)], dtype=float)
    if not weight.sum():
        raise ValueError("No same-time positive-negative pairs in fit rows")
    counts = rng.multinomial(count, weight / weight.sum())
    pos, neg = [], []
    for p,n,k in zip(positives, negatives, counts):
        if k:
            pos.append(rng.choice(p, k))
            neg.append(rng.choice(n, k))
    return np.concatenate(pos), np.concatenate(neg)


def pairwise_derivatives(prediction, positive, negative):
    q = expit(prediction[negative] - prediction[positive])
    curvature = q * (1 - q)
    size = len(prediction)
    scale = size / len(positive)
    gradient = (np.bincount(negative, weights=q, minlength=size) - np.bincount(positive, weights=q, minlength=size)) * scale
    hessian = (np.bincount(negative, weights=curvature, minlength=size) + np.bincount(positive, weights=curvature, minlength=size)) * scale
    return gradient, np.maximum(hessian, 1e-6)


def fit_model(config, x, target, steps, seed):
    weights = time_pair_weights(target, steps)
    if config["kind"] == "extra_trees":
        model = ExtraTreesClassifier(**config["parameters"], n_jobs=4, random_state=seed)
        model.fit(x, target, sample_weight=weights)
        return model
    parameters = dict(objective="binary", metric="None", verbosity=-1, num_threads=4,
                      deterministic=True, force_col_wise=True, seed=seed, feature_fraction=.9,
                      max_bin=127, **config["parameters"])
    if config["kind"] == "pairwise":
        positive, negative = sample_pairs(target, steps, config["pairs"], seed)
        def objective(prediction, dataset):
            return pairwise_derivatives(prediction, positive, negative)
        parameters["objective"] = objective
        weights = None
    data = lgb.Dataset(x, label=target, weight=weights, feature_name=FEATURE_NAMES)
    return lgb.train(parameters, data, num_boost_round=config["rounds"])


def predict(model, config, x):
    if config["kind"] == "extra_trees":
        return model.predict_proba(x)[:, 1]
    values = model.predict(x, num_threads=4)
    return expit(values) if config["kind"] == "pairwise" else values


def describe(scores, baseline_scores):
    gain = np.asarray(scores) - baseline_scores
    return dict(fold_scores=list(map(float, scores)), fold_mean=float(np.mean(scores)),
                fold_standard_error=fold_standard_error(scores), mean_gain=float(gain.mean()),
                paired_gain_se=fold_standard_error(gain), fold_wins=int(np.sum(gain > 0)))


def run(number):
    started = time.monotonic()
    config = CONFIGS[number]
    x, layout = prepare()
    contract = dict(config=config, seed=SEED, protocol="fixed-fivefold-development-only-v1",
                    feature_contract=json.loads((CACHE / "manifest.json").read_text())["contract"],
                    pipeline_sha256=sha256(__file__), lightgbm_version=lgb.__version__)
    identity = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()[:12]
    directory = ROOT / f"models/model{number}/runs/{identity}"
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "protocol.json", contract)
    prediction = np.empty(layout.row_count, dtype=np.float32)
    fold_scores = []
    for fold in range(5):
        held = np.flatnonzero(layout.row_fold == fold)
        output = directory / f"fold{fold}.npy"
        if output.exists():
            saved = np.load(output)
            if saved.shape != held.shape or not np.all(np.isfinite(saved)):
                raise ValueError("Invalid fold prediction checkpoint")
            prediction[held] = saved
        else:
            begin = time.monotonic()
            eligible = np.flatnonzero(layout.row_fold != fold)
            rng = np.random.default_rng(SEED + fold)
            fit = np.sort(rng.choice(eligible, min(config["fit_rows"], len(eligible)), replace=False))
            if np.intersect1d(layout.series_position[fit], layout.series_position[held]).size:
                raise AssertionError("Series leakage in fit/score roles")
            print(f"Model {number}, fold {fold}: fitting {len(fit):,} rows", flush=True)
            model = fit_model(config, np.asarray(x[fit]), layout.target[fit], layout.time_online[fit], SEED + fold)
            if config["kind"] == "extra_trees":
                joblib.dump(model, directory / f"fold{fold}.joblib", compress=3)
            else:
                model.save_model(str(directory / f"fold{fold}.txt"))
            for chunk in np.array_split(held, max(1, int(np.ceil(len(held)/100_000)))):
                prediction[chunk] = predict(model, config, x[chunk])
            np.save(output, prediction[held])
            write_json(directory / f"fold{fold}.json", dict(fit_rows=len(fit), held_rows=len(held), elapsed_seconds=time.monotonic()-begin))
            del model
        score = tsauc(layout.target[held], prediction[held], layout.time_online[held])
        fold_scores.append(score)
        print(f"Model {number}, fold {fold}: TSAUC={score:.6f}", flush=True)
    np.save(directory / "oof.npy", prediction)
    baseline = load_model6_incumbent_postprocess(ROOT, layout=layout)
    baseline_scores = baseline.validation.observed_fold_scores
    result = dict(model=number, name=config["name"], run_directory=str(directory.relative_to(ROOT)),
                  raw=describe(fold_scores, baseline_scores), development_series=layout.series_count,
                  development_rows=layout.row_count, sealed_holdout_labels_loaded=False,
                  interpretation="Fixed recipe on reused development folds. Blends inherit upstream adaptive selection and are exploratory; no deployment promotion.",
                  elapsed_seconds=time.monotonic()-started, blends=[])
    margin = logit(np.clip(prediction, 1e-6, 1-1e-6))
    for source in (baseline, load_model9_pairwise_stack(ROOT, layout=layout)):
        source_margin = logit(np.clip(source.probability, 1e-6, 1-1e-6))
        for alpha in BLEND_WEIGHTS:
            blended = ((1-alpha)*source_margin + alpha*margin).astype(np.float32)
            scores = [tsauc(layout.target[layout.row_fold == f], blended[layout.row_fold == f], layout.time_online[layout.row_fold == f]) for f in range(5)]
            result["blends"].append(dict(source=source.name, new_model_weight=alpha,
                                        comparison_to_source=describe(scores, source.validation.observed_fold_scores)))
    write_json(directory / "summary.json", result)
    write_json(ROOT / f"models/model{number}/runs/latest.json", {"directory": str(directory.relative_to(ROOT))})
    print(json.dumps(result, indent=2), flush=True)
    return result
