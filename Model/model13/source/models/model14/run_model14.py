#!/usr/bin/env python3
"""Pairwise stack with regularized causal elapsed-time interactions."""
from pathlib import Path
import hashlib
import json
import sys
import warnings

import numpy as np
from scipy.special import logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.model10.sources import load_development_layout, load_model6_incumbent_postprocess, load_model9_pairwise_stack
from models.model9.pairwise_stack import sample_time_matched_pairs
from models.model11.pipeline import describe, sha256, write_json
from tsauc import tsauc


def time_basis(step):
    step = np.asarray(step, dtype=float)
    return np.stack((np.ones_like(step), .35*np.exp(-step/32), .35*np.exp(-step/128)), axis=-1)


def design(logits, step, scale):
    return ((logits / scale)[:, None, :] * time_basis(step)[:, :, None]).reshape(len(step), -1)


def main():
    layout = load_development_layout(ROOT)
    upstream = ROOT / "models/model9/runs/fulldata_819a001db6/pairwise_stack/experiment_manifest.json"
    sources = json.loads(upstream.read_text())["primitive_sources"]
    latest = json.loads((ROOT / "models/model11/runs/latest.json").read_text())
    added = ROOT / latest["directory"] / "oof.npy"
    sources.append(dict(name="model11:multiscale", path=str(added.relative_to(ROOT)), sha256=sha256(added)))
    contract = dict(version="model14-time-varying-pairwise-v1", sources=sources,
                    split_sha256=sha256(ROOT / "models/model5/artifacts/split_assignments.npz"),
                    metadata_sha256=sha256(ROOT / "models/artifacts/core_features.metadata.npz"),
                    target_index_sha256=sha256(ROOT / "train/y_train_index.parquet"),
                    source_sha256=sha256(__file__), pairs=200_000, C=.001,
                    time_basis="[1, 0.35*exp(-time_online/32), 0.35*exp(-time_online/128)]",
                    maximum_iterations=200, tolerance=1e-6,
                    interpretation="Exploratory OOF stacking; upstream voters are not refitted inside meta-held folds. Development data have been reused adaptively.")
    identity = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()[:12]
    directory = ROOT / f"models/model14/runs/{identity}"
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "protocol.json", contract)
    logits = np.empty((layout.row_count, len(sources)), dtype=np.float32)
    for column, source in enumerate(sources):
        path = (ROOT / source["path"]).resolve()
        if not path.is_relative_to(ROOT) or "holdout" in str(path):
            raise ValueError("Invalid development source path")
        if sha256(path) != source["sha256"]:
            raise ValueError("Primitive source hash mismatch")
        probability = np.load(path, mmap_mode="r")
        if probability.shape != (layout.row_count,) or not np.all(np.isfinite(probability)) or np.any((probability < 0) | (probability > 1)):
            raise ValueError("Primitive source alignment or range mismatch")
        logits[:, column] = logit(np.clip(probability, 1e-6, 1-1e-6))
    prediction = np.empty(layout.row_count, dtype=np.float32)
    scores = []
    for fold in range(5):
        held = np.flatnonzero(layout.row_fold == fold)
        output = directory / f"fold{fold}.npy"
        if output.exists():
            saved = np.load(output)
            if saved.shape != held.shape or not np.all(np.isfinite(saved)):
                raise ValueError("Invalid Model 14 fold checkpoint")
            prediction[held] = saved
        else:
            print(f"Model 14, fold {fold}: fitting 200,000 same-time pairs", flush=True)
            positive, negative, detail = sample_time_matched_pairs(layout.target, layout.time_online, layout.row_fold != fold, 200_000, 9101+fold)
            difference = logits[positive]-logits[negative]
            scale = np.maximum(np.sqrt(np.mean(difference.astype(float)**2, axis=0)), 1e-3)
            difference = design(difference, layout.time_online[positive], scale)
            x = np.concatenate((difference, -difference))
            y = np.r_[np.ones(len(positive)), np.zeros(len(positive))]
            model = LogisticRegression(C=.001, fit_intercept=False, solver="lbfgs", max_iter=200, tol=1e-6)
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(x, y)
            np.savez(directory / f"fold{fold}.coefficients.npz", scale=scale, coefficient=model.coef_[0])
            for chunk in np.array_split(held, 10):
                prediction[chunk] = design(logits[chunk], layout.time_online[chunk], scale) @ model.coef_[0]
            np.save(output, prediction[held])
            write_json(directory / f"fold{fold}.json", dict(pair_sampling=detail, iterations=int(model.n_iter_[0])))
            del x, difference, model
        score = tsauc(layout.target[held], prediction[held], layout.time_online[held])
        scores.append(score)
        print(f"Model 14, fold {fold}: TSAUC={score:.6f}", flush=True)
    np.save(directory / "oof.margin.npy", prediction)
    baseline6 = load_model6_incumbent_postprocess(ROOT, layout=layout)
    baseline9 = load_model9_pairwise_stack(ROOT, layout=layout)
    summary = dict(model=14, comparison_to_model6=describe(scores, baseline6.validation.observed_fold_scores),
                   comparison_to_model9=describe(scores, baseline9.validation.observed_fold_scores),
                   interpretation=contract["interpretation"], sealed_holdout_labels_loaded=False,
                   deployment_promoted=False)
    write_json(directory / "summary.json", summary)
    write_json(ROOT / "models/model14/runs/latest.json", dict(directory=str(directory.relative_to(ROOT))))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
