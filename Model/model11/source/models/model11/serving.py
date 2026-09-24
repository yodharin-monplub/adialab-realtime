"""Crunch's generator protocol for the compact standalone models."""
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from scipy.special import expit

from models.model11.config import CONFIGS, SEED
from models.model11.features import FEATURE_NAMES, StreamingFeatures, compute_features


def load_model(directory, number):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["model"] != number or manifest["feature_names"] != FEATURE_NAMES:
        raise ValueError("Serving model feature contract does not match")
    config = CONFIGS[number]
    if config["kind"] == "extra_trees":
        model = joblib.load(directory / "model.joblib")
        model.n_jobs = 1
    else:
        model = lgb.Booster(model_file=str(directory / "model.txt"))
    feature_count = model.n_features_in_ if config["kind"] == "extra_trees" else model.num_feature()
    if feature_count != len(FEATURE_NAMES):
        raise ValueError("Fitted model has the wrong number of features")
    return model


def train(datasets, model_directory_path, number):
    from models.model11.pipeline import fit_model, write_json
    directory = Path(model_directory_path)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "manifest.json").exists():
        load_model(directory, number)
        return
    rng = np.random.default_rng(SEED)
    features, targets, steps = [], [], []
    for _, historical, online, tau in datasets:
        online = np.asarray(online, dtype=float)
        block = compute_features(historical, online)
        sampled = np.flatnonzero(rng.random(len(online)) < .10)
        features.append(block[sampled])
        targets.append(((tau is not None) & (sampled >= (int(tau) if tau is not None and tau >= 0 else len(online)))).astype(np.uint8))
        steps.append(sampled)
    if not features or sum(map(len, features)) == 0:
        raise ValueError("No sampled training rows")
    x, y, t = np.concatenate(features), np.concatenate(targets), np.concatenate(steps)
    config = CONFIGS[number]
    selected = rng.choice(len(x), min(len(x), config["fit_rows"]), replace=False)
    model = fit_model(config, x[selected], y[selected], t[selected], SEED)
    if config["kind"] == "extra_trees":
        joblib.dump(model, directory / "model.joblib", compress=3)
    else:
        model.save_model(str(directory / "model.txt"))
    write_json(directory / "manifest.json", dict(model=number, feature_names=FEATURE_NAMES,
                                                training_rows=len(selected), config=config))


def infer(datasets, model_directory_path, number):
    model = load_model(model_directory_path, number)
    kind = CONFIGS[number]["kind"]
    yield None
    for item in datasets:
        if len(item) == 2:
            historical, online = item
        elif len(item) == 3:
            _, historical, online = item
        else:
            raise ValueError("Expected (history, online) or (id, history, online)")
        features = StreamingFeatures(historical)
        for value in online:
            row = features.update(float(value)).reshape(1, -1)
            if kind == "extra_trees":
                probability = model.predict_proba(row)[0, 1]
            else:
                probability = model.predict(row, num_threads=1)[0]
                if kind == "pairwise":
                    probability = expit(probability)
            yield float(np.clip(probability, 1e-7, 1-1e-7))
