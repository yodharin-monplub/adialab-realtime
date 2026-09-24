"""Pilot-trained reliability gate for Model26 versus Model27."""
from pathlib import Path
import argparse
import fcntl
import json
import sys

import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.model10.sources import load_development_layout
from models.model11.pipeline import sha256, write_json
from models.model18.run_model18 import commit_checkpoint, digest, read_checkpoint, validate_predictions
from models.model20.features import load_real_pool
from models.model20.run_model20 import reference_weights
from models.model26 import run_model26 as incumbent
from models.model27.augmentation import CONTEXT_NAMES, contexts


HERE = Path(__file__).resolve().parent
MODEL27 = ROOT / "models/model27"
ARMS = ("model26", "context_gate")
CONFIG = dict(version=1, arms=list(ARMS), pilot_folds=[0, 1], confirmation_folds=[2, 3, 4],
    c=.05, max_iter=500, tolerance=1e-8, seed=290911,
    target_tsauc=.64, minimum_confirmation_gain=.001, minimum_confirmation_wins=2,
    bootstrap_replicates=1000, bootstrap_seed=291109, confidence=.95,
    context_names=list(CONTEXT_NAMES),
    interpretation=("One fixed linear reliability gate is trained only on pilot folds 0-1 after their "
        "Model27 predictions were observed. The gate combines Model26 and Model27 logits, time, clean-history "
        "context, and context-by-disagreement interactions. Folds 2-4 are untouched confirmation data. "
        "Five-fold results are not used as confirmatory evidence."))


def _latest(model_dir):
    latest = json.loads((model_dir / "runs/latest.json").read_text())
    directory = ROOT / latest["directory"]
    fingerprint = sha256(directory / "protocol.json")
    if fingerprint != latest["protocol_sha256"]:
        raise ValueError(f"Changed upstream protocol: {model_dir.name}")
    return directory, fingerprint


def initialize():
    if (HERE / "runs/latest.json").exists():
        return frozen_study()
    model27_dir, model27_sha = _latest(MODEL27)
    model26_dir, _, model26_sha = incumbent.frozen_study()
    files = [HERE / "run_model29.py", HERE / "report.py", HERE / "readme.txt",
        ROOT / "tests/test_model29.py", model27_dir / "protocol.json", model26_dir / "protocol.json"]
    protocol = dict(contract=dict(config=CONFIG,
        hashes={str(path.relative_to(ROOT)): sha256(path) for path in files},
        model27_protocol_sha256=model27_sha, model26_protocol_sha256=model26_sha))
    directory = HERE / "runs" / digest(protocol)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if not path.exists():
        write_json(path, protocol)
    elif json.loads(path.read_text()) != protocol:
        raise ValueError("Model29 protocol mismatch")
    write_json(HERE / "runs/latest.json", dict(directory=str(directory.relative_to(ROOT)),
        protocol_sha256=sha256(path)))
    return directory, protocol, sha256(path)


def frozen_study():
    latest = json.loads((HERE / "runs/latest.json").read_text())
    directory = ROOT / latest["directory"]
    protocol = json.loads((directory / "protocol.json").read_text())
    fingerprint = sha256(directory / "protocol.json")
    if fingerprint != latest["protocol_sha256"] or protocol["contract"]["config"] != CONFIG:
        raise ValueError("Changed Model29 protocol")
    for relative, expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f"Changed frozen Model29 input: {relative}")
    return directory, protocol, fingerprint


def raw_predictions(fold, held):
    model26_dir, _, model26_sha = incumbent.frozen_study()
    base = incumbent.prediction_checkpoint(model26_dir, "depth_mix", fold, model26_sha, held)
    model27_dir, model27_sha = _latest(MODEL27)
    saved = read_checkpoint(model27_dir / f"matched_context_fold{fold}.npz",
        dict(protocol_sha256=model27_sha, fold=fold, arm="matched_context"))
    if saved is None:
        raise RuntimeError(f"Model27 fold {fold} is not complete")
    validate_predictions(saved, held)
    return base, saved["prediction"]


def meta_features(base, new, row_context, time_online):
    base_logit = logit(np.clip(np.asarray(base, dtype=np.float64), 1e-5, 1-1e-5))
    new_logit = logit(np.clip(np.asarray(new, dtype=np.float64), 1e-5, 1-1e-5))
    disagreement = new_logit - base_logit
    context = np.asarray(row_context, dtype=np.float64)
    if context.ndim != 2 or len(context) != len(base):
        raise ValueError("Invalid Model29 context")
    time_feature = np.log1p(np.asarray(time_online, dtype=np.float64))[:, None]
    result = np.column_stack((base_logit, new_logit, disagreement,
        np.abs(disagreement), time_feature, context, context * disagreement[:, None]))
    if not np.isfinite(result).all():
        raise ValueError("Non-finite Model29 features")
    return result


def context_matrix(layout):
    pool = load_real_pool(layout)
    values = contexts(pool)
    matrix = np.zeros((len(layout.inner_series_fold), len(CONTEXT_NAMES)), dtype=np.float32)
    for position, value in values.items():
        matrix[position] = value
    return matrix


def fit_gate(directory, fingerprint, layout, context):
    meta = dict(protocol_sha256=fingerprint, arm="context_gate", pilot_folds=CONFIG["pilot_folds"])
    saved = read_checkpoint(directory / "gate.npz", meta)
    if saved is not None:
        return saved
    features, targets, times = [], [], []
    for fold in CONFIG["pilot_folds"]:
        held = np.flatnonzero(layout.row_fold == fold)
        base, new = raw_predictions(fold, held)
        features.append(meta_features(base, new, context[layout.series_position[held]], layout.time_online[held]))
        targets.append(layout.target[held]); times.append(layout.time_online[held])
    x = np.concatenate(features); y = np.concatenate(targets); t = np.concatenate(times)
    mean = x.mean(axis=0); scale = x.std(axis=0); scale[scale < 1e-8] = 1
    standardized = (x - mean) / scale
    weights = reference_weights(y, t, y, t)
    model = LogisticRegression(C=CONFIG["c"], max_iter=CONFIG["max_iter"], tol=CONFIG["tolerance"],
        solver="lbfgs", random_state=CONFIG["seed"])
    model.fit(standardized, y, sample_weight=weights)
    arrays = dict(mean=mean, scale=scale, coefficient=model.coef_[0],
        intercept=np.asarray(model.intercept_), iterations=np.asarray(model.n_iter_))
    commit_checkpoint(directory / "gate.npz", arrays, meta)
    return read_checkpoint(directory / "gate.npz", meta)


def gated_probability(saved, base, new, row_context, time_online):
    x = meta_features(base, new, row_context, time_online)
    margin = ((x - saved["mean"]) / saved["scale"]) @ saved["coefficient"] + saved["intercept"][0]
    probability = expit(margin).astype(np.float32)
    if not np.isfinite(probability).all():
        raise ValueError("Invalid Model29 probability")
    return probability


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("initialize", "run", "report"))
    args = parser.parse_args(); (HERE / "runs").mkdir(parents=True, exist_ok=True)
    with (HERE / "runs/active.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        directory, _, fingerprint = initialize()
        if args.command in ("run", "report"):
            layout = load_development_layout(ROOT); context = context_matrix(layout)
            fit_gate(directory, fingerprint, layout, context)
            from models.model29.report import report
            report(layout, context)


if __name__ == "__main__":
    main()
