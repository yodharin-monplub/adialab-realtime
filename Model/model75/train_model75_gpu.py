"""Train frozen Model75 folds on a CUDA-capable remote host."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tsauc import tsauc

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "prespec.json").read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = Path(path).with_suffix(Path(path).suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def state():
    latest = json.loads((HERE / "runs/latest.json").read_text())
    directory = ROOT / latest["directory"]
    protocol_path = directory / "protocol.json"
    if sha256(protocol_path) != latest["protocol_sha256"]:
        raise ValueError("Changed remote Model75 protocol")
    protocol = json.loads(protocol_path.read_text())
    if protocol["contract"]["config"] != CONFIG:
        raise ValueError("Changed remote Model75 configuration")
    prepared = json.loads((directory / "prepared.json").read_text())
    if prepared["protocol_sha256"] != latest["protocol_sha256"]:
        raise ValueError("Wrong prepared Model75 protocol")
    for item in [prepared["layout"], *prepared["synthetic"]]:
        if sha256(directory / item["path"]) != item["sha256"]:
            raise ValueError(f"Corrupt prepared artifact: {item['path']}")
    return directory, protocol, latest["protocol_sha256"]


def reference_weights(target, time_online):
    size = int(time_online.max(initial=0)) + 1
    counts = np.bincount(time_online, minlength=size)
    positive = np.bincount(time_online, weights=target, minlength=size)
    negative = counts - positive
    weights = np.where(target == 1, negative[time_online], positive[time_online]).astype(np.float64)
    useful = weights > 0
    if useful.any():
        weights[useful] /= weights[useful].mean()
    return weights.astype(np.float32)


def splice_weights(original, real_y, real_t, synthetic_y, synthetic_t, share):
    real_cell = 2 * real_t.astype(np.int64) + real_y
    synthetic_cell = 2 * synthetic_t.astype(np.int64) + synthetic_y
    size = int(max(real_cell.max(initial=0), synthetic_cell.max(initial=0))) + 1
    mass = np.bincount(real_cell, weights=original, minlength=size)
    count = np.bincount(synthetic_cell, minlength=size)
    real_weight = (original * np.where(count[real_cell] > 0, 1 - share, 1)).astype(np.float32)
    synthetic_weight = np.divide(share * mass, count,
        out=np.zeros(size), where=count > 0)[synthetic_cell].astype(np.float32)
    after = (np.bincount(real_cell, weights=real_weight, minlength=size) +
             np.bincount(synthetic_cell, weights=synthetic_weight, minlength=size))
    np.testing.assert_allclose(after, mass, rtol=2e-7, atol=1e-5)
    return real_weight, synthetic_weight, float(synthetic_weight.sum(dtype=np.float64) /
        original.sum(dtype=np.float64))


def fill_real(destination, rows, base, white, base_columns, white_columns):
    width = int(CONFIG["prediction_batch_rows"])
    for left in range(0, len(rows), width):
        selected = rows[left:left + width]
        destination[left:left + len(selected), :len(base_columns)] = base[selected][:, base_columns]
        destination[left:left + len(selected), len(base_columns):] = white[selected][:, white_columns]


def checkpoint(directory, fingerprint, fold, prediction, held, booster, details):
    model_path = directory / f"{CONFIG['arm']}_fold{fold}.ubj"
    booster.save_model(model_path)
    detail_path = directory / f"{CONFIG['arm']}_fold{fold}.fit.json"
    write_json(detail_path, details)
    path = directory / f"{CONFIG['arm']}_fold{fold}.npz"
    temporary = directory / f"{CONFIG['arm']}_fold{fold}.partial.npz"
    np.savez_compressed(temporary, prediction=prediction.astype(np.float32), held_rows=held)
    temporary.replace(path)
    write_json(path.with_suffix(".json"), {
        "metadata": {"protocol_sha256": fingerprint, "fold": fold, "arm": CONFIG["arm"]},
        "sha256": sha256(path),
        "artifacts": {model_path.name: sha256(model_path), detail_path.name: sha256(detail_path)},
    })


def train_fold(directory, protocol, fingerprint, layout, fold):
    path = directory / f"{CONFIG['arm']}_fold{fold}.npz"
    if path.exists() and path.with_suffix(".json").exists():
        print(f"Model75 fold {fold} already complete", flush=True)
        return
    row_fold = layout["row_fold"]
    fitting = (row_fold != fold) & (row_fold != (fold + 1) % 5)
    fit = np.flatnonzero(fitting)
    held = np.flatnonzero(row_fold == fold)
    target, online = layout["target"], layout["time_online"]

    contract = protocol["contract"]
    base_names = contract["base_names"]
    white_names = contract["union_names"][len(base_names):]
    names = contract["fold_names"][fold]
    if any(name not in base_names for name in names[:len(base_names)]):
        raise ValueError("Unexpected Model75 base feature order")
    if names[len(base_names):] != white_names:
        raise ValueError("Unexpected Model75 white feature order")
    base_columns = np.asarray([base_names.index(name) for name in names[:len(base_names)]])
    white_columns = np.asarray([white_names.index(name) for name in names[len(base_names):]])

    parent20 = ROOT / CONFIG["model20_run"]
    parent57 = ROOT / CONFIG["model57_run"]
    parent64 = ROOT / CONFIG["model64_run"]
    real_base = np.load(parent20 / "real_features.npy", mmap_mode="r", allow_pickle=False)
    real_white = np.load(parent57 / "real_white.npy", mmap_mode="r", allow_pickle=False)
    with np.load(directory / f"synthetic_fold{fold}.npz", allow_pickle=False) as saved:
        synthetic_y = saved["target"].astype(np.int8)
        synthetic_t = saved["time_online"].astype(np.int16)
    synthetic_base = np.load(parent20 / f"aligned_25_fold{fold}_features.npy",
        mmap_mode="r", allow_pickle=False)
    synthetic_missing = np.load(parent64 / f"aligned_40_fold{fold}_missing.npy",
        mmap_mode="r", allow_pickle=False)
    synthetic_white = np.load(parent57 / f"aligned_25_fold{fold}_white.npy",
        mmap_mode="r", allow_pickle=False)

    real_count = len(fit)
    x = np.empty((real_count + len(synthetic_y), 500), dtype=np.float32)
    fill_real(x[:real_count], fit, real_base, real_white, base_columns, white_columns)
    x[real_count:, :256] = synthetic_base
    x[real_count:, 256:439] = synthetic_missing
    x[real_count:, 439:] = synthetic_white
    real_y, real_t = target[fit], online[fit]
    base_weight = reference_weights(real_y, real_t)
    real_weight, synthetic_weight, actual_share = splice_weights(base_weight,
        real_y, real_t, synthetic_y, synthetic_t, CONFIG["synthetic_mass"])
    y = np.r_[real_y, synthetic_y].astype(np.int8)
    weight = np.r_[real_weight, synthetic_weight].astype(np.float32)

    params = dict(CONFIG["xgboost"])
    params.update(seed=20337784 + fold * 28079, nthread=max(1, min(16, os.cpu_count() or 1)))
    started = time.monotonic()
    print(f"Fitting Model75 fold {fold}: {len(y):,} rows x {x.shape[1]} features", flush=True)
    matrix = xgb.QuantileDMatrix(x, label=y, weight=weight, max_bin=params["max_bin"])
    booster = xgb.train(params, matrix, num_boost_round=CONFIG["rounds"])
    del matrix, y, weight

    prediction = np.empty(len(held), dtype=np.float32)
    width = int(CONFIG["prediction_batch_rows"])
    for left in range(0, len(held), width):
        rows = held[left:left + width]
        hx = np.empty((len(rows), 500), dtype=np.float32)
        fill_real(hx, rows, real_base, real_white, base_columns, white_columns)
        prediction[left:left + len(rows)] = booster.inplace_predict(hx)
    elapsed = time.monotonic() - started
    score = tsauc(target[held], prediction, online[held])
    details = {
        "fold": fold,
        "component_tsauc": score,
        "real_rows": real_count,
        "synthetic_rows": len(synthetic_y),
        "synthetic_weight_share": actual_share,
        "rounds": CONFIG["rounds"],
        "elapsed_seconds": elapsed,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "xgboost_version": xgb.__version__,
    }
    checkpoint(directory, fingerprint, fold, prediction, held, booster, details)
    print(f"Finished Model75 fold {fold}: component={score:.9f}, elapsed={elapsed:.1f}s", flush=True)
    del booster, x, prediction
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    if not set(args.folds).issubset(set(range(5))):
        raise ValueError("Folds must be in 0..4")
    if xgb.__version__ != CONFIG["xgboost_version"]:
        raise ValueError(f"Expected XGBoost {CONFIG['xgboost_version']}, found {xgb.__version__}")
    subprocess.run(["nvidia-smi"], check=True, stdout=sys.stdout, stderr=sys.stderr)
    directory, protocol, fingerprint = state()
    with np.load(directory / "layout.npz", allow_pickle=False) as saved:
        layout = {name: saved[name] for name in saved.files}
    for fold in args.folds:
        train_fold(directory, protocol, fingerprint, layout, fold)


if __name__ == "__main__":
    main()
