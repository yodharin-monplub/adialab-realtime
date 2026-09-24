"""Run Model104's nested design with nearest-time sparse-tail pooling."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model11.pipeline import sha256, write_json
from models.model104 import run_model104 as engine

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "prespec.json").read_text())
if sha256(ROOT / "models/model104/run_model104.py") != CONFIG["model104_engine_sha256"]:
    raise ValueError("Changed Model105 engine source")


def pooled_null_tail_feature(layout, evidence, reference_folds, apply_rows, rows_by_time):
    apply_rows = np.asarray(apply_rows, dtype=np.int64)
    result = np.empty(len(apply_rows), dtype=np.float32)
    apply_time = layout.time_online[apply_rows]
    apply_order = np.argsort(apply_time, kind="stable")
    sorted_apply_time = apply_time[apply_order]
    maximum = len(rows_by_time) - 1
    for step in np.unique(sorted_apply_time):
        lo, hi = np.searchsorted(sorted_apply_time, (step, step + 1))
        local = apply_order[lo:hi]
        blocks, count = [], 0
        for radius in range(maximum + 1):
            candidates = [int(step)] if radius == 0 else [int(step) - radius, int(step) + radius]
            for nearby in candidates:
                if nearby < 0 or nearby > maximum:
                    continue
                time_rows = rows_by_time[nearby]
                reference_rows = time_rows[
                    np.isin(layout.row_fold[time_rows], reference_folds) &
                    (layout.target[time_rows] == 0)]
                if len(reference_rows):
                    blocks.append(np.asarray(evidence[reference_rows], dtype=np.float64))
                    count += len(reference_rows)
            if count >= CONFIG["minimum_null_references"]:
                break
        if count < CONFIG["minimum_null_references"]:
            raise ValueError(f"Insufficient pooled Model105 null reference at step {step}: {count}")
        reference = np.sort(np.concatenate(blocks))
        observed = np.asarray(evidence[apply_rows[local]], dtype=np.float64)
        left = np.searchsorted(reference, observed, side="left")
        right = np.searchsorted(reference, observed, side="right")
        exceed = len(reference) - 0.5 * (left + right)
        probability = (exceed + 0.5) / (len(reference) + 1.0)
        result[local] = np.minimum(-np.log(probability), CONFIG["tail_clip"])
    if not np.isfinite(result).all():
        raise ValueError("Invalid Model105 pooled empirical-null feature")
    return result


engine.HERE = HERE
engine.CONFIG = CONFIG
engine.PARENT84 = ROOT / CONFIG["model84_run"]
engine.PARENT95 = ROOT / CONFIG["model95_run"]
engine.null_tail_feature = pooled_null_tail_feature


if __name__ == "__main__":
    engine.main()
    latest = json.loads((HERE / "runs/latest.json").read_text())
    directory = ROOT / latest["directory"]
    summary_path = directory / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        summary["model"] = 105
        summary["sparse_tail_calibration"] = (
            "Nearest online times pooled to at least 500 null references.")
        write_json(summary_path, summary)
    for path in (HERE / "RESULTS.md", directory / "RESULTS.md"):
        if path.exists():
            text = path.read_text().replace(
                "# Model104: nested cross-fitted empirical-null scale GLR",
                "# Model105: pooled nested empirical-null scale GLR")
            path.write_text(text)
