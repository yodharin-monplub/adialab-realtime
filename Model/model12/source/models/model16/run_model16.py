#!/usr/bin/env python3
"""Fixed combination of two independently implemented causal corrections."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
from scipy.special import logit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.model10.causal_transforms import TransformSpec, apply_transform_grid
from models.model10.sources import load_development_layout, load_model6_incumbent_postprocess, load_model9_pairwise_stack
from models.model11.pipeline import describe, sha256, write_json
from tsauc import tsauc


def main():
    layout = load_development_layout(ROOT)
    model9 = load_model9_pairwise_stack(ROOT, layout=layout)
    model6 = load_model6_incumbent_postprocess(ROOT, layout=layout)
    previous_path = ROOT/"models/model10/runs/postprocess_e4ef830f0e/summary.json"
    previous = next(r for r in json.loads(previous_path.read_text())["reports"] if r["source"] == model9.name)["best_fixed_candidate_diagnostic"]
    if previous["name"] != "rolling_max_n128_w0p1":
        raise ValueError("Historical diagnostic differs from the expected recipe")
    latest = json.loads((ROOT/"models/model11/runs/latest.json").read_text())
    path = ROOT/latest["directory"]/"oof.npy"
    probability = np.load(path)
    if probability.shape != (layout.row_count,) or not np.all(np.isfinite(probability)) or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Model 11 OOF alignment or range mismatch")
    spec = TransformSpec(kind="rolling_max_premium", window=128, correction_weight=.1)
    _, transformed = next(apply_transform_grid(model9.probability, layout.time_online, layout.series_position, [spec]))
    historical_scores = [tsauc(layout.target[layout.row_fold == f], transformed[layout.row_fold == f], layout.time_online[layout.row_fold == f]) for f in range(5)]
    np.testing.assert_allclose(historical_scores, previous["outer_fold_scores"], atol=1e-7, rtol=0)
    prediction = (.9*transformed + .1*logit(np.clip(probability, 1e-6, 1-1e-6))).astype(np.float32)
    contract = dict(version="model16-persistence-diversity-v1", new_model_weight=.1, rolling_window=128, rolling_weight=.1,
                    model11_oof_sha256=sha256(path), model10_summary_sha256=sha256(previous_path),
                    model9_oof_sha256=sha256(ROOT/"models/model9/runs/fulldata_819a001db6/pairwise_stack/development_oof.margin.f32.npy"),
                    source_sha256=sha256(__file__), transform_source_sha256=sha256(ROOT/"models/model10/causal_transforms.py"),
                    split_sha256=sha256(ROOT/"models/model5/artifacts/split_assignments.npz"),
                    target_index_sha256=sha256(ROOT/"train/y_train_index.parquet"),
                    interpretation="Post-hoc combination of reused-development diagnostics. Model 10 rejected promotion; Model 9 base refits remain not fully nested. Not confirmatory.")
    identity = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()[:12]
    directory = ROOT/f"models/model16/runs/{identity}"
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory/"protocol.json", contract)
    np.save(directory/"oof.margin.npy", prediction)
    scores = [tsauc(layout.target[layout.row_fold == f], prediction[layout.row_fold == f], layout.time_online[layout.row_fold == f]) for f in range(5)]
    summary = dict(model=16, comparison_to_model6=describe(scores, model6.validation.observed_fold_scores),
                   comparison_to_model9=describe(scores, model9.validation.observed_fold_scores),
                   comparison_to_historical_model10_diagnostic=describe(scores, historical_scores),
                   interpretation=contract["interpretation"], sealed_holdout_labels_loaded=False, deployment_promoted=False)
    write_json(directory/"summary.json", summary)
    write_json(ROOT/"models/model16/runs/latest.json", dict(directory=str(directory.relative_to(ROOT))))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
