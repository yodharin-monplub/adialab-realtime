#!/usr/bin/env python3
"""Learn a small feature-dependent correction to the strongest old stack."""
from pathlib import Path
import hashlib
import json
import sys

import lightgbm as lgb
import numpy as np
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.model10.sources import load_model6_incumbent_postprocess, load_model9_pairwise_stack
from models.model11.pipeline import CACHE, describe, prepare, sha256, write_json
from models.model11.features import FEATURE_NAMES
from tsauc import time_pair_weights, tsauc


def main():
    x, layout = prepare()
    source = load_model9_pairwise_stack(ROOT, layout=layout)
    baseline = load_model6_incumbent_postprocess(ROOT, layout=layout)
    margin = logit(np.clip(source.probability, 1e-6, 1-1e-6))
    contract = dict(version="model15-frozen-margin-residual-v1", source_sha256=sha256(__file__),
                    feature_contract=json.loads((CACHE/"manifest.json").read_text())["contract"],
                    baseline_summary_sha256=sha256(source.summary_path),
                    baseline_oof_sha256=sha256(ROOT/"models/model9/runs/fulldata_819a001db6/pairwise_stack/development_oof.margin.f32.npy"),
                    fit_rows=500_000, rounds=150, seed=20260908,
                    parameters=dict(num_leaves=7, learning_rate=.025, min_data_in_leaf=1000, lambda_l2=100),
                    interpretation="Exploratory residual stacking on Model 9 OOF; upstream base refits are not nested inside correction-held folds.")
    identity = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()[:12]
    directory = ROOT/f"models/model15/runs/{identity}"
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory/"protocol.json", contract)
    prediction = np.empty(layout.row_count, dtype=np.float32)
    scores = []
    for fold in range(5):
        held = np.flatnonzero(layout.row_fold == fold)
        output = directory/f"fold{fold}.npy"
        if output.exists():
            saved = np.load(output)
            if saved.shape != held.shape or not np.all(np.isfinite(saved)):
                raise ValueError("Invalid residual fold checkpoint")
            prediction[held] = saved
        else:
            rng = np.random.default_rng(contract["seed"]+fold)
            eligible = np.flatnonzero(layout.row_fold != fold)
            fit = np.sort(rng.choice(eligible, min(len(eligible), contract["fit_rows"]), replace=False))
            features = np.column_stack((x[fit], margin[fit]))
            weights = time_pair_weights(layout.target[fit], layout.time_online[fit])
            data = lgb.Dataset(features, label=layout.target[fit], weight=weights, init_score=margin[fit],
                               feature_name=FEATURE_NAMES+["frozen_model9_margin"])
            parameters = dict(objective="binary", metric="None", verbosity=-1, num_threads=4,
                              deterministic=True, force_col_wise=True, seed=contract["seed"]+fold,
                              max_bin=127, **contract["parameters"])
            print(f"Model 15, fold {fold}: fitting residual correction", flush=True)
            model = lgb.train(parameters, data, num_boost_round=contract["rounds"])
            model.save_model(str(directory/f"fold{fold}.txt"))
            for chunk in np.array_split(held, 10):
                block = np.column_stack((x[chunk], margin[chunk]))
                correction = model.predict(block, raw_score=True, num_threads=4)
                prediction[chunk] = expit(margin[chunk]+correction)
            np.save(output, prediction[held])
            del model, features, data
        score = tsauc(layout.target[held], prediction[held], layout.time_online[held])
        scores.append(score)
        print(f"Model 15, fold {fold}: TSAUC={score:.6f}", flush=True)
    np.save(directory/"oof.npy", prediction)
    summary = dict(model=15, name="residual_boosting", raw=describe(scores, baseline.validation.observed_fold_scores),
                   comparison_to_model9=describe(scores, source.validation.observed_fold_scores), blends=[],
                   interpretation=contract["interpretation"], sealed_holdout_labels_loaded=False, deployment_promoted=False)
    write_json(directory/"summary.json", summary)
    write_json(ROOT/"models/model15/runs/latest.json", dict(directory=str(directory.relative_to(ROOT))))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
