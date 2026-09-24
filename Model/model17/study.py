"""Small frozen Model 16/12/6 OOF blend screen; never submit or load holdout labels."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
from scipy.special import logit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.model10.sources import load_development_layout, load_model6_incumbent_postprocess
from models.model11.pipeline import describe, sha256, write_json
from models.report_next_models import WeightedTSAUC
from tsauc import tsauc

# Order is the deterministic tie-break: incumbent, simpler additions, joint blends.
RECIPES = (
    ("model16", (1., 0., 0.)),
    ("m16_95_m12_05", (.95, .05, 0.)),
    ("m16_90_m12_10", (.90, .10, 0.)),
    ("m16_95_m6_05", (.95, 0., .05)),
    ("m16_90_m6_10", (.90, 0., .10)),
    ("m16_85_m6_15", (.85, 0., .15)),
    ("m16_90_m12_05_m6_05", (.90, .05, .05)),
    ("m16_85_m12_10_m6_05", (.85, .10, .05)),
)
MINIMUM_GAIN = .00025
MINIMUM_WINS = 4
BOOTSTRAP_SEED = 170612


def blend(margins, weights):
    if len(margins) != 3 or len(weights) != 3:
        raise ValueError("Expected Model 16, Model 12, and Model 6")
    weights = np.asarray(weights, dtype=float)
    if not np.isfinite(weights).all() or np.any(weights < 0) or not np.isclose(weights.sum(), 1.):
        raise ValueError("Weights must be finite, nonnegative and sum to one")
    values = [np.asarray(m, dtype=np.float32) for m in margins]
    if any(m.ndim != 1 or m.shape != values[0].shape or not np.isfinite(m).all() for m in values):
        raise ValueError("Unaligned or nonfinite margins")
    # Accumulate in float64; the stored screening margin uses one float32 boundary.
    output = np.zeros(values[0].shape, dtype=np.float64)
    for weight, margin in zip(weights, values):
        if weight:
            output += weight * margin.astype(np.float64)
    return output.astype(np.float32)


def select_without_fold(scores, held_fold):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] < 2 or not 0 <= held_fold < scores.shape[1] or not np.isfinite(scores).all():
        raise ValueError("Invalid candidate-by-fold matrix")
    fitting = np.arange(scores.shape[1]) != held_fold
    return int(np.argmax(np.mean(scores[:, fitting], axis=1)))


def screening_gate(gain, wins, interval_low, selection_gain):
    return bool(gain >= MINIMUM_GAIN and wins >= MINIMUM_WINS and interval_low > 0 and selection_gain > 0)


def latest(number):
    directory = ROOT / json.loads((ROOT / f"models/model{number}/runs/latest.json").read_text())["directory"]
    directory.resolve().relative_to(ROOT)
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=300)
    args = parser.parse_args()
    if args.bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    m12_dir, m16_dir = latest(12), latest(16)
    paths = [m12_dir / "oof.npy", m16_dir / "oof.margin.npy", m12_dir / "summary.json",
             m16_dir / "summary.json", ROOT / "models/model5/artifacts/split_assignments.npz",
             ROOT / "models/artifacts/core_features.metadata.npz", ROOT / "models/model10/sources.py",
             ROOT / "models/report_next_models.py", ROOT / "tsauc.py", Path(__file__)]
    protocol = dict(version="model17-small-fixed-blends-v1", recipes=RECIPES, component_order=[16, 12, 6],
                    minimum_mean_gain=MINIMUM_GAIN, minimum_fold_wins=MINIMUM_WINS,
                    require_bootstrap_lower_bound_positive=True, require_leave_fold_selection_gain_positive=True,
                    bootstrap_replicates=args.bootstrap, bootstrap_seed=BOOTSTRAP_SEED,
                    source_hashes={str(p.relative_to(ROOT)): sha256(p) for p in paths},
                    interpretation="Reused development OOF only; leave-fold selection is a sensitivity check, not fully nested validation. Bootstrap conditions on saved predictions and omits adaptive selection uncertainty.")
    identity = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12]
    directory = ROOT / f"models/model17/runs/{identity}"
    if (directory / "summary.json").exists():
        print((directory / "summary.json").read_text())
        return
    # Freeze the candidate set and gates before loading target values or scoring.
    write_json(directory / "protocol.json", protocol)
    print(f"Frozen protocol: {directory.relative_to(ROOT)}", flush=True)
    layout = load_development_layout(ROOT)
    model6 = load_model6_incumbent_postprocess(ROOT, layout=layout)
    p12 = np.load(m12_dir / "oof.npy", allow_pickle=False)
    m16 = np.load(m16_dir / "oof.margin.npy", allow_pickle=False)
    if p12.shape != (layout.row_count,) or not np.isfinite(p12).all() or np.any((p12 < 0) | (p12 > 1)):
        raise ValueError("Model 12 OOF alignment/range mismatch")
    if m16.shape != (layout.row_count,) or not np.isfinite(m16).all():
        raise ValueError("Model 16 OOF alignment/range mismatch")
    masks = [layout.row_fold == f for f in range(5)]
    def score_folds(prediction):
        return [tsauc(layout.target[rows], prediction[rows], layout.time_online[rows]) for rows in masks]
    baseline = score_folds(m16)
    np.testing.assert_allclose(baseline, json.loads((m16_dir / "summary.json").read_text())["comparison_to_model6"]["fold_scores"], atol=1e-7, rtol=0)
    np.testing.assert_allclose(score_folds(p12), json.loads((m12_dir / "summary.json").read_text())["raw"]["fold_scores"], atol=1e-7, rtol=0)
    margins = [m16, logit(np.clip(p12, 1e-6, 1-1e-6)).astype(np.float32),
               logit(np.clip(model6.probability, 1e-6, 1-1e-6)).astype(np.float32)]
    candidates = []
    for name, weights in RECIPES:
        scores = describe(score_folds(blend(margins, weights)), baseline)
        candidates.append(dict(name=name, weights=list(weights), **scores))
        write_json(directory / "screen_checkpoint.json", candidates)
        print(f'{name}: {scores["fold_mean"]:.9f}; gain {scores["mean_gain"]:+.9f}; wins {scores["fold_wins"]}/5', flush=True)
    matrix = np.array([item["fold_scores"] for item in candidates])
    best_index = int(np.argmax(matrix.mean(axis=1)))
    best = candidates[best_index]
    best_prediction = blend(margins, best["weights"])
    selections = [select_without_fold(matrix, f) for f in range(5)]
    selected_scores = [matrix[index, f] for f, index in enumerate(selections)]
    selection = dict(choices=[RECIPES[i][0] for i in selections], **describe(selected_scores, baseline),
                     interpretation="The selector excludes held-fold scores; upstream models and prior research are not independent of these development folds.")
    print(f'Leave-fold selection mean gain: {selection["mean_gain"]:+.9f}', flush=True)
    np.save(directory / "best.oof.margin.npy", best_prediction)
    write_json(directory / "frozen_candidate.json", dict(name=best["name"], component_order=[16, 12, 6], weights=best["weights"],
                 selected_using="five reused development folds", deployable=False, prediction_sha256=sha256(directory / "best.oof.margin.npy")))
    scorers, fold_series = [], []
    for f, rows in enumerate(masks):
        fold_series.append(np.flatnonzero(layout.inner_series_fold == f))
        scorers.append([WeightedTSAUC(layout.target[rows], p[rows], layout.time_online[rows], layout.series_position[rows])
                        for p in (best_prediction, m16)])
    weights = np.ones(len(layout.inner_series_fold))
    np.testing.assert_allclose([pair[0].score(weights) for pair in scorers], best["fold_scores"], atol=1e-12, rtol=0)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    differences = []
    for repeat in range(args.bootstrap):
        gains = []
        for series, pair in zip(fold_series, scorers):
            weights = np.zeros(len(layout.inner_series_fold))
            weights[series] = rng.multinomial(len(series), np.full(len(series), 1 / len(series)))
            gains.append(pair[0].score(weights) - pair[1].score(weights))
        differences.append(float(np.mean(gains)))
        if (repeat + 1) % 25 == 0:
            print(f"Paired whole-series bootstrap: {repeat + 1}/{args.bootstrap}", flush=True)
    interval = np.quantile(differences, [.025, .975]).tolist()
    bootstrap = dict(replicates=args.bootstrap, standard_error=float(np.std(differences, ddof=1)),
                     percentile_95_interval=interval, interpretation=protocol["interpretation"])
    passed = screening_gate(best["mean_gain"], best["fold_wins"], interval[0], selection["mean_gain"])
    decision = ("Candidate clears the development screen; exact deployment-matched independent validation is still required."
                if passed else "Reject promotion: keep Model 16. The fixed blend study does not justify consuming the untouched holdout.")
    summary = dict(model=17, development_series=layout.series_count, development_rows=layout.row_count,
                   baseline_mean=float(np.mean(baseline)), candidates=candidates, best_diagnostic=best,
                   leave_fold_selection=selection, bootstrap=bootstrap, screening_gate_passed=passed,
                   decision=decision, sealed_holdout_labels_loaded=False, cloud_submission_created=False,
                   interpretation=protocol["interpretation"])
    write_json(directory / "summary.json", summary)
    write_json(ROOT / "models/model17/runs/latest.json", dict(directory=str(directory.relative_to(ROOT))))
    lines = ["# Model 17: fixed diversity blend screen", "", f"{layout.series_count:,} complete development series / {layout.row_count:,} online rows.", "",
             "| Model 16 / Model 12 / Model 6 weights | Mean fold TSAUC | Gain vs Model 16 | Fold wins |",
             "| --- | ---: | ---: | ---: |"]
    for item in candidates:
        label = " / ".join(f"{w:.0%}" for w in item["weights"])
        lines.append(f'| {label} | {item["fold_mean"]:.9f} | {item["mean_gain"]:+.9f} | {item["fold_wins"]}/5 |')
    lines += ["", f'Best diagnostic: {best["name"]}.',
              f'Paired whole-series bootstrap 95% interval for its gain: [{interval[0]:+.9f}, {interval[1]:+.9f}] ({args.bootstrap} replicates).',
              f'Leave-one-fold-out recipe-selection sensitivity: {selection["mean_gain"]:+.9f} mean gain; {selection["fold_wins"]}/5 wins.',
              "", "Predeclared gates: gain at least 0.00025, at least 4/5 fold wins, positive bootstrap lower endpoint, and positive leave-fold selection gain.",
              "", decision, "", protocol["interpretation"], "",
              "No holdout/test labels were loaded. No models were retrained, deployed, or submitted.", ""]
    (ROOT / "models/model17/RESULTS.md").write_text("\n".join(lines))
    print(json.dumps({"best": best, "bootstrap": bootstrap, "decision": decision}, indent=2), flush=True)


if __name__ == "__main__":
    main()
