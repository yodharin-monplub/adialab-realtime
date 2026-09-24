"""Summarize completed candidates and bootstrap whole development series."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
from scipy.special import logit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.model10.sources import load_development_layout, load_model6_incumbent_postprocess, load_model9_pairwise_stack
from models.model11.pipeline import write_json
from models.model10.causal_transforms import TransformSpec, apply_transform_grid


class WeightedTSAUC:
    """Exact rank scorer for bootstrap multiplicities, including tied scores."""
    def __init__(self, target, score, step, series):
        order = np.lexsort((score, step))
        y, s, t = target[order], score[order], step[order]
        self.series = series[order]
        self.y = y
        self.tie = np.flatnonzero(np.r_[True, (t[1:] != t[:-1]) | (s[1:] != s[:-1])])
        group_time = t[self.tie]
        self.time = np.flatnonzero(np.r_[True, group_time[1:] != group_time[:-1]])
        self.time_group = np.repeat(np.arange(len(self.time)), np.diff(np.r_[self.time, len(self.tie)]))

    def score(self, series_weight):
        weight = series_weight[self.series]
        positive = np.add.reduceat(weight*self.y, self.tie)
        negative = np.add.reduceat(weight*(1-self.y), self.tie)
        before = np.cumsum(negative) - negative
        before -= before[self.time][self.time_group]
        numerator = np.sum(positive*(before + .5*negative))
        denominator = np.sum(np.add.reduceat(positive, self.time) * np.add.reduceat(negative, self.time))
        return float(numerator/denominator) if denominator else .5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=200)
    args = parser.parse_args()
    if args.bootstrap < 2:
        raise ValueError("At least two bootstrap replicates are required")
    layout = load_development_layout(ROOT)
    baselines = [load_model6_incumbent_postprocess(ROOT, layout=layout), load_model9_pairwise_stack(ROOT, layout=layout)]
    candidates = []
    baseline_by_name = {source.name: source for source in baselines}
    for number in (11, 12, 13, 14, 15, 16):
        latest = ROOT / f"models/model{number}/runs/latest.json"
        if not latest.exists():
            raise RuntimeError(f"Model {number} has not finished")
        directory = ROOT / json.loads(latest.read_text())["directory"]
        summary = json.loads((directory/"summary.json").read_text())
        if number in (14, 16):
            name = "Model 14 time-varying stack" if number == 14 else "Model 16 persistence + diversity"
            candidates.append(dict(model=number, name=name, scores=summary["comparison_to_model6"],
                                   path=str((directory/"oof.margin.npy").relative_to(ROOT)), kind="margin"))
            continue
        name = "Model 15 residual correction" if number == 15 else f"Model {number} standalone"
        candidates.append(dict(model=number, name=name, scores=summary["raw"],
                               path=str((directory/"oof.npy").relative_to(ROOT)), kind="probability"))
        for blend in summary["blends"]:
            candidates.append(dict(model=number, name=f'{blend["source"]} + {blend["new_model_weight"]:.0%} Model {number}',
                                   scores=blend["comparison_to_source"], path=str((directory/"oof.npy").relative_to(ROOT)),
                                   kind="blend", source=blend["source"], alpha=blend["new_model_weight"]))
    candidates.sort(key=lambda item: item["scores"]["fold_mean"], reverse=True)
    for number in (11, 12, 13, 14, 15, 16):
        own = [item for item in candidates if item["model"] == number]
        lines = [f"# Model {number} results", "", "| Candidate | Mean fold TSAUC | Fold SE |", "|---|---:|---:|"]
        for item in own:
            score = item["scores"]
            lines.append(f'| {item["name"]} | {score["fold_mean"]:.6f} | {score["fold_standard_error"]:.6f} |')
        lines += ["", "All five complete-series development folds finished. No outer-holdout",
                  "or competition test labels were loaded. These are local development",
                  "diagnostics; stacking/blending inherits upstream adaptive selection.", "",
                  "No deployment was promoted. See `../NEXT_RESULTS.md` for comparisons",
                  "and the best candidate's conditional whole-series bootstrap.", ""]
        (ROOT/f"models/model{number}/RESULTS.md").write_text("\n".join(lines))
    best = candidates[0]
    prediction = np.load(ROOT/best["path"])
    if best["kind"] != "margin":
        prediction = logit(np.clip(prediction, 1e-6, 1-1e-6))
    if best["kind"] == "blend":
        baseline = baseline_by_name[best["source"]]
        prediction = ((1-best["alpha"])*logit(np.clip(baseline.probability, 1e-6, 1-1e-6)) + best["alpha"]*prediction).astype(np.float32)
    prior_path = ROOT/"models/model10/runs/postprocess_e4ef830f0e/summary.json"
    prior = next(r for r in json.loads(prior_path.read_text())["reports"] if r["source"] == baselines[1].name)["best_fixed_candidate_diagnostic"]
    spec = TransformSpec(kind="rolling_max_premium", window=128, correction_weight=.1)
    _, prior_margin = next(apply_transform_grid(baselines[1].probability, layout.time_online, layout.series_position, [spec]))
    baseline_names = [s.name for s in baselines] + ["historical_model10_rejected_diagnostic"]
    baseline_scores = [s.validation.observed_fold_scores for s in baselines] + [prior["outer_fold_scores"]]
    baseline_predictions = [s.probability for s in baselines] + [prior_margin]
    best_scores = []
    scorers, fold_series = [], []
    for fold in range(5):
        rows = layout.row_fold == fold
        fold_series.append(np.flatnonzero(layout.inner_series_fold == fold))
        scorers.append([WeightedTSAUC(layout.target[rows], p[rows], layout.time_online[rows], layout.series_position[rows])
                        for p in [prediction, *baseline_predictions]])
        best_scores.append(scorers[-1][0].score(np.ones(len(layout.inner_series_fold))))
    np.testing.assert_allclose(best_scores, best["scores"]["fold_scores"], atol=1e-7, rtol=0)
    rng = np.random.default_rng(20260907)
    differences = []
    for repeat in range(args.bootstrap):
        fold_scores = []
        for series, fold_scorers in zip(fold_series, scorers):
            weight = np.zeros(len(layout.inner_series_fold))
            weight[series] = rng.multinomial(len(series), np.full(len(series), 1/len(series)))
            fold_scores.append([scorer.score(weight) for scorer in fold_scorers])
        mean = np.mean(fold_scores, axis=0)
        differences.append(mean[0]-mean[1:])
        if (repeat+1)%25 == 0:
            print(f"Whole-series bootstrap: {repeat+1}/{args.bootstrap}", flush=True)
    differences = np.array(differences)
    bootstrap = {}
    for i, name in enumerate(baseline_names):
        bootstrap[name] = dict(observed_gain=float(np.mean(best_scores)-np.mean(baseline_scores[i])),
                                      standard_error=float(np.std(differences[:, i], ddof=1)),
                                      percentile_95_interval=np.quantile(differences[:, i], [.025, .975]).tolist(),
                                      replicates=args.bootstrap)
    result = dict(best_diagnostic=best, bootstrap=bootstrap, ranked_candidates=candidates,
                  bootstrap_interpretation="Paired complete-series resampling within original folds, conditional on saved predictions; excludes upstream refitting and adaptive model-selection uncertainty.",
                  decision="Keep Model 6 deployment. Candidate evidence remains exploratory; no submission was made.",
                  sealed_holdout_labels_loaded=False)
    write_json(ROOT/"models/next_models_results.json", result)
    lines = ["# Models 11–16 results", "", "CPU development experiments on 8,333 complete series / 4,195,720 online rows.",
             "", "| Candidate | Mean fold TSAUC | Fold SE |", "|---|---:|---:|",
             "| Model 6 incumbent | 0.625675 | 0.008440 |", "| Model 9 exploratory stack | 0.628020 | 0.007603 |",
             f'| Model 10 historical rejected diagnostic | {prior["outer_fold_mean"]:.6f} | {prior["outer_fold_standard_error"]:.6f} |']
    for entry in candidates:
        score = entry["scores"]
        lines.append(f'| {entry["name"]} | {score["fold_mean"]:.6f} | {score["fold_standard_error"]:.6f} |')
    lines += ["", "Best observed candidate: " + best["name"] + ".", "",
              "Paired bootstrap resamples complete series within each original fold:", "",
              "| Reference | Gain | Bootstrap SE | Percentile 95% interval |", "|---|---:|---:|---:|"]
    for name, value in bootstrap.items():
        low, high = value["percentile_95_interval"]
        lines.append(f'| {name} | {value["observed_gain"]:+.6f} | {value["standard_error"]:.6f} | [{low:+.6f}, {high:+.6f}] |')
    lines += ["", result["bootstrap_interpretation"], "", result["decision"], "",
              "The outer holdout and competition test labels were not loaded. These are local",
              "development scores, not leaderboard scores or evidence of a prize-winning rank.", "",
              "Reproduce: `python3 models/run_next_models.py run`, then",
              "`python3 models/model14/run_model14.py`, then",
              "`python3 models/model15/run_model15.py`, then",
              "`python3 models/model16/run_model16.py`, then",
              "`python3 models/report_next_models.py --bootstrap 200`.", ""]
    (ROOT/"models/NEXT_RESULTS.md").write_text("\n".join(lines))
    print(json.dumps({"best": best, "bootstrap": bootstrap, "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
