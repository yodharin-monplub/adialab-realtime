"""Evaluation of the equal depth-3/4/5 Model43 ensemble."""
import numpy as np
from scipy.special import expit
from models.model10.causal_transforms import apply_causal_transform
from models.model10.sources import load_development_layout
from models.model11.pipeline import describe, sha256, write_json
from models.model18.report import false_alarms
from models.model18.run_model18 import commit_checkpoint, read_checkpoint
from models.model20.report import ensemble_margin
from models.model26 import run_model26 as incumbent
from models.model38.run_model38 import transform_spec
from models.model43.run_model43 import CONFIG, HERE, ROOT, equal_depth_probability, frozen_study
from models.report_next_models import WeightedTSAUC
from tsauc import tsauc


def report():
    directory, _, fingerprint = frozen_study(); layout = load_development_layout(ROOT)
    d26, _, s26 = incumbent.frozen_study(); p26 = np.empty(layout.row_count, np.float32); pequal = np.empty_like(p26)
    for fold in range(5):
        held = np.flatnonzero(layout.row_fold == fold)
        p26[held] = incumbent.prediction_checkpoint(d26, "depth_mix", fold, s26, held)
        pequal[held] = equal_depth_probability(directory, fold, fingerprint, held)
    base = ensemble_margin(p26, layout); raw = ensemble_margin(pequal, layout)
    smooth = apply_causal_transform(expit(raw.astype(float)), layout.series_position,
        transform_spec(), layout.time_online).astype(np.float32)
    values = (base, raw, smooth); names = ("model26", "equal_depths_raw", "equal_depths_smoothed")
    scores = np.asarray([[tsauc(layout.target[layout.row_fold == f], m[layout.row_fold == f],
        layout.time_online[layout.row_fold == f]) for f in range(5)] for m in values])
    metadata = dict(protocol_sha256=fingerprint,
        predictions={p.name: sha256(p) for p in sorted(directory.glob("depth5_*.npz"))})
    saved = read_checkpoint(directory / "bootstrap.npz", metadata); boot = [] if saved is None else saved["gains"].tolist()
    scorers = []
    for f in range(5):
        rows = layout.row_fold == f; series = np.flatnonzero(layout.inner_series_fold == f)
        scorers.append((series, [WeightedTSAUC(layout.target[rows], m[rows], layout.time_online[rows],
                                               layout.series_position[rows]) for m in values]))
    rng = np.random.default_rng(CONFIG["bootstrap_seed"]); completed = len(boot)
    for repeat in range(CONFIG["bootstrap_replicates"]):
        local=[]
        for series, triple in scorers:
            multiplicity=rng.multinomial(len(series),np.full(len(series),1/len(series)))
            if repeat < completed: continue
            weights=np.zeros(len(layout.inner_series_fold)); weights[series]=multiplicity
            local.append([s.score(weights) for s in triple])
        if repeat < completed: continue
        mean=np.asarray(local).mean(axis=0); boot.append([float(mean[1]-mean[0]),float(mean[2]-mean[0])])
        if (repeat+1)%25==0: commit_checkpoint(directory/"bootstrap.npz",dict(gains=np.asarray(boot)),metadata)
    lower=np.quantile(np.asarray(boot),.025,axis=0); half=(layout.series_ids*0x9E3779B1)&1; split=[]
    for value in (0,1):
        rows=half[layout.series_position]==value
        split.append(float(tsauc(layout.target[rows],smooth[rows],layout.time_online[rows])-
                           tsauc(layout.target[rows],base[rows],layout.time_online[rows])))
    gain=float(scores[2].mean()-scores[0].mean()); wins=int(np.sum(scores[2]>scores[0]))
    passed=bool(scores[2].mean()>=CONFIG["target_tsauc"] and gain>=CONFIG["minimum_gain"]
                and wins>=CONFIG["minimum_fold_wins"] and lower[1]>0 and min(split)>0)
    results={name:describe(score,scores[0]) for name,score in zip(names,scores)}
    results["equal_depths_smoothed"].update(fold_wins=wins,bootstrap_lower_95=float(lower[1]),
        split_half_gains=split,passes=passed,
        no_break_false_alarms=false_alarms(expit(smooth.astype(float)),layout,[.5,.8]))
    summary=dict(model=43,target_tsauc=CONFIG["target_tsauc"],target_reached=bool(scores[2].mean()>=CONFIG["target_tsauc"]),
        selected_candidate="equal_depths_smoothed" if passed else None,results=results,
        interpretation="Predeclared equal probability mean of depth-3/4/5 voter pairs; confirmed smoother applied.",
        deployment="No packaging, upload, or cloud run performed.")
    write_json(directory/"summary.json",summary)
    lines=["# Model43: equal depth-3/4/5 diversity", "",
        "| Arm | Held-series OOF | Gain | Wins | Lower 95% | Split gains | Pass |",
        "|---|---:|---:|---:|---:|---|---|"]
    for i,name in enumerate(names):
        r=results[name]; lines.append(f"| {name} | {r['fold_mean']:.6f} | {r['mean_gain']:+.6f} | "
            f"{wins if i==2 else int(np.sum(scores[i]>scores[0]))}/5 | {lower[i-1] if i else 0:+.6f} | "
            f"{split if i==2 else 'diagnostic'} | {passed if i==2 else 'diagnostic'} |")
    lines += ["",f"Target: {CONFIG['target_tsauc']:.3f}; reached: {summary['target_reached']}.","",summary["interpretation"],"",
              "These are local development OOF scores, not cloud scores.","",summary["deployment"],""]
    text="\n".join(lines); (directory/"RESULTS.md").write_text(text); (HERE/"RESULTS.md").write_text(text)
    print(text,flush=True); return summary

