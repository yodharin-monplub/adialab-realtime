"""Held-series OOF report for Model32."""
import numpy as np
from scipy.special import expit
from models.model11.pipeline import describe,sha256,write_json
from models.model18.report import false_alarms
from models.model18.run_model18 import commit_checkpoint,read_checkpoint
from models.model32.run_model32 import ROOT,HERE,ARMS,CONFIG,frozen_study,_state,MODEL27,MODEL30,crossfit_margin
from models.report_next_models import WeightedTSAUC
from tsauc import tsauc

def report(layout,base_margin,features,models):
    directory,_,fingerprint=frozen_study(); margins=[base_margin,crossfit_margin(models,layout,features)]
    scores=np.asarray([[tsauc(layout.target[layout.row_fold==f],m[layout.row_fold==f],layout.time_online[layout.row_fold==f]) for f in CONFIG["folds"]] for m in margins])
    gain=float(scores[1].mean()-scores[0].mean()); wins=int(np.sum(scores[1]>scores[0])); half=(layout.series_ids*0x9E3779B1)&1; split=[]
    for value in (0,1):
        rows=(layout.row_fold>=0)&(half[layout.series_position]==value)
        split.append(float(tsauc(layout.target[rows],margins[1][rows],layout.time_online[rows])-tsauc(layout.target[rows],margins[0][rows],layout.time_online[rows])))
    d27,_=_state(MODEL27,CONFIG["model27_protocol_sha256"]); d30,_=_state(MODEL30,CONFIG["model30_protocol_sha256"])
    hashes={f"m27_{f}":sha256(d27/f"matched_context_fold{f}.npz") for f in CONFIG["folds"]}; hashes.update({f"m30_{f}":sha256(d30/f"matched_15_fold{f}.npz") for f in CONFIG["folds"]}); hashes.update({f"meta_{f}":sha256(directory/f"meta_fold{f}.npz") for f in CONFIG["folds"]})
    meta=dict(protocol_sha256=fingerprint,predictions=hashes); saved=read_checkpoint(directory/"bootstrap.npz",meta); boot=[] if saved is None else saved["gains"].tolist(); scorers=[]
    for fold in CONFIG["folds"]:
        rows=layout.row_fold==fold; series=np.flatnonzero(layout.inner_series_fold==fold)
        scorers.append((series,[WeightedTSAUC(layout.target[rows],m[rows],layout.time_online[rows],layout.series_position[rows]) for m in margins]))
    rng=np.random.default_rng(CONFIG["bootstrap_seed"]); completed=len(boot)
    for repeat in range(CONFIG["bootstrap_replicates"]):
        local=[]
        for series,pair in scorers:
            mult=rng.multinomial(len(series),np.full(len(series),1/len(series)))
            if repeat<completed: continue
            w=np.zeros(len(layout.inner_series_fold));w[series]=mult;local.append([s.score(w) for s in pair])
        if repeat<completed: continue
        local=np.asarray(local).mean(axis=0);boot.append(float(local[1]-local[0]))
        if (repeat+1)%25==0:commit_checkpoint(directory/"bootstrap.npz",dict(gains=np.asarray(boot)),meta)
    lower=float(np.quantile(boot,(1-CONFIG["confidence"])/2));passed=bool(scores[1].mean()>=CONFIG["target_tsauc"] and gain>=CONFIG["minimum_gain"] and wins>=CONFIG["minimum_fold_wins"] and lower>0 and min(split)>0)
    results={arm:describe(scores[i],scores[0]) for i,arm in enumerate(ARMS)};results["crossfit_stack"].update(fold_wins=wins,bootstrap_lower_95=lower,split_half_gains=split,passes=passed)
    summary=dict(model=32,selected_candidate="crossfit_stack" if passed else None,target_tsauc=CONFIG["target_tsauc"],target_reached=bool(scores[1].mean()>=CONFIG["target_tsauc"]),results=results,interpretation="Every meta prediction is produced by a model fitted and standardized on the other four OOF folds.",deployment="No packaging, upload, or cloud run performed.")
    write_json(directory/"summary.json",summary);lines=["# Model32: five-fold cross-fitted diversity stack","","| Arm | Held-series OOF | Gain | Wins | Lower 95% | Split gains | Pass |","|---|---:|---:|---:|---:|---|---|"]
    for arm in ARMS:
        r=results[arm];lines.append(f"| {arm} | {r['fold_mean']:.6f} | {r['mean_gain']:+.6f} | {r.get('fold_wins',0)}/5 | {r.get('bootstrap_lower_95',0):+.6f} | {r.get('split_half_gains','baseline')} | {r.get('passes','baseline')} |")
    lines += ["",f"Target held-series OOF TSAUC: {CONFIG['target_tsauc']:.3f}; reached: {summary['target_reached']}.","",summary["interpretation"],"","These are local OOF scores, not cloud scores.","",summary["deployment"],""]
    text="\n".join(lines);(directory/"RESULTS.md").write_text(text);(HERE/"RESULTS.md").write_text(text);print(text,flush=True);return summary
