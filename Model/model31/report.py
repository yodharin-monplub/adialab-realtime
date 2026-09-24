"""Confirmation report for the direct Model31 meta-stack."""
import json
import numpy as np
from scipy.special import expit
from models.model11.pipeline import describe,sha256,write_json
from models.model18.report import false_alarms
from models.model18.run_model18 import commit_checkpoint,read_checkpoint
from models.model31.run_model31 import (ROOT,HERE,ARMS,CONFIG,frozen_study,
    model30_state,model30_prediction,fit_meta,meta_margin)
from models.report_next_models import WeightedTSAUC
from tsauc import tsauc

def report(layout,base_margin):
    directory,_,fingerprint=frozen_study(); fitted=fit_meta(directory,fingerprint,layout,base_margin)
    margins=[base_margin.copy(),np.empty(layout.row_count,dtype=np.float32)]
    for fold in range(5):
        held=np.flatnonzero(layout.row_fold==fold)
        margins[1][held]=meta_margin(fitted,base_margin[held],model30_prediction(fold,held),layout.time_online[held])
    scores=np.asarray([[tsauc(layout.target[layout.row_fold==f],m[layout.row_fold==f],layout.time_online[layout.row_fold==f]) for f in range(5)] for m in margins])
    confirm=np.asarray(CONFIG["confirmation_folds"]); gain=float(scores[1,confirm].mean()-scores[0,confirm].mean()); wins=int(np.sum(scores[1,confirm]>scores[0,confirm]))
    half=(layout.series_ids*0x9E3779B1)&1; split=[]
    for value in (0,1):
        rows=np.isin(layout.row_fold,confirm)&(half[layout.series_position]==value)
        split.append(float(tsauc(layout.target[rows],margins[1][rows],layout.time_online[rows])-tsauc(layout.target[rows],margins[0][rows],layout.time_online[rows])))
    model30_dir,_=model30_state(); hashes={f"fold{f}":sha256(model30_dir/f"matched_15_fold{f}.npz") for f in range(5)}
    meta=dict(protocol_sha256=fingerprint,predictions=hashes,model=sha256(directory/"meta.npz")); saved=read_checkpoint(directory/"bootstrap.npz",meta)
    boot=[] if saved is None else saved["gains"].tolist(); scorers=[]
    for fold in confirm:
        rows=layout.row_fold==fold; series=np.flatnonzero(layout.inner_series_fold==fold)
        scorers.append((series,[WeightedTSAUC(layout.target[rows],m[rows],layout.time_online[rows],layout.series_position[rows]) for m in margins]))
    rng=np.random.default_rng(CONFIG["bootstrap_seed"]); completed=len(boot)
    for repeat in range(CONFIG["bootstrap_replicates"]):
        local=[]
        for series,pair in scorers:
            mult=rng.multinomial(len(series),np.full(len(series),1/len(series)))
            if repeat<completed: continue
            w=np.zeros(len(layout.inner_series_fold)); w[series]=mult; local.append([s.score(w) for s in pair])
        if repeat<completed: continue
        local=np.asarray(local).mean(axis=0); boot.append(float(local[1]-local[0]))
        if (repeat+1)%25==0: commit_checkpoint(directory/"bootstrap.npz",dict(gains=np.asarray(boot)),meta)
    lower=float(np.quantile(boot,(1-CONFIG["confidence"])/2)); confirmation=float(scores[1,confirm].mean())
    passed=bool(confirmation>=CONFIG["target_tsauc"] and gain>=CONFIG["minimum_confirmation_gain"] and wins>=CONFIG["minimum_confirmation_wins"] and lower>0 and min(split)>0)
    results={arm:dict(five_fold=describe(scores[i],scores[0]),pilot_score=float(scores[i,CONFIG["pilot_folds"][0]]),confirmation_scores=scores[i,confirm].tolist(),confirmation_mean=float(scores[i,confirm].mean()),no_break_false_alarms=false_alarms(expit(margins[i].astype(np.float64)),layout,[.5,.8])) for i,arm in enumerate(ARMS)}
    results["direct_meta"].update(confirmation_gain=gain,confirmation_wins=wins,bootstrap_lower_95=lower,split_half_confirmation_gains=split,passes=passed)
    summary=dict(model=31,selected_candidate="direct_meta" if passed else None,target_tsauc=CONFIG["target_tsauc"],target_reached=bool(confirmation>=CONFIG["target_tsauc"]),results=results,interpretation="Method prespecified in prespec.json after Model30 fold0 only; folds1-4 are confirmation.",deployment="No packaging, upload, or cloud run performed.")
    write_json(directory/"summary.json",summary)
    lines=["# Model31: direct pilot-trained meta-stack","","| Arm | Pilot fold0 | Confirmation folds1-4 | Gain | Wins | Lower 95% | Pass |","|---|---:|---:|---:|---:|---:|---|"]
    for arm in ARMS:
        r=results[arm]; lines.append(f"| {arm} | {r['pilot_score']:.6f} | {r['confirmation_mean']:.6f} | {r.get('confirmation_gain',0):+.6f} | {r.get('confirmation_wins',0)}/4 | {r.get('bootstrap_lower_95',0):+.6f} | {r.get('passes','baseline')} |")
    lines += ["",f"Unseen confirmation target: {CONFIG['target_tsauc']:.3f}; reached: {summary['target_reached']}.","Fold0 trains the meta-model and is excluded from confirmation.","",summary["interpretation"],"","These are local held-series OOF scores, not cloud scores.","",summary["deployment"],""]
    text="\n".join(lines); (directory/"RESULTS.md").write_text(text); (HERE/"RESULTS.md").write_text(text); print(text,flush=True); return summary
