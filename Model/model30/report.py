"""Held-series OOF report for Model30."""
import json
import numpy as np
from scipy.special import expit
from models.model10.sources import load_development_layout
from models.model11.pipeline import describe, sha256, write_json
from models.model18.report import false_alarms
from models.model18.run_model18 import commit_checkpoint, read_checkpoint
from models.model20.report import ensemble_margin
from models.model30.run_model30 import ROOT,HERE,ARMS,CONFIG,frozen_study,prediction_checkpoint
from models.report_next_models import WeightedTSAUC
from tsauc import tsauc

EXPECTED_MODEL26=np.asarray([.6427355043536926,.6085023046201091,.6422531072481089,.6218047904181766,.6467391865285458])

def report():
    directory,_,fingerprint=frozen_study(); layout=load_development_layout(ROOT)
    component=[]; margins=[]
    for arm in ARMS:
        p=np.empty(layout.row_count,dtype=np.float32)
        for fold in range(5):
            held=np.flatnonzero(layout.row_fold==fold)
            p[held]=prediction_checkpoint(directory,arm,fold,fingerprint,held)
        component.append(p); margins.append(ensemble_margin(p,layout))
    scores=np.asarray([[tsauc(layout.target[layout.row_fold==f],m[layout.row_fold==f],layout.time_online[layout.row_fold==f]) for f in range(5)] for m in margins])
    np.testing.assert_allclose(scores[0],EXPECTED_MODEL26,rtol=0,atol=1e-12)
    component_scores=np.asarray([[tsauc(layout.target[layout.row_fold==f],p[layout.row_fold==f],layout.time_online[layout.row_fold==f]) for f in range(5)] for p in component])
    gain=float(scores[1].mean()-scores[0].mean()); wins=int(np.sum(scores[1]>scores[0]))
    half=(layout.series_ids*0x9E3779B1)&1; split=[]
    for value in (0,1):
        rows=half[layout.series_position]==value
        split.append(float(tsauc(layout.target[rows],margins[1][rows],layout.time_online[rows])-tsauc(layout.target[rows],margins[0][rows],layout.time_online[rows])))
    hashes={p.name:sha256(p) for p in sorted(directory.glob("matched_15_fold*.npz"))}
    meta=dict(protocol_sha256=fingerprint,predictions=hashes); saved=read_checkpoint(directory/"bootstrap.npz",meta)
    boot=[] if saved is None else saved["gains"].tolist(); scorers=[]
    for fold in range(5):
        rows=layout.row_fold==fold; series=np.flatnonzero(layout.inner_series_fold==fold)
        scorers.append((series,[WeightedTSAUC(layout.target[rows],m[rows],layout.time_online[rows],layout.series_position[rows]) for m in margins]))
    rng=np.random.default_rng(CONFIG["bootstrap_seed"]); completed=len(boot)
    for repeat in range(CONFIG["bootstrap_replicates"]):
        local=[]
        for series,pair in scorers:
            mult=rng.multinomial(len(series),np.full(len(series),1/len(series)))
            if repeat<completed: continue
            w=np.zeros(len(layout.inner_series_fold)); w[series]=mult
            local.append([s.score(w) for s in pair])
        if repeat<completed: continue
        local=np.asarray(local).mean(axis=0); boot.append(float(local[1]-local[0]))
        if (repeat+1)%25==0: commit_checkpoint(directory/"bootstrap.npz",dict(gains=np.asarray(boot)),meta)
    lower=float(np.quantile(boot,(1-CONFIG["confidence"])/2))
    passed=bool(scores[1].mean()>=CONFIG["target_tsauc"] and gain>=CONFIG["minimum_gain"] and wins>=CONFIG["minimum_fold_wins"] and lower>0 and min(split)>0)
    results={}
    for i,arm in enumerate(ARMS):
        results[arm]=describe(scores[i],scores[0]); results[arm]["component"]=describe(component_scores[i],component_scores[0]); results[arm]["no_break_false_alarms"]=false_alarms(expit(margins[i].astype(np.float64)),layout,[.5,.8])
    results["matched15_50"].update(fold_wins=wins,bootstrap_lower_95=lower,split_half_gains=split,passes=passed)
    summary=dict(model=30,selected_candidate="matched15_50" if passed else None,target_tsauc=CONFIG["target_tsauc"],target_reached=bool(scores[1].mean()>=CONFIG["target_tsauc"]),results=results,interpretation=CONFIG["interpretation"],deployment="No packaging, upload, or cloud run performed.")
    write_json(directory/"summary.json",summary)
    lines=["# Model30: lower-shift matched augmentation","","| Arm | Component OOF | Full OOF | Gain | Wins | Lower 95% | Pass |","|---|---:|---:|---:|---:|---:|---|"]
    for arm in ARMS:
        r=results[arm]; lines.append(f"| {arm} | {r['component']['fold_mean']:.6f} | {r['fold_mean']:.6f} | {r['mean_gain']:+.6f} | {r.get('fold_wins',0)}/5 | {r.get('bootstrap_lower_95',0):+.6f} | {r.get('passes','baseline')} |")
    lines += ["",f"Target held-series OOF TSAUC: {CONFIG['target_tsauc']:.3f}; reached: {summary['target_reached']}.","",CONFIG["interpretation"],"","These are local OOF scores, not cloud scores.","",summary["deployment"],""]
    text="\n".join(lines); (directory/"RESULTS.md").write_text(text); (HERE/"RESULTS.md").write_text(text); print(text,flush=True); return summary
