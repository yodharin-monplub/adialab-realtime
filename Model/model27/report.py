"""Held-series OOF evaluation for the prespecified Model27 candidate."""
import json

import numpy as np
from scipy.special import expit

from models.model10.sources import load_development_layout
from models.model11.pipeline import describe, sha256, write_json
from models.model18.run_model18 import read_checkpoint, commit_checkpoint
from models.model18.report import simultaneous_bounds, false_alarms
from models.model20.report import ensemble_margin
from models.model27.run_model27 import ROOT,HERE,ARMS,CONFIG,frozen_study,prediction_checkpoint
from models.report_next_models import WeightedTSAUC
from tsauc import tsauc


EXPECTED_MODEL26 = np.asarray([
    .6427355043536926,.6085023046201091,.6422531072481089,
    .6218047904181766,.6467391865285458])


def report():
    directory,protocol,fingerprint=frozen_study()
    layout=load_development_layout(ROOT)
    component=[]; margins=[]
    for arm in ARMS:
        p=np.empty(layout.row_count,dtype=np.float32)
        for fold in range(5):
            held=np.flatnonzero(layout.row_fold==fold)
            p[held]=prediction_checkpoint(directory,arm,fold,fingerprint,held)
        component.append(p); margins.append(ensemble_margin(p,layout))
    scores=np.asarray([[tsauc(layout.target[layout.row_fold==fold],m[layout.row_fold==fold],
        layout.time_online[layout.row_fold==fold]) for fold in range(5)] for m in margins])
    np.testing.assert_allclose(scores[0],EXPECTED_MODEL26,rtol=0,atol=1e-12,
        err_msg='Model26 reconstruction changed')
    component_scores=np.asarray([[tsauc(layout.target[layout.row_fold==fold],p[layout.row_fold==fold],
        layout.time_online[layout.row_fold==fold]) for fold in range(5)] for p in component])
    half=(layout.series_ids*0x9E3779B1)&1
    split_gains=[]
    for value in (0,1):
        rows=half[layout.series_position]==value
        split_gains.append(float(tsauc(layout.target[rows],margins[1][rows],layout.time_online[rows])
            -tsauc(layout.target[rows],margins[0][rows],layout.time_online[rows])))
    preliminary=dict(ensemble_scores=dict(zip(ARMS,scores.tolist())),
        component_scores=dict(zip(ARMS,component_scores.tolist())),split_half_pooled_gains=split_gains,
        confidence_checks_complete=False)
    write_json(directory/'preliminary_scores.json',preliminary)
    for arm,fold_scores in zip(ARMS,scores):
        print(f'{arm}: full ensemble={fold_scores.mean():.9f}, folds={fold_scores.tolist()}',flush=True)
    prediction_hashes={p.name:sha256(p) for p in sorted(directory.glob('matched_context_fold*.npz'))}
    meta=dict(protocol_sha256=fingerprint,predictions=prediction_hashes)
    saved=read_checkpoint(directory/'bootstrap.npz',meta)
    boot=[] if saved is None else saved['means'].tolist()
    count=CONFIG['bootstrap_replicates']
    if len(boot)>count or (boot and np.asarray(boot).shape!=(len(boot),2)):
        raise ValueError('Invalid Model27 bootstrap checkpoint')
    if len(boot)<count:
        scorers=[]; positions=[]
        for fold in range(5):
            rows=layout.row_fold==fold
            positions.append(np.flatnonzero(layout.inner_series_fold==fold))
            scorers.append([WeightedTSAUC(layout.target[rows],m[rows],layout.time_online[rows],
                layout.series_position[rows]) for m in margins])
            np.testing.assert_allclose([s.score(np.ones(len(layout.inner_series_fold))) for s in scorers[-1]],
                scores[:,fold],rtol=0,atol=1e-12)
        rng=np.random.default_rng(CONFIG['bootstrap_seed']); completed=len(boot)
        for repeat in range(count):
            local=[]
            for series,fold_scorers in zip(positions,scorers):
                multiplicity=rng.multinomial(len(series),np.full(len(series),1/len(series)))
                if repeat<completed: continue
                weights=np.zeros(len(layout.inner_series_fold)); weights[series]=multiplicity
                local.append([s.score(weights) for s in fold_scorers])
            if repeat<completed: continue
            boot.append(np.mean(local,axis=0).tolist())
            if (repeat+1)%25==0 or repeat+1==count:
                commit_checkpoint(directory/'bootstrap.npz',dict(means=np.asarray(boot)),meta)
                print(f'Model27 whole-series bootstrap: {repeat+1}/{count}',flush=True)
    gain=float(scores[1].mean()-scores[0].mean())
    lower,critical=simultaneous_bounds(np.asarray([gain]),
        (np.asarray(boot)[:,1]-np.asarray(boot)[:,0])[:,None],CONFIG['confidence'])
    wins=int(np.sum(scores[1]>scores[0]))
    passed=bool(gain>=CONFIG['minimum_gain'] and wins>=CONFIG['minimum_fold_wins']
        and lower[0]>0 and min(split_gains)>0)
    results={}
    for index,arm in enumerate(ARMS):
        result=describe(scores[index],scores[0])
        result['component']=describe(component_scores[index],component_scores[0])
        result['no_break_false_alarms']=false_alarms(expit(margins[index].astype(np.float64)),layout,[.5,.8])
        if index:
            result.update(simultaneous_lower_95=float(lower[0]),fold_wins=wins,
                split_half_pooled_gains=split_gains,passes=passed,
                target_0_64_reached=bool(scores[index].mean()>=CONFIG['target_tsauc']))
        results[arm]=result
    summary=dict(model=27,results=results,selected_candidate='matched_context' if passed else None,
        target_tsauc=CONFIG['target_tsauc'],target_reached=bool(scores[1].mean()>=CONFIG['target_tsauc']),
        bootstrap_replicates=len(boot),simultaneous_critical_value=float(critical),
        interpretation=CONFIG['interpretation'],deployment='No packaging, upload, or cloud run performed.')
    write_json(directory/'summary.json',summary)
    lines=['# Model27: family-matched contrastive augmentation','',
        '| Arm | Component CV | Full ensemble CV | Gain | Fold wins | Lower 95% | Split gains | Pass |',
        '|---|---:|---:|---:|---:|---:|---|---|']
    for arm,result in results.items():
        lines.append(f"| {arm} | {result['component']['fold_mean']:.6f} | {result['fold_mean']:.6f} | "
            f"{result['mean_gain']:+.6f} | {result.get('fold_wins',0)}/5 | "
            f"{result.get('simultaneous_lower_95',0):+.6f} | {result.get('split_half_pooled_gains','baseline')} | "
            f"{result.get('passes','baseline')} |")
    lines += ['',f"Target held-series OOF TS-AUC: {CONFIG['target_tsauc']:.3f}; reached: {summary['target_reached']}.",
        'Promotion requires gain >=0.0007, >=4/5 folds, lower bound >0, and positive gains in both prespecified disjoint series halves.',
        '',CONFIG['interpretation'],'','These are local OOF scores, not cloud scores.','',summary['deployment'],'']
    text='\n'.join(lines); (directory/'RESULTS.md').write_text(text); (HERE/'RESULTS.md').write_text(text)
    print(text,flush=True)
    return summary


if __name__=='__main__': report()
