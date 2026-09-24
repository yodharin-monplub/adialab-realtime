"""Fit Model62 after adding sampled rows from its former stopping fold."""
from __future__ import annotations
import fcntl, gc, json, resource, sys, time
from pathlib import Path
import lightgbm as lgb
import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from models.model10.causal_transforms import apply_causal_transform
from models.model10.sources import load_development_layout
from models.model11.pipeline import sha256, write_json
from models.model18.run_model18 import commit_checkpoint, digest, read_checkpoint, validate_predictions
from models.model20.features import load_real_pool
from models.model20.report import ensemble_margin
from models.model20.run_model20 import reference_weights, sampled_mask, splice_weights, synthetic_tasks
from models.model21.run_model21 import read_cache
from models.model26 import run_model26 as voter
from models.model38 import run_model38 as smoother
from models.model39 import run_model39 as incumbent
from models.model57 import run_model57 as white_source
from models.model62 import run_model62 as source
from models.model67 import run_model67
from models.model84 import run_model84
from models.report_next_models import WeightedTSAUC
from tsauc import tsauc

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE/'prespec.json').read_text())
P20, P57 = ROOT/CONFIG['model20_run'], ROOT/CONFIG['model57_run']
ARM = 'four_fold_standard'


def parent_state():
    d62, p62, s62 = source.frozen_study(); d84, _, s84 = run_model84.frozen_study()
    if d62 != ROOT/CONFIG['model62_run'] or s62 != CONFIG['model62_protocol_sha256']:
        raise ValueError('Changed Model62 parent')
    if d84 != ROOT/CONFIG['model84_run'] or s84 != CONFIG['model84_protocol_sha256']:
        raise ValueError('Changed Model84 parent')
    parent, _, deps, d26, d39 = source.parent_state()
    if CONFIG['synthetic_mass'] != source.CONFIG['synthetic_mass']:
        raise ValueError('Changed synthetic mass')
    return d62, p62, d84, parent, deps, d26, d39


def initialize(d62, p62, d84, parent, deps, d26, d39):
    if (HERE/'runs/latest.json').exists(): return frozen_study()
    paths = [HERE/'run_model85.py', HERE/'prespec.json', HERE/'readme.txt',
        P20/'protocol.json', P20/'real_features.cache.json', P57/'protocol.json', P57/'real_white.json',
        d62/'protocol.json', d62/'summary.json', d84/'protocol.json', d84/'summary.json',
        d26/'protocol.json', d39/'protocol.json', ROOT/'models/model20/run_model20.py',
        ROOT/'models/model20/features.py', ROOT/'models/model20/report.py',
        ROOT/'models/model57/run_model57.py', ROOT/'models/model62/run_model62.py',
        ROOT/'models/model84/run_model84.py', ROOT/'models/model67/run_model67.py',
        ROOT/'models/model10/causal_transforms.py', ROOT/'models/model10/sources.py',
        ROOT/'models/report_next_models.py', ROOT/'train/X_train.parquet',
        ROOT/'train/y_train_index.parquet', ROOT/'models/model5/artifacts/split_assignments.npz', ROOT/'tsauc.py']
    for f in range(5): paths += [P20/f'aligned_25_fold{f}_features.cache.json', P57/f'aligned_25_fold{f}_white.json']
    protocol = {'contract': {'config': CONFIG, 'dependencies': deps,
        'hashes': {str(p.relative_to(ROOT)): sha256(p) for p in paths},
        'recipes': parent['contract']['recipes'], 'names': p62['contract']['names'],
        'union_names': p62['contract']['union_names']},
        'plans': p62['plans'], 'plans_sha256': p62['plans_sha256']}
    directory = HERE/'runs'/digest(protocol)[:16]; directory.mkdir(parents=True, exist_ok=True)
    path = directory/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol: raise ValueError('Model85 protocol mismatch')
    if not path.exists(): write_json(path, protocol)
    fp = sha256(path); write_json(HERE/'runs/latest.json', {'directory': str(directory.relative_to(ROOT)), 'protocol_sha256': fp})
    return directory, protocol, fp


def frozen_study():
    latest=json.loads((HERE/'runs/latest.json').read_text()); directory=ROOT/latest['directory']
    protocol=json.loads((directory/'protocol.json').read_text()); fp=sha256(directory/'protocol.json')
    if fp != latest['protocol_sha256'] or protocol['contract']['config'] != CONFIG: raise ValueError('Changed Model85 protocol')
    if digest(protocol['plans']) != protocol['plans_sha256']: raise ValueError('Changed donor plan')
    for rel, expected in protocol['contract']['hashes'].items():
        if sha256(ROOT/rel) != expected: raise ValueError(f'Changed Model85 input: {rel}')
    return directory, protocol, fp


def expanded_rows(layout, fold, base):
    stop = (fold+1) % 5
    mask = sampled_mask(layout.series_ids[layout.series_position], layout.time_online)
    extra = np.flatnonzero((layout.row_fold == stop) & mask)
    rows = np.sort(np.r_[base, extra])
    if len(np.unique(rows)) != len(rows) or np.any(layout.row_fold[rows] == fold):
        raise ValueError('Invalid four-fold fitting rows')
    return rows, extra


def run_fold(directory, protocol, fp, layout, pool, real, fold):
    held=np.flatnonzero(layout.row_fold==fold); meta={'protocol_sha256':fp,'fold':fold,'arm':ARM}
    saved=read_checkpoint(directory/f'{ARM}_fold{fold}.npz',meta)
    if saved is not None: validate_predictions(saved,held); print(f'Model85 fold {fold} already complete',flush=True); return
    plan=protocol['plans'][fold]; tasks,sy,st=synthetic_tasks(pool,plan,layout,True)
    p20=json.loads((P20/'protocol.json').read_text())
    base=read_cache(P20,f'aligned_25_fold{fold}_features',{'protocol_sha256':CONFIG['model20_protocol_sha256'],
        'rows':len(sy),'names':p20['contract']['names'][fold],'tasks':len(tasks)})
    white=white_source.cached_white(P57/f'aligned_25_fold{fold}_white.npy',CONFIG['model57_protocol_sha256'],len(sy))
    if white is None: raise ValueError('Missing synthetic white cache')
    sx=np.concatenate((np.asarray(base),np.asarray(white)),axis=1); del tasks,base,white
    fit,extra=expanded_rows(layout,fold,np.asarray(plan['fit_rows']))
    names,union=protocol['contract']['names'][fold],protocol['contract']['union_names']
    cols=np.asarray([union.index(n) for n in names]); y,t=layout.target[fit],layout.time_online[fit]
    fitting=layout.row_fold != fold
    weights=reference_weights(y,t,layout.target[fitting],layout.time_online[fitting])
    rw,sw,accounting=splice_weights(weights,y,t,sy,st,CONFIG['synthetic_mass'])
    x=np.concatenate((real[fit][:,cols],sx)); labels=np.r_[y,sy]; sample_weight=np.r_[rw,sw]
    recipe=dict(source.CONFIG['lightgbm']); seed=protocol['contract']['recipes'][fold]['seed']
    model=lgb.LGBMClassifier(objective='binary',random_state=seed,n_jobs=CONFIG['threads'],verbosity=-1,
        deterministic=True,force_col_wise=True,**recipe)
    started=time.monotonic(); print(f'Fitting Model85 fold {fold}: {len(fit):,} real ({len(extra):,} new) + {len(sy):,} synthetic',flush=True)
    model.fit(x,labels,sample_weight=sample_weight); pred=np.empty(len(held),dtype=np.float32)
    for left in range(0,len(held),50000):
        rows=held[left:left+50000]; pred[left:left+len(rows)]=model.predict_proba(real[rows][:,cols])[:,1]
    model_path=directory/f'{ARM}_fold{fold}.txt'; temp=model_path.with_suffix('.partial.txt'); model.booster_.save_model(temp); temp.replace(model_path)
    details=directory/f'{ARM}_fold{fold}.fit.json'; score=tsauc(layout.target[held],pred,layout.time_online[held])
    write_json(details,{'score':score,'real_rows':len(fit),'added_stopping_rows':len(extra),'synthetic_rows':len(sy),
        'weights':accounting,'elapsed_seconds':time.monotonic()-started,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    commit_checkpoint(directory/f'{ARM}_fold{fold}.npz',{'prediction':pred,'held_rows':held},meta,(model_path,details))
    print(f'Finished Model85 fold {fold}: component={score:.9f}',flush=True); del x,labels,sample_weight,model,sx,sy,st; gc.collect()


def score_folds(layout,v): return np.asarray([tsauc(layout.target[layout.row_fold==f],v[layout.row_fold==f],layout.time_online[layout.row_fold==f]) for f in range(5)])


def model84_prediction(layout):
    d,_,fp=run_model84.frozen_study(); decision=json.loads((d/'pilot_decision.json').read_text())
    original=run_model84.CONFIG['arms']; run_model84.CONFIG['arms']={decision['selected_arm']:original[decision['selected_arm']]}
    try: m67,signals=run_model84.processed_signals(d,fp,layout,range(5))
    finally: run_model84.CONFIG['arms']=original
    w=decision['selected_weight']; return ((1-w)*m67+w*signals[decision['selected_arm']]).astype(np.float32)


def new_signal(directory,fp,layout,folds,baseline):
    m26,m35,_=incumbent.margins(layout); delta=m35.astype(float)-m26.astype(float)
    d26,_,s26=voter.frozen_study(); probability=np.empty(layout.row_count,dtype=np.float32)
    for f in range(5):
        held=np.flatnonzero(layout.row_fold==f); probability[held]=voter.prediction_checkpoint(d26,'depth_mix',f,s26,held)
        if f in folds:
            saved=read_checkpoint(directory/f'{ARM}_fold{f}.npz',{'protocol_sha256':fp,'fold':f,'arm':ARM})
            if saved is None: raise ValueError(f'Missing Model85 fold {f}')
            probability[held]=saved['prediction']
    value=apply_causal_transform(expit(ensemble_margin(probability,layout).astype(float)+delta),
        layout.series_position,smoother.transform_spec(),layout.time_online).astype(np.float32)
    value[~np.isin(layout.row_fold,list(folds))]=baseline[~np.isin(layout.row_fold,list(folds))]
    return value


def decide(directory,fp,layout):
    baseline=model84_prediction(layout); signal=new_signal(directory,fp,layout,CONFIG['pilot_folds'],baseline)
    bs=score_folds(layout,baseline); gains=[]
    for w in CONFIG['blend_weights']:
        gains.append(float(score_folds(layout,((1-w)*baseline+w*signal).astype(np.float32))[CONFIG['pilot_folds']].mean()-bs[CONFIG['pilot_folds']].mean()))
    i=int(np.argmax(gains)); decision={'pilot_gains':gains,'selected_index':i,'selected_weight':CONFIG['blend_weights'][i],
        'continue_training':bool(gains[i]>=CONFIG['pilot_minimum_gain'])}; write_json(directory/'pilot_decision.json',decision)
    print(f'Model85 pilot gains: {dict(zip(CONFIG["blend_weights"],gains))}',flush=True); return decision


def final_report(directory,fp,layout,decision):
    baseline=model84_prediction(layout); signal=new_signal(directory,fp,layout,range(5),baseline); w=decision['selected_weight']
    candidate=((1-w)*baseline+w*signal).astype(np.float32); bs,cs=score_folds(layout,baseline),score_folds(layout,candidate); gains=cs-bs
    confirm=np.asarray(CONFIG['confirmation_folds']); scorers=[]
    for f in confirm:
        rows=layout.row_fold==f; series=np.flatnonzero(layout.inner_series_fold==f)
        scorers.append((series,[WeightedTSAUC(layout.target[rows],v[rows],layout.time_online[rows],layout.series_position[rows]) for v in (baseline,candidate)]))
    rng=np.random.default_rng(CONFIG['bootstrap_seed']); boot=[]
    for repeat in range(CONFIG['bootstrap_replicates']):
        local=[]
        for series,pair in scorers:
            mult=rng.multinomial(len(series),np.full(len(series),1/len(series))); weights=np.zeros(len(layout.inner_series_fold)); weights[series]=mult
            local.append([s.score(weights) for s in pair])
        mean=np.asarray(local).mean(axis=0); boot.append(float(mean[1]-mean[0]))
        if (repeat+1)%25==0: commit_checkpoint(directory/'bootstrap.npz',{'gains':np.asarray(boot)},{'protocol_sha256':fp})
    lower=float(np.quantile(boot,(1-CONFIG['confidence'])/2)); halves=(layout.series_ids*0x9E3779B1)&1; split=[]
    for half in (0,1):
        rows=np.isin(layout.row_fold,confirm)&(halves[layout.series_position]==half)
        split.append(float(tsauc(layout.target[rows],candidate[rows],layout.time_online[rows])-tsauc(layout.target[rows],baseline[rows],layout.time_online[rows])))
    cg=float(gains[confirm].mean()); cscore=float(cs[confirm].mean()); full=float(cs.mean()); wins=int(np.sum(gains[confirm]>0))
    passed=bool(cscore>=CONFIG['target_tsauc'] and full>=CONFIG['target_tsauc'] and cg>=CONFIG['minimum_confirmation_gain'] and wins>=CONFIG['minimum_confirmation_wins'] and lower>0 and min(split)>0)
    summary={'model':85,'pilot':decision,'baseline_fold_scores':bs.tolist(),'candidate_fold_scores':cs.tolist(),'fold_gains':gains.tolist(),
        'confirmation_score':cscore,'full_oof_score':full,'confirmation_gain':cg,'confirmation_wins':wins,'bootstrap_lower_95':lower,'split_half_gains':split,'passes':passed,'deployment':'No cloud activity.'}
    write_json(directory/'summary.json',summary); text=(f'# Model85: four-fold real training\n\n| Arm | Confirmation | Full OOF | Gain | Wins | Lower 95% | Pass |\n|---|---:|---:|---:|---:|---:|---|\n| Model84 | {bs[confirm].mean():.6f} | {bs.mean():.6f} | baseline | baseline | baseline | baseline |\n| Four-fold weight {w:.2f} | {cscore:.6f} | {full:.6f} | {cg:+.6f} | {wins}/3 | {lower:+.6f} | {passed} |\n\nFold gains: {gains.tolist()}. Split halves: {split}.\n')
    (directory/'RESULTS.md').write_text(text); (HERE/'RESULTS.md').write_text(text); print(text,flush=True)


def main():
    (HERE/'runs').mkdir(parents=True,exist_ok=True)
    with (HERE/'runs/active.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); d62,p62,d84,parent,deps,d26,d39=parent_state(); directory,protocol,fp=initialize(d62,p62,d84,parent,deps,d26,d39)
        layout=load_development_layout(ROOT); base=read_cache(P20,'real_features',{'protocol_sha256':CONFIG['model20_protocol_sha256'],'rows':layout.row_count,'names':parent['contract']['union_names'],'tasks':int(np.sum(layout.inner_series_fold>=0))})
        white=white_source.cached_white(P57/'real_white.npy',CONFIG['model57_protocol_sha256'],layout.row_count)
        if white is None: raise ValueError('Missing real white cache')
        real=white_source.CombinedFeatures(base,white); pool=load_real_pool(layout)
        for f in CONFIG['pilot_folds']: run_fold(directory,protocol,fp,layout,pool,real,f)
        decision=decide(directory,fp,layout)
        if not decision['continue_training']:
            text='# Model85: four-fold real training\n\nNo blend met the +0.0001 pilot gate. Target not reached; no cloud activity.\n'; (directory/'RESULTS.md').write_text(text); (HERE/'RESULTS.md').write_text(text); print(text); return
        for f in CONFIG['confirmation_folds']: run_fold(directory,protocol,fp,layout,pool,real,f)
        final_report(directory,fp,layout,decision)

if __name__=='__main__': main()
