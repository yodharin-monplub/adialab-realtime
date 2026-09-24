"""Bag Model84's expressive voter with an explicitly independent seed."""
from __future__ import annotations
import fcntl, json, sys
from pathlib import Path
import numpy as np
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from models.model10.causal_transforms import apply_causal_transform
from models.model10.sources import load_development_layout
from models.model11.pipeline import sha256, write_json
from models.model18.run_model18 import commit_checkpoint, digest, read_checkpoint, validate_predictions
from models.model20.features import load_real_pool
from models.model20.report import ensemble_margin
from models.model21.run_model21 import read_cache
from models.model26 import run_model26 as voter
from models.model38 import run_model38 as smoother
from models.model39 import run_model39 as incumbent
from models.model57 import run_model57 as white_source
from models.model67 import run_model67
from models.model84 import run_model84
from models.report_next_models import WeightedTSAUC
from tsauc import tsauc

HERE=Path(__file__).resolve().parent
CONFIG=json.loads((HERE/'prespec.json').read_text())
ARM='replica'


def state():
    d84,p84,s84=run_model84.frozen_study()
    if d84 != ROOT/CONFIG['model84_run'] or s84 != CONFIG['model84_protocol_sha256']:
        raise ValueError('Changed Model84 parent')
    p62,parent,deps,d26,d39,d62,d67=run_model84.parent_state()
    decision=json.loads((d84/'pilot_decision.json').read_text())
    if decision['selected_arm']!='expressive' or decision['selected_weight']!=0.5:
        raise ValueError('Changed Model84 decision')
    recipe=dict(run_model84.CONFIG['arms']['expressive']); recipe.update(CONFIG['replica_seeds'])
    return d84,p84,p62,parent,deps,d26,d39,d62,d67,recipe


def initialize(d84,p84,p62,parent,deps,d26,d39,d62,d67,recipe):
    if (HERE/'runs/latest.json').exists(): return frozen_study()
    paths=[HERE/'run_model86.py',HERE/'prespec.json',HERE/'readme.txt',d84/'protocol.json',d84/'summary.json',
        d62/'protocol.json',d67/'protocol.json',d26/'protocol.json',d39/'protocol.json',
        ROOT/'models/model84/run_model84.py',ROOT/'models/model67/run_model67.py',
        ROOT/'models/model20/run_model20.py',ROOT/'models/model20/features.py',ROOT/'models/model20/report.py',
        ROOT/'models/model57/run_model57.py',ROOT/'models/model10/causal_transforms.py',ROOT/'models/model10/sources.py',
        ROOT/'models/report_next_models.py',ROOT/'train/X_train.parquet',ROOT/'train/y_train_index.parquet',
        ROOT/'models/model5/artifacts/split_assignments.npz',ROOT/'tsauc.py',
        run_model84.PARENT20/'protocol.json',run_model84.PARENT20/'real_features.cache.json',
        run_model84.PARENT57/'protocol.json',run_model84.PARENT57/'real_white.json']
    for f in range(5):
        paths += [run_model84.PARENT20/f'aligned_25_fold{f}_features.cache.json',
            run_model84.PARENT57/f'aligned_25_fold{f}_white.json',d84/f'expressive_fold{f}.json']
    protocol={'contract':{'config':CONFIG,'replica_recipe':recipe,'dependencies':deps,
        'hashes':{str(p.relative_to(ROOT)):sha256(p) for p in paths},
        'recipes':parent['contract']['recipes'],'names':p62['contract']['names'],
        'union_names':p62['contract']['union_names']},'plans':p62['plans'],'plans_sha256':p62['plans_sha256']}
    directory=HERE/'runs'/digest(protocol)[:16]; directory.mkdir(parents=True,exist_ok=True); path=directory/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol: raise ValueError('Model86 protocol mismatch')
    if not path.exists(): write_json(path,protocol)
    fp=sha256(path); write_json(HERE/'runs/latest.json',{'directory':str(directory.relative_to(ROOT)),'protocol_sha256':fp})
    return directory,protocol,fp


def frozen_study():
    latest=json.loads((HERE/'runs/latest.json').read_text()); directory=ROOT/latest['directory']
    protocol=json.loads((directory/'protocol.json').read_text()); fp=sha256(directory/'protocol.json')
    if fp!=latest['protocol_sha256'] or protocol['contract']['config']!=CONFIG: raise ValueError('Changed Model86 protocol')
    if digest(protocol['plans'])!=protocol['plans_sha256']: raise ValueError('Changed donor plan')
    for rel,expected in protocol['contract']['hashes'].items():
        if sha256(ROOT/rel)!=expected: raise ValueError(f'Changed Model86 input: {rel}')
    return directory,protocol,fp


def fit(directory,protocol,fp,layout,pool,real,fold,recipe):
    original=run_model84.CONFIG['arms']
    try:
        run_model84.CONFIG['arms']={ARM:recipe}
        run_model84.run_fold(directory,protocol,fp,layout,pool,real,fold,[ARM])
    finally: run_model84.CONFIG['arms']=original


def scores(layout,v):
    return np.asarray([tsauc(layout.target[layout.row_fold==f],v[layout.row_fold==f],layout.time_online[layout.row_fold==f]) for f in range(5)])


def base_and_bag(directory,fp,layout,folds):
    d84,_,f84=run_model84.frozen_study(); parents=run_model67.parent_state(); _,m67,_=run_model67.predictions(layout,parents)
    m26,m35,_=incumbent.margins(layout); delta=m35.astype(float)-m26.astype(float)
    d26,_,s26=voter.frozen_study(); old=np.empty(layout.row_count,dtype=np.float32); replica=np.empty_like(old)
    for f in range(5):
        held=np.flatnonzero(layout.row_fold==f)
        p=voter.prediction_checkpoint(d26,'depth_mix',f,s26,held); old[held]=p; replica[held]=p
        saved=read_checkpoint(d84/f'expressive_fold{f}.npz',{'protocol_sha256':f84,'fold':f,'arm':'expressive'})
        if saved is None: raise ValueError(f'Missing Model84 expressive fold {f}')
        validate_predictions(saved,held); old[held]=saved['prediction']; replica[held]=saved['prediction']
        if f in folds:
            new=read_checkpoint(directory/f'{ARM}_fold{f}.npz',{'protocol_sha256':fp,'fold':f,'arm':ARM})
            if new is None: raise ValueError(f'Missing Model86 fold {f}')
            validate_predictions(new,held); replica[held]=new['prediction']
    def process(probability):
        adjusted=expit(ensemble_margin(probability,layout).astype(float)+delta)
        return apply_causal_transform(adjusted,layout.series_position,smoother.transform_spec(),layout.time_online).astype(np.float32)
    old_signal=process(old); bag_signal=process(((old+replica)*np.float32(.5)).astype(np.float32))
    baseline=(.5*np.asarray(m67,dtype=np.float32)+.5*old_signal).astype(np.float32)
    inactive=~np.isin(layout.row_fold,list(folds)); bag_signal[inactive]=old_signal[inactive]
    return np.asarray(m67,dtype=np.float32),baseline,bag_signal


def decision(directory,fp,layout):
    m67,baseline,bag=base_and_bag(directory,fp,layout,CONFIG['pilot_folds']); bs=scores(layout,baseline); gains=[]
    for w in CONFIG['signal_weights']:
        cand=((1-w)*m67+w*bag).astype(np.float32); gains.append(float(scores(layout,cand)[CONFIG['pilot_folds']].mean()-bs[CONFIG['pilot_folds']].mean()))
    i=int(np.argmax(gains)); value={'pilot_gains':gains,'selected_index':i,'selected_weight':CONFIG['signal_weights'][i],
        'continue_training':bool(gains[i]>=CONFIG['pilot_minimum_gain'])}; write_json(directory/'pilot_decision.json',value)
    print(f'Model86 pilot gains: {dict(zip(CONFIG["signal_weights"],gains))}',flush=True); return value


def report(directory,fp,layout,choice):
    m67,baseline,bag=base_and_bag(directory,fp,layout,range(5)); w=choice['selected_weight']; candidate=((1-w)*m67+w*bag).astype(np.float32)
    bs,cs=scores(layout,baseline),scores(layout,candidate); gains=cs-bs; confirm=np.asarray(CONFIG['confirmation_folds']); scorers=[]
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
    summary={'model':86,'pilot':choice,'baseline_fold_scores':bs.tolist(),'candidate_fold_scores':cs.tolist(),'fold_gains':gains.tolist(),
        'confirmation_score':cscore,'full_oof_score':full,'confirmation_gain':cg,'confirmation_wins':wins,'bootstrap_lower_95':lower,'split_half_gains':split,'passes':passed,'deployment':'No cloud activity.'}
    write_json(directory/'summary.json',summary); text=(f'# Model86: expressive seed bag\n\n| Arm | Confirmation | Full OOF | Gain | Wins | Lower 95% | Pass |\n|---|---:|---:|---:|---:|---:|---|\n| Model84 | {bs[confirm].mean():.6f} | {bs.mean():.6f} | baseline | baseline | baseline | baseline |\n| Seed bag weight {w:.3f} | {cscore:.6f} | {full:.6f} | {cg:+.6f} | {wins}/3 | {lower:+.6f} | {passed} |\n\nFold gains: {gains.tolist()}. Split halves: {split}.\n')
    (directory/'RESULTS.md').write_text(text); (HERE/'RESULTS.md').write_text(text); print(text,flush=True)


def main():
    (HERE/'runs').mkdir(parents=True,exist_ok=True)
    with (HERE/'runs/active.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); values=state(); directory,protocol,fp=initialize(*values[:-1],values[-1]); recipe=values[-1]
        layout=load_development_layout(ROOT); parent=values[3]
        base=read_cache(run_model84.PARENT20,'real_features',{'protocol_sha256':run_model84.CONFIG['model20_protocol_sha256'],'rows':layout.row_count,'names':parent['contract']['union_names'],'tasks':int(np.sum(layout.inner_series_fold>=0))})
        white=white_source.cached_white(run_model84.PARENT57/'real_white.npy',run_model84.CONFIG['model57_protocol_sha256'],layout.row_count)
        if white is None: raise ValueError('Missing real white cache')
        real=white_source.CombinedFeatures(base,white); pool=load_real_pool(layout)
        for f in CONFIG['pilot_folds']: fit(directory,protocol,fp,layout,pool,real,f,recipe)
        choice=decision(directory,fp,layout)
        if not choice['continue_training']:
            text='# Model86: expressive seed bag\n\nNo choice met the +0.0001 pilot gate. Target not reached; no cloud activity.\n'; (directory/'RESULTS.md').write_text(text); (HERE/'RESULTS.md').write_text(text); print(text); return
        for f in CONFIG['confirmation_folds']: fit(directory,protocol,fp,layout,pool,real,f,recipe)
        report(directory,fp,layout,choice)

if __name__=='__main__': main()
