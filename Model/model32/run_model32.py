"""Five-fold cross-fitted diversity stack."""
from pathlib import Path
import argparse
import fcntl
import json
import sys

import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from models.model10.sources import load_development_layout
from models.model11.pipeline import sha256,write_json
from models.model18.run_model18 import commit_checkpoint,digest,read_checkpoint,validate_predictions
from models.model20.report import ensemble_margin
from models.model20.run_model20 import reference_weights
from models.model26 import run_model26 as incumbent

HERE=Path(__file__).resolve().parent; PRESPEC=HERE/"prespec.json"; CONFIG=json.loads(PRESPEC.read_text())
MODEL27=ROOT/"models/model27"; MODEL30=ROOT/"models/model30"; ARMS=("model26","crossfit_stack")

def _state(model_dir,expected):
    latest=json.loads((model_dir/"runs/latest.json").read_text()); directory=ROOT/latest["directory"]
    fingerprint=sha256(directory/"protocol.json")
    if fingerprint!=latest["protocol_sha256"] or fingerprint!=expected: raise ValueError(f"Changed {model_dir.name} protocol")
    return directory,fingerprint

def initialize():
    if (HERE/"runs/latest.json").exists(): return frozen_study()
    d27,s27=_state(MODEL27,CONFIG["model27_protocol_sha256"]); d30,s30=_state(MODEL30,CONFIG["model30_protocol_sha256"])
    d26,_,s26=incumbent.frozen_study(); files=[HERE/"run_model32.py",HERE/"report.py",HERE/"readme.txt",PRESPEC,
        ROOT/"tests/test_model32.py",d27/"protocol.json",d30/"protocol.json",d26/"protocol.json"]
    protocol=dict(contract=dict(config=CONFIG,hashes={str(p.relative_to(ROOT)):sha256(p) for p in files},
        model27_protocol_sha256=s27,model30_protocol_sha256=s30,model26_protocol_sha256=s26))
    directory=HERE/"runs"/digest(protocol)[:16]; directory.mkdir(parents=True,exist_ok=True); path=directory/"protocol.json"
    if not path.exists(): write_json(path,protocol)
    elif json.loads(path.read_text())!=protocol: raise ValueError("Model32 protocol mismatch")
    write_json(HERE/"runs/latest.json",dict(directory=str(directory.relative_to(ROOT)),protocol_sha256=sha256(path)))
    return directory,protocol,sha256(path)

def frozen_study():
    latest=json.loads((HERE/"runs/latest.json").read_text()); directory=ROOT/latest["directory"]
    protocol=json.loads((directory/"protocol.json").read_text()); fingerprint=sha256(directory/"protocol.json")
    if fingerprint!=latest["protocol_sha256"] or protocol["contract"]["config"]!=CONFIG: raise ValueError("Changed Model32 protocol")
    for relative,expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT/relative)!=expected: raise ValueError(f"Changed frozen Model32 input: {relative}")
    return directory,protocol,fingerprint

def _checkpoint(model_dir,protocol_sha,name,arm,fold,held):
    saved=read_checkpoint(model_dir/f"{name}_fold{fold}.npz",dict(protocol_sha256=protocol_sha,fold=fold,arm=arm))
    if saved is None: raise RuntimeError(f"Missing {arm} fold {fold}")
    validate_predictions(saved,held); return saved["prediction"]

def level_one(layout):
    d26,_,s26=incumbent.frozen_study(); d27,s27=_state(MODEL27,CONFIG["model27_protocol_sha256"]); d30,s30=_state(MODEL30,CONFIG["model30_protocol_sha256"])
    base=np.empty(layout.row_count,dtype=np.float32); p27=np.empty_like(base); p30=np.empty_like(base)
    for fold in CONFIG["folds"]:
        held=np.flatnonzero(layout.row_fold==fold)
        base[held]=incumbent.prediction_checkpoint(d26,"depth_mix",fold,s26,held)
        p27[held]=_checkpoint(d27,s27,"matched_context","matched_context",fold,held)
        p30[held]=_checkpoint(d30,s30,"matched_15","matched_15",fold,held)
    return ensemble_margin(base,layout),p27,p30

def meta_features(base_margin,p27,p30,time_online):
    result=np.column_stack((np.asarray(base_margin,dtype=np.float64),
        logit(np.clip(np.asarray(p27,dtype=np.float64),1e-5,1-1e-5)),
        logit(np.clip(np.asarray(p30,dtype=np.float64),1e-5,1-1e-5)),
        np.log1p(np.asarray(time_online,dtype=np.float64))))
    if result.ndim!=2 or result.shape[1]!=4 or not np.isfinite(result).all(): raise ValueError("Invalid Model32 features")
    return result

def fit_models(directory,fingerprint,layout,features):
    models=[]
    for fold in CONFIG["folds"]:
        meta=dict(protocol_sha256=fingerprint,arm="crossfit_stack",fold=fold)
        saved=read_checkpoint(directory/f"meta_fold{fold}.npz",meta)
        if saved is not None: models.append(saved); continue
        train=np.isin(layout.row_fold,[f for f in CONFIG["folds"] if f!=fold]); x=features[train]
        mean=x.mean(axis=0); scale=x.std(axis=0); scale[scale<1e-8]=1
        y=layout.target[train]; t=layout.time_online[train]; weights=reference_weights(y,t,y,t)
        model=LogisticRegression(C=CONFIG["C"],tol=CONFIG["tolerance"],max_iter=CONFIG["max_iter"],
            solver=CONFIG["solver"],random_state=CONFIG["seed_base"]+fold)
        model.fit((x-mean)/scale,y,sample_weight=weights)
        commit_checkpoint(directory/f"meta_fold{fold}.npz",dict(mean=mean,scale=scale,coefficient=model.coef_[0],
            intercept=np.asarray(model.intercept_),iterations=np.asarray(model.n_iter_)),meta)
        models.append(read_checkpoint(directory/f"meta_fold{fold}.npz",meta))
    return models

def crossfit_margin(models,layout,features):
    result=np.empty(layout.row_count,dtype=np.float32)
    for fold,saved in zip(CONFIG["folds"],models):
        held=layout.row_fold==fold; x=features[held]
        result[held]=(((x-saved["mean"])/saved["scale"])@saved["coefficient"]+saved["intercept"][0]).astype(np.float32)
    if not np.isfinite(result[layout.row_fold>=0]).all(): raise ValueError("Invalid Model32 predictions")
    return result

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("command",choices=("initialize","run","report")); args=parser.parse_args()
    (HERE/"runs").mkdir(parents=True,exist_ok=True)
    with (HERE/"runs/active.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); directory,_,fingerprint=initialize()
        if args.command!="initialize":
            layout=load_development_layout(ROOT); base,p27,p30=level_one(layout); features=meta_features(base,p27,p30,layout.time_online)
            models=fit_models(directory,fingerprint,layout,features)
            from models.model32.report import report
            report(layout,base,features,models)

if __name__=="__main__": main()
