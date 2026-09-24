"""Direct meta-stack prespecified after Model30 pilot fold 0."""
from pathlib import Path
import argparse
import fcntl
import json
import sys

import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.model10.sources import load_development_layout
from models.model11.pipeline import sha256, write_json
from models.model18.run_model18 import commit_checkpoint, digest, read_checkpoint, validate_predictions
from models.model20.report import ensemble_margin
from models.model20.run_model20 import reference_weights
from models.model26 import run_model26 as incumbent


HERE = Path(__file__).resolve().parent
MODEL30 = ROOT / "models/model30"
PRESPEC = HERE / "prespec.json"
CONFIG = json.loads(PRESPEC.read_text())
ARMS = ("model26", "direct_meta")


def model30_state():
    latest = json.loads((MODEL30 / "runs/latest.json").read_text())
    directory = ROOT / latest["directory"]
    fingerprint = sha256(directory / "protocol.json")
    if fingerprint != latest["protocol_sha256"] or fingerprint != CONFIG["model30_protocol_sha256"]:
        raise ValueError("Changed Model30 protocol")
    return directory, fingerprint


def initialize():
    if (HERE / "runs/latest.json").exists(): return frozen_study()
    model30_dir, model30_sha = model30_state(); model26_dir, _, model26_sha = incumbent.frozen_study()
    files = [HERE/"run_model31.py",HERE/"report.py",HERE/"readme.txt",PRESPEC,
        ROOT/"tests/test_model31.py",model30_dir/"protocol.json",model26_dir/"protocol.json"]
    protocol = dict(contract=dict(config=CONFIG,
        hashes={str(path.relative_to(ROOT)):sha256(path) for path in files},
        model30_protocol_sha256=model30_sha,model26_protocol_sha256=model26_sha))
    directory=HERE/"runs"/digest(protocol)[:16]; directory.mkdir(parents=True,exist_ok=True)
    path=directory/"protocol.json"
    if not path.exists(): write_json(path,protocol)
    elif json.loads(path.read_text())!=protocol: raise ValueError("Model31 protocol mismatch")
    write_json(HERE/"runs/latest.json",dict(directory=str(directory.relative_to(ROOT)),protocol_sha256=sha256(path)))
    return directory,protocol,sha256(path)


def frozen_study():
    latest=json.loads((HERE/"runs/latest.json").read_text()); directory=ROOT/latest["directory"]
    protocol=json.loads((directory/"protocol.json").read_text()); fingerprint=sha256(directory/"protocol.json")
    if fingerprint!=latest["protocol_sha256"] or protocol["contract"]["config"]!=CONFIG:
        raise ValueError("Changed Model31 protocol")
    for relative,expected in protocol["contract"]["hashes"].items():
        if sha256(ROOT/relative)!=expected: raise ValueError(f"Changed frozen Model31 input: {relative}")
    return directory,protocol,fingerprint


def base_components(layout):
    directory,_,fingerprint=incumbent.frozen_study(); result=np.empty(layout.row_count,dtype=np.float32)
    for fold in range(5):
        held=np.flatnonzero(layout.row_fold==fold)
        result[held]=incumbent.prediction_checkpoint(directory,"depth_mix",fold,fingerprint,held)
    return result


def model30_prediction(fold,held):
    directory,fingerprint=model30_state()
    saved=read_checkpoint(directory/f"matched_15_fold{fold}.npz",
        dict(protocol_sha256=fingerprint,fold=fold,arm="matched_15"))
    if saved is None: raise RuntimeError(f"Model30 fold {fold} is incomplete")
    validate_predictions(saved,held); return saved["prediction"]


def meta_features(base_margin,new_probability,time_online):
    result=np.column_stack((np.asarray(base_margin,dtype=np.float64),
        logit(np.clip(np.asarray(new_probability,dtype=np.float64),1e-5,1-1e-5)),
        np.log1p(np.asarray(time_online,dtype=np.float64))))
    if result.ndim!=2 or result.shape[1]!=3 or not np.isfinite(result).all():
        raise ValueError("Invalid Model31 features")
    return result


def fit_meta(directory,fingerprint,layout,base_margin):
    meta=dict(protocol_sha256=fingerprint,arm="direct_meta",pilot_folds=CONFIG["pilot_folds"])
    saved=read_checkpoint(directory/"meta.npz",meta)
    if saved is not None: return saved
    fold=CONFIG["pilot_folds"][0]; held=np.flatnonzero(layout.row_fold==fold)
    x=meta_features(base_margin[held],model30_prediction(fold,held),layout.time_online[held])
    mean=x.mean(axis=0); scale=x.std(axis=0); scale[scale<1e-8]=1
    y=layout.target[held]; t=layout.time_online[held]; weights=reference_weights(y,t,y,t)
    model=LogisticRegression(C=CONFIG["C"],tol=CONFIG["tolerance"],max_iter=CONFIG["max_iter"],
        solver=CONFIG["solver"],random_state=CONFIG["seed"])
    model.fit((x-mean)/scale,y,sample_weight=weights)
    commit_checkpoint(directory/"meta.npz",dict(mean=mean,scale=scale,coefficient=model.coef_[0],
        intercept=np.asarray(model.intercept_),iterations=np.asarray(model.n_iter_)),meta)
    return read_checkpoint(directory/"meta.npz",meta)


def meta_margin(saved,base_margin,new_probability,time_online):
    x=meta_features(base_margin,new_probability,time_online)
    result=((x-saved["mean"])/saved["scale"])@saved["coefficient"]+saved["intercept"][0]
    if not np.isfinite(result).all(): raise ValueError("Invalid Model31 margin")
    return result.astype(np.float32)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("command",choices=("initialize","run","report"))
    args=parser.parse_args(); (HERE/"runs").mkdir(parents=True,exist_ok=True)
    with (HERE/"runs/active.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); directory,_,fingerprint=initialize()
        if args.command!="initialize":
            layout=load_development_layout(ROOT); base_margin=ensemble_margin(base_components(layout),layout)
            fit_meta(directory,fingerprint,layout,base_margin)
            from models.model31.report import report
            report(layout,base_margin)


if __name__=="__main__": main()
