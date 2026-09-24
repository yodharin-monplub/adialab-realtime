"""Parity, determinism, memory and full-path timing before cloud submission."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import importlib
import json
import multiprocessing as mp
import os
import resource
import sys
import time
import types

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT/"submissions/submission16"
UPSTREAM = ROOT/"models/model16/upstream_cloud_submission9"
OUTPUT = ROOT/"models/model16/submission_validation"
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(BUNDLE))


def records(count):
    with np.load(ROOT/"models/model5/artifacts/split_assignments.npz") as p:
        positions = np.flatnonzero(p["inner_series_fold"] >= 0)
    with np.load(ROOT/"models/artifacts/core_features.metadata.npz") as p:
        ids = p["series_ids"][np.random.default_rng(16312).choice(positions,count,replace=False)]
        total = int(p["online_lengths"].sum())
    frame = ds.dataset(ROOT/"train/X_train.parquet",format="parquet").to_table(filter=ds.field("id").isin(ids.tolist())).to_pandas().sort_index()
    output=[]
    for identifier in ids:
        s=frame.loc[identifier]
        output.append((int(identifier), s.loc[s.period==1,"value"].to_numpy(dtype=float),
                       s.loc[s.period==2,"value"].to_numpy(dtype=np.float32)))
    return output,total


def unwrap_models():
    shim=types.ModuleType("models.model6.model6_pipeline")
    shim.FittedModel=type("FittedModel",(),{})
    sys.modules[shim.__name__]=shim


def parity():
    import joblib
    import lightgbm as lgb
    from solution16 import Bundle,SeriesPredictor,logistic,logit
    from primitive_runtime import _bag_probability_rows
    from feature_pipeline import compute_features_offline
    from compact_features import compute_features
    unwrap_models()
    selected,_=records(3)
    bundle=Bundle(BUNDLE/"resources")
    maximum_detector=maximum_feature=maximum_output=0.
    tested=0
    for _,history,online in selected:
        online=online[:180]
        x=compute_features_offline(history,online,bundle.banks).astype(np.float32)
        np.nan_to_num(x,copy=False,nan=0.,posinf=1e6,neginf=-1e6)
        np.clip(x,-1e6,1e6,out=x)
        mapping={n:i for i,n in enumerate(bundle.manifest["feature_names"])}
        all_probability=[]
        for family,bank in bundle.manifest["detector_ensemble"]["bank_keys"]:
            source=ROOT/f"models/model5/artifacts/screen/{family}/{bank}"
            names=json.loads((source/"manifest.json").read_text())["feature_names"]
            columns=[mapping[n] for n in names]
            voters=np.column_stack([lgb.Booster(model_file=str(source/f"fold_{f}.txt")).predict(x[:,columns],num_threads=1) for f in range(5)])
            all_probability.append(_bag_probability_rows(voters))
        for spec in bundle.manifest["stack_feature_components"]:
            family,component=spec.split("__feature__")
            source=ROOT/f"models/model6/runs/tsroc0p64_909b8f5c6c/final_cv/{family}/{component}"
            probabilities=[]
            for fold in range(5):
                with np.load(source/f"fold_{fold}.columns.npz") as p:
                    columns=[mapping[str(n)] for n in p["names"]]
                fitted=joblib.load(source/f"fold_{fold}.joblib")
                fitted.model.set_params(n_jobs=1)
                probabilities.append(fitted.model.predict_proba(x[:,columns])[:,1])
            all_probability.append(_bag_probability_rows(np.column_stack(probabilities)))
        reference=np.column_stack(all_probability).astype(np.float32)
        exported=np.stack([bundle.primitive_probabilities(row) for row in x])
        maximum_detector=max(maximum_detector,float(np.max(np.abs(reference[:,:31]-exported[:,:31]))))
        maximum_feature=max(maximum_feature,float(np.max(np.abs(reference[:,31:]-exported[:,31:]))))
        np.testing.assert_allclose(exported,reference,atol=1e-7,rtol=0)
        margin9=(logit(reference).astype(np.float32).astype(np.float64) @ bundle.coefficient).astype(np.float32)
        p9=logistic(margin9.astype(float)).astype(np.float32)
        m9=logit(p9.astype(float),epsilon=1e-7)
        compact_x=compute_features(history,online)
        compact_p=_bag_probability_rows(np.column_stack([m.predict(compact_x,num_threads=1) for m in bundle.compact_models]))
        m11=logit(compact_p).astype(np.float32)
        expected=[]
        for i in range(len(online)):
            corrected=np.float32(m9[i]+.1*(np.max(m9[max(0,i-127):i+1])-m9[i]))
            expected.append(float(logistic(float(np.float32(.9*corrected+.1*m11[i])))))
        stream=SeriesPredictor(bundle,history)
        actual=np.array([stream.step(v) for v in online])
        maximum_output=max(maximum_output,float(np.max(np.abs(expected-actual))))
        np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=0)
        repeated=SeriesPredictor(bundle,history)
        np.testing.assert_array_equal(actual,[repeated.step(v) for v in online])
        tested+=len(online)
        print(f"Parity: {tested} online observations",flush=True)
    from main import infer,train
    consumed=[]
    def points():
        for i,v in enumerate(selected[0][2][:10]):
            consumed.append(i)
            yield v
    generator=infer([(selected[0][1],points())],str(BUNDLE/"resources"))
    assert next(generator) is None and not consumed
    for i in range(10):
        assert np.isfinite(next(generator)) and len(consumed)==i+1
    try:next(generator)
    except StopIteration:pass
    else:raise AssertionError("Extra inference output")
    class NoTrainingData:
        def __iter__(self):raise AssertionError("Training data were consumed")
    train(NoTrainingData(),str(BUNDLE/"resources"))
    result=dict(observations=tested,detector_probability_max_error=maximum_detector,
                feature_probability_max_error=maximum_feature,final_probability_max_error=maximum_output,
                repeated_inference_bitwise_identical=True,incremental_consumption_passed=True,
                train_does_not_consume_data=True,python=sys.version)
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT/"parity.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)


_worker_bundle=None
_worker_baseline=False


def initialize_worker(baseline=False):
    global _worker_bundle,_worker_baseline
    _worker_baseline=baseline
    if baseline:
        from primitive_runtime import _load_bundle
        _worker_bundle=_load_bundle(str(OUTPUT/"baseline_resources"))
    else:
        from solution16 import Bundle
        _worker_bundle=Bundle(BUNDLE/"resources")


def timed_series(record):
    if _worker_baseline:
        from primitive_runtime import SeriesPredictor
    else:
        from solution16 import SeriesPredictor
    identifier,history,online=record
    start=time.perf_counter()
    predictor=SeriesPredictor(_worker_bundle,history)
    values=np.array([predictor.step(v) for v in online],dtype=np.float32)
    return dict(id=identifier,rows=len(online),seconds=time.perf_counter()-start,
                max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                checksum=float(values.astype(float).sum()))


def benchmark(count,workers,baseline=False):
    sample,total=records(count)
    start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context("spawn"),initializer=initialize_worker,initargs=(baseline,)) as pool:
        results=list(pool.map(timed_series,sample))
    elapsed=time.perf_counter()-start
    result=dict(workers=workers,series=count,rows=sum(r["rows"] for r in results),wall_seconds=elapsed,
                summed_worker_seconds=sum(r["seconds"] for r in results),peak_worker_rss_mib=max(r["max_rss_mib"] for r in results),
                training_online_rows_reference=total,series_results=results,python=sys.version,
                note="Direct streaming benchmark, no socket/scoring overhead. Startup included in wall time; excludes parent training data.")
    OUTPUT.mkdir(exist_ok=True)
    prefix="baseline_" if baseline else ""
    (OUTPUT/f"{prefix}benchmark_w{workers}_n{count}.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!='series_results'},indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("command",choices=("parity","benchmark"))
    parser.add_argument("--series",type=int,default=24)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--baseline",action="store_true")
    args=parser.parse_args()
    if args.command=="parity":parity()
    else:benchmark(args.series,args.workers,args.baseline)
