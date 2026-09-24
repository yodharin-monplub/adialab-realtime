"""Validate exact predictions and benchmark Submission28."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse, json, multiprocessing as mp, os, resource, sys, time

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import numpy as np
import pyarrow.dataset as ds

ROOT=Path(__file__).resolve().parents[3]; BUNDLE=ROOT/'submissions/submission28'
OUT=ROOT/'models/model87/deployment_validation'; sys.path.insert(0,str(BUNDLE)); sys.path.insert(1,str(ROOT))

def write(name,value): OUT.mkdir(exist_ok=True); (OUT/name).write_text(json.dumps(value,indent=2)+'\n')

def records(count):
    with np.load(ROOT/'models/model5/artifacts/split_assignments.npz') as p: positions=np.flatnonzero(p['inner_series_fold']>=0)
    with np.load(ROOT/'models/artifacts/core_features.metadata.npz') as m:
        ids=m['series_ids'][np.random.default_rng(16312).choice(positions,count,replace=False)]; total=int(m['online_lengths'].sum())
    frame=ds.dataset(ROOT/'train/X_train.parquet',format='parquet').to_table(filter=ds.field('id').isin(ids.tolist())).to_pandas().sort_index()
    out=[]
    for sid in ids:
        series=frame.loc[sid]; out.append((int(sid),series.loc[series.period==1,'value'].to_numpy(dtype=np.float32),series.loc[series.period==2,'value'].to_numpy(dtype=np.float32)))
    return out,total

def parity():
    from solution73 import Bundle,SeriesPredictor
    from incremental_whitening import StreamingWhitening
    from model73_whitening import whitening_features
    sample,_=records(5); golden_sample,_=records(3)
    bundle=Bundle(BUNDLE/'resources'); maximum=0.; predictions=[]; rows=0
    for _,history,online in sample:
        full=whitening_features(history,online); state=StreamingWhitening(history)
        streaming=np.stack([state.step(v) for v in online]); error=float(np.max(np.abs(full-streaming)))
        maximum=max(maximum,error); np.testing.assert_array_equal(full,streaming)
    for _,history,online in golden_sample:
        predictor=SeriesPredictor(bundle,history); predictions.extend(predictor.step(v) for v in online[:180]); rows+=min(180,len(online))
    golden=np.load(ROOT/'models/model73/deployment_validation/parity_predictions.npz')['prediction']
    np.testing.assert_array_equal(predictions,golden)
    second=[]
    for _,history,online in golden_sample:
        predictor=SeriesPredictor(bundle,history); second.extend(predictor.step(v) for v in online[:180])
    np.testing.assert_array_equal(predictions,second)
    from main import infer,train
    touched=[]
    def stream():
        for i,v in enumerate(golden_sample[0][2][:10]): touched.append(i); yield v
    output=infer([(golden_sample[0][1],stream())],str(BUNDLE/'resources')); assert next(output) is None and not touched
    for i in range(10): assert np.isfinite(next(output)) and len(touched)==i+1
    class NoData:
        def __iter__(self): raise AssertionError('Training data consumed')
    train(NoData(),str(BUNDLE/'resources'))
    result={'whitening_rows':int(sum(len(x[2]) for x in sample)),'golden_rows':rows,'whitening_max_error':maximum,
        'final_max_error':float(np.max(np.abs(np.asarray(predictions)-golden))),'bitwise_identical':True,
        'incremental_consumption':True,'train_does_not_consume_data':True,'python':sys.version}
    write('parity.json',result); print(json.dumps(result,indent=2))

_bundle=None
def init_worker():
    global _bundle
    from solution73 import Bundle
    _bundle=Bundle(BUNDLE/'resources')
def timed(record):
    from solution73 import SeriesPredictor
    sid,h,o=record; started=time.perf_counter(); p=SeriesPredictor(_bundle,h); values=np.asarray([p.step(v) for v in o],dtype=np.float32)
    return {'id':sid,'rows':len(o),'seconds':time.perf_counter()-started,'rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,'checksum':float(values.astype(float).sum())}
def benchmark(count,workers):
    sample,total=records(count); started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=init_worker) as pool: results=list(pool.map(timed,sample))
    report={'workers':workers,'series':count,'rows':sum(x['rows'] for x in results),'wall_seconds':time.perf_counter()-started,
        'worker_seconds':sum(x['seconds'] for x in results),'peak_worker_rss_mib':max(x['rss_mib'] for x in results),
        'training_online_rows_reference':total,'series_results':results,'python':sys.version}
    write(f'benchmark_w{workers}_n{count}.json',report); print(json.dumps({k:v for k,v in report.items() if k!='series_results'},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=('parity','benchmark')); parser.add_argument('--series',type=int,default=24); parser.add_argument('--workers',type=int,default=2); args=parser.parse_args()
    parity() if args.command=='parity' else benchmark(args.series,args.workers)
