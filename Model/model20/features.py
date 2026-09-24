"""Selected deployed detector features, with resumable prefix-checked caches."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT/'submissions/submission16'


def detector_module():
    if str(BUNDLE) not in sys.path:
        sys.path.insert(0, str(BUNDLE))
    import feature_pipeline
    if Path(feature_pipeline.__file__).resolve() != BUNDLE/'feature_pipeline.py':
        raise ValueError('Unexpected detector implementation imported')
    return feature_pipeline


def selected_features(history, online, names):
    fp = detector_module()
    banks = sorted({n.split('__')[0] for n in names}, key=lambda n: int(n[1:]))
    available = fp.feature_names(banks)
    columns = [available.index(n) for n in names]
    return fp.compute_features_offline(history, online, banks)[:, columns]


def load_real_pool(layout):
    positions = np.flatnonzero(layout.inner_series_fold >= 0)
    ids = layout.series_ids[positions]
    frame = pd.read_parquet(ROOT/'train/X_train.parquet', columns=['value', 'period'],
                           filters=[('id', 'in', ids.tolist())])
    if not frame.index.is_monotonic_increasing or frame.index.names != ['id', 'time']:
        raise ValueError('Training values must be ordered by (id,time)')
    lookup = dict(zip(ids.tolist(), positions.tolist()))
    pool = {}
    for sid, group in frame.groupby(level='id', sort=False):
        values, periods = group.value.to_numpy(dtype=np.float64), group.period.to_numpy()
        history, online = values[periods == 1], values[periods == 2]
        p = lookup[sid]
        if len(online) != layout.online_lengths[p] or not np.array_equal(periods, np.r_[np.ones(len(history)), np.full(len(online), 2)]):
            raise ValueError('Invalid history/online layout')
        pool[p] = (history, online)
    if len(pool) != len(positions):
        raise ValueError('Missing development inputs')
    return pool


def _initialize_worker(names):
    global _FP, _BANKS, _COLUMNS, _LIMITER
    from threadpoolctl import threadpool_limits
    _LIMITER = threadpool_limits(1)
    _FP = detector_module()
    _BANKS = sorted({n.split('__')[0] for n in names}, key=lambda n: int(n[1:]))
    available = _FP.feature_names(_BANKS)
    _COLUMNS = [available.index(n) for n in names]


def _compute_task(task):
    history, online, keep = task
    values = _FP.compute_features_offline(history, online, _BANKS)[:, _COLUMNS]
    # Recompute all causal prefixes before selecting any synthetic fitting rows.
    if keep is not None:
        values = values[keep]
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite selected detector features')
    return np.ascontiguousarray(values, dtype=np.float32)


def block_digest(values):
    return hashlib.sha256(memoryview(np.ascontiguousarray(values)).cast('B')).hexdigest()


def build_cache(directory, name, tasks, rows, names, protocol_sha, workers=4):
    from models.model11.pipeline import write_json
    path = directory/f'{name}.npy'
    manifest_path = directory/f'{name}.cache.json'
    contract = dict(protocol_sha256=protocol_sha, rows=rows, names=names, tasks=len(tasks))
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['contract'] != contract or not path.exists():
            raise ValueError('Feature cache contract mismatch')
        matrix = np.load(path, mmap_mode='r+')
        if matrix.shape != (rows, len(names)) or matrix.dtype != np.float32:
            raise ValueError('Feature cache shape/dtype mismatch')
        for block in manifest['blocks']:
            if block_digest(matrix[block['start']:block['end']]) != block['sha256']:
                raise ValueError('Corrupted committed feature block')
    else:
        if shutil.disk_usage(directory).free < rows*len(names)*4 + 2*1024**3:
            raise RuntimeError('Insufficient disk for this feature cache plus 2 GiB reserve')
        matrix = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(rows, len(names)))
        manifest = dict(contract=contract, completed_tasks=0, completed_rows=0, blocks=[])
        write_json(manifest_path, manifest)
    completed, offset = manifest['completed_tasks'], manifest['completed_rows']
    if completed == len(tasks):
        if offset != rows:
            raise ValueError('Incomplete feature cache row accounting')
        return np.load(path, mmap_mode='r')
    print(f'{name}: restoring {completed}/{len(tasks)} completed series', flush=True)
    block_start = offset
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
            initializer=_initialize_worker, initargs=(names,)) as executor:
        blocks = executor.map(_compute_task, tasks[completed:], chunksize=1, buffersize=8)
        for number, block in enumerate(blocks, completed+1):
            matrix[offset:offset+len(block)] = block
            offset += len(block)
            if number % 32 == 0 or number == len(tasks):
                matrix.flush()
                manifest['blocks'].append(dict(start=block_start, end=offset,
                    sha256=block_digest(matrix[block_start:offset])))
                manifest.update(completed_tasks=number, completed_rows=offset)
                write_json(manifest_path, manifest)
                print(f'{name}: {number}/{len(tasks)} series, {offset:,}/{rows:,} rows', flush=True)
                block_start = offset
    if offset != rows:
        raise ValueError('Feature cache did not fill its declared row count')
    return np.load(path, mmap_mode='r')
