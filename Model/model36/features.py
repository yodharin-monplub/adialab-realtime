"""Resumable row-aligned cache for Model36 AR-Focus features."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import shutil

import numpy as np

from models.model11.pipeline import sha256, write_json
from models.model20.features import block_digest
from models.model36.arp_focus_features import N_FEATURES, compute_features_offline, feature_names


def _worker(task):
    position, history, online = task
    values = compute_features_offline(history, online)
    return int(position), np.ascontiguousarray(values, dtype=np.float32)


def build_cache(directory, fingerprint, layout, pool, workers=4):
    path = directory / "arp_focus.npy"
    manifest_path = directory / "arp_focus.cache.json"
    positions = np.flatnonzero(layout.inner_series_fold >= 0)
    contract = dict(protocol_sha256=fingerprint, rows=layout.row_count,
                    columns=N_FEATURES, names=feature_names(), tasks=len(positions))
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["contract"] != contract or not path.exists():
            raise ValueError("AR-Focus cache contract mismatch")
        matrix = np.load(path, mmap_mode="r+")
        if matrix.shape != (layout.row_count, N_FEATURES) or matrix.dtype != np.float32:
            raise ValueError("AR-Focus cache shape or dtype mismatch")
        for block in manifest["blocks"]:
            if block_digest(matrix[block["start"]:block["end"]]) != block["sha256"]:
                raise ValueError("Corrupted AR-Focus cache block")
    else:
        required = layout.row_count * N_FEATURES * 4
        if shutil.disk_usage(directory).free < required + 2 * 1024**3:
            raise RuntimeError("Insufficient disk for AR-Focus cache plus 2 GiB reserve")
        matrix = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float32, shape=(layout.row_count, N_FEATURES))
        manifest = dict(contract=contract, completed_tasks=0, completed_rows=0, blocks=[])
        write_json(manifest_path, manifest)
    completed = manifest["completed_tasks"]
    offset = manifest["completed_rows"]
    if completed == len(positions):
        if offset != layout.row_count:
            raise ValueError("AR-Focus cache row accounting mismatch")
        return np.load(path, mmap_mode="r")
    print(f"AR-Focus cache: restoring {completed}/{len(positions)} series", flush=True)
    tasks = ((int(position), *pool[int(position)]) for position in positions[completed:])
    block_start = offset
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        for number, (position, values) in enumerate(
            executor.map(_worker, tasks, chunksize=1, buffersize=8), completed + 1
        ):
            if not np.all(layout.series_position[offset:offset + len(values)] == position):
                raise ValueError("AR-Focus cache series alignment failed")
            matrix[offset:offset + len(values)] = values
            offset += len(values)
            if number % 64 == 0 or number == len(positions):
                matrix.flush()
                manifest["blocks"].append(dict(
                    start=block_start, end=offset,
                    sha256=block_digest(matrix[block_start:offset])))
                manifest.update(completed_tasks=number, completed_rows=offset)
                write_json(manifest_path, manifest)
                print(f"AR-Focus cache: {number}/{len(positions)} series, {offset:,} rows", flush=True)
                block_start = offset
    if offset != layout.row_count:
        raise ValueError("AR-Focus cache did not fill all development rows")
    return np.load(path, mmap_mode="r")
