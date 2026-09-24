#!/usr/bin/env python3
"""Build all-detector caches with resumable, disjoint series shards.

The six cache matrices are independent. Each cache is divided into stable
complete-series ranges, and eight worker processes write non-overlapping row
slices. Every range flushes before advancing its sidecar, so interruption can
lose at most the uncheckpointed tail of a shard. Existing sequential-prefix
sidecars from :func:`feature_pipeline.build_feature_cache` are reused.

Only ``train/`` is loaded. Reduced test files and protected competition data
are outside this program's input contract.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-model6-cache")

import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from competition_data import TrainingData  # noqa: E402
from feature_pipeline import (  # noqa: E402
    ARCHIVE_CANDIDATE_BANKS,
    ARCHIVE_CORE_BANKS,
    CANDIDATE_BANKS,
    CORE_BANKS,
    FEATURE_VERSION,
    NOVEL_BANKS,
    SPECIALIST_BANKS,
    _feature_signature,
    compute_features_offline,
    feature_names,
)
from models.model5.config import make_model5_config  # noqa: E402


CACHE_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("core_features", CORE_BANKS),
    ("candidate_features", CANDIDATE_BANKS),
    ("specialist_features", SPECIALIST_BANKS),
    ("novel_features", NOVEL_BANKS),
    ("archive_core_features", ARCHIVE_CORE_BANKS),
    ("archive_candidate_features", ARCHIVE_CANDIDATE_BANKS),
)

# More shards than CPUs permit dynamic load balancing when cheap ranges finish.
SHARD_COUNTS = {
    "core_features": 8,
    "candidate_features": 8,
    "specialist_features": 16,
    "novel_features": 4,
    "archive_core_features": 8,
    "archive_candidate_features": 2,
}
MAX_PROCESSES = 8
PROGRESS_EVERY = 100
RESERVE_BYTES = 5 * 1024**3

_WORKER_TRAINING: dict[str, TrainingData] = {}


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _allocated_bytes(path: Path) -> int:
    return path.stat().st_blocks * 512 if path.exists() else 0


def _worker_training(path: str) -> TrainingData:
    data = _WORKER_TRAINING.get(path)
    if data is None:
        data = TrainingData(Path(path))
        _WORKER_TRAINING[path] = data
    return data


def _partial_path(cache_path: Path, name: str) -> Path:
    return cache_path / f"{name}.f32.mmap.partial"


def _matrix_path(cache_path: Path, name: str) -> Path:
    return cache_path / f"{name}.f32.mmap"


def _sequential_prefix(
    cache_path: Path,
    name: str,
    rows: int,
    columns: int,
    signature: str,
    row_starts: np.ndarray,
) -> int:
    path = cache_path / f"{name}.progress.json"
    if not path.exists():
        return 0
    saved = json.loads(path.read_text(encoding="utf-8"))
    expected = (rows, columns, signature)
    actual = (
        int(saved["rows"]),
        int(saved["columns"]),
        saved["feature_signature"],
    )
    if actual != expected:
        raise ValueError(f"{path} belongs to another data/feature contract")
    prefix = int(saved["next_series"])
    if int(saved["next_row"]) != int(row_starts[prefix]):
        raise ValueError(f"{path} has an inconsistent row offset")
    return prefix


def _shard_path(cache_path: Path, name: str, start: int, end: int) -> Path:
    return cache_path / f"{name}.shard_{start}_{end}.progress.json"


def _make_tasks(
    training: TrainingData, cache_path: Path
) -> tuple[list[dict[str, Any]], int]:
    rows = training.online_row_count
    row_starts = np.r_[
        np.int64(0), np.cumsum(training.online_lengths, dtype=np.int64)
    ]
    tasks: list[dict[str, Any]] = []
    required = 0
    for name, banks in CACHE_SPECS:
        manifest = cache_path / f"{name}.manifest.json"
        if manifest.exists():
            continue
        columns = len(feature_names(banks))
        signature = _feature_signature(banks)
        partial = _partial_path(cache_path, name)
        complete_matrix = _matrix_path(cache_path, name)
        expected_bytes = rows * columns * 4
        if complete_matrix.exists() and not partial.exists():
            if complete_matrix.stat().st_size != expected_bytes:
                raise ValueError(f"unfinished {complete_matrix} has the wrong size")
            os.replace(complete_matrix, partial)
        if not partial.exists():
            matrix = np.memmap(
                partial, dtype=np.float32, mode="w+", shape=(rows, columns)
            )
            matrix.flush()
            del matrix
        elif partial.stat().st_size != expected_bytes:
            raise ValueError(f"{partial} has the wrong logical size")
        required += max(expected_bytes - _allocated_bytes(partial), 0)
        prefix = _sequential_prefix(
            cache_path, name, rows, columns, signature, row_starts
        )
        remaining = len(training) - prefix
        shard_count = min(SHARD_COUNTS[name], max(remaining, 1))
        for shard in range(shard_count):
            start = prefix + remaining * shard // shard_count
            end = prefix + remaining * (shard + 1) // shard_count
            if start == end:
                continue
            tasks.append(
                {
                    "name": name,
                    "banks": tuple(banks),
                    "rows": rows,
                    "columns": columns,
                    "signature": signature,
                    "start": start,
                    "end": end,
                }
            )
    return tasks, required


def _build_shard(
    train_path: str,
    cache_path_string: str,
    task: dict[str, Any],
) -> dict[str, Any]:
    """Write one stable, non-overlapping complete-series interval."""

    training = _worker_training(train_path)
    cache_path = Path(cache_path_string)
    name = str(task["name"])
    start = int(task["start"])
    end = int(task["end"])
    rows = int(task["rows"])
    columns = int(task["columns"])
    signature = str(task["signature"])
    banks = tuple(task["banks"])
    row_starts = np.r_[
        np.int64(0), np.cumsum(training.online_lengths, dtype=np.int64)
    ]
    progress_path = _shard_path(cache_path, name, start, end)
    next_series = start
    if progress_path.exists():
        saved = json.loads(progress_path.read_text(encoding="utf-8"))
        contract = (
            int(saved["rows"]),
            int(saved["columns"]),
            saved["feature_signature"],
            int(saved["start"]),
            int(saved["end"]),
        )
        expected = (rows, columns, signature, start, end)
        if contract != expected:
            raise ValueError(f"{progress_path} has an incompatible shard contract")
        next_series = int(saved["next_series"])
    if not start <= next_series <= end:
        raise ValueError(f"{progress_path} has an invalid next_series")
    if next_series == end:
        return {"name": name, "start": start, "end": end, "resumed": True}

    matrix = np.memmap(
        _partial_path(cache_path, name),
        dtype=np.float32,
        mode="r+",
        shape=(rows, columns),
    )
    resumed = next_series > start
    started = time.perf_counter()
    completed_at_start = next_series
    for position in range(next_series, end):
        record = training.record(position)
        block = compute_features_offline(
            record.historical, record.online, banks
        ).astype(np.float32, copy=False)
        left = int(row_starts[position])
        right = int(row_starts[position + 1])
        if block.shape != (right - left, columns):
            raise ValueError(f"unexpected block shape for {name} series {position}")
        matrix[left:right] = block
        next_series = position + 1
        if next_series % PROGRESS_EVERY == 0 or next_series == end:
            matrix.flush()
            _atomic_json(
                progress_path,
                {
                    "rows": rows,
                    "columns": columns,
                    "feature_signature": signature,
                    "start": start,
                    "end": end,
                    "next_series": next_series,
                    "next_row": int(row_starts[next_series]),
                },
            )
            elapsed = time.perf_counter() - started
            rate = (next_series - completed_at_start) / max(elapsed, 1e-9)
            print(
                f"shard[{name}:{start}-{end}] {next_series}/{end} "
                f"series ({rate:.2f}/s)",
                flush=True,
            )
    del matrix
    return {"name": name, "start": start, "end": end, "resumed": resumed}


def _write_shared_metadata(training: TrainingData, path: Path) -> None:
    row_id = np.repeat(training.series_ids, training.online_lengths).astype(np.int32)
    series_position = np.repeat(
        np.arange(len(training), dtype=np.int32), training.online_lengths
    )
    time_online = np.concatenate(
        [np.arange(length, dtype=np.int16) for length in training.online_lengths]
    )
    target = np.concatenate([record.target for record in training]).astype(np.uint8)
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            row_id=row_id,
            series_position=series_position,
            time_online=time_online,
            target=target,
            series_ids=training.series_ids.astype(np.int32),
            online_lengths=training.online_lengths.astype(np.int16),
            historical_lengths=training.historical_lengths.astype(np.int16),
            tau_index=training.tau_index.astype(np.int16),
        )


def _finalize(
    training: TrainingData, cache_path: Path, tasks: list[dict[str, Any]]
) -> None:
    tasks_by_name: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        tasks_by_name.setdefault(str(task["name"]), []).append(task)
    metadata_template = cache_path / ".all_detector.metadata.npz"
    _write_shared_metadata(training, metadata_template)
    for name, banks in CACHE_SPECS:
        manifest_path = cache_path / f"{name}.manifest.json"
        if manifest_path.exists():
            continue
        cache_tasks = tasks_by_name.get(name, [])
        if not cache_tasks:
            raise AssertionError(f"no completion tasks recorded for {name}")
        for task in cache_tasks:
            progress_path = _shard_path(
                cache_path, name, int(task["start"]), int(task["end"])
            )
            saved = json.loads(progress_path.read_text(encoding="utf-8"))
            if int(saved["next_series"]) != int(task["end"]):
                raise RuntimeError(f"incomplete shard: {progress_path}")
        columns = len(feature_names(banks))
        rows = training.online_row_count
        partial = _partial_path(cache_path, name)
        matrix = np.memmap(
            partial, dtype=np.float32, mode="r+", shape=(rows, columns)
        )
        matrix.flush()
        del matrix
        shutil.copyfile(metadata_template, cache_path / f"{name}.metadata.npz")
        os.replace(partial, _matrix_path(cache_path, name))
        _atomic_json(
            manifest_path,
            {
                "feature_version": FEATURE_VERSION,
                "feature_signature": _feature_signature(banks),
                "banks": list(banks),
                "rows": rows,
                "columns": columns,
                "dtype": "float32",
                "feature_names": feature_names(banks),
            },
        )
        (cache_path / f"{name}.progress.json").unlink(missing_ok=True)
        for task in cache_tasks:
            _shard_path(
                cache_path, name, int(task["start"]), int(task["end"])
            ).unlink(missing_ok=True)
        print(f"CACHE COMPLETE: {name} ({rows:,} x {columns:,})", flush=True)
    metadata_template.unlink(missing_ok=True)


def main() -> None:
    config = make_model5_config(PROJECT_ROOT)
    config.cache_path.mkdir(parents=True, exist_ok=True)
    training = TrainingData(config.train_path)
    tasks, required = _make_tasks(training, config.cache_path)
    available = shutil.disk_usage(config.cache_path).free
    print(
        json.dumps(
            {
                "training_series": len(training),
                "online_rows": training.online_row_count,
                "shards": len(tasks),
                "processes": min(MAX_PROCESSES, len(tasks)),
                "remaining_cache_gib": round(required / 1024**3, 3),
                "free_gib": round(available / 1024**3, 3),
                "reserved_gib": RESERVE_BYTES // 1024**3,
            },
            indent=2,
        ),
        flush=True,
    )
    if not tasks:
        print("all detector caches are already complete", flush=True)
        return
    if required + RESERVE_BYTES > available:
        raise OSError(
            f"need {required / 1024**3:.1f} GiB plus 5 GiB reserve, "
            f"but only {available / 1024**3:.1f} GiB is free"
        )
    train_path = str(config.train_path)
    cache_path = str(config.cache_path)
    del training
    # Longest expected tasks first; the executor dynamically assigns the tail.
    weights = {
        "specialist_features": 1.8,
        "core_features": 0.6,
        "candidate_features": 0.5,
        "archive_core_features": 0.4,
        "novel_features": 0.15,
        "archive_candidate_features": 0.04,
    }
    tasks.sort(
        key=lambda task: -(
            (int(task["end"]) - int(task["start"]))
            * weights[str(task["name"])]
        )
    )
    with ProcessPoolExecutor(max_workers=min(MAX_PROCESSES, len(tasks))) as executor:
        futures = {
            executor.submit(_build_shard, train_path, cache_path, task): task
            for task in tasks
        }
        for future in as_completed(futures):
            result = future.result()
            print(f"SHARD COMPLETE: {json.dumps(result, sort_keys=True)}", flush=True)
    training = TrainingData(Path(train_path))
    _finalize(training, Path(cache_path), tasks)
    print("all detector caches completed and validated", flush=True)


if __name__ == "__main__":
    main()
