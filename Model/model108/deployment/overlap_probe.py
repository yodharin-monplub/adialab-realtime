"""Match values and period boundaries, never cloud IDs or hidden labels."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pandas as pd

MODES = {"exact": "<f8", "public_float32_precision": "<f4"}


def fingerprint(values, dtype):
    values = np.array(values, dtype=dtype, copy=True).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite series values cannot be matched reliably")
    values[values == 0] = 0  # Treat signed zeros as equal.
    return len(values), hashlib.sha256(values.tobytes()).hexdigest()


def load_reference(model_directory_path):
    path = Path(model_directory_path) / "X_test.reduced.parquet"
    frame = pd.read_parquet(path)
    if frame.index.names != ["id", "time"] or set(frame.columns) != {"value", "period"}:
        raise ValueError("Unexpected public test schema")
    if frame.index.has_duplicates:
        raise ValueError("Duplicate public (id, time) rows")
    frame = frame.sort_index()
    lookup = {mode: defaultdict(list) for mode in MODES}
    count = 0
    for reference_id, group in frame.groupby(level="id", sort=True):
        periods = group.period.to_numpy()
        values = group.value.to_numpy()
        split = int(np.sum(periods == 1))
        if not (0 < split < len(group) and np.all(periods[:split] == 1)
                and np.all(periods[split:] == 2)):
            raise ValueError("Reference must have history followed by online values")
        for mode, dtype in MODES.items():
            lookup[mode][fingerprint(values[:split], dtype)].append(
                (int(reference_id), fingerprint(values[split:], dtype)))
        count += 1
    if count != 100:
        raise ValueError(f"Expected 100 public reference series, received {count}")
    return lookup, count


def percent(numerator, denominator):
    return 100.0 * numerator / denominator if denominator else None


def run_inference(datasets, model_directory_path):
    lookup, reference_count = load_reference(model_directory_path)
    states = {mode: {"history_yes": 0, "yes": 0, "found": set(),
                     "history_found": set()} for mode in MODES}
    total = 0
    complete = False
    yield None  # Crunch ready sentinel; no dataset has been consumed yet.
    try:
        for item in datasets:
            if len(item) == 2:
                history, online = item
            elif len(item) == 3:
                _, history, online = item
            else:
                raise ValueError("Expected (history, online) or (id, history, online)")
            # History is available in full; online remains a one-pass iterator.
            history = np.asarray(history).reshape(-1)
            candidates = {mode: lookup[mode].get(fingerprint(history, dtype), [])
                          for mode, dtype in MODES.items()}
            hashes = {mode: hashlib.sha256() for mode in MODES if candidates[mode]}
            length = 0
            for value in online:
                value = float(value)
                if not np.isfinite(value):
                    raise ValueError("Nonfinite online value")
                if value == 0:
                    value = 0.0
                for mode, digest in hashes.items():
                    digest.update(struct.pack("<d" if mode == "exact" else "<f", value))
                length += 1
                # Diagnostic-only score: overlap never affects predictions.
                yield 0.5
            total += 1
            for mode, state in states.items():
                if candidates[mode]:
                    state["history_yes"] += 1
                    state["history_found"].update(x[0] for x in candidates[mode])
                    key = (length, hashes[mode].hexdigest())
                    found = {ref_id for ref_id, signature in candidates[mode]
                             if signature == key}
                    state["yes"] += bool(found)
                    state["found"].update(found)
            if total % 1000 == 0:
                print(f"OVERLAP_PROGRESS completed_series={total} "
                      f"exact_yes={states['exact']['yes']}", flush=True)
        complete = True
    finally:
        report = {
            "complete": complete,
            "scope": "this infer invocation; INFER_PARALLELISM=1",
            "reference_series": reference_count,
            "completed_cloud_series": total,
            "expected_cloud_series": 10000,
            "expected_cloud_count_reached": total == 10000,
            "modes": {},
        }
        for mode, state in states.items():
            report["modes"][mode] = {
                "yes_cloud_series": state["yes"],
                "yes_percentage": percent(state["yes"], total),
                "reference_series_found": len(state["found"]),
                "reference_found_percentage": percent(len(state["found"]), reference_count),
                "history_only_yes_cloud_series": state["history_yes"],
                "history_reference_series_found": len(state["history_found"]),
            }
        print("OVERLAP_FINAL " + json.dumps(report, sort_keys=True), flush=True)
        exact = report["modes"]["exact"]
        percentage = exact["yes_percentage"]
        display = "N/A (no completed series)" if percentage is None else f"{percentage:.4f}%"
        print(f"OVERLAP {'COMPLETE' if complete else 'PARTIAL'}: "
              f"YES = {exact['yes_cloud_series']}/{total} ({display}); "
              f"public test found = {exact['reference_series_found']}/{reference_count} "
              f"({exact['reference_found_percentage']:.2f}%)", flush=True)
