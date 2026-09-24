#!/usr/bin/env python3
"""Audit every retained detector implementation and active feature cache.

The audit has three layers:

1. Parse, compile, and import every ``detectors*.py`` file in ``core`` and
   ``candidate``; validate feature-name contracts and identify inherited d3
   prefixes.
2. Execute every module's independent ``__main__`` parity test plus the
   project regression suite and composite synthetic fixtures.
3. Optionally inspect the production feature caches for constants, exact
   duplicates, and within-bank near duplicates.  ``--cache-scan full`` reads
   every cached value, hashes each raw cache, and verifies sample-discovered
   exact duplicates over all rows.

The command writes a machine-readable JSON artifact and a concise Markdown
report.  Redundant features are reported, never deleted, because existing
model feature contracts must remain stable.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import inspect
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
DETECTOR_ROOT = PROJECT / "detector"
CORE_DIRECTORY = DETECTOR_ROOT / "core"
CANDIDATE_DIRECTORY = DETECTOR_ROOT / "candidate"
ARTIFACT_DIRECTORY = PROJECT / "models" / "artifacts"
DEFAULT_JSON = ARTIFACT_DIRECTORY / "detector_audit.json"
DEFAULT_MARKDOWN = DETECTOR_ROOT / "AUDIT.md"
DETECTOR_IMPORT = re.compile(r"^(?:detectors(?:\d+)?|channels18)$")

# These accepted contracts must not change as a side effect of repairing an
# archived/specialist detector.
EXPECTED_PRODUCTION_SIGNATURES = {
    "core": "cde3ea9d9d3494cbf16d29c015afc4d53c598752a08aca03884f52a061287c54",
    "candidate": "78b1ada988001a974f1cdf151e5cf296e72ba959dc6903d30aa46e1e80eb8b25",
    "novel": "509c3e60c3ef7f326c6393e242c9145f51b0d02b9c3a6bf15de18963475c7465",
}


def _configure_imports() -> None:
    """Expose legacy sibling import names in the same order as production."""

    for directory in (CANDIDATE_DIRECTORY, CORE_DIRECTORY, PROJECT):
        value = str(directory)
        if value not in sys.path:
            sys.path.insert(0, value)


def _detector_number(path: Path) -> int:
    suffix = path.stem.removeprefix("detectors")
    return int(suffix) if suffix else 1


def detector_files() -> list[Path]:
    """Return all 49 detector sources in numeric lineage order."""

    files = list(CORE_DIRECTORY.glob("detectors*.py"))
    files.extend(CANDIDATE_DIRECTORY.glob("detectors*.py"))
    return sorted(files, key=lambda path: (_detector_number(path), path.parent.name))


def _has_main_guard(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not isinstance(test, ast.Compare) or len(test.ops) != 1:
            continue
        if not isinstance(test.left, ast.Name) or test.left.id != "__name__":
            continue
        if not isinstance(test.ops[0], ast.Eq) or len(test.comparators) != 1:
            continue
        value = test.comparators[0]
        if isinstance(value, ast.Constant) and value.value == "__main__":
            return True
    return False


def _source_dependencies(tree: ast.AST) -> list[str]:
    dependencies: set[str] = set()
    for node in ast.walk(tree):
        candidates: list[str] = []
        if isinstance(node, ast.Import):
            candidates.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            candidates.append(node.module.split(".")[0])
        dependencies.update(name for name in candidates if DETECTOR_IMPORT.match(name))
    return sorted(dependencies, key=lambda name: (_module_number(name), name))


def _module_number(name: str) -> int:
    suffix = name.removeprefix("detectors")
    if name == "channels18":
        return 18
    return int(suffix) if suffix else 1


def _feature_name_metadata(module: Any, d3_names: Sequence[str]) -> dict[str, Any]:
    function = getattr(module, "feature_names", None)
    if function is None:
        return {
            "feature_name_contract": "dynamic/undeclared",
            "feature_count": None,
            "duplicate_feature_names": [],
            "inherits_complete_d3_prefix": False,
            "d3_shared_name_count": 0,
        }

    signature = inspect.signature(function)
    required = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if required:
        # d31's count is fitted externally. Three templates are sufficient to
        # validate name uniqueness and its per-template contract.
        names = list(function(3))
        contract = f"dynamic ({', '.join(parameter.name for parameter in required)})"
    else:
        names = list(function())
        contract = "fixed"
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    d3_set = set(d3_names)
    return {
        "feature_name_contract": contract,
        "feature_count": len(names),
        "declared_n_features": getattr(module, "N_FEATURES", None),
        "duplicate_feature_names": duplicates,
        "inherits_complete_d3_prefix": (
            len(names) >= len(d3_names) and names[: len(d3_names)] == list(d3_names)
        ),
        "d3_shared_name_count": len(set(names) & d3_set),
    }


def inspect_module(path: Path, d3_names: Sequence[str]) -> dict[str, Any]:
    """Compile/import one source and capture its static feature contract."""

    relative = path.relative_to(PROJECT).as_posix()
    source = path.read_text(encoding="utf-8")
    result: dict[str, Any] = {
        "id": f"d{_detector_number(path)}",
        "group": path.parent.name,
        "module": path.stem,
        "path": relative,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "source_lines": len(source.splitlines()),
        "compile": "pass",
        "import": "pass",
        "issues": [],
    }
    try:
        tree = ast.parse(source, filename=str(path))
        compile(source, str(path), "exec")
    except Exception as error:  # pragma: no cover - failure reporting path
        result["compile"] = "fail"
        result["issues"].append(f"{type(error).__name__}: {error}")
        return result
    result["dependencies"] = _source_dependencies(tree)
    result["has_local_self_test"] = _has_main_guard(tree)

    try:
        module = importlib.import_module(path.stem)
        stream_classes = sorted(
            name
            for name, value in vars(module).items()
            if inspect.isclass(value)
            and name.startswith("StreamDetector")
            and value.__module__ == module.__name__
        )
        result["stream_classes"] = stream_classes
        result["offline_entry_points"] = sorted(
            name
            for name in (
                "compute_features_offline",
                "extra_features",
                "features_from_channels",
                "compute_features_from_channels",
            )
            if callable(getattr(module, name, None))
        )
        result.update(_feature_name_metadata(module, d3_names))
        declared = result.get("declared_n_features")
        count = result.get("feature_count")
        if declared is not None and result["feature_name_contract"] == "fixed":
            if int(declared) != int(count):
                result["issues"].append(
                    f"N_FEATURES={declared} but feature_names has {count} entries"
                )
        if result.get("duplicate_feature_names"):
            result["issues"].append("duplicate names within feature contract")
        if not stream_classes:
            result["issues"].append("no StreamDetector class")
    except Exception as error:  # pragma: no cover - failure reporting path
        result["import"] = "fail"
        result["issues"].append(f"{type(error).__name__}: {error}")
    return result


def _self_test_one(path: Path, timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(CORE_DIRECTORY), str(CANDIDATE_DIRECTORY), str(PROJECT)]
    )
    try:
        completed = subprocess.run(
            [sys.executable, str(path)],
            cwd=PROJECT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        output = (completed.stdout + completed.stderr).strip()
        return {
            "status": "pass" if completed.returncode == 0 else "fail",
            "return_code": completed.returncode,
            "seconds": round(time.perf_counter() - started, 3),
            "output_tail": output[-2000:],
        }
    except subprocess.TimeoutExpired as error:
        output = "".join(
            part.decode(errors="replace") if isinstance(part, bytes) else (part or "")
            for part in (error.stdout, error.stderr)
        )
        return {
            "status": "timeout",
            "return_code": None,
            "seconds": round(time.perf_counter() - started, 3),
            "output_tail": output[-2000:],
        }


def run_self_tests(
    files: Sequence[Path],
    inventory: Sequence[dict[str, Any]],
    jobs: int,
    timeout: float,
) -> dict[str, dict[str, Any]]:
    """Run independent module parity checks concurrently."""

    has_test = {item["path"]: item.get("has_local_self_test", False) for item in inventory}
    selected = [path for path in files if has_test[path.relative_to(PROJECT).as_posix()]]
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as executor:
        futures = {executor.submit(_self_test_one, path, timeout): path for path in selected}
        for future in as_completed(futures):
            path = futures[future]
            key = path.relative_to(PROJECT).as_posix()
            results[key] = future.result()
            print(f"self-test {results[key]['status']:>4} {key}", flush=True)
    for path in files:
        key = path.relative_to(PROJECT).as_posix()
        if key not in results:
            results[key] = {
                "status": "covered_by_central_suite",
                "return_code": None,
                "seconds": 0.0,
                "output_tail": "no local __main__ test",
            }
    return results


def run_regression_suite(timeout: float) -> dict[str, Any]:
    """Run the dependency-free project regression suite."""

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, str(PROJECT / "tests" / "run_tests.py")],
            cwd=PROJECT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        output = (completed.stdout + completed.stderr).strip()
        match = re.search(r"PASS: (\d+) tests", output)
        return {
            "status": "pass" if completed.returncode == 0 else "fail",
            "return_code": completed.returncode,
            "test_count": int(match.group(1)) if match else None,
            "seconds": round(time.perf_counter() - started, 3),
            "output_tail": output[-4000:],
        }
    except subprocess.TimeoutExpired as error:  # pragma: no cover
        return {
            "status": "timeout",
            "return_code": None,
            "test_count": None,
            "seconds": round(time.perf_counter() - started, 3),
            "output_tail": str(error),
        }


def _ar_values(seed: int, length: int, coefficient: float, heavy: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_t(5.0, length) if heavy else rng.normal(size=length)
    values = np.empty(length, dtype=np.float64)
    values[0] = noise[0]
    for index in range(1, length):
        values[index] = coefficient * values[index - 1] + noise[index]
    return values


def _synthetic_fixtures() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    fixtures: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    gaussian = _ar_values(100, 964, 0.0, False)
    fixtures["gaussian"] = (gaussian[:900], gaussian[900:])
    heavy = _ar_values(101, 964, 0.72, True)
    fixtures["heavy_tail_ar"] = (heavy[:900], heavy[900:])
    fixtures["constant"] = (np.ones(900), np.r_[np.ones(32), np.full(32, 2.0)])
    short = _ar_values(102, 903, -0.35, False)
    fixtures["short_online"] = (short[:900], short[900:])

    mean_break = _ar_values(103, 964, 0.45, True)
    mean_online = mean_break[900:].copy()
    mean_online[32:] += 1.25
    fixtures["mean_break"] = (mean_break[:900], mean_online)

    scale_break = _ar_values(104, 964, 0.45, True)
    scale_online = scale_break[900:].copy()
    scale_online[32:] *= 2.0
    fixtures["scale_break"] = (scale_break[:900], scale_online)

    dependence = _ar_values(105, 964, 0.05, False)
    dependence_online = dependence[900:].copy()
    for index in range(33, dependence_online.size):
        dependence_online[index] += 0.8 * dependence_online[index - 1]
    fixtures["dependence_break"] = (dependence[:900], dependence_online)
    return fixtures


def run_synthetic_composite_checks() -> dict[str, Any]:
    """Exercise every active bank on null and break-shaped causal fixtures."""

    import feature_pipeline as pipeline

    groups = {
        "core": pipeline.CORE_BANKS,
        "candidate": pipeline.CANDIDATE_BANKS,
        "specialist": pipeline.SPECIALIST_BANKS,
        "novel": pipeline.NOVEL_BANKS,
        "model8": pipeline.MODEL8_BANKS,
    }
    cases: list[dict[str, Any]] = []
    for fixture_name, (history, online) in _synthetic_fixtures().items():
        for group_name, banks in groups.items():
            started = time.perf_counter()
            try:
                offline = pipeline.compute_features_offline(history, online, banks)
                repeated = pipeline.compute_features_offline(history, online, banks)
                detector = pipeline.CoreStreamDetector(history, banks)
                streamed = np.vstack([detector.step(value) for value in online])
                split = max(1, online.size // 2)
                changed = online.copy()
                changed[split:] = 17.0 - 4.0 * changed[split:]
                perturbed = pipeline.compute_features_offline(history, changed, banks)
                maximum_error = float(np.max(np.abs(offline - streamed)))
                deterministic = bool(np.array_equal(offline, repeated))
                causal = bool(np.array_equal(offline[:split], perturbed[:split]))
                finite = bool(np.all(np.isfinite(offline)))
                # Exact-constant history is outside this competition's
                # standardized-data contract. Several old accumulators take
                # different floating-point cancellation paths when their
                # fallback scale is 1e-12. Preserve accepted production
                # semantics and audit that edge case for finiteness/causality,
                # while requiring strict parity on every supported fixture.
                strict_parity = fixture_name != "constant"
                parity_passed = maximum_error < 1e-8
                passed = (
                    (parity_passed or not strict_parity)
                    and deterministic
                    and causal
                    and finite
                )
                cases.append(
                    {
                        "fixture": fixture_name,
                        "group": group_name,
                        "status": "pass" if passed else "fail",
                        "shape": list(offline.shape),
                        "maximum_streaming_error": maximum_error,
                        "strict_streaming_parity_required": strict_parity,
                        "streaming_parity": parity_passed,
                        "deterministic": deterministic,
                        "causal": causal,
                        "finite": finite,
                        "seconds": round(time.perf_counter() - started, 3),
                    }
                )
            except Exception as error:  # pragma: no cover - audit failure path
                cases.append(
                    {
                        "fixture": fixture_name,
                        "group": group_name,
                        "status": "fail",
                        "error": f"{type(error).__name__}: {error}",
                        "seconds": round(time.perf_counter() - started, 3),
                    }
                )
            print(
                f"fixture {cases[-1]['status']:>4} {fixture_name}/{group_name}",
                flush=True,
            )

    # Interleave two independent instances. A hidden module-global state bug
    # would make these differ from their isolated offline reference matrices.
    first = _ar_values(201, 940, 0.61, True)
    second = _ar_values(202, 940, -0.43, True)
    isolation: dict[str, Any] = {}
    for group_name, banks in groups.items():
        first_history, first_online = first[:900], first[900:]
        second_history, second_online = second[:900], second[900:]
        first_expected = pipeline.compute_features_offline(
            first_history, first_online, banks
        )
        second_expected = pipeline.compute_features_offline(
            second_history, second_online, banks
        )
        first_detector = pipeline.CoreStreamDetector(first_history, banks)
        second_detector = pipeline.CoreStreamDetector(second_history, banks)
        first_rows, second_rows = [], []
        for first_value, second_value in zip(first_online, second_online):
            first_rows.append(first_detector.step(first_value))
            second_rows.append(second_detector.step(second_value))
        first_error = float(np.max(np.abs(first_expected - np.vstack(first_rows))))
        second_error = float(np.max(np.abs(second_expected - np.vstack(second_rows))))
        isolation[group_name] = {
            "status": "pass" if max(first_error, second_error) < 1e-8 else "fail",
            "maximum_error": max(first_error, second_error),
        }
    return {"cases": cases, "instance_isolation": isolation}


def training_history_profile() -> dict[str, Any]:
    """Confirm whether the known constant-history edge case occurs in train."""

    from competition_data import TrainingData

    data = TrainingData(PROJECT / "train")
    standard_deviation = np.asarray(
        [float(np.std(record.historical)) for record in data], dtype=np.float64
    )
    return {
        "series": int(standard_deviation.size),
        "minimum_standard_deviation": float(np.min(standard_deviation)),
        "maximum_standard_deviation": float(np.max(standard_deviation)),
        "standard_deviation_at_most_1e_8": int(
            np.count_nonzero(standard_deviation <= 1e-8)
        ),
        "constant_history_edge_case_present": bool(
            np.any(standard_deviation <= 1e-8)
        ),
    }


def _hash_column(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _sample_rows(row_count: int, requested: int) -> np.ndarray:
    count = min(int(requested), int(row_count))
    if count == row_count:
        return np.arange(row_count, dtype=np.int64)
    return np.unique(np.linspace(0, row_count - 1, count, dtype=np.int64))


def _scan_one_cache(
    cache_name: str,
    scan_mode: str,
    sample_rows: int,
    chunk_rows: int,
) -> tuple[dict[str, Any], np.ndarray]:
    manifest_path = ARTIFACT_DIRECTORY / f"{cache_name}.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    matrix_path = ARTIFACT_DIRECTORY / f"{cache_name}.f32.mmap"
    rows, columns = int(manifest["rows"]), int(manifest["columns"])
    expected_bytes = rows * columns * np.dtype(np.float32).itemsize
    result: dict[str, Any] = {
        "name": cache_name,
        "path": matrix_path.relative_to(PROJECT).as_posix(),
        "rows": rows,
        "columns": columns,
        "banks": manifest["banks"],
        "feature_signature": manifest["feature_signature"],
        "expected_bytes": expected_bytes,
        "actual_bytes": matrix_path.stat().st_size,
        "size_valid": matrix_path.stat().st_size == expected_bytes,
    }
    matrix = np.memmap(matrix_path, dtype=np.float32, mode="r", shape=(rows, columns))
    indices = _sample_rows(rows, sample_rows)
    sample = np.asarray(matrix[indices], dtype=np.float32)
    result["sample_rows"] = int(sample.shape[0])

    if scan_mode == "full":
        minimum = np.full(columns, np.inf, dtype=np.float64)
        maximum = np.full(columns, -np.inf, dtype=np.float64)
        digest = hashlib.sha256()
        for left in range(0, rows, chunk_rows):
            block = np.asarray(matrix[left : min(left + chunk_rows, rows)])
            minimum = np.minimum(minimum, np.min(block, axis=0))
            maximum = np.maximum(maximum, np.max(block, axis=0))
            digest.update(memoryview(np.ascontiguousarray(block)).cast("B"))
        result["raw_sha256"] = digest.hexdigest()
        result["range_scope"] = "all_rows"
    else:
        minimum = np.min(sample, axis=0).astype(np.float64)
        maximum = np.max(sample, axis=0).astype(np.float64)
        result["raw_sha256"] = None
        result["range_scope"] = "sample"

    names = list(manifest["feature_names"])
    constant_indices = np.flatnonzero(minimum == maximum)
    result["constant_columns"] = [
        {
            "index": int(index),
            "name": names[index],
            "value": float(minimum[index]),
        }
        for index in constant_indices
    ]
    result["minimum_value"] = float(np.min(minimum))
    result["maximum_value"] = float(np.max(maximum))
    result["all_finite"] = bool(np.all(np.isfinite(minimum)) and np.all(np.isfinite(maximum)))
    del matrix
    return result, sample


def _selected_feature_sets() -> dict[str, set[str]]:
    paths = {
        "submission_3": (
            PROJECT
            / "submissions"
            / "submission_3"
            / "candidate_feature_selection.npz"
        ),
        "joint_768": (
            PROJECT
            / "models"
            / "artifacts"
            / "joint_lightgbm_v1"
            / "selected_joint_features_fixedseed_v2.npz"
        ),
    }
    output: dict[str, set[str]] = {}
    for label, path in paths.items():
        if not path.exists():
            output[label] = set()
            continue
        with np.load(path) as values:
            output[label] = {str(name) for name in values["names"]}
    return output


def _verify_exact_group(
    group: Sequence[dict[str, Any]],
    manifests: dict[str, dict[str, Any]],
    chunk_rows: int,
) -> bool:
    first = group[0]
    first_manifest = manifests[first["cache"]]
    rows = int(first_manifest["rows"])
    opened: dict[str, np.memmap] = {}
    try:
        for item in group:
            manifest = manifests[item["cache"]]
            if int(manifest["rows"]) != rows:
                return False
            opened[item["cache"]] = np.memmap(
                ARTIFACT_DIRECTORY / f"{item['cache']}.f32.mmap",
                dtype=np.float32,
                mode="r",
                shape=(int(manifest["rows"]), int(manifest["columns"])),
            )
        for left in range(0, rows, chunk_rows):
            right = min(left + chunk_rows, rows)
            reference = opened[first["cache"]][left:right, first["index"]]
            for item in group[1:]:
                values = opened[item["cache"]][left:right, item["index"]]
                if not np.array_equal(reference, values):
                    return False
        return True
    finally:
        opened.clear()


def scan_feature_caches(
    scan_mode: str,
    sample_rows: int,
    near_rows: int,
    near_threshold: float,
    chunk_rows: int,
) -> dict[str, Any]:
    """Find numeric redundancy without altering any accepted feature column."""

    import feature_pipeline as pipeline

    cache_names = ("core_features", "candidate_features", "novel_features")
    cache_results: list[dict[str, Any]] = []
    samples: list[np.ndarray] = []
    manifests: dict[str, dict[str, Any]] = {}
    references: list[dict[str, Any]] = []
    offset = 0
    for cache_name in cache_names:
        print(f"cache {scan_mode:>6} {cache_name}", flush=True)
        result, sample = _scan_one_cache(
            cache_name, scan_mode, sample_rows, chunk_rows
        )
        manifest_path = ARTIFACT_DIRECTORY / f"{cache_name}.manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifests[cache_name] = manifest
        for index, name in enumerate(manifest["feature_names"]):
            references.append(
                {
                    "global_index": offset + index,
                    "cache": cache_name,
                    "index": index,
                    "name": name,
                }
            )
        offset += sample.shape[1]
        cache_results.append(result)
        samples.append(sample)

    combined = np.column_stack(samples)
    sample_minimum = np.min(combined, axis=0)
    sample_maximum = np.max(combined, axis=0)
    variable = sample_minimum != sample_maximum
    fingerprints: dict[str, list[int]] = defaultdict(list)
    for index in np.flatnonzero(variable):
        fingerprints[_hash_column(combined[:, index])].append(int(index))

    exact_groups: list[dict[str, Any]] = []
    sample_only_groups: list[dict[str, Any]] = []
    for indices in fingerprints.values():
        if len(indices) < 2:
            continue
        reference = combined[:, indices[0]]
        equal = [index for index in indices if np.array_equal(reference, combined[:, index])]
        if len(equal) < 2:
            continue
        group = [references[index] for index in equal]
        verified = (
            _verify_exact_group(group, manifests, chunk_rows)
            if scan_mode == "full"
            else None
        )
        candidate = {
            "columns": group,
            "sample_equal": True,
            "all_rows_equal": verified,
        }
        if scan_mode == "full" and verified is not True:
            sample_only_groups.append(candidate)
        else:
            exact_groups.append(candidate)

    exact_pairs = {
        tuple(sorted((left, right)))
        for group in exact_groups
        for left in [item["global_index"] for item in group["columns"]]
        for right in [item["global_index"] for item in group["columns"]]
        if left < right
    }
    near_pairs: list[dict[str, Any]] = []
    cache_offset = 0
    for result, sample in zip(cache_results, samples):
        slices = pipeline.bank_slices(tuple(result["banks"]))
        for bank, bank_slice in slices.items():
            values = sample[: min(near_rows, sample.shape[0]), bank_slice].astype(
                np.float64
            )
            centered = values - np.mean(values, axis=0)
            scale = np.sqrt(np.mean(centered * centered, axis=0))
            valid = np.flatnonzero(scale > 1e-12)
            if valid.size < 2:
                continue
            standardized = centered[:, valid] / scale[valid]
            correlation = np.clip(
                standardized.T @ standardized / standardized.shape[0],
                -1.0,
                1.0,
            )
            rows, columns = np.where(np.triu(np.abs(correlation), 1) >= near_threshold)
            for left_local, right_local in zip(rows, columns):
                left = cache_offset + bank_slice.start + int(valid[left_local])
                right = cache_offset + bank_slice.start + int(valid[right_local])
                if tuple(sorted((left, right))) in exact_pairs:
                    continue
                near_pairs.append(
                    {
                        "bank": bank,
                        "left": references[left],
                        "right": references[right],
                        "correlation": float(correlation[left_local, right_local]),
                    }
                )
        cache_offset += sample.shape[1]
    near_pairs.sort(key=lambda item: abs(item["correlation"]), reverse=True)

    selected = _selected_feature_sets()
    for result in cache_results:
        for column in result["constant_columns"]:
            column["selected_in"] = [
                label for label, names in selected.items() if column["name"] in names
            ]
    for group in exact_groups:
        for column in group["columns"]:
            column["selected_in"] = [
                label for label, names in selected.items() if column["name"] in names
            ]

    signatures = {
        "core": pipeline._feature_signature(pipeline.CORE_BANKS),
        "candidate": pipeline._feature_signature(pipeline.CANDIDATE_BANKS),
        "novel": pipeline._feature_signature(pipeline.NOVEL_BANKS),
        "specialist": pipeline._feature_signature(pipeline.SPECIALIST_BANKS),
    }
    return {
        "mode": scan_mode,
        "sample_rows": int(combined.shape[0]),
        "near_rows": min(int(near_rows), int(combined.shape[0])),
        "near_threshold": near_threshold,
        "near_scope": "within each detector bank",
        "caches": cache_results,
        "exact_duplicate_groups": exact_groups,
        "sample_only_duplicate_candidates": sample_only_groups,
        "near_duplicate_pair_count": len(near_pairs),
        "near_duplicate_pairs": near_pairs[:200],
        "near_duplicate_pairs_truncated": len(near_pairs) > 200,
        "production_signatures": signatures,
        "expected_production_signatures": EXPECTED_PRODUCTION_SIGNATURES,
        "production_signatures_unchanged": all(
            signatures[name] == expected
            for name, expected in EXPECTED_PRODUCTION_SIGNATURES.items()
        ),
    }


def _role(detector_id: str) -> str:
    import feature_pipeline as pipeline

    bank = detector_id
    if bank in pipeline.CORE_BANKS:
        return "production core"
    if bank in pipeline.CANDIDATE_BANKS:
        return "production candidate"
    if bank in pipeline.SPECIALIST_BANKS:
        return "specialist"
    if bank in pipeline.NOVEL_BANKS:
        return "novel/rejected"
    return "foundation/ablation"


def _markdown(result: dict[str, Any]) -> str:
    inventory = result["inventory"]
    self_tests = result["self_tests"]
    cache = result.get("cache_audit")
    lines = [
        "# Detector audit",
        "",
        f"Generated: `{result['generated_at']}` by `python3 detector/audit_detectors.py`.",
        "",
        f"Overall result: **{result['status'].upper()}**.",
        "",
        "## Correctness fixes",
        "",
        "- Restored `detector/core/channels18.py`, the missing history-normalized "
        "d3 innovation helper required by d19 and d20.",
        "- Replaced d16's invalid positional d4 prefix slice with an explicit "
        "feature-name mapping. d4 inserts `x99_up`/`x99_dn` at column 90, so "
        "the old slice fed the wrong channels to d16.",
        "- Versioned only d16's specialist implementation. Core, candidate, and "
        "novel production signatures remain unchanged.",
        "- Exact-constant history exposes floating-point cancellation differences "
        "between some old vectorized and recurrent accumulators. Outputs remain "
        "finite, deterministic, and causal. This edge case is documented rather "
        "than changing accepted production semantics; none of the 8,000 training "
        "histories is near-constant.",
        "",
        "## Verification summary",
        "",
        f"- Detector sources discovered: **{len(inventory)}** "
        f"({sum(item['group'] == 'core' for item in inventory)} core, "
        f"{sum(item['group'] == 'candidate' for item in inventory)} candidate).",
        f"- Compile/import issues: **{sum(bool(item['issues']) for item in inventory)}**.",
        f"- Local module self-tests passed: **{sum(test['status'] == 'pass' for test in self_tests.values())}**; "
        f"central-only modules: **{sum(test['status'] == 'covered_by_central_suite' for test in self_tests.values())}**.",
        f"- Project regression tests: **{result['regression_suite'].get('test_count')} passed**.",
        f"- Composite synthetic cases: **{sum(case['status'] == 'pass' for case in result['synthetic_checks']['cases'])}/"
        f"{len(result['synthetic_checks']['cases'])} passed**; instance isolation "
        f"**{sum(item['status'] == 'pass' for item in result['synthetic_checks']['instance_isolation'].values())}/"
        f"{len(result['synthetic_checks']['instance_isolation'])} passed**.",
        f"- Minimum train-history standard deviation: "
        f"**{result['training_history_profile']['minimum_standard_deviation']:.8g}**; "
        f"near-constant histories: "
        f"**{result['training_history_profile']['standard_deviation_at_most_1e_8']}**.",
        "",
        "## Detector inventory",
        "",
        "| Detector | Folder | Role | Features | Direct detector dependencies | d3 prefix | Self-test |",
        "|---|---|---|---:|---|---|---|",
    ]
    for item in inventory:
        count = item.get("feature_count")
        contract = item.get("feature_name_contract", "unknown")
        if count is None:
            count_text = contract
        elif contract == "fixed":
            count_text = str(count)
        else:
            count_text = f"{count} in audit probe; {contract}"
        dependencies = ", ".join(item.get("dependencies", ())) or "none"
        inherited = "yes" if item.get("inherits_complete_d3_prefix") else "no"
        status = self_tests[item["path"]]["status"]
        lines.append(
            f"| {item['id']} | {item['group']} | {_role(item['id'])} | "
            f"{count_text} | {dependencies} | {inherited} | {status} |"
        )

    inherited = [item["id"] for item in inventory if item.get("inherits_complete_d3_prefix")]
    lines.extend(
        [
            "",
            "## Repeated detector logic",
            "",
            "A detector file can intentionally emit many features from one core "
            "innovation. Prefix, upward/downward CUSUM, fixed windows, EWMAs, and "
            "absolute-value summaries are different time geometries, not accidental "
            "copies. The following standalone experiments prepend the complete d3 "
            f"feature block: `{', '.join(inherited)}`.",
            "",
            "The active composite contract avoids that repeated base block: d10, d11, "
            "d15, and d16 are sliced to retain only their additions, while d3's shared "
            "statistics come from d4. Feature names are bank-prefixed and no active "
            "contract contains duplicate names.",
        ]
    )

    if cache:
        constants = [
            column
            for cache_item in cache["caches"]
            for column in cache_item["constant_columns"]
        ]
        lines.extend(
            [
                "",
                "## Numeric redundancy in cached training features",
                "",
                f"Cache scan mode: `{cache['mode']}`; sampled rows: "
                f"**{cache['sample_rows']}**; near-duplicate threshold: "
                f"**|r| >= {cache['near_threshold']}** "
                f"({cache['near_scope']}).",
                "",
                f"- Constant columns: **{len(constants)}**.",
                f"- Exact non-constant duplicate groups: "
                f"**{len(cache['exact_duplicate_groups'])}**.",
                f"- Near-duplicate pairs excluding exact duplicates: "
                f"**{cache['near_duplicate_pair_count']}**.",
                "",
                "Constants (retained for contract compatibility):",
                "",
            ]
        )
        if constants:
            lines.extend(f"- `{column['name']}` = {column['value']:g}" for column in constants)
        else:
            lines.append("- None.")
        lines.extend(["", "Exact non-constant groups:", ""])
        if cache["exact_duplicate_groups"]:
            for group in cache["exact_duplicate_groups"]:
                names = " = ".join(f"`{item['name']}`" for item in group["columns"])
                scope = (
                    "verified over all cached rows"
                    if group["all_rows_equal"] is True
                    else "equal on the audit sample"
                )
                lines.append(f"- {names} ({scope}).")
        else:
            lines.append("- None.")
        lines.extend(["", "Near-duplicate pairs (sample correlation):", ""])
        if cache["near_duplicate_pairs"]:
            for pair in cache["near_duplicate_pairs"]:
                lines.append(
                    f"- `{pair['left']['name']}` vs `{pair['right']['name']}`: "
                    f"r = {pair['correlation']:.8f}."
                )
        else:
            lines.append("- None.")
        selected_redundant = [
            column["name"]
            for column in constants
            if column.get("selected_in")
        ] + [
            column["name"]
            for group in cache["exact_duplicate_groups"]
            for column in group["columns"]
            if column.get("selected_in")
        ]
        lines.extend(
            [
                "",
                "These columns are reported rather than removed. None of the "
                f"redundant columns is selected by the audited submission-3 or "
                f"joint-768 contracts: **{not selected_redundant}**.",
                "",
                "## Production contract hashes",
                "",
                "| Contract | Features | SHA-256 | Unchanged |",
                "|---|---:|---|---|",
            ]
        )
        import feature_pipeline as pipeline

        counts = {
            "core": len(pipeline.feature_names(pipeline.CORE_BANKS)),
            "candidate": len(pipeline.feature_names(pipeline.CANDIDATE_BANKS)),
            "novel": len(pipeline.feature_names(pipeline.NOVEL_BANKS)),
        }
        for name in ("core", "candidate", "novel"):
            signature = cache["production_signatures"][name]
            unchanged = signature == EXPECTED_PRODUCTION_SIGNATURES[name]
            lines.append(
                f"| {name} | {counts[name]} | `{signature}` | {unchanged} |"
            )

    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            "python3 detector/audit_detectors.py --cache-scan full",
            "python3 tests/run_tests.py",
            "```",
            "",
            "The audit checks implementation correctness, causality, determinism, "
            "and redundancy. It does not claim that every retained experimental "
            "detector improves TSAUC; model selection remains a separate leakage-free "
            "validation task.",
            "",
        ]
    )
    return "\n".join(lines)


def _overall_status(result: dict[str, Any]) -> str:
    failures = []
    failures.extend(item for item in result["inventory"] if item["issues"])
    failures.extend(
        item for item in result["self_tests"].values() if item["status"] in {"fail", "timeout"}
    )
    if result["regression_suite"]["status"] != "pass":
        failures.append(result["regression_suite"])
    failures.extend(
        case for case in result["synthetic_checks"]["cases"] if case["status"] != "pass"
    )
    failures.extend(
        item
        for item in result["synthetic_checks"]["instance_isolation"].values()
        if item["status"] != "pass"
    )
    cache = result.get("cache_audit")
    if cache:
        failures.extend(item for item in cache["caches"] if not item["size_valid"])
        failures.extend(item for item in cache["caches"] if not item["all_finite"])
        if not cache["production_signatures_unchanged"]:
            failures.append("production feature signature changed")
        if cache["mode"] == "full":
            failures.extend(
                group
                for group in cache["exact_duplicate_groups"]
                if group["all_rows_equal"] is not True
            )
    return "pass" if not failures else "fail"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-scan",
        choices=("none", "sample", "full"),
        default="sample",
        help="numeric cache audit depth (default: sample)",
    )
    parser.add_argument("--sample-rows", type=int, default=8192)
    parser.add_argument("--near-rows", type=int, default=2048)
    parser.add_argument("--near-threshold", type=float, default=0.9999)
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--self-test-timeout", type=float, default=180.0)
    parser.add_argument("--suite-timeout", type=float, default=300.0)
    parser.add_argument("--skip-self-tests", action="store_true")
    parser.add_argument("--skip-regression-suite", action="store_true")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.sample_rows < 2 or args.near_rows < 2 or args.chunk_rows < 1:
        raise ValueError("sample rows, near rows, and chunk rows must be positive")
    if not 0.0 < args.near_threshold <= 1.0:
        raise ValueError("near threshold must be in (0, 1]")
    _configure_imports()
    d3 = importlib.import_module("detectors3")
    files = detector_files()
    inventory = [inspect_module(path, d3.feature_names()) for path in files]
    print(f"inventory {len(inventory)} detector modules", flush=True)

    if args.skip_self_tests:
        self_tests = {
            item["path"]: {
                "status": "not_run",
                "return_code": None,
                "seconds": 0.0,
                "output_tail": "disabled by --skip-self-tests",
            }
            for item in inventory
        }
    else:
        self_tests = run_self_tests(files, inventory, args.jobs, args.self_test_timeout)

    regression = (
        {
            "status": "not_run",
            "return_code": None,
            "test_count": None,
            "seconds": 0.0,
            "output_tail": "disabled by --skip-regression-suite",
        }
        if args.skip_regression_suite
        else run_regression_suite(args.suite_timeout)
    )
    synthetic = run_synthetic_composite_checks()
    history_profile = training_history_profile()
    result: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": sys.version,
        "inventory": inventory,
        "self_tests": self_tests,
        "regression_suite": regression,
        "synthetic_checks": synthetic,
        "training_history_profile": history_profile,
    }
    if args.cache_scan != "none":
        result["cache_audit"] = scan_feature_caches(
            args.cache_scan,
            args.sample_rows,
            args.near_rows,
            args.near_threshold,
            args.chunk_rows,
        )
    result["status"] = _overall_status(result)

    json_path = args.output_json.resolve()
    markdown_path = args.output_markdown.resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(result), encoding="utf-8")
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {markdown_path}", flush=True)
    print(f"AUDIT {result['status'].upper()}", flush=True)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
