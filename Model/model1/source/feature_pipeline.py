"""Causal feature composition and an on-disk feature cache.

The ``detector/`` directory contains many experiments. This module defines
explicit, versioned feature contracts instead of silently importing all of
them. ``CORE_BANKS`` is the primary family, ``CANDIDATE_BANKS`` is the
independently trained complementary family, and ``SPECIALIST_BANKS`` contains
ablation candidates retained for reproducibility. The core contract contains:

* d4: calibrated location, scale, shape, dependence, AR residual, extreme
  tail, and multi-lag difference-energy scans;
* d15: finite-start-corrected EWMA copies of the proven d3 innovations;
* d22/d26: empirical historical-null calibration of suffix and EWMA scores;
* d27/d30: the same two calibrations for the complementary d4 energy/tail
  channels.

Every online feature uses only the clean history and the prefix ending at the
current observation.  ``CoreStreamDetector`` is checked against the vectorized
offline implementation in ``tests/test_feature_pipeline.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from competition_data import TrainingData


PROJECT_DIRECTORY = Path(__file__).resolve().parent
DETECTOR_DIRECTORY = PROJECT_DIRECTORY / "detector"
if str(DETECTOR_DIRECTORY) not in sys.path:
    # Detector modules intentionally use sibling imports such as
    # ``import detectors3``; putting the directory itself on sys.path keeps
    # those modules unchanged and works both locally and in Crunch's runner.
    sys.path.insert(0, str(DETECTOR_DIRECTORY))

import detectors3 as d3  # noqa: E402
import detectors4 as d4  # noqa: E402
import detectors15 as d15  # noqa: E402
import detectors22 as d22  # noqa: E402
import detectors26 as d26  # noqa: E402
import detectors27 as d27  # noqa: E402
import detectors30 as d30  # noqa: E402
import detectors10 as d10  # noqa: E402
import detectors11 as d11  # noqa: E402
import detectors12 as d12  # noqa: E402
import detectors16 as d16  # noqa: E402
import detectors21 as d21  # noqa: E402
import detectors23 as d23  # noqa: E402
import detectors24 as d24  # noqa: E402
import detectors25 as d25  # noqa: E402
import detectors28 as d28  # noqa: E402
import detectors29 as d29  # noqa: E402
import detectors32 as d32  # noqa: E402
import detectors33 as d33  # noqa: E402
import detectors35 as d35  # noqa: E402
import detectors37 as d37  # noqa: E402
import detectors40 as d40  # noqa: E402
import detectors45 as d45  # noqa: E402
import detectors51 as d51  # noqa: E402
import detectors52 as d52  # noqa: E402
import detectors53 as d53  # noqa: E402
import detectors55 as d55  # noqa: E402
import detectors57 as d57  # noqa: E402
import detectors58 as d58  # noqa: E402
import detectors36 as d36  # noqa: E402


FEATURE_VERSION = "core-v1"
CORE_BANKS = ("d4", "d15", "d22", "d26", "d27", "d30")
CANDIDATE_BANKS = (
    "d32",
    "d33",
    "d35",
    "d37",
    "d40",
    "d45",
    "d51",
    "d52",
    "d53",
    "d55",
    "d57",
    "d58",
)
SPECIALIST_BANKS = (
    "d10",
    "d11",
    "d12",
    "d16",
    "d21",
    "d23",
    "d24",
    "d25",
    "d28",
    "d29",
    "d36",
)
ALL_BANKS = CORE_BANKS + CANDIDATE_BANKS + SPECIALIST_BANKS
CLIP_ABS = 1_000_000.0


def _bank_feature_names() -> dict[str, list[str]]:
    return {
        "d4": [f"d4__{name}" for name in d4.feature_names()],
        "d15": [
            f"d15__{name}" for name in d15.feature_names()[d3.N_FEATURES :]
        ],
        "d22": [f"d22__{name}" for name in d22.feature_names()],
        "d26": [f"d26__{name}" for name in d26.feature_names()],
        "d27": [f"d27__{name}" for name in d27.feature_names()],
        "d30": [f"d30__{name}" for name in d30.feature_names()],
        # d10/d11/d16 return the d3 base followed by their additions.  The
        # d3 base is already contained in d4, so store only non-duplicate
        # columns in this bank contract.
        "d10": [f"d10__{name}" for name in d10.feature_names()[d3.N_FEATURES :]],
        "d11": [f"d11__{name}" for name in d11.feature_names()[d3.N_FEATURES :]],
        "d12": [f"d12__{name}" for name in d12.feature_names()],
        "d16": [f"d16__{name}" for name in d16.feature_names()[d3.N_FEATURES :]],
        "d21": [f"d21__{name}" for name in d21.feature_names()],
        "d23": [f"d23__{name}" for name in d23.feature_names()],
        "d24": [f"d24__{name}" for name in d24.feature_names()],
        "d25": [f"d25__{name}" for name in d25.feature_names()],
        "d28": [f"d28__{name}" for name in d28.feature_names()],
        "d29": [f"d29__{name}" for name in d29.feature_names()],
        "d36": [f"d36__{name}" for name in d36.feature_names()],
        "d32": [f"d32__{name}" for name in d32.feature_names()],
        "d33": [f"d33__{name}" for name in d33.feature_names()],
        "d35": [f"d35__{name}" for name in d35.feature_names()],
        "d37": [f"d37__{name}" for name in d37.feature_names()],
        "d40": [f"d40__{name}" for name in d40.feature_names()],
        "d45": [f"d45__{name}" for name in d45.feature_names()],
        "d51": [f"d51__{name}" for name in d51.feature_names()],
        "d52": [f"d52__{name}" for name in d52.feature_names()],
        "d53": [f"d53__{name}" for name in d53.feature_names()],
        "d55": [f"d55__{name}" for name in d55.feature_names()],
        "d57": [f"d57__{name}" for name in d57.feature_names()],
        "d58": [f"d58__{name}" for name in d58.feature_names()],
    }


BANK_FEATURE_NAMES = _bank_feature_names()


def feature_names(banks: Sequence[str] = CORE_BANKS) -> list[str]:
    """Names, in exactly the order emitted by offline/streaming features."""

    unknown = set(banks) - set(BANK_FEATURE_NAMES)
    if unknown:
        raise ValueError(f"unknown feature banks: {sorted(unknown)}")
    return [name for bank in banks for name in BANK_FEATURE_NAMES[bank]]


def bank_slices(banks: Sequence[str] = CORE_BANKS) -> dict[str, slice]:
    """Map bank name to its contiguous columns in the composite matrix."""

    output: dict[str, slice] = {}
    left = 0
    for bank in banks:
        right = left + len(BANK_FEATURE_NAMES[bank])
        output[bank] = slice(left, right)
        left = right
    return output


def sanitize_features(matrix: np.ndarray) -> np.ndarray:
    """Replace numerical pathologies deterministically and clip extremes."""

    values = np.asarray(matrix)
    np.nan_to_num(
        values,
        copy=False,
        nan=0.0,
        posinf=CLIP_ABS,
        neginf=-CLIP_ABS,
    )
    np.clip(values, -CLIP_ABS, CLIP_ABS, out=values)
    return values


def compute_features_offline(
    historical: np.ndarray,
    online: np.ndarray,
    banks: Sequence[str] = CORE_BANKS,
) -> np.ndarray:
    """Vectorized causal features for every prefix of one online segment."""

    history = np.asarray(historical, dtype=np.float64)
    stream = np.asarray(online, dtype=np.float64)
    blocks: dict[str, np.ndarray] = {}
    for bank in banks:
        if bank == "d4":
            blocks[bank] = d4.compute_features_offline(history, stream)
        elif bank == "d15":
            blocks[bank] = d15.extra_features(history, stream)
        elif bank == "d22":
            blocks[bank] = d22.extra_features(history, stream)
        elif bank == "d26":
            blocks[bank] = d26.extra_features(history, stream)
        elif bank == "d27":
            blocks[bank] = d27.extra_features(history, stream)
        elif bank == "d30":
            blocks[bank] = d30.extra_features(history, stream)
        elif bank == "d10":
            blocks[bank] = d10.compute_features_offline(history, stream)[
                :, d3.N_FEATURES :
            ]
        elif bank == "d11":
            blocks[bank] = d11.compute_features_offline(history, stream)[
                :, d3.N_FEATURES :
            ]
        elif bank == "d12":
            blocks[bank] = d12.compute_features_offline(history, stream)
        elif bank == "d16":
            # d4 starts with the exact d3 layout.
            base = blocks.get("d4")
            if base is None:
                base = d4.compute_features_offline(history, stream)
            blocks[bank] = d16.extra_from_base(base[:, : d3.N_FEATURES])
        elif bank == "d21":
            blocks[bank] = d21.extra_features(history, stream)
        elif bank == "d23":
            blocks[bank] = d23.compute_features_offline(history, stream)
        elif bank == "d24":
            blocks[bank] = d24.compute_features_offline(history, stream)
        elif bank == "d25":
            blocks[bank] = d25.compute_features_offline(history, stream)
        elif bank == "d28":
            blocks[bank] = d28.extra_features(history, stream)
        elif bank == "d29":
            blocks[bank] = d29.extra_features(history, stream)
        elif bank == "d36":
            blocks[bank] = d36.extra_features(history, stream)
        elif bank == "d32":
            blocks[bank] = d32.compute_features_offline(history, stream)
        elif bank == "d33":
            blocks[bank] = d33.compute_features_offline(history, stream)
        elif bank == "d35":
            blocks[bank] = d35.compute_features_offline(history, stream)
        elif bank == "d37":
            blocks[bank] = d37.compute_features_offline(history, stream)
        elif bank == "d40":
            blocks[bank] = d40.compute_features_offline(history, stream)
        elif bank == "d45":
            blocks[bank] = d45.extra_features(history, stream)
        elif bank == "d51":
            blocks[bank] = d51.extra_features(history, stream)
        elif bank == "d52":
            blocks[bank] = d52.extra_features(history, stream)
        elif bank == "d53":
            blocks[bank] = d53.extra_features(history, stream)
        elif bank == "d55":
            blocks[bank] = d55.extra_features(history, stream)
        elif bank == "d57":
            blocks[bank] = d57.extra_features(history, stream)
        elif bank == "d58":
            blocks[bank] = d58.extra_features(history, stream)
    result = np.column_stack([blocks[bank] for bank in banks])
    if result.shape != (stream.size, len(feature_names(banks))):
        raise AssertionError(
            f"feature shape {result.shape} violates the declared contract"
        )
    return sanitize_features(result)


class CoreStreamDetector:
    """Incremental twin of :func:`compute_features_offline`.

    The historical class name is retained for API stability, but ``banks`` may
    select any declared family. Compatible transforms share one
    ``StreamDetector4`` state; independent specialist states are advanced once
    per observation. No state sees a future value or another series.
    """

    def __init__(
        self,
        historical: np.ndarray,
        banks: Sequence[str] = CORE_BANKS,
    ) -> None:
        self.banks = tuple(banks)
        unknown = set(self.banks) - set(ALL_BANKS)
        if unknown:
            raise ValueError(f"unsupported streaming banks: {sorted(unknown)}")
        history = np.asarray(historical, dtype=np.float64)
        self.inner = d4.StreamDetector4(history)
        self.transforms: dict[str, object] = {}
        if "d15" in self.banks:
            self.transforms["d15"] = d15.StreamDetector15(history, self.inner)
        if "d22" in self.banks:
            self.transforms["d22"] = d22.StreamDetector22(history, self.inner)
        if "d26" in self.banks:
            self.transforms["d26"] = d26.StreamDetector26(history, self.inner)
        if "d27" in self.banks:
            self.transforms["d27"] = d27.StreamDetector27(history, self.inner)
        if "d30" in self.banks:
            self.transforms["d30"] = d30.StreamDetector30(history, self.inner)
        if "d10" in self.banks:
            self.transforms["d10"] = d10.StreamDetector10(history)
        if "d11" in self.banks:
            self.transforms["d11"] = d11.StreamDetector11(history)
        if "d12" in self.banks:
            self.transforms["d12"] = d12.StreamDetector12(history)
        if "d16" in self.banks:
            # Stateless transform of the d3 prefix embedded in d4.
            self.transforms["d16"] = d16
        if "d21" in self.banks:
            self.transforms["d21"] = d21.StreamDetector21(history)
        if "d23" in self.banks:
            self.transforms["d23"] = d23.StreamDetector23(history)
        if "d24" in self.banks:
            self.transforms["d24"] = d24.StreamDetector24(history, self.inner)
        if "d25" in self.banks:
            self.transforms["d25"] = d25.StreamDetector25(history)
        if "d28" in self.banks:
            self.transforms["d28"] = d28.StreamDetector28(history)
        if "d29" in self.banks:
            self.transforms["d29"] = d29.StreamDetector29(history, self.inner)
        if "d36" in self.banks:
            self.transforms["d36"] = d36.StreamDetector36(history, self.inner)
        if "d32" in self.banks:
            self.transforms["d32"] = d32.StreamDetector32(history)
        if "d33" in self.banks:
            self.transforms["d33"] = d33.StreamDetector33(history)
        if "d35" in self.banks:
            self.transforms["d35"] = d35.StreamDetector35(history)
        if "d37" in self.banks:
            self.transforms["d37"] = d37.StreamDetector37(history, self.inner)
        if "d40" in self.banks:
            self.transforms["d40"] = d40.StreamDetector40(history)
        if "d45" in self.banks:
            self.transforms["d45"] = d45.StreamDetector45(history, self.inner)
        if "d51" in self.banks:
            self.transforms["d51"] = d51.StreamDetector51(history, self.inner)
        if "d52" in self.banks:
            self.transforms["d52"] = d52.StreamDetector52(history, self.inner)
        if "d53" in self.banks:
            self.transforms["d53"] = d53.StreamDetector53(history, self.inner)
        if "d55" in self.banks:
            self.transforms["d55"] = d55.StreamDetector55(history, self.inner)
        if "d57" in self.banks:
            self.transforms["d57"] = d57.StreamDetector57(history, self.inner)
        if "d58" in self.banks:
            self.transforms["d58"] = d58.StreamDetector58(history, self.inner)

    def step(self, value: float) -> np.ndarray:
        """Consume exactly one new value and return its feature vector."""

        base4 = self.inner.step(float(value))
        blocks: dict[str, np.ndarray] = {"d4": base4}
        if "d15" in self.transforms:
            # d15's public shared-state method returns [d3 base, d15 extra].
            combined = self.transforms["d15"].step_from_inner(  # type: ignore[attr-defined]
                np.zeros(d3.N_FEATURES, dtype=np.float64)
            )
            blocks["d15"] = combined[d3.N_FEATURES :]
        for bank in ("d22", "d26", "d27", "d30"):
            if bank in self.transforms:
                blocks[bank] = self.transforms[bank].step_from_inner()  # type: ignore[attr-defined]
        for bank in ("d10", "d11"):
            if bank in self.transforms:
                blocks[bank] = self.transforms[bank].step(value)[  # type: ignore[attr-defined]
                    d3.N_FEATURES :
                ]
        if "d12" in self.transforms:
            blocks["d12"] = self.transforms["d12"].step(value)  # type: ignore[attr-defined]
        if "d16" in self.transforms:
            blocks["d16"] = d16.extra_from_base(base4[: d3.N_FEATURES])
        for bank in ("d21", "d23", "d25", "d28"):
            if bank in self.transforms:
                blocks[bank] = self.transforms[bank].step(value)  # type: ignore[attr-defined]
        if "d24" in self.transforms:
            blocks["d24"] = self.transforms["d24"].step_from_inner(value)  # type: ignore[attr-defined]
        for bank in ("d29", "d36"):
            if bank in self.transforms:
                blocks[bank] = self.transforms[bank].step_from_inner()  # type: ignore[attr-defined]
        for bank in ("d32", "d33", "d35", "d40"):
            if bank in self.transforms:
                blocks[bank] = self.transforms[bank].step(value)  # type: ignore[attr-defined]
        for bank in ("d37", "d45", "d51", "d52", "d53", "d55", "d57", "d58"):
            if bank in self.transforms:
                blocks[bank] = self.transforms[bank].step_from_inner()  # type: ignore[attr-defined]
        output = np.concatenate([blocks[bank] for bank in self.banks])
        return sanitize_features(output)


@dataclass(frozen=True)
class FeatureCache:
    """Paths and metadata for a completed float32 memory-mapped cache."""

    matrix_path: Path
    metadata_path: Path
    manifest_path: Path
    rows: int
    columns: int
    banks: tuple[str, ...]

    def open(self, mode: str = "r") -> np.memmap:
        return np.memmap(
            self.matrix_path,
            dtype=np.float32,
            mode=mode,
            shape=(self.rows, self.columns),
        )

    def load_metadata(self) -> dict[str, np.ndarray]:
        with np.load(self.metadata_path) as values:
            return {name: values[name] for name in values.files}


def _feature_signature(banks: Sequence[str]) -> str:
    payload = json.dumps(
        {
            "version": FEATURE_VERSION,
            "banks": list(banks),
            "names": feature_names(banks),
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _cache_paths(directory: Path, name: str) -> tuple[Path, Path, Path, Path]:
    return (
        directory / f"{name}.f32.mmap",
        directory / f"{name}.metadata.npz",
        directory / f"{name}.manifest.json",
        directory / f"{name}.progress.json",
    )


def load_feature_cache(
    cache_directory: str | Path = "artifacts",
    name: str = "core_features",
) -> FeatureCache:
    """Open and validate a completed cache without loading its matrix."""

    directory = Path(cache_directory)
    matrix_path, metadata_path, manifest_path, _ = _cache_paths(directory, name)
    if not manifest_path.exists():
        raise FileNotFoundError(f"feature cache manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    banks = tuple(manifest["banks"])
    if manifest["feature_signature"] != _feature_signature(banks):
        raise ValueError("feature code/name signature differs from cached features")
    expected_bytes = int(manifest["rows"]) * int(manifest["columns"]) * 4
    if matrix_path.stat().st_size != expected_bytes:
        raise ValueError("feature cache byte size does not match its manifest")
    return FeatureCache(
        matrix_path=matrix_path,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        rows=int(manifest["rows"]),
        columns=int(manifest["columns"]),
        banks=banks,
    )


def build_feature_cache(
    data: TrainingData,
    cache_directory: str | Path = "artifacts",
    name: str = "core_features",
    banks: Sequence[str] = CORE_BANKS,
    progress_every: int = 25,
    workers: int = 1,
) -> FeatureCache:
    """Build or resume the full online-row feature cache.

    A progress sidecar is flushed periodically.  If extraction is interrupted,
    rerunning resumes at the next complete series rather than discarding a
    multi-gigabyte partial cache.
    """

    selected_banks = tuple(banks)
    directory = Path(cache_directory)
    directory.mkdir(parents=True, exist_ok=True)
    matrix_path, metadata_path, manifest_path, progress_path = _cache_paths(
        directory, name
    )
    if manifest_path.exists():
        cache = load_feature_cache(directory, name)
        if cache.banks != selected_banks:
            raise ValueError(
                f"cache {name} has banks {cache.banks}, requested {selected_banks}"
            )
        return cache

    rows = data.online_row_count
    columns = len(feature_names(selected_banks))
    partial_path = matrix_path.with_suffix(matrix_path.suffix + ".partial")
    signature = _feature_signature(selected_banks)

    next_series = 0
    next_row = 0
    if progress_path.exists() and partial_path.exists():
        progress = json.loads(progress_path.read_text())
        expected = (rows, columns, signature)
        actual = (
            int(progress["rows"]),
            int(progress["columns"]),
            progress["feature_signature"],
        )
        if actual != expected:
            raise ValueError("partial cache belongs to a different feature contract")
        next_series = int(progress["next_series"])
        next_row = int(progress["next_row"])

    mode = "r+" if partial_path.exists() else "w+"
    matrix = np.memmap(
        partial_path,
        dtype=np.float32,
        mode=mode,
        shape=(rows, columns),
    )
    if workers < 1:
        raise ValueError("workers must be positive")

    def extract(position: int) -> tuple[int, np.ndarray]:
        record = data.record(position)
        block = compute_features_offline(
            record.historical, record.online, selected_banks
        ).astype(np.float32, copy=False)
        return position, block

    positions = range(next_series, len(data))
    started = time.perf_counter()
    # Detector calibration is independent across series.  Threads avoid
    # copying the large read-only parquet arrays, while NumPy/SciPy release
    # the GIL for most of the expensive kernels.  executor.map preserves
    # series order, so the cache and its resumable row offset stay simple.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        blocks = executor.map(extract, positions, buffersize=2 * workers)
        for position, block in blocks:
            right = next_row + block.shape[0]
            matrix[next_row:right] = block
            next_row = right

            if (position + 1) % progress_every == 0 or position + 1 == len(data):
                matrix.flush()
                progress_path.write_text(
                    json.dumps(
                        {
                            "rows": rows,
                            "columns": columns,
                            "banks": selected_banks,
                            "feature_signature": signature,
                            "next_series": position + 1,
                            "next_row": next_row,
                        },
                        indent=2,
                    )
                )
                elapsed = time.perf_counter() - started
                rate = (position + 1 - next_series) / max(elapsed, 1e-9)
                print(
                    f"features: {position + 1:4d}/{len(data)} series, "
                    f"{next_row:,}/{rows:,} rows, {rate:.2f} series/s",
                    flush=True,
                )

    if next_row != rows:
        raise AssertionError(f"wrote {next_row} rows, expected {rows}")
    matrix.flush()
    del matrix

    row_id = np.repeat(data.series_ids, data.online_lengths).astype(np.int32)
    series_position = np.repeat(
        np.arange(len(data), dtype=np.int32), data.online_lengths
    )
    time_online = np.concatenate(
        [np.arange(length, dtype=np.int16) for length in data.online_lengths]
    )
    target = np.concatenate([record.target for record in data]).astype(np.uint8)
    np.savez_compressed(
        metadata_path,
        row_id=row_id,
        series_position=series_position,
        time_online=time_online,
        target=target,
        series_ids=data.series_ids.astype(np.int32),
        online_lengths=data.online_lengths.astype(np.int16),
        historical_lengths=data.historical_lengths.astype(np.int16),
        tau_index=data.tau_index.astype(np.int16),
    )

    os.replace(partial_path, matrix_path)
    manifest_path.write_text(
        json.dumps(
            {
                "feature_version": FEATURE_VERSION,
                "feature_signature": signature,
                "banks": selected_banks,
                "rows": rows,
                "columns": columns,
                "dtype": "float32",
                "feature_names": feature_names(selected_banks),
            },
            indent=2,
        )
    )
    progress_path.unlink(missing_ok=True)
    return load_feature_cache(directory, name)
