"""Causal feature composition and an on-disk feature cache.

The ``detector/`` directory contains many experiments. This module defines
explicit, versioned feature contracts instead of silently importing all of
them. ``CORE_BANKS`` is the primary family, ``CANDIDATE_BANKS`` is the
independently trained complementary family, and ``SPECIALIST_BANKS`` contains
ablation candidates retained for reproducibility. ``ALL_*_DETECTOR_BANKS``
adds every archived source for Model 4 without changing the accepted
production signatures. The core contract contains:

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
CORE_DETECTOR_DIRECTORY = DETECTOR_DIRECTORY / "core"
CANDIDATE_DETECTOR_DIRECTORY = DETECTOR_DIRECTORY / "candidate"
# Detector modules intentionally use sibling imports such as
# ``import detectors3`` and candidate modules may reuse core states. Expose the
# two explicit locations while retaining the historical import names. This
# works both locally and after the same directory tree is copied into Crunch.
for directory in (
    CANDIDATE_DETECTOR_DIRECTORY,
    CORE_DETECTOR_DIRECTORY,
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import detectors as d1  # noqa: E402
import detectors2 as d2  # noqa: E402
import detectors3 as d3  # noqa: E402
import detectors4 as d4  # noqa: E402
import detectors5 as d5  # noqa: E402
import detectors6 as d6  # noqa: E402
import detectors7 as d7  # noqa: E402
import detectors8 as d8  # noqa: E402
import detectors9 as d9  # noqa: E402
import detectors15 as d15  # noqa: E402
import detectors13 as d13  # noqa: E402
import detectors17 as d17  # noqa: E402
import detectors19 as d19  # noqa: E402
import detectors20 as d20  # noqa: E402
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
import detectors31 as d31  # noqa: E402
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
import detectors60 as d60  # noqa: E402
import detectors61 as d61  # noqa: E402
import detectors62 as d62  # noqa: E402
import detectors63 as d63  # noqa: E402
import detectors64 as d64  # noqa: E402
import detectors65 as d65  # noqa: E402
import detectors66 as d66  # noqa: E402


FEATURE_VERSION = "core-v1"
# Logic revisions are included only for the affected banks.  This invalidates
# a stale specialist cache after a correctness fix without changing the
# accepted core, candidate, or novel feature signatures.
BANK_IMPLEMENTATION_VERSIONS = {
    "d16": "name-mapped-d3-v2",
    # The archived learned detectors did not retain fitted artifacts. Model 4
    # therefore uses fixed analytic bases, never labels or pretrained files.
    "d19": "fixed-identity-aspect-basis-v1",
    "d20": "fixed-analytic-emission-v1",
    "d31": "fixed-eight-dct-bin-basis-v1",
    "d61": "history-grid-rank-tests-v1",
    "d62": "causal-piecewise-scan-v1",
    "d63": "symbolic-nonlinear-v1",
    "d64": "rank-trend-dependence-v1",
    "d65": "fixed-multiwindow-aggregation-v1",
    "d66": "iqr-symlet2-v1",
}
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
# ``d60`` is kept separate so its incremental value can be measured and the
# accepted core/candidate contracts remain byte-for-byte reproducible.
NOVEL_BANKS = ("d60",)
MODEL8_BANKS = ("d61", "d62", "d63", "d64", "d65", "d66")
ALL_BANKS = (
    CORE_BANKS + CANDIDATE_BANKS + SPECIALIST_BANKS + NOVEL_BANKS + MODEL8_BANKS
)

# These sources are retained in detector/core and detector/candidate but were
# not part of Submission 3's production caches. Descendants that prepend a
# complete d3 block contribute only their additions here; d3 itself is a bank.
# This preserves every detector experiment without storing the same 236 base
# columns many times.
ARCHIVE_CORE_BANKS = (
    "d1",
    "d2",
    "d3",
    "d5",
    "d6",
    "d7",
    "d8",
    "d9",
    "d13",
    "d17",
    "d19",
    "d20",
)
ARCHIVE_CANDIDATE_BANKS = ("d31",)
CORE_SPECIALIST_BANKS = tuple(bank for bank in SPECIALIST_BANKS if bank != "d36")
MODEL4_CORE_DETECTOR_BANKS = (
    "d1",
    "d2",
    "d3",
    "d4",
    "d5",
    "d6",
    "d7",
    "d8",
    "d9",
    "d10",
    "d11",
    "d12",
    "d13",
    "d15",
    "d16",
    "d17",
    "d19",
    "d20",
    "d21",
    "d22",
    "d23",
    "d24",
    "d25",
    "d26",
    "d27",
    "d28",
    "d29",
    "d30",
)
MODEL4_CANDIDATE_DETECTOR_BANKS = (
    "d31",
    "d32",
    "d33",
    "d35",
    "d36",
    "d37",
    "d40",
    "d45",
    "d51",
    "d52",
    "d53",
    "d55",
    "d57",
    "d58",
    "d60",
)
ALL_CORE_DETECTOR_BANKS = MODEL4_CORE_DETECTOR_BANKS
ALL_CANDIDATE_DETECTOR_BANKS = MODEL4_CANDIDATE_DETECTOR_BANKS + MODEL8_BANKS
ALL_DETECTOR_BANKS = ALL_CORE_DETECTOR_BANKS + ALL_CANDIDATE_DETECTOR_BANKS
if len(ALL_CORE_DETECTOR_BANKS) != 28 or len(ALL_CANDIDATE_DETECTOR_BANKS) != 21:
    raise AssertionError("the all-detector registry must contain 28 core and 21 candidate banks")
if set(ALL_CORE_DETECTOR_BANKS) != set(
    CORE_BANKS + CORE_SPECIALIST_BANKS + ARCHIVE_CORE_BANKS
):
    raise AssertionError("core detector registry does not cover its source groups")
if set(ALL_CANDIDATE_DETECTOR_BANKS) != set(
    CANDIDATE_BANKS
    + ("d36",)
    + NOVEL_BANKS
    + MODEL8_BANKS
    + ARCHIVE_CANDIDATE_BANKS
):
    raise AssertionError("candidate detector registry does not cover its source groups")
CLIP_ABS = 1_000_000.0


def _source_columns(source_names: Sequence[str], required_names: Sequence[str]):
    """Locate a named feature contract inside a larger named contract.

    Positional prefix reuse is unsafe: d4 inserts ``x99_up`` and ``x99_dn``
    before d3's quantile aspects.  Resolve every d3 feature explicitly and
    fail during import if a future detector revision removes or duplicates it.
    """

    positions: dict[str, list[int]] = {}
    for index, name in enumerate(source_names):
        positions.setdefault(name, []).append(index)
    missing = [name for name in required_names if name not in positions]
    ambiguous = [name for name in required_names if len(positions.get(name, ())) != 1]
    if missing or ambiguous:
        raise ValueError(
            "cannot map feature contracts; "
            f"missing={missing[:5]}, ambiguous={ambiguous[:5]}"
        )
    return np.asarray([positions[name][0] for name in required_names], dtype=np.intp)


D3_COLUMNS_IN_D4 = _source_columns(d4.feature_names(), d3.feature_names())


# Fixed, label-free stand-ins for three archived APIs whose original learned
# template/model files were never retained. They make the source detectors
# executable in an honest train-only experiment without pretending that a
# missing pretrained artifact exists.
D19_FIXED_TEMPLATES = np.eye(len(d3.ASPECTS), dtype=np.float64)
_d31_position = np.arange(d31.N_BINS, dtype=np.float64) + 0.5
D31_FIXED_TEMPLATES = np.cos(
    np.pi
    * np.arange(1, 9, dtype=np.float64)[:, None]
    * _d31_position[None, :]
    / float(d31.N_BINS)
)
D31_FIXED_TEMPLATES -= D31_FIXED_TEMPLATES.mean(axis=1, keepdims=True)
D31_FIXED_TEMPLATES /= np.maximum(
    np.linalg.norm(D31_FIXED_TEMPLATES, axis=1, keepdims=True), 1e-12
)


class _FixedEmission20:
    """Deterministic analytic margin satisfying d20's model interface."""

    def predict(self, matrix: np.ndarray, raw_score: bool = False) -> np.ndarray:
        del raw_score
        values = np.asarray(matrix, dtype=np.float64)
        return (
            0.31 * values[:, 0]
            - 0.17 * values[:, 1]
            + 0.13 * np.abs(values[:, 3])
            + 0.07 * values[:, 8] * values[:, 11]
        )


D20_FIXED_EMISSION = _FixedEmission20()


def _bank_feature_names() -> dict[str, list[str]]:
    return {
        "d1": [f"d1__{name}" for name in d1.feature_names()],
        "d2": [f"d2__{name}" for name in d2.feature_names()],
        "d3": [f"d3__{name}" for name in d3.feature_names()],
        "d4": [f"d4__{name}" for name in d4.feature_names()],
        "d5": [f"d5__{name}" for name in d5.feature_names()[d3.N_FEATURES :]],
        "d6": [f"d6__{name}" for name in d6.feature_names()[d3.N_FEATURES :]],
        "d7": [f"d7__{name}" for name in d7.feature_names()[d6.N_FEATURES :]],
        "d8": [f"d8__{name}" for name in d8.feature_names()[d3.N_FEATURES :]],
        "d9": [f"d9__{name}" for name in d9.feature_names()[d3.N_FEATURES :]],
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
        "d13": [f"d13__{name}" for name in d13.feature_names()[d3.N_FEATURES :]],
        "d16": [f"d16__{name}" for name in d16.feature_names()[d3.N_FEATURES :]],
        "d17": [f"d17__{name}" for name in d17.feature_names()[d3.N_FEATURES :]],
        "d19": [
            f"d19__{name}"
            for name in d19.feature_names(D19_FIXED_TEMPLATES.shape[0])
        ],
        "d20": [f"d20__{name}" for name in d20.feature_names()],
        "d21": [f"d21__{name}" for name in d21.feature_names()],
        "d23": [f"d23__{name}" for name in d23.feature_names()],
        "d24": [f"d24__{name}" for name in d24.feature_names()],
        "d25": [f"d25__{name}" for name in d25.feature_names()],
        "d28": [f"d28__{name}" for name in d28.feature_names()],
        "d29": [f"d29__{name}" for name in d29.feature_names()],
        "d36": [f"d36__{name}" for name in d36.feature_names()],
        "d31": [
            f"d31__{name}"
            for name in d31.feature_names(D31_FIXED_TEMPLATES.shape[0])
        ],
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
        "d60": [f"d60__{name}" for name in d60.feature_names()],
        "d61": [f"d61__{name}" for name in d61.feature_names()],
        "d62": [f"d62__{name}" for name in d62.feature_names()],
        "d63": [f"d63__{name}" for name in d63.feature_names()],
        "d64": [f"d64__{name}" for name in d64.feature_names()],
        "d65": [f"d65__{name}" for name in d65.feature_names()],
        "d66": [f"d66__{name}" for name in d66.feature_names()],
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
        if bank == "d1":
            blocks[bank] = d1.compute_features_offline(history, stream)
        elif bank == "d2":
            blocks[bank] = d2.compute_features_offline(history, stream)
        elif bank == "d3":
            blocks[bank] = d3.compute_features_offline(history, stream)
        elif bank == "d4":
            blocks[bank] = d4.compute_features_offline(history, stream)
        elif bank == "d5":
            blocks[bank] = d5.compute_features_offline(history, stream)[
                :, d3.N_FEATURES :
            ]
        elif bank == "d6":
            blocks[bank] = d6.compute_features_offline(history, stream)[
                :, d3.N_FEATURES :
            ]
        elif bank == "d7":
            blocks[bank] = d7.compute_features_offline_fast(history, stream)[
                :, d6.N_FEATURES :
            ]
        elif bank == "d8":
            blocks[bank] = d8.compute_features_offline(history, stream)[
                :, d3.N_FEATURES :
            ]
        elif bank == "d9":
            base3 = blocks.get("d3")
            if base3 is None:
                base3 = d3.compute_features_offline(history, stream)
            blocks[bank] = d9.augment_base_features(history, base3)[
                :, d3.N_FEATURES :
            ]
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
        elif bank == "d13":
            base3 = blocks.get("d3")
            if base3 is None:
                base3 = d3.compute_features_offline(history, stream)
            blocks[bank] = d13.extra_features(history, base3)
        elif bank == "d16":
            # d4 contains the d3 contract but inserts extra aspects, so select
            # the shared columns by name rather than taking a positional prefix.
            base = blocks.get("d4")
            if base is None:
                base = d4.compute_features_offline(history, stream)
            blocks[bank] = d16.extra_from_base(base[:, D3_COLUMNS_IN_D4])
        elif bank == "d17":
            blocks[bank] = d17.extra_features(history, stream)
        elif bank in {"d19", "d20"}:
            # Both archived detectors transform the same causal innovation
            # channels. Share that expensive calculation when both are built.
            channels = blocks.get("__channels18")
            if channels is None:
                import channels18

                channels = channels18.compute_channels_offline(history, stream)
                blocks["__channels18"] = channels
            if bank == "d19":
                blocks[bank] = d19.compute_features_from_channels(
                    history, channels, D19_FIXED_TEMPLATES
                )
            else:
                blocks[bank] = d20.compute_features_from_channels(
                    history, channels, D20_FIXED_EMISSION
                )
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
        elif bank == "d31":
            blocks[bank] = d31.extra_features(
                history, stream, D31_FIXED_TEMPLATES
            )
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
        elif bank == "d60":
            blocks[bank] = d60.compute_features_offline(history, stream)
        elif bank == "d61":
            blocks[bank] = d61.compute_features_offline(history, stream)
        elif bank == "d62":
            blocks[bank] = d62.compute_features_offline(history, stream)
        elif bank == "d63":
            blocks[bank] = d63.compute_features_offline(history, stream)
        elif bank == "d64":
            blocks[bank] = d64.compute_features_offline(history, stream)
        elif bank == "d65":
            first = blocks.get("d61")
            if first is None:
                first = d61.compute_features_offline(history, stream)
            second = blocks.get("d64")
            if second is None:
                second = d64.compute_features_offline(history, stream)
            blocks[bank] = d65.aggregate_features(first, second)
        elif bank == "d66":
            blocks[bank] = d66.compute_features_offline(history, stream)
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
        unknown = set(self.banks) - set(ALL_DETECTOR_BANKS)
        if unknown:
            raise ValueError(f"unsupported streaming banks: {sorted(unknown)}")
        history = np.asarray(historical, dtype=np.float64)
        self.inner = d4.StreamDetector4(history)
        self.transforms: dict[str, object] = {}
        if "d1" in self.banks:
            self.transforms["d1"] = d1.StreamDetector(history)
        if "d2" in self.banks:
            self.transforms["d2"] = d2.StreamDetector2(history)
        if "d5" in self.banks:
            self.transforms["d5"] = d5.StreamDetector5(history, self.inner)
        if "d6" in self.banks:
            self.transforms["d6"] = d6.StreamDetector6(history)
        if "d7" in self.banks:
            self.transforms["d7"] = d7.StreamDetector7(history)
        if "d8" in self.banks:
            self.transforms["d8"] = d8.StreamDetector8(history, self.inner)
        if "d9" in self.banks:
            self.transforms["d9"] = d9.StreamDetector9(history, self.inner)
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
        if "d13" in self.banks:
            self.transforms["d13"] = d13.StreamDetector13(history, self.inner)
        if "d16" in self.banks:
            # Stateless transform of the named d3 columns embedded in d4.
            self.transforms["d16"] = d16
        if "d17" in self.banks:
            self.transforms["d17"] = d17.StreamDetector17(history, self.inner)
        if "d19" in self.banks:
            self.transforms["d19"] = d19.StreamDetector19(
                history, D19_FIXED_TEMPLATES
            )
        if "d20" in self.banks:
            self.transforms["d20"] = d20.StreamDetector20(
                history, D20_FIXED_EMISSION
            )
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
        if "d31" in self.banks:
            self.transforms["d31"] = d31.StreamDetector31(
                history, D31_FIXED_TEMPLATES
            )
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
        if "d60" in self.banks:
            self.transforms["d60"] = d60.StreamDetector60(history)
        if "d61" in self.banks:
            self.transforms["d61"] = d61.StreamDetector61(history)
        if "d62" in self.banks:
            self.transforms["d62"] = d62.StreamDetector62(history)
        if "d63" in self.banks:
            self.transforms["d63"] = d63.StreamDetector63(history)
        if "d64" in self.banks:
            self.transforms["d64"] = d64.StreamDetector64(history)
        if "d65" in self.banks:
            self.transforms["d65"] = d65.StreamDetector65(history)
        if "d66" in self.banks:
            self.transforms["d66"] = d66.StreamDetector66(history)

    def step(self, value: float) -> np.ndarray:
        """Consume exactly one new value and return its feature vector."""

        base4 = self.inner.step(float(value))
        blocks: dict[str, np.ndarray] = {"d4": base4}
        base3 = base4[D3_COLUMNS_IN_D4]
        for bank in ("d1", "d2"):
            if bank in self.transforms:
                blocks[bank] = self.transforms[bank].step(value)  # type: ignore[attr-defined]
        if "d3" in self.banks:
            blocks["d3"] = base3
        if "d5" in self.transforms:
            blocks["d5"] = self.transforms["d5"].step_from_inner(value)  # type: ignore[attr-defined]
        if "d6" in self.transforms:
            blocks["d6"] = self.transforms["d6"].step(value)[  # type: ignore[attr-defined]
                d3.N_FEATURES :
            ]
        if "d7" in self.transforms:
            blocks["d7"] = self.transforms["d7"].step(value)[  # type: ignore[attr-defined]
                d6.N_FEATURES :
            ]
        if "d8" in self.transforms:
            blocks["d8"] = self.transforms["d8"].step_from_inner(value)  # type: ignore[attr-defined]
        if "d9" in self.transforms:
            blocks["d9"] = self.transforms["d9"].step_from_inner(base3)  # type: ignore[attr-defined]
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
        if "d13" in self.transforms:
            blocks["d13"] = self.transforms["d13"].step_from_inner(base3)  # type: ignore[attr-defined]
        if "d16" in self.transforms:
            blocks["d16"] = d16.extra_from_base(base3)
        if "d17" in self.transforms:
            blocks["d17"] = self.transforms["d17"].step_from_inner()  # type: ignore[attr-defined]
        for bank in ("d19", "d20", "d31"):
            if bank in self.transforms:
                blocks[bank] = self.transforms[bank].step(value)  # type: ignore[attr-defined]
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
        if "d60" in self.transforms:
            blocks["d60"] = self.transforms["d60"].step(value)  # type: ignore[attr-defined]
        for bank in MODEL8_BANKS:
            if bank in self.transforms:
                blocks[bank] = self.transforms[bank].step(value)  # type: ignore[attr-defined]
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
    contract = {
        "version": FEATURE_VERSION,
        "banks": list(banks),
        "names": feature_names(banks),
    }
    implementation_versions = {
        bank: BANK_IMPLEMENTATION_VERSIONS[bank]
        for bank in banks
        if bank in BANK_IMPLEMENTATION_VERSIONS
    }
    if implementation_versions:
        contract["implementation_versions"] = implementation_versions
    payload = json.dumps(contract, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _cache_paths(directory: Path, name: str) -> tuple[Path, Path, Path, Path]:
    return (
        directory / f"{name}.f32.mmap",
        directory / f"{name}.metadata.npz",
        directory / f"{name}.manifest.json",
        directory / f"{name}.progress.json",
    )


def load_feature_cache(
    cache_directory: str | Path = "models/artifacts",
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
    cache_directory: str | Path = "models/artifacts",
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
        if sys.version_info >= (3, 14):
            blocks = executor.map(extract, positions, buffersize=2 * workers)
        else:
            # ``buffersize`` was added in Python 3.14. Crunch currently uses
            # Python 3.12, where ordered map is still correct but unbounded.
            blocks = executor.map(extract, positions)
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
                    f"features[{name}]: {position + 1:4d}/{len(data)} series, "
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
