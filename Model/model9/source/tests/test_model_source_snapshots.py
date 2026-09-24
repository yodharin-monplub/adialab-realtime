"""Contracts for the complete source copies stored with numbered models."""

from __future__ import annotations

import json
from pathlib import Path

from models.sync_model_sources import (
    MANIFEST_NAME,
    MODELS_ROOT,
    PROJECT_ROOT,
    ROOT_SOURCE_NAMES,
    _numbered_models,
    validate_snapshot,
)


def test_numbered_models_are_contiguous() -> None:
    numbers = _numbered_models()
    assert numbers
    assert numbers == tuple(range(1, max(numbers) + 1))


def test_every_numbered_model_has_a_valid_source_snapshot() -> None:
    for number in _numbered_models():
        file_count, total_bytes = validate_snapshot(number)
        assert file_count > 0
        assert total_bytes > 0


def test_historical_snapshots_preserve_frozen_source_without_weights() -> None:
    for number in (1, 2, 3):
        source = MODELS_ROOT / f"model{number}" / "source"
        manifest = json.loads((source / MANIFEST_NAME).read_text(encoding="utf-8"))
        assert manifest["provenance"] == f"submissions/submission_{number}"
        assert (source / "main.py").read_bytes() == (
            PROJECT_ROOT / "submissions" / f"submission_{number}" / "main.py"
        ).read_bytes()
        assert not any(
            "resources" in path.relative_to(source).parts for path in source.rglob("*")
        )
    assert (MODELS_ROOT / "model3" / "source" / "core_feature_selection.npz").is_file()
    assert (
        MODELS_ROOT / "model3" / "source" / "candidate_feature_selection.npz"
    ).is_file()


def test_experiment_snapshots_contain_shared_and_model_source() -> None:
    for number in (n for n in _numbered_models() if n >= 4):
        source = MODELS_ROOT / f"model{number}" / "source"
        for name in ROOT_SOURCE_NAMES:
            assert (source / name).is_file()
        assert (source / "detector" / "core" / "detectors.py").is_file()
        assert (source / "detector" / "candidate" / "detectors66.py").is_file()
        assert (source / "models" / f"model{number}" / "__init__.py").is_file()
