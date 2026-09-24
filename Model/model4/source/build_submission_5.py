"""Build the conditional no-pretrained joint-LightGBM Crunch bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from feature_pipeline import CANDIDATE_BANKS, CORE_BANKS, NOVEL_BANKS


PROJECT = Path(__file__).resolve().parent
ARTIFACT_ROOT = PROJECT / "models" / "artifacts" / "joint_lightgbm_v1"
FROZEN_CONTRACT_FILE = "frozen_contract_fixedseed_v2.json"
SELECTION_FILE = "selected_joint_features_fixedseed_v2.npz"
SEALED_RESULT_FILE = "sealed_test_results_fixedseed_v2.json"

SOURCE_FILES = (
    "solution.py", "feature_pipeline.py", "competition_data.py", "train_cv.py",
    "tsauc.py", "postprocess.py", "joint_training.py",
    "requirements.txt",
)

MAIN = '''"""Crunch entry point for the selected single joint LightGBM."""

from __future__ import annotations

from typing import Iterable

from joint_training import train_joint_model
from solution import infer as _infer


INFER_PARALLELISM = 4


def train(datasets: Iterable[tuple], model_directory_path: str) -> None:
    """Train from labels and write every model artifact under resources."""

    train_joint_model(datasets, model_directory_path)


def infer(datasets: Iterable[tuple], model_directory_path: str):
    """Yield readiness and one strictly causal probability per online value."""

    yield from _infer(datasets, model_directory_path)
'''

SUBMIT = '''# Push submission 5

This directory is the submission root and contains no pretrained model.
Crunch calls `main.train()`, which learns one selected joint LightGBM from the
labeled iterator and writes only inside its supplied `resources/` path.

```bash
cd submissions/submission_5
crunch test
crunch push --message "submission 5 selected joint LightGBM"
```

Do not run `crunch push` from the parent repository.
'''

GITIGNORE = '''__pycache__/
*.py[cod]
data/
train/
prediction/
.pytest_cache/
resources/*
!resources/.gitkeep
'''


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build(output: Path) -> Path:
    report_path = ARTIFACT_ROOT / SEALED_RESULT_FILE
    frozen_path = ARTIFACT_ROOT / FROZEN_CONTRACT_FILE
    selection_path = ARTIFACT_ROOT / SELECTION_FILE
    for path in (report_path, frozen_path, selection_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing completed experiment artifact: {path}")
    report = json.loads(report_path.read_text())
    frozen = json.loads(frozen_path.read_text())
    if not report.get("submission_5_eligible"):
        raise RuntimeError(
            "sealed joint model did not beat the same-split incumbent; "
            "submission_5 must not be built"
        )
    if report["frozen_contract_hash"] != frozen["contract_hash"]:
        raise ValueError("test report and frozen contract hashes differ")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing bundle: {output}")

    output.mkdir(parents=True)
    for filename in SOURCE_FILES:
        shutil.copy2(PROJECT / filename, output / filename)
    detector_source = PROJECT / "detector"
    detector_output = output / "detector"
    detector_output.mkdir()
    for source in sorted(detector_source.rglob("*.py")):
        relative = source.relative_to(detector_source)
        destination = detector_output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    shutil.copy2(selection_path, output / "joint_feature_selection.npz")

    config = {
        "banks": list(CORE_BANKS + CANDIDATE_BANKS + NOVEL_BANKS),
        "selection_name": frozen["selection"]["name"],
        "feature_count": frozen["selection"]["feature_count"],
        "family_counts": frozen["selection"]["family_counts"],
        "boosting_rounds": frozen["joint_model"]["best_iteration"],
        "parameters": frozen["joint_model"]["parameters"],
        "frozen_contract_hash": frozen["contract_hash"],
        "offline_test_tsauc": report["joint"]["pooled_tsauc"],
        "offline_test_standard_error": report["joint"][
            "series_subfold_standard_error"
        ],
        "training_source": "Crunch train() labeled datasets only",
    }
    (output / "joint_model_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (output / "main.py").write_text(MAIN, encoding="utf-8")
    (output / "SUBMIT.md").write_text(SUBMIT, encoding="utf-8")
    (output / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    result_document = (PROJECT / "models" / "joint_lightgbm_results.md").read_text()
    method_document = (PROJECT / "models" / "JOINT_LIGHTGBM.md").read_text()
    (output / "README.md").write_text(
        "# Submission 5: selected joint LightGBM\n\n"
        + result_document + "\n\n" + method_document,
        encoding="utf-8",
    )
    resources = output / "resources"
    resources.mkdir()
    (resources / ".gitkeep").write_text("", encoding="utf-8")
    manifest = {
        "bundle": "submission_5",
        "contains_pretrained_model": False,
        "frozen_contract_hash": frozen["contract_hash"],
        "files": {
            str(path.relative_to(output)): {
                "bytes": path.stat().st_size, "sha256": _sha256(path)
            }
            for path in sorted(output.rglob("*")) if path.is_file()
        },
    }
    (output / "bundle_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"built push-ready no-pretrained bundle: {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "submissions" / "submission_5"
    )
    args = parser.parse_args()
    build(args.output.resolve())


if __name__ == "__main__":
    main()
