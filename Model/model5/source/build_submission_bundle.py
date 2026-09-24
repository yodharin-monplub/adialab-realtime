"""Build a portable or directly uploadable Crunch submission directory.

The output contains the production source, every detector module, the two
all-series LightGBM boosters, and relative-path manifests.  It deliberately
excludes train data, OOF predictions, fold models, and rejected experiments.
With ``--crunch-layout``, it also writes Crunch's default ``main.py`` entry
point and stores the pretrained models under ``resources/``.  The command
refuses to overwrite an existing output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
SOURCE_FILES = (
    "solution.py",
    "feature_pipeline.py",
    "competition_data.py",
    "train_cv.py",
    "tsauc.py",
    "postprocess.py",
    "requirements.txt",
    "README.md",
)

CRUNCH_MAIN = '''"""Crunch submission entry point for the pretrained TSAUC ensemble.

The implementation lives in ``solution.py`` so it can be tested independently
of Crunch's filename convention.  These thin wrappers deliberately keep the
default ``main.py`` interface expected by ``crunch test`` and ``crunch push``.
"""

from __future__ import annotations

from typing import Iterable

from solution import infer as _infer
from solution import train as _train


# Streaming inference is CPU-bound while each LightGBM call is single-threaded.
# Crunch runs these workers as separate processes over disjoint series.
INFER_PARALLELISM = 4


def train(datasets: Iterable[tuple], model_directory_path: str) -> None:
    """Preserve and validate the uploaded ensemble, or train the fallback."""

    _train(datasets, model_directory_path)


def infer(datasets: Iterable[tuple], model_directory_path: str):
    """Yield Crunch's readiness sentinel and one score per streamed value."""

    yield from _infer(datasets, model_directory_path)
'''

SCRATCH_ENSEMBLE_MAIN = '''"""Crunch entry point for train-from-scratch TSAUC ensemble.

No pretrained model is bundled. ``train`` fits both selected feature families
using only Crunch's labeled training iterator; ``infer`` loads the resulting
relative-path ensemble from the supplied resources directory.
"""

from __future__ import annotations

from typing import Iterable

from ensemble_training import train_ensemble
from solution import infer as _infer


INFER_PARALLELISM = 4


def train(datasets: Iterable[tuple], model_directory_path: str) -> None:
    """Fit core and candidate models and save their configured logit ensemble."""

    train_ensemble(datasets, model_directory_path)


def infer(datasets: Iterable[tuple], model_directory_path: str):
    """Yield readiness, then one ensemble probability per streamed value."""

    yield from _infer(datasets, model_directory_path)
'''

CRUNCH_GITIGNORE = """# Local/runtime files that must never enter a submission.
__pycache__/
*.py[cod]
data/
train/
prediction/
.pytest_cache/
"""

CRUNCH_SUBMIT = """# Submit this directory

This is the default Crunch script layout:

- `main.py`: entry point exposing `train()` and streaming `infer()`;
- `resources/`: the validated pretrained core/candidate logit ensemble;
- `requirements.txt`: runtime dependencies;
- detector and feature modules imported by the entry point.

For the web uploader, select this entire directory as the submission root.

For the CLI, first use the one-time setup command shown on the competition's
**Submit via CLI** page. Copy this directory's contents into that initialized
project, then run:

```bash
crunch test
crunch push --message "pretrained equal-logit TSAUC ensemble"
```

`train()` validates and preserves the uploaded ensemble when Crunch forces a
first training call. No train data, protected test archive, feature cache, OOF
prediction, or rejected experimental model is included here.
"""

SCRATCH_ENSEMBLE_SUBMIT = """# Submit this directory

This bundle uses Crunch's default script layout and contains **no pretrained
model weights**:

- `main.py`: exposes train-from-scratch `train()` and streaming `infer()`;
- `ensemble_training.py`: fits both selected LightGBM feature families;
- `resources/`: intentionally empty until Crunch calls `train()`;
- `requirements.txt`: runtime and training dependencies;
- complete feature, detector, metric, and training support code.

`train()` consumes only the labeled tuples supplied by the competition. It
does not read a local test directory, protected archive, cached feature file,
OOF prediction, or pretrained weight. It writes this runtime structure:

```text
resources/
├── manifest.json
├── core/
│   ├── final.txt
│   └── manifest.json
└── candidate/
    ├── final.txt
    └── manifest.json
```

For the web uploader, select this entire directory as the submission root.
For the CLI, copy its contents into the project created by the competition's
setup command, then run `crunch test` and `crunch push`.
"""

SELECTED_FEATURE_REPORT = """# Frozen feature-selection contract

This is an experimental successor to `submission_2`. It contains no pretrained
model. Crunch's `train()` fits both LightGBMs from its labeled training iterator
and writes every learned weight under `resources/`.

The two small `*_feature_selection.npz` files contain only integer source-column
IDs and matching feature names. They freeze a selection derived exclusively
from the local `train/` data before the locked test was scored.

## Leakage-safe protocol

- train: whole-series folds 2-4 (4,800 series);
- validation: whole-series fold 1 (1,600 series);
- locked test: whole-series fold 0 (1,600 series);
- rankings used train rows only;
- subset methods and blend weights were chosen on validation only;
- the locked test was evaluated once after the contract was frozen.

Methods tested separately for core and candidate were all features; LightGBM
gain, split count, mean absolute SHAP, and consensus top 64/128/256/512/768;
nonzero gain; gain plus 0.98 absolute-correlation pruning; cumulative feature
banks; and a random-256 negative control excluded from deployment selection.

## Frozen configuration

- core: consensus top 768, 101 rounds, logit weight 0.55;
- candidate: gain ranking plus correlation pruning, 671 columns, 217 rounds,
  logit weight 0.45;
- contract SHA-256: `e7154880585cb472bbed94079048312641d5b4f03d6dfc912ac4a8e908861a8f`.

Validation TSAUC improved from 0.614042 to 0.621166. Locked-test TSAUC moved
from 0.630459 to 0.630597 (+0.000138). The paired five-subfold delta had SE
0.001653 and 95% t interval [-0.004187, 0.004994], so the improvement is not
statistically resolved. Treat this as a lower-dimensional cloud experiment,
not as a demonstrated replacement for submission 2.
"""

SELECTED_SUBMISSION_README_SOURCE = (
    "submissions/notes/submission_3_development.md"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build(
    output: Path,
    *,
    crunch_layout: bool = False,
    train_from_scratch_ensemble: bool = False,
    selected_feature_ensemble: bool = False,
) -> Path:
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite existing bundle: {output}; choose --output"
        )
    if selected_feature_ensemble:
        train_from_scratch_ensemble = True
    if train_from_scratch_ensemble:
        crunch_layout = True

    output.mkdir(parents=True)
    source_files = SOURCE_FILES + (
        ("ensemble_training.py",) if train_from_scratch_ensemble else ()
    )
    for name in source_files:
        shutil.copy2(PROJECT / name, output / name)
    if crunch_layout:
        main_source = (
            SCRATCH_ENSEMBLE_MAIN
            if train_from_scratch_ensemble
            else CRUNCH_MAIN
        )
        submit_source = (
            SCRATCH_ENSEMBLE_SUBMIT
            if train_from_scratch_ensemble
            else CRUNCH_SUBMIT
        )
        (output / "main.py").write_text(main_source, encoding="utf-8")
        (output / ".gitignore").write_text(CRUNCH_GITIGNORE, encoding="utf-8")
        (output / "SUBMIT.md").write_text(submit_source, encoding="utf-8")

    if selected_feature_ensemble:
        selection_root = PROJECT / "models" / "artifacts" / "feature_selection_v1"
        for family, method in (
            ("core", "consensus_top_768"),
            ("candidate", "gain_corr_0p98"),
        ):
            source = selection_root / family / "selections" / f"{method}.npz"
            if not source.is_file():
                raise FileNotFoundError(f"missing frozen selection: {source}")
            shutil.copy2(source, output / f"{family}_feature_selection.npz")
        (output / "FEATURE_SELECTION.md").write_text(
            SELECTED_FEATURE_REPORT,
            encoding="utf-8",
        )
        shutil.copy2(
            PROJECT / SELECTED_SUBMISSION_README_SOURCE,
            output / "README.md",
        )

    detector_source = PROJECT / "detector"
    detector_output = output / "detector"
    detector_output.mkdir()
    for source in sorted(detector_source.rglob("*.py")):
        relative = source.relative_to(detector_source)
        destination = detector_output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    model_directory_name = "resources" if crunch_layout else "model"
    model_output = output / model_directory_name
    model_output.mkdir()
    if not train_from_scratch_ensemble:
        for family, source_name in (
            ("core", "model_core"),
            ("candidate", "model_candidate"),
        ):
            source = PROJECT / "models" / "artifacts" / source_name
            destination = model_output / family
            destination.mkdir()
            shutil.copy2(source / "manifest.json", destination / "manifest.json")
            shutil.copy2(source / "final.txt", destination / "final.txt")

        ensemble_manifest = {
            "ensemble_method": "logit",
            "components": [
                {"directory": "core", "weight": 0.5},
                {"directory": "candidate", "weight": 0.5},
            ],
            "postprocess_penalty": None,
            "weight_status": "median of five nested leave-one-fold-out selections",
            "validation_report": "README.md#results-and-uncertainty",
        }
        (model_output / "manifest.json").write_text(
            json.dumps(ensemble_manifest, indent=2)
        )

    files = sorted(path for path in output.rglob("*") if path.is_file())
    inventory = {
        "layout": "crunch" if crunch_layout else "portable",
        "training_mode": (
            "from_scratch_selected_core_candidate"
            if selected_feature_ensemble
            else "from_scratch_core_candidate"
            if train_from_scratch_ensemble
            else "pretrained_core_candidate"
        ),
        "entry_points": {
            "train": "main.train" if crunch_layout else "solution.train",
            "infer": "main.infer" if crunch_layout else "solution.infer",
        },
        "model_directory": model_directory_name,
        "files": {
            str(path.relative_to(output)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in files
        },
    }
    (output / "bundle_manifest.json").write_text(json.dumps(inventory, indent=2))
    print(
        f"built {output}: {len(files) + 1} files, "
        f"{sum(path.stat().st_size for path in files):,} bytes"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("submissions/submission_bundle")
    )
    parser.add_argument(
        "--crunch-layout",
        action="store_true",
        help="write main.py and use the default resources/ model directory",
    )
    parser.add_argument(
        "--selected-feature-ensemble",
        action="store_true",
        help=(
            "bundle the frozen leakage-safe selected core/candidate contract; "
            "implies --train-from-scratch-ensemble"
        ),
    )
    parser.add_argument(
        "--train-from-scratch-ensemble",
        action="store_true",
        help=(
            "bundle no weights; main.train fits core and candidate models "
            "from the labeled train iterator"
        ),
    )
    args = parser.parse_args()
    build(
        args.output.resolve(),
        crunch_layout=args.crunch_layout,
        train_from_scratch_ensemble=args.train_from_scratch_ensemble,
        selected_feature_ensemble=args.selected_feature_ensemble,
    )


if __name__ == "__main__":
    main()
