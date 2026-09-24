"""Dependency-free runner for the small function-style regression suite."""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path


# Make ``python3 tests/run_tests.py`` work without requiring callers to export
# PYTHONPATH first.  The test modules use project-root imports just like the
# Crunch bundle does.
PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))


MODULES = (
    "tests.test_tsauc",
    "tests.test_feature_pipeline",
    "tests.test_detector_regressions",
    "tests.test_model4",
    "tests.test_model5",
    "tests.test_model6",
    "tests.test_model7",
    "tests.test_model8",
    "tests.test_model9",
    "tests.test_model10",
    "tests.test_model10_selection",
    "tests.test_model10_sources",
    "tests.test_model_source_snapshots",
    "tests.test_training_protocol",
    "tests.test_joint_lightgbm_experiment",
)


def main() -> None:
    completed = 0
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        for name, function in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("test_"):
                function()
                completed += 1
                print(f"PASS {module_name}.{name}")
    print(f"PASS: {completed} tests")


if __name__ == "__main__":
    main()
