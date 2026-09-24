#!/usr/bin/env python3
"""Checkpointed command-line runner for the train-only Model 7 study."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-model7")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model7.config import make_model7_config  # noqa: E402
from models.model7.model7_pipeline import (  # noqa: E402
    context_summary,
    load_context,
    run_all,
    run_ensemble,
    run_repeated_cv,
    run_screen,
)


def _compact(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    if value.get("stage") == "nested_robust_ensemble":
        bootstrap = value["complete_series_bootstrap_of_average_repeat_oof"]
        return {
            "stage": value["stage"],
            "selected_architecture": value["selected_architecture"],
            "repeated_cv_tsroc": value["selected_score"]["repeat_means"],
            "robust_repeat_score": value["selected_score"]["robust_repeat_score"],
            "pooled_average_repeat_oof": value["selected_score"][
                "pooled_average_repeat_oof"
            ],
            "bootstrap_standard_error": bootstrap["standard_error"],
            "bootstrap_95_interval": bootstrap["percentile_95_interval"],
            "target_reached": value["target_reached_on_repeated_development_cv"],
            "outer_holdout_read": False,
        }
    if value.get("stage") == "three_repeat_five_fold_cv":
        return {
            "stage": value["stage"],
            "models": {
                name: {
                    "repeat_means": report["repeat_means"],
                    "robust_repeat_score": report["robust_repeat_score"],
                    "pooled_average_repeat_oof": report[
                        "pooled_average_repeat_oof"
                    ],
                }
                for name, report in value["models"].items()
            },
            "outer_holdout_read": False,
        }
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or resume Model 7 without opening Model 6's holdout"
    )
    parser.add_argument(
        "command",
        choices=("status", "prepare", "screen", "repeated-cv", "ensemble", "run-all"),
    )
    args = parser.parse_args()
    context = load_context(make_model7_config(PROJECT_ROOT))
    if args.command in {"status", "prepare"}:
        result = context_summary(context)
    elif args.command == "screen":
        result = run_screen(context)
    elif args.command == "repeated-cv":
        result = run_repeated_cv(context)
    elif args.command == "ensemble":
        result = run_ensemble(context)
    elif args.command == "run-all":
        result = run_all(context)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(_compact(result), indent=2), flush=True)


if __name__ == "__main__":
    main()
