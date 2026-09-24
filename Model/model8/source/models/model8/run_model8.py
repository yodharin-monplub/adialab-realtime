#!/usr/bin/env python3
"""Resumable command-line runner for Model 8."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model8.config import make_model8_config  # noqa: E402
from models.model8.model8_pipeline import (  # noqa: E402
    build_frozen_outer_prediction,
    build_model8_cache,
    compact_final,
    compact_outer,
    context_summary,
    evaluate_outer_holdout,
    freeze_outer_holdout,
    load_context,
    run_detector_screen,
    run_final_evaluation,
)


def _print(value: object) -> None:
    print(json.dumps(value, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "status",
            "build-cache",
            "screen",
            "final",
            "run-all",
            "freeze-outer",
            "predict-outer",
            "evaluate-outer",
        ),
        nargs="?",
        default="status",
    )
    args = parser.parse_args()
    config = make_model8_config(PROJECT_ROOT)

    if args.command == "status":
        cache_manifest = config.cache_path / f"{config.cache_name}.manifest.json"
        context = load_context(config, require_cache=cache_manifest.exists())
        _print(context_summary(context))
        return
    if args.command == "build-cache":
        cache = build_model8_cache(config)
        _print(
            {
                "stage": "model8_cache",
                "rows": cache.rows,
                "columns": cache.columns,
                "banks": cache.banks,
                "matrix": str(cache.matrix_path.relative_to(PROJECT_ROOT)),
            }
        )
        return

    if not (config.cache_path / f"{config.cache_name}.manifest.json").exists():
        build_model8_cache(config)
    context = load_context(config, require_cache=True)
    if args.command in {"freeze-outer", "predict-outer", "evaluate-outer"}:
        # Outer scoring is never part of `run-all`; it requires this explicit
        # command after the development experiment has already been frozen.
        frozen = freeze_outer_holdout(context)
        if args.command == "freeze-outer":
            _print(
                {
                    "stage": "model8_outer_frozen",
                    "contract_hash": frozen["contract_hash"],
                    "selected_new_model_weight": frozen["deployment_blend"][
                        "selected_candidate_weight"
                    ],
                    "outer_labels_read": False,
                }
            )
            return
        manifest = build_frozen_outer_prediction(context, frozen)
        if args.command == "predict-outer":
            _print(
                {
                    "stage": "model8_outer_prediction",
                    "contract_hash": frozen["contract_hash"],
                    "outer_rows": manifest["outer_rows"],
                    "selected_new_model_weight": manifest[
                        "selected_new_model_weight"
                    ],
                    "outer_labels_read": False,
                }
            )
            return
        _print(compact_outer(evaluate_outer_holdout(context, frozen, manifest)))
        return
    if args.command == "screen":
        report = run_detector_screen(context)
        _print(
            {
                "stage": report["stage"],
                "baseline": report["baseline"],
                "banks": {
                    bank: {
                        "standalone": detail["standalone"],
                        "nested_score": detail["nested_baseline_blend"]["score"],
                        "increment": detail["increment_over_baseline"],
                        "selected": detail["selected_by_full_development_diagnostic"],
                    }
                    for bank, detail in report["banks"].items()
                },
                "outer_holdout_read": report["outer_holdout_read"],
            }
        )
        return
    if args.command == "final":
        _print(compact_final(run_final_evaluation(context)))
        return

    run_detector_screen(context)
    _print(compact_final(run_final_evaluation(context)))


if __name__ == "__main__":
    main()
