#!/usr/bin/env python3
"""Checkpointed command-line runner for the long Model 6 experiment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Avoid repeated slow fallback cache creation on restricted workstations.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-model6")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model6.config import make_model6_config  # noqa: E402
from models.model5.config import make_model5_config  # noqa: E402
from models.model5.model5_pipeline import (  # noqa: E402
    cross_validate_family_models,
    load_model5_data,
    prepare_split,
    save_selected_detectors,
    screen_family,
)
from models.model6.model6_pipeline import (  # noqa: E402
    build_feature_rankings,
    build_frozen_holdout_prediction,
    context_summary,
    evaluate_frozen_holdout,
    freeze_deployment,
    load_context,
    run_all,
    run_detector_stack,
    run_development,
    run_ensembling,
    run_feature_search,
    run_finalist_cross_validation,
    run_hyperparameter_tuning,
    run_model_search,
)


def _prepare_model5_source() -> dict[str, Any]:
    """Build/validate caches and freeze fresh detector selection for Model 6.

    Every operation uses the updated ``train/`` data and complete-series
    development folds. Detector fold checkpoints make this safe to resume.
    The outer holdout labels are never indexed by this preparation stage.
    """

    source = make_model5_config(PROJECT_ROOT)
    data = load_model5_data(source)
    split = prepare_split(data, source)
    selection_path = source.artifact_path / "selected_detectors.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
    else:
        core = screen_family(data, split, "core", source)
        candidate = screen_family(data, split, "candidate", source)
        selection = save_selected_detectors(data, core, candidate, source)

    # Model 6's fold-excluded gain/split rankings reuse these selected-family
    # LightGBM baselines. Do this even when detector selection was already
    # checkpointed: each fold model is itself resumable, and an early return
    # here previously left ``combined_cv/`` absent after a clean rebuild.
    cross_validate_family_models(data, split, source)
    return selection


def _compact_result(value: Any) -> Any:
    """Avoid printing multi-megabyte fold grids after a completed stage."""

    if not isinstance(value, dict):
        return value
    if {"run", "development_series", "sealed_holdout_series"} <= value.keys():
        return {"stage": "status", "status": "complete", **value}
    if "holdout_tsroc" in value:
        bootstrap = value["complete_series_bootstrap"]
        return {
            "holdout_tsroc": value["holdout_tsroc"],
            "bootstrap_standard_error": bootstrap["standard_error"],
            "bootstrap_95_interval": bootstrap["percentile_95_interval"],
            "target_tsroc": value["target_tsroc"],
            "target_reached": value["target_reached"],
        }
    if "global" in value and isinstance(value["global"], dict):
        metric = value["global"].get("postprocess", {}).get("nested_tsroc", {})
        return {
            "global_nested_development_tsroc": metric,
            "target_reached": value["global"].get(
                "target_reached_on_nested_development_cv"
            ),
        }
    if value.get("created_without_outer_holdout_scoring") is True:
        return {
            "stage": "freeze_deployment",
            "status": "complete",
            "contract_hash": value["contract_hash"],
            "development_nested_tsroc": value["development_nested_tsroc"],
        }
    if value.get("holdout_labels_read") is False and "prediction_path" in value:
        return {
            "stage": "predict_holdout",
            "status": "complete",
            "frozen_contract_hash": value["frozen_contract_hash"],
            "prediction_path": value["prediction_path"],
            "holdout_rows": value["holdout_rows"],
            "holdout_labels_read": False,
        }
    return {"stage": value.get("stage"), "status": "complete"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or resume the leakage-controlled Model 6 search"
    )
    parser.add_argument(
        "command",
        choices=(
            "status",
            "prepare",
            "prepare-source",
            "detector-stack",
            "rank-features",
            "feature-search",
            "model-search",
            "tune",
            "final-cv",
            "ensemble",
            "freeze",
            "predict-holdout",
            "evaluate-holdout",
            "run-development",
            "run-all",
        ),
    )
    args = parser.parse_args()

    if args.command in {"prepare-source", "run-development", "run-all"}:
        source_selection = _prepare_model5_source()
        if args.command == "prepare-source":
            print(
                json.dumps(
                    {
                        "stage": "prepare-source",
                        "status": "complete",
                        "selection_hash": source_selection["selection_hash"],
                        "selected_core_banks": source_selection[
                            "selected_core_banks"
                        ],
                        "selected_candidate_banks": source_selection[
                            "selected_candidate_banks"
                        ],
                        "outer_holdout_read": False,
                    },
                    indent=2,
                ),
                flush=True,
            )
            return
    context = load_context(make_model6_config(PROJECT_ROOT))

    if args.command in {"status", "prepare"}:
        result: Any = context_summary(context)
    elif args.command == "detector-stack":
        result = run_detector_stack(context)
    elif args.command == "rank-features":
        build_feature_rankings(context)
        result = {"stage": "rank-features"}
    elif args.command == "feature-search":
        result = run_feature_search(context)
    elif args.command == "model-search":
        result = run_model_search(context)
    elif args.command == "tune":
        result = run_hyperparameter_tuning(context)
    elif args.command == "final-cv":
        result = run_finalist_cross_validation(context)
    elif args.command == "ensemble":
        result = run_ensembling(context)
    elif args.command == "freeze":
        result = freeze_deployment(context)
    elif args.command == "predict-holdout":
        frozen = freeze_deployment(context)
        result = build_frozen_holdout_prediction(context, frozen)
    elif args.command == "evaluate-holdout":
        frozen = freeze_deployment(context)
        manifest = build_frozen_holdout_prediction(context, frozen)
        result = evaluate_frozen_holdout(context, frozen, manifest)
    elif args.command == "run-development":
        result = run_development(context)
    elif args.command == "run-all":
        result = run_all(context)
    else:  # pragma: no cover - argparse guarantees this cannot happen.
        raise AssertionError(args.command)
    print(json.dumps(_compact_result(result), indent=2), flush=True)


if __name__ == "__main__":
    main()
