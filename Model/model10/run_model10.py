#!/usr/bin/env python3
"""Checkpointed command-line runner for Model 10 post-processing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model10.config import make_model10_config  # noqa: E402
from models.model10.model10_pipeline import (  # noqa: E402
    run_postprocess_study,
    study_status,
)


def _ready(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "__dataclass_fields__"):
        return {key: _ready(item) for key, item in vars(value).items()}
    if isinstance(value, dict):
        return {str(key): _ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_ready(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _compact(summary: dict[str, Any]) -> dict[str, Any]:
    reports = []
    for report in summary["reports"]:
        paired = report["nested_selection"]["paired_gain"]
        reports.append(
            {
                "source": report["source"],
                "raw_fold_mean": report["raw"]["fold_mean"],
                "raw_fold_standard_error": report["raw"][
                    "fold_standard_error"
                ],
                "nested_held_cell_mean": report["nested_selection"][
                    "held_cell_selected_mean"
                ],
                "nested_held_cell_raw_mean": report["nested_selection"][
                    "held_cell_raw_mean"
                ],
                "mean_gain": paired.mean_gain,
                "paired_gain_standard_error": paired.paired_standard_error,
                "fold_wins": paired.wins,
                "stage_one_promotion_gate_passed": report[
                    "stage_one_promotion_gate_passed"
                ],
                "best_fixed_candidate_diagnostic": report[
                    "best_fixed_candidate_diagnostic"
                ]["name"],
                "deployment_candidate": report[
                    "all_development_deployment_candidate"
                ]["name"],
            }
        )
    return {
        "artifact_directory": summary["protocol"]["artifact_directory"],
        "sealed_outer_holdout_labels_used": False,
        "sealed_outer_holdout_rows_materialized": False,
        "reports": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Explore causal score-history corrections on frozen development OOF "
            "predictions; no outer-holdout command exists"
        )
    )
    parser.add_argument(
        "command", choices=("status", "run"), nargs="?", default="status"
    )
    parser.add_argument(
        "--model6-only",
        action="store_true",
        help="skip the exploratory, non-deployable Model 9 pairwise source",
    )
    args = parser.parse_args()
    config = make_model10_config(PROJECT_ROOT)
    if args.command == "status":
        result = study_status(config)
    else:
        result = _compact(
            run_postprocess_study(
                config, include_model9=not args.model6_only
            )
        )
    print(json.dumps(_ready(result), indent=2), flush=True)


if __name__ == "__main__":
    main()
