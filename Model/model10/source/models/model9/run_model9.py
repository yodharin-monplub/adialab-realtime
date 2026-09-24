#!/usr/bin/env python3
"""Checkpointed command-line runner for the holdout-blind Model 9 study."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-model9")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model9.config import make_model9_config  # noqa: E402
from models.model9.model9_pipeline import (  # noqa: E402
    compact_report,
    context_summary,
    load_context,
    run_all,
    run_meta_stack,
    run_promoted_cv,
    run_scale_cv,
)


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2), flush=True)


def _compact_pairwise(report: dict[str, Any]) -> dict[str, Any]:
    """Keep the CLI useful without printing source hashes and 34 coefficients."""

    return {
        "stage": report["stage"],
        "pairwise_stack": report["pairwise_stack"],
        "comparison_to_model6": report["comparison_to_model6"],
        "oof_prediction_path": report["oof_prediction_path"],
        "deployment_coefficient_path": report["deployment_coefficient_path"],
        "level_two_oof_is_exploratory": report["level_two_oof_is_exploratory"],
        "upstream_model5_bank_selection_reused_development_folds": report.get(
            "upstream_model5_bank_selection_reused_development_folds", True
        ),
        "outer_holdout_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run or resume Model 9 using development OOF data only; this "
            "runner intentionally exposes no outer-holdout command"
        )
    )
    parser.add_argument(
        "command",
        choices=(
            "status",
            "pairwise-stack",
            "scale-cv",
            "promote",
            "meta-stack",
            "run-all",
        ),
        nargs="?",
        default="status",
    )
    args = parser.parse_args()
    context = load_context(make_model9_config(PROJECT_ROOT))

    if args.command == "status":
        _print(context_summary(context))
        return
    if args.command == "pairwise-stack":
        from models.model9.pairwise_stack import run_pairwise_stack

        _print(_compact_pairwise(run_pairwise_stack(context)))
        return
    if args.command == "scale-cv":
        result = run_scale_cv(context)
    elif args.command == "promote":
        result = run_promoted_cv(context)
    elif args.command == "meta-stack":
        result = run_meta_stack(context)
    elif args.command == "run-all":
        # The pairwise stack is cheap and checkpointed, so run it first. The
        # full-row feature stages then resume independently after interruption.
        from models.model9.pairwise_stack import run_pairwise_stack

        pairwise = run_pairwise_stack(context)
        result = run_all(context)
        compact = compact_report(result)
        compact["pairwise_stack"] = _compact_pairwise(pairwise)
        _print(compact)
        return
    else:  # pragma: no cover - argparse validates choices.
        raise AssertionError(args.command)
    _print(compact_report(result))


if __name__ == "__main__":
    main()
