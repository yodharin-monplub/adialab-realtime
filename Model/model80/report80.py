"""Bind Model79's identical frozen report machinery to Model80 artifacts."""
from __future__ import annotations

import json
from pathlib import Path

from models.model11.pipeline import write_json
from models.model79 import report79 as shared

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "prespec.json").read_text())


def report(directory, fingerprint, layout):
    # Model79 and Model80 use the same parent/replacement/gating contract. Rebind
    # the parameter and output roots while retaining one checksummed evaluator.
    shared.CONFIG = CONFIG
    shared.HERE = HERE
    result = shared.report(directory, fingerprint, layout)
    summary_path = directory / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        summary["model"] = 80
        write_json(summary_path, summary)
    for path in (directory / "RESULTS.md", HERE / "RESULTS.md"):
        if path.exists():
            text = path.read_text().replace("Model79", "Model80").replace(
                "regime-matched aligned splices", "TSAUC-mass-aligned splice sampling")
            path.write_text(text)
    return result
