"""Report-only repair: load confirmation checkpoints for the selected arm.

The frozen runner correctly trains only the pilot-selected arm on confirmation
folds, but final_report originally asked processed_signals to load every arm.
This wrapper narrows the in-memory arm mapping after validating the frozen
protocol and pilot decision. It changes no prediction, fit, blend, or score.
"""
from __future__ import annotations

import json

from models.model10.sources import load_development_layout
from models.model84 import run_model84 as runner


def main():
    directory, _, fingerprint = runner.frozen_study()
    decision = json.loads((directory / "pilot_decision.json").read_text())
    selected = decision["selected_arm"]
    if not decision["continue_training"] or selected not in runner.CONFIG["arms"]:
        raise ValueError("Invalid frozen Model84 pilot decision")
    original = runner.CONFIG["arms"]
    try:
        runner.CONFIG["arms"] = {selected: original[selected]}
        layout = load_development_layout(runner.ROOT)
        runner.final_report(directory, fingerprint, layout, decision)
    finally:
        runner.CONFIG["arms"] = original
    note = ("\nReport repair: the frozen runner trained only the pilot-selected arm on "
            "confirmation folds but initially attempted to load all arms while reporting. "
            "report_fix.py restricts loading to the already-frozen selected arm; no fit, "
            "prediction, blend, or score was changed.\n")
    for path in (directory / "RESULTS.md", runner.HERE / "RESULTS.md"):
        path.write_text(path.read_text() + note)


if __name__ == "__main__":
    main()
