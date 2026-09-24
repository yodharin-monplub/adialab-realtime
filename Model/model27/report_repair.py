"""Audited reporting-only repair for the completed frozen Model27 run.

The original training and report sources remain byte-for-byte frozen.  This
wrapper replaces only the broken report-time Model26 directory lookup, then
executes the original report and records the repair provenance.
"""
from pathlib import Path
import json
import sys

ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_PATH))

from models.model11.pipeline import sha256, write_json
from models.model18.run_model18 import read_checkpoint, validate_predictions
from models.model26 import run_model26 as baseline
from models.model27 import report as original_report
from models.model27.run_model27 import ROOT, CONFIG, candidate_probability


EXPECTED_REPORT_SHA256 = "b35d10b72ae02220754cbf25fbc6916ef9eb82d378fd36f78b6907e675f34612"
EXPECTED_RUNNER_SHA256 = "8b96012920f493496a56c49bc7146b35f22796e91e3f52db12233e2b3ac3af20"


def fixed_prediction_checkpoint(directory, arm, fold, fingerprint, held):
    baseline_directory, _, baseline_fingerprint = baseline.frozen_study()
    base = baseline.prediction_checkpoint(
        baseline_directory, "depth_mix", fold, baseline_fingerprint, held)
    if arm == "model26":
        return base
    if arm != "matched_context":
        raise ValueError("Unknown Model27 arm")
    saved = read_checkpoint(directory / f"matched_context_fold{fold}.npz",
        dict(protocol_sha256=fingerprint, fold=fold, arm="matched_context"))
    if saved is None:
        raise RuntimeError("Incomplete Model27 fit")
    validate_predictions(saved, held)
    return candidate_probability(base, saved["prediction"])


def main():
    report_path = ROOT / "models/model27/report.py"
    runner_path = ROOT / "models/model27/run_model27.py"
    if sha256(report_path) != EXPECTED_REPORT_SHA256 or sha256(runner_path) != EXPECTED_RUNNER_SHA256:
        raise ValueError("Original frozen Model27 sources changed")
    directory, protocol, fingerprint = original_report.frozen_study()
    if protocol["contract"]["config"] != CONFIG:
        raise ValueError("Model27 configuration changed")
    original_report.prediction_checkpoint = fixed_prediction_checkpoint
    summary = original_report.report()
    write_json(directory / "report_repair.json", dict(
        reason="Supply the frozen Model26 run directory during report-time reconstruction; no prediction changed",
        original_report_sha256=sha256(report_path),
        original_runner_sha256=sha256(runner_path),
        repair_wrapper_sha256=sha256(Path(__file__)),
        protocol_sha256=fingerprint,
        prediction_hashes={path.name: sha256(path) for path in
            sorted(directory.glob("matched_context_fold*.npz"))},
        selected_candidate=summary["selected_candidate"],
        target_reached=summary["target_reached"],
    ))


if __name__ == "__main__":
    main()
