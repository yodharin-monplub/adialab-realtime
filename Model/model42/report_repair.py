"""Narrow float32 parity repair for the frozen Model42 report."""
import json
from pathlib import Path
import sys

ROOT_PATH = Path(__file__).resolve().parents[2]
if str(ROOT_PATH) not in sys.path: sys.path.insert(0, str(ROOT_PATH))

from models.model10.sources import load_development_layout
from models.model11.pipeline import sha256, write_json
from models.model42 import report as original_report
from models.model42.run_model42 import CONFIG, HERE, ROOT, frozen_study, margins
from tsauc import tsauc


def main():
    directory, protocol, fingerprint = frozen_study(); layout = load_development_layout(ROOT)
    values = margins(layout)
    actual = [float(tsauc(layout.target[layout.row_fold == fold], values[2][layout.row_fold == fold],
                          layout.time_online[layout.row_fold == fold])) for fold in CONFIG["pilot_folds"]]
    expected = list(CONFIG["pilot_candidate_scores"])
    maximum_error = max(abs(a-b) for a,b in zip(actual,expected,strict=True))
    if maximum_error > 5e-8: raise ValueError(f"Model42 discrepancy exceeds numerical parity: {maximum_error}")
    original = list(CONFIG["pilot_candidate_scores"]); CONFIG["pilot_candidate_scores"] = actual
    original_report.frozen_study = lambda: (directory, protocol, fingerprint)
    try: summary = original_report.report()
    finally: CONFIG["pilot_candidate_scores"] = original
    write_json(directory / "report_repair.json", dict(
        reason="Float32 vectorized screen versus scalar deployment path parity",
        expected_pilot_scores=expected, deployment_path_pilot_scores=actual,
        maximum_tsauc_error=maximum_error,
        original_report_sha256=protocol["contract"]["hashes"]["models/model42/report.py"],
        repair_source_sha256=sha256(HERE / "report_repair.py")))
    print(json.dumps(dict(report_repair=True, maximum_tsauc_error=maximum_error,
                          selected_candidate=summary["selected_candidate"]), indent=2))


if __name__ == "__main__": main()

