"""Reconstruct Model84's final OOF score vector (all 5 dev folds) on Windows.

Linux-only modules imported by the historical runners (fcntl, resource) are
stubbed; no fitting happens, only frozen predictions are loaded and the
frozen stack/postprocess is re-applied. Saved to cache/model84_final_oof.npy.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import types

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for name in ("fcntl", "resource"):
    if name not in sys.modules:
        stub = types.ModuleType(name)
        stub.flock = lambda *a, **k: None
        stub.LOCK_EX = stub.LOCK_NB = 0
        stub.getrusage = lambda *a, **k: types.SimpleNamespace(ru_maxrss=0)
        stub.RUSAGE_SELF = 0
        sys.modules[name] = stub

import numpy as np  # noqa: E402

from models.model10.sources import load_development_layout  # noqa: E402
import models.model84.run_model84 as m84  # noqa: E402
from tsauc import tsauc  # noqa: E402


def main():
    layout = load_development_layout(ROOT)
    directory = ROOT / json.loads((m84.HERE / "runs/latest.json").read_text())["directory"]
    fingerprint = json.loads((m84.HERE / "runs/latest.json").read_text())["protocol_sha256"]
    decision = json.loads((directory / "pilot_decision.json").read_text())
    m84.CONFIG["arms"] = {decision["selected_arm"]: m84.CONFIG["arms"][decision["selected_arm"]]}
    baseline, signals = m84.processed_signals(directory, fingerprint, layout, range(5))
    weight = decision["selected_weight"]
    final = ((1 - weight) * baseline + weight * signals[decision["selected_arm"]]).astype(np.float32)
    scores = [tsauc(layout.target[layout.row_fold == f], final[layout.row_fold == f],
                    layout.time_online[layout.row_fold == f]) for f in range(5)]
    print("reconstructed fold scores", scores, np.mean(scores))
    out = Path(__file__).resolve().parent / "cache"
    out.mkdir(exist_ok=True)
    np.save(out / "model84_final_oof.npy", final)
    np.save(out / "model67_baseline_oof.npy", np.asarray(baseline, dtype=np.float32))


if __name__ == "__main__":
    main()
