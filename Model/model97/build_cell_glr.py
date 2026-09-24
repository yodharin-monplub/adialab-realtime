"""Build and label-free audit Model97's OOF-aligned cell GLR feature."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model10.sources import load_development_layout
from models.model11.pipeline import sha256
from models.model20.features import load_real_pool
from models.model97.cell_glr import FEATURE_NAMES, cell_glr_features, prefix_parity

HERE = Path(__file__).resolve().parent


def main():
    layout = load_development_layout(ROOT)
    pool = load_real_pool(layout)
    path, partial = HERE / "real_cell_glr.npy", HERE / "real_cell_glr.partial.npy"
    matrix = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.float32, shape=(layout.row_count, 1))
    positions, offset = layout.series_position, 0
    started = time.monotonic()
    for number, (position, (history, online)) in enumerate(sorted(pool.items()), 1):
        left, right = np.searchsorted(positions, (position, position + 1))
        if left != offset or right - left != len(online):
            raise ValueError("Model97 real-row alignment failed")
        matrix[left:right] = (prefix_parity(history, online) if number <= 3
                              else cell_glr_features(history, online))
        offset = right
        if number % 500 == 0:
            matrix.flush()
            print(f"Model97 cell cache: {number}/{len(pool)} series", flush=True)
    if offset != layout.row_count:
        raise ValueError("Incomplete Model97 cell cache")
    matrix.flush(); del matrix; partial.replace(path)
    values = np.load(path, mmap_mode="r", allow_pickle=False)[:, 0]
    pilot = np.isin(layout.row_fold, (0, 1))
    points = (0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0)
    report = {"rows": layout.row_count, "names": list(FEATURE_NAMES),
        "sha256": sha256(path), "source_sha256": sha256(HERE / "cell_glr.py"),
        "x_train_sha256": sha256(ROOT / "train/X_train.parquet"),
        "split_sha256": sha256(ROOT / "models/model5/artifacts/split_assignments.npz"),
        "label_free": True,
        "pilot_quantiles": {str(q): float(np.quantile(values[pilot], q)) for q in points},
        "pilot_std": float(np.std(values[pilot])),
        "all_quantiles": {str(q): float(np.quantile(values, q)) for q in points},
        "all_std": float(np.std(values)), "elapsed_seconds": time.monotonic() - started}
    (HERE / "cell_glr_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
