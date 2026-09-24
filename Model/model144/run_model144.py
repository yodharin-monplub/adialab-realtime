"""Model144 runner: same student / folds / rule as stage 4 (Model140), with the split-test features.

Arms:  B = 317 + 27 split-test columns (no selection)
       E = the same with |rho| > .95 duplicates pruned
       X = 317 + 13 stage-4 tests (Model140) + 27 split tests, pruned
Usage: python models/model144/run_model144.py ARM [ARM ...] [folds]   |   report
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import models.model140.run_model140 as m140  # noqa: E402
from models.model140.fast_tests import NAMES as NAMES140  # noqa: E402
from models.model144.split_tests import NAMES  # noqa: E402

HERE = Path(__file__).resolve().parent
WHITE = __import__("os").environ.get("SP_WHITEN", "0") == "1"   # AR-whitened variant (data_white/, runs_white/)
D144 = HERE / ("data_white" if WHITE else "data")
D140 = ROOT / "models/model140/data"
m140.RUNS = HERE / ("runs_white" if WHITE else "runs")
m140.HERE = HERE


def run(arms, folds):
    m140.RUNS.mkdir(exist_ok=True)
    _, protocol, layout, real, pool, _ = m140.m111.load_everything()
    real144 = np.load(D144 / "real.npy"); assert len(real144) == layout.row_count
    real140 = np.load(D140 / "real.npy") if "X" in arms else None
    for fold in folds:
        syn144 = np.load(D144 / f"syn_fold{fold}.npy")
        plain = m140.Fold(protocol, layout, real, pool, real144, fold, syn144, NAMES) if set(arms) & {"B", "E"} else None
        for arm in arms:
            if arm == "B":
                plain.fit("B", np.arange(317 + len(NAMES)))
            elif arm == "E":
                cols = plain.corr_prune(); print(f"E fold {fold}: {len(cols)} columns kept", flush=True); plain.fit("E", cols)
            elif arm == "X":
                both = m140.Fold(protocol, layout, real, pool, np.concatenate((real140, real144), axis=1), fold,
                                 np.concatenate((np.load(D140 / f"syn_fold{fold}.npy"), syn144), axis=1),
                                 list(NAMES140) + list(NAMES))
                cols = both.corr_prune(); print(f"X fold {fold}: {len(cols)} columns kept", flush=True); both.fit("X", cols)
                del both
        del plain


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["report"]:
        m140.report()
    else:
        run([a for a in args if not a.isdigit()], [int(a) for a in args if a.isdigit()] or [0, 1])
