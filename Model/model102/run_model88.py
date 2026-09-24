"""Bind Model88's frozen extension engine to Model102's scale GLR feature."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model11.pipeline import sha256
from models.model88 import run_model88 as engine
from models.model102 import short_whitening as feature


HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "prespec.json").read_text())
if sha256(ROOT / "models/model89/bayes_evidence.py") != CONFIG["whitener_source_sha256"]:
    raise ValueError("Changed Model102 whitening source")

# Model88 was intentionally written as a generic one-block feature-extension
# runner. Bind its module globals before entering the engine; every function
# resolves these names at call time.
engine.HERE = HERE
engine.CONFIG = CONFIG
engine.PARENT20 = ROOT / CONFIG["model20_run"]
engine.PARENT57 = ROOT / CONFIG["model57_run"]
engine.PARENT62 = ROOT / CONFIG["model62_run"]
engine.PARENT67 = ROOT / CONFIG["model67_run"]
engine.PARENT84 = ROOT / CONFIG["model84_run"]
engine.ARM = "expressive_scale_glr"
engine.FEATURE_NAMES = feature.FEATURE_NAMES
engine.WINDOW = feature.WINDOW
engine.prefix_parity = feature.prefix_parity
engine.short_whitening_features = feature.short_whitening_features


if __name__ == "__main__":
    engine.main()
