"""Exact 85/15 Model 16/6 blend with shared primitive computation per point."""
from pathlib import Path
from copy import copy
import hashlib
import json

import numpy as np
from scipy.special import logit
from solution16 import Bundle as Bundle16, SeriesPredictor as Predictor16, logistic
from solution16 import validate_resources as validate_model16
from primitive_runtime import LoadedBundle, SeriesPredictor as Predictor6, _resource_path
from compact_runtime import load_compact_linear


def validate_resources(directory):
    root = Path(directory)
    manifest = json.loads((root / "ensemble17.json").read_text())
    if manifest.get("submission") != "submission_17" or manifest.get("weights") != [0.85, 0.15]:
        raise ValueError("Unexpected frozen Model 17 recipe")
    if hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest() != manifest["model16_manifest_sha256"]:
        raise ValueError("Model 16 resource manifest changed")
    base = validate_model16(root)
    old = manifest["model6"]
    if base["union_banks"] != old["union_banks"] or base["detector_ensemble"]["bank_keys"] != old["detector_ensemble"]["bank_keys"]:
        raise ValueError("Shared bank ordering differs")
    if base["feature_components"] != old["feature_components"]:
        raise ValueError("Shared feature models differ")
    for name, expected in old["file_sha256"].items():
        if hashlib.sha256(_resource_path(root, name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Model 6 resource hash mismatch: {name}")
    return manifest


class Bundle:
    def __init__(self, directory):
        import lightgbm as lgb
        self.manifest = validate_resources(directory)
        self.base = Bundle16(directory)
        root = Path(directory)
        old = self.manifest["model6"]
        specs = old["meta_models"]
        linear = load_compact_linear(_resource_path(root, specs["core_linear_l2"]["path"]))
        meta = lgb.Booster(model_file=str(_resource_path(root, specs["candidate_lightgbm_gbdt"]["path"])))
        order = {family: tuple(spec["bank_order"]) for family, spec in old["detector_families"].items()}
        if linear.feature_count != len(order["core"]) + 6 or meta.num_feature() != len(order["candidate"]) + 6:
            raise ValueError("Model 6 meta feature count changed")
        self.old = LoadedBundle(root=root, manifest=old, union_banks=self.base.banks,
            bank_order=order, detector_ensemble=self.base.detectors,
            features={feature.name: feature for feature in self.base.features}, linear_meta=linear,
            lightgbm_meta=meta, lightgbm_meta_iteration=specs["candidate_lightgbm_gbdt"]["best_iteration"],
            component_specs=tuple(old["components"]), stream_detector_class=self.base.detector_class)


class SharedFrame:
    """Per-series, per-point cache; no state is shared across different series."""
    def __init__(self, base, historical):
        self.base = base
        self.detector = base.detector_class(np.asarray(historical, dtype=np.float64), base.banks)
        self.bank_count = base.detectors.bank_count
        self.row = self.margins = None
        self.feature_values = {}

    def begin(self, value):
        self.value = float(value)
        row = np.asarray(self.detector.step(self.value), dtype=np.float32)
        np.nan_to_num(row, copy=False, nan=0., posinf=1e6, neginf=-1e6)
        np.clip(row, -1e6, 1e6, out=row)
        self.row = row
        self.margins = self.base.detectors.predict_fold_margins(row)
        self.feature_values.clear()

    def step(self, value):
        if self.row is None or float(value) != self.value:
            raise RuntimeError("Shared frame is not ready for this observation")
        return self.row

    def predict_fold_margins(self, row):
        if self.margins is None:
            raise RuntimeError("Shared detector margins are not ready")
        return self.margins


class CachedFeature:
    def __init__(self, frame, feature):
        self.frame, self.feature, self.name = frame, feature, feature.name

    def predict(self, row):
        if self.name not in self.frame.feature_values:
            self.frame.feature_values[self.name] = self.feature.predict(row)
        return self.frame.feature_values[self.name]


class SeriesPredictor:
    def __init__(self, bundle, historical):
        self.frame = SharedFrame(bundle.base, historical)
        frame = self.frame
        base = copy(bundle.base)
        base.detector_class = lambda history, banks: frame
        base.detectors = self.frame
        base.features = [CachedFeature(self.frame, feature) for feature in bundle.base.features]
        old = copy(bundle.old)
        old.stream_detector_class = base.detector_class
        old.detector_ensemble = self.frame
        old.features = {feature.name: feature for feature in base.features}
        self.model16 = Predictor16(base, historical)
        self.model6 = Predictor6(old, historical)
        self.last_model16 = self.last_model6 = None

    def step(self, value):
        self.frame.begin(value)
        p16 = self.model16.step(value)
        p6 = self.model6.step(value)
        self.last_model16, self.last_model6 = p16, p6
        # Model 16 emits sigmoid(float32 margin). Inversion recovers that margin
        # before the same final float32 weighted-sum boundary used in the study.
        margin16 = np.float32(logit(p16))
        margin6 = np.float32(logit(np.clip(np.float32(p6), 1e-6, 1-1e-6)))
        combined = np.float32(.85 * float(margin16) + .15 * float(margin6))
        return float(logistic(float(combined)))


def infer(datasets, model_directory_path):
    bundle = Bundle(model_directory_path)
    yield None
    for item in datasets:
        if len(item) == 2:
            historical, online = item
        elif len(item) == 3:
            _, historical, online = item
        else:
            raise ValueError("Expected (history, online) or (id, history, online)")
        predictor = SeriesPredictor(bundle, historical)
        for value in online:
            yield predictor.step(value)
