"""Deployment of the frozen Model 16 recipe; no training or label access."""
from collections import deque
from pathlib import Path
import hashlib
import json

import numpy as np

from primitive_runtime import (NativeDetectorEnsemble, LoadedFeatureComponent, LoadedFeatureVoter,
                               _bag_probabilities, _bag_probability_rows, _resource_path)
from compact_runtime import load_compact_forest_bag


def logistic(value):
    return 1 / (1 + np.exp(-np.clip(value, -35, 35)))


def logit(value, epsilon=1e-6):
    bounded = np.clip(value, epsilon, 1-epsilon)
    return np.log(bounded)-np.log1p(-bounded)


def validate_resources(directory):
    root = Path(directory)
    manifest = json.loads((root/"manifest.json").read_text())
    if manifest.get("submission") != "submission_16" or manifest.get("recipe") != "0.81*m9 + 0.09*rolling_max128(m9) + 0.10*m11":
        raise ValueError("Invalid Submission 16 manifest")
    for name, expected in manifest["file_sha256"].items():
        path = _resource_path(root, name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Resource checksum mismatch: {name}")
    return manifest


class Bundle:
    def __init__(self, directory):
        import lightgbm as lgb
        import xgboost as xgb
        from feature_pipeline import CoreStreamDetector, feature_names
        from compact_features import FEATURE_NAMES, StreamingFeatures

        self.root = Path(directory)
        self.manifest = validate_resources(directory)
        self.banks = tuple(self.manifest["union_banks"])
        names = feature_names(self.banks)
        if names != self.manifest["feature_names"]:
            raise ValueError("Detector feature contract mismatch")
        self.detector_class, self.compact_class = CoreStreamDetector, StreamingFeatures
        spec = self.manifest["detector_ensemble"]
        model = lgb.Booster(model_file=str(_resource_path(self.root, spec["model_path"])))
        with np.load(_resource_path(self.root, spec["lookup_path"]), allow_pickle=False) as p:
            self.detectors = NativeDetectorEnsemble(
                feature_count=int(p["feature_count"]), voters_per_bank=int(p["voters_per_bank"]),
                bank_count=int(p["bank_count"]), model=model, group_offsets=p["group_offsets"],
                leaf_offsets=p["leaf_offsets"], leaf_values=p["leaf_values"])
        by_name = {name:i for i,name in enumerate(names)}
        features = {}
        for spec in self.manifest["feature_components"]:
            if spec["kind"] == "forest":
                forest = load_compact_forest_bag(_resource_path(self.root, spec["path"]))
                features[spec["name"]] = LoadedFeatureComponent(spec["name"], "forest", forest_bag=forest)
            else:
                voters = []
                for voter in spec["voters"]:
                    model = xgb.Booster(params={"nthread":1})
                    model.load_model(_resource_path(self.root, voter["path"]))
                    model.set_param({"nthread":1, "device":"cpu"})
                    columns = np.asarray([by_name[n] for n in voter["feature_names"]], dtype=np.int32)
                    voters.append(LoadedFeatureVoter(model, columns))
                features[spec["name"]] = LoadedFeatureComponent(spec["name"], "xgboost", voters=tuple(voters))
        self.features = [features[name] for name in self.manifest["stack_feature_components"]]
        with np.load(_resource_path(self.root, "stack_coefficients.npz"), allow_pickle=False) as p:
            if p["feature_names"].tolist() != self.manifest["stack_feature_names"]:
                raise ValueError("Stack feature order mismatch")
            self.coefficient = p["raw_logit_coefficient"].astype(np.float64)
        if self.coefficient.shape != (self.detectors.bank_count+len(self.features),):
            raise ValueError("Stack dimensions mismatch")
        self.compact_models = [lgb.Booster(model_file=str(_resource_path(self.root, name)))
                               for name in self.manifest["compact_models"]]
        if any(m.feature_name() != FEATURE_NAMES for m in self.compact_models):
            raise ValueError("Compact model feature contract mismatch")

    def primitive_probabilities(self, row):
        margins = self.detectors.predict_fold_margins(row)
        banks = _bag_probability_rows(logistic(margins))
        components = [feature.predict(row) for feature in self.features]
        return np.r_[banks, components].astype(np.float32)


class SeriesPredictor:
    def __init__(self, bundle, historical):
        self.bundle = bundle
        self.detector = bundle.detector_class(np.asarray(historical, dtype=np.float64), bundle.banks)
        self.compact = bundle.compact_class(historical)
        self.peaks = deque()
        self.step_index = 0

    def step(self, value):
        row = np.asarray(self.detector.step(float(value)), dtype=np.float32)
        np.nan_to_num(row, copy=False, nan=0., posinf=1e6, neginf=-1e6)
        np.clip(row, -1e6, 1e6, out=row)
        probability = self.bundle.primitive_probabilities(row)
        # Model 9's OOF loader and Model 16 use these float32 boundaries.
        primitive_margin = logit(probability).astype(np.float32)
        margin9 = np.float32(primitive_margin.astype(np.float64) @ self.bundle.coefficient)
        p9 = np.float32(logistic(float(margin9)))
        current = float(logit(float(p9), epsilon=1e-7))
        index = self.step_index
        self.step_index += 1
        while self.peaks and self.peaks[0][0] <= index-128:
            self.peaks.popleft()
        while self.peaks and self.peaks[-1][1] <= current:
            self.peaks.pop()
        self.peaks.append((index,current))
        transformed = np.float32(current + .1*(self.peaks[0][1]-current))
        compact_row = self.compact.update(float(value))[None,:]
        p11 = np.float32(_bag_probabilities([m.predict(compact_row, num_threads=1)[0] for m in self.bundle.compact_models]))
        margin11 = np.float32(logit(p11))
        combined = np.float32(.9*transformed + .1*margin11)
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
