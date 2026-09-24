"""Model73: replace Model26's core feature voter with frozen Model62 LightGBM."""
from collections import deque
from pathlib import Path
import json

import numpy as np

from primitive_runtime import _bag_probabilities, _bag_probability_rows, _resource_path
from solution26 import Bundle as Model26Bundle, logistic, logit, validate_resources as validate_model26
from model73_whitening import FEATURE_NAMES, whitening_features


CORE = 'core__feature__xgboost__fisher_256__xgb_depth3'


def validate_resources(directory):
    root = Path(directory)
    manifest = validate_model26(root)
    spec = manifest.get('model73_white_voter')
    if manifest.get('cloud_candidate') != 'model73_direct_model62' or not isinstance(spec, dict):
        raise ValueError('Missing Model73 candidate contract')
    if spec.get('replaces') != CORE or spec.get('feature_names') != list(FEATURE_NAMES):
        raise ValueError('Changed Model73 feature contract')
    voters = spec.get('voters')
    if not isinstance(voters, list) or len(voters) != 5:
        raise ValueError('Model73 requires five LightGBM voters')
    original = set(manifest['feature_names'])
    for voter in voters:
        if voter['path'] not in manifest['file_sha256']:
            raise ValueError('Unchecksummed Model73 voter')
        names = voter['names']
        if len(names) != 317 or names[-61:] != list(FEATURE_NAMES):
            raise ValueError('Changed Model73 voter feature order')
        if any(name not in original for name in names[:-61]):
            raise ValueError('Unknown Model73 detector feature')
        _resource_path(root, voter['path'])
    return manifest


class Bundle(Model26Bundle):
    def __init__(self, directory):
        super().__init__(directory)
        self.manifest = validate_resources(directory)
        import lightgbm as lgb
        by_name = {name: number for number, name in enumerate(self.manifest['feature_names'])}
        spec = self.manifest['model73_white_voter']
        self.core_index = self.manifest['stack_feature_components'].index(CORE)
        self.white_voters = []
        for entry in spec['voters']:
            model = lgb.Booster(model_file=str(_resource_path(self.root, entry['path'])))
            if model.num_feature() != len(entry['names']):
                raise ValueError('Model73 LightGBM feature count mismatch')
            columns = np.asarray([by_name[name] for name in entry['names'][:-61]], dtype=np.int32)
            self.white_voters.append((model, columns))

    def primitive_probabilities(self, row, white_row):
        margins = self.detectors.predict_fold_margins(row)
        banks = _bag_probability_rows(logistic(margins))
        white = np.asarray(white_row, dtype=np.float32)
        if white.shape != (len(FEATURE_NAMES),) or not np.isfinite(white).all():
            raise ValueError('Invalid Model73 whitening row')
        components = []
        for index, feature in enumerate(self.features):
            if index == self.core_index:
                probabilities = []
                for model, columns in self.white_voters:
                    selected = np.concatenate((row[columns], white)).astype(np.float32)[None, :]
                    probabilities.append(np.float32(model.predict(selected, num_threads=1)[0]))
                components.append(_bag_probabilities(probabilities))
            else:
                components.append(feature.predict(row))
        return np.r_[banks, components].astype(np.float32)


class SeriesPredictor:
    def __init__(self, bundle, historical):
        self.bundle = bundle
        self.history = np.asarray(historical, dtype=np.float64)
        self.online = []
        self.detector = bundle.detector_class(self.history, bundle.banks)
        self.compact = bundle.compact_class(self.history)
        self.peaks = deque()
        self.step_index = 0

    def step(self, value):
        value = float(value)
        self.online.append(value)
        row = np.asarray(self.detector.step(value), dtype=np.float32)
        np.nan_to_num(row, copy=False, nan=0., posinf=1e6, neginf=-1e6)
        np.clip(row, -1e6, 1e6, out=row)
        white = whitening_features(self.history, np.asarray(self.online, dtype=np.float64))[-1]
        probability = self.bundle.primitive_probabilities(row, white)
        primitive_margin = logit(probability).astype(np.float32)
        margin9 = np.float32(primitive_margin.astype(np.float64) @ self.bundle.coefficient)
        p9 = np.float32(logistic(float(margin9)))
        current = float(logit(float(p9), epsilon=1e-7))
        index = self.step_index
        self.step_index += 1
        while self.peaks and self.peaks[0][0] <= index - 128:
            self.peaks.popleft()
        while self.peaks and self.peaks[-1][1] <= current:
            self.peaks.pop()
        self.peaks.append((index, current))
        transformed = np.float32(current + .1 * (self.peaks[0][1] - current))
        compact_row = self.compact.update(value)[None, :]
        p11 = np.float32(_bag_probabilities(
            [model.predict(compact_row, num_threads=1)[0] for model in self.bundle.compact_models]))
        margin11 = np.float32(logit(p11))
        combined = np.float32(.9 * transformed + .1 * margin11)
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
            raise ValueError('Expected (history, online) or (id, history, online)')
        predictor = SeriesPredictor(bundle, historical)
        for value in online:
            yield predictor.step(value)
