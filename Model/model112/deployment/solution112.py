"""Submission30: Submission28 chain plus a top-level oracle-distilled LightGBM component.

score = (chain - mu_chain) / sd_chain + weight * (logit(p_distilled) - mu_dist) / sd_dist
p_distilled is the float32 logit bag of five Model111 fold voters, evaluated on
the same detector row and incremental whitening row as the Model73 voters.
All constants are frozen in resources/manifest.json["model112_distilled"].
"""
import math
from pathlib import Path

import numpy as np

from primitive_runtime import _bag_probabilities, _resource_path
from solution26 import logistic, logit
import solution73 as base


def validate_resources(directory):
    manifest = base.validate_resources(directory)
    spec = manifest.get('model112_distilled')
    if manifest.get('top_level_candidate') != 'model112_distilled_top_blend' or not isinstance(spec, dict):
        raise ValueError('Missing Model112 candidate contract')
    voters = spec.get('voters')
    if not isinstance(voters, list) or len(voters) not in (5, 15):
        raise ValueError('Model112 requires five (or 3x5 seed-bagged) distilled voters')
    for key in ('mu_chain', 'sd_chain', 'mu_dist', 'sd_dist', 'weight'):
        if not math.isfinite(float(spec[key])):
            raise ValueError(f'Invalid Model112 constant {key}')
    if float(spec['sd_chain']) <= 0 or float(spec['sd_dist']) <= 0:
        raise ValueError('Model112 scales must be positive')
    original = set(manifest['feature_names'])
    for voter in voters:
        if voter['path'] not in manifest['file_sha256']:
            raise ValueError('Unchecksummed Model112 voter')
        names = voter['names']
        if len(names) != 317 or names[-61:] != list(base.FEATURE_NAMES):
            raise ValueError('Changed Model112 voter feature order')
        if any(name not in original for name in names[:-61]):
            raise ValueError('Unknown Model112 detector feature')
        _resource_path(Path(directory), voter['path'])
    return manifest


class Bundle(base.Bundle):
    def __init__(self, directory):
        super().__init__(directory)
        self.manifest = validate_resources(directory)
        import lightgbm as lgb
        by_name = {name: number for number, name in enumerate(self.manifest['feature_names'])}
        spec = self.manifest['model112_distilled']
        self.distilled = []
        for entry in spec['voters']:
            model = lgb.Booster(model_file=str(_resource_path(self.root, entry['path'])))
            if model.num_feature() != len(entry['names']):
                raise ValueError('Model112 LightGBM feature count mismatch')
            columns = np.asarray([by_name[name] for name in entry['names'][:-61]], dtype=np.int32)
            self.distilled.append((model, columns))
        self.mu_chain = float(spec['mu_chain'])
        self.sd_chain = float(spec['sd_chain'])
        self.mu_dist = float(spec['mu_dist'])
        self.sd_dist = float(spec['sd_dist'])
        self.weight = float(spec['weight'])

    def distilled_probability(self, row, white_row):
        white = np.asarray(white_row, dtype=np.float32)
        probabilities = []
        for model, columns in self.distilled:
            selected = np.concatenate((row[columns], white)).astype(np.float32)[None, :]
            probabilities.append(np.float32(model.predict(selected, num_threads=1)[0]))
        return _bag_probabilities(probabilities)


class SeriesPredictor(base.SeriesPredictor):
    def step(self, value):
        value = float(value)
        row = np.asarray(self.detector.step(value), dtype=np.float32)
        np.nan_to_num(row, copy=False, nan=0., posinf=1e6, neginf=-1e6)
        np.clip(row, -1e6, 1e6, out=row)
        white = self.white.step(value)
        bundle = self.bundle
        probability = bundle.primitive_probabilities(row, white)
        primitive_margin = logit(probability).astype(np.float32)
        margin9 = np.float32(primitive_margin.astype(np.float64) @ bundle.coefficient)
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
            [model.predict(compact_row, num_threads=1)[0] for model in bundle.compact_models]))
        margin11 = np.float32(logit(p11))
        combined = float(np.float32(.9 * transformed + .1 * margin11))
        pd = float(bundle.distilled_probability(row, white))
        pd = min(max(pd, 1e-6), 1 - 1e-6)
        distilled = math.log(pd) - math.log1p(-pd)
        score = ((combined - bundle.mu_chain) / bundle.sd_chain
                 + bundle.weight * (distilled - bundle.mu_dist) / bundle.sd_dist)
        return float(logistic(score))


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
