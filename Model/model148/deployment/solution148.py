"""Submission34: Submission33 with the GRU ensemble retrained on 129 inputs (102 + 27 split-test columns).

score = z(chain) + w * z(logit student147) + w_gru * z(gru148)            (same structure as before)
gru148 = mean logit of the Model148 GRUs (numpy inference, one hidden state per series and network).
Their input = Submission31's 102 standardized inputs followed by the 27 split-test columns,
standardized with the frozen constants of resources/manifest.json["model148_gru"], clipped to +/-clip
and rounded to float16 exactly like the training export.  The split-test row is computed once per
step and shared with the student.
"""
import math
from pathlib import Path

import numpy as np

from gru_numpy import NumpyGRU
from primitive_runtime import _bag_probabilities, _resource_path
from solution26 import logistic, logit
from split_tests_numpy import NAMES as SPLIT_NAMES
import solution147 as base


def validate_resources(directory):
    manifest = base.validate_resources(directory)
    spec = manifest.get('model148_gru')
    if manifest.get('gru148_candidate') != 'model148_split_input_gru' or not isinstance(spec, dict):
        raise ValueError('Missing Model148 candidate contract')
    if not spec['folds'] or len(spec['folds']) % 5:
        raise ValueError('Model148 requires 5k fold GRUs')
    if spec['split_names'] != list(SPLIT_NAMES) or len(spec['split_mu']) != len(SPLIT_NAMES) or len(spec['split_sd']) != len(SPLIT_NAMES):
        raise ValueError('Model148 split-input contract changed')
    for key in ('mu_gru', 'sd_gru', 'weight_gru', 'clip'):
        if not math.isfinite(float(spec[key])):
            raise ValueError(f'Invalid Model148 constant {key}')
    if float(spec['sd_gru']) <= 0:
        raise ValueError('Model148 scale must be positive')
    for fold in spec['folds']:
        if fold['path'] not in manifest['file_sha256']:
            raise ValueError('Unchecksummed Model148 weights')
        _resource_path(Path(directory), fold['path'])
    return manifest


class Bundle(base.Bundle):
    def __init__(self, directory):
        super().__init__(directory)
        self.manifest = validate_resources(directory)
        spec = self.manifest['model148_gru']
        self.grus148 = [NumpyGRU(dict(np.load(_resource_path(self.root, fold['path']))), int(fold['layers']))
                        for fold in spec['folds']]
        self.split_mu = np.asarray(spec['split_mu'], dtype=np.float32)
        self.split_sd = np.asarray(spec['split_sd'], dtype=np.float32)
        self.split_clip = float(spec['clip'])
        self.mu_gru148 = float(spec['mu_gru'])
        self.sd_gru148 = float(spec['sd_gru'])
        self.weight_gru148 = float(spec['weight_gru'])

    def gru148_input(self, row, white_row, split_row, step_index):
        head = self.gru_input(row, white_row, step_index)                       # Submission31's 102 inputs
        z = np.clip((np.asarray(split_row, dtype=np.float32) - self.split_mu) / self.split_sd, -self.split_clip, self.split_clip)
        return np.concatenate((head, z.astype(np.float16).astype(np.float32)))


class SeriesPredictor(base.SeriesPredictor):
    def __init__(self, bundle, historical):
        super().__init__(bundle, historical)
        self.gru148_states = [gru.initial_state() for gru in bundle.grus148]

    def step(self, value):
        value = float(value)
        row = np.asarray(self.detector.step(value), dtype=np.float32)
        np.nan_to_num(row, copy=False, nan=0., posinf=1e6, neginf=-1e6)
        np.clip(row, -1e6, 1e6, out=row)
        white = np.asarray(self.white.step(value), dtype=np.float32)
        split_row = self.split.step(value)
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
        ps = float(bundle.student_probability(row, white, split_row))
        ps = min(max(ps, 1e-6), 1 - 1e-6)
        student = math.log(ps) - math.log1p(-ps)
        x = bundle.gru148_input(row, white, split_row, index)
        gru_logit = float(np.mean([gru.step(x, state) for gru, state in zip(bundle.grus148, self.gru148_states)],
                                  dtype=np.float32))
        score = ((combined - bundle.mu_chain) / bundle.sd_chain
                 + bundle.weight_student * (student - bundle.mu_student) / bundle.sd_student
                 + bundle.weight_gru148 * (gru_logit - bundle.mu_gru148) / bundle.sd_gru148)
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
