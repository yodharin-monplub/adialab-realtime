"""Submission32: Submission31 with the distilled student replaced by the Model147 student.

score = z(chain) + w * z(logit student147) + w_gru * z(gru)          (same structure as Submission31)
student147 = logit bag of five fold LightGBM voters trained on
  * the fold's pruned columns out of: 256 detector columns + 61 whitening columns + 27 split-test columns
    (two-sample tests between the clean reference and the post segment at an ESTIMATED break, on
    AR-whitened values - models/model144, the user's idea),
  * the usual rows plus "move the start" re-cuts of the training series (models/model147).
The 27 split-test columns are computed causally per step by split_tests_numpy.SplitTests (pure numpy,
bit-identical to the numba code that built the training rows).  Constants are frozen in
resources/manifest.json["model147_student"].
"""
import math
from pathlib import Path

import numpy as np

from model73_whitening import FEATURE_NAMES
from primitive_runtime import _bag_probabilities, _resource_path
from solution26 import logistic, logit
from split_tests_numpy import NAMES as SPLIT_NAMES, SplitTests
import solution124 as base


def validate_resources(directory):
    manifest = base.validate_resources(directory)
    spec = manifest.get('model147_student')
    if manifest.get('student_candidate') != 'model147_recut_split_student' or not isinstance(spec, dict):
        raise ValueError('Missing Model147 candidate contract')
    voters = spec.get('voters')
    if not isinstance(voters, list) or not voters or len(voters) % 5:
        raise ValueError('Model147 requires five fold voters per bagged student')
    for key in ('mu_dist', 'sd_dist', 'weight'):
        if not math.isfinite(float(spec[key])):
            raise ValueError(f'Invalid Model147 constant {key}')
    if float(spec['sd_dist']) <= 0:
        raise ValueError('Model147 scale must be positive')
    known = set(manifest['feature_names']) | set(FEATURE_NAMES) | set(SPLIT_NAMES)
    for voter in voters:
        if voter['path'] not in manifest['file_sha256']:
            raise ValueError('Unchecksummed Model147 voter')
        if any(name not in known for name in voter['names']):
            raise ValueError('Unknown Model147 feature')
        _resource_path(Path(directory), voter['path'])
    return manifest


class Bundle(base.Bundle):
    def __init__(self, directory):
        super().__init__(directory)
        self.manifest = validate_resources(directory)
        import lightgbm as lgb
        spec = self.manifest['model147_student']
        detector = {name: i for i, name in enumerate(self.manifest['feature_names'])}
        white = {name: i for i, name in enumerate(FEATURE_NAMES)}
        split = {name: i for i, name in enumerate(SPLIT_NAMES)}
        self.students = []
        for entry in spec['voters']:
            model = lgb.Booster(model_file=str(_resource_path(self.root, entry['path'])))
            names = entry['names']
            if model.num_feature() != len(names):
                raise ValueError('Model147 LightGBM feature count mismatch')
            # training order: detector block, whitening block, split-test block
            blocks = [[n for n in names if n in split], [n for n in names if n in white and n not in split]]
            det_names = [n for n in names if n not in split and n not in white]
            if det_names + blocks[1] + blocks[0] != list(names):
                raise ValueError('Model147 voter column order changed')
            self.students.append((model, np.asarray([detector[n] for n in det_names], dtype=np.int32),
                                  np.asarray([white[n] for n in blocks[1]], dtype=np.int32),
                                  np.asarray([split[n] for n in blocks[0]], dtype=np.int32)))
        self.mu_student = float(spec['mu_dist'])
        self.sd_student = float(spec['sd_dist'])
        self.weight_student = float(spec['weight'])

    def student_probability(self, row, white_row, split_row):
        probabilities = []
        for model, det, wh, sp in self.students:
            selected = np.concatenate((row[det], white_row[wh], split_row[sp])).astype(np.float32)[None, :]
            probabilities.append(np.float32(model.predict(selected, num_threads=1)[0]))
        return _bag_probabilities(probabilities)


class SeriesPredictor(base.SeriesPredictor):
    def __init__(self, bundle, historical):
        super().__init__(bundle, historical)
        self.split = SplitTests(np.asarray(historical, dtype=np.float64))

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
        x = bundle.gru_input(row, white, index)
        gru_logit = float(np.mean([gru.step(x, state) for gru, state in zip(bundle.grus, self.gru_states)],
                                  dtype=np.float32))
        score = ((combined - bundle.mu_chain) / bundle.sd_chain
                 + bundle.weight_student * (student - bundle.mu_student) / bundle.sd_student
                 + bundle.weight_gru * (gru_logit - bundle.mu_gru) / bundle.sd_gru)
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
