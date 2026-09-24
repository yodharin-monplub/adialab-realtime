"""Submission31: Submission30 plus a causal GRU student component (Model124).

score = z(chain) + w_dist * z(distilled) + w_gru * z(gru)
gru = mean over five fold GRUs of the per-step logit; each GRU keeps its own
hidden state per series (O(1) per step, deterministic float32 numpy).
Inputs per step: 61 whitening columns + 40 detector columns + log1p(t),
standardized with the training constants and rounded to float16 exactly as
the training export was.  Constants in resources/manifest.json["model124_gru"].
"""
import math
from pathlib import Path

import numpy as np

from gru_numpy import NumpyGRU
from model73_whitening import FEATURE_NAMES
from primitive_runtime import _resource_path
from solution26 import logistic
import solution112 as base


def validate_resources(directory):
    manifest = base.validate_resources(directory)
    spec = manifest.get('model124_gru')
    if manifest.get('gru_candidate') != 'model124_gru_top_blend' or not isinstance(spec, dict):
        raise ValueError('Missing Model124 candidate contract')
    if len(spec['folds']) % 5 or any(int(f['layers']) < 1 for f in spec['folds']):
        raise ValueError('Model124 requires 5k fold GRUs with a layer count each')
    contracts = spec.get('input_contracts') or [spec['input_contract']]
    original = set(manifest['feature_names'])
    for contract in contracts:
        n = len(contract['names'])
        if n < 62 or len(contract['mu']) != n or len(contract['sd']) != n:
            raise ValueError('Model124 input contract is malformed')
        if contract['names'][:61] != list(FEATURE_NAMES) or contract['names'][-1] != 'log1p_t':
            raise ValueError('Model124 input contract order changed')
        if any(name not in original for name in contract['names'][61:-1]):
            raise ValueError('Unknown Model124 detector feature')
    if any(int(f.get('contract', 0)) >= len(contracts) for f in spec['folds']):
        raise ValueError('Model124 fold refers to a missing input contract')
    for key in ('mu_gru', 'sd_gru', 'weight_gru'):
        if not math.isfinite(float(spec[key])):
            raise ValueError(f'Invalid Model124 constant {key}')
    if float(spec['sd_gru']) <= 0:
        raise ValueError('Model124 scale must be positive')
    for fold in spec['folds']:
        if fold['path'] not in manifest['file_sha256']:
            raise ValueError('Unchecksummed Model124 weights')
        _resource_path(Path(directory), fold['path'])
    return manifest


class Bundle(base.Bundle):
    def __init__(self, directory):
        super().__init__(directory)
        self.manifest = validate_resources(directory)
        spec = self.manifest['model124_gru']
        by_name = {name: number for number, name in enumerate(self.manifest['feature_names'])}
        contracts = spec.get('input_contracts') or [spec['input_contract']]
        self.gru_contracts = [(np.asarray([by_name[name] for name in c['names'][61:-1]], dtype=np.int32),
                               np.asarray(c['mu'], dtype=np.float32), np.asarray(c['sd'], dtype=np.float32),
                               float(c['clip'])) for c in contracts]
        self.grus = [(NumpyGRU(dict(np.load(_resource_path(self.root, fold['path']))), int(fold['layers'])),
                      int(fold.get('contract', 0))) for fold in spec['folds']]
        self.mu_gru = float(spec['mu_gru'])
        self.sd_gru = float(spec['sd_gru'])
        self.weight_gru = float(spec['weight_gru'])

    def gru_input(self, row, white_row, step_index, contract=0):
        columns, mu, sd, clip = self.gru_contracts[contract]
        raw = np.concatenate((np.asarray(white_row, dtype=np.float32),
                              np.asarray(row, dtype=np.float32)[columns],
                              np.asarray([math.log1p(step_index)], dtype=np.float32)))
        np.nan_to_num(raw, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
        z = np.clip((raw - mu) / sd, -clip, clip)
        return z.astype(np.float16).astype(np.float32)

    def gru_inputs(self, row, white_row, step_index):
        return [self.gru_input(row, white_row, step_index, i) for i in range(len(self.gru_contracts))]


class SeriesPredictor(base.SeriesPredictor):
    def __init__(self, bundle, historical):
        super().__init__(bundle, historical)
        self.gru_states = [gru.initial_state() for gru, _ in bundle.grus]

    def step(self, value):
        value = float(value)
        row = np.asarray(self.detector.step(value), dtype=np.float32)
        np.nan_to_num(row, copy=False, nan=0., posinf=1e6, neginf=-1e6)
        np.clip(row, -1e6, 1e6, out=row)
        white = self.white.step(value)
        bundle = self.bundle
        from solution26 import logit
        from primitive_runtime import _bag_probabilities
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
        xs = bundle.gru_inputs(row, white, index)
        gru_logit = float(np.mean([gru.step(xs[c], state) for (gru, c), state in zip(bundle.grus, self.gru_states)],
                                  dtype=np.float32))
        score = ((combined - bundle.mu_chain) / bundle.sd_chain
                 + bundle.weight * (distilled - bundle.mu_dist) / bundle.sd_dist
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
