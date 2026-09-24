"""Frozen stack with two specified causal correction strengths."""
import numpy as np
from scipy.special import logit
from models.model10.sources import _logistic
from models.model10.causal_transforms import TransformSpec, apply_transform_grid
from models.model20.run_model20 import ROOT, SOURCE, STACK, COMPONENT

def correction_weight(arm):
    return {'model20':.1,'model23':.1,'premium_05':.05,'premium_20':.2}[arm]

def primitive_logit(probability):
    bounded = np.clip(np.asarray(probability, dtype=np.float32), 1e-6, 1-1e-6)
    return np.log(bounded)-np.log1p(-bounded)


def replace_stack_column(original_margin, original_probability, replacement, coefficient):
    delta = primitive_logit(replacement).astype(np.float64)-primitive_logit(original_probability).astype(np.float64)
    return (np.asarray(original_margin, dtype=np.float64)+coefficient*delta).astype(np.float32)


def ensemble_margin(probability, layout, arm):
    old = np.load(SOURCE/'raw_oof.f32.npy')
    margin = np.load(STACK/'development_oof.margin.f32.npy')
    updated = np.empty_like(margin)
    for f in range(5):
        with np.load(STACK/f'pairwise_cv/fold_{f}.coefficients.npz') as coefficients:
            names = coefficients['feature_names'].tolist()
            weight = float(coefficients['raw_logit_coefficient'][names.index(COMPONENT)])
        rows = layout.row_fold == f
        updated[rows] = replace_stack_column(margin[rows], old[rows], probability[rows], weight)
    spec = TransformSpec(kind='rolling_max_premium', window=128, correction_weight=correction_weight(arm))
    _, corrected = next(apply_transform_grid(_logistic(updated).astype(np.float32),
        layout.time_online, layout.series_position, [spec]))
    compact = np.load(ROOT/'models/model11/runs/a982309521d5/oof.npy')
    return (.9*corrected+.1*logit(np.clip(compact, 1e-6, 1-1e-6))).astype(np.float32)


