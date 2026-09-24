"""Score student voter sets on the sealed holdout OFFLINE from the saved components (holdout_components.py).

final = sigmoid( z(chain) + w * z(student logit-bag) + w_gru * z(gru) ), all constants as in
models/model147/holdout_eval147.py (Submission31 bundle + a models/model112 constants file).
A voter set = runs/{arm}_fold{k}.txt/.npz of Model147-style arms (column list in the npz).
Usage (python models/model152/offline_holdout.py ...):
  check                                   parity vs holdout_predictions_BAG3.npz on the finished series
  score TAG CONST ARMDIR:ARM[,ARM...] ... [--alpha-sd]   e.g. score BAG3 constants_m147_bag3.json model147:SRE40,SRE40b,SRE40c
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
H = HERE / 'holdout'
BUNDLE = ROOT / 'submissions/submission31'
sys.path.insert(0, str(BUNDLE)); sys.path.insert(1, str(ROOT)); sys.path.insert(2, str(ROOT / 'models/model144'))
PROTOCOL = ROOT / 'models/model84/runs/e1e98a6e8af2df25/protocol.json'


def bundle_constants():
    import solution124
    b = solution124.Bundle(BUNDLE / 'resources')
    return dict(weight=float(b.weight), weight_gru=float(b.weight_gru), mu_chain=float(b.mu_chain), sd_chain=float(b.sd_chain),
                mu_gru=float(b.mu_gru), sd_gru=float(b.sd_gru))


def x_columns():
    from model73_whitening import FEATURE_NAMES
    from split_tests import NAMES
    det = [n for n in json.loads(PROTOCOL.read_text())['contract']['union_names'] if not n.startswith('white_')]
    return {n: i for i, n in enumerate(det + list(FEATURE_NAMES) + list(NAMES))}


def student_logit(X, voters):
    """voters: list of (model_txt_path, npz_path, fold).  Logit of the logit-mean bag probability."""
    import lightgbm as lgb
    from split_tests import NAMES
    protocol = json.loads(PROTOCOL.read_text())
    index = x_columns()
    z = np.zeros(len(X), np.float64)
    for txt, npz, fold in voters:
        names = list(protocol['contract']['names'][fold]) + list(NAMES)
        saved = np.load(npz)
        cols = saved['columns'] if 'columns' in saved.files else np.arange(len(protocol['contract']['names'][fold]))
        idx = np.asarray([index[names[c]] for c in cols])
        booster = lgb.Booster(model_file=str(txt)); assert booster.num_feature() == len(idx)
        p = np.zeros(len(X), np.float64)
        for lo in range(0, len(X), 200_000):
            p[lo:lo + 200_000] = booster.predict(np.asarray(X[lo:lo + 200_000][:, idx]), num_threads=8)
        p = np.clip(p.astype(np.float32).astype(np.float64), 1e-7, 1 - 1e-7)
        z += np.log(p) - np.log1p(-p)
    z /= len(voters)
    pe = np.clip(1 / (1 + np.exp(-z)), 1e-6, 1 - 1e-6)
    return np.log(pe) - np.log1p(-pe)


def voters_for(spec):
    folder, arms = spec.split(':')
    d = ROOT / 'models' / folder / 'runs'
    return [(d / f'{a}_fold{k}.txt', d / f'{a}_fold{k}.npz', k) for a in arms.split(',') for k in range(5)]


def blend(dist, const, bc):
    chain, gru = np.load(H / 'chain.npy', mmap_mode='r'), np.load(H / 'gru.npy', mmap_mode='r')
    s = ((chain - bc['mu_chain']) / bc['sd_chain'] + bc['weight'] * (dist - const['mu_dist']) / const['sd_dist']
         + bc['weight_gru'] * (gru - bc['mu_gru']) / bc['sd_gru'])
    return 1 / (1 + np.exp(-s))


def finished_rows():
    ref = np.load(ROOT / 'models/model123/holdout_predictions.npz')
    done = set(np.load(H / 'done.npy').tolist())
    starts = np.flatnonzero(ref['step'] == 0); ends = np.r_[starts[1:], len(ref['step'])]
    m = np.zeros(len(ref['step']), bool)
    for sid, a, b in zip(ref['ids'], starts, ends):
        if int(sid) in done:
            m[a:b] = True
    return m, ref


def main():
    from tsauc import tsauc
    args = sys.argv[1:]
    bc = bundle_constants()
    X = np.load(H / 'X.npy', mmap_mode='r')
    m, ref = finished_rows()
    rows = np.flatnonzero(m)
    y, t = ref['target'][rows], ref['step'][rows]
    if args[0] == 'check':
        const = json.loads((ROOT / 'models/model112/constants_m147_bag3.json').read_text())
        dist = student_logit(X[rows], voters_for('model147:SRE40,SRE40b,SRE40c'))
        chain, gru = np.load(H / 'chain.npy')[rows], np.load(H / 'gru.npy')[rows]
        s = ((chain - bc['mu_chain']) / bc['sd_chain'] + bc['weight'] * (dist - const['mu_dist']) / const['sd_dist']
             + bc['weight_gru'] * (gru - bc['mu_gru']) / bc['sd_gru'])
        ours = 1 / (1 + np.exp(-s))
        theirs = np.load(ROOT / 'models/model147/holdout_predictions_BAG3.npz')['final'][rows]
        c31 = ref['chain'][rows]
        print(f'rows {len(rows):,}; max |final - BAG3 run| {np.abs(ours - theirs).max():.2e}; '
              f'chain vs model123 chain max diff {np.abs(1 / (1 + np.exp(-chain)) - c31).max():.2e} / {np.abs(chain - c31).max():.2e}')
        return
    tag, const_file, specs = args[1], args[2], args[3:]
    const = json.loads((ROOT / 'models/model112' / const_file).read_text())
    voters = [v for s in specs for v in voters_for(s)]
    dist = student_logit(X[rows], voters)
    chain, gru = np.load(H / 'chain.npy')[rows], np.load(H / 'gru.npy')[rows]
    s = ((chain - bc['mu_chain']) / bc['sd_chain'] + bc['weight'] * (dist - const['mu_dist']) / const['sd_dist']
         + bc['weight_gru'] * (gru - bc['mu_gru']) / bc['sd_gru'])
    final = 1 / (1 + np.exp(-s))
    s31 = np.load(ROOT / 'models/model124/holdout_predictions31.npz')['final'][rows]
    bag3 = np.load(ROOT / 'models/model147/holdout_predictions_BAG3.npz')['final'][rows]
    res = dict(tag=tag, rows=int(len(rows)), student_raw=tsauc(y, dist, t), final=tsauc(y, final, t),
               sub31=tsauc(y, s31, t), bag3=tsauc(y, bag3, t))
    res['gain_vs_bag3'] = res['final'] - res['bag3']
    np.savez(HERE / f'holdout_student_{tag}.npz', dist=dist.astype(np.float32), final=final.astype(np.float32), rows=rows)
    (HERE / f'holdout_offline_{tag}.json').write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
