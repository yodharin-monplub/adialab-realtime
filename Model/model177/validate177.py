"""Model177: run Submission43 (lattice gate) and Submission42 on the local test set and compare.

Usage: python Model/model177/validate177.py [N] [--ids 10002,10007]
  * every series whose history is not a lattice must score bit-identically in both bundles;
  * the lattice series (locally: ids found by scan177.py test) must switch to 1.0 at the break;
  * TS-AUC of both bundles on the chosen series is printed (y_test.reduced is the local truth).
Each bundle is imported from submissions/<name> exactly as the platform does (main.train, main.infer),
with the online values fed one at a time through a generator.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import time
from pathlib import Path

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / 'Data' / 'ADIA Lab Structural Break Challenge Real-Time Edition'
sys.path.insert(0, str(ROOT / 'Other'))
from tsauc import tsauc  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith('--')]
N = int(args[0]) if args else 100
MUST = []
if '--ids' in sys.argv:
    MUST = [int(x) for x in sys.argv[sys.argv.index('--ids') + 1].split(',')]


def load():
    X = pd.read_parquet(DATA / 'X_test.reduced.parquet')
    Y = pd.read_parquet(DATA / 'y_test.reduced.parquet')
    tau = pd.read_parquet(DATA / 'y_test_index.reduced.parquet')
    ids = list(dict.fromkeys(X.index.get_level_values(0)))
    chosen = [i for i in MUST if i in ids] + [i for i in ids if i not in MUST]
    chosen = chosen[:max(N, len(MUST))]
    datasets, targets, steps, lengths = [], [], [], []
    for sid in chosen:
        g = X.loc[sid]
        history = g['value'][g['period'] == 1].to_numpy(np.float64).tolist()
        online = g['value'][g['period'] == 2].to_numpy(np.float64).tolist()
        y = Y.loc[sid]['target'].to_numpy()
        assert len(y) == len(online)
        datasets.append((history, online))
        targets.append(y)
        steps.append(np.arange(len(online)))
        lengths.append(len(online))
    return chosen, datasets, np.concatenate(targets), np.concatenate(steps), lengths, tau


def run_bundle(name, datasets):
    bundle = ROOT / 'submissions' / name
    for module in [m for m in sys.modules if m.startswith(('solution', 'main', 'lattice_gate', 'split_tests_numpy',
                                                           'gru_numpy', 'primitive_runtime', 'feature_pipeline',
                                                           'compact_', 'model73_', 'incremental_', 'pairing',
                                                           'detector'))]:
        sys.modules.pop(module)
    sys.path.insert(0, str(bundle))
    try:
        main = importlib.import_module('main')
        main.train(iter(()), str(bundle / 'resources'))

        def stream():
            for history, online in datasets:
                yield history, iter(online)

        t0 = time.perf_counter()
        out = [p for p in main.infer(stream(), str(bundle / 'resources')) if p is not None]
        return np.asarray(out, np.float64), time.perf_counter() - t0
    finally:
        sys.path.remove(str(bundle))


def main():
    chosen, datasets, target, step, lengths, tau = load()
    print(f'{len(chosen)} local test series, {sum(lengths)} online steps')
    sys.path.insert(0, str(HERE))
    from lattice_gate import fit_lattice
    lattice = [sid for sid, (h, _) in zip(chosen, datasets) if fit_lattice(h) is not None]
    print('lattice series among them:', lattice)
    out = {}
    for name in ('submission42', 'submission43'):
        out[name], seconds = run_bundle(name, datasets)
        assert len(out[name]) == sum(lengths), (name, len(out[name]), sum(lengths))
        print(f'{name}: {seconds:.1f} s, TS-AUC {tsauc(target, out[name], step):.6f}')
    offsets = np.cumsum([0] + lengths)
    report = {'ids': chosen, 'lattice': lattice, 'tsauc42': tsauc(target, out['submission42'], step),
              'tsauc43': tsauc(target, out['submission43'], step), 'series': {}}
    identical = 0
    for k, sid in enumerate(chosen):
        a = out['submission42'][offsets[k]:offsets[k + 1]]
        b = out['submission43'][offsets[k]:offsets[k + 1]]
        if sid not in lattice:
            if not np.array_equal(a, b):
                raise SystemExit(f'series {sid}: non-lattice scores differ (max {np.max(np.abs(a - b))})')
            identical += 1
            continue
        tp = int(tau.loc[sid, 'tau_index'])
        first_one = int(np.argmax(b >= 1.0)) if np.any(b >= 1.0) else -1
        report['series'][sid] = {'tau_pos': tp, 'first_1.0': first_one, 'n': int(lengths[k]),
                                 'max_before': float(b[:first_one].max()) if first_one > 0 else None}
        print(f'  lattice series {sid}: tau_pos {tp}, gate fires at {first_one}, '
              f'scores before: max {report["series"][sid]["max_before"]}, Submission42 at tau: {a[tp] if tp >= 0 else None}')
    print(f'{identical} non-lattice series bit-identical between the bundles')
    (HERE / f'validation_{len(chosen)}.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
