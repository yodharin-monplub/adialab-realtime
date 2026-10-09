"""Model177: scan the training (or local test) set with lattice_gate and evaluate the trigger rules.

Usage: python Model/model177/scan177.py [train|test]
Writes Model/model177/scan_<split>.csv (one row per lattice series) and prints the summary that is
copied into readme.txt.  The gate is driven exactly as in the bundle: fit on the history, then one
observe() per online value in order.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / 'Data' / 'ADIA Lab Structural Break Challenge Real-Time Edition'
sys.path.insert(0, str(HERE))
from lattice_gate import LatticeGate, fit_lattice, on_grid  # noqa: E402

split = sys.argv[1] if len(sys.argv) > 1 else 'train'
xname, yname = {'train': ('X_train.parquet', 'y_train_index.parquet'),
                'test': ('X_test.reduced.parquet', 'y_test_index.reduced.parquet')}[split]
X = pd.read_parquet(DATA / xname)
tau_index = pd.read_parquet(DATA / yname)

rows = []
n_series = 0
for sid, g in X.groupby(level=0, sort=False):
    n_series += 1
    values = g['value'].to_numpy(dtype=np.float64)
    period = g['period'].to_numpy()
    history, online = values[period == 1], values[period == 2]
    t_online = g.index.get_level_values(1).to_numpy()[period == 2]
    tau = int(tau_index['tau'].get(sid, -1))
    tau_pos = int(tau_index['tau_index'].get(sid, -1))
    gates = {rule: LatticeGate(history, rule=rule) for rule in ('first', 'burst', 'auto')}
    if not gates['auto'].active:
        continue
    flags = []
    trigger = {}
    for i, value in enumerate(online):
        for rule, gate in gates.items():
            was = gate.triggered
            flag = gate.observe(value)
            if gate.triggered and not was:
                trigger[rule] = i
        flags.append(flag)
    flags = np.asarray(flags)
    fit = gates['auto'].fit
    row = dict(id=sid, step=fit['step'], step_ulps=fit['step_ulps'], div=fit['div'], fraction=fit['fraction'],
               chance=fit['chance'], history_off=gates['auto'].history_off,
               history_decidable=gates['auto'].history_decidable, n_online=len(online), tau_pos=tau_pos, tau=tau,
               online_off=int(np.nansum(flags == 0)), online_undecidable=int(np.isnan(flags).sum()))
    for rule in gates:
        row['trig_' + rule] = trigger.get(rule, -1)
    if tau_pos >= 0:
        post = flags[tau_pos:]
        row['post_off_rate'] = float(np.nanmean(post == 0)) if np.isfinite(post).any() else np.nan
        pre = flags[:tau_pos]
        row['pre_off_rate'] = float(np.nanmean(pre == 0)) if np.isfinite(pre).any() else np.nan
    rows.append(row)

table = pd.DataFrame(rows).set_index('id')
table.to_csv(HERE / f'scan_{split}.csv')
print(f'{split}: {n_series} series, {len(table)} lattice series ({len(table) / n_series:.2%}); '
      f'breaks {int((table.tau >= 0).sum())}, no break {int((table.tau < 0).sum())}')
print('step_ulps quantiles', table.step_ulps.quantile([0, .1, .5, .9, 1]).round(0).to_dict())
print('unclean histories (history_off > 0):', int((table.history_off > 0).sum()))
nb, br = table[table.tau < 0], table[table.tau >= 0]
for rule in ('first', 'burst', 'auto'):
    trig = 'trig_' + rule
    exact = int((br[trig] == br.tau_pos).sum())
    late5 = int(((br[trig] > br.tau_pos) & (br[trig] <= br.tau_pos + 5)).sum())
    later = int((br[trig] > br.tau_pos + 5).sum())
    early = int(((br[trig] >= 0) & (br[trig] < br.tau_pos)).sum())
    never = int((br[trig] < 0).sum())
    false_alarm = int((nb[trig] >= 0).sum())
    delay = (br[trig] - br.tau_pos)[(br[trig] >= br.tau_pos)]
    print(f'rule {rule:5s}: breaks exact {exact}/{len(br)}, late<=5 {late5}, later {later}, early {early}, '
          f'never {never}; no-break false alarms {false_alarm}/{len(nb)}; mean delay {delay.mean():.2f}')
print('\nbreak series not exact under rule auto:')
bad = br[br.trig_auto != br.tau_pos]
print(bad[['step_ulps', 'fraction', 'history_off', 'n_online', 'tau_pos', 'trig_first', 'trig_burst', 'trig_auto',
           'pre_off_rate', 'post_off_rate']].round(4).to_string())
print('\nno-break series with any online off-grid value:')
print(nb[nb.online_off > 0][['step_ulps', 'fraction', 'history_off', 'n_online', 'online_off', 'trig_auto']].to_string())
