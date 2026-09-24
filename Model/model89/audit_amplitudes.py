"""Compare held-fold-safe whitened break amplitudes with Model89's fixed priors."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model10.sources import load_development_layout
from models.model20.features import load_real_pool
from models.model89.bayes_evidence import whitened_innovations


def correlation(values):
    x = np.asarray(values, dtype=np.float64)
    if len(x) < 3 or np.std(x[:-1]) < 1e-8 or np.std(x[1:]) < 1e-8:
        return 0.0
    value = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    return value if np.isfinite(value) else 0.0


def quantiles(values):
    x = np.asarray(values, dtype=np.float64)
    return {str(q): float(np.quantile(x, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}


def main():
    layout = load_development_layout(ROOT)
    pool = load_real_pool(layout)
    records = []
    for number, position in enumerate(sorted(pool), 1):
        tau = int(layout.tau_index[position])
        if tau < 0:
            continue
        history, online = pool[position]
        historical, current = whitened_innovations(history, online)
        post = current[tau:]
        if len(post) < 20:
            continue
        scale = abs(float(np.log(max(np.std(post), 1e-8) /
                                 max(np.std(historical), 1e-8))))
        dependence = abs(correlation(post) - correlation(historical))
        records.append((int(layout.inner_series_fold[position]), scale, dependence))
        if number % 1000 == 0:
            print(f"Model89 amplitude audit: {number}/{len(pool)} series", flush=True)
    values = np.asarray(records, dtype=np.float64)
    report = {
        "diagnostic_only": True,
        "priors_were_predeclared": True,
        "eligible_break_series": int(len(values)),
        "all": {"scale_abs_log_sigma": quantiles(values[:, 1]),
                "dependence_abs_delta_acf1": quantiles(values[:, 2])},
        "fold_excluded": {},
    }
    for fold in range(5):
        fitting = values[values[:, 0] != fold]
        report["fold_excluded"][str(fold)] = {
            "series": int(len(fitting)),
            "scale_abs_log_sigma": quantiles(fitting[:, 1]),
            "dependence_abs_delta_acf1": quantiles(fitting[:, 2]),
        }
    path = Path(__file__).resolve().parent / "amplitude_audit.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
