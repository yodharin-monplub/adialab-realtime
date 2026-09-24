#!/usr/bin/env bash
set -euo pipefail

for attempt in {1..480}; do
    if .venv/bin/python -c 'import catboost' >/dev/null 2>&1; then
        echo "CatBoost is ready; starting Model46."
        OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
            .venv/bin/python models/model46/run_model46.py run
        exit 0
    fi
    sleep 30
done

echo "CatBoost was not available after four hours." >&2
exit 1
