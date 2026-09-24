#!/usr/bin/env bash
# Durable end-to-end refresh for the 10,000-series training dataset.
#
# Stage 1 builds all six cache blocks with resumable complete-series shards.
# Stage 2 performs the fresh Model 5 TSAUC detector screen. Stage 3 runs Model
# 6 development selection, freezes the deployment contract, predicts the outer
# holdout without labels, then opens that holdout exactly once for TSAUC/SE.

set -euo pipefail

PROJECT_ROOT="/home/hp/resources/structural-break-real-time-minimum-mantis"
LOG_PATH="$PROJECT_ROOT/models/model6/refresh_10000.log"
LOCK_PATH="$PROJECT_ROOT/models/model6/refresh_10000.lock"

# Prefer the repository environment documented in AGENTS.md.  An explicit
# override remains available for another machine; the system interpreter is a
# final fallback instead of relying on a user venv that may no longer exist.
if [[ -n "${MODEL6_PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="$MODEL6_PYTHON_BIN"
elif [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
else
    PYTHON_BIN="$(command -v python3)"
fi

cd "$PROJECT_ROOT"
exec 9>"$LOCK_PATH"
if ! flock -n 9; then
    echo "A Model 6 refresh already owns $LOCK_PATH" >&2
    exit 1
fi
exec >>"$LOG_PATH" 2>&1

export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/matplotlib-model6-refresh

echo "[$(date --iso-8601=seconds)] START 10,000-series Model 6 refresh"
echo "Python: $PYTHON_BIN"
"$PYTHON_BIN" -c "import catboost, lightgbm, sklearn, xgboost"
"$PYTHON_BIN" -u models/model6/build_detector_caches.py
echo "[$(date --iso-8601=seconds)] CACHE BUILD COMPLETE"
"$PYTHON_BIN" -u models/model6/run_model6.py run-all
echo "[$(date --iso-8601=seconds)] MODEL 6 REFRESH COMPLETE"
