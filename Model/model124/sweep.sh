#!/bin/bash
# Model124 config sweep on the GPU box; waits for the first run to finish.
set -uo pipefail
cd /workspace/model124
while ! grep -q ALL_DONE run.log 2>/dev/null; do sleep 10; done
run () {  # tag hidden layers epochs lr batch
  for k in 0 1 2 3 4; do
    [ -f "$1fold${k}_pred.npz" ] && continue
    python -u train_gru.py --data sequences.npz --fold "$k" --hidden "$2" --layers "$3" --epochs "$4" --lr "$5" --batch "$6" --tag "$1" 2>&1 | grep -E "best|epoch (0|5|11|17|23):" | tee -a sweep.log
  done
}
run h64_   64  2 16 2e-3 48
run h96l3_ 96  3 24 1e-3 48
run h128_  128 2 24 1e-3 32
run h96d_  96  2 24 2e-3 96
echo SWEEP_DONE | tee -a sweep.log
