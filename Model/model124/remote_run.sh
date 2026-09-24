#!/bin/bash
# Model124 remote runner: train the GRU student for all five folds on the GPU box.
# Usage (on the instance): bash remote_run.sh [epochs] [hidden]
set -euo pipefail
cd /workspace/model124
EPOCHS=${1:-8}
HIDDEN=${2:-96}
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
for k in 0 1 2 3 4; do
  if [ -f "fold${k}_pred.npz" ]; then echo "fold $k done"; continue; fi
  python -u train_gru.py --data sequences.npz --fold "$k" --epochs "$EPOCHS" --hidden "$HIDDEN" 2>&1 | tee -a "train_fold${k}.log"
done
echo ALL_DONE
