#!/bin/bash
# Model129 stage 2b on the GPU box: 15 nested GRUs (k held, j training fold).
# Needs: train_gru.py tsauc.py train_gru_nested.py sequences.npz syn_fold{0..4}.npz
set -uo pipefail
cd /workspace/model129
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
for k in 0 1 2 3 4; do
  k1=$(( (k + 1) % 5 ))
  for j in 0 1 2 3 4; do
    if [ "$j" = "$k" ] || [ "$j" = "$k1" ]; then continue; fi
    if [ -f "nested_k${k}_j${j}.npz" ]; then echo "k$k j$j done"; continue; fi
    python -u train_gru_nested.py --k "$k" --j "$j" --hidden 96 --layers 3 --epochs 24 --lr 1e-3 2>&1 | tee -a nested.log
  done
done
echo NESTED_DONE | tee -a nested.log
