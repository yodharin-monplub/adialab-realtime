#!/bin/bash
# Model131 on the GPU box: 15 bidirectional-GRU teachers (k held, j target fold).
# Needs: train_gru.py tsauc.py train_teacher_bigru.py sequences.npz
set -uo pipefail
cd /workspace/model131
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
for k in 0 1 2 3 4; do
  k1=$(( (k + 1) % 5 ))
  for j in 0 1 2 3 4; do
    if [ "$j" = "$k" ] || [ "$j" = "$k1" ]; then continue; fi
    if [ -f "teacher_k${k}_j${j}.npz" ]; then echo "k$k j$j done"; continue; fi
    python -u train_teacher_bigru.py --k "$k" --j "$j" --hidden 96 --layers 2 --epochs 12 --lr 1e-3 2>&1 | grep -E "best|epoch (0|5|11):" | tee -a teacher.log
  done
done
echo TEACHER_DONE | tee -a teacher.log
