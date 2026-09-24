Model143 - end-to-end sequence model on the RAW series (user's question 02:13:
"have you considered more neural network more like 1d cnn lstm gru etc.").

What existed: Model124 GRU on engineered features (61 whitened + 40 detector
columns per step; in Submission31, +0.0016 holdout). Nothing had read the raw
values end to end. This is "option A" of the stage-5 mail.

export_raw.py -> data/raw.npz: per online row z = (x - hist mean)/hist sd,
|z|, first difference (seeded with the last history point), log1p(t); per
series the last 512 standardized history points (+ length). Soft targets,
folds, weights copied from Model124's export (identical teachers / folds).
train_tcn.py: history encoder (causal TCN 2->32, 5 blocks, masked mean + last
-> 32-d context) ; online model [4 channels + context] -> dilated causal TCN
(64 ch, 8 blocks, receptive field 511) -> GRU(96) -> per-step logit. Loss =
weighted BCE on 0.5*y + 0.5*teacher, time-pair weights; best epoch by held
TS-AUC (same protocol as Model124). Runs on CPU (torch) or a vast.ai GPU.

Smoke run 02:18 (CPU, 8 threads, 1 epoch, fold 0): 222 s/epoch, held TS-AUC
0.618134 after one epoch (gate 1 passed). Full run at lr 2e-3 (OneCycle) was unstable: fold 0 epochs 0-4 = 0.6358,
0.6268, 0.5955, 0.6061, 0.6116 with the loss flat at 66.2 (train_lr2e-3_
fold0_abandoned.log) - killed 02:42. Restarted 02:45 with lr 5e-4: five
folds x 12 epochs on CPU (~3.5 h), train_lr5e-4.log, fold{k}_pred.npz /
fold{k}_weights.npz.
LEAK FOUND 03:30 (results above are void): CausalBlock used GroupNorm(1, ch)
on (B, C, T), which normalizes over channels AND time, so every step saw the
whole online series' statistics. Symptom: fold 0 "0.6477 raw" and an
impossible +0.05 blend gain; per-step AUC 0.81 at t=5 where the causal stack
is at chance. Fixed with per-step LayerNorm over channels; causality unit
test (perturb steps >= 300, outputs < 300 unchanged) added before the rerun.
Leaky artefacts kept as leaky_fold0_*.npz / train_leaky_groupnorm.log.
Gates (pre-registered, as for Model124): fold-0 raw >= 0.615; dev blend gain
>= +0.003 over Submission31; sealed holdout; emailed approval before upload.
Serving, if it earns it: numpy per-step forward with a 511-step buffer per
TCN layer (incremental causal convolution) + GRU state.
