Submission26: pretrained CPU-only depth-diversity ensemble.
Original Model23 resources retained; ten depth4 XGBoost voters added.
For each fold: average depth3 aligned25/60 probabilities in float32,
average depth4 aligned25/60 probabilities in float32, then average both.
Keep the original five-fold logit bag, stack and rolling128 postprocess.
No training data consumed. forceFirstTrain=False, trainFrequency=0.
12 inference workers, one numerical thread per worker; 16 CPU / 64 GiB target.
Local CV 0.632406979 versus Model23 0.632094726; 4/5 fold wins.
Reused development evidence does not guarantee cloud improvement.
Serving validation and runtime estimate required before upload.
