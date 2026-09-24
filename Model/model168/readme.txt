Model168 - what is a BIGGER GRU ENSEMBLE worth? (2026-09-23)

Before renting a GPU to train more neural students, measure the payoff.
gru_count.py scores the 15 deployed GRUs individually on the sealed holdout
(inputs from Model152's saved components) and averages random subsets.
  single nets: 0.597 ... 0.621 (mean ~0.609)
   1 net   raw 0.6117
   3 nets  raw 0.6175
   5 nets  raw 0.6182
  10 nets  raw 0.6178
  15 nets  raw 0.6191
-> the ensemble SATURATES after ~3 networks: going 3 -> 15 buys +0.0016 raw.
   Training 15-30 more of the same size is nearly worthless.
   (The "final" column printed by the script uses the in-sample AP3 student
   and is meaningless; only the raw column matters here.)
Conclusion: a GPU is only worth renting for a BIGGER/different neural model,
not for more copies of the deployed one -> Model169.
Cache: data/gru_logits.npy (15 x 840,797 holdout logits, reusable).
