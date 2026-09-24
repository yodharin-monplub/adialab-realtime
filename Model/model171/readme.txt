Model171 - blend weights re-checked with the NEW neural part (2026-09-24).

Submission41 replaces the 15 deployed GRUs with 5 bigger ones (Model169).
A stronger component can shift the best blend weights, so the Model148
protocol was re-run with the new nets' development out-of-fold predictions
(weights_eval171.py new; student = Model164's SRP40, chain = Model112's dev
chain; weights picked on pilot folds 0-1, confirmed on folds 2-4):
  reference ws=0.5 wg=0.5 -> dev mean 0.64119
  ws=1.0 wg=0.5 : pilot +0.00095  confirm +0.00079  5/5
  ws=1.5 wg=0.5 : pilot +0.00109  confirm +0.00074  3/3 confirmation wins
  ws=1.5 wg=0.75: pilot +0.00100  confirm +0.00051  5/5
  PILOT-SELECTED ws=1.5, wg=0.5
-> exactly the weights already in Submission41 (student 1.5, neural 0.5);
   no change needed.  Dropping the chain still costs -0.0033, so all three
   components stay.
Usage: python models/model171/weights_eval171.py new   (or `old` for the
deployed GRU set).
