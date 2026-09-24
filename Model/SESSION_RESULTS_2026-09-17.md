# Session results, 2026-09-16/17 (Models 109–123, Submission30)

## Outcome

| | Public TS-AUC | Rank |
|---|---:|---:|
| Before (Model26, run 114949) | 0.624253 | 166 / 727 |
| Submission30 (run 117218) | 0.629568 | 122 / 727 |
| **Submission31 (run 117482, 18 Sep 03:34)** | **0.630873** | **115 / 736** |
| Prize zone (10th) | 0.651639 | |
| Leader | 0.663010 | |

Submission30 = Submission28 chain (Model73 stack with exact incremental whitening)
plus a top-level z-score blend (weight 0.5) with an **oracle-distilled** LightGBM
voter (Model111). Cloud run 3 h 56 m, determinism check passed. Sealed archive:
`submissions/submission30.zip`; build/validate/seal scripts in `models/model112/deployment/`.

## The two findings that matter

1. **Oracle distillation works, modestly.** A teacher that sees the whole online
   segment (privileged features: max-over-onset Gaussian/dependence LLRs, onset
   posterior, post-segment moments; `models/model109/teacher_features.py`) is
   cross-fitted and its probabilities become soft targets `0.5*y + 0.5*teacher`
   for the causal student. Raw voter +0.0039 on all five dev folds (Model111),
   +0.0053 on the cloud after the top-level blend.
2. **The development folds are ~0.006 optimistic; the sealed holdout equals the
   cloud.** Running the Submission30 bundle on the 1,667 never-used holdout series
   (Model123): Sub28 0.6297, Sub30 0.6306 — versus cloud 0.6296 and dev-fold 0.6369.
   The long-standing "local→cloud offset" is adaptive selection on the dev folds
   (100+ models chosen against them), not a metric or distribution issue; the
   official `scoring.py` and `tsauc.py` agree to 12 decimals on 840,797 rows.
   **Protocol change:** judge future bundles on the holdout
   (`models/model123/holdout_eval.py`, ~1.5 h with 8 workers); never tune on it.

## What did not help (all on top of Model111 / Submission30)

| Model | Change | Result |
|---|---|---|
| 110 | look-ahead teacher (future causal features) | no student gain |
| 113 | second distilled voter, α = 0.25 | no blend gain |
| 114 | distilled 500-feature voter | +0.0012 raw, no blend gain |
| 115 | 3× real rows | −0.0024 / −0.0001 |
| 116 | learned temporal stacker | tree −0.023 (quantization ties); linear −0.006 |
| 117 | seed-averaged teachers | ±0 |
| 118 | 127-leaf / 2000-tree student | −0.002 / −0.007 |
| 119 | α = 0.75 | +0.0016 / −0.0052 |
| 120 | age-softened labels (τ-only teacher) | ≈ hard labels → the teacher's gain is not just early-post-break noise |
| 121 | age-softening + teacher | −0.0006 / −0.0009 |
| 122 | 3-seed student bag | −0.0008 vs deployed seed (which was a lucky draw) |
| 123 | blend weight 0.35 / t<32 gate, confirmed on holdout | +0.0002 / +0.0003 (noise) |

## Update 07:55 — Models 124/125 and Submission31

| Model | Change | Result |
|---|---|---|
| 125 | soft targets on the synthetic rows too | +0.0054 / −0.0040 / +0.0002 / +0.0011 / −0.0006 → noise |
| 124 | causal GRU student (61 whitening + 40 detector cols + log1p t, distilled targets), trained on a vast.ai RTX 4090 (~$2) | raw 0.6208 (ensemble of 3 configs) — weaker than the tree, but as a **second top-level component** it gains on all five folds: pilot +0.0018, confirmation +0.0020 (3/3), dev 0.63885 |

**Submission31** = Submission30 + GRU ensemble (15 numpy GRUs, O(1)/step; parity 1.2e-7;
runtime 1.195× Sub30 ≈ 4.7 h). **Sealed holdout: 0.63222 vs Sub30 0.63064 (+0.0016)**, both
ID halves positive, gains concentrated at t < 256; series bootstrap 95% CI [−0.0016, +0.0048],
P(gain > 0) ≈ 0.82. Awaiting the user's approval for the cloud run.

## Update 11:50 — Models 126/127 and Submission32

| Model | Change | Result |
|---|---|---|
| 126 | GRU with 200 detector columns (262 inputs) | same raw (0.6337/0.5985) but blend value drops to +0.0006 — more tree-like inputs remove the diversity that makes the GRU useful |
| 127 | GRU on the 61 whitening channels only (62 inputs) | raw only 0.585, yet as a 4th ensemble member: pilot +0.0022 / confirmation +0.0021 vs Sub30 |

**Submission32** = Submission30 + 4-config GRU ensemble (20 numpy GRUs, two input contracts):
dev 0.63908 vs Submission31 0.63885 (+0.0002); parity 1.8e-7; ~4.3 h cloud; sealed.
**Sealed holdout: 0.63201 vs Submission31 0.63222 (−0.0002) → dropped.** Submission31 remains
the candidate; GRU-variant gains beyond the first ensemble are at the holdout's noise level.
Model128 (GRU on raw values only, no features): ~0.51 after 6 epochs — chance; stopped. Lesson: the GRU's contribution is *diversity from the distilled tree*, not raw
strength — the weakest-looking member (whitening-only) helps most per unit of raw score.

## Status at 19:40 and next steps

The user chose not to run Submission31 (+0.0016 holdout) and to wait for a clearly higher
score. Nothing tried after Submission31 (Models 125–128, Submission32, a time-gated GRU weight)
moved the sealed holdout by more than ±0.0002. Public: 0.6296, rank 124/730.

What could plausibly give a *larger* step (not done; each is a multi-hour build):
1. **Feature-level stacking**: feed the GRU ensemble's logit (nested, fold-clean) into the
   distilled LightGBM as extra columns instead of blending scores. Blocked tonight because the
   400k synthetic splice rows have no full sequences (caches hold only sampled rows); needs the
   feature pipeline run on the spliced series (`build_cache`, ~1 h) and 15 nested GRUs (~20 min
   on a GPU). Expected: +0.002–0.005 raw if the GRU signal is complementary, unproven.
2. **Distil the chain's other components** (31 detector-bank voters, compact model) with the
   teacher targets — the Model6 pipeline's caches are missing, so this means rebuilding them.
3. **Teacher with true τ as a feature** for the *synthetic* rows only, where the splice point is
   known exactly — Model125 used learned teachers there (+0.0004); a τ-aware synthetic teacher
   is untested.
Everything else in the distillation/GRU family is at the holdout's noise floor; do not spend
cloud runs on it. Sealed, validated bundles ready if wanted: submission31.zip (+0.0016
holdout), submission32.zip (−0.0002 vs 31).

## Update 22:35 — Models 129–132 (user: keep going, no time limit)

| Model | Change | Result (raw, folds 0/1 unless noted) |
|---|---|---|
| 129 | GRU logit as a LightGBM column (feature-level stacking); synthetic rows given real GRU values via full-sequence features | +0.0003 / +0.0004 / −0.0021 (folds 0–2) → worse than the score blend; nested GPU stage cancelled. (The NaN-on-synthetic pilot's +0.0053 was an artefact.) |
| 130 | synthetic mass 0.2 / 0.6 under distillation | 0.2: +0.0063 / −0.0091; 0.6: −0.0086 / +0.0035 → 0.4 stays |
| 131 | bidirectional-GRU teacher (GPU, 15 nested models, ~$0.5) | teacher 0.65–0.70 < tree teacher 0.69–0.73; student −0.0035 / +0.0028 |
| 132 | cross-family teacher average | −0.0030 / +0.0041 |

Every target-side change since Model119 shows the same **opposite-sign split between folds 0
and 1**; nothing is deployable from it on its own. Total vast.ai spend ≈ $2.5.

Bagging check (saved pilot predictions, logit average): Model111 + Model131 → 0.64405 / 0.61009
(+0.0000 / +0.0029 vs Model111); + Model132 → 0.64357 / 0.61103. Students correlate 0.97, so
this is variance reduction across teacher families: ≈ +0.0009 raw over five folds (4/5). On top
of the Submission31 blend, however, bagging 111+131 (or +125) changes nothing (±0.0001): the GRU
component already supplies the variance reduction. Closed.

## 08:45 (18 Sep) — Staged protocol, stage 1 (Model137: data)

Same student, same folds, only training data changed (gain vs current 0.62337, 5-fold mean ± s.e.):
splices 20% +0.0008 ± .0027 (4/5); 60% −0.0043 (1/5); original-only −0.0052 (2/5); all 2.5M real
rows −0.0012 (1/5); user's exhaustive-pairing idea as 8× splices +0.0043 / −0.0015 on the pilot
(folds 2–4 queued). Rule (gain > 2 s.e. and ≥ 4/5): **no change wins; current data kept.**
Stage 3 identity results: 79/130 single-test features significant at p<.01; scale, distribution
and change-point tests lead (CUSUM-of-squares 0.556, Levene 0.550, Wasserstein 0.551); signed
location tests are at chance. Transforms and stage 5 search running.

## 03:00 (18 Sep) — Model136, the user's suggestion (monotone scores)

Cumulative-max (monotone) scores: −0.038 dev / −0.042 holdout; 25% mix −0.002; 10% ≈ 0. The
optimal score is not monotone (false alarms must be revisable); the chain's rolling-max-128 at
10% is the only helpful form. Closed.

## 01:30 (18 Sep) — Model134, the user's suggestion (bootstrap dataset)

Block-bootstrap replicas of all 8,333 dev series (644,592 rows × 500 cols, 55 min). Distilled
student: replicas instead of splices +0.0002 / −0.0220; added to splices −0.0015 / −0.0051.
Replicas carry no new break information; the splices matter because they create *new* breaks.
→ Model135: parametric break injection (mean/scale/AR/shape at real-break magnitudes) into
break-free series, 648k rows: instead of splices −0.011 / −0.017; added −0.003 / −0.001.
**The synthetic-data lever is exhausted**: only real post-break tails (splices) teach the model;
neither resampling nor parametric shifts reproduce the organisers' post-break behaviour.

## 00:05 (18 Sep) — Model133, the user's suggestion (transforms + higher moments)

Inventory: |x| (1,086 cols), x² (68), empirical-CDF/quantile cells (1,717), Haar (70) existed;
sqrt/log transforms, Fourier bands and 3rd/4th/5th online-window moments did not. Model133 adds
23 such causal columns (real + synthetic rows). Distilled student: −0.0009 / −0.0007 on folds
0/1; columns used (8% gain share) but redundant with the quantile cells. Standalone TS-AUC
0.51–0.55. Closed.

## 23:00 — Submission31 submitted on the user's instruction

Email 22:48: "submit 31 and wait". Uploaded to `sympathetic-trout` (bitter-krittanut) as
submission #5 (id 81831); cloud run **117482** started 22:59 (inference only, ~4.7 h).
`.env` now holds two keys (KRITTANUT → sympathetic-trout; YODHARIN → minimum-mantis).

User question, answered by measurement (not applied): perfect dictionary answers on 1% of series
(the 100 public series) would inflate the public TS-AUC by ≈ +0.0074 on the holdout simulation
(20 draws: +0.0049 … +0.0101); zero effect on the private test.

## 03:50–17:10 (18 Sep) — the user's staged protocol (Models 137–141)

One factor at a time, same Model111 student, five dev folds; a change wins if its
mean gain is > 2 s.e. and it wins ≥ 4/5 folds.

| stage | model | result |
|---|---|---|
| 1 data | 137 | no arm beats 400k real + 1,000 splices @40%: m20 +0.0008 (4/5, 0.3 s.e.), m60 −0.0043, orig-only −0.0052, all 2.5M rows −0.0012, 8× pairing +0.0043/−0.0015 pilot |
| 2+3 tests × transforms | 138/139 | 48 tests × 3 windows × 8 transforms on 60k rows, series-bootstrap s.e.; identity: 79/130 sig at .01, best single test CUSUM-of-squares 0.556; abs: location scans lead (Buishand 0.563); other transforms within 1 s.e. |
| 4 features + selection | 140 | 13 stage-3 winners as causal features (numba build, numpy serving): B (+13) +0.0017 3/5; **E (+13, \|ρ\|>.95 pruned) +0.0030 ± 0.0009, 5/5** — first winner; control P (pruning only) −0.0005 |
| 5 hyperparameters / seeds / ensemble | 141 | 8 random configs: best cfg03 +0.0042 pilot but +0.0012 3/5 → keep; 5-seed bag +0.0006; 243-blend grid over bag5/GRU/E/cfg03/8× → best = GRU .5 + E .5 (+0.0044 vs chain, Sub31 structure +0.0034) = the holdout-tested candidate |
| 4 follow-up | 140 ED | tests on diff / abs series added to E: +0.0026, 3/5 → not better than E |

Stage 1 8× pairing five folds: +0.0043 −0.0015 +0.0065 +0.0029 −0.0040 = +0.0016 ± 0.0019, 3/5 → fails.
Protocol finished 2026-09-19 01:40; Submission31 remains the best deployable model.

Stage-4 E on the dev blend with the chain: +0.0027 (vs a050 +0.0015). Sealed holdout
(`models/model140/holdout_eval140.py`, Submission31 with the E voters, 17:07–19:12):
0.63245 vs Sub31 0.63222 = **+0.0002** — the dev gain does not transfer; not uploaded.
Session outage 10:30–15:40 (API); background jobs were killed by low memory and restarted one
at a time — LightGBM fits at 13 GB do not coexist with the 6-worker stage-3 builder.

## Honest assessment

With this feature set the deployable ceiling is about 0.630 public. The
distillation lever is exhausted (14 variants); teacher strength/noise is not the
limiter, the student's causal features are. Reaching the prize zone (+0.022)
needs a structurally different model, which could not be built and validated
safely in this session. The 100 public labeled series are bit-identical members
of the public test (Model108); this was deliberately not exploited.

## Reproduce / continue

```
.venv\Scripts\python.exe models\model109\teacher_features.py
.venv\Scripts\python.exe -u models\model111\run_model111.py 0 1 2 3 4
.venv\Scripts\python.exe -u models\model112\select_blend.py a050
.venv\Scripts\python.exe models\model112\deployment\build.py ; validate.py parity ; seal.py
.venv\Scripts\python.exe -u models\model123\holdout_eval.py --workers 8
.venv\Scripts\python.exe models\run_status.py RUN_ID ; models\run_score.py RUN_ID
```
Windows notes: legacy runners import `fcntl`/`resource` (stubbed in
`models/model111/run_model111.py`); `xgboost` was installed into `.venv`.
