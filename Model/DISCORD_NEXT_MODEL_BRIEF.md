# Discord research brief for the next model

Prepared 2026-09-15 from the discussion pasted by the user. All external
scores below are participant reports, not independently verified results or
current leaderboard checks. This is an experiment brief, not a trained model.

## Main takeaway

Prioritize better estimates of scale and dependence changes in causally
whitened series, with empirically calibrated amplitude priors and correctly
weighted integration over candidate break times. Test a few separate evidence
features against the current stack. The discussion gives much less support
for simply adding model capacity, broad feature banks, or more synthetic data.

## What was reported

- Baseline: approximately 0.6263 TS-AUC using AR + EWMA + empirical-CDF
  whitening, about 119 window features, and a GBM/MLP/ExtraTrees blend.
- Using raw series instead of whitened series made the same oracle statistics
  about 2.5 times worse by the author's comparison; the exact interpretation
  of that ratio is unspecified.
- Five-moment oracle with the true break time, tau, gained about +0.0191;
  integrating over unknown tau initially gained about +0.0035. The initial
  claim that unknown tau explains 82% of the available headroom was withdrawn
  as an information-bound interpretation.
- GRU, dilated TCN, causal transformer, column bagging, and larger ensembles
  added at most approximately +0.0007 in that participant's probes.
- Predicting online segment length from 23 historical descriptors produced
  negative out-of-fold R-squared. That approach found no useful length prior;
  it does not prove all possible priors are useless.
- The full calibrated Bayesian evidence block reportedly added +0.0051 on
  8,901 series, with a series-level bootstrap. Scale and dependence supplied
  the benefit; the mean channel scored about 0.507 on its own.
- Bundling eight evidence columns once added only +0.0001, although an
  individual column had added +0.0008. More evidence features can dilute gains.

## Oracle correction: these are not performance ceilings

Knowing tau does not make a restricted, noisy statistic an optimal detector.
The cell-based empirical-law and five-moment estimators use the same observed
window but achieve different results. Reported gains after length calibration:

| Oracle combination | TS-AUC gain |
| --- | ---: |
| Cells only | +0.0054 |
| Five moments only | +0.0191 |
| 75% moments / 25% cells | +0.0169 |
| 50% / 50% | +0.0134 |
| 25% moments / 75% cells | +0.0080 |
| Maximum of both | +0.0103 |

An earlier cell comparison gave +0.0074; do not silently equate the two setups.
At true post-break ages 6-10, cells reportedly gained +0.054 over production
while moments lost -0.083. An oracle choosing by true age reached +0.0206,
but true age is unavailable at inference. Elapsed online time is observable
and is not the same as age since the break.

These results are consistent with estimator noise and ineffective combination;
they do not by themselves prove variance is the only limitation. Treat oracle
gains as evidence of achievable improvement with extra information, not an
upper bound on the best deployable model. A leader exceeding these oracles
does not establish that the leader uses foundation models or different data.

The author also reported identical +0.0191 gains when supplying the true
break channel versus using an omnibus statistic, then declared routing/MoE
closed. Preserve the observation, but not that universal conclusion: equality
for one oracle family is not a proof that all history-based routing or
alternative-dependent ranking is useless. Any deployable routing must use
information available for every series at prediction time.

## Concrete Bayesian feature recipe

### Integrate uniformly over candidate break times, not grid nodes

Reported suffix lengths:

    4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768

For each channel, combine amplitude alternatives within each suffix, then
combine suffix log evidence using:

    logsumexp(log_BF(m_j) + log(w_j)) - log(n)

Here n is the number of candidate change points, and w_j is the count of
candidate tau values represented by grid cell j, approximately m_j-m_(j-1).
The grid approximates a uniform prior over tau. Equal weights over geometric
grid points overrepresent short windows.

Implementation details to settle explicitly: cover lengths below 4, clip
cells to the observed prefix, include the final partial cell, and specify
coverage beyond 768. Require sum(w_j)=n; if intentionally truncating the
candidate set, normalize by its actual size and document the changed prior.
The pasted discussion does not fully specify these boundaries.

### Keep channels separate and calibrate amplitude priors

| Channel | Reported alternatives | Reported empirical break size |
| --- | --- | --- |
| Scale | Inverse-gamma on variance: shape a in {100,30,12,6,3}, scale parameter b=a-1 under density proportional to v^(-a-1) exp(-b/v) | absolute log post/pre standard deviation: q25 .047, median .109, p90 .404 |
| Mean | Normal shift, standard deviation in {.05,.12,.30} | median .064, p90 .211 |
| Dependence | AR(1) coefficient in {+/-.06,+/-.12,+/-.20,+/-.35} | median .075, p90 .208 |

The scale prior has expected variance 1. The author quoted an approximate
spread 1/sqrt(a-2); do not confuse that variance dispersion with the standard
deviation of log sigma. Verify the parameterization before coding.

Mix alternatives uniformly within each channel. Emit scale, dependence, and
mean as separate columns, rather than summing all channels into one score.
The dependence prior originally started at .15, above approximately 79% of
the observed break amplitudes; lowering it to .06 reportedly made it useful.
An improvement changing sign between series halves can reflect a misplaced
prior dominated by rare large breaks, although that is not the only cause.

Measure our amplitude distributions on fitting series only. True break labels
may guide training-fold calibration but must not enter validation features,
held-fold prior fitting, or online inference. The reported grids are starting
hypotheses, not automatically calibrated priors for our representation.

## Learning curves: small gains, substantial uncertainty

The author corrected a stale, small-sample-overfitting configuration
(4,000 estimators, colsample .7) and evaluated the actual full production
stack, 119 columns, four seeds, on 1,973 untouched series:

| Training series | Reported TS-AUC |
| ---: | ---: |
| 600 | .6051 +/- .0073 |
| 1,000 | .6100 +/- .0069 |
| 1,700 | .6173 +/- .0019 |
| 2,800 | .6190 +/- .0052 |
| 4,700 | .6233 +/- .0029 |
| 7,917 | .6239 +/- .0009 |

For a curve ceiling - b/n^c, fixing c=.5 gave ceiling .6322; fitting c gave
c=.575 and ceiling .6302, with a broad reported c interval [.24,.95].
Fitting the first five sizes overpredicted the sixth by .0013 (tree alone:
.0015). The .0013 miss was about one prediction-standard-deviation, not the
originally claimed 3.1. This weakly suggests faster saturation; it does not
establish a hard asymptote or a representation bound.

Using the author's own +.0026 validation-to-leaderboard offset, projections
were about .632 at 10 times the data and .633 at 100 times the data; their
informal ceiling was revised to about .635. The quoted leader was .6524.
Those offsets and absolute scores do not transfer to our validation split.
The corrected last segment gained about .0008 per doubling of series versus
.0003 per doubling of columns. Synthetic augmentation reportedly gained
.0013 on that stack. Do not budget large augmentation or GPU runs on the
retracted scaling estimates.

## Additional discussion: possible public-leaderboard contamination

A participant speculates that 100 publicly available leaderboard series
could be included in training or recognized exactly at inference. They claim
memorization could raise a score near 64% to roughly 64.7%: about +0.007 AUC,
or 0.7 percentage points, not +0.7 AUC. This is an unverified estimate; the
effect depends on those series' time-stratum coverage, pair weights, and
prediction ranks, not merely their count.

Separate three claims:

- Exact recognition with known answers could improve evaluation on matching
  public series. It offers no direct benefit on completely disjoint private
  series. Perfect answers on a subset also do not automatically imply perfect
  cross-sectional ranking against the rest of the evaluation set.
- Training on additional labeled public series might help generalization,
  but it could also hurt through overfitting or distribution mismatch. The
  discussion's assertion that it cannot hurt is not established.
- The claim that such use is permitted has not been checked against official
  rules. There is no evidence here that any particular leader used either
  approach, or even verification of the public data's contents and overlap.

Implication: public leaderboard gaps and validation-to-leaderboard offsets
are weaker evidence of transferable model quality if evaluation-series
exposure differs between competitors. For illustration only, subtracting
the claimed .007 effect from the quoted .6524 leader gives .6454; this is
not an estimate of that leader's uncontaminated performance.

For the next experiment, keep the primary evidence on held-out complete
series with recorded data provenance and overlap checks. If public labeled
data are later considered, first verify their provenance and permitted use,
then evaluate any training augmentation separately on untouched series.
Record public-data exposure with leaderboard results so public-only gains
are not mistaken for progress toward private performance and a prize.
This brief does not add public-series lookup logic or change training data.

## Repository context and recommended next experiment

The inspected readmes show:

- Model51 already tried AR(5) rank whitening and Gaussian shift priors over
  six suffix lengths, aggregated across windows, channels, and amplitudes.
  Model57 describes that unsupervised experiment as unsuccessful. That does
  not reject the calibrated, channel-separated recipe above.
- Model57 supplies history-fitted AR selection, EWMA normalization, empirical
  CDF whitening, and seven innovation channels. Reuse this infrastructure.
- Model84 uses the 317-feature contract with the frozen 40% synthetic pool.
  Model88's readme calls Model84 the .637888 local leader; confirm against
  actual completed reports before choosing the next baseline.
- Model88 already tests width-5 whitened features. Inspect its result before
  duplicating short-window work.
- Model87 provides incremental whitening with documented parity checks;
  preserve streaming efficiency when adding new inference features.

Proposed CPU experiment after checking current run status and numbering:

1. Freeze the verified incumbent, folds, rows, synthetic weighting, learner,
   and postprocessing. Audit Model51 code to identify precisely which prior,
   weighting, and aggregation choices were already tested.
2. Estimate scale/dependence break sizes using fitting-series labels only;
   compare them with the proposed grids and assess estimation noise.
3. Build two causal features on existing whitening: tau-integrated scale
   evidence and tau-integrated dependence evidence. Reserve mean as a
   secondary ablation given its weak reported result.
4. Pilot baseline + scale, baseline + dependence, then baseline + both.
   Measure incremental TS-AUC on the complete stack. Keep the search small.
5. Freeze the selected recipe before confirmation folds. Report paired
   series-bootstrap uncertainty, fixed-ID-half gains, and performance by
   observable online time. Use true break age only for offline diagnostics.
6. Check prefix invariance, valid grid mass, finite outputs, and streaming
   parity/runtime. Keep held-out/test labels closed. Treat adaptively reused
   development folds as exploratory rather than independent confirmation.
7. Record results in the next unused model<N>/readme.txt. Only a validated,
   packaged candidate can supply public-leaderboard evidence; this discussion
   alone cannot predict a prize or a transferable score gain.

Success hypothesis: a compact, correctly calibrated pair of evidence features
adds complementary ranking signal to our current stack. The external +.0051
is a motivation for testing, not an expected gain or a promise.
