# Candidate detector lineage

This folder contains all 21 complementary/candidate-lineage implementations
from `d31` onward.

The twelve production candidate banks are:

`d32`, `d33`, `d35`, `d37`, `d40`, `d45`, `d51`, `d52`, `d53`, `d55`, `d57`,
and `d58`.

The non-production candidate experiments are `d31`, `d36`, and the new
winner-inspired `d60` bank. `d60` was evaluated in the joint-model study but
was not promoted because the joint model failed its sealed-test gate.

Model 8 adds six experimental banks: `d61` (rank/distribution tests), `d62`
(causal Sup-F and piecewise scans), `d63` (symbolic entropy and nonlinear
dependence), `d64` (rank trends and ACF/PACF shifts), `d65` (label-free
evidence aggregation), and `d66` (IQR-robust and Symlet energy).

They cover spectral, local-pattern, distribution, volatility, serial-copula,
adjacent-block, sequential, higher-order shape, conditional-dependence,
conformal, and nonlinear forecast-surprise evidence. Some reuse shared or core
state to avoid calculating the same causal statistics twice.

Models 4-6 use a frozen 15-bank candidate contract. The repository-wide
`ALL_CANDIDATE_DETECTOR_BANKS` contract includes all 21. Because the
original d31 fitted templates were not retained, its Model 4 adapter uses eight
fixed label-free DCT bin directions and records that choice in the experiment
manifest.
