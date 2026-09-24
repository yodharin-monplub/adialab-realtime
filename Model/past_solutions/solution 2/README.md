# Structural Break Challenge — ADIA Lab & CrunchDAO

This repository contains my solution for the [Structural Break Detection Challenge](https://hub.crunchdao.com/competitions/structural-break), organized by **ADIA Lab** and **CrunchDAO**.

The task is to determine **whether a structural break occurred** at a specified point in each univariate time series.

---

## Feature Engineering

Features were extracted from:
- Full time series
- Segments **before** and **after** the boundary point
- Corresponding **return series**

### Statistical Features
- `mean`, `std`, `kurtosis`, `skew`
- Quantiles: `q10`, `q25`, `q75`
- Signal-to-noise ratio: `|mean| / std`

### Rolling Features
- Rolling statistics for mean, std, kurtosis, skew  
- Mean and standard deviation of rolling windows

### Autocorrelation & Trend
- Lag-1 autocorrelation  
- Linear regression slope over time

### Local Window Analysis
For windows around the boundary point (`±7, ±15, ±30, ±100`):
- Difference in standard deviation (before vs after)
- Levene’s test statistic for variance equality

### Variance Ratio Dynamics
For sliding windows (`size = 100`, `step = 50`):
- Compute variance ratio before/after  
- Extract:
  - Variance change **magnitude at boundary**
  - **Maximum** variance change magnitude  
  - **Relative distance** between detected and true boundary (how far the strongest variance change occurs)

### Higher-Order and Statistical Tests
- Hurst exponent (long-term memory / mean reversion)
- Distribution tests:
  - Kolmogorov–Smirnov (KS)
  - Anderson–Darling
  - Cramér–von Mises
  - Mood’s median test
  - Fligner–Killeen variance test
- Energy statistic:  
  `2 * cross_dist - before_dist - after_dist`

### Frequency & Complexity Features
- Wavelet decomposition using **db4**
- Sample entropy
- AR(5) model coefficients
- Selected **tsfresh** library features

---

## Modeling Approach

Tree-based models performed best in this task: Catboost, LightGBM, XGBoost
