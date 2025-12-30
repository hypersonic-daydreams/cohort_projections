# G02: Small-Sample Inference & Forecasting Review

## Context

You are reviewing the statistical methodology for a journal article on forecasting international migration to North Dakota. The analysis uses **n=15 annual observations (2010-2024)**, which presents significant challenges for classical statistical inference.

A previous review (by ChatGPT 5-2 Pro) identified several critical issues that require resolution:

### Key Issues to Address

1. **KPSS Interpretation Contradiction**: The article narrative claims KPSS "fails to reject stationarity in levels," but the robustness table shows KPSS rejecting stationarity (marked **). These cannot both be true. The actual test results need to be reconciled.

2. **Unit Root vs Structural Break Confusion**: With a major break around 2020-2021, standard ADF tests may mistake broken-trend stationarity for unit roots (and vice versa). Declaring the series a random walk without drift based on AIC selection in a 15-point annual series is not a safe inferential leap.

3. **Scenario Arithmetic Issues**:
   - "8% annual growth" does not appear to compound to the reported 2045 value from the 2024 baseline (5,126)
   - "Continue 2010-2019 slope (+72/year)" yields a lower 2045 projection (2,517) than the 2024 baseline (5,126), which is logically inconsistent unless anchoring to a pre-2020 level
   - The paper reports CV = 82.5% in descriptive stats but the Monte Carlo uses CV = 0.39 -- this discrepancy needs explanation

4. **Terminology Issues**:
   - Monte Carlo intervals are called "credible intervals" but the procedure is frequentist simulation/parametric bootstrap (not Bayesian)
   - Need consistent use of "prediction intervals" vs "confidence intervals"

5. **Backtesting Gap**: No proper out-of-sample validation is presented. Reviewers will expect rolling-origin evaluation comparing to naive benchmarks.

## Files Attached

Please review the following files (numbered for reference):

1. **01_critique.md** - The full critique from ChatGPT 5-2 Pro identifying issues
2. **02_results_unit_root.json** - Unit root test results (ADF, KPSS, Phillips-Perron)
3. **03_results_breaks.json** - Structural break test results (Bai-Perron, CUSUM, Chow)
4. **04_results_arima.json** - ARIMA model selection and forecasts
5. **05_results_scenario.json** - Scenario modeling and Monte Carlo results
6. **06_results_summary.json** - Summary statistics (including CV calculations)
7. **07_data_nd_migration.csv** - The actual migration data (15 observations)
8. **08_script_unit_root.py** - Python script that generated unit root tests
9. **09_script_scenario.py** - Python script that generated scenario projections

## Tasks

### Task 1: KPSS Resolution

Examine files 02 and 08. The KPSS test results show:
- Level series: statistic = 0.323, p = 0.10 (fails to reject at 5%)
- Differenced series: statistic = 0.500, p = 0.042 (rejects at 5%)

Clarify:
- What the KPSS results actually show
- Whether there is an internal contradiction or just imprecise language
- How to correctly describe these results in a journal article
- What the combined ADF+KPSS evidence tells us about integration order

### Task 2: Break-Robust Stationarity Testing

Given n=15 and a known structural break around 2020:
- Recommend break-robust alternatives to standard unit root tests
- Suggest appropriate language for describing stationarity findings in small samples
- Propose how to handle the 2020 COVID outlier (value=30 vs mean=1,796)

### Task 3: Backtesting Design

Design a feasible backtesting procedure for n=15:
- Specify a rolling-origin protocol (e.g., train 2010-2016, predict 2017; train 2010-2017, predict 2018; etc.)
- List appropriate metrics (MAE, RMSE, interval coverage)
- Define at least 3 baseline comparisons:
  1. Naive last-observation (random walk)
  2. Mean/median benchmark
  3. Simple regression with a national driver
- Address the constraint that with n=15, leave-one-out creates very small training sets

### Task 4: Scenario Verification

Examine files 05, 06, and 07. Verify or correct the scenario arithmetic:
- CBO Full: Does 8% annual growth from baseline 5,126 yield the reported 2045 value (~19,318)?
- Pre-2020 Trend: If the 2010-2019 slope is +72.43/year and we anchor at the pre-2020 endpoint (634 in 2019), does continuing this slope yield 2,517 by 2045?
- CV Discrepancy: The summary stats show CV = 82.5% (=1482/1796), but Monte Carlo uses CV = 0.39. Explain why these differ or flag as an error.

### Task 5: Terminology Audit

Review all statistical terminology and flag corrections needed:
- "Credible intervals" vs "prediction intervals" vs "confidence intervals"
- "Unit root" vs "non-stationary" usage
- Any other imprecise or incorrect statistical language

## Output Format

**IMPORTANT: In your response text (before the file links), please include a brief executive summary (3-5 paragraphs) covering:**
1. Resolution of the KPSS interpretation issue
2. Key findings from scenario verification (any arithmetic errors found)
3. Top 3 priority terminology/methodology corrections
4. Your assessment of what the small-sample analysis can and cannot credibly claim

**Then produce the following downloadable files:**

### 1. G02_kpss_resolution.md

A markdown document containing:
- Correct interpretation of the KPSS results
- Resolution of the apparent contradiction
- Recommended language for the revised article
- Summary table of all unit root test conclusions

### 2. G02_backtesting_spec.md

A markdown document containing:
- Detailed backtesting protocol specification
- Code pseudocode or Python snippet for implementation
- Expected output table format
- Discussion of limitations given n=15

### 3. G02_terminology_corrections.csv

A CSV table with columns:
```
original_phrase,replacement_phrase,context
```

List every terminology correction needed, including:
- Statistical term fixes
- Hedging language additions (e.g., "suggests" instead of "establishes")
- Any arithmetic corrections found

### 4. G02_recommendations.md

A summary markdown document containing:
- Prioritized list of changes (critical / important / minor)
- Recommended additional analyses if any
- Suggested appendix content for methodological transparency
- Key talking points for responding to reviewers

## Additional Context

The paper's target is a top-tier demography journal. The core contribution is demonstrating that "rigorous analysis remains feasible in small samples" for small-state migration forecasting. The revision must:

1. Align claims with identification strength
2. Use appropriate hedging language throughout
3. Ensure all arithmetic is reproducible
4. Address the unique challenges of n=15 annual observations

The attached Python scripts show exactly how the analyses were conducted - please examine these to understand the methodology and identify any implementation issues.
