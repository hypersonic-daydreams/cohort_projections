# Statistical Analysis of North Dakota Immigration Flow Patterns

## Comprehensive Summary Report

**Analysis Date:** December 29, 2025
**Data Period:** 2002-2024 (varies by module)
**Geographic Focus:** North Dakota within national context

---

## Executive Summary

This report synthesizes findings from a 9-module statistical analysis pipeline examining international migration patterns to North Dakota. The analysis employed diverse methodological approaches ranging from descriptive statistics to causal inference, machine learning, and survival analysis.

### Key Findings

1. **ND receives a disproportionate share of US international migration** relative to its population (0.17% of US international migration vs. 0.23% of US population), with high volatility (CV = 83%).

2. **Immigration composition is diversifying but regionally concentrated**: The Herfindahl-Hirschman Index shows unconcentrated country-level origins (HHI = 616-942) but highly concentrated regional sources (HHI = 2,600-5,500), dominated by Asia (52.5%) and Africa (27.9%).

3. **Time series exhibit non-stationarity**: ND international migration is I(1), requiring first-differencing for modeling. The ARIMA(0,1,0) model indicates a random walk process with high uncertainty.

4. **Structural breaks are policy-driven**: Chow tests identify significant breaks at 2020 (COVID, p = 0.0006) and 2021 (recovery, p = 0.003). The Travel Ban reduced refugee arrivals from affected countries by approximately 75%.

5. **Network effects are weak to moderate**: Gravity model elasticity of diaspora stock on new admissions is 0.096 (controlling for mass variables), while panel analysis shows stronger persistence (0.85 elasticity of lagged stock).

6. **Machine learning confirms population size as dominant predictor**: Random Forest feature importance shows log_population accounts for 93% of explained variance in international migration share.

7. **Immigration waves have median duration of 3 years**: Survival analysis of refugee arrival waves shows higher-intensity waves persist longer (median 4 years for top quartile vs. 2 years for bottom).

8. **Scenario projections range widely**: 2045 international migration to ND projected between 2,517 (pre-2020 trend) and 19,318 (CBO-elevated) persons annually, with Monte Carlo 95% CI of 3,183-14,104.

### Overall Assessment of Rigor

| Module | Data Quality | Method Appropriateness | Statistical Significance | Robustness | Overall Rating |
|--------|-------------|----------------------|------------------------|------------|----------------|
| 1.1-1.2 | Good | Excellent | N/A (descriptive) | High | Strong |
| 2.1-2.2 | Adequate | Appropriate | Mixed | Low (small n) | Moderate |
| 3.1-3.2 | Good | Excellent | Significant | High | Strong |
| 4 | Good | Excellent | Mixed | High | Strong |
| 5 | Good | Appropriate | Significant | Moderate | Good |
| 6 | Good | Appropriate | N/A (ML) | High | Strong |
| 7 | Good | Excellent | Significant | High | Very Strong |
| 8 | Good | Excellent | Significant | High | Very Strong |
| 9 | Synthetic | Appropriate | N/A | Low | Exploratory |

---

## 1. Methodology Overview

### 1.1 Analysis Framework

The pipeline consists of nine interconnected module groups:

| Module | Analysis Type | Primary Methods |
|--------|--------------|-----------------|
| 1.1 | Descriptive Statistics | Summary statistics, HP filter decomposition |
| 1.2 | Concentration Analysis | HHI, Location Quotients |
| 2.1 | Univariate Time Series | Unit root tests, ARIMA, Structural breaks |
| 2.2 | Multivariate Time Series | VAR, Cointegration, Granger causality |
| 3.1 | Panel Analysis | Fixed/Random Effects, Hausman test |
| 3.2 | Network Analysis | Origin-specific growth, Stock-flow correlation |
| 4 | Regression Extensions | Quantile, Robust M-estimators, Beta regression |
| 5 | Gravity & Networks | PPML gravity, Network elasticity |
| 6 | Machine Learning | Elastic Net, Random Forest, Clustering |
| 7 | Causal Inference | DiD, Synthetic Control, Bartik IV |
| 8 | Duration Analysis | Kaplan-Meier, Cox PH, Wave lifecycle |
| 9 | Scenario Modeling | Monte Carlo, Model averaging |

### 1.2 Data Sources

- **Census Bureau PEP**: Components of population change (2010-2024)
- **DHS LPR Data**: Lawful permanent resident admissions by state and country (FY2023)
- **ACS Foreign-Born**: Foreign-born population by origin (2009-2023)
- **Refugee Arrivals**: State-level refugee resettlement by nationality (2002-2020)

### 1.3 Critical Methodological Decisions

1. **COVID-2020 Handling**: Retained in descriptive analysis but flagged as structural break; used as treatment in causal analysis.

2. **HP Filter Lambda**: Used lambda = 6.25 for annual data (Ravn-Uhlig recommendation).

3. **Minimum Segment Size**: Set to 2 years for structural break detection given short series.

4. **PPML Estimation**: Selected over log-linear OLS for gravity models to handle zeros and heteroskedasticity.

5. **Model Averaging**: AIC-based weights for time series; R-squared weights for cross-sectional models.

---

## 2. Key Findings by Module Group

### 2.1 Descriptive & Concentration Analysis (Modules 1.1, 1.2)

#### Summary Statistics (2010-2024)

| Variable | Mean | Std Dev | CV | Min | Max |
|----------|------|---------|-----|-----|-----|
| ND International Migration | 1,796 | 1,482 | 82.5% | 30 | 5,126 |
| US International Migration | 1,010,744 | 743,049 | 73.5% | 19,885 | 2,786,119 |
| ND Share of US (%) | 0.173 | 0.054 | 31.3% | 0.102 | 0.303 |

**Key Findings:**

- ND international migration is highly volatile (CV = 82.5%)
- Distribution is right-skewed (skewness = 1.10) with no extreme outliers under IQR method
- Shapiro-Wilk test marginally fails to reject normality (p = 0.058)
- HP filter trend component shows U-shaped pattern: declining 2010-2014, rising through 2024

#### Geographic Concentration (FY2023)

**Country-Level HHI: 1,162** (unconcentrated, based on DHS LPR data)
- Philippines dominates with 31% of LPR admissions (HHI contribution: 961)
- Top 10 countries account for 89% of HHI

**Regional-Level HHI: 3,712** (highly concentrated)
- Asia: 52.5% of admissions
- Africa: 27.9%
- North America: 12.3%

**Note:** ACS foreign-born data shows country-level HHI ranging from 482 to 942 over 2009-2023, reflecting a different population (stock vs. flow) and time period. See Figure `summary_dashboard.png` Panel C for top origin countries by Location Quotient.

#### Location Quotients (2023)

Countries with highest LQ (overrepresentation in ND relative to national):
| Country | LQ | ND Foreign-Born |
|---------|-----|-----------------|
| Egypt | 15.13 | 373 |
| India | 9.86 | 329 |
| Other Western Africa | 9.23 | 400 |
| Sudan | 8.21 | 164 |
| Other Australian/NZ | 7.46 | 5,740 |

**Interpretation:** ND attracts disproportionately high shares of migrants from African and Middle Eastern origins, likely driven by refugee resettlement programs.

### 2.2 Time Series Analysis (Modules 2.1, 2.2)

#### Unit Root Tests

| Variable | ADF Level p-value | ADF Diff p-value | Integration Order |
|----------|-------------------|------------------|-------------------|
| ND Intl Migration | 0.556 | 0.002 | I(1) |
| ND Share of US | 0.285 | 0.000 | I(1) |
| US Intl Migration | 0.002 | 0.157 | I(0) |

**Implication:** ND variables require differencing for stationarity; mixed integration orders complicate cointegration analysis.

#### ARIMA Modeling

- **Selected Model:** ARIMA(0,1,0) - Random walk
- **AIC:** 238.42
- **Ljung-Box (lag 5):** p = 0.404 (no autocorrelation in residuals)
- **5-Year Forecasts:** Point estimate of 5,126 annually with widening 95% CI from [2,928, 7,324] in 2025 to [212, 10,040] in 2029

**Assessment:** The random walk specification indicates high unpredictability. Wide prediction intervals reflect small sample size (n=15) and COVID volatility.

#### Structural Break Analysis

| Break Year | Chow F-stat | p-value | Regime Shift |
|------------|-------------|---------|--------------|
| 2017 | 1.29 | 0.314 | Not significant |
| 2020 | 16.01 | 0.0006 | Mean +91.1% |
| 2021 | 10.28 | 0.003 | Mean +161.6% |

**Key Finding:** COVID-2020 and post-COVID recovery represent the only statistically significant structural breaks. The travel ban (2017) shows no detectable break at the aggregate ND level.

#### VAR & Cointegration

- **Engle-Granger:** Suggests cointegration (ADF on residuals: -4.48, p < 0.001)
- **Johansen:** No cointegration detected (trace stat = 7.18 < CV 15.49)
- **Granger Causality:** No bidirectional causality at 5% level
- **FEVD:** ND migration explains 82% of its own forecast variance at 10 periods; US migration contributes 18%

**Assessment:** Conflicting cointegration results likely reflect small sample limitations. VAR model shows weak interdependence.

### 2.3 Panel & Network Effects (Modules 3.1, 3.2)

#### Panel Data Structure

- **Observations:** 765 (51 states x 15 years)
- **Balanced:** Yes
- **Hausman Test:** p = 1.00 (Random Effects preferred)
- **Breusch-Pagan LM:** chi2 = 1,502, p < 0.001 (RE needed vs. pooled OLS)

#### State Fixed Effects

| State | Fixed Effect | Interpretation |
|-------|--------------|----------------|
| Florida | +125,526 | Highest international migration |
| California | +106,259 | Second highest |
| Texas | +89,232 | Third highest |
| North Dakota | -18,022 | Below-average |
| Wyoming | -19,332 | Lowest |

**COVID Impact:** Mean international migration in 2020 was 390 vs. 21,206 for other years (98% reduction).

#### Network Elasticity (Stock-Flow Relationship)

| Specification | Elasticity | 95% CI | R-squared |
|--------------|------------|--------|-----------|
| Simple Panel OLS | 0.851 | [0.761, 0.940] | 0.331 |
| With Controls | 0.677 | - | 0.393 |
| Cross-sectional PPML | -0.152 | [-0.179, -0.125] | 0.044 |

**Interpretation:** The panel elasticity of 0.85 indicates decelerating dynamics - a 1% increase in lagged stock predicts 0.85% increase in current stock. The negative cross-sectional result suggests composition effects or displacement.

#### Origin-Specific Growth Analysis (2009-2023)

**Fastest Growing Origins (CAGR):**
| Country | CAGR | 2023 Population |
|---------|------|-----------------|
| Kenya | 34.1% | 11,920 |
| Iran | 20.0% | 1,261 |
| Other Eastern Africa | 17.8% | 3,582 |
| Iraq | 17.0% | 788 |
| Croatia | 16.8% | 526 |

**Declining Origins:**
| Country | CAGR | 2023 Population |
|---------|------|-----------------|
| Nepal | -23.3% | 7 |
| Liberia | -18.4% | 19 |
| Philippines | -17.8% | 34 |
| Central America | -17.7% | 71 |

### 2.4 Advanced Regression (Module 4)

#### Quantile Regression (ND International Migration)

| Quantile | Time Trend | COVID-2020 Effect | Pseudo R-squared |
|----------|------------|-------------------|------------------|
| 10th | -1.4 | -424 | 0.160 |
| 25th | 71.5 | -1,360 | 0.146 |
| 50th | 225.9** | -2,805 | 0.314 |
| 75th | 314.0 | -3,787 | 0.588 |
| 90th | 301.3 | -3,891 | 0.697 |

**Key Finding:** Trend effects are heterogeneous across the distribution - stronger at higher quantiles. COVID impact increases in magnitude at higher quantiles.

#### Robust Regression Comparison

| Estimator | Trend Coefficient | COVID Coefficient | Outliers Identified |
|-----------|-------------------|-------------------|---------------------|
| OLS | 211.0** | -2,571* | - |
| Huber M | 237.3*** | -2,838** | 2019, 2021 |
| Tukey Biweight | 257.7*** | -3,001** | 2019, 2021 |

**Finding:** Robust estimators identify 2019 and 2021 as outliers and produce trend estimates 12-22% higher than OLS.

### 2.5 Gravity & Machine Learning Models (Modules 5, 6)

#### Gravity Model (PPML, n = 2,680 state-origin pairs)

| Model | Log(Diaspora) | Log(Origin Mass) | Log(Dest Mass) | Pseudo R-squared |
|-------|---------------|------------------|----------------|------------------|
| Simple Network | 0.359*** | - | - | 0.186 |
| Full Gravity | 0.096*** | 0.039*** | 0.755*** | 0.382 |
| State FE | 0.115*** | - | - | 0.401 |

**Interpretation:** The network effect attenuates dramatically (from 0.36 to 0.10) when controlling for origin and destination mass, suggesting much of the raw diaspora effect reflects selection into larger receiving states.

#### Machine Learning Feature Importance

**Random Forest (OOB R-squared = 0.953):**
| Feature | Impurity Importance | Permutation Importance |
|---------|---------------------|----------------------|
| log_population | 93.0% | 2.152 |
| death_rate | 3.0% | 0.074 |
| dom_mig_rate | 2.6% | 0.044 |
| birth_rate | 0.7% | 0.006 |
| natural_rate | 0.6% | 0.005 |

**Elastic Net (CV R-squared = 0.599):**
- Optimal parameters: alpha = 0.0007, L1 ratio = 0.90
- Non-zero coefficients: log_population, natural_change, birth_rate
- Sparsity: 40% of features zeroed out

#### State Clustering

- **Optimal K:** 2 (silhouette = 0.644)
- **Cluster 0:** 52 states including ND (typical immigration patterns)
- **Cluster 1:** Puerto Rico only (outlier due to negative net international migration)

### 2.6 Causal Inference (Module 7)

#### Travel Ban Difference-in-Differences

| Component | Estimate | Std Error | 95% CI |
|-----------|----------|-----------|--------|
| ATT (log scale) | -1.384** | 0.481 | [-2.326, -0.441] |
| Percentage Effect | -74.9% | - | - |
| Pre-trend coefficient | 0.087 | - | p = 0.183 (parallel trends satisfied) |

**Finding:** Refugee arrivals from banned countries (Iran, Iraq, Libya, Somalia, Sudan, Syria, Yemen) decreased by approximately 75% post-ban, with parallel trends assumption satisfied.

#### COVID Interrupted Time Series

| Effect | Estimate | Std Error | p-value |
|--------|----------|-----------|---------|
| Level shift | -19,503*** | 4,363 | <0.001 |
| Trend change | +14,113*** | 3,149 | <0.001 |

**Interpretation:** COVID caused an immediate drop of 19,503 international migrants (nationwide panel), followed by accelerated recovery of +14,113 per year.

#### Synthetic Control for North Dakota

- **Donor Weights:** Wyoming (41.9%), Vermont (24.5%), Rhode Island (20.1%), Washington (8.3%), Oregon (2.7%), Florida (2.5%)
- **Pre-treatment RMSPE:** 0.020 (excellent fit)
- **Post-treatment Effects:** Mean effect = 0.57 per 1000 population; RMSPE ratio = 48.6

**Assessment:** Synthetic ND closely tracks actual ND pre-2017, suggesting valid counterfactual. Post-2017 shows positive divergence.

#### Bartik Shift-Share Instrument

- **First-stage F-statistic:** 22.46 (> 10 threshold for strong instrument)
- **Bartik coefficient:** 4.36*** (SE = 0.92)
- **Model R-squared:** 0.852

### 2.7 Duration & Scenario Analysis (Modules 8, 9)

#### Immigration Wave Survival Analysis

**Wave Definition:** 50%+ above baseline for 2+ consecutive years

| Statistic | Value |
|-----------|-------|
| Total waves identified | 940 |
| Unique nationalities | 56 |
| Unique states | 48 |
| Mean duration | 3.54 years |
| Median survival | 3.0 years |

**Kaplan-Meier by Intensity Quartile:**
| Quartile | Median Survival | Mean Duration |
|----------|-----------------|---------------|
| Q1 (Low) | 2.0 years | 2.43 years |
| Q2 | 2.0 years | 2.80 years |
| Q3 | 3.0 years | 3.61 years |
| Q4 (High) | 4.0 years | 5.35 years |

Log-rank test: chi2 = 278.7, p < 10^-60 (significant intensity effect)

**Cox Proportional Hazards (C-index = 0.769):**

| Predictor | Hazard Ratio | 95% CI | p-value |
|-----------|--------------|--------|---------|
| log_intensity | 0.412*** | [0.342, 0.496] | <0.001 |
| early_wave | 1.361*** | [1.177, 1.572] | <0.001 |
| peak_arrivals | 0.656*** | [0.520, 0.826] | <0.001 |
| Americas region | 1.711** | [1.176, 2.489] | 0.005 |
| Europe region | 1.568** | [1.157, 2.126] | 0.004 |

**Interpretation:** Higher-intensity waves and those with higher peak arrivals persist significantly longer. Waves from the Americas and Europe end more quickly than African/Asian waves.

#### Scenario Projections (2025-2045)

| Scenario | Description | 2045 Projection |
|----------|-------------|-----------------|
| CBO Full | Elevated immigration policy | 19,318 |
| Moderate | Dampened historical trend | 7,048 |
| Pre-2020 Trend | Continue 2010-2019 slope | 2,517 |
| Zero | No international migration | 0 |

**Monte Carlo Uncertainty (1,000 draws):**
| Year | Median | 50% CI | 95% CI |
|------|--------|--------|--------|
| 2030 | 5,314 | [3,511, 6,993] | [1,042, 9,473] |
| 2045 | 8,672 | [6,164, 10,962] | [3,183, 14,104] |

---

## 3. Cross-Module Insights

### 3.1 Reinforcing Findings

1. **COVID as Major Discontinuity:** All applicable modules detect COVID-2020 as a significant break:
   - Structural break tests (Module 2.1.2): p = 0.0006
   - ITS level shift (Module 7): -19,503
   - Panel fixed effects (Module 3.1): 2020 time effect = -19,429

2. **Population Size Dominates Migration Patterns:**
   - Gravity model: log(state_total) coefficient = 0.755
   - Random Forest: log_population explains 93% of variance
   - Panel FE: Largest states have largest positive fixed effects

3. **Modest Network/Diaspora Effects:**
   - Gravity elasticity (0.10) is smaller than expected from migration literature
   - Panel persistence (0.68-0.85) suggests path dependence
   - Stock-flow correlation is weak (-0.06, not significant)

4. **High Uncertainty in Projections:**
   - ARIMA forecasts have 95% CIs spanning factor of 3-4x
   - Monte Carlo 95% CI for 2045: 3,183-14,104 (factor of 4.4x)
   - Reflects short time series and structural instability

### 3.2 Conflicting or Ambiguous Findings

1. **Cointegration:** Engle-Granger supports, Johansen does not (sample size limitation likely)

2. **Network Elasticity Sign:** Panel shows positive (0.85), cross-sectional shows negative (-0.15) - suggests dynamic vs. static effects differ

3. **Travel Ban Impact at State Level:** No significant break in ND aggregate data (Module 2.1.2), but significant at nationality level in DiD (Module 7)

### 3.3 Key Uncertainties

1. **Projection Baseline Sensitivity:** 2024 value of 5,126 may be outlier or new regime
2. **Policy Regime Changes:** Immigration policy shifts are not forecastable
3. **Origin Composition Shifts:** Changing source countries affect ND differently than other states
4. **Small Sample Limitations:** n=15 years limits power for many time series methods

---

## 4. Limitations and Caveats

### 4.1 Data Limitations

| Limitation | Affected Modules | Mitigation |
|------------|------------------|------------|
| Short time series (n=15) | 2.1, 2.2, 9 | Wide CIs, cautious interpretation |
| LPR data only FY2023 | 1.2, 5 | Cross-sectional analysis only |
| ACS sampling error | 1.2, 3.2 | Margins of error reported |
| Refugee data ends 2020 | 7, 8 | Cannot assess recent trends |
| COVID partial year data | All | Flagged throughout |

### 4.2 Methodological Limitations

1. **Unit Root Test Power:** With n=15, ADF tests have limited power to reject non-stationarity.

2. **VAR Over-parameterization:** Given small sample, even VAR(1) may be over-specified.

3. **Synthetic Control Donor Pool:** Some donors (e.g., Wyoming at 42%) may not be ideal comparators.

4. **Causal Identification:** Travel ban DiD relies on parallel trends assumption; COVID ITS has no control group.

5. **Gravity Model Endogeneity:** Diaspora stock and migration flows are jointly determined.

### 4.3 Interpretation Caveats

1. **ND is an outlier state:** Small population, unique economic drivers (oil, agriculture), unusual origin composition

2. **Immigration categories conflated:** Analysis combines refugee, LPR, and other categories with different dynamics

3. **Net vs. gross flows:** Some data sources are net, others gross - not directly comparable

---

## 5. Implications for Population Projections

### 5.1 Recommended Approaches

1. **Use scenario-based projections** rather than point estimates given high uncertainty

2. **Baseline scenario** should use dampened historical trend (Moderate scenario: ~7,000 by 2045)

3. **High scenario** (CBO-elevated: ~19,000) appropriate for planning upper bounds

4. **Low scenario** (Pre-2020 trend: ~2,500) appropriate for conservative planning

5. **Monitor post-COVID recovery trajectory** - recent data (2022-2024) suggests accelerating growth

### 5.2 Key Parameters for Projection Models

| Parameter | Recommended Value | Source Module | Uncertainty |
|-----------|-------------------|---------------|-------------|
| Baseline migration (2024) | 5,126 | 1.1 | Low |
| Trend (annual growth) | +120 to +240 | 4 | Medium |
| Volatility (CV) | 0.39 | 9 | Low |
| Network elasticity | 0.68 | 3.2 | Medium |
| Wave median duration | 3 years | 8 | Low |

### 5.3 Assumptions to Document

Any population projection using these results should explicitly state:
- Immigration policy scenario assumed
- Treatment of 2020 COVID year
- Whether pre-2020 or post-2020 trends are extrapolated
- Confidence level and interval interpretation

---

## 6. Technical Appendix

### 6.1 Model Specifications

#### ARIMA (Module 2.1)
```
Model: ARIMA(0,1,0)
DV: nd_intl_migration
Differencing: First difference (d=1) based on ADF test
Selection: auto_arima with AIC criterion
Residual diagnostics: Ljung-Box(5) p=0.404, Shapiro-Wilk p=0.820
```

#### VAR (Module 2.2)
```
Model: VAR(1)
Variables: nd_intl_migration, us_intl_migration
Lag selection: AIC optimal at lag 1
Deterministic: Constant
Covariance: Standard errors (not cluster-robust)
```

#### Panel Fixed Effects (Module 3.1)
```
Model: Two-way FE (entity + time)
DV: intl_migration
Entities: 51 states
Periods: 15 years
SE: Clustered by state
```

#### Cox Proportional Hazards (Module 8)
```
Model: Cox PH
DV: Wave duration (time to end)
Covariates: log_intensity, high_intensity, early_wave, peak_arrivals,
            nationality_region dummies, state_region dummies
Ties: Breslow method
Censoring: Right censored at 2020
```

#### DiD Travel Ban (Module 7)
```
Model: Two-way FE OLS
DV: log(arrivals + 1)
Treatment: Banned countries x Post-2017
FE: Nationality, Year
SE: HC3 robust
```

### 6.2 Diagnostic Checks Summary

| Diagnostic | Module | Test | Result | Pass/Fail |
|------------|--------|------|--------|-----------|
| Residual normality | 2.1 | Jarque-Bera | p=0.776 | Pass |
| Autocorrelation | 2.1 | Ljung-Box(5) | p=0.404 | Pass |
| Parallel trends | 7 | Pre-trend test | p=0.183 | Pass |
| PH assumption | 8 | Global Schoenfeld | Passed | Pass |
| Instrument strength | 7 | First-stage F | 22.46 | Pass |
| Heteroskedasticity | 4 | Visual residual plots | Present | Handled by robust SE |

### 6.3 Sample Sizes by Analysis

| Analysis | N (observations) | N (units) | N (periods) |
|----------|------------------|-----------|-------------|
| ND time series | 15 | 1 | 15 |
| State panel | 765 | 51 | 15 |
| Origin panel (ND) | 707 | 88 | 14 |
| Gravity model | 2,680 | 51x71 | 1 |
| DiD refugee | 1,137 | 126 | 18 |
| Survival analysis | 940 | 940 | 2-13 |

### 6.4 Software and Packages

- Python 3.11+
- statsmodels (time series, panel, GLM)
- linearmodels (panel fixed/random effects)
- scikit-learn (machine learning)
- lifelines (survival analysis)
- ruptures (change point detection)

---

## References

Key methodological references incorporated:
- Ravn, M.O. and Uhlig, H. (2002). On adjusting the Hodrick-Prescott filter.
- Santos Silva, J.M.C. and Tenreyro, S. (2006). The log of gravity.
- Bai, J. and Perron, P. (1998). Estimating and testing linear models with multiple structural changes.

---

## 7. Generated Visualizations

The following summary visualizations are generated by `create_summary_visuals.py` and stored in the `figures/` directory:

| Figure | Description |
|--------|-------------|
| `summary_rigor_assessment.png` | Heatmap of methodological rigor across 9 module groups |
| `summary_key_estimates.png` | Forest plot comparing coefficient estimates across methods |
| `summary_scenarios.png` | Fan chart of scenario projections with Monte Carlo uncertainty bands |
| `summary_consistency.png` | Cross-method consistency matrix for key findings |
| `summary_dashboard.png` | Four-panel dashboard summarizing key trends and effects |

These figures complement the detailed findings in this report and are suitable for presentations and executive summaries.

---

*Report generated: December 29, 2025*
*Analysis pipeline version: 1.0*
*Contact: North Dakota Cohort Component Population Projection System*
