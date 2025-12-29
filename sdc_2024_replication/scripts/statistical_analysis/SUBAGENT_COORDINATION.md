# Sub-Agent Coordination Manifest: Immigration Statistical Analysis

## Overview

This document provides complete specifications for sub-agents implementing the Statistical Analysis Plan for North Dakota immigration flow modeling. Each sub-agent should be able to operate independently with the information provided here.

**Project Root:** `/home/nigel/cohort_projections`
**Analysis Output Directory:** `data/processed/immigration/analysis/`
**Scripts Directory:** `sdc_2024_replication/scripts/statistical_analysis/`

---

## Environment Setup

### Python Environment

```bash
# CRITICAL: Always use this environment
micromamba run -n cohort_proj python <script.py>

# Or activate first
micromamba activate cohort_proj
```

### Installed Packages (Available)

| Package | Version | Capabilities |
|---------|---------|--------------|
| pandas | 2.3.3 | DataFrames, time series |
| numpy | 2.4.0 | Numerical computing |
| scipy | 1.16.3 | Statistical functions, optimization |
| statsmodels | 0.14.6 | Time series, panel data, regression |
| matplotlib | 3.10.8 | Visualization |
| seaborn | 0.13.2 | Statistical visualization |

### Additional Packages (Already Installed)

All required packages have been pre-installed. No additional installation needed.

| Package | Version | Capabilities |
|---------|---------|--------------|
| arch | 8.0.0 | GARCH, structural breaks |
| ruptures | 1.1.10 | Change point detection (PELT, Binseg) |
| pmdarima | 2.1.1 | Auto-ARIMA, model selection |
| linearmodels | 7.0 | Panel data (FE, RE, IV, GMM) |
| scikit-learn | 1.8.0 | ML models, clustering, preprocessing |
| lifelines | 0.30.0 | Survival analysis, hazard models |

---

## Data Dictionary

### Primary Datasets

#### 1. nd_migration_summary.csv
**Path:** `data/processed/immigration/analysis/nd_migration_summary.csv`
**Shape:** 15 rows (years 2010-2024)
**Key for:** Time series analysis, share modeling

| Column | Type | Description |
|--------|------|-------------|
| year | int | Calendar year (2010-2024) |
| nd_intl_migration | int | ND international migration count |
| us_intl_migration | int | US total international migration |
| nd_share_of_us_intl_pct | float | ND share as percentage (0.10-0.30%) |
| nd_share_of_us_pop_pct | float | ND share of US population (~0.23%) |

**Sample Data:**
```
year,nd_intl_migration,us_intl_migration,nd_share_of_us_intl_pct
2010,468,179893,0.260
2017,2875,948392,0.303  # Peak year (Bakken)
2020,30,19885,0.151     # COVID minimum
2024,5126,2786119,0.184 # Recent high
```

---

#### 2. combined_components_of_change.csv
**Path:** `data/processed/immigration/analysis/combined_components_of_change.csv`
**Shape:** 795 rows (53 states/territories × 15 years)
**Key for:** Panel data analysis, state comparisons

| Column | Type | Description |
|--------|------|-------------|
| state | str | State name |
| state_fips | int | FIPS code |
| year | int | Calendar year |
| population | int | Total population |
| pop_change | int | Year-over-year change |
| births | int | Annual births |
| deaths | int | Annual deaths |
| natural_change | int | births - deaths |
| intl_migration | int | International migration |
| domestic_migration | int | Domestic migration |
| net_migration | int | intl + domestic |

**Notes:**
- 2020 has anomalous low values (partial year, COVID)
- Includes all 50 states + DC + PR + territories

---

#### 3. dhs_lpr_by_state_time.parquet
**Path:** `data/processed/immigration/analysis/dhs_lpr_by_state_time.parquet`
**Shape:** 540 rows (54 states × 10 years)
**Key for:** LPR time series, state comparisons

| Column | Type | Description |
|--------|------|-------------|
| state_or_territory | str | State name |
| fiscal_year | int | DHS fiscal year (2014-2023) |
| lpr_count | int | Lawful permanent residents admitted |

---

#### 4. dhs_lpr_by_state_country.parquet
**Path:** `data/processed/immigration/analysis/dhs_lpr_by_state_country.parquet`
**Shape:** 8,619 rows
**Key for:** Origin-country analysis, network effects

| Column | Type | Description |
|--------|------|-------------|
| region_country_of_birth | str | Country or region name |
| state | str | US state |
| lpr_count | int | LPR count |
| fiscal_year | int | FY 2023 only |
| is_region | bool | True if aggregate region |

**LIMITATION:** Only FY2023 data available for state × country detail.

---

#### 5. acs_foreign_born_by_state_origin.parquet
**Path:** `data/processed/immigration/analysis/acs_foreign_born_by_state_origin.parquet`
**Shape:** 125,580 rows
**Key for:** Foreign-born stocks, diaspora analysis

| Column | Type | Description |
|--------|------|-------------|
| year | int | 2009-2023 (15 years) |
| state_fips | int | FIPS code |
| state_name | str | State name |
| variable | str | ACS variable ID |
| region | str | World region |
| sub_region | str | Subregion |
| country | str | Country of birth |
| detail | str | Additional detail level |
| level | str | Hierarchy level |
| foreign_born_pop | int | Population estimate |
| margin_of_error | int | ACS MOE |

---

#### 6. acs_foreign_born_nd_share.parquet
**Path:** `data/processed/immigration/analysis/acs_foreign_born_nd_share.parquet`
**Shape:** 2,414 rows
**Key for:** ND-specific origin analysis, location quotients

| Column | Type | Description |
|--------|------|-------------|
| year | int | 2009-2023 |
| nd_foreign_born | int | ND foreign-born from origin |
| nd_moe | int | ND margin of error |
| national_foreign_born | int | US total from origin |
| national_moe | int | National MOE |
| nd_share_of_national | float | ND/US share |
| country | str | Country of origin |
| region | str | World region |

---

#### 7. refugee_arrivals_by_state_nationality.parquet
**Path:** `data/processed/immigration/analysis/refugee_arrivals_by_state_nationality.parquet`
**Shape:** 15,447 rows
**Key for:** Refugee analysis, policy impact

| Column | Type | Description |
|--------|------|-------------|
| fiscal_year | int | FY 2002-2020 |
| state | str | State name |
| nationality | str | Refugee nationality |
| arrivals | int | Number of arrivals |
| data_source | str | WRAPS or ORR |
| national_total | int | US total for nationality |
| state_share_of_nationality | float | State's share |

---

#### 8. dhs_naturalizations_by_state.parquet
**Path:** `data/processed/immigration/analysis/dhs_naturalizations_by_state.parquet`
**Shape:** 590 rows
**Key for:** Integration analysis, lagged flows

| Column | Type | Description |
|--------|------|-------------|
| state | str | State name |
| fiscal_year | int | FY 2014-2023 |
| naturalizations | int | Naturalization count |
| population_2022 | int | State population |
| naturalizations_per_100k | float | Rate per 100k |

---

#### 9. dhs_naturalizations_by_state_historical.parquet
**Path:** `data/processed/immigration/analysis/dhs_naturalizations_by_state_historical.parquet`
**Shape:** 2,366 rows
**Key for:** Long-run trends

| Column | Type | Description |
|--------|------|-------------|
| state | str | State name |
| fiscal_year | int | FY 1986-2023 (38 years!) |
| naturalizations | int | Naturalization count |

---

#### 10. migration_analysis_results.json
**Path:** `data/processed/immigration/analysis/migration_analysis_results.json`
**Key for:** Baseline parameters, existing analysis

Contains:
- Transfer coefficient: 0.001777 (ND gets 0.178% of US intl migration)
- Linear model: slope=0.00191, R²=0.918
- Policy scenarios: CBO projections, baseline values
- Summary statistics: mean ND share = 0.173%, SD = 0.054%

---

## Module Specifications

**IMPORTANT:** All modules must adhere to:
- **[SPSS-Style Output Standards](#spss-style-output-standards)** - Comprehensive statistics for each test type
- **[Visualization Standards](#visualization-standards)** - Publication-quality figures in PNG and PDF
- **[Autonomous Execution Guidelines](#autonomous-execution-guidelines)** - Decision-making and error handling

### Wave 1: Foundational Analyses (No Dependencies)

#### Module 1.1: Descriptive Statistics Agent

**Output Standards Reference:** See [Descriptive Statistics](#descriptive-statistics-all-variables) in SPSS-Style Output Standards

**Input Files:**
- nd_migration_summary.csv
- combined_components_of_change.csv
- All parquet files (for stream-specific stats)

**Analyses:**
1. Summary statistics by immigration stream (mean, median, SD, IQR, CV, skewness, kurtosis)
2. Hodrick-Prescott trend decomposition (λ=6.25 for annual)
3. First differences and log-differences
4. Distributional tests (Shapiro-Wilk, Jarque-Bera)

**Required SPSS-Style Outputs:**
- Full descriptive statistics table (N, mean, SE, CI, median, mode, SD, variance, skewness, kurtosis, percentiles)
- Normality tests (Shapiro-Wilk, Kolmogorov-Smirnov) with statistics and p-values
- Outlier identification using IQR method

**Output Files:**
- `results/module_1_1_summary_statistics.json`
- `results/module_1_1_trend_decomposition.parquet`
- `figures/module_1_1_time_series_plots.png`
- `figures/module_1_1_histogram_*.png` (one per key variable)
- `figures/module_1_1_qq_plot_*.png` (one per key variable)
- `figures/module_1_1_boxplot_*.png` (one per key variable)

**Code Template:**
```python
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.filters.hp_filter import hpfilter

def hp_decompose(series, lamb=6.25):
    cycle, trend = hpfilter(series, lamb=lamb)
    return {'trend': trend, 'cycle': cycle}
```

---

#### Module 1.2: Geographic Concentration Agent

**Output Standards Reference:** See [Descriptive Statistics](#descriptive-statistics-all-variables) in SPSS-Style Output Standards

**Input Files:**
- dhs_lpr_by_state_country.parquet
- acs_foreign_born_by_state_origin.parquet
- acs_foreign_born_nd_share.parquet

**Analyses:**
1. Herfindahl-Hirschman Index (HHI) by origin country
2. Location Quotients for ND vs US by origin
3. Concentration trends over time (using ACS data)

**Required SPSS-Style Outputs:**
- Descriptive statistics for HHI and LQ distributions
- Correlation matrix between concentration measures
- Time trend analysis with confidence intervals

**Output Files:**
- `results/module_1_2_hhi_by_origin.json`
- `results/module_1_2_location_quotients.parquet`
- `figures/module_1_2_concentration_heatmap.png`
- `figures/module_1_2_lq_bar_chart.png`
- `figures/module_1_2_concentration_trends.png`

**Formulas:**
```
HHI = Σ(share_i)²  where share_i = country_i / total

LQ_nd,c = (Immigrants_ND,c / Immigrants_ND,total) / (Immigrants_US,c / Immigrants_US,total)
```

---

#### Module 2.1.1: Unit Root Testing Agent

**Output Standards Reference:** See [Unit Root Tests](#unit-root-tests) in SPSS-Style Output Standards

**Input Files:**
- nd_migration_summary.csv (nd_share_of_us_intl_pct column)
- combined_components_of_change.csv (for ND rows)

**Analyses:**
1. Augmented Dickey-Fuller test (H0: unit root)
2. KPSS test (H0: stationary)
3. Phillips-Perron test
4. Determine integration order

**Required SPSS-Style Outputs:**
- ADF: test statistic, p-value, optimal lag, critical values (1%, 5%, 10%), regression coefficients
- KPSS: test statistic, p-value, critical values, bandwidth
- Phillips-Perron: test statistic, p-value, critical values
- Integration order conclusion with evidence summary

**Output Files:**
- `results/module_2_1_1_unit_root_tests.json`
- `figures/module_2_1_1_original_series.png`
- `figures/module_2_1_1_differenced_series.png`
- `figures/module_2_1_1_acf_original.png`
- `figures/module_2_1_1_acf_differenced.png`

**Code Template:**
```python
from statsmodels.tsa.stattools import adfuller, kpss

def unit_root_tests(series):
    adf = adfuller(series, autolag='AIC')
    kpss_result = kpss(series, regression='c')
    return {
        'adf': {'statistic': adf[0], 'pvalue': adf[1], 'lags': adf[2]},
        'kpss': {'statistic': kpss_result[0], 'pvalue': kpss_result[1]}
    }
```

---

#### Module 3.1: Panel Data Setup Agent

**Output Standards Reference:** See [Panel Data (Fixed Effects / Random Effects)](#panel-data-fixed-effects--random-effects) in SPSS-Style Output Standards

**Input Files:**
- combined_components_of_change.csv

**Analyses:**
1. Construct balanced panel (50 states × 15 years = 750 obs)
2. Calculate state-level variables (share of US, growth rates)
3. Fixed effects model: Immigration_st = α_s + γ_t + ε_st
4. Random effects model and Hausman test

**Required SPSS-Style Outputs:**
- Panel info: N entities, N periods, N observations, balanced/unbalanced status
- Coefficients with SE, t-values, p-values, 95% CI
- R-squared (within, between, overall)
- F-test for fixed effects, Breusch-Pagan LM test
- Hausman test with recommendation
- Entity effects summary statistics

**Output Files:**
- `results/module_3_1_panel_data.parquet` (constructed panel)
- `results/module_3_1_fixed_effects.json`
- `results/module_3_1_hausman_test.json`
- `figures/module_3_1_entity_effects.png`
- `figures/module_3_1_residuals_vs_fitted.png`
- `figures/module_3_1_within_variation.png`

**Code Template:**
```python
import statsmodels.api as sm
from statsmodels.regression.linear_model import PanelOLS

# Create panel index
panel = df.set_index(['state', 'year'])
# Fixed effects
model = PanelOLS(panel['intl_migration'], panel[['const']], entity_effects=True, time_effects=True)
```

---

#### Module 3.2: Origin-Country Panel Agent

**Input Files:**
- dhs_lpr_by_state_country.parquet
- acs_foreign_born_by_state_origin.parquet

**Analyses:**
1. Diaspora network effect: Does existing stock predict new arrivals?
2. Estimate: log(NewArrivals_c,s) = α + β·log(Stock_c,s,t-1) + ε
3. Network elasticity coefficient

**LIMITATION:** LPR state×country only available for FY2023. Use ACS stock changes as proxy.

**Output Files:**
- `results/module_3_2_network_elasticity.json`
- `results/module_3_2_origin_panel.parquet`

---

### Wave 2: Time Series Modeling (Depends on Wave 1)

#### Module 2.1: ARIMA Modeling Agent

**Output Standards Reference:** See [Time Series - ARIMA/SARIMA](#time-series---arimasarima) in SPSS-Style Output Standards

**Dependencies:** Module 2.1.1 (unit root tests)

**Input Files:**
- nd_migration_summary.csv
- results/module_2_1_1_unit_root_tests.json

**Analyses:**
1. Based on unit root results, determine d (differencing order)
2. Fit ARIMA(p,d,q) with automatic order selection
3. Model diagnostics (Ljung-Box, residual plots)
4. Generate forecasts 2025-2035

**Required SPSS-Style Outputs:**
- Model specification (p,d,q) with selection criteria
- AR/MA coefficients with SE, z-values, p-values, 95% CI
- Fit statistics: AIC, BIC, HQIC, log-likelihood
- Ljung-Box Q at lags 5, 10, 15, 20
- Residual diagnostics: mean, SD, skewness, kurtosis, normality tests
- Forecasts with point estimate, SE, 80% CI, 95% CI

**Output Files:**
- `results/module_2_1_arima_model.json`
- `results/module_2_1_arima_forecasts.parquet`
- `figures/module_2_1_series_with_fitted.png`
- `figures/module_2_1_acf_original.png`
- `figures/module_2_1_pacf_original.png`
- `figures/module_2_1_acf_residuals.png`
- `figures/module_2_1_pacf_residuals.png`
- `figures/module_2_1_residual_histogram.png`
- `figures/module_2_1_residual_qq.png`
- `figures/module_2_1_forecast_fan_chart.png`

**Code Template:**
```python
from statsmodels.tsa.arima.model import ARIMA
# Or install pmdarima for auto_arima
# pip install pmdarima
# from pmdarima import auto_arima
```

---

#### Module 2.1.2: Structural Break Agent

**Output Standards Reference:** See [Structural Break Tests](#structural-break-tests) in SPSS-Style Output Standards

**Dependencies:** Module 2.1.1 (stationarity)

**Input Files:**
- nd_migration_summary.csv
- combined_components_of_change.csv (ND rows)

**Analyses:**
1. Bai-Perron test for multiple breaks
2. Test hypothesized breaks: 2014-15 (Bakken peak), 2017 (Travel Ban), 2020 (COVID)
3. Regime-specific parameters

**Required SPSS-Style Outputs:**
- Number of breaks detected with break dates and confidence intervals
- Test statistics (SupF, UDmax, WDmax if Bai-Perron)
- Regime-specific intercept and slope with SE
- Model comparison: RSS with/without breaks, improvement percentage
- Chow test F-statistics and p-values for each hypothesized break

**Output Files:**
- `results/module_2_1_2_structural_breaks.json`
- `figures/module_2_1_2_series_with_breaks.png`
- `figures/module_2_1_2_regime_lines.png`
- `figures/module_2_1_2_cusum.png`
- `figures/module_2_1_2_rolling_params.png`

**Code Template:**
```python
# May need: pip install ruptures
import ruptures as rpt

# PELT algorithm for change point detection
algo = rpt.Pelt(model="l2").fit(series.values)
breaks = algo.predict(pen=3)
```

---

#### Module 2.2: VAR and Cointegration Agent

**Output Standards Reference:** See [VAR and Cointegration](#var-and-cointegration) in SPSS-Style Output Standards

**Dependencies:** Module 2.1.1 (unit root tests)

**Input Files:**
- combined_components_of_change.csv (ND and US totals)

**Analyses:**
1. Vector Autoregression: [ND_intl, US_intl, ND_domestic]
2. Granger causality tests
3. Johansen cointegration test
4. If cointegrated: Error Correction Model
5. Impulse response functions

**Required SPSS-Style Outputs:**
- Lag selection criteria (AIC, BIC, HQIC, FPE) for lags 1-max
- VAR coefficients with SE, t-values, p-values
- Granger causality F-statistics and p-values (full matrix)
- Johansen trace and max-eigenvalue statistics with critical values
- Cointegrating vectors and adjustment coefficients (if applicable)
- IRF at horizons 1-20 with 95% confidence bands
- FEVD table
- Model stability (eigenvalues inside unit circle)

**Output Files:**
- `results/module_2_2_var_model.json`
- `results/module_2_2_granger_causality.json`
- `results/module_2_2_cointegration.json`
- `figures/module_2_2_time_series_all.png`
- `figures/module_2_2_cross_correlation_heatmap.png`
- `figures/module_2_2_lag_selection.png`
- `figures/module_2_2_impulse_response.png`
- `figures/module_2_2_fevd.png`
- `figures/module_2_2_stability.png`

**Code Template:**
```python
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# VAR model
model = VAR(data[['nd_intl', 'us_intl', 'nd_domestic']])
results = model.fit(maxlags=3, ic='aic')

# Johansen test
johansen = coint_johansen(data, det_order=0, k_ar_diff=2)
```

---

#### Module 4: Beta and Quantile Regression Agent

**Output Standards Reference:** See [Quantile Regression](#quantile-regression) in SPSS-Style Output Standards

**Dependencies:** Module 1.1 (distributional properties)

**Input Files:**
- nd_migration_summary.csv
- combined_components_of_change.csv

**Analyses:**
1. Beta regression for ND share (bounded 0-1)
2. Quantile regression at τ = {0.10, 0.25, 0.50, 0.75, 0.90}
3. Robust regression (M-estimation) for outlier handling

**Required SPSS-Style Outputs:**
- OLS reference coefficients for comparison
- Quantile coefficients at each tau with SE (bootstrap), t-value, p-value, 95% CI
- Pseudo R-squared at each quantile
- Slope equality test across quantiles
- Beta regression coefficients with precision parameter

**Output Files:**
- `results/module_4_beta_regression.json`
- `results/module_4_quantile_regression.json`
- `results/module_4_robust_regression.json`
- `figures/module_4_quantile_coefficients.png`
- `figures/module_4_quantile_lines.png`
- `figures/module_4_ols_comparison.png`

**Code Template:**
```python
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.robust.robust_linear_model import RLM

# Quantile regression
qr = QuantReg(y, X)
for tau in [0.1, 0.25, 0.5, 0.75, 0.9]:
    res = qr.fit(q=tau)
```

---

### Wave 3: Advanced Analyses (Depends on Waves 1-2)

#### Module 5: Gravity and Network Agent

**Dependencies:** Module 3.2 (origin panel)

**Input Files:**
- dhs_lpr_by_state_country.parquet
- acs_foreign_born_by_state_origin.parquet
- results/module_3_2_origin_panel.parquet

**Analyses:**
1. Gravity model specification
2. PPML estimation (Poisson Pseudo-Maximum Likelihood)
3. Diaspora network elasticity with controls

**Output Files:**
- `results/module_5_gravity_model.json`
- `results/module_5_network_effects.json`

**Code Template:**
```python
import statsmodels.api as sm
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Poisson

# PPML
model = GLM(y, X, family=Poisson())
```

---

#### Module 6: Machine Learning Agent

**Output Standards Reference:** See [Clustering Analysis](#clustering-analysis) in SPSS-Style Output Standards

**Dependencies:** All Wave 1 results

**Input Files:**
- combined_components_of_change.csv
- All parquet files
- results/module_1_1_*.json

**Analyses:**
1. Elastic Net regularized regression
2. Random Forest / Gradient Boosting (if sample size permits)
3. Feature importance (SHAP if available)
4. State clustering by immigration profile

**Required SPSS-Style Outputs:**
- Elastic Net: alpha, l1_ratio, coefficients, cross-validation scores
- Feature importance rankings with relative importance scores
- Clustering: method, linkage, distance metric, N clusters
- Cluster sizes and centroid characteristics
- Quality metrics: silhouette score (overall and per cluster), Calinski-Harabasz, Davies-Bouldin
- Optimal K analysis: elbow, silhouette, gap statistic

**Output Files:**
- `results/module_6_elastic_net.json`
- `results/module_6_feature_importance.json`
- `results/module_6_state_clusters.parquet`
- `figures/module_6_cluster_dendrogram.png`
- `figures/module_6_elbow_plot.png`
- `figures/module_6_silhouette_plot.png`
- `figures/module_6_pca_biplot.png`
- `figures/module_6_feature_importance.png`

**Code Template:**
```python
# May need: pip install scikit-learn
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import AgglomerativeClustering
```

---

#### Module 7: Causal Inference Agent

**Output Standards Reference:** See [Causal Inference (Difference-in-Differences)](#causal-inference-difference-in-differences) in SPSS-Style Output Standards

**Dependencies:** Module 3.1 (panel data)

**Input Files:**
- combined_components_of_change.csv
- refugee_arrivals_by_state_nationality.parquet
- results/module_3_1_panel_data.parquet

**Analyses:**
1. Difference-in-Differences for policy events (2017 Travel Ban, 2020 COVID)
2. Event study specification
3. Synthetic Control Method for ND counterfactual
4. Shift-share instrument (Bartik)

**Required SPSS-Style Outputs:**
- Sample info: N treatment/control units, N pre/post periods
- Pre-trend test: joint F-test and period-specific coefficients
- ATT estimate with SE (cluster-robust), t-stat, p-value, 95% CI
- Event study coefficients for each relative time period with CI
- Robustness checks: alternative specifications, different control groups
- Placebo test results

**Output Files:**
- `results/module_7_did_estimates.json`
- `results/module_7_event_study.parquet`
- `results/module_7_synthetic_control.json`
- `figures/module_7_parallel_trends.png`
- `figures/module_7_event_study_plot.png`
- `figures/module_7_did_means.png`
- `figures/module_7_treatment_effect_time_series.png`

**Code Template:**
```python
# DiD specification
# Y_st = α + β(Treated_s × Post_t) + γ_s + δ_t + ε

# Synthetic control - may need custom implementation or package
# pip install SyntheticControlMethods
```

---

#### Module 8: Duration Analysis Agent

**Output Standards Reference:** See [Survival/Hazard Analysis](#survivalhazard-analysis) in SPSS-Style Output Standards

**Dependencies:** Module 1.1 (summary stats)

**Input Files:**
- refugee_arrivals_by_state_nationality.parquet
- acs_foreign_born_by_state_origin.parquet

**Analyses:**
1. Define "immigration wave" (>50% above baseline)
2. Kaplan-Meier survival curves for wave duration
3. Cox proportional hazards model
4. Refugee origin lifecycle analysis (initiation → peak → decline)

**Required SPSS-Style Outputs:**
- Kaplan-Meier: survival probabilities, median survival, 95% CI
- Events summary: N at risk, N events, N censored at each time
- Log-rank test for group comparisons
- Cox PH: hazard ratios with 95% CI, Wald test, p-values
- Model fit: log-likelihood, AIC, BIC, concordance index
- PH assumption: Schoenfeld residuals test (per covariate and global)

**Output Files:**
- `results/module_8_wave_durations.json`
- `results/module_8_hazard_model.json`
- `figures/module_8_survival_curves.png`
- `figures/module_8_cumulative_hazard.png`
- `figures/module_8_forest_plot.png`
- `figures/module_8_schoenfeld_residuals.png`

**Code Template:**
```python
# Survival analysis available in lifelines (may need install)
# pip install lifelines
from lifelines import KaplanMeierFitter, CoxPHFitter
```

---

### Final Wave: Synthesis

#### Module 9: Scenario Modeling Agent

**Dependencies:** All previous modules

**Input Files:**
- All results/*.json files
- migration_analysis_results.json

**Analyses:**
1. Combine forecasts using model averaging
2. Generate scenarios: CBO Full, Moderate, Zero, Pre-2020 Trend
3. Monte Carlo simulation (1000 draws)
4. Confidence intervals and fan charts

**Output Files:**
- `results/module_9_combined_forecasts.parquet`
- `results/module_9_scenario_projections.parquet`
- `results/module_9_monte_carlo.parquet`
- `figures/module_9_fan_chart.png`

---

## Output Consolidation

### Results Directory Structure

```
sdc_2024_replication/scripts/statistical_analysis/
├── SUBAGENT_COORDINATION.md     # This file
├── results/
│   ├── module_1_1_*.json
│   ├── module_1_2_*.json
│   ├── module_2_1_*.json
│   ├── ...
│   └── FINAL_STATISTICAL_REPORT.md
└── figures/
    ├── module_1_1_*.png
    ├── ...
    └── combined_results_dashboard.png
```

### JSON Output Schema

All result files should follow this structure:

```json
{
  "module": "1.1",
  "analysis": "summary_statistics",
  "generated": "2025-12-29T00:00:00Z",
  "input_files": ["file1.csv", "file2.parquet"],
  "parameters": {},
  "results": {},
  "diagnostics": {},
  "warnings": [],
  "next_steps": []
}
```

---

## Execution Order

### Parallel Execution Groups

**Group 1 (Wave 1) - No dependencies:**
- Module 1.1: Descriptive Statistics
- Module 1.2: Geographic Concentration
- Module 2.1.1: Unit Root Testing
- Module 3.1: Panel Data Setup
- Module 3.2: Origin-Country Panel

**Group 2 (Wave 2) - Depends on Group 1:**
- Module 2.1: ARIMA Modeling
- Module 2.1.2: Structural Breaks
- Module 2.2: VAR and Cointegration
- Module 4: Beta/Quantile Regression

**Group 3 (Wave 3) - Depends on Groups 1-2:**
- Module 5: Gravity and Network
- Module 6: Machine Learning
- Module 7: Causal Inference
- Module 8: Duration Analysis

**Group 4 (Final) - Depends on all:**
- Module 9: Scenario Modeling
- Final Report Synthesis

---

## Known Data Limitations

1. **Short time series (n=15):** Limits ARIMA complexity, VAR lag orders
2. **2020 COVID anomaly:** Consider excluding or special handling
3. **LPR state×country:** Only FY2023 available for detailed origin analysis
4. **Refugee data ends 2020:** No post-pandemic refugee arrivals data
5. **Historical naturalizations:** Parsing issues in by-country file (unnamed columns)

---

## SPSS-Style Output Standards

All statistical analyses must produce comprehensive outputs similar to SPSS, not just the main test statistic. This enables "set it and forget it" autonomous execution where no manual feedback is required. Each test type has specific output requirements detailed below.

### Descriptive Statistics (All Variables)

**Mandatory Outputs:**
| Category | Statistics |
|----------|------------|
| Central Tendency | N, Mean, Median, Mode |
| Dispersion | SD, SE, Variance, Range, Min, Max, IQR |
| Shape | Skewness (with SE), Kurtosis (with SE) |
| Percentiles | 5th, 10th, 25th, 50th, 75th, 90th, 95th |
| Derived | Coefficient of Variation (CV = SD/Mean) |
| Confidence | 95% CI for Mean, 95% CI for Median (bootstrap) |
| Normality Tests | Shapiro-Wilk (statistic, p-value), Kolmogorov-Smirnov (statistic, p-value) |

**Mandatory Visualizations:**
1. Histogram with normal curve overlay
2. Q-Q plot (quantile-quantile)
3. Box plot with outlier identification
4. Density plot

**JSON Output Format:**
```json
{
  "variable": "nd_share_of_us_intl_pct",
  "n": 15,
  "mean": 0.173,
  "se_mean": 0.014,
  "ci_mean_95": [0.145, 0.201],
  "median": 0.177,
  "ci_median_95": [0.151, 0.195],
  "mode": null,
  "sd": 0.054,
  "variance": 0.0029,
  "skewness": 0.234,
  "se_skewness": 0.580,
  "kurtosis": -0.876,
  "se_kurtosis": 1.121,
  "range": 0.152,
  "min": 0.103,
  "max": 0.303,
  "iqr": 0.078,
  "cv": 0.312,
  "percentiles": {"5": 0.110, "10": 0.121, "25": 0.144, "50": 0.177, "75": 0.222, "90": 0.265, "95": 0.289},
  "normality": {
    "shapiro_wilk": {"statistic": 0.923, "p_value": 0.215},
    "kolmogorov_smirnov": {"statistic": 0.143, "p_value": 0.678}
  },
  "outliers": {"mild": [], "extreme": []}
}
```

---

### Correlation and Simple Regression

**Mandatory Outputs:**
| Category | Statistics |
|----------|------------|
| Correlation | Pearson r, Spearman rho, Kendall tau (each with p-value and 95% CI) |
| Model Fit | R, R-squared, Adjusted R-squared, SE of Estimate |
| ANOVA Table | Sum of Squares (Regression, Residual, Total), df, Mean Square, F, p-value |
| Coefficients | Unstandardized B, SE of B, Standardized Beta, t-value, p-value, 95% CI for B |
| Assumptions | Durbin-Watson, Homoscedasticity test (Breusch-Pagan), Normality of residuals |

**Mandatory Visualizations:**
1. Scatter plot with regression line and 95% CI band
2. Predicted vs. Actual values plot
3. Residual vs. Predicted plot (for heteroscedasticity)
4. Q-Q plot of residuals
5. Histogram of residuals with normal curve

**JSON Output Format:**
```json
{
  "correlation": {
    "pearson": {"r": 0.958, "p_value": 0.000, "ci_95": [0.874, 0.987]},
    "spearman": {"rho": 0.943, "p_value": 0.000},
    "kendall": {"tau": 0.829, "p_value": 0.000}
  },
  "model_summary": {
    "r": 0.958,
    "r_squared": 0.918,
    "adj_r_squared": 0.912,
    "se_estimate": 0.016,
    "durbin_watson": 1.842
  },
  "anova": {
    "regression": {"ss": 0.0374, "df": 1, "ms": 0.0374, "f": 145.67, "p_value": 0.000},
    "residual": {"ss": 0.0033, "df": 13, "ms": 0.00025},
    "total": {"ss": 0.0407, "df": 14}
  },
  "coefficients": {
    "intercept": {"b": -3.651, "se": 0.321, "t": -11.37, "p_value": 0.000, "ci_95": [-4.345, -2.957]},
    "predictor": {"b": 0.00191, "se": 0.00016, "beta": 0.958, "t": 12.07, "p_value": 0.000, "ci_95": [0.00157, 0.00225]}
  },
  "assumptions": {
    "breusch_pagan": {"statistic": 2.34, "p_value": 0.126},
    "shapiro_wilk_residuals": {"statistic": 0.956, "p_value": 0.634},
    "durbin_watson": 1.842
  }
}
```

---

### Multiple Regression

**Mandatory Outputs (in addition to Simple Regression):**
| Category | Statistics |
|----------|------------|
| Collinearity | Tolerance, VIF for each predictor |
| Influence | Cook's Distance (max, mean, observations > 1), Leverage values (max, mean) |
| Change Stats | R-squared change, F change, df1, df2, p-value of change (for hierarchical) |
| Part Correlations | Zero-order, Partial, Part (semi-partial) for each predictor |

**Additional Visualizations:**
1. Partial regression plots for each predictor
2. Cook's Distance plot
3. Leverage plot
4. Correlation matrix heatmap
5. VIF bar chart

**JSON Output Format (additions):**
```json
{
  "collinearity": {
    "predictor_1": {"tolerance": 0.654, "vif": 1.53},
    "predictor_2": {"tolerance": 0.654, "vif": 1.53}
  },
  "influence": {
    "cooks_distance": {"max": 0.234, "mean": 0.067, "n_gt_1": 0, "problematic_obs": []},
    "leverage": {"max": 0.312, "mean": 0.133, "threshold": 0.40}
  },
  "correlations": {
    "predictor_1": {"zero_order": 0.834, "partial": 0.723, "part": 0.612}
  }
}
```

---

### Time Series - ARIMA/SARIMA

**Mandatory Outputs:**
| Category | Statistics |
|----------|------------|
| Model Specification | Order (p,d,q), Seasonal Order (P,D,Q,s) if applicable |
| Parameters | AR/MA coefficients with SE, z-values, p-values, 95% CI |
| Model Fit | AIC, BIC, HQIC, Log-likelihood |
| Diagnostics | Ljung-Box Q at lags 5, 10, 15, 20 (statistic, df, p-value) |
| Residuals | Mean, SD, Skewness, Kurtosis, normality tests |
| Forecasts | Point forecast, SE, 80% CI, 95% CI for each horizon |

**Mandatory Visualizations:**
1. Original series with fitted values
2. ACF of original series (with significance bands)
3. PACF of original series (with significance bands)
4. ACF of residuals
5. PACF of residuals
6. Residual histogram with normal curve
7. Q-Q plot of residuals
8. Residual time series plot
9. Forecast plot with confidence intervals (fan chart)

**JSON Output Format:**
```json
{
  "model": "ARIMA(1,1,1)",
  "parameters": {
    "ar_1": {"coef": 0.654, "se": 0.145, "z": 4.51, "p_value": 0.000, "ci_95": [0.370, 0.938]},
    "ma_1": {"coef": -0.234, "se": 0.178, "z": -1.31, "p_value": 0.189, "ci_95": [-0.583, 0.115]},
    "sigma2": {"coef": 0.00234, "se": 0.00089}
  },
  "fit_statistics": {
    "aic": -45.67,
    "bic": -42.34,
    "hqic": -44.89,
    "log_likelihood": 25.84
  },
  "ljung_box": {
    "lag_5": {"statistic": 3.45, "df": 3, "p_value": 0.327},
    "lag_10": {"statistic": 8.23, "df": 8, "p_value": 0.412},
    "lag_15": {"statistic": 12.45, "df": 13, "p_value": 0.492},
    "lag_20": {"statistic": 16.78, "df": 18, "p_value": 0.540}
  },
  "residual_diagnostics": {
    "mean": 0.0003,
    "sd": 0.048,
    "skewness": 0.234,
    "kurtosis": 2.89,
    "shapiro_wilk": {"statistic": 0.967, "p_value": 0.834}
  },
  "forecasts": [
    {"horizon": 1, "point": 0.185, "se": 0.048, "ci_80": [0.123, 0.247], "ci_95": [0.091, 0.279]},
    {"horizon": 2, "point": 0.189, "se": 0.062, "ci_80": [0.110, 0.268], "ci_95": [0.067, 0.311]}
  ]
}
```

---

### Panel Data (Fixed Effects / Random Effects)

**Mandatory Outputs:**
| Category | Statistics |
|----------|------------|
| Model Info | N entities, N time periods, N observations, balanced/unbalanced |
| Coefficients | Estimate, SE, t-value, p-value, 95% CI (robust SE if applicable) |
| Fit Statistics | R-squared (within, between, overall), F-statistic, p-value |
| Entity Effects | Individual fixed effects estimates (or variance component for RE) |
| Time Effects | Time fixed effects estimates (if included) |
| Model Tests | F-test for fixed effects, Breusch-Pagan LM test for random effects |
| Model Selection | Hausman test statistic, df, p-value, recommendation |
| Cluster Diagnostics | Cluster-robust SE if used, number of clusters |

**Mandatory Visualizations:**
1. Entity effects distribution (histogram or dot plot)
2. Time effects plot (if included)
3. Residual vs. fitted plot
4. Within-entity variation plot (spaghetti plot)
5. Between-entity comparison plot

**JSON Output Format:**
```json
{
  "model_type": "Two-way Fixed Effects",
  "panel_info": {
    "n_entities": 50,
    "n_periods": 15,
    "n_observations": 750,
    "balanced": true
  },
  "coefficients": {
    "x1": {"estimate": 0.523, "se": 0.089, "t": 5.88, "p_value": 0.000, "ci_95": [0.348, 0.698]},
    "x2": {"estimate": -0.134, "se": 0.045, "t": -2.98, "p_value": 0.003, "ci_95": [-0.222, -0.046]}
  },
  "r_squared": {
    "within": 0.456,
    "between": 0.234,
    "overall": 0.312
  },
  "f_test_fe": {"statistic": 45.67, "df1": 49, "df2": 685, "p_value": 0.000},
  "breusch_pagan_lm": {"statistic": 234.56, "p_value": 0.000},
  "hausman_test": {"statistic": 34.56, "df": 2, "p_value": 0.000, "recommendation": "Fixed Effects"},
  "entity_effects_summary": {"mean": 0.0, "sd": 2345.6, "min": -5678.9, "max": 4567.8},
  "cluster_robust_se": true
}
```

---

### Unit Root Tests

**Mandatory Outputs:**
| Category | Statistics |
|----------|------------|
| ADF Test | Test statistic, p-value, optimal lag (by AIC), critical values (1%, 5%, 10%) |
| ADF Regression | Coefficients (rho-1, trend, constant), SE, t-values |
| KPSS Test | Test statistic, p-value, critical values, bandwidth |
| Phillips-Perron | Test statistic, p-value, critical values |
| Integration Order | Conclusion on d (0, 1, or 2), evidence summary |

**Mandatory Visualizations:**
1. Original series plot
2. First-differenced series plot
3. ACF of original series
4. ACF of differenced series

**JSON Output Format:**
```json
{
  "series": "nd_share_of_us_intl_pct",
  "n_observations": 15,
  "adf": {
    "statistic": -2.345,
    "p_value": 0.156,
    "used_lag": 2,
    "n_obs_used": 12,
    "critical_values": {"1%": -3.959, "5%": -3.081, "10%": -2.681},
    "regression": {
      "rho_minus_1": {"coef": -0.456, "se": 0.194, "t": -2.35},
      "constant": {"coef": 0.089, "se": 0.034, "t": 2.62}
    }
  },
  "kpss": {
    "statistic": 0.234,
    "p_value": 0.100,
    "critical_values": {"1%": 0.739, "5%": 0.463, "10%": 0.347},
    "lags_used": 3
  },
  "phillips_perron": {
    "statistic": -2.567,
    "p_value": 0.098,
    "critical_values": {"1%": -3.959, "5%": -3.081, "10%": -2.681}
  },
  "conclusion": {
    "integration_order": 1,
    "evidence": "ADF fails to reject unit root at 5%, KPSS fails to reject stationarity - borderline I(1)"
  }
}
```

---

### Structural Break Tests

**Mandatory Outputs:**
| Category | Statistics |
|----------|------------|
| Break Detection | Number of breaks detected, break dates, confidence intervals for dates |
| Test Statistics | SupF, UDmax, WDmax statistics (if Bai-Perron) |
| Segment Parameters | Intercept and slope for each regime, SE, significance |
| Comparison | RSS pre-break, RSS post-break, RSS full sample |
| Chow Test | F-statistic, df, p-value for each hypothesized break |

**Mandatory Visualizations:**
1. Time series with vertical lines at break points
2. Segment-specific regression lines
3. CUSUM plot
4. Residual plot by segment
5. Parameter stability plot (rolling estimates)

**JSON Output Format:**
```json
{
  "method": "Bai-Perron",
  "n_breaks_detected": 2,
  "breaks": [
    {"date": 2015, "ci_90": [2014, 2016], "f_statistic": 12.34, "p_value": 0.001},
    {"date": 2020, "ci_90": [2019, 2020], "f_statistic": 23.45, "p_value": 0.000}
  ],
  "regime_parameters": {
    "regime_1": {"period": "2010-2014", "intercept": 0.145, "se": 0.023, "slope": 0.012, "slope_se": 0.004},
    "regime_2": {"period": "2015-2019", "intercept": 0.234, "se": 0.019, "slope": -0.008, "slope_se": 0.005},
    "regime_3": {"period": "2020-2024", "intercept": 0.156, "se": 0.031, "slope": 0.015, "slope_se": 0.008}
  },
  "model_comparison": {
    "rss_no_break": 0.0456,
    "rss_with_breaks": 0.0234,
    "improvement_pct": 48.7
  },
  "chow_tests": {
    "2015": {"f_statistic": 12.34, "df1": 2, "df2": 10, "p_value": 0.002},
    "2017": {"f_statistic": 8.67, "df1": 2, "df2": 10, "p_value": 0.007},
    "2020": {"f_statistic": 23.45, "df1": 2, "df2": 10, "p_value": 0.000}
  }
}
```

---

### VAR and Cointegration

**Mandatory Outputs:**
| Category | Statistics |
|----------|------------|
| Lag Selection | AIC, BIC, HQIC, FPE for lags 1 through max_lag, optimal lag |
| VAR Coefficients | Full coefficient matrices with SE, t-values, p-values |
| Granger Causality | F-statistic, df, p-value for each direction (full matrix) |
| Model Diagnostics | Portmanteau test, normality test, stability (eigenvalues) |
| Johansen Test | Trace and Max-Eigenvalue statistics, critical values, rank |
| Cointegrating Vectors | Beta coefficients, loading coefficients (alpha), SE |
| IRF | Impulse responses at horizons 1-20 with 95% confidence bands |
| FEVD | Forecast error variance decomposition table |

**Mandatory Visualizations:**
1. Time series of all variables
2. Cross-correlation matrix heatmap
3. Lag selection criteria plot
4. Impulse response function plots (grid for all variable pairs)
5. Forecast error variance decomposition stacked area plots
6. VAR stability plot (roots inside unit circle)
7. Cointegrating relationship plot (if cointegrated)

**JSON Output Format:**
```json
{
  "variables": ["nd_intl", "us_intl", "nd_domestic"],
  "lag_selection": {
    "criteria": {
      "lag_1": {"aic": -12.34, "bic": -11.89, "hqic": -12.15, "fpe": 0.000012},
      "lag_2": {"aic": -12.67, "bic": -11.78, "hqic": -12.34, "fpe": 0.000011},
      "lag_3": {"aic": -12.45, "bic": -11.12, "hqic": -11.98, "fpe": 0.000013}
    },
    "optimal": {"aic": 2, "bic": 1, "hqic": 2, "selected": 2}
  },
  "granger_causality": {
    "us_intl -> nd_intl": {"f_stat": 4.56, "df": [2, 10], "p_value": 0.034},
    "nd_intl -> us_intl": {"f_stat": 0.89, "df": [2, 10], "p_value": 0.456},
    "nd_domestic -> nd_intl": {"f_stat": 2.34, "df": [2, 10], "p_value": 0.145}
  },
  "johansen_test": {
    "trace": [
      {"h0_rank": 0, "statistic": 45.67, "critical_5pct": 29.68, "p_value": 0.001},
      {"h0_rank": 1, "statistic": 12.34, "critical_5pct": 15.41, "p_value": 0.234}
    ],
    "max_eigenvalue": [
      {"h0_rank": 0, "statistic": 33.33, "critical_5pct": 20.97, "p_value": 0.002},
      {"h0_rank": 1, "statistic": 8.45, "critical_5pct": 14.07, "p_value": 0.345}
    ],
    "conclusion": {"rank": 1, "evidence": "One cointegrating vector"}
  },
  "cointegrating_vector": {
    "beta": [1.000, -0.178, 0.045],
    "alpha": [-0.234, 0.012, 0.089],
    "se_alpha": [0.089, 0.034, 0.056]
  },
  "stability": {
    "eigenvalues": [0.89, 0.76, 0.65, 0.54, 0.43, 0.32],
    "all_inside_unit_circle": true
  },
  "diagnostics": {
    "portmanteau": {"statistic": 23.45, "df": 18, "p_value": 0.178},
    "normality": {"statistic": 8.67, "df": 6, "p_value": 0.193}
  }
}
```

---

### Quantile Regression

**Mandatory Outputs:**
| Category | Statistics |
|----------|------------|
| Quantiles | Results for tau = 0.10, 0.25, 0.50, 0.75, 0.90 |
| Coefficients | Estimate, SE (bootstrap), t-value, p-value, 95% CI at each quantile |
| Fit Statistics | Pseudo R-squared at each quantile |
| Comparison | OLS coefficients for reference, equality tests across quantiles |
| Bandwidth | Bandwidth parameter used |

**Mandatory Visualizations:**
1. Coefficient plot across quantiles (with CI bands)
2. Scatter plot with quantile regression lines at all quantiles
3. Comparison with OLS line
4. Residual plots at each quantile

**JSON Output Format:**
```json
{
  "ols_reference": {"intercept": 0.123, "slope": 0.456, "se_slope": 0.089},
  "quantile_results": {
    "0.10": {
      "intercept": {"coef": 0.098, "se": 0.045, "t": 2.18, "p_value": 0.048, "ci_95": [0.009, 0.187]},
      "slope": {"coef": 0.389, "se": 0.112, "t": 3.47, "p_value": 0.004, "ci_95": [0.170, 0.608]},
      "pseudo_r2": 0.234
    },
    "0.25": {
      "intercept": {"coef": 0.112, "se": 0.038, "t": 2.95, "p_value": 0.011, "ci_95": [0.037, 0.187]},
      "slope": {"coef": 0.423, "se": 0.098, "t": 4.32, "p_value": 0.001, "ci_95": [0.231, 0.615]},
      "pseudo_r2": 0.287
    },
    "0.50": {
      "intercept": {"coef": 0.134, "se": 0.032, "t": 4.19, "p_value": 0.001, "ci_95": [0.071, 0.197]},
      "slope": {"coef": 0.456, "se": 0.087, "t": 5.24, "p_value": 0.000, "ci_95": [0.286, 0.626]},
      "pseudo_r2": 0.334
    },
    "0.75": {
      "intercept": {"coef": 0.156, "se": 0.041, "t": 3.80, "p_value": 0.002, "ci_95": [0.076, 0.236]},
      "slope": {"coef": 0.489, "se": 0.103, "t": 4.75, "p_value": 0.000, "ci_95": [0.287, 0.691]},
      "pseudo_r2": 0.312
    },
    "0.90": {
      "intercept": {"coef": 0.178, "se": 0.052, "t": 3.42, "p_value": 0.004, "ci_95": [0.076, 0.280]},
      "slope": {"coef": 0.523, "se": 0.134, "t": 3.90, "p_value": 0.002, "ci_95": [0.261, 0.785]},
      "pseudo_r2": 0.278
    }
  },
  "slope_equality_test": {"f_stat": 2.34, "df1": 4, "df2": 50, "p_value": 0.067}
}
```

---

### Clustering Analysis

**Mandatory Outputs:**
| Category | Statistics |
|----------|------------|
| Cluster Summary | Number of clusters, method, linkage (if hierarchical), distance metric |
| Cluster Sizes | N in each cluster |
| Centroids | Mean values for each variable in each cluster |
| Quality Metrics | Silhouette score (overall and per cluster), Calinski-Harabasz, Davies-Bouldin |
| Cluster Membership | Assignment for each observation |
| Optimal K | Elbow method, silhouette method, gap statistic results |

**Mandatory Visualizations:**
1. Dendrogram (if hierarchical)
2. Elbow plot (for K-means)
3. Silhouette plot
4. PCA biplot with cluster coloring
5. Cluster profile radar chart
6. Heatmap of cluster centroids

**JSON Output Format:**
```json
{
  "method": "Agglomerative",
  "linkage": "ward",
  "distance_metric": "euclidean",
  "n_clusters": 4,
  "cluster_sizes": {"1": 12, "2": 18, "3": 8, "4": 12},
  "quality_metrics": {
    "silhouette_overall": 0.456,
    "silhouette_by_cluster": {"1": 0.512, "2": 0.423, "3": 0.489, "4": 0.401},
    "calinski_harabasz": 234.56,
    "davies_bouldin": 0.789
  },
  "centroids": {
    "cluster_1": {"var1": 0.234, "var2": 1234.5, "var3": 0.567},
    "cluster_2": {"var1": 0.567, "var2": 2345.6, "var3": 0.234}
  },
  "optimal_k_analysis": {
    "elbow": {"suggested_k": 4, "inertias": [1000, 500, 300, 200, 180, 170]},
    "silhouette": {"suggested_k": 3, "scores": [0.312, 0.423, 0.456, 0.445, 0.398]},
    "gap_statistic": {"suggested_k": 4, "gaps": [0.234, 0.345, 0.423, 0.456, 0.412]}
  },
  "cluster_assignments": {"North Dakota": 2, "Minnesota": 2, "Montana": 3}
}
```

---

### Survival/Hazard Analysis

**Mandatory Outputs:**
| Category | Statistics |
|----------|------------|
| Kaplan-Meier | Survival probabilities at time points, median survival, 95% CI |
| Events Summary | N at risk, N events, N censored at each time |
| Log-rank Test | Chi-square statistic, df, p-value (for group comparisons) |
| Cox PH Model | Hazard ratios, 95% CI, Wald test, p-values for each covariate |
| Model Fit | Log-likelihood, AIC, BIC, concordance index (C-statistic) |
| PH Assumption | Schoenfeld residuals test for each covariate, global test |

**Mandatory Visualizations:**
1. Kaplan-Meier survival curves with 95% CI
2. Cumulative hazard plot
3. Risk table below survival plot
4. Forest plot of hazard ratios
5. Schoenfeld residuals plot (for PH assumption)
6. Log-log plot (for PH assumption)
7. Martingale residuals vs. covariates

**JSON Output Format:**
```json
{
  "kaplan_meier": {
    "n_subjects": 50,
    "n_events": 35,
    "n_censored": 15,
    "median_survival": 8.5,
    "median_ci_95": [6.2, 11.3],
    "survival_table": [
      {"time": 1, "at_risk": 50, "events": 5, "survival": 0.90, "se": 0.042, "ci_95": [0.82, 0.98]},
      {"time": 2, "at_risk": 45, "events": 8, "survival": 0.74, "se": 0.062, "ci_95": [0.62, 0.86]}
    ]
  },
  "log_rank_test": {"chi_square": 12.34, "df": 3, "p_value": 0.006},
  "cox_ph": {
    "coefficients": {
      "size": {"coef": 0.456, "hr": 1.578, "se": 0.123, "z": 3.71, "p_value": 0.000, "hr_ci_95": [1.24, 2.01]},
      "type": {"coef": -0.234, "hr": 0.791, "se": 0.098, "z": -2.39, "p_value": 0.017, "hr_ci_95": [0.65, 0.96]}
    },
    "fit": {
      "log_likelihood": -145.67,
      "aic": 295.34,
      "bic": 299.87,
      "concordance": 0.723,
      "concordance_se": 0.034
    },
    "ph_test": {
      "size": {"rho": 0.089, "chi_square": 0.78, "p_value": 0.378},
      "type": {"rho": -0.123, "chi_square": 1.45, "p_value": 0.229},
      "global": {"chi_square": 2.23, "df": 2, "p_value": 0.328}
    }
  }
}
```

---

### Causal Inference (Difference-in-Differences)

**Mandatory Outputs:**
| Category | Statistics |
|----------|------------|
| Pre-Trend Test | Coefficient on treatment × pre-period interactions, joint F-test |
| Treatment Effect | ATT estimate, SE (clustered if applicable), t-value, p-value, 95% CI |
| Event Study | Coefficients for each relative time period, SE, 95% CI |
| Parallel Trends | Visual assessment, placebo test results |
| Robustness | Alternative specifications, different control groups |
| Sample Sizes | N treatment, N control, N pre-period, N post-period |

**Mandatory Visualizations:**
1. Parallel trends plot (pre-treatment period)
2. Event study coefficient plot with 95% CI
3. Difference-in-differences plot (2x2 means)
4. Treatment effect time series
5. Placebo test distribution

**JSON Output Format:**
```json
{
  "sample_info": {
    "n_treatment_units": 1,
    "n_control_units": 49,
    "n_pre_periods": 7,
    "n_post_periods": 3,
    "total_observations": 500
  },
  "pre_trend_test": {
    "joint_f_test": {"f_stat": 1.23, "df1": 6, "df2": 485, "p_value": 0.289},
    "period_coefficients": {
      "t_minus_6": {"coef": 0.012, "se": 0.089, "p_value": 0.893},
      "t_minus_5": {"coef": -0.034, "se": 0.085, "p_value": 0.689}
    }
  },
  "did_estimate": {
    "att": -0.234,
    "se": 0.089,
    "se_type": "cluster-robust",
    "n_clusters": 50,
    "t_stat": -2.63,
    "p_value": 0.011,
    "ci_95": [-0.412, -0.056]
  },
  "event_study": {
    "t_minus_3": {"coef": 0.012, "se": 0.078, "ci_95": [-0.141, 0.165]},
    "t_minus_2": {"coef": 0.023, "se": 0.076, "ci_95": [-0.126, 0.172]},
    "t_minus_1": {"coef": 0.000, "se": 0.000, "ci_95": [0.000, 0.000]},
    "t_0": {"coef": -0.189, "se": 0.082, "ci_95": [-0.350, -0.028]},
    "t_plus_1": {"coef": -0.234, "se": 0.089, "ci_95": [-0.408, -0.060]},
    "t_plus_2": {"coef": -0.278, "se": 0.095, "ci_95": [-0.464, -0.092]}
  },
  "robustness": {
    "no_controls": {"att": -0.245, "se": 0.092, "p_value": 0.008},
    "state_trends": {"att": -0.212, "se": 0.098, "p_value": 0.031},
    "drop_border_states": {"att": -0.256, "se": 0.094, "p_value": 0.007}
  }
}
```

---

## Visualization Standards

All visualizations produced by sub-agents must meet publication-quality standards to ensure consistency and professional presentation.

### Technical Requirements

| Specification | Requirement |
|--------------|-------------|
| Resolution | 300 DPI minimum (use `dpi=300` in `savefig()`) |
| Formats | Save as both PNG and PDF for each figure |
| Figure Size | Default 10x8 inches, adjust for multi-panel |
| Font Family | Sans-serif (Arial, Helvetica) for readability |
| Font Sizes | Title: 14pt, Axis labels: 12pt, Tick labels: 10pt, Legend: 10pt |
| Line Widths | Main lines: 2pt, Reference lines: 1pt, CI bounds: 1pt dashed |
| Color Scheme | Use colorblind-safe palette (see below) |

### Standard Color Palette

Use this colorblind-safe palette consistently across all modules:

```python
# Primary palette (colorblind-safe)
COLORS = {
    'primary': '#0072B2',      # Blue
    'secondary': '#D55E00',    # Vermillion/Orange
    'tertiary': '#009E73',     # Teal/Green
    'quaternary': '#CC79A7',   # Pink
    'highlight': '#F0E442',    # Yellow
    'neutral': '#999999',      # Gray
    'ci_fill': '#0072B2',      # Blue with alpha=0.2
}

# For sequential data (single hue)
SEQUENTIAL = 'Blues'  # matplotlib colormap

# For diverging data (pos/neg)
DIVERGING = 'RdBu_r'  # matplotlib colormap

# For categorical (many groups)
CATEGORICAL = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442', '#56B4E9', '#E69F00', '#999999']
```

### Required Figure Elements

Every figure must include:

1. **Title**: Descriptive, stating the analysis and time period
2. **Axis Labels**: Clear labels with units (e.g., "International Migration (persons)")
3. **Legend**: When multiple series/groups present, positioned to not obscure data
4. **Source Note**: "Source: [data source]" in bottom-left
5. **Grid**: Light gray gridlines for readability (alpha=0.3)
6. **Tight Layout**: Use `plt.tight_layout()` to prevent label clipping

### Figure Naming Convention

```
module_{module_number}_{analysis_type}_{variant}.{png|pdf}

Examples:
- module_1_1_histogram_nd_share.png
- module_2_1_arima_forecast.pdf
- module_7_event_study_plot.png
```

### Standard Code Template

```python
import matplotlib.pyplot as plt
import seaborn as sns

def setup_figure(figsize=(10, 8)):
    """Standard figure setup for all visualizations."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    return fig, ax

def save_figure(fig, filepath_base, title, source_note):
    """Save figure in both PNG and PDF formats."""
    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.text(0.02, 0.02, f"Source: {source_note}", fontsize=8,
             fontstyle='italic', transform=fig.transFigure)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Leave room for title and source

    # Save both formats
    fig.savefig(f"{filepath_base}.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(f"{filepath_base}.pdf", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)

# Example usage
fig, ax = setup_figure()
ax.plot(x, y, color='#0072B2', linewidth=2, label='ND Share')
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Share of US International Migration (%)', fontsize=12)
ax.legend(loc='best', fontsize=10)
save_figure(fig, 'figures/module_1_1_time_series',
            'North Dakota Share of US International Migration (2010-2024)',
            'Census Bureau Population Estimates Program')
```

### Minimum Visualizations Per Module

Each module should produce at least the following number of figures:

| Module | Minimum Figures | Required Types |
|--------|-----------------|----------------|
| 1.1 Descriptive | 4 | Histogram, Q-Q, Box plot, Time series |
| 1.2 Geographic | 3 | Heatmap, LQ bar chart, Concentration trends |
| 2.1 ARIMA | 5 | ACF, PACF, Residuals, Forecast, Diagnostics |
| 2.1.1 Unit Root | 2 | Original series, Differenced series |
| 2.1.2 Structural Breaks | 3 | Series with breaks, CUSUM, Regime params |
| 2.2 VAR | 5 | IRF grid, FEVD, Stability, Time series, Granger |
| 3.1 Panel | 3 | Entity effects, Residuals, Within variation |
| 3.2 Origin Panel | 3 | Network effects, Origin heatmap, Elasticity |
| 4 Quantile | 3 | Quantile coefficients, Multi-line regression, Comparison |
| 5 Gravity | 3 | Gravity fit, Network effects, PPML diagnostics |
| 6 ML | 4 | Feature importance, Clusters, Dendrogram, Validation |
| 7 Causal | 4 | Pre-trends, Event study, DiD means, Robustness |
| 8 Duration | 3 | KM curves, Hazard plot, Cox diagnostics |
| 9 Scenarios | 3 | Fan chart, Scenario comparison, Monte Carlo |

---

## Autonomous Execution Guidelines

Sub-agents are expected to execute independently without manual feedback. These guidelines ensure consistent decision-making and robust error handling.

### Decision-Making Principles

1. **Default to Conservative Assumptions**
   - When choosing between assumptions, select the more conservative option
   - Example: If testing at 5% vs 10% significance, use 5%
   - Example: If uncertain about outlier removal, keep the data point

2. **Document All Decisions**
   - Every non-trivial decision must be logged with rationale
   - Include in the `decisions` field of output JSON
   - Format: `{"decision": "...", "rationale": "...", "alternatives_considered": [...]}`

3. **Proceed with Alternatives When Blocked**
   - If primary analysis fails, attempt backup approaches
   - Example: If ARIMA(2,1,2) fails to converge, try ARIMA(1,1,1)
   - Example: If Johansen test requires more data, report limitation and skip

4. **Never Halt Execution for Missing Optional Data**
   - Required data missing: Log error, skip module, continue
   - Optional data missing: Use available data, note limitation

### Error Handling Protocol

```python
def execute_analysis_safely(analysis_func, *args, **kwargs):
    """Standard error handling wrapper for all analyses."""
    result = {
        "status": "success",
        "warnings": [],
        "errors": [],
        "fallback_used": False
    }

    try:
        output = analysis_func(*args, **kwargs)
        result["output"] = output
    except Exception as e:
        result["status"] = "failed"
        result["errors"].append({
            "error_type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc()
        })

        # Attempt fallback if available
        if hasattr(analysis_func, 'fallback'):
            try:
                output = analysis_func.fallback(*args, **kwargs)
                result["status"] = "partial_success"
                result["fallback_used"] = True
                result["output"] = output
            except Exception as e2:
                result["errors"].append({
                    "error_type": type(e2).__name__,
                    "message": f"Fallback also failed: {str(e2)}"
                })

    return result
```

### Standard Decision Log Format

```json
{
  "decisions": [
    {
      "decision_id": "D001",
      "timestamp": "2025-12-29T10:30:00Z",
      "category": "data_handling",
      "decision": "Excluded year 2020 from time series analysis",
      "rationale": "COVID-19 caused anomalous values (intl migration = 30) that would distort trend estimation",
      "alternatives_considered": [
        "Include with dummy variable",
        "Winsorize to 1st percentile",
        "Include without adjustment"
      ],
      "evidence": "Value is 16.5 standard deviations below mean",
      "reversible": true
    },
    {
      "decision_id": "D002",
      "timestamp": "2025-12-29T10:35:00Z",
      "category": "model_selection",
      "decision": "Selected ARIMA(1,1,0) over ARIMA(2,1,1)",
      "rationale": "Lower BIC (difference = 2.3), MA term not significant (p=0.34)",
      "alternatives_considered": [
        "ARIMA(2,1,1) - lower AIC but higher BIC",
        "ARIMA(0,1,1) - higher both criteria"
      ],
      "evidence": "BIC values: (1,1,0)=-42.3, (2,1,1)=-40.0",
      "reversible": true
    }
  ]
}
```

### Handling Common Scenarios

| Scenario | Action | Log Level |
|----------|--------|-----------|
| Data file not found | Skip module, log error, continue | ERROR |
| Insufficient observations for test | Use simpler alternative or skip test | WARNING |
| Model fails to converge | Try simpler specification | WARNING |
| Assumption violated (mild) | Proceed with robust SE, note limitation | WARNING |
| Assumption violated (severe) | Use alternative method, document | WARNING |
| Multiple comparison issue | Apply Bonferroni correction | INFO |
| Outliers detected | Keep in primary, run sensitivity without | INFO |
| Missing values | Use listwise deletion, report N lost | INFO |
| Multicollinearity (VIF > 10) | Remove one variable, document choice | WARNING |

### Quality Assurance Checklist

Each module must verify before completing:

```python
QUALITY_CHECKLIST = {
    "data_validation": [
        "Input files exist and are readable",
        "No unexpected missing values in key columns",
        "Data types match expectations",
        "Sample size meets minimum requirements"
    ],
    "analysis_integrity": [
        "All required statistics computed",
        "Confidence intervals calculated",
        "Assumption tests performed",
        "Diagnostics generated"
    ],
    "output_completeness": [
        "JSON results file created and valid",
        "All required figures generated",
        "PNG and PDF versions saved",
        "Decision log included"
    ],
    "documentation": [
        "Warnings logged with context",
        "Any deviations from standard protocol documented",
        "Next steps identified"
    ]
}
```

### Inter-Module Communication

When a module depends on another:

1. **Check dependency outputs exist** before starting
2. **If missing:** Log which dependency is missing, skip current module
3. **If malformed:** Attempt to parse what's available, log issues
4. **Pass forward:** Include `upstream_warnings` from dependencies in current output

```json
{
  "dependencies": {
    "module_2_1_1": {
      "status": "loaded",
      "file": "results/module_2_1_1_unit_root_tests.json",
      "key_results_used": ["integration_order", "adf_pvalue"]
    }
  },
  "upstream_warnings": [
    "Module 2.1.1 noted short time series (n=15) may reduce test power"
  ]
}
```

### Timeout and Resource Management

| Operation | Timeout | Action on Timeout |
|-----------|---------|-------------------|
| Data loading | 60 seconds | Fail module |
| Single model fit | 300 seconds | Try simpler model |
| Bootstrap (1000 iter) | 600 seconds | Reduce to 500 iter |
| Monte Carlo (1000 draws) | 900 seconds | Reduce to 500 draws |
| Figure generation | 120 seconds | Skip figure, log error |

### Final Output Validation

Before marking a module complete, verify:

1. **JSON is valid:** Parse with `json.load()` to confirm
2. **Required keys present:** Check against schema
3. **Numeric values reasonable:** No NaN, Inf, or impossible values
4. **Figures render:** Open PNG files to verify non-corrupted
5. **Log file written:** Confirm execution log saved

---

## Quality Standards

Each sub-agent must:

1. **Validate inputs:** Check file existence, data types, missing values
2. **Document assumptions:** Record any data transformations or exclusions
3. **Report diagnostics:** Model fit statistics, residual tests
4. **Handle errors gracefully:** Log issues, continue with available data
5. **Save reproducible outputs:** Include random seeds, version info

---

## Contact Points

- **Project Root:** `/home/nigel/cohort_projections`
- **Statistical Plan:** `data/processed/immigration/analysis/STATISTICAL_ANALYSIS_PLAN.md`
- **Existing Analysis:** `data/processed/immigration/analysis/migration_analysis_results.json`
- **Primary Data:** `data/processed/immigration/analysis/`
