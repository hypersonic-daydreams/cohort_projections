# Statistical Test Explanation

## Test: Granger Causality Test

**Full Name:** Granger Causality Test
**Category:** VAR Causality / Temporal Predictive Relationships
**Paper Section:** 2.3.4 Vector Autoregression

---

## What This Test Does

The Granger Causality Test determines whether one time series is useful for forecasting another. Named after Nobel laureate Clive Granger, this test operationalizes a specific notion of "causality" based on temporal precedence and predictive power: variable X is said to "Granger-cause" variable Y if past values of X contain information that helps predict Y, beyond what is contained in past values of Y alone.

In the context of forecasting international migration to North Dakota, we use the Granger test to assess whether economic indicators such as oil employment can predict future migration flows. If oil employment Granger-causes migration, then incorporating lagged employment data into our forecasting model should improve predictive accuracy. This does not imply that oil employment literally causes migration in a mechanistic sense, but rather that there exists a statistically detectable temporal predictive relationship that can be exploited for forecasting purposes.

---

## The Hypotheses

| Hypothesis | Statement |
|------------|-----------|
| **Null (H_0):** | Variable X does not Granger-cause variable Y (lagged values of X do not improve the prediction of Y given lagged values of Y) |
| **Alternative (H_1):** | Variable X Granger-causes variable Y (lagged values of X contain useful predictive information for Y) |

---

## Test Statistic

The Granger Causality Test is typically implemented as an F-test (or equivalent chi-squared test) comparing a restricted model against an unrestricted model:

**Unrestricted Model (includes lags of both Y and X):**
$$
Y_t = \alpha + \sum_{i=1}^{p} \beta_i Y_{t-i} + \sum_{j=1}^{p} \gamma_j X_{t-j} + \varepsilon_t
$$

**Restricted Model (includes only lags of Y):**
$$
Y_t = \alpha + \sum_{i=1}^{p} \beta_i Y_{t-i} + \varepsilon_t
$$

**F-Statistic:**
$$
F = \frac{(RSS_R - RSS_U) / p}{RSS_U / (T - 2p - 1)}
$$

where:
- $RSS_R$ = Residual sum of squares from the restricted model
- $RSS_U$ = Residual sum of squares from the unrestricted model
- $p$ = Number of lags included
- $T$ = Number of observations

**Distribution under H_0:** F-distribution with $(p, T - 2p - 1)$ degrees of freedom

Alternatively, the test can be conducted using a chi-squared statistic:
$$
\chi^2 = T \cdot \frac{RSS_R - RSS_U}{RSS_R}
$$

**Distribution under H_0:** Chi-squared with $p$ degrees of freedom

---

## Decision Rule

| Significance Level | Critical Value (F with p=2, df2=30) | Decision |
|-------------------|----------------|----------|
| alpha = 0.01 | 5.39 | Reject H_0 if F > 5.39 |
| alpha = 0.05 | 3.32 | Reject H_0 if F > 3.32 |
| alpha = 0.10 | 2.49 | Reject H_0 if F > 2.49 |

*Note: Critical values depend on the number of lags (p) and sample size. The values above are illustrative for p=2 lags and approximately 30 residual degrees of freedom.*

**P-value approach:** Reject H_0 if p-value < alpha

---

## When to Use This Test

**Use when:**
- You want to determine if one time series helps predict another
- You have two or more stationary time series (or appropriately differenced series)
- You are building a VAR model and want to identify meaningful predictive relationships
- You want empirical justification for including exogenous predictors in a forecasting model
- You seek to establish temporal precedence in predictive relationships

**Don't use when:**
- The series are non-stationary and not cointegrated (spurious regression risk)
- You want to establish true causal mechanisms (Granger causality is about prediction, not causation)
- You have insufficient observations relative to the number of lags tested
- The relationship is contemporaneous rather than lagged
- You have strong theoretical reasons that contradict the empirical findings

---

## Key Assumptions

1. **Stationarity:** Both time series must be covariance stationary (constant mean and variance over time). Non-stationary series must be differenced or the test should be conducted within a cointegration framework. Violation leads to spurious results.

2. **Linear Relationship:** The test assumes a linear relationship between variables. Nonlinear predictive relationships will not be detected by the standard Granger test.

3. **Correct Lag Specification:** The number of lags must be appropriately chosen. Too few lags may miss the true predictive relationship; too many lags reduce power and waste degrees of freedom. Use information criteria (AIC, BIC) for lag selection.

4. **No Omitted Variables:** The predictive relationship should not be spuriously induced by a third, omitted variable that causes both X and Y. In practice, this is addressed by including relevant controls or testing within a multivariate VAR.

5. **No Structural Breaks:** The relationship should be stable over the sample period. Structural breaks (such as policy changes) may invalidate the test or require sub-sample analysis.

---

## Worked Example

**Context:** Testing whether oil employment in North Dakota Granger-causes international migration to the state.

**Data:**
- Annual observations from 2000 to 2023 (T = 24 years)
- Y = Log of net international migration to ND
- X = Log of oil sector employment in ND
- Both series confirmed stationary via ADF test after first differencing
- Lag length p = 2 selected by AIC

**Calculation:**
```
Step 1: Estimate the restricted model (migration on its own lags)
        Restricted RSS = 0.4521

Step 2: Estimate the unrestricted model (add oil employment lags)
        Unrestricted RSS = 0.3187

Step 3: Calculate F-statistic
        F = [(0.4521 - 0.3187) / 2] / [0.3187 / (24 - 4 - 1)]
        F = [0.1334 / 2] / [0.3187 / 19]
        F = 0.0667 / 0.01677
        F = 3.98

Step 4: Compare to critical value
        F_0.05(2, 19) = 3.52
        F-statistic (3.98) > Critical value (3.52)

P-value = 0.036
```

**Interpretation:**
At the 5% significance level, we reject the null hypothesis that oil employment does not Granger-cause international migration (F = 3.98, p = 0.036). This suggests that lagged oil employment values contain statistically significant predictive information for forecasting future migration flows to North Dakota, beyond what is captured by migration's own history.

---

## Interpreting Results

**If we reject H_0:**
Oil employment Granger-causes international migration. This means that past values of oil sector employment provide statistically significant predictive power for future migration flows, beyond the information contained in migration's own past. Practically, this justifies including oil employment as a predictor in migration forecasting models. The direction and magnitude of the effect should be examined through the estimated coefficients in the VAR model.

**If we fail to reject H_0:**
We do not have sufficient evidence that oil employment Granger-causes migration. This does NOT mean oil employment is unrelated to migration; it means that in this particular sample, with this lag structure, the lagged values of oil employment do not significantly improve predictions of migration beyond migration's own history. The relationship might be contemporaneous rather than lagged, or the effect may be too small to detect with the available sample size.

---

## Common Pitfalls

- **Confusing Granger causality with true causality:** Granger causality is purely about temporal predictive precedence. Finding that X Granger-causes Y does not mean X mechanistically causes Y. A third variable Z could be causing both with different timing, creating the appearance of Granger causality.

- **Testing non-stationary series:** Applying Granger tests to non-stationary, non-cointegrated series can yield spurious results. Always verify stationarity first with unit root tests (ADF, KPSS, PP).

- **Incorrect lag selection:** Using too few lags may miss the true relationship; too many lags waste degrees of freedom and reduce power. Use information criteria for objective lag selection.

- **Ignoring bidirectionality:** Granger causality can run both ways. Always test in both directions. Migration might Granger-cause oil employment (through labor supply effects) as well as the reverse.

- **Over-interpreting marginally significant results:** With small samples common in annual macroeconomic data, p-values near 0.05 should be interpreted cautiously. Consider multiple testing corrections if examining many variable pairs.

- **Ignoring structural breaks:** Policy changes (Travel Ban 2017, COVID-19 2020) may alter the predictive relationship. Sub-sample or break-adjusted analysis may be necessary.

---

## Related Tests

| Test | Use When |
|------|----------|
| **Toda-Yamamoto Procedure** | When series may be non-stationary; avoids pre-testing for unit roots |
| **Johansen Cointegration Test** | When you want to test for long-run equilibrium relationships between non-stationary series |
| **Impulse Response Functions** | When you want to trace the dynamic effects of shocks in a VAR system |
| **Forecast Error Variance Decomposition** | When you want to quantify how much of the forecast variance is attributable to each variable |
| **Instantaneous Causality Test** | When you suspect contemporaneous rather than lagged relationships |

---

## Python Implementation

```python
"""
Granger Causality Test for Migration Forecasting
Tests whether oil employment predicts international migration to North Dakota
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.api import VAR


def check_stationarity(series: pd.Series, name: str, alpha: float = 0.05) -> bool:
    """
    Check if series is stationary using ADF test.

    Parameters
    ----------
    series : pd.Series
        Time series to test
    name : str
        Name of the series for reporting
    alpha : float
        Significance level (default 0.05)

    Returns
    -------
    bool
        True if stationary, False otherwise
    """
    result = adfuller(series.dropna(), autolag='AIC')
    adf_stat, p_value = result[0], result[1]
    is_stationary = p_value < alpha

    print(f"ADF Test for {name}:")
    print(f"  ADF Statistic: {adf_stat:.4f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Stationary: {is_stationary}")

    return is_stationary


def run_granger_causality_test(
    data: pd.DataFrame,
    cause_var: str,
    effect_var: str,
    max_lag: int = 4,
    verbose: bool = True
) -> dict:
    """
    Perform Granger Causality Test.

    Tests whether cause_var Granger-causes effect_var.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing both time series
    cause_var : str
        Name of the potential causal variable (X)
    effect_var : str
        Name of the dependent variable (Y)
    max_lag : int
        Maximum number of lags to test (default 4)
    verbose : bool
        Whether to print detailed results

    Returns
    -------
    dict
        Dictionary containing test results for each lag
    """
    # Prepare data: effect variable first, then cause variable
    # (grangercausalitytests expects [Y, X] ordering)
    test_data = data[[effect_var, cause_var]].dropna()

    if verbose:
        print(f"\n{'='*60}")
        print(f"Granger Causality Test")
        print(f"H0: {cause_var} does NOT Granger-cause {effect_var}")
        print(f"H1: {cause_var} Granger-causes {effect_var}")
        print(f"{'='*60}\n")

    # Run the test
    results = grangercausalitytests(test_data, maxlag=max_lag, verbose=verbose)

    # Extract key results
    summary = {}
    for lag in range(1, max_lag + 1):
        f_test = results[lag][0]['ssr_ftest']
        chi2_test = results[lag][0]['ssr_chi2test']

        summary[lag] = {
            'f_statistic': f_test[0],
            'f_pvalue': f_test[1],
            'df_num': f_test[2],
            'df_denom': f_test[3],
            'chi2_statistic': chi2_test[0],
            'chi2_pvalue': chi2_test[1],
            'chi2_df': chi2_test[2]
        }

    return summary


def bidirectional_granger_test(
    data: pd.DataFrame,
    var1: str,
    var2: str,
    max_lag: int = 4
) -> pd.DataFrame:
    """
    Test Granger causality in both directions.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing both time series
    var1, var2 : str
        Names of the two variables to test
    max_lag : int
        Maximum lag to test

    Returns
    -------
    pd.DataFrame
        Summary of bidirectional test results
    """
    results = []

    # Test var1 -> var2
    gc_1to2 = run_granger_causality_test(data, var1, var2, max_lag, verbose=False)
    for lag, stats in gc_1to2.items():
        results.append({
            'direction': f'{var1} -> {var2}',
            'lag': lag,
            'f_stat': stats['f_statistic'],
            'p_value': stats['f_pvalue'],
            'significant_05': stats['f_pvalue'] < 0.05,
            'significant_10': stats['f_pvalue'] < 0.10
        })

    # Test var2 -> var1
    gc_2to1 = run_granger_causality_test(data, var2, var1, max_lag, verbose=False)
    for lag, stats in gc_2to1.items():
        results.append({
            'direction': f'{var2} -> {var1}',
            'lag': lag,
            'f_stat': stats['f_statistic'],
            'p_value': stats['f_pvalue'],
            'significant_05': stats['f_pvalue'] < 0.05,
            'significant_10': stats['f_pvalue'] < 0.10
        })

    return pd.DataFrame(results)


# Example usage with North Dakota migration data
if __name__ == "__main__":
    # Simulated data (replace with actual data in practice)
    np.random.seed(42)
    n_years = 24  # 2000-2023

    # Generate correlated series with temporal structure
    oil_employment = np.cumsum(np.random.randn(n_years) * 0.1) + np.linspace(0, 2, n_years)
    migration = (
        0.3 * np.roll(oil_employment, 1) +  # Lagged effect
        0.2 * np.roll(oil_employment, 2) +  # Second lag effect
        np.cumsum(np.random.randn(n_years) * 0.08)
    )
    migration[0:2] = migration[2]  # Fix edge effects

    # Create DataFrame
    years = range(2000, 2000 + n_years)
    df = pd.DataFrame({
        'year': years,
        'oil_employment': oil_employment,
        'migration': migration
    })

    # Check stationarity
    print("Checking stationarity...\n")
    stat_oil = check_stationarity(df['oil_employment'], 'Oil Employment')
    stat_mig = check_stationarity(df['migration'], 'Migration')

    # If non-stationary, difference the series
    if not stat_oil or not stat_mig:
        print("\nDifferencing series for stationarity...")
        df['d_oil_employment'] = df['oil_employment'].diff()
        df['d_migration'] = df['migration'].diff()
        oil_var = 'd_oil_employment'
        mig_var = 'd_migration'
    else:
        oil_var = 'oil_employment'
        mig_var = 'migration'

    # Run Granger Causality Test
    print("\n" + "="*60)
    print("TESTING: Does Oil Employment Granger-cause Migration?")
    print("="*60)

    results = run_granger_causality_test(
        data=df,
        cause_var=oil_var,
        effect_var=mig_var,
        max_lag=3,
        verbose=True
    )

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for lag, stats in results.items():
        sig = "***" if stats['f_pvalue'] < 0.01 else "**" if stats['f_pvalue'] < 0.05 else "*" if stats['f_pvalue'] < 0.10 else ""
        print(f"Lag {lag}: F = {stats['f_statistic']:.3f}, p = {stats['f_pvalue']:.4f} {sig}")

    # Bidirectional test
    print("\n" + "="*60)
    print("BIDIRECTIONAL GRANGER CAUSALITY SUMMARY")
    print("="*60)
    bidir_results = bidirectional_granger_test(df, oil_var, mig_var, max_lag=3)
    print(bidir_results.to_string(index=False))
```

---

## Output Interpretation

```
Granger Causality Test
H0: d_oil_employment does NOT Granger-cause d_migration
H1: d_oil_employment Granger-causes d_migration

Granger Causality
number of lags (no zero) 1
ssr based F test:         F=4.2315, p=0.0531, df_denom=19, df_num=1
ssr based chi2 test:   chi2=4.6789, p=0.0305, df=1

Granger Causality
number of lags (no zero) 2
ssr based F test:         F=3.9842, p=0.0367, df_denom=17, df_num=2
ssr based chi2 test:   chi2=9.0123, p=0.0111, df=2

SUMMARY
Lag 1: F = 4.232, p = 0.0531 *
Lag 2: F = 3.984, p = 0.0367 **
Lag 3: F = 2.891, p = 0.0712 *

INTERPRETATION:
- At lag 2, we reject H0 at the 5% level (p = 0.037)
- Oil employment Granger-causes migration with optimal lag of 2 years
- This suggests oil sector changes precede migration changes by 1-2 years
- The predictive relationship is statistically significant and can be
  exploited for forecasting purposes in a VAR model
```

---

## References

- Granger, C. W. J. (1969). "Investigating Causal Relations by Econometric Models and Cross-spectral Methods." *Econometrica*, 37(3), 424-438. [Original paper introducing Granger causality]

- Toda, H. Y., & Yamamoto, T. (1995). "Statistical inference in vector autoregressions with possibly integrated processes." *Journal of Econometrics*, 66(1-2), 225-250. [Alternative approach for potentially non-stationary series]

- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press. Chapter 11. [Standard textbook treatment]

- Lutkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer. [Comprehensive VAR methodology]

- Stock, J. H., & Watson, M. W. (2001). "Vector Autoregressions." *Journal of Economic Perspectives*, 15(4), 101-115. [Accessible overview]

- Enders, W. (2014). *Applied Econometric Time Series* (4th ed.). Wiley. Chapter 5. [Applied treatment with examples]
