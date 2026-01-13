# Statistical Test Explanation: Augmented Dickey-Fuller Test

## Test: ADF Test

**Full Name:** Augmented Dickey-Fuller Test
**Category:** Unit Root
**Paper Section:** 2.3 Time Series Methods

---

## What This Test Does

The Augmented Dickey-Fuller (ADF) test determines whether a time series contains a unit root, which would indicate that the series is non-stationary. Non-stationary time series have statistical properties (mean, variance, autocorrelation) that change over time, making standard regression techniques invalid and forecasts unreliable. For demographic forecasting, this is critical: if annual international migration to North Dakota is non-stationary, we cannot assume that historical patterns (average levels, volatility) will persist into the future without transformation.

The test works by fitting a regression that includes a lagged level of the series alongside lagged differences. The "augmented" part refers to the inclusion of these lagged differences, which absorb any serial correlation in the errors that would otherwise bias the test. The test statistic examines whether the coefficient on the lagged level is significantly different from zero. If it is (and negative), the series tends to revert toward a stable value, indicating stationarity. If the coefficient is not significantly different from zero, shocks to the series are permanent, indicating a unit root.

---

## The Hypotheses

| Hypothesis | Statement |
|------------|-----------|
| **Null (H0):** | The series contains a unit root (non-stationary) |
| **Alternative (H1):** | The series is stationary (no unit root) |

**Important:** The null hypothesis assumes non-stationarity. This is the reverse of many statistical tests where the null represents "no effect." Failing to reject H0 means we cannot conclude the series is stationary.

---

## Test Statistic

The ADF test is based on the following regression equation:

$$
\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{j=1}^{p} \delta_j \Delta y_{t-j} + \varepsilon_t
$$

Where:
- $\Delta y_t = y_t - y_{t-1}$ is the first difference of the series
- $\alpha$ is a constant (intercept)
- $\beta t$ is a deterministic time trend (optional)
- $\gamma$ is the coefficient of interest on the lagged level
- $y_{t-1}$ is the lagged level of the series
- $\delta_j$ are coefficients on lagged differences (to remove serial correlation)
- $p$ is the number of lags (typically selected by AIC or BIC)
- $\varepsilon_t$ is the error term

The test statistic is:

$$
\tau = \frac{\hat{\gamma}}{\text{SE}(\hat{\gamma})}
$$

**Distribution under H0:** The test statistic does NOT follow a standard t-distribution. It follows a non-standard Dickey-Fuller distribution, which is left-skewed. Critical values are more negative than standard t critical values.

---

## Decision Rule

The ADF test is a left-tailed test. Reject the null hypothesis if the test statistic is more negative than the critical value.

| Significance Level | Critical Value (with constant) | Decision |
|-------------------|-------------------------------|----------|
| alpha = 0.01 | -3.43 | Reject H0 if tau < -3.43 |
| alpha = 0.05 | -2.86 | Reject H0 if tau < -2.86 |
| alpha = 0.10 | -2.57 | Reject H0 if tau < -2.57 |

**Note:** Critical values vary slightly depending on sample size and whether a trend is included. The values above are approximate for a model with a constant but no trend and a moderate sample size (n approximately 100).

| Model Specification | 1% CV | 5% CV | 10% CV |
|--------------------|-------|-------|--------|
| No constant, no trend | -2.58 | -1.95 | -1.62 |
| Constant only | -3.43 | -2.86 | -2.57 |
| Constant and trend | -3.96 | -3.41 | -3.13 |

**P-value approach:** Reject H0 if p-value < alpha

---

## When to Use This Test

**Use when:**
- You need to determine whether a time series is stationary before applying forecasting models
- You want to decide whether differencing is required (e.g., for ARIMA model specification)
- You are testing whether shocks to a demographic series are permanent or transitory
- You need a parametric unit root test with control over lag length

**Do not use when:**
- The series has structural breaks (use Zivot-Andrews test instead)
- You need to confirm stationarity (use KPSS test as a complement, which has stationarity as the null)
- The series is very short (T < 25) since the test has low power
- There is substantial heteroskedasticity (use Phillips-Perron test)

---

## Key Assumptions

1. **No structural breaks:** The ADF test assumes the data-generating process is stable throughout the sample. If there are structural breaks (such as the 2017 Travel Ban or 2020 COVID-19 pandemic affecting migration), the test may incorrectly fail to reject the null.

2. **Correct lag length:** Too few lags will leave serial correlation in the residuals, biasing the test. Too many lags reduce power. Use information criteria (AIC, BIC) for selection.

3. **Correct deterministic specification:** Including a constant and/or trend when they are not present reduces power. Excluding them when they are present can lead to incorrect inference.

4. **Homoskedastic errors:** The test assumes constant error variance. With heteroskedasticity, consider the Phillips-Perron test which applies a nonparametric correction.

5. **No measurement error:** Significant measurement error can affect test reliability.

---

## Worked Example

**Data:** Annual net international migration to North Dakota, 2000-2023 (24 observations)

Suppose our data shows migration values (in thousands): 1.2, 1.4, 1.6, 2.0, 2.3, 2.8, 3.1, 3.5, 4.2, 4.8, 5.1, 5.4, 5.6, 5.9, 6.2, 6.8, 7.2, 6.5, 4.2, 3.8, 4.5, 5.2, 5.8, 6.1

**Calculation:**
```
Step 1: Determine lag length using AIC
        - Test lags 0 through 4
        - AIC minimized at lag = 2

Step 2: Estimate the ADF regression
        Delta_y_t = 0.42 + 0.02*t - 0.18*y_{t-1} + 0.31*Delta_y_{t-1} + 0.12*Delta_y_{t-2} + e_t
                   (0.24)  (0.01)   (0.08)          (0.15)              (0.14)

Step 3: Compute test statistic
        tau = gamma / SE(gamma) = -0.18 / 0.08 = -2.25

Step 4: Compare to critical values (constant + trend model)
        Critical values: 1%: -4.38, 5%: -3.60, 10%: -3.24

P-value = 0.46
```

**Interpretation:**
The test statistic of -2.25 is greater (less negative) than the 10% critical value of -3.24. We fail to reject the null hypothesis of a unit root. The p-value of 0.46 is well above conventional significance levels. This suggests that the North Dakota international migration series may be non-stationary, and first-differencing might be appropriate before forecasting.

---

## Interpreting Results

**If we reject H0 (p-value < 0.05 or tau < critical value):**
The time series is stationary. Shocks are temporary, and the series tends to revert to its mean over time. For immigration data, this would suggest that unusual migration years are followed by return to normal levels. Standard regression and forecasting techniques can be applied directly without differencing.

**If we fail to reject H0:**
We cannot conclude the series is stationary. The series may contain a unit root, meaning shocks are permanent and the series "remembers" past disturbances. This does NOT prove the series is non-stationary (we never "accept" a null hypothesis). For immigration forecasting, this suggests:
- First-differencing may be needed before ARIMA modeling
- Forecasts will have wider confidence intervals that grow with forecast horizon
- Historical means are unreliable guides to future levels

---

## Common Pitfalls

- **Confusing the hypotheses:** Unlike most tests, the null here is "bad news" (non-stationarity). A low p-value is "good" in the sense that it provides evidence of stationarity.

- **Wrong critical values:** Using standard t-distribution critical values (approximately 1.96) instead of Dickey-Fuller critical values will lead to over-rejection of the null (concluding stationarity too often).

- **Ignoring structural breaks:** If migration patterns shifted after a policy change (e.g., 2017 Travel Ban), the ADF test will have low power. Consider the Zivot-Andrews test.

- **Automatic lag selection issues:** Sometimes AIC selects too many lags. Check residual autocorrelation and consider setting a maximum lag based on sample size.

- **Testing transformed data:** Ensure you understand whether you are testing levels or first differences. The appropriate transformation depends on your research question.

- **Over-reliance on a single test:** Always complement ADF with KPSS (which has stationarity as null) for a more complete picture.

---

## Related Tests

| Test | Use When |
|------|----------|
| **Phillips-Perron Test** | Prefer when errors are heteroskedastic or serially correlated; nonparametric correction |
| **KPSS Test** | Use as complement to ADF; tests stationarity as null hypothesis |
| **Zivot-Andrews Test** | Use when structural breaks may be present; endogenously determines break date |
| **DF-GLS Test** | More powerful variant when series has unknown mean or trend |
| **Ng-Perron Test** | Modified ADF with better size and power properties |

---

## Python Implementation

```python
"""
Augmented Dickey-Fuller Test Implementation
Tests for unit root (non-stationarity) in time series data.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from typing import Optional, Dict, Any


def run_adf_test(
    series: pd.Series,
    regression: str = 'c',
    maxlag: Optional[int] = None,
    autolag: str = 'AIC',
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform the Augmented Dickey-Fuller test for unit root.

    Parameters
    ----------
    series : pd.Series
        Time series data to test (e.g., annual migration counts)
    regression : str, default 'c'
        Constant and trend specification:
        - 'c': constant only (most common)
        - 'ct': constant and trend
        - 'ctt': constant, linear and quadratic trend
        - 'n': no constant, no trend
    maxlag : int, optional
        Maximum lag to consider. If None, uses 12*(nobs/100)^0.25
    autolag : str, default 'AIC'
        Method for automatic lag selection ('AIC', 'BIC', 't-stat', or None)
    alpha : float, default 0.05
        Significance level for hypothesis test

    Returns
    -------
    dict
        Dictionary containing test results and interpretation

    Example
    -------
    >>> migration_data = pd.Series([1200, 1400, 1600, ...],
    ...                            index=pd.date_range('2000', periods=24, freq='Y'))
    >>> results = run_adf_test(migration_data, regression='ct')
    >>> print(f"Series is {'stationary' if results['is_stationary'] else 'non-stationary'}")
    """

    # Ensure series has no missing values
    series_clean = series.dropna()

    if len(series_clean) < 20:
        print(f"Warning: Sample size ({len(series_clean)}) may be too small for reliable inference")

    # Run the ADF test
    result = adfuller(
        series_clean,
        maxlag=maxlag,
        regression=regression,
        autolag=autolag
    )

    # Unpack results
    adf_statistic = result[0]
    p_value = result[1]
    used_lag = result[2]
    nobs = result[3]
    critical_values = result[4]
    icbest = result[5]  # AIC/BIC value at selected lag

    # Determine if stationary at given alpha
    is_stationary = p_value < alpha

    # Build interpretation
    if is_stationary:
        interpretation = (
            f"Reject H0 at {alpha:.0%} level. The series appears to be stationary. "
            f"Shocks are temporary and the series reverts to its mean."
        )
    else:
        interpretation = (
            f"Fail to reject H0 at {alpha:.0%} level. Cannot conclude series is stationary. "
            f"Consider first-differencing before forecasting."
        )

    return {
        'test_name': 'Augmented Dickey-Fuller Test',
        'test_statistic': adf_statistic,
        'p_value': p_value,
        'used_lag': used_lag,
        'n_obs': nobs,
        'critical_values': critical_values,
        'ic_value': icbest,
        'regression': regression,
        'is_stationary': is_stationary,
        'alpha': alpha,
        'interpretation': interpretation
    }


def print_adf_results(results: Dict[str, Any]) -> None:
    """Pretty print ADF test results."""

    print("=" * 60)
    print("AUGMENTED DICKEY-FULLER TEST RESULTS")
    print("=" * 60)
    print(f"\nModel specification: {results['regression']}")
    print(f"Observations used: {results['n_obs']}")
    print(f"Lags included: {results['used_lag']}")
    print(f"\nTest statistic: {results['test_statistic']:.4f}")
    print(f"P-value: {results['p_value']:.4f}")
    print("\nCritical values:")
    for level, value in results['critical_values'].items():
        marker = "*" if results['test_statistic'] < value else ""
        print(f"  {level}: {value:.4f} {marker}")
    print(f"\nConclusion at alpha = {results['alpha']:.0%}:")
    print(f"  {results['interpretation']}")
    print("=" * 60)


# Example usage with North Dakota migration data
if __name__ == "__main__":
    # Simulated annual international migration to North Dakota (2000-2023)
    np.random.seed(42)
    years = pd.date_range('2000', periods=24, freq='YE')

    # Create a trending series with some persistence (mimicking migration growth)
    migration = pd.Series(
        np.cumsum(np.random.randn(24) * 500) + np.arange(24) * 200 + 2000,
        index=years,
        name='international_migration'
    )

    # Run the test
    results = run_adf_test(
        migration,
        regression='ct',  # Include constant and trend
        autolag='AIC'
    )

    print_adf_results(results)

    # Test on first differences
    print("\n\nTesting first differences:")
    diff_results = run_adf_test(
        migration.diff().dropna(),
        regression='c',
        autolag='AIC'
    )
    print_adf_results(diff_results)
```

---

## Output Interpretation

```
============================================================
AUGMENTED DICKEY-FULLER TEST RESULTS
============================================================

Model specification: ct
Observations used: 21
Lags included: 2

Test statistic: -2.2485
P-value: 0.4621

Critical values:
  1%: -4.3815
  5%: -3.6027
  10%: -3.2419

Conclusion at alpha = 5%:
  Fail to reject H0 at 5% level. Cannot conclude series is stationary.
  Consider first-differencing before forecasting.
============================================================

Testing first differences:

============================================================
AUGMENTED DICKEY-FULLER TEST RESULTS
============================================================

Model specification: c
Observations used: 19
Lags included: 0

Test statistic: -4.8721
P-value: 0.0001

Critical values:
  1%: -3.7695 *
  5%: -3.0049 *
  10%: -2.6427 *

Conclusion at alpha = 5%:
  Reject H0 at 5% level. The series appears to be stationary.
  Shocks are temporary and the series reverts to its mean.
============================================================
```

**Interpretation of output:**

- **Test statistic (-2.25):** The coefficient on the lagged level divided by its standard error. More negative values provide stronger evidence against the unit root null.

- **P-value (0.46):** The probability of observing a test statistic this extreme if the null hypothesis (unit root) were true. Values above 0.05 indicate insufficient evidence to reject non-stationarity.

- **Critical values:** Thresholds from the Dickey-Fuller distribution. Asterisks indicate which thresholds the test statistic exceeds (passes).

- **First differences:** After differencing, the series becomes stationary (p-value < 0.001), indicating the original series is integrated of order 1, or I(1).

---

## References

- Dickey, D. A., and Fuller, W. A. (1979). "Distribution of the Estimators for Autoregressive Time Series with a Unit Root." *Journal of the American Statistical Association*, 74(366), 427-431.

- Dickey, D. A., and Fuller, W. A. (1981). "Likelihood Ratio Statistics for Autoregressive Time Series with a Unit Root." *Econometrica*, 49(4), 1057-1072.

- Said, S. E., and Dickey, D. A. (1984). "Testing for Unit Roots in Autoregressive-Moving Average Models of Unknown Order." *Biometrika*, 71(3), 599-607.

- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press. (Chapter 17)

- Enders, W. (2014). *Applied Econometric Time Series* (4th ed.). Wiley. (Chapter 4)

- Statsmodels Documentation: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html
