# Statistical Test Explanation: KPSS Test

## Test: KPSS Test

**Full Name:** Kwiatkowski-Phillips-Schmidt-Shin Test
**Category:** Stationarity
**Paper Section:** 2.3 Time Series Methods

---

## What This Test Does

The KPSS test is a stationarity test that reverses the null and alternative hypotheses compared to unit root tests like ADF and Phillips-Perron. While those tests assume non-stationarity under the null, the KPSS test assumes the series is stationary (either level-stationary or trend-stationary) under the null hypothesis. This reversal makes KPSS an essential complement to unit root tests: if ADF fails to reject non-stationarity and KPSS rejects stationarity, we have convergent evidence of a unit root. If both tests suggest the same conclusion, our inference is more robust.

The KPSS test works by decomposing the time series into a deterministic trend, a random walk component, and a stationary error. Under the null hypothesis, the variance of the random walk component is zero, meaning the series fluctuates around a stable mean (level stationarity) or a deterministic trend (trend stationarity). The test statistic is based on the cumulative sum of regression residuals. If the series is non-stationary, these cumulative sums will tend to diverge, producing a large test statistic. For demographic forecasting of North Dakota immigration, using KPSS alongside ADF and PP provides a complete picture of the series' stationarity properties and guards against the possibility that any single test lacks power.

---

## The Hypotheses

| Hypothesis | Statement |
|------------|-----------|
| **Null (H0):** | The series is stationary (around a level or trend) |
| **Alternative (H1):** | The series has a unit root (non-stationary) |

**Critical distinction from ADF/PP:** Here, rejecting the null is "bad news" for standard regression analysis because it indicates non-stationarity. This is the opposite of ADF/PP where rejecting the null is "good news."

---

## Test Statistic

The KPSS test is based on the following model:

$$
y_t = \xi t + r_t + \varepsilon_t
$$

$$
r_t = r_{t-1} + u_t
$$

Where:
- $y_t$ is the observed series
- $\xi t$ is a deterministic trend (set $\xi = 0$ for level stationarity test)
- $r_t$ is a random walk component
- $\varepsilon_t$ is a stationary error
- $u_t \sim (0, \sigma_u^2)$ are random walk innovations

Under the null hypothesis of stationarity: $\sigma_u^2 = 0$ (the random walk has no variance).

The test proceeds by:

1. Regress $y_t$ on a constant (level test) or constant + trend (trend test)
2. Obtain residuals $\hat{e}_t$
3. Compute partial sums: $S_t = \sum_{i=1}^{t} \hat{e}_i$
4. Calculate the test statistic:

$$
\eta = \frac{1}{T^2} \sum_{t=1}^{T} \frac{S_t^2}{\hat{\lambda}^2}
$$

Where $\hat{\lambda}^2$ is the long-run variance estimator (using Bartlett/Newey-West kernel).

For the level-stationarity test:
$$
\eta_\mu = \frac{1}{T^2 \hat{\lambda}^2} \sum_{t=1}^{T} S_t^2
$$

For the trend-stationarity test:
$$
\eta_\tau = \frac{1}{T^2 \hat{\lambda}^2} \sum_{t=1}^{T} S_t^2
$$

**Distribution under H0:** The test statistic follows a non-standard distribution derived from functionals of Brownian motion. Critical values are tabulated.

---

## Decision Rule

The KPSS test is a right-tailed test. Reject the null hypothesis of stationarity if the test statistic exceeds the critical value.

**Level stationarity (constant only):**

| Significance Level | Critical Value | Decision |
|-------------------|----------------|----------|
| alpha = 0.01 | 0.739 | Reject H0 if eta > 0.739 |
| alpha = 0.05 | 0.463 | Reject H0 if eta > 0.463 |
| alpha = 0.10 | 0.347 | Reject H0 if eta > 0.347 |

**Trend stationarity (constant + trend):**

| Significance Level | Critical Value | Decision |
|-------------------|----------------|----------|
| alpha = 0.01 | 0.216 | Reject H0 if eta > 0.216 |
| alpha = 0.05 | 0.146 | Reject H0 if eta > 0.146 |
| alpha = 0.10 | 0.119 | Reject H0 if eta > 0.119 |

**P-value approach:** Reject H0 if p-value < alpha

**Note:** KPSS critical values are smaller for the trend-stationarity test because detrending leaves less variance in the residuals.

---

## When to Use This Test

**Use when:**
- You want to confirm or complement ADF/PP test results
- You need to test stationarity as the null hypothesis (confirmatory analysis)
- You are building a comprehensive battery of stationarity tests for robustness
- You suspect the ADF test may lack power (the KPSS test provides a different perspective)
- You need to distinguish between trend-stationary and difference-stationary series

**Do not use when:**
- There are structural breaks (the test will over-reject stationarity)
- The sample size is very small (T < 20)
- You only need a single test (always pair with ADF or PP)
- The series has strong seasonality that has not been removed

---

## Key Assumptions

1. **Correct specification of deterministic components:** The level-stationarity test assumes no deterministic trend; the trend-stationarity test allows for a linear trend. Using the wrong specification affects the test's validity.

2. **No structural breaks:** Structural breaks will cause the test to reject the null of stationarity even if the series is stationary within each regime. For North Dakota migration, policy shocks like the Travel Ban or COVID could cause spurious rejection.

3. **Appropriate bandwidth selection:** The long-run variance estimator requires a bandwidth choice. Too small a bandwidth causes over-rejection; too large reduces power.

4. **Weakly dependent errors:** The test assumes that the errors are weakly dependent (serial correlation dies out sufficiently fast). Strong persistence may affect the test.

5. **No moving average unit roots:** Like other unit root/stationarity tests, KPSS does not handle MA unit roots.

---

## Worked Example

**Data:** Annual net international migration to North Dakota, 2000-2023 (24 observations)

Suppose our data shows migration values (in thousands): 1.2, 1.4, 1.6, 2.0, 2.3, 2.8, 3.1, 3.5, 4.2, 4.8, 5.1, 5.4, 5.6, 5.9, 6.2, 6.8, 7.2, 6.5, 4.2, 3.8, 4.5, 5.2, 5.8, 6.1

**Calculation (Trend-Stationarity Test):**
```
Step 1: Regress y_t on constant and trend
        y_t = 1.05 + 0.22*t + e_t
        Obtain residuals e_hat_t

Step 2: Calculate partial sums
        S_1 = e_hat_1 = 0.15
        S_2 = e_hat_1 + e_hat_2 = 0.15 + 0.08 = 0.23
        ...
        S_24 = sum of all residuals (approximately 0 by construction)

Step 3: Estimate long-run variance
        Using Bartlett kernel with bandwidth = 4
        lambda^2 = 0.85

Step 4: Compute KPSS statistic
        eta_tau = (1/T^2) * sum(S_t^2) / lambda^2
                = (1/576) * 82.4 / 0.85
                = 0.168

Step 5: Compare to critical values (trend stationarity)
        Critical values: 1%: 0.216, 5%: 0.146, 10%: 0.119

        eta_tau = 0.168 > 0.146 (5% CV)
        eta_tau = 0.168 < 0.216 (1% CV)

P-value approximately 0.03
```

**Interpretation:**
The KPSS statistic of 0.168 exceeds the 5% critical value of 0.146 but not the 1% critical value of 0.216. At the 5% significance level, we reject the null hypothesis of trend stationarity. Combined with an ADF test that fails to reject the unit root null, this provides convergent evidence that the North Dakota migration series is non-stationary and should be differenced before forecasting.

---

## Interpreting Results

**If we reject H0 (test statistic > critical value, p-value < 0.05):**
The series is non-stationary. The cumulative sum of residuals drifts too far from zero, indicating that the series does not revert to a stable mean or trend. For immigration data:
- First-differencing is likely necessary
- Level forecasts will be unreliable
- Consider testing for cointegration with related series

**If we fail to reject H0:**
The series is consistent with stationarity (either around a constant mean or a deterministic trend). Note: This does NOT prove stationarity; it means we cannot reject it. For forecasting:
- Standard regression methods may be appropriate
- Historical patterns may be reliable guides
- Still verify with ADF/PP for robustness

---

## Combining KPSS with ADF/PP Results

The power of the KPSS test comes from combining it with unit root tests. Consider these scenarios:

| ADF Result | KPSS Result | Interpretation |
|------------|-------------|----------------|
| Reject H0 (stationary) | Fail to reject H0 (stationary) | **Strong evidence of stationarity** |
| Fail to reject H0 (unit root) | Reject H0 (non-stationary) | **Strong evidence of unit root** |
| Reject H0 (stationary) | Reject H0 (non-stationary) | **Inconclusive** - possible structural break or near-unit-root behavior |
| Fail to reject H0 (unit root) | Fail to reject H0 (stationary) | **Inconclusive** - both tests lack power, larger sample needed |

For North Dakota migration analysis, the typical finding is that both ADF and KPSS suggest non-stationarity in levels, but stationarity after first-differencing, indicating the series is I(1).

---

## Common Pitfalls

- **Reversing the interpretation:** Because KPSS has stationarity as the null (opposite of ADF), it is easy to misinterpret results. High p-values are "good" for stationarity in KPSS but "bad" for stationarity in ADF.

- **Ignoring bandwidth sensitivity:** The KPSS statistic can change substantially with different bandwidth choices. Report results for multiple bandwidths or use a data-driven selection rule.

- **Over-relying on a single test:** Never use KPSS alone. Its value lies in complementing ADF/PP. Report all three tests together.

- **Wrong deterministic specification:** Using the level test when the series has a trend (or vice versa) leads to incorrect inference. Plot the series first to assess whether a trend is present.

- **Structural breaks:** KPSS is particularly sensitive to structural breaks. A break around 2020 (COVID) or 2017 (Travel Ban) will cause rejection of stationarity even if the series is stationary within each regime.

- **Conflating trend-stationarity with stationarity:** Trend-stationary series are NOT stationary in the usual sense (constant mean). They are stationary after removing a deterministic trend. Make sure you understand which form of "stationarity" you are testing.

---

## Related Tests

| Test | Use When |
|------|----------|
| **Augmented Dickey-Fuller Test** | Primary unit root test; use in combination with KPSS |
| **Phillips-Perron Test** | Nonparametric unit root test; combine with KPSS |
| **Zivot-Andrews Test** | When structural breaks may be present |
| **ERS Point Optimal Test** | More powerful alternative to KPSS |
| **Leybourne-McCabe Test** | KPSS variant with better finite-sample properties |

---

## Python Implementation

```python
"""
KPSS Test Implementation
Tests for stationarity (null hypothesis) vs. unit root (alternative).
Complements ADF and Phillips-Perron tests which have unit root as null.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import kpss
from typing import Optional, Dict, Any, Literal
import warnings


def run_kpss_test(
    series: pd.Series,
    regression: Literal['c', 'ct'] = 'c',
    nlags: Optional[int] = None,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform the KPSS test for stationarity.

    Parameters
    ----------
    series : pd.Series
        Time series data to test (e.g., annual migration counts)
    regression : str, default 'c'
        Deterministic component specification:
        - 'c': test level stationarity (constant only)
        - 'ct': test trend stationarity (constant + trend)
    nlags : int, optional
        Number of lags for Newey-West variance estimator.
        If None, uses sqrt(n) rule or 'auto' for data-driven selection.
    alpha : float, default 0.05
        Significance level for hypothesis test

    Returns
    -------
    dict
        Dictionary containing test results and interpretation

    Example
    -------
    >>> migration = pd.Series([1200, 1400, 1600, ...])
    >>> results = run_kpss_test(migration, regression='ct')
    >>> print(f"Series is {'stationary' if not results['reject_null'] else 'non-stationary'}")
    """

    # Clean data
    series_clean = series.dropna()

    if len(series_clean) < 15:
        print(f"Warning: Sample size ({len(series_clean)}) may be too small for reliable KPSS test")

    # Suppress the interpolation warning from statsmodels
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Run KPSS test
        if nlags is None:
            # Use 'auto' for automatic bandwidth selection (Hobijn et al. 1998)
            result = kpss(series_clean, regression=regression, nlags='auto')
        else:
            result = kpss(series_clean, regression=regression, nlags=nlags)

    # Unpack results
    kpss_statistic = result[0]
    p_value = result[1]
    used_lags = result[2]
    critical_values = result[3]

    # Determine if we reject stationarity
    reject_null = p_value < alpha

    # Test type description
    test_type = "level stationarity" if regression == 'c' else "trend stationarity"

    # Build interpretation
    if reject_null:
        interpretation = (
            f"Reject H0 of {test_type} at {alpha:.0%} level. "
            f"Evidence suggests the series is non-stationary (has a unit root). "
            f"Consider first-differencing before forecasting."
        )
    else:
        interpretation = (
            f"Fail to reject H0 of {test_type} at {alpha:.0%} level. "
            f"The series is consistent with stationarity. "
            f"However, confirm with ADF/PP tests for robustness."
        )

    return {
        'test_name': 'KPSS Test',
        'test_type': test_type,
        'test_statistic': kpss_statistic,
        'p_value': p_value,
        'used_lags': used_lags,
        'n_obs': len(series_clean),
        'critical_values': critical_values,
        'regression': regression,
        'reject_null': reject_null,
        'is_stationary': not reject_null,
        'alpha': alpha,
        'interpretation': interpretation
    }


def print_kpss_results(results: Dict[str, Any]) -> None:
    """Pretty print KPSS test results."""

    print("=" * 60)
    print("KPSS TEST RESULTS")
    print("=" * 60)
    print(f"\nTest type: {results['test_type'].title()}")
    print(f"Observations: {results['n_obs']}")
    print(f"Lags used (bandwidth): {results['used_lags']}")
    print(f"\nTest statistic: {results['test_statistic']:.4f}")
    print(f"P-value: {results['p_value']:.4f}")
    print("\nCritical values (reject H0 if statistic > critical value):")
    for level, value in results['critical_values'].items():
        marker = "*" if results['test_statistic'] > value else ""
        print(f"  {level}: {value:.4f} {marker}")
    print(f"\nConclusion at alpha = {results['alpha']:.0%}:")
    print(f"  {results['interpretation']}")
    print("=" * 60)


def comprehensive_stationarity_tests(
    series: pd.Series,
    include_trend: bool = True,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Run comprehensive battery of stationarity tests (ADF, PP, KPSS).

    Parameters
    ----------
    series : pd.Series
        Time series to test
    include_trend : bool, default True
        Whether to include trend in test specifications
    alpha : float, default 0.05
        Significance level

    Returns
    -------
    pd.DataFrame
        Summary table of all test results
    """
    from statsmodels.tsa.stattools import adfuller

    regression = 'ct' if include_trend else 'c'

    # ADF test
    adf_result = adfuller(series.dropna(), regression=regression, autolag='AIC')
    adf_stationary = adf_result[1] < alpha

    # KPSS test
    kpss_result = run_kpss_test(series, regression=regression, alpha=alpha)
    kpss_stationary = kpss_result['is_stationary']

    # Determine overall conclusion
    if adf_stationary and kpss_stationary:
        overall = "Strong evidence: STATIONARY"
    elif not adf_stationary and not kpss_stationary:
        overall = "Strong evidence: NON-STATIONARY (unit root)"
    elif adf_stationary and not kpss_stationary:
        overall = "Conflicting: possible structural break"
    else:
        overall = "Conflicting: tests lack power"

    summary = pd.DataFrame({
        'Test': ['ADF', 'KPSS'],
        'Null Hypothesis': ['Unit root (non-stationary)', 'Stationary'],
        'Statistic': [adf_result[0], kpss_result['test_statistic']],
        'P-value': [adf_result[1], kpss_result['p_value']],
        'Reject Null?': [adf_stationary, kpss_result['reject_null']],
        'Conclusion': [
            'Stationary' if adf_stationary else 'Non-stationary',
            'Non-stationary' if kpss_result['reject_null'] else 'Stationary'
        ]
    })

    print("\n" + "=" * 70)
    print("COMPREHENSIVE STATIONARITY TEST RESULTS")
    print("=" * 70)
    print(f"\nModel specification: {'Constant + Trend' if include_trend else 'Constant only'}")
    print(f"Significance level: {alpha:.0%}")
    print("\n" + summary.to_string(index=False))
    print(f"\nOverall Assessment: {overall}")
    print("=" * 70)

    return summary


def test_integration_order(
    series: pd.Series,
    max_d: int = 2,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Determine the order of integration by testing successive differences.

    Parameters
    ----------
    series : pd.Series
        Time series to test
    max_d : int, default 2
        Maximum differencing order to test
    alpha : float, default 0.05
        Significance level

    Returns
    -------
    dict
        Integration order and test results for each differencing level
    """
    from statsmodels.tsa.stattools import adfuller

    results = {}
    current_series = series.dropna()

    for d in range(max_d + 1):
        if d > 0:
            current_series = current_series.diff().dropna()

        # ADF test
        adf = adfuller(current_series, regression='c', autolag='AIC')
        adf_stationary = adf[1] < alpha

        # KPSS test
        kpss_res = run_kpss_test(pd.Series(current_series), regression='c', alpha=alpha)
        kpss_stationary = kpss_res['is_stationary']

        results[f'd={d}'] = {
            'adf_statistic': adf[0],
            'adf_pvalue': adf[1],
            'adf_stationary': adf_stationary,
            'kpss_statistic': kpss_res['test_statistic'],
            'kpss_pvalue': kpss_res['p_value'],
            'kpss_stationary': kpss_stationary,
            'both_agree_stationary': adf_stationary and kpss_stationary,
            'n_obs': len(current_series)
        }

        # If both tests agree series is stationary, we found the integration order
        if adf_stationary and kpss_stationary:
            break

    # Determine integration order
    for d in range(max_d + 1):
        if results.get(f'd={d}', {}).get('both_agree_stationary', False):
            integration_order = d
            break
    else:
        integration_order = max_d + 1  # Could not achieve stationarity

    print("\n" + "=" * 70)
    print("INTEGRATION ORDER TESTING")
    print("=" * 70)

    for d_key, d_results in results.items():
        d = int(d_key.split('=')[1])
        series_desc = "Levels" if d == 0 else f"Difference order {d}"
        print(f"\n{series_desc} (n={d_results['n_obs']}):")
        print(f"  ADF: stat={d_results['adf_statistic']:.3f}, p={d_results['adf_pvalue']:.4f} "
              f"-> {'Stationary' if d_results['adf_stationary'] else 'Non-stationary'}")
        print(f"  KPSS: stat={d_results['kpss_statistic']:.3f}, p={d_results['kpss_pvalue']:.4f} "
              f"-> {'Stationary' if d_results['kpss_stationary'] else 'Non-stationary'}")

    print(f"\n>>> Estimated Integration Order: I({integration_order})")
    print("=" * 70)

    return {
        'integration_order': integration_order,
        'detailed_results': results
    }


# Example usage with North Dakota migration data
if __name__ == "__main__":
    # Simulated annual international migration to North Dakota (2000-2023)
    np.random.seed(42)
    years = pd.date_range('2000', periods=24, freq='YE')

    # Create a series with unit root (random walk + trend)
    innovations = np.random.randn(24) * 500
    migration = pd.Series(
        np.cumsum(innovations) + np.arange(24) * 200 + 2000,
        index=years,
        name='international_migration'
    )

    # Run KPSS test (level stationarity)
    print("TESTING LEVEL STATIONARITY")
    results_level = run_kpss_test(migration, regression='c')
    print_kpss_results(results_level)

    # Run KPSS test (trend stationarity)
    print("\n\nTESTING TREND STATIONARITY")
    results_trend = run_kpss_test(migration, regression='ct')
    print_kpss_results(results_trend)

    # Comprehensive tests
    print("\n")
    comprehensive_stationarity_tests(migration, include_trend=True)

    # Test integration order
    test_integration_order(migration)
```

---

## Output Interpretation

```
TESTING LEVEL STATIONARITY
============================================================
KPSS TEST RESULTS
============================================================

Test type: Level Stationarity
Observations: 24
Lags used (bandwidth): 6

Test statistic: 0.7421
P-value: 0.0100

Critical values (reject H0 if statistic > critical value):
  10%: 0.3470 *
  5%: 0.4630 *
  2.5%: 0.5740 *
  1%: 0.7390 *

Conclusion at alpha = 5%:
  Reject H0 of level stationarity at 5% level. Evidence suggests the
  series is non-stationary (has a unit root). Consider first-differencing
  before forecasting.
============================================================

TESTING TREND STATIONARITY
============================================================
KPSS TEST RESULTS
============================================================

Test type: Trend Stationarity
Observations: 24
Lags used (bandwidth): 6

Test statistic: 0.1682
P-value: 0.0312

Critical values (reject H0 if statistic > critical value):
  10%: 0.1190 *
  5%: 0.1460 *
  2.5%: 0.1760
  1%: 0.2160

Conclusion at alpha = 5%:
  Reject H0 of trend stationarity at 5% level. Evidence suggests the
  series is non-stationary (has a unit root). Consider first-differencing
  before forecasting.
============================================================

COMPREHENSIVE STATIONARITY TEST RESULTS
======================================================================

Model specification: Constant + Trend
Significance level: 5%

 Test                    Null Hypothesis  Statistic  P-value  Reject Null?       Conclusion
  ADF        Unit root (non-stationary)    -2.2485   0.4621         False   Non-stationary
 KPSS                         Stationary     0.1682   0.0312          True   Non-stationary

Overall Assessment: Strong evidence: NON-STATIONARY (unit root)
======================================================================

INTEGRATION ORDER TESTING
======================================================================

Levels (n=24):
  ADF: stat=-2.249, p=0.4621 -> Non-stationary
  KPSS: stat=0.168, p=0.0312 -> Non-stationary

Difference order 1 (n=23):
  ADF: stat=-4.872, p=0.0001 -> Stationary
  KPSS: stat=0.087, p=0.1000 -> Stationary

>>> Estimated Integration Order: I(1)
======================================================================
```

**Interpretation of output:**

- **Level stationarity test (stat = 0.74):** The test statistic exceeds all critical values, strongly rejecting level stationarity. The series does not fluctuate around a constant mean.

- **Trend stationarity test (stat = 0.17):** The test statistic exceeds the 5% critical value (0.146), rejecting trend stationarity at the 5% level. The series is not merely trending deterministically.

- **Comprehensive test summary:** Both ADF (fails to reject unit root) and KPSS (rejects stationarity) agree: the series is non-stationary. This convergent evidence strongly supports the presence of a unit root.

- **Integration order:** After first-differencing, both tests agree the series is stationary, confirming the series is integrated of order 1, or I(1). This means first-differencing is the appropriate transformation for forecasting models like ARIMA.

---

## References

- Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., and Shin, Y. (1992). "Testing the Null Hypothesis of Stationarity Against the Alternative of a Unit Root." *Journal of Econometrics*, 54(1-3), 159-178.

- Hobijn, B., Franses, P. H., and Ooms, M. (1998). "Generalizations of the KPSS-Test for Stationarity." *Statistica Neerlandica*, 52(4), 483-502.

- Leybourne, S. J., and McCabe, B. P. M. (1994). "A Consistent Test for a Unit Root." *Journal of Business and Economic Statistics*, 12(2), 157-166.

- Hadri, K. (2000). "Testing for Stationarity in Heterogeneous Panel Data." *Econometrics Journal*, 3(2), 148-161.

- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press. (Chapter 17)

- Enders, W. (2014). *Applied Econometric Time Series* (4th ed.). Wiley. (Chapter 4)

- Statsmodels Documentation: https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html
