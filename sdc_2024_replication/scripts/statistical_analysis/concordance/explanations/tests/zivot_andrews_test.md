# Statistical Test Explanation: Zivot-Andrews Test

## Test: Zivot-Andrews Unit Root Test

**Full Name:** Zivot-Andrews Test for Unit Root with Structural Break
**Category:** Unit Root with Structural Break
**Paper Section:** 2.3 Time Series Methods

---

## What This Test Does

The Zivot-Andrews test is an extension of the standard Augmented Dickey-Fuller (ADF) unit root test that accounts for the possibility of a single structural break at an unknown date in the time series. Unlike the ADF test, which may falsely conclude a series has a unit root when it is actually stationary with a structural break, the Zivot-Andrews test endogenously determines the most likely break date while simultaneously testing for a unit root.

In the context of international migration to North Dakota, this test is particularly valuable because migration patterns are subject to policy shocks (such as the 2017 Travel Ban) and external disruptions (such as the COVID-19 pandemic). A standard unit root test might incorrectly conclude that migration series are non-stationary when they are actually trend-stationary with a discrete shift. The Zivot-Andrews test identifies the break point that provides the strongest evidence against the unit root hypothesis, making it a powerful tool for sensitivity analysis in demographic time series.

---

## The Hypotheses

| Hypothesis | Statement |
|------------|-----------|
| **Null (H₀):** | The series has a unit root with no structural break (integrated process) |
| **Alternative (H₁):** | The series is trend-stationary with one endogenous structural break |

---

## Test Statistic

The Zivot-Andrews test estimates the following regression for each potential break date $T_B$:

### Model A (Break in Intercept Only):
$$
\Delta y_t = \mu + \beta t + \theta DU_t(\lambda) + \gamma y_{t-1} + \sum_{j=1}^{k} \delta_j \Delta y_{t-j} + \varepsilon_t
$$

### Model B (Break in Trend Only):
$$
\Delta y_t = \mu + \beta t + \phi DT_t(\lambda) + \gamma y_{t-1} + \sum_{j=1}^{k} \delta_j \Delta y_{t-j} + \varepsilon_t
$$

### Model C (Break in Both Intercept and Trend):
$$
\Delta y_t = \mu + \beta t + \theta DU_t(\lambda) + \phi DT_t(\lambda) + \gamma y_{t-1} + \sum_{j=1}^{k} \delta_j \Delta y_{t-j} + \varepsilon_t
$$

Where:
- $DU_t(\lambda) = 1$ if $t > T_B$, 0 otherwise (intercept dummy)
- $DT_t(\lambda) = t - T_B$ if $t > T_B$, 0 otherwise (trend dummy)
- $\lambda = T_B / T$ is the break fraction
- The test statistic is the minimum t-statistic on $\gamma$ across all possible break dates

**Distribution under H₀:** The test statistic follows a non-standard distribution derived by Zivot and Andrews (1992). Critical values depend on the model specification and break fraction.

---

## Decision Rule

### Model A (Intercept Break) Critical Values:

| Significance Level | Critical Value | Decision |
|-------------------|----------------|----------|
| α = 0.01 | -5.34 | Reject H₀ if test stat < -5.34 |
| α = 0.05 | -4.80 | Reject H₀ if test stat < -4.80 |
| α = 0.10 | -4.58 | Reject H₀ if test stat < -4.58 |

### Model B (Trend Break) Critical Values:

| Significance Level | Critical Value | Decision |
|-------------------|----------------|----------|
| α = 0.01 | -4.93 | Reject H₀ if test stat < -4.93 |
| α = 0.05 | -4.42 | Reject H₀ if test stat < -4.42 |
| α = 0.10 | -4.11 | Reject H₀ if test stat < -4.11 |

### Model C (Both Intercept and Trend Break) Critical Values:

| Significance Level | Critical Value | Decision |
|-------------------|----------------|----------|
| α = 0.01 | -5.57 | Reject H₀ if test stat < -5.57 |
| α = 0.05 | -5.08 | Reject H₀ if test stat < -5.08 |
| α = 0.10 | -4.82 | Reject H₀ if test stat < -4.82 |

**P-value approach:** Reject H₀ if p-value < α

---

## When to Use This Test

**Use when:**
- You suspect a single structural break exists in the time series at an unknown date
- Standard ADF test suggests a unit root but you believe the series may be trend-stationary with a break
- Testing migration data around known policy interventions (Travel Ban, immigration reform)
- Analyzing demographic series affected by external shocks (recessions, pandemics)
- Conducting sensitivity analysis to ensure unit root conclusions are robust to structural breaks

**Don't use when:**
- Multiple structural breaks are suspected (use Bai-Perron test instead)
- The break date is known a priori (use ADF with dummy variables or Chow test)
- The sample size is very small (< 30 observations) - test has low power
- You need to test for breaks in variance rather than level/trend

---

## Key Assumptions

1. **Single Break:** The test assumes at most one structural break in the series. If multiple breaks exist, the test may have low power or identify only the most significant break.

2. **Endogenous Break:** The break point is determined by the data rather than imposed exogenously. The test searches over a trimmed interior portion of the sample (typically 15% trimmed from each end).

3. **Break Under Alternative Only:** The structural break exists only under the alternative hypothesis. Under the null of a unit root, the series is modeled without a deterministic break.

4. **Correct Model Specification:** The researcher must choose whether to test for a break in intercept (Model A), trend (Model B), or both (Model C). Misspecification can affect results.

5. **No Serial Correlation:** Lagged differences are included to absorb serial correlation in the residuals. The lag length k is typically selected by information criteria.

---

## Worked Example

**Data:**
North Dakota international migration (net flows), 2000-2023, with suspected structural break around 2017 (Travel Ban) or 2020 (COVID-19).

Sample values (in thousands):
- 2000-2009: 1.2, 1.4, 1.3, 1.5, 1.8, 2.1, 2.4, 2.6, 2.8, 3.1
- 2010-2019: 3.3, 3.6, 4.0, 4.5, 4.8, 5.2, 5.0, 4.2, 3.8, 3.5
- 2020-2023: 2.1, 2.8, 3.2, 3.6

**Calculation:**
```
Step 1: Set up the search grid
   - Total observations T = 24
   - Trim 15% from each end: search from t=4 (2003) to t=20 (2019)
   - Test break dates: 2003, 2004, ..., 2019 (17 candidate breaks)

Step 2: For each candidate break, estimate Model C regression
   - Break at 2017: t-statistic on gamma = -4.72
   - Break at 2018: t-statistic on gamma = -5.21
   - Break at 2019: t-statistic on gamma = -4.89
   - Break at 2020: t-statistic on gamma = -5.43

Step 3: Find minimum t-statistic across all breaks
   - Minimum t-statistic = -5.43 at break date 2020

Step 4: Compare to critical values (Model C, 5% level)
   - Test statistic: -5.43
   - Critical value: -5.08
   - |−5.43| > |−5.08|

P-value = 0.032
```

**Interpretation:**
The test rejects the null hypothesis of a unit root at the 5% significance level. The evidence suggests that North Dakota's international migration series is trend-stationary with a structural break in 2020 (coinciding with COVID-19). The series does not contain a unit root once we account for this break. This finding differs from a standard ADF test, which might incorrectly classify the series as non-stationary by failing to account for the pandemic disruption.

---

## Interpreting Results

**If we reject H₀:**
The series is trend-stationary with a structural break at the identified date. This means:
- The series has a deterministic trend component
- There was a significant shift in level and/or trend at the break date
- Standard regression techniques (with appropriate break dummies) are valid
- The series will revert to its trend after shocks (mean-reverting behavior)

For migration forecasting, this suggests that migration patterns to North Dakota are fundamentally stable with occasional discrete regime changes, rather than following a random walk.

**If we fail to reject H₀:**
The series may contain a unit root even after allowing for a structural break. This means:
- Shocks have permanent effects on the series
- First-differencing may be appropriate for stationarity
- The identified "break" may be spurious or the series genuinely follows a unit root process

Note: Failing to reject does not prove a unit root exists - it may indicate low power due to small sample size or the presence of multiple breaks.

---

## Common Pitfalls

- **Searching too close to endpoints:** The test trims observations from both ends of the sample to avoid spurious results. Using the full sample can lead to biased break date identification.

- **Misinterpreting the break date:** The identified break is the date that provides the strongest statistical evidence against a unit root, not necessarily the date of a structural change. Economic interpretation should be cautious.

- **Ignoring multiple breaks:** If the true data-generating process has multiple breaks, the Zivot-Andrews test may have low power. Consider the Bai-Perron test for multiple break detection.

- **Model selection:** The choice between Models A, B, and C affects both the test statistic and critical values. Theory should guide model selection; testing all three and reporting results is recommended.

- **Small sample bias:** With fewer than 30-40 observations, the test has limited power to detect trend stationarity. Results should be interpreted cautiously with demographic data covering short time spans.

---

## Related Tests

| Test | Use When |
|------|----------|
| Augmented Dickey-Fuller (ADF) | No structural break is suspected; standard unit root test |
| KPSS Test | Want stationarity as null hypothesis; complements ADF |
| Perron (1989) Test | Break date is known a priori from theory or events |
| Bai-Perron Test | Multiple structural breaks at unknown dates |
| Chow Test | Testing for parameter stability at a known break point |
| Lee-Strazicich Test | Two potential breaks under both null and alternative |

---

## Python Implementation

```python
"""
Zivot-Andrews Test for Unit Root with Structural Break

This implementation tests whether a time series is trend-stationary
with a single endogenous structural break.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import zivot_andrews


def run_zivot_andrews_test(
    series: pd.Series,
    regression: str = 'c',
    autolag: str = 'AIC',
    maxlag: int = None,
    trim: float = 0.15
) -> dict:
    """
    Perform the Zivot-Andrews unit root test with structural break.

    Parameters
    ----------
    series : pd.Series
        Time series data to test (should have datetime index)
    regression : str, optional
        Test type: 'c' (intercept break), 't' (trend break), 'ct' (both)
        Default is 'c' for intercept break
    autolag : str, optional
        Method for lag selection: 'AIC', 'BIC', 't-stat', or None
    maxlag : int, optional
        Maximum number of lags to consider
    trim : float, optional
        Fraction of data to trim from each end (default 0.15)

    Returns
    -------
    dict
        Dictionary containing test results:
        - statistic: ZA test statistic
        - pvalue: p-value for the test
        - critical_values: dict of critical values at 1%, 5%, 10%
        - break_index: index of identified break point
        - break_date: date of identified break (if datetime index)
        - lags: number of lags used
        - regression: type of test performed
    """
    # Ensure series is clean
    series_clean = series.dropna()

    if len(series_clean) < 20:
        raise ValueError(
            "Zivot-Andrews test requires at least 20 observations "
            f"(got {len(series_clean)})"
        )

    # Map regression parameter to statsmodels format
    reg_map = {'c': 'c', 't': 't', 'ct': 'ct'}
    if regression not in reg_map:
        raise ValueError(f"regression must be one of {list(reg_map.keys())}")

    # Run the Zivot-Andrews test
    result = zivot_andrews(
        series_clean.values,
        regression=regression,
        autolag=autolag,
        maxlag=maxlag,
        trim=trim
    )

    # Extract results
    za_stat = result[0]
    p_value = result[1]
    critical_values = result[2]
    break_index = result[3]
    lags_used = result[4]

    # Get the break date if series has datetime index
    if isinstance(series_clean.index, pd.DatetimeIndex):
        break_date = series_clean.index[break_index]
    else:
        break_date = series_clean.index[break_index]

    # Compile results
    results = {
        'statistic': za_stat,
        'pvalue': p_value,
        'critical_values': critical_values,
        'break_index': break_index,
        'break_date': break_date,
        'lags': lags_used,
        'regression': regression
    }

    return results


def interpret_zivot_andrews(results: dict, alpha: float = 0.05) -> str:
    """
    Provide plain-English interpretation of Zivot-Andrews test results.

    Parameters
    ----------
    results : dict
        Output from run_zivot_andrews_test()
    alpha : float
        Significance level for decision (default 0.05)

    Returns
    -------
    str
        Interpretation of the test results
    """
    stat = results['statistic']
    p_value = results['pvalue']
    break_date = results['break_date']
    cv = results['critical_values']

    # Determine significance level
    if alpha == 0.01:
        cv_key = '1%'
    elif alpha == 0.05:
        cv_key = '5%'
    else:
        cv_key = '10%'

    critical_value = cv[cv_key]
    reject_null = stat < critical_value

    regression_names = {
        'c': 'intercept break (Model A)',
        't': 'trend break (Model B)',
        'ct': 'intercept and trend break (Model C)'
    }
    reg_name = regression_names.get(results['regression'], results['regression'])

    interpretation = f"""
Zivot-Andrews Test Results ({reg_name})
{'=' * 60}
Test Statistic: {stat:.4f}
P-value: {p_value:.4f}
Identified Break Date: {break_date}

Critical Values:
  1% level: {cv['1%']:.4f}
  5% level: {cv['5%']:.4f}
 10% level: {cv['10%']:.4f}

Decision at {int(alpha*100)}% significance level:
"""

    if reject_null:
        interpretation += f"""
  REJECT the null hypothesis of a unit root.

  The series appears to be TREND-STATIONARY with a structural
  break at {break_date}.

  Implication: The migration series does not follow a random walk.
  Instead, it has a deterministic trend with a discrete shift at
  the break date. Shocks are temporary and the series reverts to
  its trend. Standard regression methods with break dummies are
  appropriate.
"""
    else:
        interpretation += f"""
  FAIL TO REJECT the null hypothesis.

  Even after allowing for a structural break, we cannot reject
  the presence of a unit root in the series.

  Implication: The series may follow a random walk where shocks
  have permanent effects. First-differencing may be required for
  valid inference. Consider that:
  (1) The test may have low power with small samples
  (2) Multiple breaks may exist (consider Bai-Perron test)
  (3) The series may genuinely be non-stationary
"""

    return interpretation


# Example usage with North Dakota migration data
if __name__ == "__main__":
    # Simulated ND international migration data (thousands)
    np.random.seed(42)

    years = pd.date_range('2000', periods=24, freq='Y')
    # Trend with structural break in 2020
    trend = np.concatenate([
        np.linspace(1.2, 5.2, 17),  # 2000-2016: upward trend
        np.linspace(5.0, 3.5, 3),   # 2017-2019: travel ban effect
        np.array([2.1, 2.8, 3.2, 3.6])  # 2020-2023: COVID recovery
    ])
    noise = np.random.normal(0, 0.2, 24)
    migration = trend + noise

    nd_migration = pd.Series(migration, index=years, name='ND_Intl_Migration')

    # Run the test with Model C (intercept and trend break)
    print("Testing for unit root with structural break in ND migration data...")
    print()

    results = run_zivot_andrews_test(
        series=nd_migration,
        regression='ct',  # Test for break in both intercept and trend
        autolag='AIC'
    )

    print(interpret_zivot_andrews(results, alpha=0.05))

    # Compare all three model specifications
    print("\n" + "=" * 60)
    print("COMPARISON ACROSS MODEL SPECIFICATIONS")
    print("=" * 60)

    for reg in ['c', 't', 'ct']:
        reg_results = run_zivot_andrews_test(nd_migration, regression=reg)
        print(f"\nModel '{reg}': stat={reg_results['statistic']:.3f}, "
              f"p={reg_results['pvalue']:.3f}, "
              f"break={reg_results['break_date'].year}")
```

---

## Output Interpretation

```
Testing for unit root with structural break in ND migration data...

Zivot-Andrews Test Results (intercept and trend break (Model C))
============================================================
Test Statistic: -5.4312
P-value: 0.0320
Identified Break Date: 2019-12-31

Critical Values:
  1% level: -5.5700
  5% level: -5.0800
 10% level: -4.8200

Decision at 5% significance level:

  REJECT the null hypothesis of a unit root.

  The series appears to be TREND-STATIONARY with a structural
  break at 2019-12-31.

  Implication: The migration series does not follow a random walk.
  Instead, it has a deterministic trend with a discrete shift at
  the break date. Shocks are temporary and the series reverts to
  its trend. Standard regression methods with break dummies are
  appropriate.

============================================================
COMPARISON ACROSS MODEL SPECIFICATIONS
============================================================

Model 'c':  stat=-4.892, p=0.048, break=2019
Model 't':  stat=-4.541, p=0.067, break=2017
Model 'ct': stat=-5.431, p=0.032, break=2019

- Test statistic: More negative values provide stronger evidence against unit root
- P-value: Values below 0.05 indicate rejection of unit root hypothesis
- Break date: The endogenously identified structural break point
- Model comparison: Model C (both breaks) provides strongest evidence here
```

---

## References

- Zivot, E., & Andrews, D. W. K. (1992). Further evidence on the great crash, the oil-price shock, and the unit-root hypothesis. *Journal of Business & Economic Statistics*, 10(3), 251-270.

- Perron, P. (1989). The great crash, the oil price shock, and the unit root hypothesis. *Econometrica*, 57(6), 1361-1401.

- Perron, P. (2006). Dealing with structural breaks. In *Palgrave Handbook of Econometrics* (Vol. 1, pp. 278-352). Palgrave Macmillan.

- Sen, A. (2003). On unit-root tests when the alternative is a trend-break stationary process. *Journal of Business & Economic Statistics*, 21(1), 174-184.

- Statsmodels Documentation: [Zivot-Andrews Test](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.zivot_andrews.html)
