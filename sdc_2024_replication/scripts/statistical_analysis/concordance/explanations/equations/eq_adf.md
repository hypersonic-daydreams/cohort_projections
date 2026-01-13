# Equation Explanation: Augmented Dickey-Fuller Regression

**Number in Paper:** Eq. 4
**Category:** Unit Root Test
**Paper Section:** 2.3.1 Unit Root Tests

---

## What This Equation Does

The Augmented Dickey-Fuller (ADF) regression tests whether a time series is "stationary" or contains a "unit root." A stationary series has a constant mean and variance over time, meaning it tends to revert back to some average level. A series with a unit root, by contrast, can wander anywhere and has no tendency to return to a mean value - shocks to the series have permanent effects.

This matters for migration forecasting because standard regression techniques assume stationarity. If you apply these techniques to a non-stationary series, you can get misleading results (spurious correlations). The ADF test helps determine whether the raw migration data can be modeled directly, or whether you first need to transform it (typically by taking differences) to make it stationary.

---

## The Formula

$$
\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \sum_{j=1}^{p} \delta_j \Delta y_{t-j} + \varepsilon_t
$$

---

## Symbol-by-Symbol Breakdown

| Symbol | Meaning | Type | Domain |
|--------|---------|------|--------|
| $\Delta y_t$ | First difference of the series at time $t$ (i.e., $y_t - y_{t-1}$) | Dependent variable | Real numbers |
| $\alpha$ | Constant (intercept) | Parameter | Real numbers |
| $\beta$ | Trend coefficient | Parameter | Real numbers |
| $t$ | Time trend (1, 2, 3, ..., T) | Variable | Positive integers |
| $\gamma$ | Coefficient on the lagged level - this is the key test coefficient | Parameter | Real numbers (negative if stationary) |
| $y_{t-1}$ | Value of the series at time $t-1$ (lagged level) | Variable | Real numbers |
| $\delta_j$ | Coefficients on lagged differences | Parameters | Real numbers |
| $p$ | Number of lags included (selected by AIC) | Integer | 0, 1, 2, ... |
| $\varepsilon_t$ | Error term (white noise) | Random variable | Real numbers |

---

## Step-by-Step Interpretation

1. **The left side ($\Delta y_t$):** We model the change in the series from one period to the next, not the level itself. This is because if a series has a unit root, its level is non-stationary but its differences may be stationary.

2. **Constant term ($\alpha$):** Allows for a non-zero average change. If migration tends to increase by a fixed amount each year on average, this captures that drift.

3. **Trend term ($\beta t$):** Allows for a linear time trend. If there is a secular upward or downward trajectory in migration beyond just drift, this captures it.

4. **The key term ($\gamma y_{t-1}$):** This is what we are testing. If $\gamma = 0$, the lagged level has no effect on the current change, implying the series has a unit root (non-stationary). If $\gamma < 0$, the series tends to revert toward its mean (stationary). The more negative $\gamma$ is, the faster the mean reversion.

5. **Lagged differences ($\sum_{j=1}^{p} \delta_j \Delta y_{t-j}$):** These "augment" the basic Dickey-Fuller test by accounting for serial correlation in the errors. Without them, autocorrelation in the data would bias the test. The number of lags $p$ is typically selected by minimizing the AIC.

6. **Error term ($\varepsilon_t$):** Random noise assumed to be independent and identically distributed.

---

## Worked Example

**Setup:**
Suppose we have annual international migration to North Dakota for 5 years:

| Year | Migration ($y_t$) | $\Delta y_t$ | $y_{t-1}$ |
|------|-------------------|--------------|-----------|
| 2019 | 2,100 | - | - |
| 2020 | 1,800 | -300 | 2,100 |
| 2021 | 1,950 | 150 | 1,800 |
| 2022 | 2,200 | 250 | 1,950 |
| 2023 | 2,150 | -50 | 2,200 |

**Calculation:**
Suppose we run the ADF regression (with no lags, for simplicity) and get:

```
Estimated equation:
Delta_y_t = 500 + 0.01*t - 0.25*y_{t-1} + error

Key coefficient: gamma = -0.25
Standard error of gamma: 0.10
ADF test statistic = gamma / SE(gamma) = -0.25 / 0.10 = -2.50
```

**Interpretation:**
The ADF test statistic is -2.50. We compare this to critical values (at 5% significance, roughly -2.86 for a small sample with constant and trend). Since -2.50 > -2.86, we fail to reject the null hypothesis. This suggests the series may have a unit root (non-stationary).

However, with only 4 observations used in estimation, the test has very low power. In practice, more data points are needed for reliable inference.

---

## Key Assumptions

1. **The error term is white noise:** After including enough lags, the residuals should be uncorrelated and have constant variance. The lag selection (via AIC) is designed to ensure this.

2. **The data generating process matches the specified model:** If the true process includes features not in the model (e.g., structural breaks), the test may give misleading results.

3. **No structural breaks in the series:** The ADF test assumes parameters are stable over time. If there was a policy change (like the 2017 Travel Ban), this assumption may be violated.

4. **Sufficient sample size:** The ADF test has low power with small samples, meaning it may fail to detect stationarity even when the series is truly stationary.

---

## Common Pitfalls

- **Choosing too few lags:** If $p$ is too small, the residuals will be autocorrelated, biasing the test toward false rejection of the unit root.

- **Choosing too many lags:** If $p$ is too large, the test loses power (less likely to detect stationarity when it exists). Using AIC for lag selection balances these concerns.

- **Ignoring structural breaks:** If the series has a break (e.g., COVID-19 disruption), the ADF test may falsely indicate a unit root. Consider the Zivot-Andrews test as a robustness check.

- **Misinterpreting the null hypothesis:** The null is that there IS a unit root. Failing to reject does not prove non-stationarity; it just means we lack evidence of stationarity.

---

## Related Tests

- **Phillips-Perron Test:** A nonparametric alternative that corrects for serial correlation differently. Good robustness check.
- **KPSS Test:** Has the opposite null hypothesis (series is stationary). Using both ADF and KPSS provides stronger evidence about the series properties.
- **Zivot-Andrews Test:** A unit root test that allows for one structural break at an unknown date.

---

## Python Implementation

```python
from statsmodels.tsa.stattools import adfuller
import pandas as pd

# Load your migration series
# series = pd.Series([...])  # Your data here

# Run the ADF test
# regression='ct' includes constant and trend; use 'c' for constant only
result = adfuller(series, maxlag=None, regression='ct', autolag='AIC')

# Unpack results
adf_statistic = result[0]
p_value = result[1]
used_lag = result[2]
n_obs = result[3]
critical_values = result[4]
icbest = result[5]

# Interpretation
print(f"ADF Statistic: {adf_statistic:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Lags Used: {used_lag}")
print(f"Number of Observations: {n_obs}")
print("Critical Values:")
for key, value in critical_values.items():
    print(f"  {key}: {value:.4f}")

if p_value < 0.05:
    print("\nConclusion: Reject H0 - Series is stationary")
else:
    print("\nConclusion: Fail to reject H0 - Series may have unit root")
```

---

## References

- Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. *Journal of the American Statistical Association*, 74(366), 427-431.

- Said, S. E., & Dickey, D. A. (1984). Testing for unit roots in autoregressive-moving average models of unknown order. *Biometrika*, 71(3), 599-607.

- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press. (Chapter 17)
