# Equation Explanation: Hodrick-Prescott Filter

**Number in Paper:** Eq. 1
**Category:** Trend Decomposition
**Paper Section:** 2.2 Descriptive and Concentration Methods

---

## What This Equation Does

The Hodrick-Prescott (HP) Filter is a mathematical technique for separating a time series into two components: a smooth underlying trend and a cyclical (fluctuating) component. Think of it like drawing a smooth curve through noisy data points. For example, if you have annual migration data that bounces up and down year to year, the HP filter helps you see the long-term direction (trend) separate from the short-term fluctuations (cycles).

The filter works by finding the "best" trend line that balances two competing goals: (1) staying close to the actual data points, and (2) being as smooth as possible. The smoothing parameter (lambda) controls this trade-off. A higher lambda produces a smoother trend line, while a lower lambda allows the trend to follow the data more closely. For annual data, the standard value of lambda = 6.25 (based on Ravn and Uhlig's 2002 research) produces trends that capture multi-year patterns while filtering out year-to-year noise.

---

## The Formula

$$
\min_{\{g_t\}} \left\{ \sum_{t=1}^{T} (y_t - g_t)^2 + \lambda \sum_{t=2}^{T-1} [(g_{t+1} - g_t) - (g_t - g_{t-1})]^2 \right\}
$$

---

## Symbol-by-Symbol Breakdown

| Symbol | Meaning | Type | Domain |
|--------|---------|------|--------|
| $y_t$ | Observed value at time t (e.g., annual migration count) | Variable | Real numbers |
| $g_t$ | Trend component at time t (what we're solving for) | Variable | Real numbers |
| $T$ | Total number of time periods in the series | Constant | Positive integer |
| $\lambda$ | Smoothing parameter (6.25 for annual data) | Parameter | Positive real number |
| $\min_{\{g_t\}}$ | "Find the values of $g_t$ that minimize..." | Operator | - |

---

## Step-by-Step Interpretation

1. **The Goodness-of-Fit Term:** $\sum_{t=1}^{T} (y_t - g_t)^2$

   This sums up the squared differences between the actual data ($y_t$) and the estimated trend ($g_t$) at each time point. Smaller values mean the trend stays closer to the original data. This is essentially measuring how well the trend "fits" the data.

2. **The Smoothness Penalty Term:** $\sum_{t=2}^{T-1} [(g_{t+1} - g_t) - (g_t - g_{t-1})]^2$

   This measures how "wiggly" the trend line is. The expression $(g_{t+1} - g_t) - (g_t - g_{t-1})$ calculates the change in the slope of the trend (the second derivative, or "acceleration"). When this equals zero, the trend is changing at a constant rate (a straight line). Large values indicate sudden direction changes. Squaring and summing these penalizes abrupt changes.

3. **The Lambda Weight:** $\lambda$

   This parameter controls the trade-off between fit and smoothness. With lambda = 6.25 (annual data), we're saying smoothness is moderately important. If lambda were 0, we'd just get the original data back (perfect fit, no smoothing). If lambda were infinite, we'd get a straight line (maximum smoothness, poor fit).

4. **The Minimization:** $\min_{\{g_t\}}$

   The filter finds the trend values $g_1, g_2, ..., g_T$ that make the entire expression as small as possible, balancing fit against smoothness.

---

## Worked Example

**Setup:**
Suppose we have 5 years of migration data: [100, 120, 110, 140, 130]. We want to extract the underlying trend.

**Calculation:**
```
Original Data (y_t):    [100, 120, 110, 140, 130]

The HP filter finds trend values (g_t) that minimize:
- Sum of squared deviations from data (fit term)
- Plus 6.25 times sum of squared second differences (smoothness term)

After optimization (computed by software):
Trend (g_t):            [103, 112, 121, 130, 139]
Cycle (y_t - g_t):      [-3,   8, -11,  10,  -9]
```

**Interpretation:**
- The trend shows a steady upward pattern, increasing by about 9 units per year
- The cycle captures year-to-year deviations from this trend
- Year 3's dip (110) and Year 4's spike (140) are mostly attributed to cyclical fluctuations
- The underlying trend suggests consistent growth throughout the period

---

## Key Assumptions

1. **The data can be meaningfully decomposed:** The series genuinely consists of a smooth trend plus cyclical deviations (not pure noise or structural breaks)

2. **Lambda is appropriate for the data frequency:** Lambda = 6.25 is calibrated for annual data. Quarterly data typically uses 1,600; monthly data uses 129,600

3. **Stationarity of the cycle:** The cyclical component is assumed to be stationary (constant variance and no trend in the fluctuations)

4. **No structural breaks:** Abrupt, permanent shifts in the series may produce misleading trends near the break point

---

## Common Pitfalls

- **End-point bias:** The HP filter performs poorly at the beginning and end of the series because it cannot "look ahead" or "look back" sufficiently. Interpretation of the trend in the first and last 2-3 periods should be cautious.

- **Spurious cycles:** If the data has a unit root (non-stationary), the HP filter can create artificial cyclical patterns that don't actually exist. Always test for stationarity first.

- **Wrong lambda value:** Using quarterly-data lambda (1,600) on annual data produces trends that are too smooth, potentially hiding real multi-year patterns.

- **Over-interpretation:** The decomposition is a statistical convenience, not a physical separation. The "trend" and "cycle" are constructed quantities, not directly observable phenomena.

---

## Related Tests

- **ADF Test (Eq. 4):** Used to check whether the original series is stationary; helps validate whether HP filtering is appropriate
- **KPSS Test:** Alternative stationarity test to complement ADF
- **Shapiro-Wilk Test:** Can assess whether the extracted cycle follows a normal distribution, as often assumed

---

## Python Implementation

```python
from statsmodels.tsa.filters.hp_filter import hpfilter
import pandas as pd
import numpy as np

# Example: HP filtering of annual migration data
# Using lambda = 6.25 for annual data (Ravn-Uhlig 2002 recommendation)

# Sample data
migration_data = pd.Series(
    [1000, 1050, 980, 1100, 1150, 1080, 1200, 1250],
    index=pd.date_range('2015', periods=8, freq='Y')
)

# Apply HP filter
cycle, trend = hpfilter(migration_data, lamb=6.25)

# Results
print("Original Data:")
print(migration_data.values)

print("\nTrend Component:")
print(trend.values)

print("\nCyclical Component:")
print(cycle.values)

# The cycle can be used to identify boom/bust periods
# Positive cycle = above-trend migration
# Negative cycle = below-trend migration
```

---

## References

- Hodrick, R. J., & Prescott, E. C. (1997). Postwar U.S. Business Cycles: An Empirical Investigation. *Journal of Money, Credit and Banking*, 29(1), 1-16.
- Ravn, M. O., & Uhlig, H. (2002). On Adjusting the Hodrick-Prescott Filter for the Frequency of Observations. *Review of Economics and Statistics*, 84(2), 371-376.
- Hamilton, J. D. (2018). Why You Should Never Use the Hodrick-Prescott Filter. *Review of Economics and Statistics*, 100(5), 831-843. [Critical perspective]
