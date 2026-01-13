# Equation Explanation: Bai-Perron Structural Break Detection

**Number in Paper:** Eq. 6
**Category:** Structural Break Detection
**Paper Section:** 2.3.3 Structural Break Tests

---

## What This Equation Does

The Bai-Perron method finds "structural breaks" in a time series - points in time where the underlying pattern of the data fundamentally changes. Unlike tests such as the Chow test (which checks for a break at a pre-specified date), Bai-Perron discovers break dates automatically by searching for the points that best divide the data into distinct segments.

For migration data, structural breaks might occur due to policy changes (like the 2017 Travel Ban), economic shocks (the 2008 financial crisis), or unprecedented events (COVID-19 pandemic). Identifying these breaks is crucial because they indicate when historical patterns no longer apply, affecting both interpretation of past trends and forecasting of future migration.

---

## The Formula

$$
(\hat{T}_1, \ldots, \hat{T}_m) = \arg\min_{T_1, \ldots, T_m} \sum_{j=0}^{m} \sum_{t=T_j+1}^{T_{j+1}} (y_t - \bar{y}_j)^2
$$

---

## Symbol-by-Symbol Breakdown

| Symbol | Meaning | Type | Domain |
|--------|---------|------|--------|
| $(\hat{T}_1, \ldots, \hat{T}_m)$ | The estimated break dates (optimal break points) | Output | Time indices |
| $\arg\min$ | "The values that minimize" - find the break dates that minimize the expression | Operator | - |
| $m$ | Number of structural breaks | Integer | 0, 1, 2, ... |
| $T_j$ | The $j$-th break point (time index) | Variable | Integer time indices |
| $T_0$ | Start of the series (convention: period before first observation) | Boundary | Fixed |
| $T_{m+1}$ | End of the series | Boundary | Fixed at $T$ (total observations) |
| $y_t$ | Observed value at time $t$ | Data | Real numbers |
| $\bar{y}_j$ | Mean of segment $j$ (observations between $T_j$ and $T_{j+1}$) | Computed | Real numbers |
| $(y_t - \bar{y}_j)^2$ | Squared deviation from segment mean | Computed | Non-negative real |

---

## Step-by-Step Interpretation

1. **Divide the series into segments:** The break points $T_1, T_2, \ldots, T_m$ split the full time series into $m+1$ distinct segments. For example, with 2 breaks, you get 3 segments: before the first break, between the two breaks, and after the second break.

2. **Calculate segment means ($\bar{y}_j$):** For each segment $j$, compute the average value of all observations within that segment. This mean represents the "typical" level during that period.

3. **Measure within-segment variation:** For each observation, calculate how far it is from its segment's mean, then square this difference. The squared deviations $(y_t - \bar{y}_j)^2$ measure how well each segment is approximated by its mean.

4. **Sum across all segments:** Add up all the squared deviations from all segments. This gives the total "residual sum of squares" (RSS) - a measure of how much variation remains unexplained by the segmented model.

5. **Find optimal breaks:** Search over all possible combinations of break dates to find the set that minimizes the total RSS. The winning combination identifies the dates where the data most clearly shifts from one regime to another.

6. **Constraint:** In practice, each segment must contain a minimum number of observations (e.g., at least 15% of the sample) to ensure reliable estimation within each regime.

---

## Worked Example

**Setup:**
Suppose we have 20 years of migration data (2004-2023) and want to find the optimal location for 1 structural break:

| Years | Average Migration |
|-------|-------------------|
| 2004-2010 | 1,200 |
| 2011-2023 | 2,800 |

We consider all possible single break points from 2007 to 2020 (allowing at least 3 years on each side).

**Calculation:**
For each candidate break year, we compute the RSS:

```
Break at 2008 (segments: 2004-2008, 2009-2023):
  Segment 1 mean: 1,100 (5 years)
  Segment 2 mean: 2,500 (15 years)
  RSS = sum of squared deviations = 850,000

Break at 2010 (segments: 2004-2010, 2011-2023):
  Segment 1 mean: 1,200 (7 years)
  Segment 2 mean: 2,800 (13 years)
  RSS = 620,000  <-- MINIMUM

Break at 2012 (segments: 2004-2012, 2013-2023):
  Segment 1 mean: 1,600 (9 years)
  Segment 2 mean: 2,900 (11 years)
  RSS = 780,000

Break at 2015 (segments: 2004-2015, 2016-2023):
  Segment 1 mean: 1,900 (12 years)
  Segment 2 mean: 2,700 (8 years)
  RSS = 920,000
```

**Interpretation:**
The optimal break is at 2010 because it minimizes the RSS at 620,000. This means migration to North Dakota appears to have fundamentally shifted around 2010-2011, perhaps due to the oil boom attracting more international workers. The pre-2011 average was about 1,200 migrants per year; post-2010, it jumped to about 2,800.

---

## Key Assumptions

1. **Breaks are discrete regime changes:** The method assumes that at break points, the process shifts abruptly to a new regime rather than transitioning gradually. If the true transition is smooth, Bai-Perron may not accurately characterize it.

2. **Parameters are stable within segments:** Between break points, the data-generating process is assumed to be constant. Each segment follows the same rules throughout.

3. **Minimum segment length:** Each segment must contain enough observations for reliable estimation. This constraint prevents spurious breaks near the boundaries and ensures statistical stability.

4. **Number of breaks is specified or determined by information criterion:** You either pre-specify how many breaks to find, or use BIC/AIC to determine the optimal number.

---

## Common Pitfalls

- **Too many breaks:** Without proper penalization, the algorithm might identify every small fluctuation as a "break." Using BIC to select the number of breaks helps prevent overfitting.

- **Ignoring minimum segment length:** Allowing very short segments can lead to unstable estimates and spurious breaks. Enforce a minimum (typically 10-15% of the sample).

- **Confusing breaks with outliers:** A single extreme year might look like two breaks (before and after the outlier). Visual inspection and domain knowledge help distinguish real regime changes from data anomalies.

- **Asymmetric trimming:** The method cannot detect breaks near the very beginning or end of the series (need enough observations on both sides). This is a feature, not a bug, but means recent potential breaks may be undetectable.

---

## Related Tests

- **Chow Test:** Tests for a break at a known, pre-specified date. Use when you have a specific hypothesis (e.g., "Did the 2017 Travel Ban cause a structural break?").

- **CUSUM Test:** Detects parameter instability through cumulative sums of recursive residuals. Useful for monitoring gradual changes.

- **Zivot-Andrews Test:** A unit root test that simultaneously searches for a structural break. Combines stationarity testing with break detection.

- **Quandt-Andrews Test (SupF):** Tests for a single break at an unknown date. Bai-Perron extends this to multiple breaks.

---

## Python Implementation

```python
import numpy as np
import ruptures as rpt
import pandas as pd
import matplotlib.pyplot as plt

# Load your migration series
# series = pd.Series([...], index=pd.date_range('2000', periods=24, freq='Y'))

# Convert to numpy array
signal = series.values.reshape(-1, 1)

# Method 1: Specify number of breaks
n_breaks = 2
algo = rpt.Binseg(model="l2").fit(signal)
breakpoints = algo.predict(n_bkps=n_breaks)

print(f"Break points (with {n_breaks} breaks):")
for bp in breakpoints[:-1]:  # Last value is always n (end of series)
    print(f"  Position: {bp}, Year: {series.index[bp-1]}")

# Method 2: Use penalty to automatically determine number of breaks
# Higher penalty = fewer breaks; lower penalty = more breaks
penalty = np.log(len(signal)) * 2  # BIC-like penalty
algo = rpt.Binseg(model="l2").fit(signal)
breakpoints_auto = algo.predict(pen=penalty)

print(f"\nBreak points (auto-detected with BIC penalty):")
for bp in breakpoints_auto[:-1]:
    print(f"  Position: {bp}, Year: {series.index[bp-1]}")

# Calculate segment means
def get_segment_stats(series, breakpoints):
    """Calculate statistics for each segment."""
    segments = []
    prev = 0
    for bp in breakpoints:
        segment_data = series.iloc[prev:bp]
        segments.append({
            'start': series.index[prev],
            'end': series.index[bp-1],
            'mean': segment_data.mean(),
            'std': segment_data.std(),
            'n_obs': len(segment_data)
        })
        prev = bp
    return pd.DataFrame(segments)

segment_stats = get_segment_stats(series, breakpoints_auto)
print("\nSegment Statistics:")
print(segment_stats.to_string(index=False))

# Visualization
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(series.index, series.values, 'b-', linewidth=1.5, label='Migration')

# Add vertical lines at break points
for bp in breakpoints_auto[:-1]:
    ax.axvline(x=series.index[bp-1], color='red', linestyle='--',
               linewidth=2, label='Structural Break' if bp == breakpoints_auto[0] else '')

# Add segment means
prev = 0
for bp in breakpoints_auto:
    segment_mean = series.iloc[prev:bp].mean()
    ax.hlines(y=segment_mean, xmin=series.index[prev], xmax=series.index[bp-1],
              colors='green', linestyles='-', linewidth=2, alpha=0.7)
    prev = bp

ax.set_xlabel('Year')
ax.set_ylabel('International Migration')
ax.set_title('Bai-Perron Structural Break Detection')
ax.legend()
plt.tight_layout()
plt.savefig('bai_perron_breaks.png', dpi=150)
plt.show()
```

---

## References

- Bai, J., & Perron, P. (1998). Estimating and testing linear models with multiple structural changes. *Econometrica*, 66(1), 47-78.

- Bai, J., & Perron, P. (2003). Computation and analysis of multiple structural change models. *Journal of Applied Econometrics*, 18(1), 1-22.

- Zeileis, A., Kleiber, C., Kramer, W., & Hornik, K. (2003). Testing and dating of structural changes in practice. *Computational Statistics & Data Analysis*, 44(1-2), 109-123.
