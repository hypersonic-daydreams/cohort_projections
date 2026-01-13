# Equation Explanation: Interrupted Time Series

**Number in Paper:** Eq. 13
**Category:** Causal Inference
**Paper Section:** 2.7.2 Interrupted Time Series

---

## What This Equation Does

Interrupted Time Series (ITS) is a quasi-experimental method for evaluating the effect of an intervention or event that occurs at a known point in time. Unlike Difference-in-Differences, ITS does not require a control group. Instead, it uses the pre-intervention trend as a counterfactual: we assume that without the intervention, the pre-existing trend would have continued. By modeling both the pre and post-intervention periods, ITS can detect two types of effects: an immediate "level shift" (a sudden jump or drop when the intervention occurs) and a "slope change" (a change in the rate of growth or decline after the intervention).

In this paper, ITS is used to estimate the effect of COVID-19 on international migration to US states. The intervention point is 2020, when the pandemic caused widespread travel restrictions and economic disruption. The analysis examines whether COVID caused an immediate drop in migration (level shift) and whether it changed the trajectory of migration recovery (slope change). By including state fixed effects, the model can account for permanent differences between states while estimating a common COVID effect.

---

## The Formula

$$
y_{st} = \alpha_s + \beta_1 t + \beta_2 \text{Post}_{2020,t} + \beta_3 (t - 2020)\text{Post}_{2020,t} + \varepsilon_{st}
$$

---

## Symbol-by-Symbol Breakdown

| Symbol | Meaning | Type | Domain |
|--------|---------|------|--------|
| $y_{st}$ | Net international migration for state $s$ in year $t$ | Outcome variable | Real numbers (can be negative) |
| $\alpha_s$ | State fixed effect (captures permanent state characteristics) | Parameter | Real numbers |
| $t$ | Time variable (year) | Independent variable | Integer years |
| $\beta_1$ | Pre-treatment trend (annual change in migration before 2020) | Parameter | Real numbers |
| $\text{Post}_{2020,t}$ | Indicator: 1 if $t \geq 2020$, 0 otherwise | Binary indicator | {0, 1} |
| $\beta_2$ | Immediate level shift at intervention (COVID effect at onset) | Parameter of interest | Real numbers |
| $(t - 2020)$ | Years elapsed since 2020 | Time since intervention | Non-negative integers |
| $(t - 2020)\text{Post}_{2020,t}$ | Interaction: years since 2020, only for post-period | Variable | Non-negative integers or 0 |
| $\beta_3$ | Change in trend after 2020 (recovery rate change) | Parameter of interest | Real numbers |
| $\varepsilon_{st}$ | Error term | Random variable | Real numbers |

---

## Step-by-Step Interpretation

1. **State fixed effects $\alpha_s$:** Each state has its own intercept, capturing all time-invariant factors affecting that state's immigration (e.g., population size, economic base, immigrant networks). This allows us to compare within-state changes over time rather than across states.

2. **Pre-treatment trend $\beta_1 t$:** This captures how migration was changing each year before COVID. If $\beta_1 = 500$, immigration was increasing by about 500 persons per year on average. This trend is assumed to continue in the post-period, representing the counterfactual.

3. **Level shift $\beta_2 \text{Post}_{2020,t}$:** This coefficient measures the immediate, one-time change in migration when COVID hit. A negative $\beta_2$ means migration suddenly dropped when the pandemic began. This captures the acute impact of travel restrictions, economic shutdowns, and health concerns.

4. **Slope change $\beta_3 (t-2020)\text{Post}_{2020,t}$:** This measures whether the trend changed after 2020. If $\beta_3 > 0$, migration was recovering faster than the pre-COVID trend. If $\beta_3 < 0$, the decline continued to accelerate or recovery was slower than before. The total post-period trend is $\beta_1 + \beta_3$.

5. **Separating level and slope effects:** A key strength of ITS is distinguishing these two effects. COVID might have caused a large immediate drop ($\beta_2$ very negative) but rapid recovery afterward ($\beta_3$ positive), or it might have caused a smaller immediate drop but lasting damage to growth rates ($\beta_3$ negative).

---

## Worked Example

**Setup:**
Consider North Dakota's net international migration from 2015-2023:
- Pre-COVID period: 2015-2019
- Intervention: 2020
- Post-COVID period: 2020-2023

**Data:**
| Year | Time (t) | Migration | Post | t - 2020 | (t-2020) x Post |
|------|----------|-----------|------|----------|-----------------|
| 2015 | 2015 | 2,800 | 0 | -5 | 0 |
| 2016 | 2016 | 3,000 | 0 | -4 | 0 |
| 2017 | 2017 | 3,100 | 0 | -3 | 0 |
| 2018 | 2018 | 3,300 | 0 | -2 | 0 |
| 2019 | 2019 | 3,500 | 0 | -1 | 0 |
| 2020 | 2020 | 1,500 | 1 | 0 | 0 |
| 2021 | 2021 | 2,000 | 1 | 1 | 1 |
| 2022 | 2022 | 2,700 | 1 | 2 | 2 |
| 2023 | 2023 | 3,200 | 1 | 3 | 3 |

**Calculation:**
```
Step 1: Estimate pre-treatment trend (from 2015-2019)

Using simple linear regression on pre-period:
Migration = intercept + beta_1 * year
From the data: Migration increased roughly 175 per year
beta_1 (approximately) = 175

Step 2: Calculate counterfactual for 2020

Expected 2020 (if trend continued) = 3,500 + 175 = 3,675

Step 3: Estimate level shift

beta_2 = Actual 2020 - Expected 2020
beta_2 = 1,500 - 3,675 = -2,175

This represents the immediate COVID impact.

Step 4: Estimate post-treatment trend

Post-2020 change per year (from 2020-2023):
(3,200 - 1,500) / 3 = 567 per year

Post-trend = beta_1 + beta_3
567 = 175 + beta_3
beta_3 = 392

Step 5: Interpret the results

- Level shift (beta_2): -2,175 persons
  COVID caused an immediate drop of 2,175 migrants below trend.

- Slope change (beta_3): +392 persons/year
  After COVID, migration recovered faster than the pre-COVID trend.
  Pre-COVID trend: +175/year
  Post-COVID trend: +567/year (175 + 392)

- By 2023, predicted vs actual:
  Without COVID: 3,675 + 175*3 = 4,200
  With COVID: 1,500 + 567*3 = 3,201
  Cumulative loss through 2023: approximately 999 migrants
```

**Interpretation:**
COVID caused an immediate loss of about 2,175 international migrants in 2020, representing a 59% drop from the expected level. However, recovery was faster than the pre-COVID trend, with migration increasing by 567 persons per year instead of 175. Despite faster recovery, ND had still not returned to the projected no-COVID trajectory by 2023.

---

## Key Assumptions

1. **Stable Pre-Intervention Trend:** The pre-intervention trend accurately represents what would have happened without the intervention. If the trend was about to change anyway (e.g., due to a separate policy), ITS will incorrectly attribute that change to the intervention.

2. **No Concurrent Events:** No other major events occurred at the same time as the intervention that could explain the observed changes. For COVID analysis, this is challenging because many policies changed simultaneously.

3. **Correct Specification of Trend:** The pre-treatment trend is correctly modeled (linear vs. non-linear). If the true trend was accelerating before COVID but we model it as linear, we may over or underestimate the COVID effect.

4. **Immediate and Sustained Effects:** The model assumes the intervention has both an immediate effect (level shift) and a sustained effect on trajectory (slope change). If effects are delayed or temporary, this specification may not capture them well.

5. **Independence of Time Points:** Observations are not too strongly autocorrelated. Strong autocorrelation can inflate type I error rates.

---

## Common Pitfalls

- **Assuming linearity when trends are non-linear:** Always plot the data first. If pre-intervention trends were curving, a linear model will give biased counterfactual predictions. Consider polynomial or log transformations.

- **Ignoring seasonality:** For monthly or quarterly data, seasonal patterns must be modeled explicitly. Annual data (as used here) avoids this issue but loses granularity.

- **Choosing the wrong intervention point:** The timing of the intervention must be known and precise. For COVID, was the intervention March 2020 (first US cases), April 2020 (lockdowns), or the full year 2020? The choice affects results.

- **Neglecting autocorrelation:** Time series errors are often correlated. Use Newey-West standard errors or model the autocorrelation structure directly. For panel ITS, cluster standard errors by state.

- **Overfitting with short series:** ITS requires sufficient pre and post-intervention observations to estimate trends reliably. With only 3-4 years before and after, estimates will be imprecise.

- **Confusing level and slope effects:** A positive slope change after a negative level shift does not mean "recovery to normal"--it means recovery is faster than pre-treatment growth. Check whether the series has actually returned to counterfactual levels.

---

## Related Tests

- **Durbin-Watson Test:** Tests for autocorrelation in residuals. Significant autocorrelation suggests the model may need adjustment or robust standard errors.

- **Cumby-Huizinga Test:** More flexible test for autocorrelation at various lags, useful for ITS models where autocorrelation structure is unknown.

- **Segmented Regression (Alternative Parameterization):** Some software implements ITS with separate intercepts and slopes for each segment rather than level and slope change parameters. Results are mathematically equivalent but interpretation differs.

- **Bai-Perron Test:** Can be used to detect structural breaks endogenously if the intervention timing is uncertain.

---

## Python Implementation

```python
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import pandas as pd
import numpy as np

# Create ITS variables
df['time'] = df['year'] - df['year'].min()
df['post_covid'] = (df['year'] >= 2020).astype(int)
df['time_since_covid'] = np.maximum(0, df['year'] - 2020)

# Fit with state FE and clustered SE
X = pd.DataFrame({
    'time': df['time'],
    'post_covid': df['post_covid'],
    'time_since_covid': df['time_since_covid']
})
state_dummies = pd.get_dummies(df['state'], drop_first=True)
X = pd.concat([X, state_dummies], axis=1)
X = sm.add_constant(X)

model = OLS(df['intl_migration'], X).fit(
    cov_type='cluster',
    cov_kwds={'groups': df['state']}
)
level_shift = model.params['post_covid']
trend_change = model.params['time_since_covid']
```

---

## References

- Wagner, A. Kathryn, et al. (2002). "Segmented Regression Analysis of Interrupted Time Series Studies in Medication Use Research." *Journal of Clinical Pharmacy and Therapeutics* 27(4): 299-309.

- Bernal, James Lopez, Steven Cummins, and Antonio Gasparrini (2017). "Interrupted Time Series Regression for the Evaluation of Public Health Interventions: A Tutorial." *International Journal of Epidemiology* 46(1): 348-355.

- Linden, Ariel (2015). "Conducting Interrupted Time-Series Analysis for Single and Multiple-Group Comparisons." *Stata Journal* 15(2): 480-500.

- Kontopantelis, Evangelos, et al. (2015). "Regression Based Quasi-Experimental Approach When Randomisation Is Not an Option: Interrupted Time Series Analysis." *BMJ* 350: h2750.
