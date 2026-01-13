# Equation Explanation: Difference-in-Differences

**Number in Paper:** Eq. 12
**Category:** Causal Inference
**Paper Section:** 2.7.1 Difference-in-Differences

---

## What This Equation Does

Difference-in-Differences (DiD) is a method for estimating the causal effect of a policy or event by comparing how outcomes changed over time between a group that was affected by the policy (the "treatment group") and a group that was not affected (the "control group"). The key insight is that by taking the difference of two differences, we can cancel out factors that affect both groups equally and isolate the true effect of the treatment.

In the context of this paper, DiD is used to estimate how the 2017 Travel Ban affected immigration from banned countries to North Dakota. The treatment group consists of countries affected by the ban, while the control group consists of countries not affected. By comparing how immigration changed before and after 2017 for both groups, we can estimate how much of the decline in immigration from banned countries was actually caused by the ban itself, rather than other factors affecting all immigration.

---

## The Formula

$$
\ln(y_{ct} + 1) = \alpha_c + \lambda_t + \delta \cdot (\text{Affected}_c \times \text{Post}_t) + \varepsilon_{ct}
$$

---

## Symbol-by-Symbol Breakdown

| Symbol | Meaning | Type | Domain |
|--------|---------|------|--------|
| $y_{ct}$ | Number of immigrant arrivals from country $c$ in year $t$ | Outcome variable | Non-negative integers |
| $\ln(y_{ct} + 1)$ | Natural log of arrivals plus one (to handle zeros) | Transformed outcome | Real numbers |
| $\alpha_c$ | Country fixed effect (captures permanent differences between countries) | Parameter | Real numbers |
| $\lambda_t$ | Year fixed effect (captures shocks affecting all countries in year $t$) | Parameter | Real numbers |
| $\delta$ | Average Treatment Effect on the Treated (ATT) | Parameter of interest | Real numbers |
| $\text{Affected}_c$ | Indicator variable: 1 if country $c$ was affected by Travel Ban, 0 otherwise | Binary indicator | {0, 1} |
| $\text{Post}_t$ | Indicator variable: 1 if year $t$ is 2018 or later, 0 otherwise | Binary indicator | {0, 1} |
| $\text{Affected}_c \times \text{Post}_t$ | Interaction term: 1 only for affected countries after the ban | Binary indicator | {0, 1} |
| $\varepsilon_{ct}$ | Error term (unexplained variation) | Random variable | Real numbers |

---

## Step-by-Step Interpretation

1. **Log transformation $\ln(y_{ct} + 1)$:** We take the natural logarithm of arrivals to make the model estimate percentage changes rather than absolute changes. The "+1" is added because log(0) is undefined, and some country-year combinations have zero arrivals.

2. **Country fixed effects $\alpha_c$:** These capture all time-invariant characteristics of each origin country that affect immigration levels (e.g., population size, distance to ND, historical ties). By including these, we control for baseline differences between countries.

3. **Year fixed effects $\lambda_t$:** These capture shocks that affect immigration from all countries in a given year (e.g., US economic conditions, overall immigration policy changes). By including these, we control for common temporal trends.

4. **The interaction term $\text{Affected}_c \times \text{Post}_t$:** This is the heart of DiD. It equals 1 only for observations from affected countries in the post-ban period. The coefficient $\delta$ on this term measures the additional change in immigration for affected countries after the ban, beyond what we would expect based on overall time trends and country characteristics.

5. **The key coefficient $\delta$ (ATT):** If $\delta$ is negative and statistically significant, it means affected countries experienced an additional decline in immigration beyond what unaffected countries experienced. Because we used log transformation, we can interpret the percentage effect as: $(\exp(\delta) - 1) \times 100\%$.

---

## Worked Example

**Setup:**
Consider a simplified scenario with 2 countries over 4 years (2016-2019):
- Somalia (affected by Travel Ban)
- India (not affected)
- Treatment begins in 2018

**Data:**
| Country | Year | Arrivals | Affected | Post | Affected x Post |
|---------|------|----------|----------|------|-----------------|
| Somalia | 2016 | 100 | 1 | 0 | 0 |
| Somalia | 2017 | 110 | 1 | 0 | 0 |
| Somalia | 2018 | 40 | 1 | 1 | 1 |
| Somalia | 2019 | 35 | 1 | 1 | 1 |
| India | 2016 | 200 | 0 | 0 | 0 |
| India | 2017 | 220 | 0 | 0 | 0 |
| India | 2018 | 210 | 0 | 1 | 0 |
| India | 2019 | 200 | 0 | 1 | 0 |

**Calculation:**
```
Step 1: Calculate average log arrivals for each group and period

Somalia Pre-ban:  avg(ln(101), ln(111)) = avg(4.62, 4.71) = 4.665
Somalia Post-ban: avg(ln(41), ln(36))   = avg(3.71, 3.58) = 3.645

India Pre-ban:    avg(ln(201), ln(221)) = avg(5.30, 5.40) = 5.350
India Post-ban:   avg(ln(211), ln(201)) = avg(5.35, 5.30) = 5.325

Step 2: Calculate the difference for each group

Somalia change: 3.645 - 4.665 = -1.020 (large decline)
India change:   5.325 - 5.350 = -0.025 (small decline)

Step 3: Difference-in-Differences

delta = Somalia change - India change
delta = -1.020 - (-0.025)
delta = -0.995

Step 4: Convert to percentage effect

Percentage effect = (exp(-0.995) - 1) x 100%
                  = (0.370 - 1) x 100%
                  = -63%
```

**Interpretation:**
The Travel Ban caused an estimated 63% reduction in immigration from affected countries to North Dakota, beyond what would have occurred based on overall trends. India's slight decline of 2.5% represents the counterfactual--what would have happened to Somalia without the ban. The ban's true effect (-63%) is isolated by subtracting this common trend.

---

## Key Assumptions

1. **Parallel Trends:** In the absence of treatment, the treatment and control groups would have followed parallel trends. This is the most critical assumption and cannot be directly tested, though we can examine pre-treatment trends for evidence.

2. **No Anticipation Effects:** Units did not change their behavior in anticipation of treatment. If affected countries' immigration changed before the official ban date, this would violate the assumption.

3. **No Spillover Effects (SUTVA):** The treatment of affected countries does not affect outcomes for unaffected countries. For example, if immigrants who would have come from Somalia instead come from India, this would bias results.

4. **Treatment Timing Accuracy:** The post-treatment period is correctly identified. For the Travel Ban, the policy was announced in 2017 but effects may have started immediately or with a lag.

5. **Common Support:** Both groups have observations in both pre and post periods.

---

## Common Pitfalls

- **Failing to test parallel trends:** Always examine whether pre-treatment trends were similar. If control countries were already trending differently before the ban, DiD will give biased estimates. Use event study plots and formal pre-trend tests.

- **Ignoring clustering:** When observations are correlated within groups (e.g., same country over time), standard errors must be clustered at the group level. Failing to cluster leads to overly small standard errors and false positives.

- **Too few clusters:** With only a small number of treated units (e.g., 7 banned countries), standard cluster-robust inference may be unreliable. Use wild cluster bootstrap or randomization inference for small samples.

- **Misinterpreting the log coefficient:** A coefficient of -0.5 does not mean a 50% decline. Use the formula $(\exp(\delta) - 1) \times 100\%$ for the correct percentage interpretation.

- **Staggered treatment timing:** If different units are treated at different times, the standard two-way fixed effects estimator can be biased. Use newer robust DiD estimators (Callaway-Sant'Anna, Sun-Abraham) when treatment timing varies.

---

## Related Tests

- **Parallel Trends Test (Pre-Treatment F-Test):** Tests whether treatment and control groups had similar trends before treatment. Fail to reject null hypothesis supports parallel trends assumption.

- **Wild Cluster Bootstrap:** Provides robust p-values when the number of clusters is small (fewer than 20-30). Essential for Travel Ban analysis with only 7 affected countries.

- **Randomization Inference:** Permutes treatment assignment to construct exact p-values under the Fisher sharp null. Provides finite-sample valid inference regardless of cluster count.

- **Event Study / Dynamic DiD:** Extends DiD to estimate separate effects for each pre and post period, allowing visual assessment of parallel trends and effect dynamics.

---

## Python Implementation

```python
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import pandas as pd
import numpy as np

# Create interaction term
df['treated_x_post'] = df['treated'] * df['post']

# Add fixed effects dummies
y = df['log_arrivals'].values
X = pd.get_dummies(df[['treated_x_post', 'nationality', 'year']], drop_first=True)
X = sm.add_constant(X)

# Fit with clustered standard errors
model = OLS(y, X).fit(
    cov_type='cluster',
    cov_kwds={'groups': df['nationality']}
)
att = model.params['treated_x_post']
pct_effect = (np.exp(att) - 1) * 100
```

---

## References

- Card, David, and Alan B. Krueger (1994). "Minimum Wages and Employment: A Case Study of the Fast-Food Industry in New Jersey and Pennsylvania." *American Economic Review* 84(4): 772-793.

- Angrist, Joshua D., and Jorn-Steffen Pischke (2009). *Mostly Harmless Econometrics: An Empiricist's Companion.* Princeton University Press.

- Cameron, A. Colin, and Douglas L. Miller (2015). "A Practitioner's Guide to Cluster-Robust Inference." *Journal of Human Resources* 50(2): 317-372.

- Callaway, Brantly, and Pedro H.C. Sant'Anna (2021). "Difference-in-Differences with Multiple Time Periods." *Journal of Econometrics* 225(2): 200-230.
