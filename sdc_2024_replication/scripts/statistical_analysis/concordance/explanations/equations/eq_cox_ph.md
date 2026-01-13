# Equation Explanation: Cox Proportional Hazards Model

**Number in Paper:** Eq. 17
**Category:** Survival Analysis
**Paper Section:** 2.8 Duration Analysis

---

## What This Equation Does

The Cox Proportional Hazards model answers the question: "What factors influence how long immigration waves last?" While the Kaplan-Meier estimator (Eq. 16) simply describes survival patterns, the Cox model identifies which characteristics make events happen faster or slower.

The key output is the "hazard ratio" - a multiplier that tells you how much a particular factor changes the risk of an event occurring. For example, a hazard ratio of 2.0 for "high intensity waves" would mean these waves have twice the risk of terminating at any given moment compared to low intensity waves. Importantly, the Cox model does not require you to specify exactly how the baseline risk changes over time - it focuses purely on how covariates multiply that risk.

---

## The Formula

$$
h(t|\mathbf{x}) = h_0(t) \exp(\boldsymbol{\beta}'\mathbf{x})
$$

---

## Symbol-by-Symbol Breakdown

| Symbol | Meaning | Type | Domain |
|--------|---------|------|--------|
| $h(t\|\mathbf{x})$ | Hazard rate at time t given covariates x | Output | Positive real numbers |
| $h_0(t)$ | Baseline hazard function (risk when all covariates = 0) | Function | Positive real numbers |
| $\exp$ | Exponential function (e raised to a power) | Function | - |
| $\boldsymbol{\beta}$ | Vector of regression coefficients | Parameters | Real numbers |
| $\mathbf{x}$ | Vector of covariates (predictor variables) | Variables | Real numbers |
| $\boldsymbol{\beta}'\mathbf{x}$ | Linear combination: $\beta_1 x_1 + \beta_2 x_2 + ...$ | Scalar | Real numbers |

---

## Step-by-Step Interpretation

1. **Understand the hazard rate $h(t\|\mathbf{x})$:** This is the instantaneous risk of the event happening at time $t$, given that it has not happened yet. Think of it as "the probability of ending right now, given you have lasted this long." A higher hazard means a higher risk of termination.

2. **The baseline hazard $h_0(t)$:** This is the hazard when all covariates equal zero. It can vary over time in any way - the Cox model does not require us to specify its shape. This flexibility is a major advantage of the model.

3. **The exponential term $\exp(\boldsymbol{\beta}'\mathbf{x})$:** This is a multiplier that adjusts the baseline hazard based on covariate values. The exponential ensures the multiplier is always positive (you cannot have negative risk).

4. **Linear predictor $\boldsymbol{\beta}'\mathbf{x}$:** Each covariate contributes additively to the log-hazard. For example, if $\beta_1 = 0.3$ and $x_1 = 2$, that covariate contributes $0.6$ to the linear predictor.

5. **Hazard Ratios:** The quantity $\exp(\beta_j)$ is the hazard ratio for covariate $j$. It tells you how much the hazard multiplies for a one-unit increase in $x_j$, holding other covariates constant.

---

## Worked Example

**Setup:**
We are analyzing what affects how long immigration waves last. We have two covariates:
- Wave intensity (high = 1, low = 0)
- Origin region (European = 1, Other = 0)

Our fitted model gives:
- $\beta_{\text{intensity}} = 0.693$ (hazard ratio = $e^{0.693} = 2.0$)
- $\beta_{\text{european}} = -0.511$ (hazard ratio = $e^{-0.511} = 0.60$)

**Calculation:**
```
For a high-intensity European wave (x_intensity = 1, x_european = 1):

Step 1: Calculate linear predictor
  beta'x = 0.693(1) + (-0.511)(1) = 0.182

Step 2: Calculate hazard multiplier
  exp(0.182) = 1.20

Step 3: Interpret
  h(t|x) = h_0(t) x 1.20
  This wave has 1.2 times the baseline hazard

For a low-intensity non-European wave (x_intensity = 0, x_european = 0):

Step 1: Calculate linear predictor
  beta'x = 0.693(0) + (-0.511)(0) = 0

Step 2: Calculate hazard multiplier
  exp(0) = 1.00

Step 3: Interpret
  h(t|x) = h_0(t) x 1.00
  This is the baseline hazard itself (reference group)
```

**Interpretation:**
- High-intensity waves have 2 times the termination risk of low-intensity waves (HR = 2.0). They tend to end sooner.
- European-origin waves have 0.60 times the termination risk, or a 40% reduction (HR = 0.60). They tend to last longer.
- Combining both effects: a high-intensity European wave has 2.0 x 0.60 = 1.2 times the baseline risk.

---

## Key Assumptions

1. **Proportional Hazards:** The hazard ratio between any two subjects is constant over time. If wave A has twice the risk of wave B at year 1, it also has twice the risk at year 5, year 10, etc. This is the key assumption that gives the model its name.

2. **Non-informative censoring:** Censored observations (waves still ongoing when we stop observing) are not systematically different in their underlying hazard from uncensored ones.

3. **Correct functional form:** Covariates have a log-linear relationship with the hazard. For continuous variables, this means a one-unit increase always has the same multiplicative effect, regardless of starting value.

4. **Independence:** Individual observations (immigration waves) are independent of each other.

---

## Common Pitfalls

- **Ignoring proportional hazards violations:** If hazard ratios change over time, the model gives misleading "average" effects. Always test this assumption using Schoenfeld residuals. Remedy: include time-covariate interactions or stratify by the problematic variable.

- **Misinterpreting hazard ratios:** A hazard ratio of 2.0 does NOT mean "twice as many events" or "events happen at twice the rate in absolute terms." It means the instantaneous risk multiplier is 2 at every point in time.

- **Forgetting reference categories:** For categorical variables, one level is always the reference (HR = 1.0 by definition). Make sure you know which level is the reference when interpreting.

- **Overfitting with too many covariates:** Rule of thumb: you need roughly 10-15 events per covariate. With few events, coefficient estimates become unstable.

- **Confusing coefficients and hazard ratios:** The coefficient $\beta$ is on the log-hazard scale. The hazard ratio $\exp(\beta)$ is multiplicative. Report hazard ratios for interpretation.

---

## Related Tests

- **Schoenfeld Residual Test:** Tests whether the proportional hazards assumption holds. Non-significant p-values (> 0.05) suggest the assumption is reasonable.

- **Log-Rank Test:** A simpler test comparing survival curves between groups, equivalent to a Cox model with one binary covariate.

- **Concordance Index (C-index):** Measures the model's predictive accuracy - the probability that for two randomly chosen subjects, the one with higher predicted risk actually experienced the event first. C = 0.5 is random guessing; C = 1.0 is perfect.

- **Likelihood Ratio Test:** Tests whether a covariate (or set of covariates) significantly improves model fit.

---

## Python Implementation

```python
from lifelines import CoxPHFitter
import pandas as pd
import numpy as np

# Example data: immigration wave durations with covariates
df_model = pd.DataFrame({
    'duration': [2, 4, 4, 5, 7, 8, 10, 12, 15, 20],  # Years observed
    'event': [1, 1, 1, 1, 1, 0, 0, 1, 0, 0],  # 1=terminated, 0=censored
    'intensity': [1, 1, 0, 1, 0, 0, 1, 0, 0, 1],  # High=1, Low=0
    'european': [0, 0, 1, 0, 1, 1, 0, 1, 1, 0]  # European=1, Other=0
})

# Fit the Cox Proportional Hazards model
cph = CoxPHFitter()
cph.fit(df_model, duration_col='duration', event_col='event')

# View summary with hazard ratios
print(cph.summary)

# Extract specific results
hazard_ratios = cph.summary['exp(coef)']
coefficients = cph.summary['coef']
p_values = cph.summary['p']
concordance_index = cph.concordance_index_

print(f"\nConcordance Index: {concordance_index:.3f}")
print(f"\nHazard Ratios:")
for var in hazard_ratios.index:
    hr = hazard_ratios[var]
    pval = p_values[var]
    print(f"  {var}: HR = {hr:.2f} (p = {pval:.3f})")

# Check proportional hazards assumption
# If p-values > 0.05, assumption is satisfied
cph.check_assumptions(df_model, p_value_threshold=0.05, show_plots=True)

# Plot covariate effects
cph.plot()

# Predict survival curves for specific covariate patterns
# High-intensity, European wave
high_euro = pd.DataFrame({'intensity': [1], 'european': [1]})
cph.predict_survival_function(high_euro).plot(label='High intensity, European')

# Low-intensity, non-European wave
low_other = pd.DataFrame({'intensity': [0], 'european': [0]})
cph.predict_survival_function(low_other).plot(label='Low intensity, Other')

import matplotlib.pyplot as plt
plt.xlabel('Time (years)')
plt.ylabel('Survival Probability')
plt.title('Predicted Survival by Wave Characteristics')
plt.legend()
plt.show()
```

---

## References

- Cox, D. R. (1972). Regression models and life-tables. *Journal of the Royal Statistical Society: Series B*, 34(2), 187-220.

- Schoenfeld, D. (1982). Partial residuals for the proportional hazards regression model. *Biometrika*, 69(1), 239-241.

- Hosmer, D. W., Lemeshow, S., & May, S. (2008). *Applied Survival Analysis: Regression Modeling of Time-to-Event Data* (2nd ed.). Wiley.

- Davidson-Pilon, C. (2019). *lifelines: survival analysis in Python*. Journal of Open Source Software, 4(40), 1317.

- Therneau, T. M., & Grambsch, P. M. (2000). *Modeling Survival Data: Extending the Cox Model*. Springer.
