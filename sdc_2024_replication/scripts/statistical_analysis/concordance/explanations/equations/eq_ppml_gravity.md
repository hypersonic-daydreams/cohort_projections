# Equation Explanation: PPML Gravity Model

**Number in Paper:** Eq. 9
**Category:** Cross-Sectional Allocation Models
**Paper Section:** 2.5 Cross-Sectional Allocation Models

---

## What This Equation Does

The PPML (Poisson Pseudo-Maximum Likelihood) Gravity Model predicts how many legal permanent resident (LPR) admissions will flow from a specific origin country to a specific destination state. The name "gravity model" comes from physics: just as gravitational attraction depends on mass and distance, migration flows depend on the "mass" (size of populations) and connections between places.

The key insight of this model is that migration follows predictable patterns. If there's already a large community from Country X living in State Y (the "diaspora" or "stock"), new immigrants from Country X are more likely to settle in State Y. This is called chain migration or network effects. The model quantifies this relationship while controlling for other factors like the overall size of the origin-country population in the US and the overall attractiveness of the destination state. The PPML estimator is specifically designed for count data (like number of migrants) and handles zeros well, making it ideal for migration analysis where many origin-destination pairs may have zero or very few migrants.

---

## The Formula

$$
E[M_{od}] = \exp(\beta_0 + \beta_1 \ln \text{Stock}_{od} + \beta_2 \ln \text{OriginTotal}_o + \beta_3 \ln \text{DestTotal}_d + \boldsymbol{\gamma}'\mathbf{Z}_{od})
$$

---

## Symbol-by-Symbol Breakdown

| Symbol | Meaning | Type | Domain |
|--------|---------|------|--------|
| $E[M_{od}]$ | Expected (predicted) LPR admissions from origin $o$ to destination $d$ | Outcome variable | Non-negative count |
| $\exp(\cdot)$ | Exponential function--ensures predictions are always positive | Function | Positive real numbers |
| $\beta_0$ | Intercept (baseline log-count when all predictors equal zero) | Parameter | Real number |
| $\beta_1$ | Diaspora elasticity--the coefficient on existing immigrant stock | Key parameter | Real number (typically positive) |
| $\ln \text{Stock}_{od}$ | Natural log of existing foreign-born from origin $o$ living in destination $d$ | Predictor | Log of positive count |
| $\beta_2$ | Coefficient on origin country's total US presence | Parameter | Real number |
| $\ln \text{OriginTotal}_o$ | Log of total foreign-born from origin $o$ in the entire US | Predictor | Log of positive count |
| $\beta_3$ | Coefficient on destination state attractiveness | Parameter | Real number |
| $\ln \text{DestTotal}_d$ | Log of total foreign-born in destination state $d$ (from any origin) | Predictor | Log of positive count |
| $\boldsymbol{\gamma}$ | Vector of coefficients on additional controls | Parameters | Real numbers |
| $\mathbf{Z}_{od}$ | Additional bilateral control variables (if any) | Predictors | Varies |
| $o$ | Origin country index | Subscript | Categorical (countries) |
| $d$ | Destination state index | Subscript | Categorical (US states) |

---

## Step-by-Step Interpretation

1. **The exponential wrapper ($\exp(\cdot)$):** The entire right-hand side is wrapped in an exponential function. This serves two purposes: (a) it guarantees that predicted migration counts are always positive (you can't have negative migrants), and (b) it means the effects are multiplicative rather than additive. A one-unit increase in a predictor multiplies the expected count rather than adding to it.

2. **The intercept ($\beta_0$):** This is the baseline log-count. By itself, it's not very interpretable because it represents the expected flow when all log-predictors equal zero (which would mean all the predictor values equal 1, a very small flow).

3. **The diaspora effect ($\beta_1 \ln \text{Stock}_{od}$):** This is the most important term. It captures network effects: immigrants tend to go where people from their home country already live. The coefficient $\beta_1$ is an elasticity. If $\beta_1 = 0.5$, a 10% increase in the existing stock from origin $o$ in destination $d$ is associated with a roughly 5% increase in new admissions to that destination from that origin.

4. **The origin-country size effect ($\beta_2 \ln \text{OriginTotal}_o$):** Controls for the overall "supply" of immigrants from origin country $o$. Countries with more total immigrants in the US will naturally send more to any given state.

5. **The destination attractiveness effect ($\beta_3 \ln \text{DestTotal}_d$):** Controls for how attractive destination $d$ is overall. States with large immigrant populations (like California or Texas) attract more immigrants from everywhere.

6. **Additional controls ($\boldsymbol{\gamma}'\mathbf{Z}_{od}$):** In traditional gravity models, this would include distance. However, the paper notes that distance is omitted because within-US variation is minimal--once you're in the US, traveling from one state to another is relatively easy compared to international migration.

---

## Worked Example

**Setup:**
Consider predicting LPR admissions from India to North Dakota in FY2023. Suppose:

- $\text{Stock}_{\text{India, ND}} = 5,000$ (5,000 Indian-born people already live in ND)
- $\text{OriginTotal}_{\text{India}} = 2,800,000$ (2.8 million Indian-born in the entire US)
- $\text{DestTotal}_{\text{ND}} = 25,000$ (25,000 total foreign-born in ND)
- Estimated coefficients: $\beta_0 = -2.5$, $\beta_1 = 0.7$, $\beta_2 = 0.3$, $\beta_3 = 0.4$

**Calculation:**
```
Step 1: Calculate log values
  ln(5,000) = 8.52
  ln(2,800,000) = 14.85
  ln(25,000) = 10.13

Step 2: Compute the linear predictor
  Linear predictor = -2.5 + (0.7 x 8.52) + (0.3 x 14.85) + (0.4 x 10.13)
                   = -2.5 + 5.96 + 4.46 + 4.05
                   = 11.97

Step 3: Apply exponential transformation
  E[M] = exp(11.97) = 158,000 (approximately)

Wait--that seems too high! Let me recalculate with more realistic coefficients.
Suppose instead: beta_0 = -8, beta_1 = 0.6, beta_2 = 0.15, beta_3 = 0.2

Step 2 (revised):
  Linear predictor = -8 + (0.6 x 8.52) + (0.15 x 14.85) + (0.2 x 10.13)
                   = -8 + 5.11 + 2.23 + 2.03
                   = 1.37

Step 3 (revised):
  E[M] = exp(1.37) = 3.9

Result: Expected LPR admissions from India to ND is approximately 4 people.
```

**Interpretation:**
The model predicts about 4 Indian nationals will receive LPR status in North Dakota. The strong diaspora effect ($\beta_1 = 0.6$) means that ND's existing Indian community (5,000 people) is the main driver, but it's moderated by the fact that ND is a relatively small destination overall (low $\text{DestTotal}$). If ND's Indian diaspora were twice as large (10,000), the predicted flow would increase by a factor of $2^{0.6} \approx 1.52$, so about 6 admissions.

---

## Key Assumptions

1. **Conditional mean specification:** The expected value follows the exponential form. The model doesn't assume Poisson-distributed errors, just that the mean follows this functional form (hence "pseudo" maximum likelihood).

2. **No unobserved bilateral factors:** After controlling for Stock, OriginTotal, DestTotal, and any Z variables, remaining variation is random noise. If there are unobserved factors that make certain origin-destination pairs systematically different (e.g., historical ties, specific visa programs), estimates may be biased.

3. **Log-linearity in effects:** The multiplicative structure implies constant elasticities. A 10% increase in diaspora stock always leads to the same percentage increase in expected flow, regardless of the level.

4. **Stock is predetermined:** The existing diaspora (Stock) is measured at the start of the period and isn't influenced by current-year admissions. This is typically satisfied when using lagged or baseline stock measures.

5. **No perfect multicollinearity:** The predictors must not be perfectly correlated with each other.

---

## Common Pitfalls

- **Confusing elasticities with marginal effects:** $\beta_1$ is an elasticity (percentage change in $M$ per percentage change in Stock), not a marginal effect (change in $M$ per unit change in Stock). For marginal effects, you need to multiply by the predicted value.

- **Ignoring zeros in the stock variable:** If $\text{Stock}_{od} = 0$, then $\ln(0)$ is undefined. Common solutions: add 1 before logging ($\ln(\text{Stock} + 1)$), or use the inverse hyperbolic sine transformation.

- **Causal interpretation without instruments:** The diaspora effect ($\beta_1$) is a predictive association, not necessarily causal. Unobserved factors might drive both existing stock and new flows. For causal claims, you'd need instrumental variables or other identification strategies.

- **Ignoring heteroskedasticity:** PPML is robust to certain forms of heteroskedasticity, but you should still use robust standard errors.

- **Omitting relevant controls:** Traditional gravity models include distance, language similarity, colonial ties, etc. Omitting relevant variables can bias the diaspora elasticity.

---

## Related Tests

- **Ramsey RESET test:** Tests for model misspecification by checking whether polynomial terms of the fitted values improve the model.

- **Comparison with OLS on logs:** A classic result (Santos Silva & Tenreyro, 2006) shows OLS on $\ln(M_{od})$ is biased when there's heteroskedasticity or zeros. PPML is preferred.

- **Overdispersion test:** If the data shows overdispersion (variance exceeds mean), negative binomial regression might be preferred, though PPML remains consistent.

---

## Python Implementation

```python
import statsmodels.api as sm
from statsmodels.genmod.families import Poisson
import numpy as np
import pandas as pd

# Assume df has columns: M_od (count), Stock_od, OriginTotal_o, DestTotal_d
# Add small constant to handle zeros in stock
df['ln_stock'] = np.log(df['Stock_od'] + 1)
df['ln_origin_total'] = np.log(df['OriginTotal_o'])
df['ln_dest_total'] = np.log(df['DestTotal_d'])

# Prepare predictors
X = df[['ln_stock', 'ln_origin_total', 'ln_dest_total']]
X = sm.add_constant(X)  # Add intercept

# Fit PPML (Poisson regression with robust standard errors)
model = sm.GLM(
    endog=df['M_od'],
    exog=X,
    family=Poisson()
)
result = model.fit(cov_type='HC1')  # Robust standard errors

# Display results
print(result.summary())

# Extract key coefficient (diaspora elasticity)
diaspora_elasticity = result.params['ln_stock']
print(f"\nDiaspora elasticity (beta_1): {diaspora_elasticity:.4f}")
print(f"Interpretation: 10% increase in existing stock associated with")
print(f"                {10 * diaspora_elasticity:.2f}% increase in expected admissions")

# Calculate predicted flows
df['predicted_M'] = result.predict(X)

# Example prediction for specific origin-destination pair
example_row = pd.DataFrame({
    'const': [1],
    'ln_stock': [np.log(5000 + 1)],
    'ln_origin_total': [np.log(2800000)],
    'ln_dest_total': [np.log(25000)]
})
predicted_flow = result.predict(example_row)[0]
print(f"\nPredicted flow (India -> ND example): {predicted_flow:.1f}")
```

---

## References

- Santos Silva, J.M.C. & Tenreyro, S. (2006). "The Log of Gravity." *Review of Economics and Statistics*, 88(4), 641-658. [Foundational paper showing PPML is preferred to OLS-on-logs]

- Anderson, J.E. & van Wincoop, E. (2003). "Gravity with Gravitas: A Solution to the Border Puzzle." *American Economic Review*, 93(1), 170-192. [Theoretical foundations of gravity models]

- Beine, M., Docquier, F., & Ozden, C. (2011). "Diasporas." *Journal of Development Economics*, 95(1), 30-41. [Application to migration and diaspora effects]

- Head, K. & Mayer, T. (2014). "Gravity Equations: Workhorse, Toolkit, and Cookbook." *Handbook of International Economics*, Vol. 4. Elsevier. [Comprehensive guide to gravity model estimation]

- Wooldridge, J.M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press. Chapter 18 (Count data models).
