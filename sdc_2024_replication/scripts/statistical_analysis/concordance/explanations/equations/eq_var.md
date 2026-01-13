# Equation Explanation: Vector Autoregression (VAR)

**Number in Paper:** Eq. 7
**Category:** Multivariate Time Series
**Paper Section:** 2.3.4 Vector Autoregression

---

## What This Equation Does

A Vector Autoregression (VAR) models multiple time series variables together, capturing how each variable depends not only on its own past values but also on the past values of the other variables. Unlike single-variable models (like ARIMA), VAR explicitly models the dynamic interactions between variables.

In the migration context, VAR allows us to model North Dakota's international migration alongside U.S. national migration, asking questions like: Does a surge in national migration predict future increases in North Dakota? Does North Dakota's migration influence the national pattern? By modeling both series jointly, VAR can test for "Granger causality" - whether knowing the history of one variable helps predict the other beyond what the other's own history provides.

---

## The Formula

$$
\mathbf{y}_t = \mathbf{c} + \sum_{i=1}^{p} \mathbf{A}_i \mathbf{y}_{t-i} + \boldsymbol{\varepsilon}_t
$$

---

## Symbol-by-Symbol Breakdown

| Symbol | Meaning | Type | Domain |
|--------|---------|------|--------|
| $\mathbf{y}_t$ | Vector of endogenous variables at time $t$, e.g., $[y_t^{ND}, y_t^{US}]'$ | Vector | Real numbers (typically 2 or more variables) |
| $\mathbf{c}$ | Vector of constants (intercepts), one for each variable | Vector | Real numbers |
| $p$ | Lag order (how many past periods are included) | Integer | Positive integers (1, 2, 3, ...) |
| $\mathbf{A}_i$ | Coefficient matrix for lag $i$ | Matrix | Real numbers (dimension: n x n) |
| $\mathbf{y}_{t-i}$ | Vector of variables at $i$ periods ago | Vector | Real numbers |
| $\boldsymbol{\varepsilon}_t$ | Vector of error terms (white noise process) | Vector | Real numbers (assumed mean zero, uncorrelated over time) |

---

## Step-by-Step Interpretation

For a two-variable VAR (North Dakota migration $y^{ND}$ and U.S. migration $y^{US}$), the equation expands to:

$$
\begin{bmatrix} y_t^{ND} \\ y_t^{US} \end{bmatrix} = \begin{bmatrix} c_1 \\ c_2 \end{bmatrix} + \sum_{i=1}^{p} \begin{bmatrix} a_{11}^{(i)} & a_{12}^{(i)} \\ a_{21}^{(i)} & a_{22}^{(i)} \end{bmatrix} \begin{bmatrix} y_{t-i}^{ND} \\ y_{t-i}^{US} \end{bmatrix} + \begin{bmatrix} \varepsilon_t^{ND} \\ \varepsilon_t^{US} \end{bmatrix}
$$

**Row by row:**

1. **North Dakota equation:**
   $$y_t^{ND} = c_1 + \sum_{i=1}^{p} (a_{11}^{(i)} y_{t-i}^{ND} + a_{12}^{(i)} y_{t-i}^{US}) + \varepsilon_t^{ND}$$

   - $c_1$: The baseline level when all lagged values are zero
   - $a_{11}^{(i)}$: How ND migration $i$ periods ago affects current ND migration
   - $a_{12}^{(i)}$: How U.S. migration $i$ periods ago affects current ND migration (the cross-effect)

2. **U.S. equation:**
   $$y_t^{US} = c_2 + \sum_{i=1}^{p} (a_{21}^{(i)} y_{t-i}^{ND} + a_{22}^{(i)} y_{t-i}^{US}) + \varepsilon_t^{US}$$

   - $a_{21}^{(i)}$: How ND migration affects national migration (likely small given ND's size)
   - $a_{22}^{(i)}$: How past U.S. migration affects current U.S. migration

3. **Lag order ($p$):** Selected via AIC or BIC. With $p=2$, each variable depends on the two most recent periods of both variables.

4. **Error terms ($\boldsymbol{\varepsilon}_t$):** Random shocks that are unpredictable given past information. They may be correlated contemporaneously (a national shock affects both ND and US simultaneously) but are uncorrelated over time.

---

## Worked Example

**Setup:**
Suppose we have a VAR(1) model (one lag) for ND and US migration (in thousands):

$$
\begin{bmatrix} y_t^{ND} \\ y_t^{US} \end{bmatrix} = \begin{bmatrix} 0.5 \\ 100 \end{bmatrix} + \begin{bmatrix} 0.4 & 0.002 \\ 5.0 & 0.6 \end{bmatrix} \begin{bmatrix} y_{t-1}^{ND} \\ y_{t-1}^{US} \end{bmatrix} + \begin{bmatrix} \varepsilon_t^{ND} \\ \varepsilon_t^{US} \end{bmatrix}
$$

Last year's values: $y_{t-1}^{ND} = 2.5$ thousand, $y_{t-1}^{US} = 1,000$ thousand.

**Calculation:**
```
ND equation:
  y_t^ND = 0.5 + 0.4 * 2.5 + 0.002 * 1000
         = 0.5 + 1.0 + 2.0
         = 3.5 (thousand migrants)

US equation:
  y_t^US = 100 + 5.0 * 2.5 + 0.6 * 1000
         = 100 + 12.5 + 600
         = 712.5 (thousand migrants)
```

**Interpretation:**
- ND migration is predicted to be 3,500. The 0.002 coefficient on US migration means that for every additional 1,000 U.S. migrants last year, ND gains about 2 more migrants. This captures ND's share of national inflows.
- The 5.0 coefficient on ND in the US equation seems large but remember ND is measured in thousands while US is in thousands too - in context, this suggests ND contributes modestly to the national total.

---

## Key Assumptions

1. **Stationarity:** All variables should be stationary (no unit roots). If variables have unit roots, you should either difference them or use a Vector Error Correction Model (VECM) if they are cointegrated.

2. **Correct lag order:** The model includes enough lags to capture the dynamics but not so many that it overfits. Use AIC or BIC for selection.

3. **Structural stability:** The coefficient matrices $\mathbf{A}_i$ are assumed constant over time. Structural breaks would violate this assumption.

4. **No contemporaneous relationships in estimation:** Standard VAR estimation treats all variables as endogenous. If you need to model contemporaneous causation, you need a Structural VAR (SVAR).

---

## Common Pitfalls

- **Non-stationary variables:** Estimating VAR on non-stationary data leads to spurious results. Always test for unit roots first (using ADF) and difference if necessary.

- **Too many lags:** With limited data, each additional lag uses up degrees of freedom. A VAR(p) with $n$ variables has $n + p \times n^2$ parameters. With 2 variables and 3 lags, that is 20 parameters - problematic for short time series.

- **Ignoring cointegration:** If variables are I(1) (non-stationary in levels but stationary in first differences) and cointegrated, you should use VECM rather than VAR in differences.

- **Misinterpreting coefficients:** VAR coefficients show partial correlations, not causal effects. Granger causality tests whether lags of X improve predictions of Y, not whether X causes Y in the philosophical sense.

---

## Related Tests

- **Granger Causality Test:** Tests whether lagged values of one variable help predict another. In the VAR context, this tests whether certain coefficients in $\mathbf{A}_i$ are jointly zero.

- **Johansen Cointegration Test:** If variables are I(1), tests whether they share a long-run equilibrium relationship.

- **Lag Selection Criteria:** AIC, BIC, and HQIC help determine the optimal $p$.

- **Impulse Response Functions (IRF):** Show how a shock to one variable propagates through the system over time.

---

## Python Implementation

```python
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load your data
# data = pd.DataFrame({
#     'nd_migration': [...],
#     'us_migration': [...]
# }, index=pd.date_range('2000', periods=24, freq='Y'))

# Step 1: Check stationarity (both series should be stationary)
def check_stationarity(series, name):
    result = adfuller(series, autolag='AIC')
    print(f"{name}:")
    print(f"  ADF Statistic: {result[0]:.4f}")
    print(f"  p-value: {result[1]:.4f}")
    print(f"  Stationary: {'Yes' if result[1] < 0.05 else 'No'}")
    return result[1] < 0.05

is_nd_stationary = check_stationarity(data['nd_migration'], 'ND Migration')
is_us_stationary = check_stationarity(data['us_migration'], 'US Migration')

# If non-stationary, difference the data
if not (is_nd_stationary and is_us_stationary):
    data_diff = data.diff().dropna()
    print("\nUsing first differences for VAR estimation")
else:
    data_diff = data

# Step 2: Fit VAR model
model = VAR(data_diff)

# Select optimal lag order
lag_order_results = model.select_order(maxlags=4)
print("\nLag Order Selection:")
print(lag_order_results.summary())

# Fit with AIC-optimal lag
optimal_lag = lag_order_results.aic
fitted = model.fit(maxlags=optimal_lag, ic='aic')

print("\nVAR Results Summary:")
print(fitted.summary())

# Step 3: Granger Causality Tests
print("\n" + "="*50)
print("Granger Causality Tests")
print("="*50)

# Does US migration Granger-cause ND migration?
gc_us_to_nd = fitted.test_causality('nd_migration', 'us_migration', kind='f')
print(f"\nUS -> ND: F-stat = {gc_us_to_nd.test_statistic:.4f}, p-value = {gc_us_to_nd.pvalue:.4f}")
if gc_us_to_nd.pvalue < 0.05:
    print("  Conclusion: US migration Granger-causes ND migration")
else:
    print("  Conclusion: US migration does NOT Granger-cause ND migration")

# Does ND migration Granger-cause US migration?
gc_nd_to_us = fitted.test_causality('us_migration', 'nd_migration', kind='f')
print(f"\nND -> US: F-stat = {gc_nd_to_us.test_statistic:.4f}, p-value = {gc_nd_to_us.pvalue:.4f}")

# Step 4: Impulse Response Functions
irf = fitted.irf(periods=10)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot IRFs
irf.plot(orth=False, impulse='nd_migration', response='nd_migration', ax=axes[0,0])
irf.plot(orth=False, impulse='us_migration', response='nd_migration', ax=axes[0,1])
irf.plot(orth=False, impulse='nd_migration', response='us_migration', ax=axes[1,0])
irf.plot(orth=False, impulse='us_migration', response='us_migration', ax=axes[1,1])

plt.tight_layout()
plt.savefig('var_impulse_response.png', dpi=150)
plt.show()

# Step 5: Forecast
n_forecast = 5
forecast = fitted.forecast(data_diff.values[-optimal_lag:], steps=n_forecast)
forecast_df = pd.DataFrame(
    forecast,
    columns=['nd_migration', 'us_migration'],
    index=pd.date_range(data.index[-1], periods=n_forecast+1, freq='Y')[1:]
)
print(f"\n{n_forecast}-Year Forecast:")
print(forecast_df)
```

---

## References

- Sims, C. A. (1980). Macroeconomics and reality. *Econometrica*, 48(1), 1-48.

- Lutkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.

- Hamilton, J. D. (1994). *Time Series Analysis*. Princeton University Press. (Chapters 10-11)

- Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. *Econometrica*, 37(3), 424-438.
