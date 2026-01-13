# Equation Explanation: Akaike Information Criterion

**Number in Paper:** Eq. 5
**Category:** Model Selection
**Paper Section:** 2.3.2 ARIMA Model Selection

---

## What This Equation Does

The Akaike Information Criterion (AIC) helps choose between competing statistical models. When you have multiple candidate models (for example, ARIMA models with different lag orders), AIC provides a principled way to decide which one is "best" by balancing two competing goals: how well the model fits the data versus how complex the model is.

A model with more parameters will always fit the training data at least as well as a simpler model. But overly complex models "overfit" - they capture noise rather than true patterns, and perform poorly on new data. AIC penalizes complexity, so the model with the lowest AIC value is considered optimal. In the context of migration forecasting, AIC helps determine how many autoregressive and moving average terms to include in an ARIMA model.

---

## The Formula

$$
\text{AIC} = -2 \ln(\hat{L}) + 2k
$$

---

## Symbol-by-Symbol Breakdown

| Symbol | Meaning | Type | Domain |
|--------|---------|------|--------|
| AIC | Akaike Information Criterion value | Computed statistic | Real numbers |
| $\hat{L}$ | Maximized likelihood of the model | Computed value | Positive real numbers (0,1] for proper likelihoods |
| $\ln(\hat{L})$ | Natural logarithm of the likelihood | Computed value | Real numbers (typically negative) |
| $k$ | Number of estimated parameters in the model | Integer | Positive integers (1, 2, 3, ...) |
| $2$ | Penalty weight per parameter | Constant | Fixed at 2 |

---

## Step-by-Step Interpretation

1. **Likelihood term ($\hat{L}$):** The likelihood measures how probable the observed data would be if the model were true. Higher likelihood means better fit. We use the "maximized" likelihood - the likelihood evaluated at the best possible parameter values.

2. **Log-likelihood ($\ln(\hat{L})$):** We take the logarithm because likelihoods are often very small numbers. Working with logs is computationally convenient and converts products into sums.

3. **Negative sign ($-2\ln(\hat{L})$):** Since log-likelihoods are typically negative (likelihoods are less than 1), multiplying by $-2$ makes this term positive. Better-fitting models have larger likelihoods, hence larger log-likelihoods, hence smaller values of $-2\ln(\hat{L})$.

4. **Penalty term ($2k$):** This adds a cost for each parameter in the model. An ARIMA(2,1,2) model has more parameters than ARIMA(1,1,1), so it faces a larger penalty. This discourages overfitting.

5. **The balance:** The model with the lowest AIC achieves the best trade-off between fitting the data well (low $-2\ln(\hat{L})$) and staying simple (low $2k$).

---

## Worked Example

**Setup:**
Suppose we are comparing three ARIMA models for North Dakota migration data:

| Model | Parameters ($k$) | Log-Likelihood ($\ln(\hat{L})$) |
|-------|------------------|--------------------------------|
| ARIMA(1,1,0) | 2 | -45.3 |
| ARIMA(1,1,1) | 3 | -43.1 |
| ARIMA(2,1,2) | 5 | -41.8 |

**Calculation:**
```
ARIMA(1,1,0):
  AIC = -2 * (-45.3) + 2 * 2
      = 90.6 + 4
      = 94.6

ARIMA(1,1,1):
  AIC = -2 * (-43.1) + 2 * 3
      = 86.2 + 6
      = 92.2

ARIMA(2,1,2):
  AIC = -2 * (-41.8) + 2 * 5
      = 83.6 + 10
      = 93.6
```

**Interpretation:**
The ARIMA(1,1,1) model has the lowest AIC (92.2), so it is preferred. Although ARIMA(2,1,2) has a better likelihood (fits the data more closely), the improvement does not justify the extra parameters. The ARIMA(1,1,1) model achieves the best balance between fit and parsimony.

---

## Key Assumptions

1. **Models are nested or comparable:** AIC works best when comparing models estimated on the same data. All candidate models should use identical observations.

2. **Sample size is adequate:** AIC is asymptotically valid (works well with large samples). For small samples, consider using AICc (corrected AIC), which adds an additional penalty: $\text{AICc} = \text{AIC} + \frac{2k(k+1)}{n-k-1}$.

3. **Maximum likelihood estimation:** AIC assumes parameters were estimated by maximum likelihood. If you used a different estimation method, AIC may not be appropriate.

4. **True model is in the candidate set:** AIC selects the best approximating model from those considered, not necessarily the "true" model.

---

## Common Pitfalls

- **Comparing AIC across different datasets:** AIC values are only comparable when computed on the exact same observations. If one model drops observations (e.g., due to differencing or lag requirements), you cannot directly compare AICs.

- **Ignoring the scale:** The absolute value of AIC is meaningless; only differences matter. An AIC of 500 is not inherently "bad" - what matters is whether it is lower or higher than alternatives.

- **Small sample bias:** With small samples (roughly $n/k < 40$), standard AIC tends to select overly complex models. Use AICc instead.

- **Over-reliance on a single criterion:** AIC is one tool among many. Consider also BIC (Bayesian Information Criterion), which penalizes complexity more heavily, and out-of-sample validation.

---

## Related Tests

- **BIC (Bayesian Information Criterion):** Similar to AIC but with penalty $k \ln(n)$ instead of $2k$. BIC penalizes complexity more heavily and tends to select simpler models, especially with large samples.

- **Ljung-Box Test:** Used after model selection to verify that residuals are white noise. Even if a model has the best AIC, it should still pass residual diagnostics.

- **Cross-validation:** An alternative approach to model selection that directly estimates out-of-sample prediction error.

---

## Python Implementation

```python
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd

# Load your migration series
# series = pd.Series([...])  # Your data here

# Define candidate model orders
orders = [
    (1, 1, 0),
    (1, 1, 1),
    (2, 1, 1),
    (2, 1, 2),
    (0, 1, 1),
]

# Fit each model and collect AIC values
results = []
for order in orders:
    try:
        model = ARIMA(series, order=order)
        fitted = model.fit()
        results.append({
            'order': order,
            'aic': fitted.aic,
            'bic': fitted.bic,
            'log_likelihood': fitted.llf,
            'n_params': len(fitted.params)
        })
    except Exception as e:
        print(f"Order {order} failed: {e}")

# Create comparison table
comparison = pd.DataFrame(results)
comparison = comparison.sort_values('aic')

print("Model Comparison by AIC:")
print(comparison.to_string(index=False))

# Best model
best = comparison.iloc[0]
print(f"\nBest Model: ARIMA{best['order']}")
print(f"AIC: {best['aic']:.2f}")
print(f"BIC: {best['bic']:.2f}")
print(f"Log-Likelihood: {best['log_likelihood']:.2f}")

# For small samples, use AICc
n = len(series)
comparison['aicc'] = comparison['aic'] + (2 * comparison['n_params'] * (comparison['n_params'] + 1)) / (n - comparison['n_params'] - 1)
print("\nWith AICc correction:")
print(comparison.sort_values('aicc')[['order', 'aic', 'aicc']].to_string(index=False))
```

---

## References

- Akaike, H. (1974). A new look at the statistical model identification. *IEEE Transactions on Automatic Control*, 19(6), 716-723.

- Burnham, K. P., & Anderson, D. R. (2002). *Model Selection and Multimodel Inference: A Practical Information-Theoretic Approach* (2nd ed.). Springer.

- Hurvich, C. M., & Tsai, C. L. (1989). Regression and time series model selection in small samples. *Biometrika*, 76(2), 297-307. (AICc derivation)
