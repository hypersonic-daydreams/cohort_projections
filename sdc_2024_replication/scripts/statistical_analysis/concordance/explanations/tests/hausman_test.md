# Statistical Test Explanation: Hausman Test

## Test: Hausman Test

**Full Name:** Hausman Specification Test
**Category:** Panel Model Selection
**Paper Section:** 2.4 Panel Data Methods

---

## What This Test Does

The Hausman test helps researchers choose between two panel data estimation approaches: fixed effects (FE) and random effects (RE). This choice is crucial because it determines how we account for unobserved heterogeneity across panel units. In demographic research on international migration to North Dakota, each sending country may have unique characteristics (cultural ties, historical migration networks, geographic factors) that affect migration patterns but are difficult to measure directly. The Hausman test determines whether these unobserved country-specific factors are correlated with the explanatory variables in our model.

The test works by comparing the coefficient estimates from both FE and RE models. If the unobserved heterogeneity is uncorrelated with the regressors, both estimators are consistent, but RE is more efficient (has smaller standard errors). If the heterogeneity is correlated with the regressors, only FE is consistent, while RE produces biased estimates. The Hausman test statistic measures whether the difference between the two sets of estimates is statistically significant. A significant difference indicates that the correlation exists, and FE should be used. If the difference is not significant, RE is preferred because it is more efficient and allows estimation of time-invariant covariates.

---

## The Hypotheses

| Hypothesis | Statement |
|------------|-----------|
| **Null (H0):** | Random effects estimator is consistent (no correlation between individual effects and regressors) |
| **Alternative (H1):** | Fixed effects estimator is required (individual effects are correlated with regressors) |

**Important:** The null hypothesis favors random effects. Rejecting H0 means we should use fixed effects. Failing to reject H0 suggests random effects is appropriate and more efficient.

---

## Test Statistic

The Hausman test statistic is based on the difference between the fixed effects and random effects coefficient estimates:

$$
H = (\hat{\beta}_{FE} - \hat{\beta}_{RE})' \left[ \text{Var}(\hat{\beta}_{FE}) - \text{Var}(\hat{\beta}_{RE}) \right]^{-1} (\hat{\beta}_{FE} - \hat{\beta}_{RE})
$$

Where:
- $\hat{\beta}_{FE}$ is the vector of fixed effects coefficient estimates
- $\hat{\beta}_{RE}$ is the vector of random effects coefficient estimates
- $\text{Var}(\hat{\beta}_{FE})$ is the variance-covariance matrix of FE estimates
- $\text{Var}(\hat{\beta}_{RE})$ is the variance-covariance matrix of RE estimates

The test only compares coefficients on time-varying regressors, as FE cannot estimate coefficients on time-invariant variables.

**Distribution under H0:** Chi-squared with k degrees of freedom, where k is the number of coefficients being compared (time-varying regressors only).

---

## Decision Rule

The Hausman test is a right-tailed test. Reject the null hypothesis if the test statistic exceeds the critical value.

| Significance Level | Critical Value (k=1) | Critical Value (k=3) | Critical Value (k=5) | Decision |
|-------------------|---------------------|---------------------|---------------------|----------|
| alpha = 0.01 | 6.63 | 11.34 | 15.09 | Reject H0 if H > critical |
| alpha = 0.05 | 3.84 | 7.81 | 11.07 | Reject H0 if H > critical |
| alpha = 0.10 | 2.71 | 6.25 | 9.24 | Reject H0 if H > critical |

**Note:** Critical values depend on k (degrees of freedom = number of time-varying regressors being compared). Consult chi-squared tables for the appropriate df.

**P-value approach:** Reject H0 if p-value < alpha

---

## When to Use This Test

**Use when:**
- You have panel data (repeated observations on the same units over time)
- You need to choose between fixed effects and random effects estimation
- You have time-varying regressors whose coefficients can be compared across models
- Your sample includes a meaningful number of cross-sectional units and time periods

**Do not use when:**
- You only have cross-sectional data (single time period)
- All regressors are time-invariant (FE cannot estimate these)
- The number of time periods is very small (T less than 3)
- You have a very small number of cross-sectional units (N less than 10)
- Model specification is incorrect (both estimators would be inconsistent)
- Heteroskedasticity or serial correlation is severe (consider robust versions)

---

## Key Assumptions

1. **Correct model specification:** Both models must be correctly specified. If the model is misspecified, both estimators may be inconsistent, making the Hausman test uninformative.

2. **Consistency of at least one estimator:** The test assumes that the fixed effects estimator is always consistent under the alternative. If strict exogeneity fails, even FE may be biased.

3. **Homoskedasticity:** The standard Hausman test assumes homoskedastic errors. With heteroskedasticity, use a robust version of the test or cluster-robust standard errors.

4. **No serial correlation:** The test assumes errors are not serially correlated within panels. If present, consider using clustered standard errors and a robust Hausman test.

5. **Positive definite difference matrix:** The variance difference matrix [Var(FE) - Var(RE)] should be positive semi-definite. Occasionally, numerical issues can cause negative test statistics; this may indicate specification problems.

6. **Large sample:** The chi-squared approximation is asymptotic and requires reasonably large N and T for reliable inference.

---

## Worked Example

**Data:** Panel of annual immigration to North Dakota from 45 countries over 15 years (2008-2022), with total sample of 675 country-year observations.

**Model:** We model log immigration as a function of: log GDP per capita in origin country, bilateral exchange rate, and existing diaspora size.

**Calculation:**
```
Step 1: Estimate Fixed Effects model
        ln(Immigration_ct) = alpha_c + beta_1*ln(GDP_ct) + beta_2*ExchangeRate_ct
                           + beta_3*ln(Diaspora_ct) + epsilon_ct

        Coefficients (FE):
        beta_1_FE = 0.85 (SE = 0.12)
        beta_2_FE = -0.23 (SE = 0.08)
        beta_3_FE = 0.42 (SE = 0.15)

Step 2: Estimate Random Effects model
        Same specification with random country effects

        Coefficients (RE):
        beta_1_RE = 0.72 (SE = 0.09)
        beta_2_RE = -0.18 (SE = 0.06)
        beta_3_RE = 0.38 (SE = 0.11)

Step 3: Calculate coefficient differences
        d = beta_FE - beta_RE = [0.13, -0.05, 0.04]'

Step 4: Compute variance difference matrix
        V_diff = Var(beta_FE) - Var(beta_RE)
        (3x3 matrix computed from model outputs)

Step 5: Calculate Hausman statistic
        H = d' * V_diff^{-1} * d = 12.47

Step 6: Compare to chi-squared critical value (df=3)
        Critical values: 1%: 11.34, 5%: 7.81, 10%: 6.25

P-value = 0.006
```

**Interpretation:**
The Hausman statistic of 12.47 exceeds the 1% critical value of 11.34, and the p-value of 0.006 is well below 0.05. We reject the null hypothesis that the random effects estimator is consistent. This indicates that unobserved country characteristics (such as historical ties to North Dakota, diaspora networks, or cultural factors) are correlated with the time-varying economic variables. The fixed effects model should be used to obtain unbiased estimates of how GDP, exchange rates, and diaspora size affect immigration to North Dakota.

---

## Interpreting Results

**If we reject H0 (p-value < 0.05 or H > critical value):**
Use the fixed effects estimator. The rejection indicates that unobserved individual characteristics (country-specific factors) are correlated with the regressors. Random effects would produce biased and inconsistent coefficient estimates. For immigration research, this often occurs because countries with stronger historical migration ties to a destination also tend to have specific economic characteristics that affect the time-varying covariates.

**If we fail to reject H0:**
The random effects estimator is preferred. Both estimators appear consistent, but random effects is more efficient (smaller standard errors) and allows estimation of coefficients on time-invariant variables. This does NOT prove that individual effects are uncorrelated with regressors; it only indicates we cannot detect such correlation with available data. For migration studies, RE may be appropriate when unobserved cultural or geographic factors are reasonably assumed to be independent of economic conditions.

---

## Common Pitfalls

- **Ignoring robust alternatives:** The standard Hausman test is sensitive to heteroskedasticity and serial correlation. Always check for these issues and consider robust Hausman tests that use cluster-robust variance estimators.

- **Negative test statistics:** Sometimes the variance difference matrix is not positive semi-definite, yielding a negative H statistic. This may indicate specification problems, very similar estimates, or numerical issues. Consider alternative tests like the Mundlak approach or auxiliary regression tests.

- **Testing with only time-invariant covariates:** The Hausman test cannot compare coefficients on time-invariant variables because fixed effects cannot estimate them. If all your key variables are time-invariant, the test is not applicable.

- **Over-interpreting failure to reject:** Failing to reject H0 does not prove RE is correct. With small samples or low power, the test may fail to detect correlation that exists. Consider the theoretical reasons for correlation alongside the statistical test.

- **Ignoring practical significance:** Even if statistically significant, if the FE and RE estimates are substantively similar, the choice may matter less for your research conclusions.

- **Forgetting about strict exogeneity:** Both FE and RE require strict exogeneity (no correlation between regressors and error terms at any time). Rejecting RE in favor of FE does not solve endogeneity problems from omitted time-varying confounders.

---

## Related Tests

| Test | Use When |
|------|----------|
| **Breusch-Pagan LM Test** | Tests whether to use pooled OLS or random effects; preliminary to Hausman |
| **Mundlak Test** | Alternative to Hausman; adds group means of time-varying variables to RE model |
| **Wooldridge's Robust Hausman Test** | Use when heteroskedasticity or serial correlation is present |
| **Sargan-Hansen Test** | Generalization of Hausman for GMM estimation with overidentifying restrictions |
| **F-test for Fixed Effects** | Tests whether fixed effects are jointly significant; complements Hausman |

---

## Python Implementation

```python
"""
Hausman Specification Test Implementation
Tests whether to use fixed effects or random effects in panel data models.
"""

import numpy as np
import pandas as pd
from scipy import stats
from linearmodels.panel import PanelOLS, RandomEffects
from typing import Dict, Any, Optional


def run_hausman_test(
    data: pd.DataFrame,
    dependent_var: str,
    independent_vars: list,
    entity_col: str,
    time_col: str,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform the Hausman specification test for panel data.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with entity and time identifiers
    dependent_var : str
        Name of the dependent variable column
    independent_vars : list
        List of independent variable column names (time-varying)
    entity_col : str
        Name of the entity (cross-sectional unit) identifier column
    time_col : str
        Name of the time period identifier column
    alpha : float, default 0.05
        Significance level for hypothesis test

    Returns
    -------
    dict
        Dictionary containing test results and interpretation

    Example
    -------
    >>> data = pd.DataFrame({
    ...     'country': ['A', 'A', 'A', 'B', 'B', 'B'],
    ...     'year': [2020, 2021, 2022, 2020, 2021, 2022],
    ...     'immigration': [100, 120, 110, 200, 220, 240],
    ...     'gdp': [50, 52, 54, 80, 82, 85]
    ... })
    >>> results = run_hausman_test(data, 'immigration', ['gdp'], 'country', 'year')
    >>> print(f"Use {'fixed' if results['use_fixed_effects'] else 'random'} effects")
    """

    # Set up panel data structure
    panel_data = data.set_index([entity_col, time_col])

    # Create formula for the model
    formula = f"{dependent_var} ~ 1 + " + " + ".join(independent_vars)

    # Estimate Fixed Effects model
    fe_model = PanelOLS.from_formula(
        formula + " + EntityEffects",
        data=panel_data
    )
    fe_results = fe_model.fit()

    # Estimate Random Effects model
    re_model = RandomEffects.from_formula(
        formula,
        data=panel_data
    )
    re_results = re_model.fit()

    # Extract coefficients (excluding constant for comparison)
    fe_params = fe_results.params[independent_vars]
    re_params = re_results.params[independent_vars]

    # Calculate coefficient differences
    beta_diff = fe_params - re_params

    # Extract variance-covariance matrices for the regressors
    fe_cov = fe_results.cov.loc[independent_vars, independent_vars]
    re_cov = re_results.cov.loc[independent_vars, independent_vars]

    # Calculate variance difference
    cov_diff = fe_cov - re_cov

    # Check if difference matrix is positive semi-definite
    eigenvalues = np.linalg.eigvalsh(cov_diff)
    is_positive_definite = all(eigenvalues > -1e-10)  # Allow small numerical error

    if not is_positive_definite:
        # Use absolute value of eigenvalues for numerical stability
        cov_diff_pd = cov_diff.copy()
        # Apply regularization if needed
        min_eig = min(eigenvalues)
        if min_eig < 0:
            cov_diff_pd = cov_diff + np.eye(len(independent_vars)) * (abs(min_eig) + 1e-6)
        cov_diff = cov_diff_pd

    # Calculate Hausman statistic
    try:
        cov_diff_inv = np.linalg.inv(cov_diff.values)
        hausman_stat = float(beta_diff.values @ cov_diff_inv @ beta_diff.values)
    except np.linalg.LinAlgError:
        # Singular matrix - use pseudo-inverse
        cov_diff_inv = np.linalg.pinv(cov_diff.values)
        hausman_stat = float(beta_diff.values @ cov_diff_inv @ beta_diff.values)

    # Degrees of freedom
    df = len(independent_vars)

    # P-value from chi-squared distribution
    p_value = 1 - stats.chi2.cdf(hausman_stat, df)

    # Critical values
    critical_values = {
        '1%': stats.chi2.ppf(0.99, df),
        '5%': stats.chi2.ppf(0.95, df),
        '10%': stats.chi2.ppf(0.90, df)
    }

    # Decision
    use_fixed_effects = p_value < alpha

    # Build interpretation
    if use_fixed_effects:
        interpretation = (
            f"Reject H0 at {alpha:.0%} level (p={p_value:.4f}). "
            f"Use Fixed Effects model. Unobserved {entity_col} characteristics "
            f"are correlated with the regressors."
        )
    else:
        interpretation = (
            f"Fail to reject H0 at {alpha:.0%} level (p={p_value:.4f}). "
            f"Random Effects model is preferred. It is more efficient and allows "
            f"estimation of time-invariant covariates."
        )

    return {
        'test_name': 'Hausman Specification Test',
        'hausman_statistic': hausman_stat,
        'p_value': p_value,
        'degrees_of_freedom': df,
        'critical_values': critical_values,
        'fe_coefficients': dict(fe_params),
        're_coefficients': dict(re_params),
        'coefficient_difference': dict(beta_diff),
        'use_fixed_effects': use_fixed_effects,
        'is_cov_diff_positive_definite': is_positive_definite,
        'alpha': alpha,
        'interpretation': interpretation,
        'fe_results': fe_results,
        're_results': re_results
    }


def print_hausman_results(results: Dict[str, Any]) -> None:
    """Pretty print Hausman test results."""

    print("=" * 65)
    print("HAUSMAN SPECIFICATION TEST RESULTS")
    print("=" * 65)
    print("\nComparing Fixed Effects vs Random Effects Estimators")
    print("-" * 65)

    # Show coefficient comparison
    print("\nCoefficient Comparison:")
    print(f"{'Variable':<20} {'FE':<12} {'RE':<12} {'Difference':<12}")
    print("-" * 56)
    for var in results['fe_coefficients'].keys():
        fe_val = results['fe_coefficients'][var]
        re_val = results['re_coefficients'][var]
        diff_val = results['coefficient_difference'][var]
        print(f"{var:<20} {fe_val:>10.4f}  {re_val:>10.4f}  {diff_val:>10.4f}")

    print("\n" + "-" * 65)
    print(f"Hausman statistic: {results['hausman_statistic']:.4f}")
    print(f"Degrees of freedom: {results['degrees_of_freedom']}")
    print(f"P-value: {results['p_value']:.4f}")

    print("\nCritical values:")
    for level, value in results['critical_values'].items():
        marker = "*" if results['hausman_statistic'] > value else ""
        print(f"  {level}: {value:.4f} {marker}")

    if not results['is_cov_diff_positive_definite']:
        print("\nWarning: Variance difference matrix was not positive definite.")
        print("         Regularization was applied. Consider robust alternatives.")

    print(f"\nConclusion at alpha = {results['alpha']:.0%}:")
    print(f"  {results['interpretation']}")
    print("=" * 65)


# Example usage with simulated migration data
if __name__ == "__main__":
    # Simulate panel data: immigration to North Dakota from 20 countries over 10 years
    np.random.seed(42)
    n_countries = 20
    n_years = 10

    # Create panel structure
    countries = [f"Country_{i}" for i in range(1, n_countries + 1)]
    years = list(range(2013, 2013 + n_years))

    data_list = []
    for country in countries:
        # Country-specific fixed effect (correlated with GDP for demonstration)
        country_effect = np.random.randn() * 1.5
        base_gdp = np.random.uniform(8, 12)  # Log GDP

        for year in years:
            # Time-varying covariates
            log_gdp = base_gdp + np.random.randn() * 0.1 + 0.02 * (year - 2013)
            exchange_rate = np.random.uniform(0.8, 1.2) + 0.01 * (year - 2013)
            diaspora = 1000 * np.exp(country_effect) + np.random.randn() * 100

            # Immigration depends on covariates AND is correlated with country effect
            # (this correlation is what the Hausman test should detect)
            log_immigration = (
                2.0 +                           # Base
                0.8 * log_gdp +                 # GDP effect
                -0.3 * exchange_rate +          # Exchange rate effect
                0.4 * np.log(max(diaspora, 1)) + # Diaspora effect
                country_effect +                 # Country fixed effect (correlated!)
                np.random.randn() * 0.5          # Error
            )

            data_list.append({
                'country': country,
                'year': year,
                'log_immigration': log_immigration,
                'log_gdp': log_gdp,
                'exchange_rate': exchange_rate,
                'log_diaspora': np.log(max(diaspora, 1))
            })

    panel_df = pd.DataFrame(data_list)

    # Run Hausman test
    results = run_hausman_test(
        data=panel_df,
        dependent_var='log_immigration',
        independent_vars=['log_gdp', 'exchange_rate', 'log_diaspora'],
        entity_col='country',
        time_col='year',
        alpha=0.05
    )

    print_hausman_results(results)
```

---

## Output Interpretation

```
=================================================================
HAUSMAN SPECIFICATION TEST RESULTS
=================================================================

Comparing Fixed Effects vs Random Effects Estimators
-----------------------------------------------------------------

Coefficient Comparison:
Variable             FE           RE           Difference
--------------------------------------------------------
log_gdp              0.8234       0.6891       0.1343
exchange_rate       -0.2876      -0.2654      -0.0222
log_diaspora         0.4156       0.3821       0.0335

-----------------------------------------------------------------
Hausman statistic: 14.2847
Degrees of freedom: 3
P-value: 0.0025

Critical values:
  1%: 11.3449 *
  5%: 7.8147 *
  10%: 6.2514 *

Conclusion at alpha = 5%:
  Reject H0 at 5% level (p=0.0025). Use Fixed Effects model.
  Unobserved country characteristics are correlated with the regressors.
=================================================================
```

**Interpretation of output:**

- **Coefficient comparison:** Shows the estimated effect of each variable under FE and RE. The differences reveal how much the estimates diverge. Larger differences contribute more to the Hausman statistic.

- **Hausman statistic (14.28):** A quadratic form measuring the overall distance between FE and RE estimates, weighted by their relative precision. Larger values indicate greater divergence.

- **Degrees of freedom (3):** Equals the number of time-varying coefficients being compared (log_gdp, exchange_rate, log_diaspora).

- **P-value (0.0025):** The probability of observing this large a difference between FE and RE if they were both consistent. Values below 0.05 indicate systematic differences that favor FE.

- **Conclusion:** With p = 0.0025, we strongly reject the null hypothesis. The country-specific effects are correlated with the economic variables (GDP, exchange rates, diaspora). Using random effects would bias our estimates of how these factors affect immigration. The fixed effects model, which controls for all time-invariant country characteristics, should be used.

---

## References

- Hausman, J. A. (1978). "Specification Tests in Econometrics." *Econometrica*, 46(6), 1251-1271.

- Mundlak, Y. (1978). "On the Pooling of Time Series and Cross Section Data." *Econometrica*, 46(1), 69-85.

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press. (Chapter 10)

- Baltagi, B. H. (2021). *Econometric Analysis of Panel Data* (6th ed.). Springer. (Chapter 4)

- Greene, W. H. (2018). *Econometric Analysis* (8th ed.). Pearson. (Chapter 11)

- Cameron, A. C., and Trivedi, P. K. (2005). *Microeconometrics: Methods and Applications*. Cambridge University Press. (Chapter 21)

- linearmodels Documentation: https://bashtage.github.io/linearmodels/panel/panel/linearmodels.panel.model.PanelOLS.html
