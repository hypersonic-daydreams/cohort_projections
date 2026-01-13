# Statistical Test Explanation: Breusch-Pagan Lagrange Multiplier Test

## Test: Breusch-Pagan LM Test

**Full Name:** Breusch-Pagan Lagrange Multiplier Test for Random Effects
**Category:** Panel Heterogeneity
**Paper Section:** 3.5 Panel Data Results

---

## What This Test Does

The Breusch-Pagan Lagrange Multiplier (LM) test determines whether there is significant unobserved heterogeneity across panel units that warrants using a panel model rather than simple pooled OLS regression. In the context of international migration to North Dakota, this test answers a fundamental question: do different origin countries have systematically different baseline migration propensities that persist over time, or can we treat all country-year observations as independent draws from a single population?

The test examines the variance of the random effects component. Under pooled OLS, we assume all observations share common parameters with no group-specific variation. If significant country-specific effects exist (such as persistent differences due to historical ties, geographic proximity, or diaspora networks), the error terms within each country will be correlated over time. The LM test detects this correlation by checking whether the variance of the group-specific error component is zero. A significant test result indicates that countries differ in ways not captured by the observed covariates, and a panel model (either fixed or random effects) should be used to account for this heterogeneity. This test is typically performed before the Hausman test, as it first establishes whether panel structure matters at all.

---

## The Hypotheses

| Hypothesis | Statement |
|------------|-----------|
| **Null (H0):** | Variance of individual effects is zero (no unobserved heterogeneity; pooled OLS is adequate) |
| **Alternative (H1):** | Variance of individual effects is positive (significant unobserved heterogeneity; use panel model) |

**Formally:** H0: sigma_u^2 = 0 vs H1: sigma_u^2 > 0, where sigma_u^2 is the variance of the random individual effects.

**Important:** Rejecting H0 indicates that a panel model is necessary. However, this test does not specify whether to use fixed effects or random effects; the Hausman test addresses that choice.

---

## Test Statistic

The Breusch-Pagan LM test statistic is derived from the residuals of a pooled OLS regression:

$$
LM = \frac{nT}{2(T-1)} \left[ \frac{\sum_{i=1}^{n} \left( \sum_{t=1}^{T} \hat{e}_{it} \right)^2}{\sum_{i=1}^{n} \sum_{t=1}^{T} \hat{e}_{it}^2} - 1 \right]^2
$$

Where:
- $n$ is the number of cross-sectional units (e.g., origin countries)
- $T$ is the number of time periods
- $\hat{e}_{it}$ is the residual from the pooled OLS regression for unit $i$ at time $t$
- The numerator inside the brackets sums squared group totals of residuals
- The denominator is the total sum of squared residuals

**Simplified interpretation:** The test compares within-group residual correlation to what would be expected under pooled OLS. If residuals within each group tend to have the same sign (indicating persistent group effects), the numerator will be large relative to the denominator.

**Distribution under H0:** Chi-squared with 1 degree of freedom.

---

## Decision Rule

The Breusch-Pagan LM test is a right-tailed test. Reject the null hypothesis if the test statistic exceeds the critical value.

| Significance Level | Critical Value (df=1) | Decision |
|-------------------|----------------------|----------|
| alpha = 0.01 | 6.635 | Reject H0 if LM > 6.635 |
| alpha = 0.05 | 3.841 | Reject H0 if LM > 3.841 |
| alpha = 0.10 | 2.706 | Reject H0 if LM > 2.706 |

**Note:** The test always has 1 degree of freedom because we are testing a single variance parameter (sigma_u^2 = 0).

**P-value approach:** Reject H0 if p-value < alpha

---

## When to Use This Test

**Use when:**
- You have panel data and want to determine if pooled OLS is adequate
- You need to decide whether to use panel methods before choosing between FE and RE
- You want to test for the presence of unobserved heterogeneity across cross-sectional units
- You have a balanced or moderately unbalanced panel
- This is typically the first specification test in panel data analysis

**Do not use when:**
- You only have cross-sectional data (no panel structure)
- You have already decided theoretically that panel effects must exist
- The panel is extremely unbalanced (consider alternatives like cluster-robust tests)
- You are more interested in fixed effects (consider F-test for joint significance of fixed effects)
- There is substantial serial correlation beyond what random effects would imply

---

## Key Assumptions

1. **Correct pooled OLS specification:** The test uses residuals from pooled OLS. If the pooled model is misspecified (wrong functional form, omitted variables), the test may give misleading results.

2. **Random sampling of cross-sectional units:** The test assumes that the n cross-sectional units are randomly sampled from a larger population. For immigration data, this means origin countries can be viewed as draws from a population of potential sending countries.

3. **Balanced or nearly balanced panel:** The standard LM test formula assumes a balanced panel. For unbalanced panels, modified versions exist, but the basic test may have size distortions.

4. **Normality of errors:** The chi-squared approximation relies on approximately normal errors. With highly non-normal errors, consider bootstrap alternatives.

5. **Homoskedastic errors:** The standard test assumes homoskedasticity. Heteroskedasticity can inflate the test statistic and lead to over-rejection of H0.

6. **No serial correlation beyond random effects:** The test assumes the only within-group correlation comes from the random effect. Additional serial correlation requires different tests or robust versions.

---

## Worked Example

**Data:** Panel of annual immigration to North Dakota from 30 countries over 12 years (2011-2022), yielding 360 country-year observations.

**Model:** We regress log immigration on log GDP per capita and a binary indicator for English-speaking origin country.

**Calculation:**
```
Step 1: Estimate pooled OLS regression
        ln(Immigration_ct) = beta_0 + beta_1*ln(GDP_ct) + beta_2*English_c + epsilon_ct

        Pooled OLS Results:
        beta_0 = 2.15 (SE = 0.32)
        beta_1 = 0.68 (SE = 0.11)
        beta_2 = 0.45 (SE = 0.18)
        R-squared = 0.34

Step 2: Obtain residuals from pooled OLS
        e_hat_ct for all 360 observations

Step 3: Calculate sum of residuals within each country
        For each country i: sum_t(e_hat_it) for t = 1,...,12

Step 4: Compute LM statistic components
        - Numerator: sum over i of [sum_t(e_hat_it)]^2 = 847.3
        - Denominator: sum over all i,t of (e_hat_it)^2 = 423.6

Step 5: Calculate LM statistic
        LM = (30 * 12) / (2 * 11) * [(847.3 / 423.6) - 1]^2
        LM = (360 / 22) * [2.00 - 1]^2
        LM = 16.36 * 1.00
        LM = 16.36

Step 6: Compare to chi-squared critical value (df=1)
        Critical values: 1%: 6.635, 5%: 3.841, 10%: 2.706

P-value < 0.001
```

**Interpretation:**
The LM statistic of 16.36 far exceeds the 1% critical value of 6.635, and the p-value is less than 0.001. We strongly reject the null hypothesis that pooled OLS is adequate. There is significant unobserved heterogeneity across origin countries that persists over time. Some countries consistently send more (or fewer) immigrants to North Dakota than their observed characteristics (GDP, language) would predict. A panel model (fixed or random effects) is necessary. The next step is to conduct a Hausman test to choose between fixed and random effects.

---

## Interpreting Results

**If we reject H0 (p-value < 0.05 or LM > 3.841):**
There is significant unobserved heterogeneity across panel units. Pooled OLS is inadequate because it ignores the correlation of errors within groups. A panel model (fixed effects or random effects) should be used. For migration research, this indicates that countries have persistent differences in migration propensity not explained by observed variables (such as historical ties, network effects, or cultural factors). Proceed to the Hausman test to choose between FE and RE.

**If we fail to reject H0:**
We cannot reject that pooled OLS is adequate. The variance of individual effects is not significantly different from zero. This does NOT prove heterogeneity is absent; it may exist but be too small to detect given sample size. For migration data, this might occur if:
- The observed covariates fully explain country differences
- The sample is too small to detect existing heterogeneity
- Country-specific factors vary substantially over time rather than being persistent

In practice, even with insignificant LM tests, researchers often prefer panel methods when theory suggests group-level heterogeneity exists.

---

## Common Pitfalls

- **Using the wrong version for unbalanced panels:** The standard formula assumes balanced panels. For unbalanced data, use modified LM tests that account for varying Ti across groups. Most statistical software handles this automatically.

- **Confusing with heteroskedasticity test:** The Breusch-Pagan test also exists in a heteroskedasticity testing context (Breusch-Pagan-Godfrey test). Ensure you are using the random effects version (LM test) for panel data, not the heteroskedasticity version.

- **Ignoring serial correlation:** The LM test assumes random effects are the only source of within-group correlation. If there is additional serial correlation (such as AR(1) errors), the test may over-reject H0. Consider Wooldridge's test for serial correlation.

- **Over-reliance on the test for theoretical decisions:** If theory strongly suggests country-specific effects exist (as in migration with diaspora networks), use panel methods even if the LM test is marginally insignificant due to low power.

- **Forgetting to follow up with Hausman test:** The LM test only establishes that panel structure matters. It does not indicate whether to use FE or RE. Always follow a significant LM test with a Hausman test.

- **Misinterpreting direction of the test:** This is a one-sided test for positive variance. The null is sigma_u^2 = 0, not sigma_u^2 = some positive value. Negative LM statistics (which can occur due to numerical issues) should be treated as zero.

---

## Related Tests

| Test | Use When |
|------|----------|
| **Hausman Test** | Follow-up test after LM; chooses between fixed and random effects |
| **F-test for Fixed Effects** | Alternative test for joint significance of entity fixed effects |
| **Honda Test** | One-sided version of LM test with better power against random effects |
| **King-Wu Test** | Modified LM test for small samples with better size properties |
| **Wooldridge Serial Correlation Test** | Tests for serial correlation in panel errors; complements LM test |
| **Cluster-robust LM Test** | Robust version when heteroskedasticity is present |

---

## Python Implementation

```python
"""
Breusch-Pagan Lagrange Multiplier Test Implementation
Tests for the presence of random effects (unobserved heterogeneity) in panel data.
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from typing import Dict, Any, Optional


def run_breusch_pagan_lm_test(
    data: pd.DataFrame,
    dependent_var: str,
    independent_vars: list,
    entity_col: str,
    time_col: str,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Perform the Breusch-Pagan Lagrange Multiplier test for random effects.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with entity and time identifiers
    dependent_var : str
        Name of the dependent variable column
    independent_vars : list
        List of independent variable column names
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
    ...     'country': ['A', 'A', 'B', 'B'],
    ...     'year': [2020, 2021, 2020, 2021],
    ...     'immigration': [100, 110, 200, 220],
    ...     'gdp': [50, 52, 80, 82]
    ... })
    >>> results = run_breusch_pagan_lm_test(data, 'immigration', ['gdp'], 'country', 'year')
    >>> print(f"Use {'panel model' if results['use_panel_model'] else 'pooled OLS'}")
    """

    # Get panel dimensions
    n_entities = data[entity_col].nunique()
    n_periods = data[time_col].nunique()
    n_obs = len(data)

    # Check if panel is balanced
    obs_per_entity = data.groupby(entity_col).size()
    is_balanced = obs_per_entity.nunique() == 1

    if is_balanced:
        T = obs_per_entity.iloc[0]
    else:
        # For unbalanced panel, use average T (approximate)
        T = obs_per_entity.mean()

    # Estimate pooled OLS
    y = data[dependent_var].values
    X = sm.add_constant(data[independent_vars].values)

    pooled_model = sm.OLS(y, X).fit()
    residuals = pooled_model.resid

    # Add residuals to dataframe for grouping
    data_with_resid = data.copy()
    data_with_resid['residual'] = residuals

    # Calculate sum of residuals within each entity
    entity_resid_sums = data_with_resid.groupby(entity_col)['residual'].sum()
    sum_squared_entity_totals = (entity_resid_sums ** 2).sum()

    # Total sum of squared residuals
    total_ss_resid = (residuals ** 2).sum()

    # For balanced panel, use standard formula
    if is_balanced:
        n = n_entities
        ratio = sum_squared_entity_totals / total_ss_resid
        lm_statistic = (n * T) / (2 * (T - 1)) * (ratio - 1) ** 2
    else:
        # For unbalanced panel, use modified formula
        Ti = obs_per_entity.values
        n = n_entities

        # Calculate A = sum over i of (sum_t e_it)^2 / Ti
        A = ((entity_resid_sums ** 2) / obs_per_entity).sum()

        # Calculate B = sum over all i,t of e_it^2
        B = total_ss_resid

        # Modified LM statistic for unbalanced panel
        # This is a simplified version; full formula in Baltagi (2021)
        ratio = A / B
        sum_Ti = Ti.sum()
        sum_Ti_sq = (Ti ** 2).sum()

        lm_statistic = (sum_Ti ** 2) / (2 * (sum_Ti_sq - sum_Ti)) * (ratio - 1) ** 2

    # P-value from chi-squared distribution (df = 1)
    df = 1
    p_value = 1 - stats.chi2.cdf(lm_statistic, df)

    # Critical values
    critical_values = {
        '1%': stats.chi2.ppf(0.99, df),
        '5%': stats.chi2.ppf(0.95, df),
        '10%': stats.chi2.ppf(0.90, df)
    }

    # Decision
    use_panel_model = p_value < alpha

    # Calculate variance components (informative but not formally part of test)
    # sigma_u^2 estimate from ratio
    if is_balanced:
        sigma_e_sq = total_ss_resid / (n * T - X.shape[1])
        rho_estimate = 1 - (1 / np.sqrt(ratio)) if ratio > 1 else 0
    else:
        sigma_e_sq = total_ss_resid / (n_obs - X.shape[1])
        rho_estimate = None  # Complex for unbalanced

    # Build interpretation
    if use_panel_model:
        interpretation = (
            f"Reject H0 at {alpha:.0%} level (p={p_value:.4f}). "
            f"Significant unobserved heterogeneity exists across {entity_col}s. "
            f"Use a panel model (FE or RE) rather than pooled OLS. "
            f"Proceed to Hausman test to choose between FE and RE."
        )
    else:
        interpretation = (
            f"Fail to reject H0 at {alpha:.0%} level (p={p_value:.4f}). "
            f"Cannot reject that pooled OLS is adequate. "
            f"No significant evidence of unobserved {entity_col}-level heterogeneity."
        )

    return {
        'test_name': 'Breusch-Pagan LM Test for Random Effects',
        'lm_statistic': lm_statistic,
        'p_value': p_value,
        'degrees_of_freedom': df,
        'critical_values': critical_values,
        'n_entities': n_entities,
        'n_periods': n_periods,
        'n_observations': n_obs,
        'is_balanced': is_balanced,
        'pooled_ols_r_squared': pooled_model.rsquared,
        'sigma_e_squared': sigma_e_sq,
        'use_panel_model': use_panel_model,
        'alpha': alpha,
        'interpretation': interpretation,
        'pooled_results': pooled_model
    }


def print_bp_lm_results(results: Dict[str, Any]) -> None:
    """Pretty print Breusch-Pagan LM test results."""

    print("=" * 65)
    print("BREUSCH-PAGAN LM TEST FOR RANDOM EFFECTS")
    print("=" * 65)
    print("\nTests whether pooled OLS is adequate vs. panel model needed")
    print("-" * 65)

    print("\nPanel Dimensions:")
    print(f"  Cross-sectional units (n): {results['n_entities']}")
    print(f"  Time periods (T): {results['n_periods']}")
    print(f"  Total observations: {results['n_observations']}")
    print(f"  Panel balance: {'Balanced' if results['is_balanced'] else 'Unbalanced'}")

    print("\nPooled OLS Summary:")
    print(f"  R-squared: {results['pooled_ols_r_squared']:.4f}")
    print(f"  Error variance (sigma_e^2): {results['sigma_e_squared']:.4f}")

    print("\n" + "-" * 65)
    print(f"LM statistic: {results['lm_statistic']:.4f}")
    print(f"Degrees of freedom: {results['degrees_of_freedom']}")
    print(f"P-value: {results['p_value']:.4f}")

    print("\nCritical values (chi-squared, df=1):")
    for level, value in results['critical_values'].items():
        marker = "*" if results['lm_statistic'] > value else ""
        print(f"  {level}: {value:.4f} {marker}")

    print(f"\nConclusion at alpha = {results['alpha']:.0%}:")
    print(f"  {results['interpretation']}")
    print("=" * 65)


def run_full_panel_specification_tests(
    data: pd.DataFrame,
    dependent_var: str,
    independent_vars: list,
    entity_col: str,
    time_col: str,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Run the full sequence of panel specification tests:
    1. Breusch-Pagan LM test (pooled vs. panel)
    2. If panel needed, Hausman test (FE vs. RE)

    Returns comprehensive results with recommendation.
    """

    from linearmodels.panel import PanelOLS, RandomEffects

    # Step 1: Breusch-Pagan LM test
    bp_results = run_breusch_pagan_lm_test(
        data, dependent_var, independent_vars, entity_col, time_col, alpha
    )

    results = {
        'breusch_pagan': bp_results,
        'hausman': None,
        'recommendation': None
    }

    if not bp_results['use_panel_model']:
        results['recommendation'] = 'pooled_ols'
        results['recommendation_text'] = (
            "Use Pooled OLS. The Breusch-Pagan test found no significant "
            f"unobserved heterogeneity across {entity_col}s."
        )
    else:
        # Step 2: Hausman test
        panel_data = data.set_index([entity_col, time_col])
        formula = f"{dependent_var} ~ 1 + " + " + ".join(independent_vars)

        # Fit FE and RE
        fe_model = PanelOLS.from_formula(formula + " + EntityEffects", data=panel_data)
        fe_results = fe_model.fit()

        re_model = RandomEffects.from_formula(formula, data=panel_data)
        re_results = re_model.fit()

        # Calculate Hausman test
        fe_params = fe_results.params[independent_vars]
        re_params = re_results.params[independent_vars]
        beta_diff = fe_params - re_params

        fe_cov = fe_results.cov.loc[independent_vars, independent_vars]
        re_cov = re_results.cov.loc[independent_vars, independent_vars]
        cov_diff = fe_cov - re_cov

        try:
            cov_diff_inv = np.linalg.inv(cov_diff.values)
            hausman_stat = float(beta_diff.values @ cov_diff_inv @ beta_diff.values)
        except np.linalg.LinAlgError:
            cov_diff_inv = np.linalg.pinv(cov_diff.values)
            hausman_stat = float(beta_diff.values @ cov_diff_inv @ beta_diff.values)

        hausman_df = len(independent_vars)
        hausman_p = 1 - stats.chi2.cdf(hausman_stat, hausman_df)
        use_fe = hausman_p < alpha

        results['hausman'] = {
            'statistic': hausman_stat,
            'p_value': hausman_p,
            'df': hausman_df,
            'use_fixed_effects': use_fe
        }

        if use_fe:
            results['recommendation'] = 'fixed_effects'
            results['recommendation_text'] = (
                "Use Fixed Effects. The Breusch-Pagan test found significant "
                f"heterogeneity, and the Hausman test (p={hausman_p:.4f}) indicates "
                "individual effects are correlated with regressors."
            )
        else:
            results['recommendation'] = 'random_effects'
            results['recommendation_text'] = (
                "Use Random Effects. The Breusch-Pagan test found significant "
                f"heterogeneity, and the Hausman test (p={hausman_p:.4f}) does not "
                "reject consistency of RE. RE is more efficient."
            )

    return results


# Example usage with simulated migration data
if __name__ == "__main__":
    # Simulate panel data: immigration to ND from 25 countries over 12 years
    np.random.seed(123)
    n_countries = 25
    n_years = 12

    countries = [f"Country_{i:02d}" for i in range(1, n_countries + 1)]
    years = list(range(2011, 2011 + n_years))

    data_list = []
    for country in countries:
        # Country-specific random effect (what the LM test should detect)
        country_effect = np.random.randn() * 2.0

        for year in years:
            # Time-varying covariates
            log_gdp = np.random.uniform(8, 11) + 0.03 * (year - 2011)
            english_speaking = 1 if np.random.rand() < 0.3 else 0

            # Log immigration with country effect
            log_immigration = (
                3.0 +                           # Base
                0.5 * log_gdp +                 # GDP effect
                0.8 * english_speaking +        # Language effect
                country_effect +                 # Persistent country effect
                np.random.randn() * 0.8          # Idiosyncratic error
            )

            data_list.append({
                'country': country,
                'year': year,
                'log_immigration': log_immigration,
                'log_gdp': log_gdp,
                'english_speaking': english_speaking
            })

    panel_df = pd.DataFrame(data_list)

    # Run Breusch-Pagan LM test
    bp_results = run_breusch_pagan_lm_test(
        data=panel_df,
        dependent_var='log_immigration',
        independent_vars=['log_gdp', 'english_speaking'],
        entity_col='country',
        time_col='year',
        alpha=0.05
    )

    print_bp_lm_results(bp_results)

    print("\n")

    # Run full specification test sequence
    print("=" * 65)
    print("FULL PANEL SPECIFICATION TEST SEQUENCE")
    print("=" * 65)

    full_results = run_full_panel_specification_tests(
        data=panel_df,
        dependent_var='log_immigration',
        independent_vars=['log_gdp', 'english_speaking'],
        entity_col='country',
        time_col='year'
    )

    print(f"\nStep 1 - Breusch-Pagan LM Test:")
    print(f"  LM statistic: {full_results['breusch_pagan']['lm_statistic']:.4f}")
    print(f"  P-value: {full_results['breusch_pagan']['p_value']:.4f}")
    print(f"  Panel model needed: {full_results['breusch_pagan']['use_panel_model']}")

    if full_results['hausman'] is not None:
        print(f"\nStep 2 - Hausman Test:")
        print(f"  Hausman statistic: {full_results['hausman']['statistic']:.4f}")
        print(f"  P-value: {full_results['hausman']['p_value']:.4f}")
        print(f"  Use Fixed Effects: {full_results['hausman']['use_fixed_effects']}")

    print(f"\nFinal Recommendation: {full_results['recommendation'].upper()}")
    print(f"  {full_results['recommendation_text']}")
    print("=" * 65)
```

---

## Output Interpretation

```
=================================================================
BREUSCH-PAGAN LM TEST FOR RANDOM EFFECTS
=================================================================

Tests whether pooled OLS is adequate vs. panel model needed
-----------------------------------------------------------------

Panel Dimensions:
  Cross-sectional units (n): 25
  Time periods (T): 12
  Total observations: 300
  Panel balance: Balanced

Pooled OLS Summary:
  R-squared: 0.2847
  Error variance (sigma_e^2): 4.8321

-----------------------------------------------------------------
LM statistic: 187.4523
Degrees of freedom: 1
P-value: 0.0000

Critical values (chi-squared, df=1):
  1%: 6.6349 *
  5%: 3.8415 *
  10%: 2.7055 *

Conclusion at alpha = 5%:
  Reject H0 at 5% level (p=0.0000). Significant unobserved
  heterogeneity exists across countrys. Use a panel model (FE or RE)
  rather than pooled OLS. Proceed to Hausman test to choose between
  FE and RE.
=================================================================

=================================================================
FULL PANEL SPECIFICATION TEST SEQUENCE
=================================================================

Step 1 - Breusch-Pagan LM Test:
  LM statistic: 187.4523
  P-value: 0.0000
  Panel model needed: True

Step 2 - Hausman Test:
  Hausman statistic: 3.2156
  P-value: 0.2004
  Use Fixed Effects: False

Final Recommendation: RANDOM_EFFECTS
  Use Random Effects. The Breusch-Pagan test found significant
  heterogeneity, and the Hausman test (p=0.2004) does not reject
  consistency of RE. RE is more efficient.
=================================================================
```

**Interpretation of output:**

- **Panel dimensions:** Shows the structure of the data. A balanced panel with 25 countries over 12 years yields 300 observations, providing good power for the test.

- **Pooled OLS R-squared (0.28):** The observed covariates (GDP, language) explain only 28% of variation in log immigration. This leaves substantial unexplained variation that may reflect country-specific effects.

- **LM statistic (187.45):** An extremely large value, far exceeding all critical values. This indicates strong evidence that residuals within countries are correlated over time.

- **P-value (< 0.0001):** Essentially zero, providing overwhelming evidence against pooled OLS. Countries have persistent differences in migration propensity beyond what GDP and language explain.

- **Full sequence results:** After confirming panel structure matters (LM test), the Hausman test (p=0.20) fails to reject consistency of random effects. Since both FE and RE appear consistent, RE is preferred for its efficiency and ability to estimate effects of time-invariant covariates like language.

---

## References

- Breusch, T. S., and Pagan, A. R. (1980). "The Lagrange Multiplier Test and its Applications to Model Specification in Econometrics." *Review of Economic Studies*, 47(1), 239-253.

- Honda, Y. (1985). "Testing the Error Components Model with Non-Normal Disturbances." *Review of Economic Studies*, 52(4), 681-690.

- Baltagi, B. H. (2021). *Econometric Analysis of Panel Data* (6th ed.). Springer. (Chapter 4)

- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.). MIT Press. (Chapter 10)

- Greene, W. H. (2018). *Econometric Analysis* (8th ed.). Pearson. (Chapter 11)

- Hsiao, C. (2014). *Analysis of Panel Data* (3rd ed.). Cambridge University Press. (Chapter 3)

- Moulton, B. R., and Randolph, W. C. (1989). "Alternative Tests of the Error Components Model." *Econometrica*, 57(3), 685-693.

- linearmodels Documentation: https://bashtage.github.io/linearmodels/panel/panel/linearmodels.panel.model.RandomEffects.html
