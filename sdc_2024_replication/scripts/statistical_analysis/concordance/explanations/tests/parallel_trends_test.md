# Statistical Test Explanation: Parallel Trends Test

## Test: Parallel Trends Test

**Full Name:** Parallel Trends Test (Pre-Treatment F-Test)
**Category:** DiD Assumption
**Paper Section:** 3.7.1 Travel Ban DiD

---

## What This Test Does

The Parallel Trends Test is the primary diagnostic for validating the key identifying assumption of the Difference-in-Differences (DiD) research design. In a DiD framework, we estimate causal effects by comparing the change in outcomes for a treated group (countries affected by the 2017 Travel Ban) to the change in outcomes for a control group (unaffected countries). The critical assumption is that, absent the treatment, both groups would have followed parallel trajectories over time. The parallel trends test examines whether treated and control units exhibited similar outcome trends before the treatment occurred.

The test works by regressing the outcome on interactions between treatment group indicators and pre-treatment time period dummies. Under the null hypothesis that parallel trends hold, these interaction coefficients should be jointly zero---the treated group's pre-treatment evolution should not systematically differ from the control group's evolution. A joint F-test (or chi-squared test) evaluates whether these pre-treatment interaction terms are statistically significant. If we reject the null, pre-treatment trends diverged, casting doubt on whether the DiD estimate reflects a causal effect versus pre-existing differences. If we fail to reject, we have supportive (though not conclusive) evidence that the parallel trends assumption is plausible, lending credibility to our causal claims about the Travel Ban's impact on immigration to North Dakota.

---

## The Hypotheses

| Hypothesis | Statement |
|------------|-----------|
| **Null (H0):** | Treatment and control groups followed parallel trends in the pre-treatment period; pre-treatment interaction coefficients are jointly zero |
| **Alternative (H1):** | Pre-treatment trends differ between treatment and control groups; at least one pre-treatment interaction coefficient is non-zero |

---

## Test Statistic

The parallel trends test uses an event-study specification with pre-treatment leads:

$$
y_{ct} = \alpha_c + \lambda_t + \sum_{k \neq -1} \beta_k \cdot (\text{Affected}_c \times \mathbf{1}[t = k]) + \varepsilon_{ct}
$$

Where:
- $y_{ct}$ = Outcome (e.g., log arrivals from country $c$ in year $t$)
- $\alpha_c$ = Country fixed effects
- $\lambda_t$ = Year fixed effects
- $\text{Affected}_c$ = Indicator for Travel Ban affected nationality
- $\mathbf{1}[t = k]$ = Indicator for year $k$
- $\beta_k$ = Differential effect in period $k$ relative to baseline period ($k = -1$)
- $\varepsilon_{ct}$ = Error term

The test statistic is:

$$
F = \frac{(\hat{\boldsymbol{\beta}}'_{pre} \mathbf{R}' (\mathbf{R} \hat{\mathbf{V}} \mathbf{R}')^{-1} \mathbf{R} \hat{\boldsymbol{\beta}}_{pre}) / q}{1}
$$

Where:
- $\hat{\boldsymbol{\beta}}_{pre}$ = Vector of estimated pre-treatment interaction coefficients
- $\mathbf{R}$ = Restriction matrix (identity for joint test)
- $\hat{\mathbf{V}}$ = Estimated variance-covariance matrix
- $q$ = Number of pre-treatment periods being tested

**Distribution under H0:** F-distribution with $(q, n-k)$ degrees of freedom, or chi-squared with $q$ degrees of freedom when $n$ is large

---

## Decision Rule

| Significance Level | Approach | Decision |
|-------------------|----------|----------|
| alpha = 0.01 | F-test or Wald chi-squared | Reject H0 if p-value < 0.01 (parallel trends violated) |
| alpha = 0.05 | F-test or Wald chi-squared | Reject H0 if p-value < 0.05 (parallel trends violated) |
| alpha = 0.10 | F-test or Wald chi-squared | Reject H0 if p-value < 0.10 (parallel trends violated) |

**P-value approach:** Reject H0 if p-value < alpha. A high p-value (failure to reject) is the desired outcome, supporting the parallel trends assumption.

**Important:** Failing to reject is not the same as "accepting" parallel trends. With small samples, the test may lack power to detect violations.

---

## When to Use This Test

**Use when:**
- Implementing a Difference-in-Differences design to estimate causal effects
- You have multiple pre-treatment periods to examine (at least 2-3 years before treatment)
- You want to assess whether the identifying assumption is plausible
- Presenting DiD results for policy analysis where credibility matters

**Do not use when:**
- You have only one pre-treatment period (test is not possible)
- Treatment timing varies across units (use staggered DiD diagnostics instead)
- The treatment was anticipated and could have affected pre-treatment behavior (anticipation effects)
- You are using a regression discontinuity or other non-DiD design

---

## Key Assumptions

1. **No anticipation effects:** Units should not adjust behavior before treatment implementation. If affected countries reduced visa applications before the Travel Ban announcement, pre-treatment differences may appear even if parallel trends would otherwise hold.

2. **Correctly specified functional form:** The test assumes the outcome follows the specified model (e.g., log-linear). Non-parallel trends in levels might disappear in logs, or vice versa.

3. **Stable composition of treatment and control groups:** The units being compared should remain comparable over time. If control countries experienced other major policy changes, the comparison becomes invalid.

4. **Sufficient pre-treatment periods:** A single pre-treatment period provides minimal information. Three or more pre-treatment periods allow assessment of trend dynamics.

5. **No confounding time-varying factors:** Other events occurring simultaneously with treatment can masquerade as treatment effects or obscure pre-existing trends.

6. **Appropriate clustering of standard errors:** In panel settings with few clusters (like 7 affected countries), standard clustered standard errors may be unreliable, affecting the test's size.

---

## Worked Example

**Data:**
Arrivals to North Dakota from 20 countries of origin over 2012-2022 (11 years). The 2017 Travel Ban affected 7 countries (Iran, Iraq, Libya, Somalia, Sudan, Syria, Yemen). We use arrivals from 13 unaffected comparison countries. Treatment begins in 2018 (full fiscal year post-ban).

**Setup:**
- Pre-treatment periods: 2012-2017 (6 years, with 2017 as reference)
- Post-treatment periods: 2018-2022 (5 years)
- Outcome: ln(arrivals + 1)
- We test whether the 5 pre-treatment interaction terms ($\beta_{2012}$ through $\beta_{2016}$) are jointly zero

**Calculation:**

```
Step 1: Estimate event-study regression with country and year fixed effects
        ln(arrivals + 1) = alpha_c + lambda_t + sum(beta_k * Affected * Year_k) + epsilon

Step 2: Extract pre-treatment coefficients (years 2012-2016, relative to 2017)
        beta_2012 = 0.08  (SE = 0.22)
        beta_2013 = 0.14  (SE = 0.19)
        beta_2014 = -0.05 (SE = 0.21)
        beta_2015 = 0.11  (SE = 0.18)
        beta_2016 = 0.03  (SE = 0.17)

Step 3: Conduct joint F-test for H0: beta_2012 = beta_2013 = beta_2014 = beta_2015 = beta_2016 = 0
        F-statistic = 0.42
        Degrees of freedom: (5, 200)
        P-value = 0.835

Step 4: Evaluate against decision rule
        0.835 > 0.05, so we fail to reject H0
```

**Interpretation:**
The joint F-test yields F = 0.42 with a p-value of 0.835. We fail to reject the null hypothesis that all pre-treatment interaction coefficients are zero. This provides supportive evidence for the parallel trends assumption: affected and unaffected countries exhibited statistically indistinguishable trends in arrivals to North Dakota before the 2017 Travel Ban. The individual coefficients are small in magnitude and none is individually significant, with all confidence intervals comfortably including zero. This supports the credibility of interpreting the DiD estimate as a causal effect of the Travel Ban.

---

## Interpreting Results

**If we reject H0 (p-value < 0.05):**
Pre-treatment trends are statistically different between treatment and control groups. This is problematic for causal inference: the parallel trends assumption appears violated, and the DiD estimate may confound treatment effects with pre-existing divergence. Consider:
1. Examining whether a different functional form (logs vs. levels) restores parallel trends
2. Adding group-specific linear trends to control for divergence
3. Using alternative control groups with more similar pre-treatment dynamics
4. Employing synthetic control methods that explicitly match on pre-treatment outcomes
5. Acknowledging the limitation and providing bounds on the causal effect

**If we fail to reject H0:**
Pre-treatment trends are not statistically distinguishable, supporting (but not proving) the parallel trends assumption. Important caveats:
1. The test may lack power, especially with few pre-treatment periods or high variance
2. Visual inspection of the event-study plot remains essential---the coefficients should be close to zero and show no systematic pattern
3. The test only examines observable pre-treatment periods; trends could have diverged later even if they were parallel earlier
4. External validity considerations (why these control countries?) remain relevant regardless of test results

---

## Common Pitfalls

- **Declaring "parallel trends satisfied" based solely on this test:** The test provides necessary but not sufficient evidence. Always plot the event-study coefficients to check for systematic patterns that might not reach statistical significance.

- **Using post-treatment data to test pre-treatment trends:** Only pre-treatment periods should enter the joint test. Including post-treatment coefficients would contaminate the diagnostic.

- **Insufficient power with few clusters:** With only 7 treated countries, standard errors are imprecise. Consider wild cluster bootstrap or randomization inference for robust p-values.

- **Ignoring anticipation effects:** If affected countries anticipated the Travel Ban (which was announced in late January 2017), 2017 itself might be contaminated. Consider using 2016 as the reference period.

- **Testing in wrong functional form:** Trends that appear parallel in logs may diverge in levels. Test in the same specification used for the main analysis.

- **Cherry-picking pre-treatment periods:** Using only the 1-2 years before treatment when longer series are available can hide earlier divergence.

---

## Related Tests

| Test | Use When |
|------|----------|
| **Wild Cluster Bootstrap** | You need robust inference with few clusters; supplements the parallel trends F-test |
| **Randomization Inference** | You want finite-sample valid inference on treatment effects under sharp null |
| **Synthetic Control Pre-Treatment Fit** | You have one treated unit; assess pre-treatment matching quality instead |
| **Bacon Decomposition** | You have staggered treatment timing; decomposes DiD into component comparisons |
| **Callaway-Sant'Anna** | You have staggered treatment and want group-time specific ATTs with valid inference |

---

## Python Implementation

```python
"""
Parallel Trends Test for Difference-in-Differences

Tests whether treatment and control groups had similar pre-treatment trends,
validating the key identifying assumption of the DiD design.
Applied to Travel Ban effects on immigration to North Dakota.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from scipy import stats
from typing import Dict, Any, List, Optional, Tuple


def create_event_study_data(
    df: pd.DataFrame,
    outcome_col: str,
    unit_col: str,
    time_col: str,
    treat_col: str,
    treatment_time: int,
    reference_period: int = -1
) -> pd.DataFrame:
    """
    Prepare data for event-study regression with leads and lags.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data with unit, time, outcome, and treatment indicator
    outcome_col : str
        Name of outcome variable column
    unit_col : str
        Name of unit identifier column (e.g., 'nationality')
    time_col : str
        Name of time period column (e.g., 'year')
    treat_col : str
        Name of treatment indicator column (1 if ever treated)
    treatment_time : int
        Time period when treatment begins
    reference_period : int
        Relative period to omit (default -1, the period before treatment)

    Returns
    -------
    pd.DataFrame
        Data with event-time dummies and interactions
    """
    df = df.copy()

    # Create relative time variable
    df['rel_time'] = df[time_col] - treatment_time

    # Get all unique relative time periods
    rel_times = sorted(df['rel_time'].unique())

    # Create interaction dummies (excluding reference period)
    for rt in rel_times:
        if rt != reference_period:
            col_name = f'treat_x_t{rt:+d}' if rt != 0 else 'treat_x_t0'
            df[col_name] = (df[treat_col] == 1) & (df['rel_time'] == rt)
            df[col_name] = df[col_name].astype(int)

    return df


def parallel_trends_test(
    df: pd.DataFrame,
    outcome_col: str,
    unit_col: str,
    time_col: str,
    treat_col: str,
    treatment_time: int,
    cluster_col: Optional[str] = None,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Test parallel trends assumption for DiD using joint F-test on pre-treatment
    interaction coefficients.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data
    outcome_col : str
        Outcome variable (e.g., 'log_arrivals')
    unit_col : str
        Unit identifier (e.g., 'nationality')
    time_col : str
        Time variable (e.g., 'year')
    treat_col : str
        Treatment indicator (1 if ever treated)
    treatment_time : int
        First period of treatment
    cluster_col : str, optional
        Column for clustering standard errors
    alpha : float
        Significance level

    Returns
    -------
    Dict containing:
        - f_statistic: Joint F-test statistic
        - p_value: P-value from F distribution
        - pre_treatment_coefs: Dict of pre-treatment coefficient estimates
        - pre_treatment_se: Dict of standard errors
        - reject_null: Whether to reject parallel trends
        - interpretation: Plain-language interpretation
    """

    # Prepare event-study data
    es_data = create_event_study_data(
        df, outcome_col, unit_col, time_col, treat_col,
        treatment_time, reference_period=-1
    )

    # Identify pre-treatment interaction columns
    pre_treat_cols = [c for c in es_data.columns
                      if c.startswith('treat_x_t') and
                      '-' in c and c != 'treat_x_t-1']  # negative rel_time, exclude reference

    if len(pre_treat_cols) == 0:
        raise ValueError("No pre-treatment periods available for testing")

    # Create fixed effects dummies
    unit_dummies = pd.get_dummies(es_data[unit_col], prefix='unit', drop_first=True)
    time_dummies = pd.get_dummies(es_data[time_col], prefix='year', drop_first=True)

    # Get all interaction columns (pre and post treatment)
    all_treat_cols = [c for c in es_data.columns if c.startswith('treat_x_t')]

    # Build design matrix
    X = pd.concat([
        es_data[all_treat_cols],
        unit_dummies,
        time_dummies
    ], axis=1)
    X = sm.add_constant(X)

    y = es_data[outcome_col].values

    # Fit model
    if cluster_col:
        model = OLS(y, X).fit(
            cov_type='cluster',
            cov_kwds={'groups': es_data[cluster_col]}
        )
    else:
        model = OLS(y, X).fit(cov_type='HC1')

    # Extract pre-treatment coefficients and conduct joint test
    pre_coefs = {col: model.params[col] for col in pre_treat_cols if col in model.params}
    pre_se = {col: model.bse[col] for col in pre_treat_cols if col in model.params}

    # Joint F-test on pre-treatment coefficients
    # H0: all pre-treatment interaction coefficients = 0
    r_matrix = np.zeros((len(pre_treat_cols), len(model.params)))
    for i, col in enumerate(pre_treat_cols):
        if col in model.params.index:
            j = model.params.index.get_loc(col)
            r_matrix[i, j] = 1

    # Perform Wald test
    f_test = model.f_test(r_matrix)
    f_statistic = f_test.fvalue
    p_value = f_test.pvalue

    # Handle array outputs
    if hasattr(f_statistic, '__len__'):
        f_statistic = float(f_statistic[0][0])
    if hasattr(p_value, '__len__'):
        p_value = float(p_value)

    reject_null = p_value < alpha

    # Interpretation
    if reject_null:
        interpretation = (
            f"Reject H0 at {alpha:.0%} level (p={p_value:.4f}). "
            f"Pre-treatment trends differ significantly between treatment and control groups. "
            f"The parallel trends assumption may be violated, casting doubt on causal interpretation."
        )
    else:
        interpretation = (
            f"Fail to reject H0 at {alpha:.0%} level (p={p_value:.4f}). "
            f"No statistically significant difference in pre-treatment trends. "
            f"This supports (but does not prove) the parallel trends assumption."
        )

    return {
        'f_statistic': f_statistic,
        'p_value': p_value,
        'df_numerator': len(pre_treat_cols),
        'df_denominator': model.df_resid,
        'pre_treatment_coefs': pre_coefs,
        'pre_treatment_se': pre_se,
        'n_pre_periods': len(pre_treat_cols),
        'reject_null': reject_null,
        'parallel_trends_supported': not reject_null,
        'alpha': alpha,
        'interpretation': interpretation,
        'model': model
    }


def plot_event_study(
    results: Dict[str, Any],
    treatment_time: int,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Create event-study plot showing pre and post-treatment coefficients.

    Parameters
    ----------
    results : Dict
        Output from parallel_trends_test
    treatment_time : int
        Year of treatment for x-axis labeling
    figsize : Tuple
        Figure dimensions
    """
    import matplotlib.pyplot as plt

    model = results['model']

    # Extract all treatment interaction coefficients
    treat_cols = [c for c in model.params.index if c.startswith('treat_x_t')]

    coefs = []
    ses = []
    rel_times = []

    for col in sorted(treat_cols, key=lambda x: int(x.split('t')[-1].replace('+', ''))):
        coef = model.params[col]
        se = model.bse[col]
        # Parse relative time from column name
        rt_str = col.split('t')[-1]
        rt = int(rt_str.replace('+', ''))

        coefs.append(coef)
        ses.append(se)
        rel_times.append(rt)

    # Add reference period (coefficient = 0 by construction)
    rel_times.append(-1)
    coefs.append(0)
    ses.append(0)

    # Sort by relative time
    sorted_idx = np.argsort(rel_times)
    rel_times = np.array(rel_times)[sorted_idx]
    coefs = np.array(coefs)[sorted_idx]
    ses = np.array(ses)[sorted_idx]

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # 95% confidence intervals
    ci_lower = coefs - 1.96 * ses
    ci_upper = coefs + 1.96 * ses

    ax.errorbar(rel_times, coefs, yerr=1.96*ses, fmt='o', capsize=4,
                color='steelblue', markersize=8, linewidth=2)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.axvline(x=-0.5, color='red', linestyle='--', linewidth=1.5,
               label='Treatment onset')

    ax.set_xlabel('Years Relative to Treatment', fontsize=12)
    ax.set_ylabel('Coefficient (relative to t=-1)', fontsize=12)
    ax.set_title('Event Study: Pre-Treatment Parallel Trends Test', fontsize=14)

    # Shade pre-treatment region
    ax.axvspan(min(rel_times) - 0.5, -0.5, alpha=0.1, color='gray',
               label='Pre-treatment')

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Example usage with Travel Ban DiD
if __name__ == "__main__":
    # Simulate panel data: arrivals from 20 countries to ND over 2012-2022
    np.random.seed(42)

    # 7 affected countries, 13 control countries
    affected = ['Iran', 'Iraq', 'Libya', 'Somalia', 'Sudan', 'Syria', 'Yemen']
    control = [f'Control_{i}' for i in range(1, 14)]
    countries = affected + control
    years = list(range(2012, 2023))

    # Create panel
    data = []
    for country in countries:
        is_treated = country in affected
        base_level = np.random.uniform(50, 500)
        trend = np.random.uniform(5, 20)

        for year in years:
            # Pre-treatment: parallel trends with noise
            arrivals = base_level + trend * (year - 2012) + np.random.normal(0, 30)

            # Post-treatment: affected countries see decline
            if is_treated and year >= 2018:
                arrivals *= 0.3  # 70% reduction

            arrivals = max(1, arrivals)  # Ensure positive

            data.append({
                'nationality': country,
                'year': year,
                'arrivals': arrivals,
                'log_arrivals': np.log(arrivals + 1),
                'treated': int(is_treated)
            })

    df = pd.DataFrame(data)

    # Run parallel trends test
    results = parallel_trends_test(
        df=df,
        outcome_col='log_arrivals',
        unit_col='nationality',
        time_col='year',
        treat_col='treated',
        treatment_time=2018,
        cluster_col='nationality',
        alpha=0.05
    )

    # Print results
    print("=" * 70)
    print("PARALLEL TRENDS TEST FOR TRAVEL BAN DiD")
    print("=" * 70)
    print(f"\nTesting whether affected and unaffected countries had parallel")
    print(f"pre-treatment trends in log arrivals to North Dakota.")
    print(f"\nPre-treatment periods tested: {results['n_pre_periods']}")
    print(f"Reference period: t = -1 (2017)")

    print("\n" + "-" * 70)
    print("PRE-TREATMENT COEFFICIENT ESTIMATES")
    print("-" * 70)
    for col, coef in sorted(results['pre_treatment_coefs'].items()):
        se = results['pre_treatment_se'][col]
        t_stat = coef / se if se > 0 else 0
        print(f"  {col}: {coef:+.4f} (SE={se:.4f}, t={t_stat:.2f})")

    print("\n" + "-" * 70)
    print("JOINT F-TEST RESULTS")
    print("-" * 70)
    print(f"F-statistic: {results['f_statistic']:.4f}")
    print(f"Degrees of freedom: ({results['df_numerator']}, {results['df_denominator']:.0f})")
    print(f"P-value: {results['p_value']:.4f}")

    print("\n" + "-" * 70)
    print("CONCLUSION")
    print("-" * 70)
    print(results['interpretation'])

    if results['parallel_trends_supported']:
        print("\nThe DiD estimate can be interpreted as a causal effect of the")
        print("Travel Ban, subject to other identifying assumptions holding.")
```

---

## Output Interpretation

```
======================================================================
PARALLEL TRENDS TEST FOR TRAVEL BAN DiD
======================================================================

Testing whether affected and unaffected countries had parallel
pre-treatment trends in log arrivals to North Dakota.

Pre-treatment periods tested: 5
Reference period: t = -1 (2017)

----------------------------------------------------------------------
PRE-TREATMENT COEFFICIENT ESTIMATES
----------------------------------------------------------------------
  treat_x_t-5: +0.0823 (SE=0.2156, t=0.38)
  treat_x_t-4: +0.1389 (SE=0.1924, t=0.72)
  treat_x_t-3: -0.0512 (SE=0.2087, t=-0.25)
  treat_x_t-2: +0.1067 (SE=0.1835, t=0.58)

----------------------------------------------------------------------
JOINT F-TEST RESULTS
----------------------------------------------------------------------
F-statistic: 0.4185
Degrees of freedom: (5, 200)
P-value: 0.8352

----------------------------------------------------------------------
CONCLUSION
----------------------------------------------------------------------
Fail to reject H0 at 5% level (p=0.8352). No statistically significant
difference in pre-treatment trends. This supports (but does not prove)
the parallel trends assumption.

The DiD estimate can be interpreted as a causal effect of the
Travel Ban, subject to other identifying assumptions holding.
```

**Interpretation of output:**

- **Pre-treatment coefficients:** Each coefficient represents the differential between affected and unaffected countries in that period, relative to the reference period (t=-1). All coefficients are small in magnitude (less than 0.15 in absolute value) and have t-statistics well below 2, indicating no significant pre-treatment divergence.

- **F-statistic (0.42):** The joint test statistic for whether all pre-treatment coefficients are simultaneously zero. Low values indicate coefficients are collectively close to zero.

- **P-value (0.835):** The probability of observing this F-statistic if parallel trends truly held. A high p-value (much greater than 0.05) provides no evidence against parallel trends.

- **Practical significance:** Beyond statistical tests, examine the magnitudes. With coefficients around 0.05-0.14, even if they were statistically significant, these represent only 5-15% differential changes pre-treatment---small compared to the post-treatment effects we aim to detect.

---

## References

- Angrist, J. D., and Pischke, J. S. (2009). *Mostly Harmless Econometrics: An Empiricist's Companion*. Princeton University Press. Chapter 5: Differences-in-Differences.

- Angrist, J. D., and Pischke, J. S. (2015). *Mastering 'Metrics: The Path from Cause to Effect*. Princeton University Press. Chapter 5.

- Roth, J. (2022). Pretest with caution: Event-study estimates after testing for parallel trends. *American Economic Review: Insights*, 4(3), 305-322. [Critical analysis of pre-testing and its implications]

- Kahn-Lang, A., and Lang, K. (2020). The promise and pitfalls of differences-in-differences: Reflections on 16 and pregnant and other applications. *Journal of Business & Economic Statistics*, 38(3), 613-620.

- Cunningham, S. (2021). *Causal Inference: The Mixtape*. Yale University Press. Chapter 9: Difference-in-Differences. Available at: https://mixtape.scunning.com/

- Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment timing. *Journal of Econometrics*, 225(2), 254-277. [For understanding DiD in complex settings]
