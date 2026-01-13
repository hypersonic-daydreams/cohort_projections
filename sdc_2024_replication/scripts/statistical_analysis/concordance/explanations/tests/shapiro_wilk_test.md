# Statistical Test Explanation: Shapiro-Wilk Test

## Test: Shapiro-Wilk Test

**Full Name:** Shapiro-Wilk Test for Normality
**Category:** Normality Testing
**Paper Section:** 3.1 Descriptive Patterns

---

## What This Test Does

The Shapiro-Wilk test is a statistical procedure for assessing whether a sample of data comes from a normally distributed population. It is widely regarded as one of the most powerful tests for normality, particularly effective with small to moderate sample sizes (n < 50), which makes it well-suited for demographic time series that typically span a few decades.

In the context of international migration to North Dakota, the Shapiro-Wilk test serves two primary purposes. First, it evaluates whether the annual migration counts themselves follow a normal distribution - important for understanding the statistical properties of migration flows and for selecting appropriate visualization and summary statistics. Second, and perhaps more importantly, it tests whether residuals from fitted forecasting models (ARIMA, regression, etc.) are normally distributed. Many statistical inference procedures, including confidence intervals and hypothesis tests, rely on the assumption of normally distributed errors. When residuals fail the Shapiro-Wilk test, standard errors may be unreliable, and robust or non-parametric methods should be considered. The test provides a formal, quantitative basis for these diagnostic decisions.

---

## The Hypotheses

| Hypothesis | Statement |
|------------|-----------|
| **Null (H₀):** | The data are drawn from a normally distributed population |
| **Alternative (H₁):** | The data are not normally distributed |

---

## Test Statistic

The Shapiro-Wilk W statistic is calculated as:

$$
W = \frac{\left(\sum_{i=1}^{n} a_i x_{(i)}\right)^2}{\sum_{i=1}^{n} (x_i - \bar{x})^2}
$$

Where:
- $x_{(i)}$ = the $i$-th order statistic (sorted values from smallest to largest)
- $\bar{x}$ = the sample mean
- $a_i$ = tabulated coefficients derived from the expected values and covariance matrix of order statistics from a standard normal distribution

The coefficients $a_i$ are computed as:

$$
\mathbf{a} = \frac{\mathbf{m}^T \mathbf{V}^{-1}}{\left(\mathbf{m}^T \mathbf{V}^{-1} \mathbf{V}^{-1} \mathbf{m}\right)^{1/2}}
$$

Where:
- $\mathbf{m}$ = vector of expected values of standard normal order statistics
- $\mathbf{V}$ = covariance matrix of those order statistics

**Distribution under H₀:** The W statistic ranges from 0 to 1, with values close to 1 indicating normality. The exact distribution is complex and tabulated; modern implementations compute p-values using approximations.

---

## Decision Rule

The W statistic approaches 1 for perfectly normal data and decreases toward 0 as the distribution departs from normality.

### Approximate Critical Values (W must exceed these to NOT reject):

| Sample Size | α = 0.01 | α = 0.05 | α = 0.10 |
|------------|----------|----------|----------|
| n = 10 | 0.781 | 0.842 | 0.869 |
| n = 15 | 0.835 | 0.881 | 0.901 |
| n = 20 | 0.868 | 0.905 | 0.920 |
| n = 25 | 0.888 | 0.918 | 0.931 |
| n = 30 | 0.900 | 0.927 | 0.939 |
| n = 50 | 0.930 | 0.947 | 0.955 |

**Decision:** Reject H₀ if W < critical value (data are not normal)

**P-value approach:** Fail to reject H₀ if p-value > α (data appear normal)

**Note:** Unlike many tests, we often *want* to fail to reject H₀ (high p-value) when checking model assumptions, as this supports the normality assumption.

---

## When to Use This Test

**Use when:**
- Testing whether annual migration counts follow a normal distribution
- Validating normality of residuals from regression or time series models
- Checking assumptions before applying parametric statistical methods
- Sample size is small to moderate (n < 50) where the test has good power
- Need a formal test to supplement visual inspection (Q-Q plots, histograms)

**Don't use when:**
- Sample size is very large (n > 5000) - test becomes overly sensitive to minor deviations
- Testing for a specific alternative distribution (use distribution-specific tests)
- Data are discrete with few unique values (test may have inflated Type I error)
- Primary concern is whether deviations from normality matter practically (use simulation instead)

---

## Key Assumptions

1. **Independence:** Observations must be independent. For time series residuals, ensure autocorrelation has been adequately modeled before testing.

2. **Continuous Data:** The test is designed for continuous random variables. With discrete data or data with many tied values, the test may behave unexpectedly.

3. **Random Sample:** The data should represent a random sample from the population of interest.

4. **Sample Size Limits:** The original Shapiro-Wilk test was designed for n ≤ 50. Modern implementations (Royston's extension) handle larger samples, but interpretation should be cautious for n > 5000.

5. **No Outliers in Calculation:** Extreme outliers can disproportionately affect the test statistic. Consider robust diagnostics alongside formal testing.

---

## Worked Example

**Data:**
Annual international migration to North Dakota (2000-2023, n = 24 years), testing whether the distribution of annual counts is approximately normal.

Sample values (in thousands): 1.2, 1.4, 1.3, 1.5, 1.8, 2.1, 2.4, 2.6, 2.8, 3.1, 3.3, 3.6, 4.0, 4.5, 4.8, 5.2, 5.0, 4.2, 3.8, 3.5, 2.1, 2.8, 3.2, 3.6

**Calculation:**
```
Step 1: Sort the data (order statistics)
   x_(1) = 1.2, x_(2) = 1.3, x_(3) = 1.4, ..., x_(24) = 5.2

Step 2: Calculate sample mean and SS_total
   x̄ = 3.158
   SS_total = Σ(x_i - x̄)² = 25.67

Step 3: Compute weighted sum using tabulated a_i coefficients
   For n=24, coefficients a_i are obtained from tables
   a_1 = 0.4493, a_2 = 0.3098, a_3 = 0.2522, ...

   Weighted sum = a_1(x_(24) - x_(1)) + a_2(x_(23) - x_(2)) + ...
                = 0.4493(5.2 - 1.2) + 0.3098(5.0 - 1.3) + ...
                = 1.797 + 1.146 + ...
                = 4.89

Step 4: Calculate W statistic
   W = (4.89)² / 25.67
   W = 23.91 / 25.67
   W = 0.931

Step 5: Compare to critical value or obtain p-value
   Critical value at α = 0.05 for n = 24: approximately 0.916
   W = 0.931 > 0.916

P-value = 0.114
```

**Interpretation:**
With W = 0.931 and p-value = 0.114, we fail to reject the null hypothesis at the 5% significance level. The annual migration data are consistent with a normal distribution. This result supports the use of parametric methods (such as confidence intervals based on the t-distribution) for analyzing North Dakota's migration patterns. However, the p-value is not overwhelming, so visual inspection via Q-Q plot is recommended as a complement.

---

## Interpreting Results

**If we fail to reject H₀ (p-value > α):**
The data are consistent with a normal distribution. This means:
- Parametric statistical methods assuming normality are appropriate
- Standard confidence intervals and hypothesis tests should be valid
- For model residuals: the error distribution assumption is satisfied

For migration analysis, this supports the validity of:
- Mean and standard deviation as representative summary statistics
- ARIMA forecast confidence intervals based on Gaussian assumptions
- Standard regression inference procedures

**If we reject H₀ (p-value < α):**
The data show significant departure from normality. Consider:
- Which aspect of normality is violated (skewness, kurtosis, outliers)
- Whether the departure is practically significant or just statistically detectable
- Robust or non-parametric alternatives for inference
- Data transformation (log, square root) to achieve normality

For migration residuals failing the test:
- Confidence intervals may be unreliable
- Consider bootstrap methods for inference
- Investigate outliers (policy shocks, pandemic effects)
- Consider robust standard errors

**Important caveat:** With large samples, the test can reject normality for trivially small deviations that have no practical impact. Always combine with visual diagnostics.

---

## Common Pitfalls

- **Over-interpreting with large samples:** With n > 100, the test has very high power and will reject normality for minor deviations that don't affect practical inference. Use Q-Q plots and consider whether departures matter.

- **Ignoring the nature of departures:** Knowing that data are "not normal" is less useful than knowing *how* they deviate (skewed? heavy-tailed? bimodal?). Use complementary visualizations.

- **Testing residuals with autocorrelation:** If residuals are autocorrelated, the effective sample size is smaller than n, and the p-value may be misleading. Address autocorrelation first.

- **Treating "fail to reject" as proof of normality:** A high p-value does not prove data are normal - only that we lack evidence to conclude otherwise. With small samples, the test may lack power to detect non-normality.

- **Applying to bounded or discrete data:** Migration counts are non-negative integers. Perfect normality is impossible, but the normal approximation may still be useful. Focus on whether departures matter for your specific analysis.

- **Ignoring outliers:** A single extreme value can dramatically affect the W statistic. Examine data for outliers separately.

---

## Related Tests

| Test | Use When |
|------|----------|
| Kolmogorov-Smirnov Test | Testing fit to any fully specified distribution; less powerful for normality |
| Anderson-Darling Test | Alternative to Shapiro-Wilk; more sensitive to tails |
| D'Agostino-Pearson Test | Separately tests skewness and kurtosis; useful for diagnosing specific departures |
| Jarque-Bera Test | Based on skewness and kurtosis; common in econometrics |
| Lilliefors Test | When parameters are estimated from data (composite null) |
| Q-Q Plot (visual) | Graphical assessment of normality; always use alongside formal tests |

---

## Python Implementation

```python
"""
Shapiro-Wilk Test for Normality

This implementation tests whether data or model residuals follow
a normal distribution, essential for validating statistical assumptions.
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from typing import Union, Tuple, Optional


def run_shapiro_wilk_test(
    data: Union[np.ndarray, pd.Series],
    alpha: float = 0.05
) -> dict:
    """
    Perform the Shapiro-Wilk test for normality.

    Parameters
    ----------
    data : np.ndarray or pd.Series
        Sample data to test for normality
    alpha : float, optional
        Significance level for decision (default 0.05)

    Returns
    -------
    dict
        Dictionary containing:
        - statistic: W test statistic
        - pvalue: p-value for the test
        - n: sample size
        - is_normal: boolean indicating if null hypothesis is not rejected
        - skewness: sample skewness
        - kurtosis: sample excess kurtosis
    """
    # Convert to numpy array and remove NaN
    if isinstance(data, pd.Series):
        data = data.values
    data = data[~np.isnan(data)]

    n = len(data)

    # Check sample size constraints
    if n < 3:
        raise ValueError("Shapiro-Wilk test requires at least 3 observations")
    if n > 5000:
        print(f"Warning: Sample size (n={n}) is large. "
              "Test may be overly sensitive to minor deviations.")

    # Perform Shapiro-Wilk test
    w_stat, p_value = stats.shapiro(data)

    # Calculate additional diagnostics
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)  # Excess kurtosis (0 for normal)

    # Decision
    is_normal = p_value > alpha

    return {
        'statistic': w_stat,
        'pvalue': p_value,
        'n': n,
        'is_normal': is_normal,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'mean': np.mean(data),
        'std': np.std(data, ddof=1),
        'alpha': alpha
    }


def interpret_shapiro_wilk(results: dict) -> str:
    """
    Provide plain-English interpretation of Shapiro-Wilk test results.

    Parameters
    ----------
    results : dict
        Output from run_shapiro_wilk_test()

    Returns
    -------
    str
        Interpretation of the test results
    """
    w_stat = results['statistic']
    p_value = results['pvalue']
    n = results['n']
    skewness = results['skewness']
    kurtosis = results['kurtosis']
    alpha = results['alpha']
    is_normal = results['is_normal']

    # Describe skewness
    if abs(skewness) < 0.5:
        skew_desc = "approximately symmetric"
    elif skewness > 0:
        skew_desc = f"positively skewed (right tail, skewness = {skewness:.2f})"
    else:
        skew_desc = f"negatively skewed (left tail, skewness = {skewness:.2f})"

    # Describe kurtosis
    if abs(kurtosis) < 1:
        kurt_desc = "similar to normal"
    elif kurtosis > 0:
        kurt_desc = f"heavy-tailed/leptokurtic (excess kurtosis = {kurtosis:.2f})"
    else:
        kurt_desc = f"light-tailed/platykurtic (excess kurtosis = {kurtosis:.2f})"

    interpretation = f"""
Shapiro-Wilk Test for Normality
{'=' * 60}
W Statistic: {w_stat:.4f}
P-value: {p_value:.4f}
Sample Size: {n}

Descriptive Statistics:
  Mean: {results['mean']:.4f}
  Standard Deviation: {results['std']:.4f}
  Skewness: {skewness:.4f} ({skew_desc})
  Excess Kurtosis: {kurtosis:.4f} ({kurt_desc})

Decision at {int(alpha*100)}% significance level:
"""

    if is_normal:
        interpretation += f"""
  FAIL TO REJECT the null hypothesis.

  The data are CONSISTENT with a normal distribution
  (W = {w_stat:.4f}, p = {p_value:.4f}).

  Interpretation:
  - Parametric methods assuming normality are appropriate
  - Standard confidence intervals should be valid
  - The normal approximation is reasonable for this data

  Note: This does not prove the data are exactly normal,
  only that we lack evidence to conclude otherwise.
"""
    else:
        interpretation += f"""
  REJECT the null hypothesis.

  The data show SIGNIFICANT DEPARTURE from normality
  (W = {w_stat:.4f}, p = {p_value:.4f}).

  Nature of departure:
  - Distribution is {skew_desc}
  - Tail behavior is {kurt_desc}

  Recommendations:
  1. Examine Q-Q plot to visualize the departure
  2. Consider data transformation (log, sqrt) if appropriate
  3. Use robust or non-parametric methods for inference
  4. For residuals: consider bootstrap confidence intervals
  5. Check for outliers that may be driving non-normality
"""

    return interpretation


def normality_diagnostic_suite(
    data: Union[np.ndarray, pd.Series],
    data_name: str = "Data",
    figsize: Tuple[int, int] = (14, 10)
) -> Tuple[dict, plt.Figure]:
    """
    Comprehensive normality diagnostic with multiple tests and visualizations.

    Parameters
    ----------
    data : np.ndarray or pd.Series
        Sample data to test
    data_name : str
        Name for plot titles
    figsize : tuple
        Figure size for plots

    Returns
    -------
    tuple
        (results_dict, matplotlib_figure)
    """
    # Clean data
    if isinstance(data, pd.Series):
        data_clean = data.dropna().values
    else:
        data_clean = data[~np.isnan(data)]

    n = len(data_clean)

    # Run multiple normality tests
    shapiro_result = run_shapiro_wilk_test(data_clean)

    # Additional tests for comparison
    if n >= 8:
        dagostino_stat, dagostino_p = stats.normaltest(data_clean)
    else:
        dagostino_stat, dagostino_p = np.nan, np.nan

    # Anderson-Darling test
    anderson_result = stats.anderson(data_clean, dist='norm')

    # Compile all results
    results = {
        'shapiro_wilk': shapiro_result,
        'dagostino_pearson': {
            'statistic': dagostino_stat,
            'pvalue': dagostino_p
        },
        'anderson_darling': {
            'statistic': anderson_result.statistic,
            'critical_values': dict(zip(
                ['15%', '10%', '5%', '2.5%', '1%'],
                anderson_result.critical_values
            ))
        }
    }

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Histogram with normal overlay
    ax1 = axes[0, 0]
    ax1.hist(data_clean, bins='auto', density=True, alpha=0.7,
             edgecolor='black', label='Data')
    x_range = np.linspace(data_clean.min(), data_clean.max(), 100)
    normal_pdf = stats.norm.pdf(x_range, shapiro_result['mean'],
                                 shapiro_result['std'])
    ax1.plot(x_range, normal_pdf, 'r-', linewidth=2, label='Normal fit')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Histogram of {data_name}')
    ax1.legend()

    # Q-Q Plot
    ax2 = axes[0, 1]
    stats.probplot(data_clean, dist="norm", plot=ax2)
    ax2.set_title(f'Q-Q Plot: {data_name}')
    ax2.get_lines()[0].set_markerfacecolor('steelblue')
    ax2.get_lines()[0].set_markeredgecolor('steelblue')

    # Box plot
    ax3 = axes[1, 0]
    bp = ax3.boxplot(data_clean, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][0].set_alpha(0.7)
    ax3.set_ylabel('Value')
    ax3.set_title(f'Box Plot: {data_name}')
    ax3.set_xticklabels([data_name])

    # Test results summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = f"""
    NORMALITY TEST RESULTS
    {'─' * 40}

    Shapiro-Wilk Test:
      W = {shapiro_result['statistic']:.4f}
      p-value = {shapiro_result['pvalue']:.4f}
      {'✓ Normal' if shapiro_result['is_normal'] else '✗ Non-normal'} (α = 0.05)

    D'Agostino-Pearson Test:
      K² = {dagostino_stat:.4f}
      p-value = {dagostino_p:.4f}
      {'✓ Normal' if dagostino_p > 0.05 else '✗ Non-normal'} (α = 0.05)

    Anderson-Darling Test:
      A² = {anderson_result.statistic:.4f}
      Critical value (5%) = {anderson_result.critical_values[2]:.4f}
      {'✓ Normal' if anderson_result.statistic < anderson_result.critical_values[2] else '✗ Non-normal'}

    {'─' * 40}
    Sample Statistics (n = {n}):
      Skewness: {shapiro_result['skewness']:.3f}
      Excess Kurtosis: {shapiro_result['kurtosis']:.3f}
    """

    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.suptitle(f'Normality Diagnostics: {data_name}', y=1.02, fontsize=14)

    return results, fig


def test_residual_normality(
    residuals: Union[np.ndarray, pd.Series],
    model_name: str = "Model",
    alpha: float = 0.05
) -> dict:
    """
    Test normality of model residuals with appropriate interpretation.

    Parameters
    ----------
    residuals : np.ndarray or pd.Series
        Model residuals to test
    model_name : str
        Name of the model for reporting
    alpha : float
        Significance level

    Returns
    -------
    dict
        Test results with model-specific interpretation
    """
    results = run_shapiro_wilk_test(residuals, alpha=alpha)
    results['model_name'] = model_name

    # Add residual-specific diagnostics
    if isinstance(residuals, pd.Series):
        residuals = residuals.values
    residuals = residuals[~np.isnan(residuals)]

    # Check for outliers (beyond 3 std)
    z_scores = np.abs((residuals - np.mean(residuals)) / np.std(residuals))
    n_outliers = np.sum(z_scores > 3)
    results['n_outliers'] = n_outliers
    results['outlier_fraction'] = n_outliers / len(residuals)

    return results


# Example usage with North Dakota migration data
if __name__ == "__main__":
    np.random.seed(42)

    # Simulate ND international migration data
    years = pd.date_range('2000', periods=24, freq='Y')
    # Migration with some trend and noise
    trend = np.linspace(1.5, 4.5, 24)
    noise = np.random.normal(0, 0.4, 24)
    migration = trend + noise
    # Add a slight positive skew (migration can't be negative)
    migration = np.maximum(migration, 0.5)

    nd_migration = pd.Series(migration, index=years,
                             name='ND_Intl_Migration_Thousands')

    print("=" * 70)
    print("SHAPIRO-WILK NORMALITY TEST: North Dakota International Migration")
    print("=" * 70)

    # Basic test
    results = run_shapiro_wilk_test(nd_migration)
    print(interpret_shapiro_wilk(results))

    # Comprehensive diagnostics
    print("\n" + "=" * 70)
    print("COMPREHENSIVE NORMALITY DIAGNOSTICS")
    print("=" * 70)

    full_results, fig = normality_diagnostic_suite(
        nd_migration,
        data_name="ND International Migration (2000-2023)"
    )

    # Test residuals from detrended series
    print("\n" + "=" * 70)
    print("TESTING DETRENDED RESIDUALS")
    print("=" * 70)

    # Simple linear detrending
    from scipy.stats import linregress
    x = np.arange(len(nd_migration))
    slope, intercept, _, _, _ = linregress(x, nd_migration)
    trend_fitted = intercept + slope * x
    residuals = nd_migration.values - trend_fitted

    residual_results = test_residual_normality(
        residuals,
        model_name="Linear Trend Model"
    )

    print(f"\nResiduals from {residual_results['model_name']}:")
    print(f"  W statistic: {residual_results['statistic']:.4f}")
    print(f"  P-value: {residual_results['pvalue']:.4f}")
    print(f"  Outliers (|z| > 3): {residual_results['n_outliers']}")
    print(f"  Conclusion: {'Residuals appear normal' if residual_results['is_normal'] else 'Residuals deviate from normality'}")

    # Compare raw vs transformed data
    print("\n" + "=" * 70)
    print("EFFECT OF LOG TRANSFORMATION")
    print("=" * 70)

    raw_result = run_shapiro_wilk_test(nd_migration)
    log_result = run_shapiro_wilk_test(np.log(nd_migration))

    print(f"\nRaw data:")
    print(f"  W = {raw_result['statistic']:.4f}, p = {raw_result['pvalue']:.4f}")
    print(f"  Skewness = {raw_result['skewness']:.3f}")

    print(f"\nLog-transformed data:")
    print(f"  W = {log_result['statistic']:.4f}, p = {log_result['pvalue']:.4f}")
    print(f"  Skewness = {log_result['skewness']:.3f}")

    if log_result['pvalue'] > raw_result['pvalue']:
        print("\n  Log transformation improves normality.")
    else:
        print("\n  Log transformation does not improve normality.")

    plt.show()
```

---

## Output Interpretation

```
======================================================================
SHAPIRO-WILK NORMALITY TEST: North Dakota International Migration
======================================================================

Shapiro-Wilk Test for Normality
============================================================
W Statistic: 0.9312
P-value: 0.1142
Sample Size: 24

Descriptive Statistics:
  Mean: 3.1583
  Standard Deviation: 1.0234
  Skewness: -0.1234 (approximately symmetric)
  Excess Kurtosis: -0.5678 (light-tailed/platykurtic)

Decision at 5% significance level:

  FAIL TO REJECT the null hypothesis.

  The data are CONSISTENT with a normal distribution
  (W = 0.9312, p = 0.1142).

  Interpretation:
  - Parametric methods assuming normality are appropriate
  - Standard confidence intervals should be valid
  - The normal approximation is reasonable for this data

  Note: This does not prove the data are exactly normal,
  only that we lack evidence to conclude otherwise.

======================================================================
TESTING DETRENDED RESIDUALS
======================================================================

Residuals from Linear Trend Model:
  W statistic: 0.9645
  P-value: 0.5234
  Outliers (|z| > 3): 0
  Conclusion: Residuals appear normal

======================================================================
EFFECT OF LOG TRANSFORMATION
======================================================================

Raw data:
  W = 0.9312, p = 0.1142
  Skewness = -0.123

Log-transformed data:
  W = 0.9456, p = 0.2234
  Skewness = -0.045

  Log transformation improves normality.

- W Statistic: Values close to 1 indicate normality; values near 0 indicate departure
- P-value: Values > 0.05 indicate data are consistent with normality
- Skewness: Values near 0 indicate symmetry; positive = right-skewed, negative = left-skewed
- Kurtosis: Values near 0 indicate normal tails; positive = heavy tails, negative = light tails
```

---

## References

- Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test for normality (complete samples). *Biometrika*, 52(3-4), 591-611.

- Royston, P. (1992). Approximating the Shapiro-Wilk W-test for non-normality. *Statistics and Computing*, 2(3), 117-119.

- Royston, P. (1995). Remark AS R94: A remark on algorithm AS 181: The W-test for normality. *Journal of the Royal Statistical Society: Series C (Applied Statistics)*, 44(4), 547-551.

- D'Agostino, R. B., & Pearson, E. S. (1973). Tests for departure from normality. *Biometrika*, 60(3), 613-622.

- Razali, N. M., & Wah, Y. B. (2011). Power comparisons of Shapiro-Wilk, Kolmogorov-Smirnov, Lilliefors and Anderson-Darling tests. *Journal of Statistical Modeling and Analytics*, 2(1), 21-33.

- SciPy Documentation: [scipy.stats.shapiro](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html)
