# Agent Report: 2 - Statistical Transition Analysis

## Metadata

| Field | Value |
|-------|-------|
| Agent | 2 |
| Title | Statistical Transition Analysis: Detecting Vintage-Related Artifacts in North Dakota International Migration Time Series |
| Date | 2026-01-01 |
| Status | Complete |
| Confidence Level | Medium |

---

## Executive Summary

This investigation applied multiple statistical tests to detect vintage-related artifacts in North Dakota's international migration time series (2000-2024). The analysis found **significant evidence of a level shift at the 2009-2010 vintage boundary** (t-test p=0.003, Cohen's d=1.56) and **significant variance heterogeneity across vintages** (Levene's p=0.002, variance ratio=29:1). However, the Chow structural break test at the 2009-2010 transition was **not significant** (F=0.84, p=0.45), matching placebo tests at other years. The 2019-2020 transition analysis is confounded by COVID-19.

**Bottom Line**: Statistical evidence is mixed. Mean levels differ significantly across vintages, but trend-based structural break tests do not detect the first transition as unusual compared to other years.

**Recommendation**: Proceed with caution; the significant level shift warrants sensitivity analyses, but the lack of structural break detection suggests the effect may be modest relative to overall variability.

---

## Scope and Objectives

### Primary Question
Is there statistical evidence of vintage-related artifacts (level shifts, variance changes, structural breaks) at the Census Bureau methodology transition points in the North Dakota international migration time series?

### Boundaries
- **In Scope**:
  - Quantitative statistical tests at 2009-2010 transition (Vintage 2009 to Vintage 2020)
  - Quantitative statistical tests at 2019-2020 transition (Vintage 2020 to Vintage 2024)
  - Placebo tests at non-transition years for comparison
  - Power analysis and limitations discussion
- **Out of Scope**:
  - Census Bureau methodology documentation review (Agent 1)
  - External validation against other data sources (Agent 3)
  - Correction factor development (Agent 4)
- **Dependencies**: None for this analysis; findings may inform subsequent agents

---

## Methodology

### Data Sources

| Source | Type | Coverage | Location/Citation |
|--------|------|----------|-------------------|
| State Migration Components | Primary | 2000-2024 | `/data/processed/immigration/state_migration_components_2000_2024.csv` |
| North Dakota subset | Derived | 2000-2024, n=25 | STATE=38, 3 vintages |

### Analytical Methods

1. **Level Shift Analysis**: Two-sample t-test, Welch's t-test, Mann-Whitney U test
   - Rationale: Test mean differences at vintage boundaries
   - Parameters: alpha=0.05, two-tailed tests

2. **Variance Homogeneity Tests**: Levene's test (robust), Bartlett's test (assumes normality)
   - Rationale: Detect heteroskedasticity that might indicate methodology changes
   - Parameters: alpha=0.05

3. **Structural Break Tests**: Chow test with linear trend model
   - Rationale: Formal test for parameter instability at known break points
   - Parameters: Break at indices 10 (2009-2010) and 20 (2019-2020)

4. **Placebo Tests**: Chow tests at non-transition years (2005, 2007, 2012, 2015, 2017)
   - Rationale: Compare transition-point statistics to random years
   - Parameters: Same specification as primary Chow tests

5. **Autocorrelation Analysis**: ACF, Ljung-Box test
   - Rationale: Assess time series structure and independence assumptions
   - Parameters: Up to 5 lags for full series, 3 lags for vintages

6. **Trend Analysis**: OLS linear trend by vintage
   - Rationale: Compare trend slopes across vintages
   - Parameters: Simple linear model (intercept + time)

### Limitations

- **Small sample sizes**: n=10 per vintage for first two, n=5 for third. Power to detect moderate effects (d=0.5) is approximately 18%.
- **COVID-19 confound**: The 2019-2020 transition coincides with the pandemic, making effects inseparable.
- **Multiple testing**: No formal correction applied; results should be interpreted holistically.
- **Linear trend assumption**: Chow test assumes linear relationships; non-linear patterns may be missed.

---

## Findings

### Finding 1: Significant Level Shift at 2009-2010 Transition

**Summary**: Mean international migration increased significantly from Vintage 2009 (mean=457 persons) to Vintage 2020 (mean=1,378 persons), a difference of 921 persons. All three tests (t-test, Welch's, Mann-Whitney) reject the null hypothesis of equal means/distributions.

**Evidence**:
- Two-sample t-test: t=-3.49, df=18, p=0.0026
- Welch's t-test: t=-3.49, p=0.0034
- Mann-Whitney U: U=10, p=0.0028
- Cohen's d = 1.56 (large effect)

**Uncertainty**: Medium - While statistically significant, small samples mean true effect size uncertainty is high. The CI for Cohen's d is wide.

**Implications**: This suggests a meaningful change in measured migration at the vintage boundary. However, this could reflect:
1. Census methodology change (artifact)
2. Real demographic shift coinciding with 2010
3. Economic factors (e.g., Bakken oil boom began 2008-2010)

---

### Finding 2: Significant Variance Heterogeneity Across Vintages

**Summary**: Variances differ dramatically across vintages, with Vintage 2024 showing variance 29 times larger than Vintage 2009. Levene's test confirms this is statistically significant.

**Evidence**:
- Variance Vintage 2009: 177,261
- Variance Vintage 2020: 518,245
- Variance Vintage 2024: 5,211,818
- Variance ratio (max/min): 29.4
- Levene's test (all vintages): W=8.49, p=0.0018
- Bartlett's test: chi2=20.1, p<0.0001

**Uncertainty**: Medium - Vintage 2024 has only n=5 observations, making variance estimates unstable. The extreme values (30 in 2020, 5,126 in 2024) drive much of the variance.

**Implications**: Increasing variance over time could indicate:
1. More volatile international migration patterns (real)
2. COVID-era disruption (2020-2024)
3. Methodology changes affecting measurement precision

---

### Finding 3: No Structural Break Detected at 2009-2010 (Chow Test)

**Summary**: The Chow test for structural break at the 2009-2010 transition is NOT significant, suggesting the linear trend relationship does not change detectably at this boundary. This contrasts with the significant level shift finding.

**Evidence**:
- Chow test at 2009-2010: F=0.84, p=0.447 (not significant)
- Chow test at 2019-2020: F=25.75, p<0.001 (significant, but COVID confound)
- Placebo tests show similar F-statistics at non-transition years:
  - 2005: F=0.42, p=0.66
  - 2007: F=0.75, p=0.48
  - 2012: F=0.88, p=0.43
  - 2015: F=1.20, p=0.32
  - 2017: F=2.95, p=0.07

**Uncertainty**: High - The Chow test's power is limited with small samples. The discrepancy between level shift tests (significant) and structural break tests (not significant) may reflect that:
1. The level shift is real but the trend slope is similar before/after
2. The test lacks power to detect the break
3. The simple linear model is misspecified

**Implications**: The first transition point does not stand out from placebo years in structural break testing. This suggests the vintage transition may not introduce a dramatic discontinuity in the trend relationship, even if mean levels differ.

---

### Finding 4: Strong Positive Trend in Vintage 2024 Period

**Summary**: While Vintages 2009 and 2020 show no significant linear trends, Vintage 2024 (2020-2024) exhibits an extremely steep positive trend of 1,401 persons/year.

**Evidence**:
- Vintage 2009 trend: slope=39 persons/year, p=0.43 (not significant)
- Vintage 2020 trend: slope=72 persons/year, p=0.39 (not significant)
- Vintage 2024 trend: slope=1,401 persons/year, p=0.006 (significant, R2=0.94)
- Full series trend: slope=119 persons/year, p<0.001 (significant)

**Uncertainty**: High - Vintage 2024 has only n=5 observations. The steep trend is heavily influenced by COVID suppression in 2020 (value=30) followed by recovery and surge (2024 value=5,126).

**Implications**: The dramatic trend in recent years reflects post-COVID immigration recovery, possibly combined with new immigration patterns. This is likely real signal rather than artifact.

---

### Finding 5: Significant Autocorrelation in Full Series

**Summary**: The full 25-year series shows significant positive autocorrelation at lag 1 (ACF=0.56), indicating that high-migration years tend to follow high-migration years. This violates the independence assumption of standard t-tests.

**Evidence**:
- Ljung-Box at lag 1: Q=8.98, p=0.003
- Ljung-Box at lag 2: Q=10.45, p=0.005
- ACF at lag 1: 0.56 (exceeds critical value 0.39)
- Within-vintage ACF patterns differ: V2009 shows slight negative lag-1 (-0.23), V2020 shows near-zero lag-1 (0.06)

**Uncertainty**: Medium - Small samples within each vintage make ACF estimates unreliable.

**Implications**: The presence of autocorrelation suggests:
1. Standard tests may overstate significance (effective sample size is smaller)
2. The difference in ACF structure across vintages could indicate different data-generating processes
3. Time series methods might be more appropriate than cross-sectional tests

---

## Quantitative Summary

```json
{
  "agent_id": 2,
  "report_date": "2026-01-01",
  "metrics": {
    "level_shift_2009_2010": {
      "value": 921.1,
      "unit": "persons",
      "confidence_interval_95": null,
      "interpretation": "Mean increase at first vintage transition"
    },
    "level_shift_pvalue": {
      "value": 0.0026,
      "unit": "p-value",
      "confidence_interval_95": null,
      "interpretation": "Statistically significant at alpha=0.05"
    },
    "cohens_d_transition_1": {
      "value": 1.56,
      "unit": "standardized effect",
      "confidence_interval_95": null,
      "interpretation": "Large effect size"
    },
    "variance_ratio": {
      "value": 29.4,
      "unit": "ratio",
      "confidence_interval_95": null,
      "interpretation": "Extreme heteroskedasticity across vintages"
    },
    "chow_F_transition_1": {
      "value": 0.84,
      "unit": "F-statistic",
      "confidence_interval_95": null,
      "interpretation": "No structural break detected"
    },
    "chow_p_transition_1": {
      "value": 0.447,
      "unit": "p-value",
      "confidence_interval_95": null,
      "interpretation": "Not significant at alpha=0.05"
    }
  },
  "categorical_findings": {
    "level_shift_detected": {
      "conclusion": "Yes, at 2009-2010",
      "confidence": "Medium",
      "evidence_strength": "Strong (3 tests agree)"
    },
    "structural_break_detected": {
      "conclusion": "No at 2009-2010; Yes at 2019-2020 (confounded)",
      "confidence": "Low",
      "evidence_strength": "Weak (limited power)"
    },
    "variance_homogeneous": {
      "conclusion": "No",
      "confidence": "Medium-High",
      "evidence_strength": "Strong (ratio=29:1)"
    }
  },
  "overall_assessment": {
    "recommendation": "Proceed with caution",
    "confidence_level": "Medium",
    "key_uncertainties": [
      "Small sample sizes limit power",
      "Conflicting test results (level shift vs. structural break)",
      "Cannot separate artifact from real demographic change",
      "COVID confounds 2019-2020 analysis"
    ]
  }
}
```

---

## Uncertainty Quantification

### Epistemic Uncertainty (What We Don't Know)

| Unknown | Impact on Conclusion | Reducible? |
|---------|---------------------|------------|
| True effect size at 2009-2010 transition | High - determines if level shift is practically significant | Partially (more data in future) |
| Whether level shift is artifact or real | High - core question remains unanswered | Yes (Agent 1 methodology review) |
| Appropriate statistical model | Medium - linear trend may be misspecified | Yes (alternative models possible) |

### Aleatory Uncertainty (Inherent Variability)

| Source | Magnitude | Handling |
|--------|-----------|----------|
| Year-to-year migration volatility | High (std=421-2283 by vintage) | Acknowledged in effect size calculations |
| Small sample estimation error | High (n=10, 10, 5) | Noted in all findings; power analysis conducted |

### Sensitivity to Assumptions

| Assumption | If Wrong, Impact | Alternative Interpretation |
|------------|------------------|---------------------------|
| Independence of observations | t-test p-values may be inflated | True significance could be weaker |
| Linear trend model | Chow test may miss non-linear breaks | Vintage transitions could affect non-linear dynamics |
| COVID effect is purely real, not methodological | 2019-2020 analysis remains confounded | May need COVID-specific adjustment |

---

## Areas Flagged for External Review

### Review Request 1: Reconciling Level Shift vs. Structural Break Results

**Question**: How should we interpret the discrepancy between significant t-test results (p=0.003 for level shift) and non-significant Chow test results (p=0.45 for structural break) at the 2009-2010 transition?

**Context**: The t-test compares means directly across periods. The Chow test assesses whether linear regression parameters (intercept and slope) change at the break point. Both analyze the same data.

**Our Tentative Answer**: The level shift (mean increase) may be real, but the trend slope is similar before and after 2010. This would explain why t-test detects a difference but Chow test does not - the series "jumps up" at 2010 but continues with similar growth rate.

**Why External Review Needed**: This interpretation has implications for how to correct for any vintage artifact. If slope is preserved but level shifts, a simple additive adjustment might suffice.

---

### Review Request 2: Validity of Placebo Comparison

**Question**: Is comparing Chow test F-statistics at transition points versus random years a valid way to assess whether transition breaks are "special"?

**Context**: Our placebo tests showed F-statistics at transition points (F=0.84 at 2009-2010) were comparable to non-transition years (F=0.42 to 2.95).

**Our Tentative Answer**: This suggests the 2009-2010 transition is not detectably different from random years in terms of structural breaks. However, the power of this comparison is limited.

**Why External Review Needed**: This informal comparison lacks formal statistical framework. Is there a more rigorous way to test "transition vs. non-transition" differences?

---

### Review Request 3: Implications of Extreme Variance Ratio

**Question**: What does a 29:1 variance ratio across vintages imply for analysis and any potential corrections?

**Context**: Variance increases monotonically: V2009=177k, V2020=518k, V2024=5.2M. This could reflect increasing measurement uncertainty, real volatility, or both.

**Our Tentative Answer**: This heteroskedasticity complicates pooled analyses and may require variance-weighted methods or vintage-specific treatment.

**Why External Review Needed**: We need guidance on whether this variance pattern suggests methodological concerns or expected demographic behavior.

---

## Artifacts Produced

| Artifact | Filename | Format | Purpose |
|----------|----------|--------|---------|
| ND Migration Data | `agent2_nd_migration_data.csv` | CSV | Raw data for replication |
| Test Results | `agent2_test_results.csv` | CSV | Complete statistical test outputs |
| Transition Metrics | `agent2_transition_metrics.json` | JSON | Summary statistics at transitions |
| Findings Summary | `agent2_findings_summary.json` | JSON | Machine-readable conclusions |
| Methods Sources | `agent2_sources.json` | JSON | Statistical method references |
| Calculations | `agent2_calculations.md` | Markdown | Step-by-step verification |
| Time Series Plot | `agent2_fig1_timeseries_with_vintages.png` | PNG | Visualization of data by vintage |
| Variance Plot | `agent2_fig2_variance_by_vintage.png` | PNG | Box plots and variance comparison |
| Structural Breaks | `agent2_fig3_structural_breaks.png` | PNG | Break points and Chow test results |
| ACF Analysis | `agent2_fig4_acf_by_vintage.png` | PNG | Autocorrelation by vintage |

### Artifact Descriptions

#### agent2_nd_migration_data.csv
- **Contents**: North Dakota international migration by year with vintage labels
- **Schema**: year, intl_migration, vintage, vintage_period
- **Usage**: Replicate all analyses; verify data extraction

#### agent2_test_results.csv
- **Contents**: All 25 statistical tests with full output
- **Schema**: test_id, test_name, hypothesis, test_statistic, statistic_name, df, p_value, significance_level, reject_null, effect_size, effect_size_name, interpretation, notes
- **Usage**: Review individual test results; verify calculations

#### agent2_transition_metrics.json
- **Contents**: Descriptive statistics and test summaries by vintage
- **Schema**: Nested JSON with level_shift_analysis, variance_analysis, structural_break_analysis, autocorrelation_analysis, trend_analysis
- **Usage**: Quick reference for key metrics; input to synthesis

---

## Conclusion

### Answer to Primary Question

**Is there statistical evidence of vintage-related artifacts in the ND international migration time series?**

The evidence is **mixed and inconclusive**:

1. **Level shift tests say YES**: Significant mean differences at the 2009-2010 boundary (p=0.003, d=1.56)
2. **Structural break tests say NO**: Chow test at 2009-2010 is not significant (p=0.45) and similar to placebo years
3. **Variance tests say YES**: Significant heteroskedasticity across vintages (p=0.002, ratio=29:1)
4. **Power limitations**: Small samples (n=10 per vintage) severely constrain our ability to detect moderate effects

The most defensible interpretation is that **there may be a level shift at the 2009-2010 vintage boundary, but the evidence is insufficient to definitively attribute it to methodology changes rather than real demographic shifts** (e.g., the oil boom). The structural break test's failure to detect an unusual break at the transition point is noteworthy.

### Confidence Assessment

| Aspect | Confidence | Explanation |
|--------|------------|-------------|
| Data Quality | High | Data extracted correctly from source; verified against original |
| Method Appropriateness | Medium | Standard tests applied, but power is limited |
| Conclusion Robustness | Medium | Multiple tests give mixed signals; interpretation depends on emphasis |
| **Overall** | **Medium** | **Evidence insufficient for definitive conclusion** |

### Implications for Extension Decision

| Option | Supported? | Confidence |
|--------|-----------|------------|
| A: Extend with corrections | Partial | Low - unclear what corrections are needed |
| B: Extend with caveats | Yes | Medium - acknowledge vintage effects in uncertainty |
| C: Hybrid approach | Yes | Medium - use recent vintages with more confidence |
| D: Maintain n=15 | Partial | Low - may be overly conservative |

---

## Sources and References

### Primary Sources Consulted

1. Census Bureau Population Estimates Components of Change (state-level, 2000-2024)

### Data Files Used

1. `/data/processed/immigration/state_migration_components_2000_2024.csv` - Multi-vintage migration components

### Methods References

1. Student (1908). "The probable error of a mean." *Biometrika*, 6(1), 1-25.
2. Welch, B.L. (1947). "The generalization of 'Student's' problem." *Biometrika*, 34(1-2), 28-35.
3. Mann, H.B., Whitney, D.R. (1947). "On a test of whether one of two random variables is stochastically larger than the other." *Annals of Mathematical Statistics*, 18(1), 50-60.
4. Levene, H. (1960). "Robust tests for equality of variances." *Contributions to Probability and Statistics*.
5. Chow, G.C. (1960). "Tests of equality between sets of coefficients in two linear regressions." *Econometrica*, 28(3), 591-605.
6. Ljung, G.M., Box, G.E.P. (1978). "On a measure of lack of fit in time series models." *Biometrika*, 65(2), 297-303.
7. Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.

---

## Appendix: Technical Details

### A.1 Descriptive Statistics by Vintage

| Vintage | Years | n | Mean | Std | Variance | Median | Min | Max |
|---------|-------|---|------|-----|----------|--------|-----|-----|
| 2009 | 2000-2009 | 10 | 456.8 | 421.0 | 177,261 | 528 | -545 | 1,025 |
| 2020 | 2010-2019 | 10 | 1,377.9 | 719.9 | 518,245 | 1,251 | 468 | 2,875 |
| 2024 | 2020-2024 | 5 | 2,633.0 | 2,282.9 | 5,211,818 | 3,287 | 30 | 5,126 |

### A.2 Raw Data

```
Year  IntlMig  Vintage
2000      258    2009
2001      651    2009
2002      264    2009
2003     -545    2009
2004    1,025    2009
2005      535    2009
2006      815    2009
2007      461    2009
2008      583    2009
2009      521    2009
---- TRANSITION ----
2010      468    2020
2011    1,209    2020
2012    1,295    2020
2013    1,254    2020
2014      961    2020
2015    2,247    2020
2016    1,589    2020
2017    2,875    2020
2018    1,247    2020
2019      634    2020
---- TRANSITION ----
2020       30    2024
2021      453    2024
2022    3,287    2024
2023    4,269    2024
2024    5,126    2024
```

### A.3 Power Analysis

For two-sample t-test with n1=n2=10, alpha=0.05, two-tailed:
- Power to detect d=0.8 (large): ~39%
- Power to detect d=0.5 (medium): ~18%
- Power to detect d=0.2 (small): ~7%

Implication: Non-significant results should NOT be interpreted as "no effect."

---

## Revision History

| Date | Version | Author | Changes |
|------|---------|--------|---------|
| 2026-01-01 | 1.0 | Agent 2 | Initial report |

---

*End of Report*
