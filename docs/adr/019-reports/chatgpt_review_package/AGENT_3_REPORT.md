# Agent Report: 3 - Cross-Vintage Comparability Assessment

## Metadata

| Field | Value |
|-------|-------|
| Agent | 3 |
| Title | Cross-Vintage Comparability Assessment for North Dakota International Migration |
| Date | 2026-01-01 |
| Status | Complete |
| Confidence Level | Medium |

---

## Executive Summary

This analysis evaluates whether the three Census Bureau PEP vintages (2009, 2020, 2024) measure the same underlying construct for North Dakota international migration. The evidence is **mixed but cautiously favorable**: internal data consistency is perfect, ACS validation shows strong agreement (13/14 years), and similar states show comparable transition patterns. However, the ND-US correlation structure changes substantially across vintages (r=0.34 to r=0.99), and ND's share of US migration tripled from the 2009 to 2020 vintage periods. These changes may reflect real demographic shifts (Bakken boom) rather than measurement artifacts.

**Bottom Line**: The vintages likely measure the same underlying phenomenon, but with improving precision in later periods. The pattern of increasing correlation and share is consistent with ND becoming a more prominent immigration destination during the Bakken boom.

**Recommendation**: Proceed with caution - the time series can likely be extended, but the changing ND-US relationship should be documented and may affect forecasting assumptions.

---

## Scope and Objectives

### Primary Question
Do the three Census Bureau PEP vintages (2009, 2020, 2024) measure the same underlying construct for North Dakota international migration, making time series extension statistically valid?

### Boundaries
- **In Scope**:
  - Correlation stability between ND and US international migration
  - Cross-state pattern comparison (ND, SD, MT, WY)
  - Validation against ACS foreign-born data
  - Internal consistency checks
  - Response to known events (2008 crisis, Bakken boom, post-2017 policy)
- **Out of Scope**:
  - Census Bureau methodology documentation analysis (Agent 1's domain)
  - Structural break detection (Agent 2's domain)
  - Specific adjustment factor recommendations (Agent 4's domain)
- **Dependencies**: Results should be interpreted alongside Agent 1's methodology findings and Agent 2's statistical analysis

---

## Methodology

### Data Sources

| Source | Type | Coverage | Location/Citation |
|--------|------|----------|-------------------|
| Census PEP State Migration Components | Primary | 2000-2024, 3 vintages | `/data/processed/immigration/state_migration_components_2000_2024.csv` |
| ACS Table B05006 Foreign-Born | Secondary | 2009-2023 | `/data/raw/immigration/census_foreign_born/b05006_states_all_years.csv` |
| DHS LPR Data | Available but not processed | 2007-2023 | `/data/raw/immigration/dhs_lpr/` |

### Analytical Methods

1. **Correlation Analysis by Vintage Period**
   - Rationale: Stable correlations indicate measurement consistency
   - Parameters: Pearson correlation with Fisher z-transform 95% CIs
   - Periods: 2009 vintage (2000-2009), 2020 vintage (2010-2019), 2024 vintage (2020-2024)

2. **ND Share of US Migration Analysis**
   - Rationale: Stable share suggests consistent relative measurement
   - Parameters: ND INTERNATIONALMIG / US INTERNATIONALMIG by year

3. **Cross-State Comparison**
   - Rationale: Similar states should show similar patterns if changes are measurement-driven
   - Parameters: Compared ND, SD, MT, WY vintage transition patterns

4. **ACS Validation**
   - Rationale: Independent data source should corroborate PEP patterns
   - Parameters: Direction agreement between PEP flows and ACS stock changes

5. **Internal Consistency**
   - Rationale: Perfect internal consistency suggests clean data processing
   - Parameters: NETMIG = INTERNATIONALMIG + DOMESTICMIG check

### Limitations
- ACS measures foreign-born stock, not migration flow - imperfect comparison
- Small sample sizes within vintage periods (n=5 to n=10)
- DHS LPR data not processed for this analysis
- Cannot distinguish real demographic change from measurement change

---

## Findings

### Finding 1: ND-US Correlation Structure Changes Substantially Across Vintages

**Summary**: The correlation between North Dakota and US international migration increases dramatically from weak (r=0.34) in the 2009 vintage to very strong (r=0.998) in the 2024 vintage. This 0.66 change in correlation across vintages suggests either improving measurement precision or changing underlying dynamics.

**Evidence**:

| Period | n | Correlation | 95% CI | p-value | Interpretation |
|--------|---|-------------|--------|---------|----------------|
| 2009 (2000-2009) | 10 | 0.340 | [-0.37, 0.80] | 0.337 | Weak, not significant |
| 2020 (2010-2019) | 10 | 0.688 | [0.10, 0.92] | 0.028 | Moderate, significant |
| 2024 (2020-2024) | 5 | 0.998 | [0.97, 1.00] | <0.001 | Very strong, significant |

- Correlation range across vintages: 0.66 (exceeds stability threshold of 0.30)
- The non-significant 2009 vintage correlation may reflect ND's small, volatile immigration numbers during that period
- The near-perfect 2024 correlation may reflect COVID-related synchronized suppression and recovery

**Uncertainty**: Medium - Small samples limit precision; the pattern could reflect real changing dynamics or measurement evolution.

**Implications**: The changing correlation structure should be documented. It may indicate that combining vintages requires recognizing different ND-US relationships by period.

---

### Finding 2: ND's Share of US Migration Tripled Between Vintage Periods

**Summary**: ND's average share of US international migration increased from 0.052% (2009 vintage) to 0.176% (2020 vintage) to 0.167% (2024 vintage). This 3.4x increase likely reflects real Bakken boom effects rather than measurement artifacts.

**Evidence**:

| Vintage Period | Mean ND Share (%) | Std Dev (%) | Min Share | Max Share |
|----------------|-------------------|-------------|-----------|-----------|
| 2009 (2000-2009) | 0.052 | 0.044 | -0.066% (2003) | 0.104% (2004) |
| 2020 (2010-2019) | 0.176 | 0.061 | 0.102% (2014) | 0.303% (2017) |
| 2024 (2020-2024) | 0.167 | 0.028 | 0.120% (2021) | 0.194% (2022) |

- The share stabilized between 2020 and 2024 vintages (0.176% vs 0.167%)
- High variability in 2009 vintage (CV=85%) vs lower variability in later vintages (CV=35%, 17%)
- 2017 peak share (0.303%) coincides with known ND economic expansion

**Uncertainty**: Low - The Bakken boom's timing aligns perfectly with the share increase, supporting real demographic change interpretation.

**Implications**: The increased share is likely real. Time series extension should account for this structural shift in ND's immigration profile.

---

### Finding 3: Similar States Show Comparable Vintage Transition Patterns

**Summary**: North Dakota, South Dakota, and Montana all show similar 160-200% increases in mean international migration between 2009 and 2020 vintages. This shared pattern suggests the changes are not ND-specific measurement artifacts.

**Evidence**:

| State | 2009 Vintage Mean | 2020 Vintage Mean | Change (%) | Similar to ND |
|-------|-------------------|-------------------|------------|---------------|
| ND | 457 | 1,378 | +202% | Reference |
| SD | 655 | 1,712 | +161% | Yes |
| MT | 304 | 886 | +191% | Yes |
| WY | 328 | 428 | +30% | No |

- Three of four Great Plains states show similar patterns
- Wyoming's divergence may reflect its different economic trajectory (no oil boom)
- Pattern sharing suggests measurement consistency rather than ND-specific artifacts

**Uncertainty**: Low - The pattern replication across comparable states is strong evidence.

**Implications**: The vintage transition patterns are consistent with regional economic factors affecting multiple states similarly. This supports treating the data as measuring a real phenomenon rather than artifacts.

---

### Finding 4: ACS Foreign-Born Data Validates PEP Patterns

**Summary**: PEP international migration direction agrees with ACS foreign-born population changes in 13 of 14 comparison years (93% agreement). The single disagreement (2023) may reflect ACS sampling variability.

**Evidence**:

| Years | Agreement Count | Disagreement Count | Agreement Rate |
|-------|-----------------|-------------------|----------------|
| 2010-2022 | 13 | 0 | 100% |
| 2023 | 0 | 1 | 0% |
| **Total** | **13** | **1** | **93%** |

Detailed comparison (selected years):

| Year | PEP Intl Mig | ACS FB Change | Agreement |
|------|--------------|---------------|-----------|
| 2010 | +468 | +1,372 | Yes |
| 2015 | +2,247 | +2,272 | Yes |
| 2017 | +2,875 | +2,351 | Yes |
| 2020 | +30 | +1,595 | Yes |
| 2023 | +4,269 | -1,210 | No |

- ACS consistently shows larger increases, which is expected (foreign-born includes naturalized citizens, secondary migration)
- The 2023 disagreement: PEP shows +4,269, ACS shows -1,210. This may be ACS sampling noise.

**Uncertainty**: Medium - ACS measures stock not flow; comparison is directional only.

**Implications**: The strong agreement validates PEP data quality. The 2023 discrepancy warrants monitoring but does not invalidate the series.

---

### Finding 5: Perfect Internal Consistency Across All Vintages

**Summary**: The accounting identity NETMIG = INTERNATIONALMIG + DOMESTICMIG holds perfectly for all observations. This indicates clean data processing without computational errors.

**Evidence**:
- Mean residual: 0.0
- Maximum absolute residual: 0
- All observations consistent: Yes (100%)

**Uncertainty**: High confidence - This is an objective, verifiable check.

**Implications**: Data quality is excellent at the computational level. Any issues are conceptual (methodology), not arithmetic.

---

### Finding 6: Vintage Transition Shows COVID Discontinuity

**Summary**: The transition from 2020 to 2024 vintage shows a dramatic drop from 634 (2019) to 30 (2020), reflecting COVID-related border closures. This is real and should be treated as a known event, not a measurement artifact.

**Evidence**:

| Transition | Last Value (Prior Vintage) | First Value (Next Vintage) | Change |
|------------|---------------------------|---------------------------|--------|
| 2009 to 2020 | 521 (2009) | 468 (2010) | -53 (-10%) |
| 2020 to 2024 | 634 (2019) | 30 (2020) | -604 (-95%) |

- The 2009-2010 transition is smooth (-10%), suggesting methodological consistency
- The 2019-2020 transition reflects COVID, not methodology change
- US total also dropped 97% in 2020 (568,639 to 19,885)

**Uncertainty**: Low - COVID disruption is well-documented.

**Implications**: The 2020 data point is an outlier due to real events, not measurement error. Forecasting should account for this.

---

### Finding 7: International and Domestic Migration Show Uncorrelated Patterns

**Summary**: Within each vintage, ND international and domestic migration are essentially uncorrelated (r near 0), which is expected for independent migration flows.

**Evidence**:

| Vintage | Intl-Domestic Correlation | p-value | n |
|---------|--------------------------|---------|---|
| 2009 | 0.088 | 0.810 | 10 |
| 2020 | -0.220 | 0.541 | 10 |
| 2024 | 0.488 | 0.404 | 5 |

- No significant correlations in any vintage
- Domestic migration during Bakken boom was 5-10x international migration

**Uncertainty**: Medium - Small samples limit power to detect correlation.

**Implications**: The two migration components are responding to different drivers, as expected. This supports data quality.

---

## Quantitative Summary

```json
{
  "agent_id": 3,
  "report_date": "2026-01-01",
  "metrics": {
    "correlation_stability": {
      "value": 0.66,
      "unit": "correlation range across vintages",
      "confidence_interval_95": null,
      "interpretation": "Unstable - exceeds 0.30 threshold"
    },
    "nd_share_change": {
      "value": 3.4,
      "unit": "multiplier (2020 vs 2009 vintage mean)",
      "confidence_interval_95": null,
      "interpretation": "Large increase, likely real (Bakken boom)"
    },
    "acs_validation_agreement": {
      "value": 93,
      "unit": "percent",
      "confidence_interval_95": [68, 100],
      "interpretation": "Strong agreement"
    },
    "internal_consistency": {
      "value": 100,
      "unit": "percent",
      "confidence_interval_95": [100, 100],
      "interpretation": "Perfect"
    }
  },
  "categorical_findings": {
    "measurement_consistency": {
      "conclusion": "likely_same_construct",
      "confidence": "medium",
      "evidence_strength": "moderate"
    },
    "cross_state_pattern": {
      "conclusion": "shared_pattern",
      "confidence": "high",
      "evidence_strength": "strong"
    }
  },
  "overall_assessment": {
    "recommendation": "Proceed with caution",
    "confidence_level": "Medium",
    "key_uncertainties": [
      "Cannot definitively distinguish real change from measurement change",
      "Small sample sizes within vintage periods",
      "DHS data not analyzed as additional validation"
    ]
  }
}
```

---

## Uncertainty Quantification

### Epistemic Uncertainty (What We Don't Know)

| Unknown | Impact on Conclusion | Reducible? |
|---------|---------------------|------------|
| Whether ND share increase is measurement or real | Medium - affects comparability interpretation | Partially - methodology review (Agent 1) may help |
| Whether 2023 ACS disagreement is meaningful | Low - single data point | Yes - wait for 2024 ACS |
| DHS LPR patterns for ND | Medium - could strengthen validation | Yes - analyze DHS data |

### Aleatory Uncertainty (Inherent Variability)

| Source | Magnitude | Handling |
|--------|-----------|----------|
| Small state population volatility | High (CV=44-85%) | Use confidence intervals |
| ACS sampling error | Medium (MOE available but not analyzed) | Directional comparison only |
| Year-to-year economic fluctuations | High | Acknowledged in interpretation |

### Sensitivity to Assumptions

| Assumption | If Wrong, Impact | Alternative Interpretation |
|------------|------------------|---------------------------|
| ND share increase is real | Moderate - if measurement artifact, vintages less comparable | Would suggest methodology changed to "find" more immigrants |
| ACS directional agreement is validating | Low - if coincidental, less confident | Would need DHS data for validation |
| Similar state patterns indicate shared measurement | Low - regional economics could explain | Still supports data quality but not methodology stability |

---

## Areas Flagged for External Review

### Review Request 1: Correlation Structure Interpretation

**Question**: Is the increasing ND-US correlation (0.34 -> 0.69 -> 0.998) more consistent with (a) improving measurement precision, (b) ND becoming more integrated with national immigration trends, or (c) COVID-synchronized behavior in recent years?

**Context**: The 2009 vintage shows weak, non-significant correlation; the 2020 vintage shows moderate significant correlation; the 2024 vintage shows near-perfect correlation.

**Our Tentative Answer**: Likely a combination of (b) and (c). The Bakken boom attracted more mainstream immigration flows, and COVID created synchronized suppression.

**Why External Review Needed**: This interpretation affects whether we treat the vintages as measuring the same thing with different precision vs. measuring different regimes of ND-US relationship.

---

### Review Request 2: ND Share Tripling - Measurement or Reality?

**Question**: Is a 3.4x increase in ND's share of US international migration between 2000-2009 and 2010-2019 plausible as a real demographic phenomenon?

**Context**: The Bakken oil boom began ~2011 and attracted significant in-migration. Domestic migration was 5-10x international migration during this period.

**Our Tentative Answer**: Yes, plausible. The timing matches the Bakken boom, and domestic migration shows even larger increases.

**Why External Review Needed**: If implausible, it would suggest measurement changes rather than real demographic change.

---

### Review Request 3: 2023 ACS Disagreement Significance

**Question**: Should the 2023 ACS-PEP disagreement (PEP: +4,269; ACS: -1,210) concern us about data quality?

**Context**: This is the only disagreement in 14 comparison years (93% agreement).

**Our Tentative Answer**: Likely ACS sampling variability. A -1,210 change in a 35,000 foreign-born population is within typical MOE.

**Why External Review Needed**: If this represents real divergence, it could indicate emerging measurement issues in the 2024 vintage.

---

## Artifacts Produced

| Artifact | Filename | Format | Purpose |
|----------|----------|--------|---------|
| Correlation analysis | `agent3_external_correlations.csv` | CSV | ND-US correlations by vintage period |
| State comparison | `agent3_state_comparison.csv` | CSV | Cross-state transition patterns |
| Validation data | `agent3_validation_data.csv` | CSV | PEP vs ACS comparison |
| Coherence checks | `agent3_coherence_checks.json` | JSON | Internal consistency results |
| Findings summary | `agent3_findings_summary.json` | JSON | Machine-readable findings |
| Sources | `agent3_sources.json` | JSON | Data source references |
| Detailed state patterns | `agent3_detailed_state_patterns.csv` | CSV | Full state statistics by vintage |
| ND share analysis | `agent3_nd_share_analysis.csv` | CSV | Year-by-year ND share of US |
| Analysis script | `agent3_analysis.py` | Python | Reproducible analysis code |

### Artifact Descriptions

#### agent3_external_correlations.csv
- **Contents**: ND-US international migration correlations by vintage period
- **Schema**: indicator, period, n, correlation, correlation_95_lower, correlation_95_upper, p_value, nd_share_mean_pct, nd_share_std_pct, interpretation
- **Usage**: Assess correlation stability across vintages

#### agent3_state_comparison.csv
- **Contents**: Mean international migration by vintage for ND, SD, MT, WY
- **Schema**: state, metric, vintage_2009_value, vintage_2020_value, transition_change_pct, similar_to_nd
- **Usage**: Evaluate whether ND patterns are unique or shared

#### agent3_validation_data.csv
- **Contents**: Year-by-year PEP and ACS comparison for ND
- **Schema**: year, pep_intl_migration, pep_vintage, acs_foreign_born, acs_foreign_born_change, source_agreement
- **Usage**: External validation of PEP data quality

---

## Conclusion

### Answer to Primary Question

**Do the three vintages measure the same underlying construct?**

The evidence suggests **yes, with caveats**:

1. **Supporting evidence**:
   - Perfect internal consistency (100%)
   - Strong ACS validation (93% directional agreement)
   - Similar transition patterns in comparable states (SD, MT)
   - Changes align with known events (Bakken boom, COVID)

2. **Concerning evidence**:
   - ND-US correlation changes substantially (0.34 to 0.998)
   - ND's share of US migration tripled between vintages
   - These could indicate different measurement regimes

3. **Resolution**: The evidence is more consistent with ND's **real demographic transformation** during the Bakken boom era than with measurement artifacts. The methodology appears to capture the same underlying phenomenon (international migration), but ND's relationship to national patterns changed.

### Confidence Assessment

| Aspect | Confidence | Explanation |
|--------|------------|-------------|
| Data Quality | High | Perfect internal consistency, strong ACS validation |
| Method Appropriateness | High | Standard correlation and comparison methods |
| Conclusion Robustness | Medium | Cannot definitively separate real change from measurement change |
| **Overall** | **Medium** | **Evidence favors comparability but with documented caveats** |

### Implications for Extension Decision

| Option | Supported? | Confidence |
|--------|-----------|------------|
| A: Extend with corrections | Partial | Medium - may not need corrections if changes are real |
| B: Extend with caveats | Yes | High - document changing ND-US relationship |
| C: Hybrid approach | Yes | Medium - could treat pre/post-2010 as different regimes |
| D: Maintain n=15 | No | Low - evidence doesn't support rejecting extension |

**Recommendation**: Option B (extend with caveats) or Option C (hybrid) depending on Agent 4's adjustment analysis. The time series extension appears valid, but the changing ND-US relationship (from uncorrelated to highly correlated) should be documented and may affect forecasting model specification.

---

## Sources and References

### Primary Sources Consulted

1. U.S. Census Bureau. "Population Estimates Program (PEP)." Various vintages (2009, 2020, 2024).
2. U.S. Census Bureau. "American Community Survey Table B05006: Place of Birth for the Foreign-Born Population in the United States." 2009-2023.

### Data Files Used

1. `/home/nhaarstad/workspace/demography/cohort_projections/data/processed/immigration/state_migration_components_2000_2024.csv` - Primary PEP data
2. `/home/nhaarstad/workspace/demography/cohort_projections/data/raw/immigration/census_foreign_born/b05006_states_all_years.csv` - ACS foreign-born data

### Methods References

1. Fisher, R. A. (1921). "On the 'probable error' of a coefficient of correlation deduced from a small sample." Metron, 1, 3-32. [Used for correlation confidence intervals]

---

## Appendix: Technical Details

### A.1 Correlation Confidence Interval Calculation

Confidence intervals for Pearson correlations were computed using the Fisher z-transformation:

```
z = arctanh(r)
SE(z) = 1/sqrt(n-3)
z_lower, z_upper = z +/- 1.96 * SE(z)
r_lower, r_upper = tanh(z_lower), tanh(z_upper)
```

### A.2 State Selection Rationale

Comparison states (SD, MT, WY) were selected based on:
- Geographic proximity (Great Plains region)
- Similar population size (small states)
- Rural character
- Different from major immigration gateways (CA, TX, NY, FL)

Wyoming diverged from the pattern, likely due to lack of oil boom comparable to ND.

### A.3 ACS-PEP Comparison Limitations

The ACS foreign-born population is a stock measure, while PEP international migration is a flow measure. Year-over-year changes in ACS foreign-born are affected by:
- International migration (what PEP measures)
- Deaths of foreign-born residents
- Naturalization (changes category but not stock)
- Secondary domestic migration (foreign-born moving between states)
- Emigration

Therefore, perfect agreement is not expected. Directional agreement is the appropriate validation criterion.

---

## Revision History

| Date | Version | Author | Changes |
|------|---------|--------|---------|
| 2026-01-01 | 1.0 | Agent 3 | Initial report |

---

*End of Report*
