# External Review Briefing: Census Vintage Methodology Investigation

## For ChatGPT 5.2 Pro Review

---

## Document Information

| Field | Value |
|-------|-------|
| Prepared By | Claude Code (Opus 4.5) Phase A Investigation |
| Date | 2026-01-01 |
| Investigation ID | ADR-019 |
| Review Type | Phase A Synthesis |

---

## 1. Context and Background

### 1.1 The Research Question

We are conducting a statistical analysis of North Dakota's international migration patterns for a peer-reviewed journal article targeting top-tier demographic methods journals (Demography, Population Studies). The core question is:

**Should we extend our time series from n=15 (2010-2024) to n=25 (2000-2024)?**

### 1.2 Why This Matters

The current n=15 series has significant statistical limitations:
- Unit root tests require n >= 25 for reliable asymptotic properties
- VAR/cointegration analysis lacks degrees of freedom (Granger causality tests failed)
- Structural break detection power is constrained
- Wide prediction intervals reduce interpretability
- ARIMA forecasting limited to 5-year horizons

Extending to n=25 would substantially improve statistical power, BUT the extension requires combining data from three different Census Bureau "vintages" with potentially different methodologies.

### 1.3 The Methodological Concern

Census Bureau Population Estimates Program (PEP) data is organized by "vintage" - estimates produced with a particular methodology and benchmark population. The data we would combine:

| Vintage | Years | Benchmark | NIM Estimation Method |
|---------|-------|-----------|----------------------|
| Vintage 2009 | 2000-2009 | 2000 Census | Residual method using decennial census foreign-born counts |
| Vintage 2020 | 2010-2019 | 2010 Census | ROYA (Residence One Year Ago) using ACS survey data |
| Vintage 2024 | 2020-2024 | 2020 Census | ROYA + DHS administrative data adjustment for humanitarian migrants |

**Risk**: If methodologies differ substantially, combining vintages could introduce artifacts that we might misinterpret as real demographic phenomena.

### 1.4 The Data

Here is the complete North Dakota international migration time series (net international migration count):

```
Year  IntlMig  Vintage  Notes
----  -------  -------  -----
2000      258    2009   Pre-Bakken era
2001      651    2009
2002      264    2009
2003     -545    2009   Only negative year in series
2004    1,025    2009
2005      535    2009
2006      815    2009
2007      461    2009
2008      583    2009   Financial crisis year
2009      521    2009   Last year of Vintage 2009
----  -------  -------  ----- METHODOLOGY TRANSITION -----
2010      468    2020   First year of Vintage 2020
2011    1,209    2020   Bakken boom begins
2012    1,295    2020
2013    1,254    2020
2014      961    2020
2015    2,247    2020   Peak Bakken-era migration
2016    1,589    2020
2017    2,875    2020   Travel Ban year
2018    1,247    2020
2019      634    2020   Last year before COVID
----  -------  -------  ----- METHODOLOGY TRANSITION -----
2020       30    2024   COVID pandemic (anomaly)
2021      453    2024   Post-COVID recovery
2022    3,287    2024
2023    4,269    2024
2024    5,126    2024   Highest value in series
```

**Key statistics by vintage:**
- Vintage 2009 (2000-2009): Mean = 456, SD = 419, n = 10
- Vintage 2020 (2010-2019): Mean = 1,378, SD = 711, n = 10
- Vintage 2024 (2020-2024): Mean = 2,633, SD = 2,273, n = 5

---

## 2. Investigation Structure

Three agents conducted Phase A investigations:

1. **Agent 1**: Census Bureau methodology documentation review
2. **Agent 2**: Statistical transition analysis (quantitative tests)
3. **Agent 3**: Cross-vintage comparability assessment

---

## 3. Agent 1 Summary: Methodology Documentation

### Key Findings

1. **Major Methodology Shift at 2010 Transition**: The Census Bureau fundamentally changed its approach to estimating foreign-born immigration. The 2000s used a residual method from decennial census foreign-born counts; the 2010s introduced the ROYA method using ACS survey data. This is a well-documented, intentional change.

2. **Vintage 2024 DHS Adjustment**: Vintage 2024 incorporated a major methodological change - using DHS administrative data to adjust ACS-based estimates upward by approximately 75% of estimated humanitarian migrants. This increased national NIM estimates for 2021-2022 by 700,000 persons over previous vintage.

3. **Census Bureau Explicitly Warns Against Combining Vintages**: Direct quote from Census Bureau documentation: "Data from separate vintages should not be combined."

4. **NIM is the Most Uncertain Component**: Net international migration accounts for approximately 40% of variance in Demographic Analysis estimates despite comprising only 12% of the population under 65.

5. **State Allocation Uncertainty**: Census Bureau acknowledges that state-level allocation of national NIM adjustment "likely does not accurately reflect the distribution of these humanitarian migrants across states."

6. **Small States Face Additional Challenges**: Research indicates "rates for smaller population bases will be unstable" - relevant for North Dakota (~800,000 residents).

### Confidence Level: Medium

### Critical Uncertainties
- Magnitude of methodology-induced level shift at 2009-2010 transition is not quantified by Census Bureau
- Accuracy of state-level allocation for North Dakota specifically
- Whether 2000s residual method systematically over/under-estimated NIM relative to ROYA method

### Attached Artifacts
- `AGENT_1_REPORT.md` - Full report
- `agent1_findings_summary.json` - Machine-readable findings
- `agent1_methodology_matrix.csv` - Side-by-side vintage comparison
- `agent1_census_quotes.json` - 24 direct quotes from Census Bureau documentation
- `agent1_sources.json` - Bibliography with 16 sources

---

## 4. Agent 2 Summary: Statistical Analysis

### Key Findings

1. **Significant Level Shift at 2009-2010 Transition**:
   - Mean increased by 921 persons (456 → 1,378)
   - t-test p-value = 0.003 (significant)
   - Welch's test p-value = 0.003 (significant)
   - Cohen's d = 1.56 (large effect)

2. **Non-Significant Level Shift at 2019-2020 Transition**:
   - Mean increased by 1,255 persons (1,378 → 2,633)
   - t-test p-value = 0.13 (not significant)
   - **Critical caveat**: COVID-19 pandemic confounds this test; cannot separate vintage effect from COVID effect

3. **Significant Variance Heterogeneity**:
   - Variance ratio across vintages: 29:1 (Vintage 2024 variance 29x higher than Vintage 2009)
   - Levene's test p-value = 0.002 (significant)
   - Suggests different data-generating processes across vintages

4. **Non-Significant Structural Break at 2009-2010**:
   - Chow test F = 0.84, p = 0.45 (not significant)
   - Placebo tests at other years show similar or larger F-statistics
   - **Interpretation**: The 2009-2010 transition does not stand out as unusual compared to breaks at other years

5. **Autocorrelation Present**:
   - Overall series ACF lag-1 = 0.56 (p = 0.003, significant)
   - Suggests genuine time series dynamics, not just white noise

### Confidence Level: Low (due to small samples)

### Critical Uncertainties
- Very small samples (n=10, 10, 5 per vintage) severely limit statistical power
- Discrepancy between significant t-test and non-significant Chow test needs interpretation
- Cannot distinguish methodology change from real demographic shifts (Bakken boom began 2011)

### Attached Artifacts
- `AGENT_2_REPORT.md` - Full report
- `agent2_nd_migration_data.csv` - Raw data for replication
- `agent2_test_results.csv` - Complete results for 25 statistical tests
- `agent2_transition_metrics.json` - Key quantitative metrics
- `agent2_calculations.md` - Step-by-step calculations
- `agent2_fig1_timeseries_with_vintages.png` - Time series visualization
- `agent2_fig2_variance_by_vintage.png` - Variance comparison
- `agent2_fig3_structural_breaks.png` - Structural break analysis
- `agent2_fig4_acf_by_vintage.png` - ACF plots by vintage

---

## 5. Agent 3 Summary: Comparability Assessment

### Key Findings

1. **ND-US Correlation Instability**:
   - Vintage 2009 (2000-2009): r = 0.34 (weak, not significant)
   - Vintage 2020 (2010-2019): r = 0.69 (moderate, significant)
   - Vintage 2024 (2020-2024): r = 0.998 (very strong, significant)
   - **Interpretation**: ND's relationship to national patterns changed substantially, but this may reflect Bakken boom economics rather than measurement artifacts

2. **ND Share of US Migration Tripled**:
   - Vintage 2009: Mean 0.052%
   - Vintage 2020: Mean 0.176%
   - **Interpretation**: Likely reflects Bakken oil boom creating genuine pull factors rather than methodology change

3. **Cross-State Pattern Shared**:
   - ND, SD, MT all show similar 160-200% increases in mean international migration between vintages
   - WY (without oil boom) shows only 30% increase
   - **Interpretation**: Shared regional pattern suggests real economic drivers, not measurement artifacts

4. **High ACS Validation**:
   - 93% agreement (13/14 years) between PEP international migration direction and ACS foreign-born population changes
   - Suggests PEP captures genuine patterns

5. **Perfect Internal Consistency**:
   - NETMIG = INTERNATIONALMIG + DOMESTICMIG holds exactly for all 25 observations
   - No accounting discrepancies

### Confidence Level: Medium

### Critical Uncertainties
- Limited sample size within vintage periods
- ACS validation is stock vs. flow comparison (imperfect)
- No direct validation against administrative immigration data for state level

### Attached Artifacts
- `AGENT_3_REPORT.md` - Full report
- `agent3_findings_summary.json` - Machine-readable findings
- `agent3_external_correlations.csv` - ND-US correlations by vintage
- `agent3_state_comparison.csv` - Cross-state transition patterns
- `agent3_validation_data.csv` - PEP vs ACS comparison
- `agent3_coherence_checks.json` - Internal consistency results

---

## 6. Synthesis and Tensions

### 6.1 Areas of Agreement

All three agents converge on:
- **Recommendation**: All recommend "proceed with caution"
- **Methodology changes are real**: Census Bureau documentation confirms distinct methodological eras
- **Extension is feasible**: No agent recommends outright rejection of extension (Option D)
- **Documentation required**: All agree that any extension must document vintage transitions explicitly
- **Sample size limitations**: All acknowledge small samples constrain definitive conclusions

### 6.2 Areas of Disagreement or Tension

| Tension | Description |
|---------|-------------|
| **Institutional vs. Research Use** | Agent 1 emphasizes Census Bureau's explicit warning against combining vintages; Agents 2-3 find patterns suggesting extension may be valid for research purposes |
| **Methodology vs. Bakken Effect** | Agent 2 finds significant level shift at 2009-2010; Agent 3 attributes similar patterns to Bakken oil boom economics. Timing coincidence makes attribution impossible |
| **t-test vs. Chow Test Discrepancy** | Agent 2 found significant t-test (p=0.003) but non-significant Chow test (p=0.45) at same transition. These tests have different null hypotheses but the discrepancy needs interpretation |
| **Correction Feasibility** | Agent 1 does not support Option A (corrections); Agents 2-3 are uncertain about whether corrections are needed or feasible |

### 6.3 Unresolved Questions

1. **Attribution problem**: How much of the 2009-2010 level shift is methodology vs. Bakken boom timing?
2. **Variance heterogeneity implications**: What statistical approaches are appropriate given 29:1 variance ratio?
3. **Research context exception**: Under what conditions might combining vintages be defensible despite Census Bureau guidance?
4. **Effect size interpretation**: Is the d=1.56 level shift practically significant for our modeling purposes?

---

## 7. Specific Review Requests

### 7.1 Reconciling Conflicting Test Results

**Question**: How should we interpret the discrepancy between the significant t-test/Welch's test (p=0.003, d=1.56) and non-significant Chow test (p=0.45) at the 2009-2010 vintage transition?

**Context**: Both tests examine the 2009-2010 boundary. The t-test compares means across vintage periods. The Chow test examines whether a structural break exists in the regression relationship. Agent 2 notes that the Chow F-statistic at 2009-2010 (F=0.84) is actually smaller than placebo tests at several non-transition years.

**Agent Findings**: Agent 2 flagged this as a key uncertainty requiring external review.

**What We Need**: Explanation of whether these tests are measuring different phenomena, and which is more relevant for our decision.

---

### 7.2 Methodological Defensibility of Extension

**Question**: Given the Census Bureau's explicit guidance against combining vintages and the significant level shift detected at 2009-2010, is Option C (hybrid approach with primary analysis on n=15 and sensitivity analyses on n=25) methodologically defensible for publication in a top-tier demography journal?

**Context**: The Census Bureau's guidance is oriented toward general users. Our use case is research where we explicitly document methodology changes and report sensitivity analyses. The journal article will transparently acknowledge vintage transitions.

**Agent Findings**:
- Agent 1: Census explicitly warns against combining; recommends Option B or C
- Agent 2: Statistical evidence inconclusive; recommends sensitivity analyses
- Agent 3: 3/4 comparability indicators positive; recommends Option B or C

**What We Need**: Assessment of whether the weight of evidence supports a defensible path forward, and what documentation/caveats would be required.

---

### 7.3 Handling Variance Heterogeneity

**Question**: Given the 29:1 variance ratio across vintages, what statistical approaches should be used if we proceed with time series extension?

**Context**: Standard time series methods (ARIMA, unit root tests, structural break tests) typically assume homoscedasticity. The variance in Vintage 2024 (2020-2024) is 29 times higher than in Vintage 2009 (2000-2009).

**Agent Findings**: Agent 2 detected significant heteroskedasticity (Levene's p=0.002) but did not investigate correction methods (deferred to potential Phase B).

**What We Need**: Recommended statistical approaches that are robust to heteroskedasticity, or methods to model time-varying variance.

---

### 7.4 Phase B Decision

**Question**: Should we proceed to Phase B (correction methods investigation) or is the evidence sufficient to proceed directly with Option B or C?

**Context**: Phase B would investigate splicing techniques, chain-linking, regime-switching models, and sensitivity analysis frameworks. This would add investigative effort but may or may not yield actionable corrections.

**Agent Findings**: Synthesis recommends "uncertain" on Phase B need. The key question is whether the detected effects warrant correction or can be handled through documentation and sensitivity analyses.

**What We Need**: Your assessment of whether corrections are likely to improve on a simple "document and report sensitivity" approach.

---

## 8. Decision Framework

We need to choose between four options:

### Option A: Extend with Statistical Corrections
- Apply corrections (e.g., vintage dummy variables, splicing adjustments, regime-switching)
- Document all adjustments
- Report sensitivity analyses

**Choose if**: Artifacts are detectable AND correctable, and n=25 benefits outweigh residual risks

### Option B: Extend with Caveats Only
- Use 2000-2024 data as-is
- Document known methodology changes in paper
- Interpret cautiously around transition points

**Choose if**: No significant artifacts detected, or artifacts are small relative to signal of interest

### Option C: Hybrid Approach
- Primary analysis on 2010-2024 (n=15, consistent methodology)
- Robustness checks on 2000-2024 (n=25)
- Report both sets of results

**Choose if**: Uncertain whether artifacts exist, or artifacts are moderate but not correctable

### Option D: Maintain Current Approach
- Stay with n=15 (2010-2024)
- Accept statistical power limitations
- Use small-sample robust methods throughout

**Choose if**: Significant artifacts detected that cannot be reliably corrected, making extension misleading

---

## 9. What We Ask of You

Please provide:

1. **Validation of Agent Analyses**: Are the statistical methods appropriate? Any errors in reasoning?

2. **Assessment of Evidence Strength**: How confident should we be in these findings given sample sizes?

3. **Recommendation**: Which option (A/B/C/D) do the findings support?

4. **Alternative Interpretations**: Are there perspectives the agents missed?

5. **Suggested Additional Analyses**: What else should we examine before deciding?

---

## 10. Attached Materials

### 10.1 Full Agent Reports
- [x] `AGENT_1_REPORT.md` - Census methodology documentation review (27 KB)
- [x] `AGENT_2_REPORT.md` - Statistical transition analysis (22 KB)
- [x] `AGENT_3_REPORT.md` - Cross-vintage comparability assessment (23 KB)

### 10.2 Data Files
- [x] `agent2_nd_migration_data.csv` - Raw ND data (25 observations)
- [x] `agent2_test_results.csv` - Complete test results (25 tests)
- [x] `agent2_transition_metrics.json` - Quantitative transition metrics
- [x] `agent3_external_correlations.csv` - ND-US correlations by period
- [x] `agent3_state_comparison.csv` - Cross-state patterns
- [x] `synthesis_findings_matrix.csv` - Cross-agent synthesis
- [x] `synthesis_recommendations.json` - Aggregate recommendations

### 10.3 Visualizations
- [x] `agent2_fig1_timeseries_with_vintages.png` - Time series with vintage boundaries
- [x] `agent2_fig2_variance_by_vintage.png` - Variance comparison
- [x] `agent2_fig3_structural_breaks.png` - Structural break analysis
- [x] `agent2_fig4_acf_by_vintage.png` - ACF plots

### 10.4 Machine-Readable Summaries
- [x] `agent1_findings_summary.json`
- [x] `agent2_findings_summary.json`
- [x] `agent3_findings_summary.json`
- [x] `agent1_methodology_matrix.csv`
- [x] `agent1_census_quotes.json`

---

## 11. Response Format Request

Please structure your response as:

```markdown
## Executive Assessment
[1-2 paragraph overall assessment of the evidence and situation]

## Option Recommendation
[Which option (A/B/C/D) and detailed rationale]

## Agent Report Validation
### Agent 1 Assessment
[Are methods and reasoning sound? Any concerns?]

### Agent 2 Assessment
[Are statistical tests appropriate? Interpretation correct?]

### Agent 3 Assessment
[Are comparability indicators valid? Any missed considerations?]

## Key Question Responses
### Q1: t-test vs. Chow Test Discrepancy
[Your interpretation]

### Q2: Methodological Defensibility
[Your assessment]

### Q3: Variance Heterogeneity Approaches
[Recommended methods]

### Q4: Phase B Decision
[Proceed or skip?]

## Alternative Interpretations
[Any perspectives the agents missed]

## Remaining Uncertainties
[What we still don't know after this investigation]

## Suggested Next Steps
[Specific recommendations for proceeding]
```

---

## 12. Contact for Clarification

If any information is unclear or additional context is needed, please note the specific question in your response and we will provide supplementary information.

---

*End of Briefing Document*
