# Agent Report: 1 - Census Bureau Methodology Documentation Review

## Metadata

| Field | Value |
|-------|-------|
| Agent | 1 |
| Title | Census Bureau Population Estimates Program (PEP) Methodology Review: Net International Migration Across Vintages |
| Date | 2026-01-01 |
| Status | Complete |
| Confidence Level | Medium |

---

## Executive Summary

The Census Bureau has fundamentally changed how it estimates Net International Migration (NIM) at least three times over the period 2000-2024, with major methodology shifts occurring at each decennial census transition and additional refinements within the 2020s decade. The 1990s-2000s methodology relied heavily on residual methods using decennial census foreign-born counts, while the 2010s introduced the Residence One Year Ago (ROYA) method using ACS data, and Vintage 2024 added DHS administrative data adjustments for humanitarian migrants. The Census Bureau explicitly warns against combining data across vintages and acknowledges that NIM represents the largest source of uncertainty in population estimates.

**Bottom Line**: Census Bureau methodology for estimating state-level NIM has changed substantially between the periods covered by Vintage 2009 (2000-2009) and Vintage 2020/2024 (2010-2024), with different data sources, estimation techniques, and allocation methods creating fundamental comparability concerns.

**Recommendation**: Proceed with caution - extending the time series from n=15 to n=25 requires explicit acknowledgment of methodological discontinuities at vintage transitions.

---

## Scope and Objectives

### Primary Question
Did the Census Bureau fundamentally change how they estimate state-level international migration between vintages, and does this create comparability issues for time series analysis?

### Boundaries
- **In Scope**:
  - Census Bureau methodology documentation for NIM estimation
  - Data sources used in Vintages 2009, 2020, 2021, 2022, 2023, and 2024
  - Census Bureau's own caveats about data limitations
  - State-level allocation methodology
  - Academic literature on PEP accuracy

- **Out of Scope**:
  - Statistical analysis of actual North Dakota NIM data (Agent 2's task)
  - Comparison with external data sources for validation (Agent 3's task)
  - Specific adjustment methodology proposals

- **Dependencies**: None - this is the foundational methodology review

---

## Methodology

### Data Sources

| Source | Type | Coverage | Location/Citation |
|--------|------|----------|-------------------|
| Census Bureau Random Samplings Blog | Primary | 2024 methodology changes | https://www.census.gov/newsroom/blogs/random-samplings/2024/12/international-migration-population-estimates.html |
| Census Bureau Working Paper POP-twps0051 | Primary | 1990-2000 NIM methodology | https://www.census.gov/library/working-papers/2001/demo/POP-twps0051.html |
| Census Bureau Working Paper POP-twps0097 | Primary | 2010 Demographic Analysis NIM | https://www.census.gov/library/working-papers/2013/demo/POP-twps0097.html |
| Census Bureau 2000-2010 Intercensal Methodology | Primary | 2000s decade methods | https://www2.census.gov/programs-surveys/popest/technical-documentation/methodology/intercensal/2000-2010-intercensal-estimates-methodology.pdf |
| Census Bureau PEP Methodology Page | Primary | Current methodology | https://www.census.gov/programs-surveys/popest/technical-documentation/methodology.html |
| Cornell PAD Vintage 2024 Analysis | Secondary | State-level impacts | https://pad.human.cornell.edu/papers/downloads/StateEstimatesV2024Memo.pdf |

### Analytical Methods

1. **Document Review**: Systematic review of Census Bureau technical documentation, working papers, and methodology statements
   - Rationale: Establish authoritative understanding of official methodology
   - Parameters: Focused on NIM components and state-level allocation

2. **Temporal Comparison**: Cross-vintage comparison of documented methodologies
   - Rationale: Identify changes that could affect time series comparability
   - Parameters: Compared 2000s decade vs. 2010s decade vs. 2020s decade

3. **Literature Review**: Search for peer-reviewed assessments of PEP accuracy
   - Rationale: Identify independent evaluations of methodology limitations
   - Parameters: Google Scholar and academic journal searches

### Limitations
- Many Census Bureau PDFs were not fully extractable; relied on summaries and metadata
- Some older methodology documentation (pre-2000) was difficult to locate
- Limited peer-reviewed literature specifically evaluating state-level NIM accuracy

---

## Findings

### Finding 1: Major Methodology Shift at 2010 Transition - Introduction of ROYA Method

**Summary**: The Census Bureau fundamentally changed its approach to estimating foreign-born immigration between the 1990s-2000s and the 2010s. The 1990s-2000s methodology relied primarily on residual techniques using decennial census foreign-born counts, while the 2010s introduced the Residence One Year Ago (ROYA) method using American Community Survey data.

**Evidence**:
- Census Bureau Working Paper POP-twps0051 (2001): "Net international migration is the most complex determinant of population change, and its complete measurement involves the measurement of several sub-components, each one using separate methodology."
- The ACS was fully implemented only in 2005, meaning pre-2005 estimates could not use the ROYA method
- "Over the past decade [as of 2010], the Census Bureau undertook a major initiative to improve its ability to measure net international migration. The implementation of the American Community Survey (ACS) provides critical demographic information between decennial censuses that was not available in previous decades."
- The ROYA method "is considered an improvement over previously used measures of immigration"

**Uncertainty**: Medium - The transition is well-documented, but the exact magnitude of impact on state-level estimates is not quantified

**Implications**: Data from 2000-2009 was estimated using fundamentally different methods than data from 2010 onwards. This creates a structural break at the vintage transition that is independent of real-world migration patterns.

---

### Finding 2: Vintage 2024 Introduced DHS Administrative Data Adjustment for Humanitarian Migrants

**Summary**: The Vintage 2024 release incorporated a major methodological change: using Department of Homeland Security administrative data to adjust ACS-based estimates upward by approximately 75% of estimated humanitarian migrants. This changed estimates retroactively back to 2021.

**Evidence**:
- "To develop the Vintage 2024 population estimates and to better reflect current trends and potentially underrepresented populations, the Census Bureau used newly available administrative data to adjust the usually survey-based estimates of foreign-born immigration."
- "Data from the Department of Homeland Security (DHS) is used to make an estimate of the immigration of humanitarian migrants."
- "For July 1, 2021, to June 30, 2022, the bureau's Vintage 2024 estimate for net international migration was almost 1.7 million, an increase of 700,000 over the previous Vintage 2023 (and 2022) estimates for that year."
- Adjustment was "75% of the humanitarian migrants in their Benchmark Database"

**Uncertainty**: Low - This change is explicitly documented and quantified

**Implications**: Even within the 2020s decade, methodological changes create non-comparability between vintages. The Census Bureau explicitly warns: "Due to periodic methodological updates, such as this year's immigration adjustment, year-to-year comparisons in the estimates should only be done within the same vintage."

---

### Finding 3: State-Level Allocation Method May Not Accurately Reflect Humanitarian Migrant Distribution

**Summary**: The Census Bureau acknowledges that its method for distributing national NIM estimates to states may not accurately capture where humanitarian migrants are actually settling. The national adjustment was distributed using ACS patterns, but the ACS undersamples the populations being adjusted for.

**Evidence**:
- "The adjustment for humanitarian migrants was applied to the national total and then distributed down to states and counties using the Census Bureau's usual method."
- "Since this adjustment was made to account for the under-sampling of humanitarian migrants in the ACS, this approach likely does not accurately reflect the distribution of these humanitarian migrants across states."
- "The Census Bureau is currently researching the best way to distribute this nationwide international migration adjustment to states and counties."
- "The ACS is used to distribute recent immigrants but is not keeping up with changing patterns. Research is ongoing to improve allocation of new immigrants to the states and counties."

**Uncertainty**: High - The Census Bureau acknowledges uncertainty but does not quantify potential error at the state level

**Implications**: State-level NIM estimates in Vintage 2024 carry acknowledged uncertainty in their geographic allocation. This is particularly concerning for states like North Dakota that receive proportionally fewer immigrants than gateway states but may have unique settlement patterns.

---

### Finding 4: Census Bureau Explicitly Warns Against Combining Vintages

**Summary**: The Census Bureau explicitly states that data from different vintages should not be combined and that time series comparisons should only be made within a single vintage.

**Evidence**:
- "The release of a new vintage supersedes any previous estimates series because the new series incorporates the most up-to-date data and methodological improvements."
- "With each new release of annual estimates, the entire time series of estimates is revised for all years back to the last census. All previously published estimates (e.g., old vintages) are superseded and archived."
- "Data from separate vintages should not be combined."
- "Due to periodic methodological updates, such as this year's immigration adjustment, year-to-year comparisons in the estimates should only be done within the same vintage."

**Uncertainty**: Low - This guidance is explicit and unambiguous

**Implications**: The proposed extension from n=15 to n=25 would require combining Vintage 2009 (2000-2009) with Vintage 2020/2024 (2010-2024), which the Census Bureau explicitly advises against.

---

### Finding 5: NIM Represents the Largest Source of Uncertainty in Population Estimates

**Summary**: The Census Bureau acknowledges that net international migration is the most uncertain component of population change estimates, representing approximately 40% of the variance in Demographic Analysis estimates.

**Evidence**:
- "Although the NIM component comprises approximately twelve percent of the total DA resident population under age 65, it represents a large portion of the uncertainty in the DA estimates."
- "The largest share of uncertainty in the range of Demographic Analysis estimates, nearly 40%, comes from estimates of the foreign-born population."
- "Migration is hard to estimate because there are not good administrative records like the vital records from birth and death certificates. A person is only born once or dies once, but people can migrate multiple times throughout their lifetime."
- "International migration is difficult to estimate because of its complexity and dynamic nature."

**Uncertainty**: Low - This is well-documented in Census Bureau publications

**Implications**: NIM estimates carry inherent uncertainty that compounds with methodological changes across vintages.

---

### Finding 6: Small States Face Additional Accuracy Challenges

**Summary**: Research indicates that population projections and estimates are less accurate for smaller population bases due to rate instability in small cells.

**Evidence**:
- "Previous studies indicate that the population size affects the accuracy of population projections. This is mainly due to the relationship between 'true demographic rates' (fertility, mortality, migration rates) and the population size."
- "Since the detailed demographic rates for small states will be likely to have many small numbers in each cell or many empty cells, these rates for smaller population bases will be unstable."
- State-level ACS estimates have sampling error that is proportionally larger for small states
- North Dakota's total population (~800,000) is among the smallest in the nation

**Uncertainty**: Medium - General principle is established, but specific impact on North Dakota NIM estimates is not quantified

**Implications**: North Dakota's small population size may amplify uncertainty in NIM estimates relative to larger states.

---

## Quantitative Summary

```json
{
  "agent_id": 1,
  "report_date": "2026-01-01",
  "metrics": {
    "methodology_transitions_identified": {
      "value": 3,
      "unit": "distinct methodology eras",
      "confidence_interval_95": null,
      "interpretation": "2000s residual method, 2010s ROYA method, 2020s ROYA+DHS method"
    },
    "vintage_2024_nim_revision_magnitude": {
      "value": 700000,
      "unit": "persons (national)",
      "confidence_interval_95": null,
      "interpretation": "Increase over previous vintage for 2021-2022 period"
    },
    "nim_share_of_da_uncertainty": {
      "value": 40,
      "unit": "percent",
      "confidence_interval_95": null,
      "interpretation": "NIM accounts for ~40% of variance in DA estimates"
    }
  },
  "categorical_findings": {
    "methodology_changed_between_decades": {
      "conclusion": "Yes - fundamental changes in data sources and methods",
      "confidence": "high",
      "evidence_strength": "strong"
    },
    "census_recommends_vintage_combination": {
      "conclusion": "No - explicitly warns against combining vintages",
      "confidence": "high",
      "evidence_strength": "strong"
    },
    "state_allocation_methodology_stable": {
      "conclusion": "No - allocation methods have evolved and carry acknowledged uncertainty",
      "confidence": "medium",
      "evidence_strength": "moderate"
    }
  },
  "overall_assessment": {
    "recommendation": "proceed_with_caution",
    "confidence_level": "medium",
    "key_uncertainties": [
      "Magnitude of methodology-induced level shift at 2009-2010 transition",
      "Accuracy of state-level allocation of national NIM adjustment",
      "Small-state estimation error specific to North Dakota"
    ]
  }
}
```

---

## Uncertainty Quantification

### Epistemic Uncertainty (What We Don't Know)

| Unknown | Impact on Conclusion | Reducible? |
|---------|---------------------|------------|
| Exact magnitude of methodology-induced level shift at 2009-2010 | High | Partially - could estimate via comparison with external data |
| State-level allocation error for North Dakota specifically | High | No - Census Bureau is still researching this |
| How small-state effects interact with methodology changes | Medium | Partially - could compare ND to similar states |
| Whether 2000s residual method systematically over/under-estimated NIM | Medium | Partially - retrospective analysis with intercensal data |

### Aleatory Uncertainty (Inherent Variability)

| Source | Magnitude | Handling |
|--------|-----------|----------|
| ACS sampling error for state-level foreign-born estimates | Unknown for ND specifically, but proportionally larger for small states | Census Bureau provides margins of error for ACS estimates |
| Year-to-year variation in actual migration flows | Substantial - real migration is volatile | Not addressable through methodology |
| Emigration estimation uncertainty | High - no administrative data on departures | Residual estimation inherently uncertain |

### Sensitivity to Assumptions

| Assumption | If Wrong, Impact | Alternative Interpretation |
|------------|------------------|---------------------------|
| 2000s residual method produced unbiased estimates | Level shift at 2010 could be artifact not real | Time series trend interpretation would change |
| ACS captures representative sample of recent immigrants | State allocation could be systematically biased | Some states over/under-allocated NIM |
| DHS humanitarian adjustment (75%) is correct | National total could be too high or low | Affects all state estimates proportionally |

---

## Areas Flagged for External Review

### Review Request 1: Quantifying Methodology-Induced Level Shifts

**Question**: What statistical approaches are appropriate for detecting and potentially adjusting for methodology-induced level shifts in a time series with only two known transition points (2009-2010 and 2019-2020) and no overlap period?

**Context**: The Census Bureau provides no overlapping estimates using both old and new methodologies. We have 10 years pre-2010 and 15 years post-2010, with a known methodology change at the boundary.

**Our Tentative Answer**: Structural break tests could detect if there is a statistically significant level shift, but cannot distinguish methodology artifact from real change in migration patterns.

**Why External Review Needed**: This is a statistical methodology question where domain expertise in time series econometrics would be valuable.

---

### Review Request 2: Validity of Intercensal Estimate Extension

**Question**: Given that the Census Bureau explicitly warns against combining vintages, under what conditions (if any) might it be methodologically defensible to create an extended time series for research purposes?

**Context**: Our goal is to extend a migration time series for demographic projection modeling. The Census Bureau's guidance is oriented toward general users, not researchers who may apply adjustments.

**Our Tentative Answer**: Extension might be defensible if: (1) analysis accounts for methodology change as a potential source of variance, (2) sensitivity analysis tests conclusions with and without pre-2010 data, (3) results are clearly caveated.

**Why External Review Needed**: Methodological expertise in demographic time series and standards in the field.

---

### Review Request 3: Small-State Reliability Assessment

**Question**: Are there established standards or guidelines in demography for minimum population thresholds below which migration estimates become unreliable for time series modeling?

**Context**: North Dakota has approximately 800,000 residents and receives perhaps 1,000-3,000 net international migrants annually. The signal-to-noise ratio in these estimates is unclear.

**Our Tentative Answer**: We could not find specific guidance on minimum thresholds for NIM reliability at the state level.

**Why External Review Needed**: This requires knowledge of demographic standards that may not be in publicly available Census Bureau documentation.

---

## Artifacts Produced

| Artifact | Filename | Format | Purpose |
|----------|----------|--------|---------|
| Findings Summary | `agent1_findings_summary.json` | JSON | Machine-readable summary for synthesis |
| Sources Bibliography | `agent1_sources.json` | JSON | Complete citations with key excerpts |
| Methodology Matrix | `agent1_methodology_matrix.csv` | CSV | Side-by-side vintage comparison |
| Census Quotes | `agent1_census_quotes.json` | JSON | Direct quotes supporting findings |
| Data Sources Timeline | `agent1_data_sources_timeline.csv` | CSV | When each data source was used |

### Artifact Descriptions

#### agent1_findings_summary.json
- **Contents**: JSON representation of key findings with confidence levels
- **Schema**: Follows ARTIFACT_SPECIFICATIONS.md findings summary schema
- **Usage**: Input for synthesis phase and ChatGPT 5.2 Pro review

#### agent1_sources.json
- **Contents**: Complete bibliography with source metadata and key excerpts
- **Schema**: Follows ARTIFACT_SPECIFICATIONS.md sources schema
- **Usage**: Verification of citations and further research

#### agent1_methodology_matrix.csv
- **Contents**: Comparison of methodology across Vintage 2009, 2020, and 2024
- **Schema**: aspect, vintage_2009, vintage_2020, vintage_2024, difference_severity, impact_on_comparability, source
- **Usage**: Quick reference for methodology differences

#### agent1_census_quotes.json
- **Contents**: Direct quotes from Census Bureau documentation with context
- **Schema**: id, source_id, quote, context, relevance, finding_supported
- **Usage**: Authoritative evidence for findings

#### agent1_data_sources_timeline.csv
- **Contents**: Timeline of when each administrative data source was incorporated
- **Schema**: data_source, start_year, end_year, vintage_used, notes
- **Usage**: Understanding data input evolution over time

---

## Conclusion

### Answer to Primary Question

Yes, the Census Bureau fundamentally changed how they estimate state-level international migration between vintages. Key changes include:

1. **Data Sources**: Shifted from residual methods using decennial census foreign-born counts (2000s) to ROYA method using ACS data (2010s) to ROYA plus DHS administrative data adjustment (2024)

2. **Estimation Technique**: Moved from residual estimation to direct measurement via survey residence question, with recent additions of administrative data corrections

3. **State Allocation**: Methods for distributing national NIM to states have evolved and carry acknowledged uncertainty, particularly for humanitarian migrants in Vintage 2024

4. **Census Bureau Guidance**: The Bureau explicitly warns against combining data from different vintages and recommends time series comparisons only within a single vintage

### Confidence Assessment

| Aspect | Confidence | Explanation |
|--------|------------|-------------|
| Data Quality | Medium | Census Bureau documentation is authoritative but some older documents were inaccessible |
| Method Appropriateness | High | Standard document review and comparison approach |
| Conclusion Robustness | Medium-High | Core finding about methodology changes is well-supported; quantitative impacts less certain |
| **Overall** | **Medium** | **Key finding confirmed but magnitude of impact uncertain** |

### Implications for Extension Decision

| Option | Supported? | Confidence |
|--------|-----------|------------|
| A: Extend with corrections | Partial | Low - no validated correction method available |
| B: Extend with caveats | Yes | Medium - explicit acknowledgment of limitations |
| C: Hybrid approach | Partial | Medium - depends on approach details |
| D: Maintain n=15 | Yes | High - consistent with Census Bureau guidance |

---

## Sources and References

### Primary Sources Consulted

1. U.S. Census Bureau. "Census Bureau Improves Methodology to Better Estimate Increase in Net International Migration." Random Samplings Blog, December 19, 2024. https://www.census.gov/newsroom/blogs/random-samplings/2024/12/international-migration-population-estimates.html

2. U.S. Census Bureau. "U.S. Census Bureau Measurement of Net International Migration to the United States: 1990 to 2000." Working Paper No. 51, December 2001. https://www.census.gov/library/working-papers/2001/demo/POP-twps0051.html

3. U.S. Census Bureau. "Estimating Net International Migration for 2010 Demographic Analysis: An Overview of Methods and Results." Working Paper No. 97, February 2013. https://www.census.gov/library/working-papers/2013/demo/POP-twps0097.html

4. U.S. Census Bureau. "Plans for Producing Estimates of Net International Migration for the 2010 Demographic Analysis Estimates." Working Paper No. 90, 2011. https://www.census.gov/library/working-papers/2011/demo/POP-twps0090.html

5. U.S. Census Bureau. "The Census Bureau Approach for Allocating International Migration to States, Counties, and Places: 1981-1991." Working Paper No. 1, October 1992. https://www.census.gov/library/working-papers/1992/demo/POP-twps0001.html

6. U.S. Census Bureau. "2000-2010 Intercensal Estimates Methodology." Technical Documentation. https://www2.census.gov/programs-surveys/popest/technical-documentation/methodology/intercensal/2000-2010-intercensal-estimates-methodology.pdf

7. U.S. Census Bureau. "Population Estimates Program Methodology." https://www.census.gov/programs-surveys/popest/technical-documentation/methodology.html

8. U.S. Census Bureau. "New Population Estimates Show COVID-19 Pandemic Significantly Disrupted Migration Across Borders." December 2021. https://www.census.gov/library/stories/2021/12/net-international-migration-at-lowest-levels-in-decades.html

9. Cornell Program on Applied Demographics. "Vintage 2024 New York State Estimates." December 2024. https://pad.human.cornell.edu/papers/downloads/StateEstimatesV2024Memo.pdf

10. CTData Collaborative. "Census Bureau Releases Vintage 2024 State Population Estimates." December 2024. https://www.ctdata.org/blog/vintage-2024-state-population-estimates

### Data Files Used

1. Census Bureau methodology statements (web pages and PDFs as cited above)
2. Census Bureau working papers (as cited above)

### Methods References

1. Das Gupta, Prithwis. Methods for producing intercensal estimates. Referenced in Census Bureau intercensal methodology documentation.
2. Residence One Year Ago (ROYA) method. Census Bureau standard methodology for foreign-born immigration estimation using ACS.

---

## Appendix: Technical Details

### A.1 Components of Net International Migration

The Census Bureau's NIM estimate comprises several sub-components:

1. **Foreign-born immigration**: Estimated using ACS ROYA question (residence abroad one year ago)
2. **Foreign-born emigration**: Estimated using residual methods based on survival/attrition
3. **Net native migration**: Net of U.S.-born moving to/from abroad
4. **Net Puerto Rico migration**: Movement between states and Puerto Rico (not technically international but included in NIM)
5. **Armed Forces movement**: Net movement of military between U.S. and overseas

### A.2 State Allocation Methodology Summary

**Pre-2010**: National NIM distributed to states using patterns derived from decennial census data on foreign-born population and administrative records from INS.

**2010s**: National NIM distributed to states using 3-year pooled ACS 1-year files for state demographic distributions. County distributions use ACS 5-year files.

**Vintage 2024**: Same ACS-based distribution applied to DHS-adjusted national total. Census Bureau acknowledges this may not accurately reflect humanitarian migrant settlement patterns.

### A.3 Key Methodology Transition Dates

| Transition | Year | Nature of Change |
|------------|------|------------------|
| Full ACS implementation | 2005 | ROYA data available nationally |
| Vintage 2020 base | 2020 | Blended base incorporating 2020 Census, Vintage 2020 estimates, and Demographic Analysis |
| COVID-19 adjustment | 2021 | Special adjustments for pandemic impact on migration |
| DHS data integration | 2024 | Administrative data supplement to ACS for humanitarian migrants |

---

## Revision History

| Date | Version | Author | Changes |
|------|---------|--------|---------|
| 2026-01-01 | 1.0 | Agent 1 (Claude Opus 4.5) | Initial report |

---

*End of Report*
