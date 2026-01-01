# ADR-020a: Vintage Methodology Investigation Plan

## Purpose

This document specifies the rigorous, methodical investigation to be conducted by sub-agents to assess the validity risks of using Census Bureau PEP data across vintage methodology transitions (2009 to 2010 and 2019 to 2020) for the North Dakota international migration time series.

**External Review**: Phase A findings will be reviewed by ChatGPT 5.2 Pro prior to deciding whether to proceed with Phase B.

## Investigation Structure

The investigation proceeds in two phases:
1. **Phase A**: Assess validity risks (Agents 1-3)
2. **Phase B**: Evaluate correction methods (Agents 4-5), contingent on Phase A findings and external review

## Output Standards

All agents must follow standardized output formats to enable:
- Cross-agent comparison
- External AI review (ChatGPT 5.2 Pro has no local file access)
- Reproducibility and verification

**Required Templates and Specifications**:
- Report Template: `docs/adr/019-reports/REPORT_TEMPLATE.md`
- Artifact Specifications: `docs/adr/019-reports/ARTIFACT_SPECIFICATIONS.md`
- Briefing Template: `docs/adr/019-reports/CHATGPT_BRIEFING_TEMPLATE.md`

## Phase A: Validity Risk Assessment

### Agent 1: Census Bureau Methodology Documentation Review

**Objective**: Establish authoritative understanding of how PEP methodology differs across vintages

**Tasks**:

1. **Locate and review Census Bureau technical documentation**
   - Search for methodology statements for Vintage 2009, 2020, and 2024
   - Focus specifically on Net International Migration (NIM) estimation methods
   - Identify data sources used in each vintage (administrative records, surveys, residual methods)

2. **Document known methodology changes**
   - What changed in NIM estimation between 2000s and 2010s?
   - What changed between Vintage 2020 and Vintage 2024?
   - Are there published accounts of methodology impacts on estimates?

3. **Identify Census Bureau caveats**
   - What limitations does Census Bureau acknowledge for state-level NIM?
   - Are there specific concerns for small states like North Dakota?
   - What is the recommended use case for these data?

4. **Search for peer-reviewed assessments**
   - Academic papers evaluating PEP accuracy
   - Studies comparing PEP to alternative data sources
   - Demographic literature on vintage reconciliation

**Deliverables** (per ARTIFACT_SPECIFICATIONS.md):
- `AGENT_1_REPORT.md` - Full report using REPORT_TEMPLATE.md
- `agent1_findings_summary.json` - Machine-readable findings
- `agent1_sources.json` - Complete bibliography with key excerpts
- `agent1_methodology_matrix.csv` - Side-by-side vintage comparison
- `agent1_census_quotes.json` - Direct quotes supporting findings
- `agent1_data_sources_timeline.csv` - Administrative data source history

**Search locations**:
- Census Bureau website (census.gov)
- Census Bureau Working Papers series
- Demography, Population Studies, and related journals
- Google Scholar for academic assessments

---

### Agent 2: Statistical Transition Analysis

**Objective**: Conduct quantitative tests to detect vintage-related artifacts in the ND time series

**Tasks**:

1. **Level shift analysis at transition points**
   - Test for mean shift at 2009 to 2010 boundary
   - Test for mean shift at 2019 to 2020 boundary (controlling for COVID)
   - Compare pre-transition vs. post-transition levels

2. **Variance analysis across vintages**
   - Calculate variance within Vintage 2009 (2000-2009)
   - Calculate variance within Vintage 2020 (2010-2019)
   - Calculate variance within Vintage 2024 (2020-2024)
   - Test for heteroskedasticity across vintages (Levene's test, Bartlett's test)

3. **Structural break testing at known transition points**
   - Chow test with known break at 2009/2010
   - Chow test with known break at 2019/2020
   - Compare break statistics to those at non-transition years

4. **Autocorrelation analysis**
   - Examine ACF/PACF patterns within each vintage
   - Test whether autocorrelation structure changes at vintage boundaries
   - Look for evidence of artificial smoothing or discontinuities

5. **Trend analysis by vintage**
   - Fit linear trends within each vintage
   - Compare trend slopes across vintages
   - Test for trend breaks at transition points

**Deliverables** (per ARTIFACT_SPECIFICATIONS.md):
- `AGENT_2_REPORT.md` - Full report using REPORT_TEMPLATE.md
- `agent2_findings_summary.json` - Machine-readable findings
- `agent2_sources.json` - Methods references
- `agent2_nd_migration_data.csv` - Raw data used
- `agent2_test_results.csv` - Complete statistical test results
- `agent2_transition_metrics.json` - Key quantitative metrics
- `agent2_fig1_timeseries_with_vintages.png` - Annotated time series
- `agent2_fig2_variance_by_vintage.png` - Variance comparison
- `agent2_fig3_structural_breaks.png` - Break test visualization
- `agent2_fig4_acf_by_vintage.png` - Autocorrelation plots
- `agent2_calculations.md` or `.ipynb` - Reproducible calculations

**Data source**: `/data/processed/immigration/state_migration_components_2000_2024.csv`

---

### Agent 3: Cross-Vintage Comparability Assessment

**Objective**: Evaluate whether the three vintages measure the same underlying construct

**Tasks**:

1. **Correlation with external indicators by vintage**
   - Correlate ND international migration with US total by vintage
   - Correlate with economic indicators (unemployment, oil prices) by vintage
   - Compare correlation patterns across vintages

2. **Relationship to known events by vintage**
   - Does the series respond to 2008 financial crisis as expected?
   - Does the series capture Bakken boom effects appropriately?
   - Are policy events (Travel Ban) visible in the expected vintage?

3. **Comparison with alternative data sources**
   - DHS Legal Permanent Resident data (available 2007-2023)
   - ACS foreign-born population changes
   - Calculate correlations between PEP and alternative sources by period

4. **Pattern replication across states**
   - Do similar vintage transition patterns appear for other small states?
   - Is the ND pattern unique or part of a broader measurement pattern?
   - Compare ND to similar states (SD, MT, WY)

5. **Coherence checks**
   - Do ND international and domestic migration show consistent patterns?
   - Is the ratio of ND to US migration stable across vintages?
   - Do components (births, deaths, migration) sum correctly?

**Deliverables** (per ARTIFACT_SPECIFICATIONS.md):
- `AGENT_3_REPORT.md` - Full report using REPORT_TEMPLATE.md
- `agent3_findings_summary.json` - Machine-readable findings
- `agent3_sources.json` - References
- `agent3_external_correlations.csv` - Correlation analysis by period
- `agent3_state_comparison.csv` - Cross-state pattern comparison
- `agent3_validation_data.csv` - PEP vs alternative sources
- `agent3_coherence_checks.json` - Internal consistency results

---

## Phase A Synthesis and External Review

### Synthesis Process

After all three agents complete, a synthesis step will:

1. **Aggregate findings** across agents
2. **Identify agreements and tensions** in conclusions
3. **Prepare external review briefing** using CHATGPT_BRIEFING_TEMPLATE.md

**Synthesis Deliverables**:
- `synthesis_findings_matrix.csv` - Cross-reference of all findings
- `synthesis_recommendations.json` - Aggregate recommendations
- `CHATGPT_BRIEFING.md` - Completed briefing document

### External Review by ChatGPT 5.2 Pro

**Context**: ChatGPT 5.2 Pro has no access to local files. All context must be provided as:
- Text (Markdown reports)
- Uploadable files (CSV, JSON, PNG)
- Complete journal article draft (relevant sections)

**Materials to Provide**:
1. All three agent reports (Markdown)
2. Key data files (CSV)
3. Summary JSON files
4. Visualizations (PNG, 300 DPI minimum)
5. Completed briefing document with specific review requests

**Review Requests Should Include**:
- Validation of statistical methods
- Assessment of evidence strength
- Identification of alternative interpretations
- Specific decision recommendation (Option A/B/C/D)
- Suggested additional analyses if warranted

### Decision Point

After external review, decide:

```
├── If no significant artifacts detected → Proceed with extension (Option B from ADR-020)
├── If artifacts detected but correctable → Proceed to Phase B
├── If uncertain → May request additional analysis
└── If severe artifacts detected → Maintain current approach (Option D)
```

---

## Phase B: Correction Methods Investigation

*Proceed only after Phase A agents complete, synthesis is prepared, and external review recommends Phase B*

### Agent 4: Academic Literature on Time Series Corrections

**Objective**: Identify methods used in academic literature to handle methodology breaks in time series

**Tasks**:

1. **Search for methodology break correction literature**
   - Survey literature
   - National accounts reconciliation methods
   - Epidemiological time series with testing methodology changes

2. **Document splicing/bridging techniques**
   - Chain-linking methods
   - Ratio splicing
   - Regression-based bridging
   - Overlap-based calibration

3. **Identify regime-switching approaches**
   - Markov-switching models with measurement regimes
   - Threshold models
   - Structural break models with known break dates

4. **Review sensitivity analysis frameworks**
   - How do researchers report results with known methodology changes?
   - What are best practices for transparency?
   - How are confidence intervals adjusted?

5. **Assess applicability to our context**
   - Which methods work with n=25?
   - Which require overlapping observations (we don't have)?
   - What assumptions do each method require?

**Deliverables**: Per ARTIFACT_SPECIFICATIONS.md format

---

### Agent 5: Correction Method Implementation Assessment

**Objective**: Evaluate specific correction approaches for the ND time series

**Tasks**:

1. **Evaluate available correction data**
   - Do we have any overlapping vintage estimates for same years?
   - What external anchors could be used for calibration?
   - What validation data is available?

2. **Assess dummy variable approaches**
   - Model with vintage indicators
   - Estimate vintage fixed effects
   - Test whether vintage effects are significant

3. **Evaluate detrending approaches**
   - Separate trend estimation within each vintage
   - Combine detrended series
   - Assess whether this preserves signal of interest

4. **Consider robust estimation methods**
   - Methods less sensitive to level shifts
   - Rank-based approaches
   - Trimmed estimation

5. **Design sensitivity analysis protocol**
   - What analyses should be run on both 2010-2024 and 2000-2024?
   - How should results be compared?
   - What decision rule determines if extension is justified?

**Deliverables**: Per ARTIFACT_SPECIFICATIONS.md format, including specific code/pseudocode

---

## Execution Sequence

```
Phase A (Parallel):
├── Agent 1: Census Methodology Documentation → Report + Artifacts
├── Agent 2: Statistical Transition Analysis → Report + Artifacts
└── Agent 3: Cross-Vintage Comparability → Report + Artifacts

Synthesis:
├── Aggregate all findings into synthesis artifacts
├── Prepare completed CHATGPT_BRIEFING.md
└── Assemble all materials for external review

External Review (ChatGPT 5.2 Pro):
├── Provide all reports, data, and visualizations
├── Request specific analysis and recommendations
└── Receive structured response

Decision Point:
├── If Option B supported → Proceed with extension
├── If Phase B needed → Continue to Agents 4-5
├── If Option D supported → Maintain n=15
└── If inconclusive → Additional analysis

Phase B (if needed, Parallel):
├── Agent 4: Correction Methods Literature → Methods compendium
└── Agent 5: Implementation Assessment → Specific recommendations

Final Decision: Select Option A, B, C, or D from ADR-020
```

## Success Criteria

The investigation succeeds if it produces:

1. **Clear characterization** of vintage methodology differences (Agent 1)
2. **Quantitative bounds** on potential artifact magnitudes (Agent 2)
3. **Validity assessment** of cross-vintage comparability (Agent 3)
4. **External validation** of findings and reasoning (ChatGPT 5.2 Pro)
5. **Actionable correction methods** if artifacts are detected (Agents 4-5)
6. **Defensible decision** on whether to extend the time series

## Documentation Requirements

Each agent must:

1. **Follow the report template** exactly (REPORT_TEMPLATE.md)
2. **Produce all required artifacts** per ARTIFACT_SPECIFICATIONS.md
3. **Quantify uncertainty explicitly** for every finding
4. **Flag review requests** for external AI clearly
5. **Use machine-readable formats** for key outputs (JSON, CSV)

## File Organization

All Phase A outputs go to: `docs/adr/019-reports/`

```
docs/adr/019-reports/
├── REPORT_TEMPLATE.md           # Template for agent reports
├── ARTIFACT_SPECIFICATIONS.md   # What artifacts to produce
├── CHATGPT_BRIEFING_TEMPLATE.md # Template for external briefing
├── AGENT_1_REPORT.md            # Agent 1 completed report
├── AGENT_2_REPORT.md            # Agent 2 completed report
├── AGENT_3_REPORT.md            # Agent 3 completed report
├── CHATGPT_BRIEFING.md          # Completed briefing for external review
├── CHATGPT_RESPONSE.md          # External review response (when received)
├── agent1_*.{csv,json}          # Agent 1 artifacts
├── agent2_*.{csv,json,png}      # Agent 2 artifacts
├── agent3_*.{csv,json}          # Agent 3 artifacts
└── synthesis_*.{csv,json}       # Synthesis artifacts
```

## Quality Assurance

Before synthesis, verify each agent's output:

- [ ] Report follows template structure completely
- [ ] All required artifacts are present
- [ ] JSON files are valid and parseable
- [ ] CSV files have correct headers
- [ ] Visualizations are legible (300 DPI, clear labels)
- [ ] Uncertainty is quantified for all findings
- [ ] Sources are properly cited
- [ ] Review requests are clearly articulated

## Related Documents

- [ADR-020: Extended Time Series Methodology Analysis](./019-extended-time-series-methodology-analysis.md) - Parent decision record
- [state_migration_components_2000_2024.csv](../../data/processed/immigration/state_migration_components_2000_2024.csv) - Combined vintage data
- [combine_census_vintages.py](../../sdc_2024_replication/data_immigration_policy/scripts/combine_census_vintages.py) - Script that created combined file

## References

- Census Bureau Population Estimates Program methodology
- Census Bureau Working Papers on net international migration estimation
- Academic literature on time series methodology breaks
