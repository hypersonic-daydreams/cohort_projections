# ADR-019: Extended Time Series Methodology Analysis

## Status
**PROPOSED** - Awaiting rigorous methodology impact analysis

## Date
2025-12-31

## Context

The journal article's statistical analysis currently uses **n=15 annual observations (2010-2024)** for North Dakota international migration. This short time series creates significant limitations:

- Unit root tests require n≥25 for reliable asymptotic properties
- VAR/cointegration analysis has insufficient degrees of freedom
- Structural break detection is constrained
- Prediction intervals are very wide
- Granger causality tests fail due to insufficient data

**Discovery**: Census Bureau Population Estimates Program (PEP) data exists for 2000-2024, potentially extending the series to **n=25**.

## Proposed Extension

### Available Data Sources

| Vintage | Coverage | File | Status |
|---------|----------|------|--------|
| Vintage 2009 | 2000-2009 | `NST-EST2009-ALLDATA.csv` | Available |
| Vintage 2020 | 2010-2019 | `NST-EST2020-ALLDATA.csv` | Available |
| Vintage 2024 | 2020-2024 | `NST-EST2024-ALLDATA.csv` | Available |

### North Dakota International Migration by Vintage

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

## Validity Risks

### Risk 1: Vintage Methodology Changes

Census Bureau PEP methodology evolves between vintages. Key concerns:

1. **Estimation methods differ**: Each decennial census creates a new base, and intercensal estimates use different administrative data sources
2. **Retroactive revisions**: When new census data becomes available, prior estimates are revised but only within the same vintage
3. **International migration estimation**: The methods for estimating net international migration have evolved significantly:
   - Pre-2010: Residual-based estimates
   - 2010-2020: Foreign-born population flows from ACS
   - Post-2020: Enhanced administrative records integration

### Risk 2: Structural vs. Methodological Breaks

The series shows apparent breaks that could be:
- **Real phenomena**: 2008 financial crisis, 2010 Bakken boom, 2017 Travel Ban, 2020 COVID
- **Methodological artifacts**: Vintage 2009→2020 transition, Vintage 2020→2024 transition

**Critical question**: How do we distinguish between genuine structural breaks and measurement artifacts?

### Risk 3: Level Shifts from Vintage Changes

Observed patterns at transition points:
- **2009→2010**: 521 → 468 (modest decline, -10%)
- **2019→2020**: 634 → 30 (massive decline, -95%, but COVID is a known real event)

The 2009→2010 transition is concerning because:
- It coincides with the 2010 Census (major methodology update)
- The Bakken boom began in 2011, not 2010
- Any apparent discontinuity could be measurement artifact

### Risk 4: Variance Heterogeneity

Pre-2010 values (258-1,025) have different variance characteristics than post-2010 (30-5,126). This could indicate:
- Real increased volatility in migration patterns
- Improved measurement precision in newer vintages
- Different underlying data quality

### Risk 5: Comparability Across Decades

The 2000s data reflects:
- Different economic conditions (pre-Bakken)
- Different immigration policy environment
- Different Census Bureau data infrastructure

Using these data assumes the underlying measurement construct is comparable.

## Required Analysis

### Phase 1: Methodology Documentation Review

Sub-agents should investigate:

1. **Census Bureau technical documentation** for each vintage
   - What are the exact methods for estimating net international migration?
   - What administrative data sources are used?
   - How are foreign-born flows estimated?

2. **Published methodology comparisons**
   - Has Census Bureau documented vintage-to-vintage differences?
   - Are there peer-reviewed studies comparing PEP vintage accuracy?

3. **Known issues and caveats**
   - What limitations does Census Bureau acknowledge?
   - Are there state-specific concerns for small states like ND?

### Phase 2: Statistical Diagnostics

Sub-agents should conduct:

1. **Vintage transition analysis**
   - Test for level shifts at 2009→2010 and 2019→2020 boundaries
   - Compare within-vintage vs. across-vintage variance
   - Examine autocorrelation patterns before/after transitions

2. **Structural break testing**
   - Apply Chow tests at vintage transition points
   - Apply Bai-Perron multiple break detection
   - Compare break dates to known methodological vs. real events

3. **Distributional analysis**
   - Test for variance homogeneity across vintages
   - Examine skewness/kurtosis changes
   - Look for outlier patterns by vintage

### Phase 3: Correction Methods Investigation

If Phase 1-2 identify significant methodology effects, sub-agents should investigate:

1. **Splicing techniques**
   - Chain-linking across vintage transitions
   - Level adjustment based on overlapping years
   - Ratio-based calibration

2. **Statistical adjustments**
   - Regime-switching models that account for measurement changes
   - Dummy variables for vintage periods
   - Heteroskedasticity-robust estimation

3. **Sensitivity analysis approaches**
   - Run all analyses on 2010-2024 and 2000-2024 separately
   - Report results with and without vintage controls
   - Bound estimates using alternative assumptions

4. **Alternative data sources for validation**
   - DHS LPR data (2007-2023)
   - ACS foreign-born population changes
   - State-level administrative records

## Decision Framework

After Phase 1-3 analysis, decide between:

### Option A: Use Extended Series with Corrections
- Apply identified correction methods
- Document all adjustments transparently
- Report sensitivity analyses

### Option B: Use Extended Series with Caveats
- Use 2000-2024 data as-is
- Document known methodology changes
- Interpret results cautiously around transition points

### Option C: Use Hybrid Approach
- Primary analysis on 2010-2024 (consistent methodology)
- Robustness checks on 2000-2024
- Report both sets of results

### Option D: Maintain Current Approach
- Continue with n=15 (2010-2024)
- Accept statistical power limitations
- Focus on methods robust to small samples

## Sub-Agent Investigation Plan

### Agent 1: Census Methodology Documentation
**Objective**: Gather authoritative documentation on PEP vintage methodology differences
**Scope**: Census Bureau technical papers, working papers, methodology statements
**Deliverable**: Summary of known methodology changes affecting international migration estimates

### Agent 2: Statistical Transition Analysis
**Objective**: Conduct diagnostic tests for vintage-related artifacts
**Scope**: Apply statistical tests to the ND time series at transition points
**Deliverable**: Quantitative assessment of transition effects

### Agent 3: Cross-Vintage Comparability Assessment
**Objective**: Evaluate whether vintages measure the same underlying construct
**Scope**: Compare patterns, correlations, and relationships across vintages
**Deliverable**: Assessment of data comparability for time series analysis

### Agent 4: Correction Methods Research
**Objective**: Identify and evaluate potential correction approaches
**Scope**: Academic literature on time series splicing, vintage reconciliation
**Deliverable**: Recommended correction methods with pros/cons

### Agent 5: Validation Data Analysis
**Objective**: Use alternative data sources to validate patterns
**Scope**: DHS, ACS, state records
**Deliverable**: External validation of observed patterns

## Related Documents

- [state_migration_components_2000_2024.csv](../../data/processed/immigration/state_migration_components_2000_2024.csv) - Combined vintage data
- [combine_census_vintages.py](../../sdc_2024_replication/data_immigration_policy/scripts/combine_census_vintages.py) - Script that created combined file
- Module 2.1.1 unit root tests - Documents n=15 limitations
- Module 2.1.2 structural breaks - Documents power constraints

## References

- Census Bureau Population Estimates Program methodology
- Census Bureau Working Papers on net international migration estimation
- Academic literature on time series methodology breaks
