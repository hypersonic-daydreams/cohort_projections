# ADR-035: Census PEP Components of Change for Migration Inputs

## Status
Accepted

## Date
2026-02-03

## Context

### Problem Discovery

During analysis of North Dakota 2025-2045 population projections, a significant divergence was discovered between current baseline projections (decline to 754,882 by 2045) and the 2024 State Data Center projections (growth to 950,000+ by 2050). Investigation revealed that **migration assumptions account for ~170,000 people difference by 2045**.

Full analysis documented in: [docs/reports/PROJECTION_DIVERGENCE_ANALYSIS.md](../../reports/PROJECTION_DIVERGENCE_ANALYSIS.md)

### Current Implementation (IRS County-to-County Flows)

**Data Source**: IRS Statistics of Income county-to-county migration flows, 2019-2022

**Characteristics**:
- Domestic migration only (based on tax return address changes)
- County-level directional flows (origin → destination)
- Recent 4-year period averaging
- Administrative records (high accuracy for domestic moves)

**Limitations Identified**:
1. **Missing international migration**: IRS captures domestic address changes only (~1,100-1,200 international migrants/year for ND)
2. **Temporal bias**: 2019-2022 period covers:
   - COVID-19 pandemic disruptions (2020-2021)
   - Bakken oil region bust aftermath (2020-2021)
   - Western ND counties experiencing massive out-migration
3. **Misses recent recovery**: 2023-2024 data shows major rebound (+5,181 and +4,835 net migration) with international component driving growth (+4,227/year average)
4. **Short historical window**: Only 4 years, cannot capture longer-term trends or oil boom/bust cycles

**Result**: Average net migration of **-987 people/year** (2019-2022), representing the worst recent period rather than typical conditions.

### Census PEP Components of Change Alternative

**Data Source**: Census Population Estimates Program (PEP) components of change, 2000-2024

**Characteristics**:
- Comprehensive migration (domestic + international)
- Residual method: Net migration = population change - births + deaths
- 24 years of historical data
- State and county level available
- Captures all migration regardless of mechanism

**Historical North Dakota Trends** (from state-level PEP data):

| Period | Avg Net Migration/Year | Regime | Notes |
|--------|----------------------|--------|-------|
| 2000-2010 | -1,267 | Pre-Bakken | Baseline out-migration |
| 2011-2015 | +11,650 | Bakken Boom | Peak in-migration |
| 2016-2021 | -2,391 | Bust + COVID | Volatility + pandemic |
| 2019-2022 | -987 | IRS Period | Worst recent conditions |
| 2022-2024 | +3,545 | Recovery | International-driven growth |
| 2000-2024 | +1,624 | Full Period | Long-term average |

**Key Insight**: IRS data (2019-2022) captured net -987/year, while full PEP period (2000-2024) shows +1,624/year average. This 2,611 people/year difference compounds to ~52,000 people by 2045.

### Methodological Alignment

**SDC 2024 Methodology**: Used Census residual method (2000-2020) with 60% Bakken dampening, resulting in net in-migration assumption.

**Our Current Baseline**: Uses IRS (2019-2022), resulting in net out-migration assumption.

**Divergence**: These different assumptions create the ~170,000 people gap by 2045.

### Infrastructure Availability

Census PEP data archive already established per [ADR-034: Census PEP Data Archive](./034-census-pep-data-archive.md):

- **Location**: `~/workspace/shared-data/census/popest/`
- **Coverage**: State and county data, 2000-2024 (multiple vintages)
- **Status**: All 5 phases complete (download, parquet conversion, archiving, documentation, PostgreSQL analytics layer)
- **Data volume**: ~278 MB raw staging, harmonized in PostgreSQL
- **Access**: Via `CENSUS_POPEST_DIR` and `CENSUS_POPEST_PG_DSN` environment variables

County-level components of change data is available and ready for extraction.

## Decision

**We will replace IRS county-to-county migration flows with Census Population Estimates Program (PEP) components of change data as the primary migration input for cohort-component projections.**

### Scope

**In Scope (This ADR)**:
1. Use Census PEP net migration (domestic + international combined) as migration input
2. Extract county-level net migration from PEP components of change datasets
3. Calculate multi-period averages to capture historical trends and regime shifts
4. ~~Implement regime-aware averaging (pre-Bakken, boom, bust, recovery)~~ → Superseded by ADR-036: BEBR multi-period averaging + Census Bureau interpolation
5. Rigorous validation against known population changes
6. Age/sex distribution allocation of county net migration totals

**Deferred to Future Work**:
1. Blending PEP net migration totals with IRS directional flow patterns
2. Origin-destination matrices for internal migration modeling
3. Place-level (city) migration estimation using apportionment methods
4. International migration forecasting and scenario modeling

### Implementation Approach

**Phase 1: Data Extraction (County-Level PEP Components)**
- Extract net migration from county PEP components of change files (2000-2024)
- Validate against state totals (county sums must equal state)
- Document vintage differences and methodology changes
- Create harmonized county time series

**Phase 2: Regime Analysis (Granular Historical Understanding)**
- Identify migration regimes by county (oil vs. non-oil counties)
- Calculate period-specific averages:
  - Pre-Bakken baseline (2000-2010)
  - Bakken boom (2011-2015)
  - Bust + COVID (2016-2021)
  - Recent recovery (2022-2024)
- Document county-specific patterns (Williams, McKenzie, Ward vs. Cass, Burleigh)

**Phase 3: Migration Rate Calculation (Rigorous Methodology)**
- Calculate age/sex-specific migration rates using:
  - PEP net migration totals (county level)
  - ACS population distributions by age/sex (for allocation)
  - Multi-period smoothing to reduce year-to-year volatility
- Implement dampening strategies for extreme oil boom/bust periods
- Create scenario-specific rate sets (conservative, moderate, optimistic)

**Phase 4: Integration & Validation (Quality Assurance)**
- Replace IRS inputs in projection pipeline
- Run test projections for validation counties
- Compare results to:
  - SDC 2024 projections (methodology alignment check)
  - Recent Census PEP estimates (short-term accuracy)
  - Historical intercensal estimates (long-term plausibility)
- Document divergence from IRS-based projections

**Phase 5: Scenario Development (Uncertainty Quantification)**
- **Baseline**: Blend 2000-2024 average with dampened boom/bust extremes
- **Low**: Recent trends (2019-2024) with conservative international assumptions
- **High**: Include partial boom recovery for oil-producing counties
- **SDC 2024**: Replicate SDC methodology for comparability

## Rationale

### Why Census PEP Over IRS

1. **Comprehensiveness**: PEP captures both domestic and international migration; IRS misses ~1,100-1,200 international migrants/year for North Dakota
2. **Temporal Coverage**: 24 years (2000-2024) vs. 4 years (2019-2022); captures full boom/bust cycles
3. **Current Data**: Includes 2022-2024 recovery period showing return to growth
4. **Methodological Consistency**: Aligns with SDC 2024 approach and standard demographic practice
5. **Infrastructure**: Data already archived and accessible per ADR-034

### Impact Quantification

**Missing International Component**: ~1,100/year × 20 years = **~22,000 people by 2045**

**Period Selection Bias**: Using -987/year (IRS 2019-2022) vs. +1,624/year (PEP 2000-2024) average = **~52,000 people by 2045**

**Total Impact**: **~74,000-80,000 people difference by 2045** from data source choice alone (about 10% of ND population)

### Why Defer IRS Directional Flows

While IRS data provides valuable directional flow information (which counties gain/lose to which destinations), implementing this requires:
- Complex origin-destination matrix estimation
- Age/sex-specific flow modeling
- Uncertain value-add over net migration totals for aggregate projections
- Significant implementation complexity

**Decision**: Demonstrate value of PEP net migration first, then assess whether directional detail justifies additional complexity.

## Alternatives Considered

### Alternative 1: Continue with IRS Data Only
**Rejected**: Misses international component and captures anomalous period. Results in pessimistic bias.

### Alternative 2: Blend IRS + PEP (Immediate Implementation)
**Deferred**: Adds complexity before establishing baseline. Better to validate PEP approach first, then enhance with directional flows if needed.

### Alternative 3: Use ACS Migration Tables
**Rejected**: ACS migration data has higher sampling error at county level and shorter time series than PEP components.

### Alternative 4: Status Quo (Keep Current Projections)
**Rejected**: Current projections fail face validity due to missing ~74,000-80,000 people from data source limitations.

## Consequences

### Positive

1. **Comprehensiveness**: Captures full migration picture (domestic + international)
2. **Historical Depth**: 24 years of data supports regime analysis and trend identification
3. **Current Relevance**: Includes 2022-2024 recovery data
4. **Methodological Soundness**: Aligns with standard demographic practice and SDC 2024 approach
5. **Infrastructure Ready**: Leverages existing PEP data archive (ADR-034)
6. **Flexibility**: Can layer IRS directional flows later without discarding PEP foundation
7. **Scenario Capability**: Rich historical data supports multiple defensible scenarios

### Negative

1. **Loss of Directional Detail**: Net migration only, no origin-destination information
2. **Residual Method Limitations**: PEP migration is calculated residually (includes estimation error)
3. **Age/Sex Allocation**: Net migration totals require age/sex distribution estimation (not directly observed)
4. **Implementation Work**: Requires new data extraction and rate calculation scripts
5. **Baseline Shift**: Results will differ from current projections (feature, not bug)

### Risks and Mitigations

**Risk**: PEP residual method includes statistical noise and administrative corrections
- **Mitigation**: Multi-period averaging; regime-aware smoothing; validation against independent sources

**Risk**: Age/sex allocation of net migration totals may introduce bias
- **Mitigation**: Use ACS microdata for empirical age/sex patterns; validate against detailed PEP demographic files where available

**Risk**: County-level PEP components may have vintages/methodology breaks
- **Mitigation**: Careful documentation review; harmonization across vintages; explicit handling of geographic boundary changes

**Risk**: Results will diverge from current baseline (stakeholder confusion)
- **Mitigation**: Clear documentation of rationale; scenario comparison; validation against SDC 2024 and recent trends

## Implementation Plan

### Phase 1: Data Extraction and Harmonization (Rigorous Foundation)
**Timeline**: Session 1 (Today)
**Deliverables**:
- [ ] Extract county-level net migration from PEP components (2000-2024)
- [ ] Validate county sums = state totals (hierarchical consistency)
- [ ] Document vintage differences and methodology changes
- [ ] Create harmonized time series dataset: `data/processed/pep_county_components_2000_2024.parquet`
- [ ] Generate validation report comparing to state totals and published estimates

### Phase 2: Regime Analysis and Period Identification (Granular Understanding)
**Timeline**: Session 2
**Deliverables**:
- [ ] County-level regime classification (oil vs. non-oil; metro vs. rural)
- [ ] Period-specific migration averages by regime:
  - Pre-Bakken (2000-2010)
  - Boom (2011-2015)
  - Bust+COVID (2016-2021)
  - Recovery (2022-2024)
- [ ] Visualization of migration patterns by county and regime
- [ ] Documentation: `docs/analysis/migration_regime_analysis.md`

### Phase 3: Rate Calculation and Age/Sex Allocation (Technical Rigor)
**Timeline**: Session 3
**Deliverables**:
- [ ] Age/sex-specific migration rate calculation methodology
- [ ] Extraction of ACS age/sex patterns for migration allocation
- [ ] Multi-period smoothing and outlier handling
- [ ] Scenario-specific rate sets (baseline, low, high, sdc_2024)
- [ ] Output: `data/processed/migration_rates_pep.parquet` (replaces current `migration_rates.parquet`)
- [ ] Validation: Compare implied age/sex distributions to ACS migration tables

### Phase 4: Pipeline Integration and Testing (Quality Assurance)
**Timeline**: Session 4
**Deliverables**:
- [ ] Update data processing pipeline to use PEP migration rates
- [ ] Run projections for test counties (Burleigh, Cass, Williams, McKenzie)
- [ ] Compare results to:
  - Current IRS-based projections
  - SDC 2024 projections
  - Recent PEP estimates (2022-2024 trends)
- [ ] Generate comparison report with visualizations
- [ ] Document methodology in `docs/methods/migration_methodology.md`

### Phase 5: Full Projection Run and Scenario Analysis (Production Deployment)
**Timeline**: Session 5
**Deliverables**:
- [ ] Run full state projections (all 53 counties) with PEP migration inputs
- [ ] Generate baseline and scenario projections (low, high, sdc_2024)
- [ ] Create comprehensive comparison to IRS-based projections
- [ ] Update [PROJECTION_DIVERGENCE_ANALYSIS.md](../../reports/PROJECTION_DIVERGENCE_ANALYSIS.md) with resolution
- [ ] Generate stakeholder briefing materials
- [ ] Update [DEVELOPMENT_TRACKER.md](../../../DEVELOPMENT_TRACKER.md) with completion status

### Future Enhancement: IRS Directional Flow Blending (Deferred)
**Timeline**: TBD (post-validation)
**Approach**:
- Use PEP net migration totals as control totals
- Use IRS flows to distribute net migration across origin-destination pairs
- Implement iterative proportional fitting (IPF) to reconcile totals with flows
- Validate added value for aggregate county projections

## Data Locations and Configuration

### Input Data (Census PEP Archive)
**Source**: [ADR-034 Census PEP Data Archive](./034-census-pep-data-archive.md)
- **State-level**: `data/processed/immigration/state_migration_components_2000_2024.csv`
- **County-level** (to be extracted from PEP archive):
  - Vintage 2020-2024: `~/workspace/shared-data/census/popest/parquet/2020-2024/county/*.parquet`
  - Vintage 2010-2020: `~/workspace/shared-data/census/popest/parquet/2010-2020/county/*.parquet`
  - Vintage 2000-2010: `~/workspace/shared-data/census/popest/parquet/2000-2010/county/*.parquet`

### Output Data (Processed Migration Rates)
- **Harmonized components**: `data/processed/pep_county_components_2000_2024.parquet`
- **Migration rates**: `data/processed/migration_rates_pep.parquet` (replaces current `migration_rates.parquet`)
- **Regime analysis**: `data/processed/migration_regimes_by_county.csv`
- **Validation reports**: `data/output/validation/pep_migration_validation.html`

### Configuration Updates
**File**: [config/projection_config.yaml](../../../config/projection_config.yaml)

```yaml
rates:
  migration:
    domestic:
      method: "PEP_components"  # Changed from "IRS_county_flows"
      source: "Census_PEP"
      averaging_period: "regime_aware"  # Multi-period with regime weights
      smooth_extreme_outliers: true
      dampening_factor: 0.60  # For extreme boom/bust years (matches SDC 2024)
    international:
      method: "PEP_included"  # Included in PEP net migration
      allocation: "proportional"  # Distribute to counties proportionally
```

## Validation Criteria

### Must Pass (Go/No-Go)
1. County net migration sums match state totals within 0.1% (hierarchical consistency)
2. 2020-2024 projections track within 5% of actual PEP estimates (short-term accuracy)
3. Age/sex distributions plausible (no negative cohorts, reasonable sex ratios)
4. Western ND counties (Williams, McKenzie) show higher volatility than eastern counties (Cass, Grand Forks)

### Success Metrics (Quality Indicators)
1. Projections align with SDC 2024 methodology (within 10% by 2045)
2. Baseline scenario shows population growth consistent with recent recovery trends
3. Historical back-testing: applying method to 2000-2020 should track intercensal estimates within 15%
4. Regime-specific rates capture known boom/bust patterns (Williams County boom > 5,000/year 2011-2014)

### Documentation Requirements
1. Full methodology write-up in `docs/methods/migration_methodology.md`
2. Regime analysis report in `docs/analysis/migration_regime_analysis.md`
3. Validation report comparing PEP vs. IRS approaches with quantified differences
4. Update [PROJECTION_DIVERGENCE_ANALYSIS.md](../../reports/PROJECTION_DIVERGENCE_ANALYSIS.md) with resolution details

## Related ADRs

- **ADR-036: Migration Averaging Methodology** - Refines how PEP data is averaged; replaces regime-weighted approach (Phases 2–3) with BEBR multi-period and Census Bureau interpolation methods
- **ADR-034: Census PEP Data Archive** - Infrastructure for PEP data access
- **ADR-033: City-Level Projection Methodology** - Consumer of migration rates
- **ADR-016: Raw Data Management Strategy** - Git/rclone hybrid for data
- **ADR-010: Geographic Scope and Granularity** - County-level focus

## References

### Documentation
1. [PROJECTION_DIVERGENCE_ANALYSIS.md](../../reports/PROJECTION_DIVERGENCE_ANALYSIS.md) - Root cause analysis
2. [DEVELOPMENT_TRACKER.md](../../../DEVELOPMENT_TRACKER.md) - Project status
3. [ADR-034: Census PEP Data Archive](./034-census-pep-data-archive.md) - Data infrastructure

### Data Sources
1. Census PEP Methodology: https://www.census.gov/programs-surveys/popest/technical-documentation/methodology.html
2. Census PEP Datasets: https://www2.census.gov/programs-surveys/popest/datasets/
3. Census Components of Change: https://www.census.gov/data/tables/time-series/demo/popest/2020s-counties-detail.html

### Analysis
1. State-level PEP components: `data/processed/immigration/state_migration_components_2000_2024.csv`
2. Current IRS inputs: `data/raw/migration/nd_migration_processed.csv`
3. SDC 2024 rates: `data/processed/sdc_2024/migration_rates_sdc_2024.csv`

## Revision History

- **2026-02-03**: Initial version (ADR-035) - Decision to replace IRS with PEP migration data
- Focus: Rigor and granular analysis; defer IRS directional flow blending
- **2026-02-12**: Added reference to ADR-036 which refines the averaging methodology (regime-weighted → BEBR multi-period + Census Bureau interpolation)

## Next Steps

1. **Immediate** (Today): Phase 1 - Extract and harmonize county-level PEP components
2. **Session 2**: Phase 2 - Regime analysis and period identification
3. **Session 3**: Phase 3 - Rate calculation and age/sex allocation
4. **Session 4**: Phase 4 - Pipeline integration and validation
5. **Session 5**: Phase 5 - Full projection run and scenario analysis

**Starting Point**: Use existing state-level PEP analysis as template; extend to county level with rigorous validation.
