# ADR-033: City-Level Projection Methodology

## Status
Accepted

## Date
2026-02-02

## Implemented
2026-03-01

## Context

The cohort-component projection system currently produces projections at state and county levels (all 53 North Dakota counties, 2025-2055). There is demand for city/place-level projections for municipal planning and local decision-making.

### Challenge

Applying cohort-component methods directly to cities is problematic:

1. **Rate availability**: Fertility, mortality, and migration rates are only available at county level (or coarser)
2. **Small population volatility**: Many ND cities have populations <2,500; cohort-level projections become statistically unstable
3. **False precision**: Projecting 1,092 cohorts (age × sex × race) for a city of 500 people implies precision that doesn't exist

### Alternative Approach

Rather than applying cohort-component rates to cities, use **empirical share-trending**:

1. Cities have annual population estimates (Census PEP) going back 20+ years
2. Calculate each city's share of its containing county over time
3. Identify trends (cities gaining or losing share)
4. Project shares forward and apply to already-projected county totals

This approach:
- Uses empirical city-level data rather than imputed rates
- Maintains consistency with county cohort-component projections
- Allows uncertainty quantification based on historical forecast errors
- Is standard practice in sub-county demographic projections

## Decision

### Decision 1: Share-Trending as Primary Method

**Decision**: Use ratio-trending (share-of-county) method for city-level projections rather than cohort-component.

**Method**:
```
city_share_t = city_population_t / county_population_t

# Fit trend to historical shares (2000-2024)
trend_model = fit_share_trend(city_shares)

# Project share forward
city_share_2045 = trend_model.predict(2045)

# Apply to county projection
city_population_2045 = county_projection_2045 * city_share_2045
```

**Rationale**:
- Empirically grounded in actual city population history
- Avoids fiction of city-level demographic rates
- Maintains county-level constraint (shares sum to ≤100%)
- Standard practice in state demography offices

### Decision 2: Population-Based Confidence Tiers

**Decision**: Assign confidence tiers based on population size.

| Population | Tier | Output Detail | Uncertainty Band |
|------------|------|---------------|------------------|
| >10,000 | HIGH | Full age groups | ±10% |
| 2,500-10,000 | MODERATE | Broad age groups | ±15% |
| 500-2,500 | LOWER | Total only | ±25% |
| <500 | EXCLUDED | Not projected | — |

**Rationale**:
- Smaller populations have higher relative volatility
- Prevents misuse of unreliable detail
- Communicates uncertainty clearly to users

### Decision 3: Constraint to County Totals

**Decision**: City projections must be consistent with county projections.

**Constraints**:
- Sum of city shares within a county ≤ 100%
- Remainder = unincorporated/rural areas
- If trended shares sum to >100%, proportionally scale back

**Rationale**:
- Maintains hierarchical consistency
- Allows derivation of "balance of county" estimates
- Standard practice in demographic projections

### Decision 4: Uncertainty Quantification via Backtesting

**Decision**: Derive prediction intervals from historical forecast errors.

**Method**:
1. Use data 2000-2014 to project 2015-2024
2. Compare projections to actuals
3. Calculate error distribution by population tier
4. Use empirical error distribution to construct prediction intervals

**Rationale**:
- Empirically grounded uncertainty (not assumed)
- Accounts for actual volatility in ND cities
- Allows honest communication of projection reliability

## Implementation Plan (Proposal Baseline)

The plan below captures the proposed phase structure from initial ADR drafting.
Implementation is now complete; see **Implementation Results** for executed
outputs, metrics, and approval status.

### Phase 1: Data Assembly
- Assemble city population time series 2000-2024
- Calculate city/county share ratios
- Visualize trends by city size class
- **Status**: Completed (see Implementation Results)

### Phase 2: Trend Analysis
- Fit trend models to share time series
- Identify regime changes (e.g., oil boom effects)
- Classify cities by trend type (growing share, stable, declining)

### Phase 3: Model Building
- Implement share extrapolation with constraints
- Handle edge cases (new cities, annexations, boundary changes)
- Build age-distribution allocation for larger cities

### Phase 4: Integration
- Link to county cohort-component outputs
- Generate consistent multi-level outputs
- Update output writers for city-level files

### Phase 5: Validation
- Backtest against 2015-2024 actuals
- Calculate accuracy metrics by population tier
- Document prediction intervals

### Phase 6: Documentation & Communication
- User guide for interpreting city projections
- Clear labeling of confidence tiers
- Caveats for small-city projections

## Consequences

### Positive
1. **Empirically grounded**: Uses actual city population history
2. **Honest uncertainty**: Communicates reliability clearly
3. **Consistent**: Cities sum to county (minus unincorporated)
4. **Standard practice**: Follows established methodology
5. **Avoids false precision**: Doesn't pretend to have city-level rates

### Negative
1. **No cohort detail for small cities**: Age structure only for larger places
2. **Trend assumption**: Assumes recent trends continue
3. **Boundary changes**: City annexations can distort historical shares
4. **Additional complexity**: Adds a modeling layer beyond cohort-component

### Risks and Mitigations

**Risk**: Historical trends don't predict future (e.g., new development)
- **Mitigation**: Scenario-based projections (trend continues vs. stabilizes)
- **Mitigation**: Clear documentation that projections assume trend continuation

**Risk**: Boundary changes distort share calculations
- **Mitigation**: Identify and flag cities with annexations
- **Mitigation**: Use consistent geographic definitions (Census vintage)

**Risk**: Users misinterpret uncertainty
- **Mitigation**: Prominent confidence tier labeling
- **Mitigation**: User guide with interpretation examples

## Data Requirements

### Available
- City population estimates 2020-2024: `data/raw/geographic/nd_places.csv`
- County population estimates: `data/processed/nd_county_population.csv`
- County projections 2025-2055: `data/projections/*/county/*.parquet`

### Needed
- City population estimates 2000-2019 (Census PEP historical)
- City-to-county mapping with consistent boundaries

## Alternatives Considered

### Alternative 1: Cohort-Component at City Level
Apply county demographic rates to city base populations.

**Rejected because**:
- Implies false precision for small cities
- No city-level rate data exists
- Produces unreliable age-detail for small populations

### Alternative 2: Constant Share
Assume cities maintain current share of county.

**Rejected because**:
- Ignores urbanization/rural decline trends
- Historical data shows shares do change
- Less accurate than trending shares

### Alternative 3: Housing-Unit Method
Project housing units, apply persons-per-household.

**Considered for future**:
- Useful for short-term (5-year) projections
- Requires housing development data
- Could complement share-trending for near-term

## Historical Deferral Rationale (Resolved)

On 2026-02-28 this ADR remained deferred for sequencing reasons during
publication-priority work. That deferral state is now resolved:

1. ADR-054 aggregation prerequisites remained in place for county-constrained share modeling.
2. PP-003 implementation milestones (IMP-01 through IMP-19) were executed.
3. Human sign-off for IMP-19 was recorded on 2026-03-01.

## Implementation Results (2026-03-01)

### Overall Outcome

- PP-003 Phase 1 implementation for place projections is complete through IMP-19.
- Production winner selected in IMP-09: **Variant B-II** (`wls` + `cap_and_redistribute`), score `3.0757840903894844`.
- IMP-19 end-to-end validation passed for all active scenarios (`baseline`, `restricted_growth`, `high_growth`) with human sign-off recorded.

### Backtest Acceptance Metrics (Primary Window, Winner B-II)

- `HIGH` tier: `tier_medape=3.002588`, `tier_p90_mape=4.314164`, `abs_tier_mean_me=1.106081` (PASS)
- `MODERATE` tier: `tier_medape=1.835121`, `tier_p90_mape=17.781376`, `abs_tier_mean_me=4.280532` (PASS)
- `LOWER` tier: `tier_medape=4.248954`, `tier_p90_mape=11.049137`, `abs_tier_mean_me=0.775915` (PASS)
- Primary acceptance: `all_scored_tiers_pass_primary=true`

### Prediction Interval Snapshot (Primary Window, Winner B-II)

- `HIGH`: PI80 +/-`4.377292%`, PI90 +/-`5.597436%`
- `MODERATE`: PI80 +/-`11.321926%`, PI90 +/-`23.437520%`
- `LOWER`: PI80 +/-`8.337710%`, PI90 +/-`13.039024%`

### Production Validation Snapshot (IMP-19)

- Per-scenario outputs: `90` projected places + `46` balance-of-county rows.
- Hard constraints: PASS (share constraints, non-negative populations, place totals constrained to county totals, scenario ordering).
- QA artifacts: full required set emitted per scenario, including reconciliation magnitude diagnostics.

### Structural-Break Exclusion Disposition

- Human review accepted structural-break interpretations (Horace, Williston).
- No structural-break exclusions were applied to production outputs.

### Evidence Artifacts

- `docs/reviews/2026-02-28-pp3-imp09-backtest-results.md`
- `docs/reviews/pp3-backtest-outlier-narrative.md`
- `docs/reviews/pp3-end-to-end-validation.md`
- `docs/reviews/2026-03-01-pp3-imp19-approval-gate.md`
- `docs/reviews/2026-03-01-pp3-imp20-results.md`

## References

1. Smith, S. K., Tayman, J., & Swanson, D. A. (2001). *State and Local Population Projections: Methodology and Analysis*. Kluwer Academic/Plenum Publishers. Chapter 8: Subcounty Population Projections.

2. Wilson, T. (2015). "New evaluations of simple models for small area population forecasts." *Population, Space and Place*, 21(4), 335-353.

3. Rayer, S., & Smith, S. K. (2010). "Factors affecting the accuracy of subcounty population forecasts." *Journal of Planning Education and Research*, 30(2), 147-161.

## Revision History

- **2026-03-01**: Status changed from Deferred to Accepted; implemented date added; implementation results added (winner B-II, backtest metrics, IMP-19 validation and human sign-off)
- **2026-02-28**: Deferral rationale updated to reflect ADR-054 implementation; city-level work remains deferred for sequencing/readiness, not aggregation inconsistency
- **2026-02-23**: Deferred — state-county aggregation discrepancy (ADR-054) must be resolved before adding city layer
- **2026-02-02**: Initial proposal (ADR-033)

## Related ADRs

- ADR-004: Core projection engine (county-level cohort-component)
- ADR-010: Geographic scope and granularity (three-level hierarchy)
- ADR-012: Output formats (would need city-level additions)
