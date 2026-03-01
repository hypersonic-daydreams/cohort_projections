# ADR-060: Housing-Unit Method for Place Projections

## Status
Proposed

## Date
2026-03-01

## Context

ADR-033 identified the housing-unit method as an alternative approach for place-level projections: project housing units, then apply persons-per-household (PPH) ratios to derive population. While share-of-county trending (the Phase 1 winner) is well-suited for the 30-year projection horizon (2025-2055), the housing-unit method provides a complementary short-term (5-10 year) perspective that can serve as a cross-check and may be more appropriate for municipal planning use cases where housing development patterns are the primary driver.

### Requirements
- Acquire place-level housing unit counts and average household size from Census ACS
- Trend housing units for each place (linear or log-linear)
- Apply projected PPH ratios to derive short-term population estimates
- Cross-validate against share-trending outputs for consistency diagnostics
- Produce supplementary output that complements (not replaces) the primary share-trending projections

### Challenges
- Census ACS place-level housing data may have gaps for small places
- PPH ratios change over time — need trend or scenario-based handling
- Building permit data (local sources) is not readily available for all ND places
- This is a complementary method, not a replacement — need clear communication about its role

## Decision

*To be completed during implementation.*

## Consequences

### Positive
1. Provides independent cross-check on share-trending projections
2. More intuitive for municipal stakeholders (housing = population driver)
3. Better suited for short-term planning horizons (5-10 years)
4. Can incorporate local housing development intelligence when available

### Negative
1. Requires new data pipeline (Census ACS housing tables)
2. Limited to short-term horizons where housing trend extrapolation is defensible
3. May diverge from share-trending outputs, requiring explanation

## Implementation Notes

### Key Functions/Classes
- `trend_housing_units()`: Fit trend model to housing unit history
- `project_population_from_hu()`: Apply PPH ratios to projected HUs
- `run_housing_unit_projections()`: Per-place orchestration
- Extends: `CensusDataFetcher` with ACS housing variables (B25001, B25010)

### Configuration Integration
New `housing_unit_method` block in `projection_config.yaml`.

### Testing Strategy
Unit tests for HU trending, PPH application, projection orchestration. Integration tests for pipeline stage. Cross-validation diagnostic against share-trending outputs.

### Documentation
- [ ] Update `docs/methodology.md` with housing-unit method documentation

## References

1. ADR-033: City-Level Projection Methodology (Alternative 3: housing-unit method)
2. Census ACS Table B25001: Housing Units
3. Census ACS Table B25010: Average Household Size
4. Sibling housing analysis repo: `~/workspace/demography/housing/`

## Revision History

- **2026-03-01**: Initial proposal (ADR-060) - Housing-unit method for complementary place projections

## Related ADRs

- ADR-033: City-Level Projection Methodology (complementary approach)
- ADR-006: Data Pipeline Architecture (data ingestion patterns)
