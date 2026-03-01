# ADR-058: Multi-County Place Splitting

## Status
Proposed

## Date
2026-03-01

## Context

Seven North Dakota incorporated places span multiple counties. Phase 1 place projections (PP-003) assign each place to its primary county only, ignoring the multi-county distribution. While this is conservative and affects only 7 of 357 places (1.96%), proper handling would allocate population across constituent counties, project per county, and reaggregate to the place level.

The crosswalk builder (`build_place_county_crosswalk.py`) already computes TIGER-based area overlaps for these places, stored in `place_county_crosswalk_2020_multicounty_detail.csv` (14 rows, 7 places).

### Requirements
- Allocate multi-county place population to constituent counties using defensible weights
- Project each county's share of the place independently using the existing share-trending engine
- Reaggregate county-level projections back to the place level
- Maintain county-balance consistency across all constituent counties

### Challenges
- Historical population data for multi-county places may not be disaggregated by county
- Area-based allocation may not reflect actual population distribution within a place
- County-balance reconciliation becomes more complex with shared places

## Decision

*To be completed during implementation.*

## Consequences

### Positive
1. More accurate projections for multi-county places
2. Proper county-balance accounting across jurisdictions
3. Eliminates the primary-county-only simplification

### Negative
1. Adds complexity to the projection orchestrator
2. Allocation weights introduce a methodological assumption
3. Affects only 7 places — limited practical impact

## Implementation Notes

### Key Functions/Classes
- `split_multicounty_place()`: Fan-out place to constituent county shares
- `reaggregate_multicounty_place()`: Fan-in county projections back to place total
- Reuses: `trend_all_places_in_county()`, `reconcile_county_shares()` from existing modules

### Configuration Integration
New `multicounty_allocation` block in `projection_config.yaml`.

### Testing Strategy
Unit tests for split/reaggregate roundtrip, county-balance invariants, allocation weight computation.

### Documentation
- [ ] Update `docs/methodology.md` with multi-county handling approach

## References

1. ADR-033: City-Level Projection Methodology (parent methodology)
2. TIGER 2020 shapefiles: `data/interim/geographic/tiger2020/`
3. Multi-county detail: `data/processed/geographic/place_county_crosswalk_2020_multicounty_detail.csv`

## Revision History

- **2026-03-01**: Initial proposal (ADR-058) - Multi-county place splitting methodology

## Related ADRs

- ADR-033: City-Level Projection Methodology (extends place handling)
- ADR-010: Geographic Scope (county/place FIPS relationships)
