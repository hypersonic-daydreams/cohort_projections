# ADR-058: Multi-County Place Splitting

## Status
Accepted

## Date
2026-03-01

## Context

Seven North Dakota incorporated places span multiple counties. Phase 1 place projections (PP-003) assign each place to its primary county only, ignoring the multi-county distribution. While this is conservative and affects only 7 of 357 places (1.96%), proper handling allocates population across constituent counties, projects per county, then reaggregates to the place level.

The crosswalk builder (`build_place_county_crosswalk.py`) already computes TIGER-based area overlaps for these places, stored in `place_county_crosswalk_2020_multicounty_detail.csv` (14 rows, 7 places x 2 counties each).

### The 7 Multi-County Places

| Place | Primary County | Secondary County | Primary Area Share |
|-------|---------------|-----------------|-------------------|
| Enderlin (3824260) | Ransom (38073) | Cass (38017) | 98.9% |
| Grandin (3832300) | Cass (38017) | Traill (38097) | 86.3% |
| Lehr (3845740) | Logan (38051) | LaMoure (38047) | 60.1% |
| Reynolds (3866260) | Traill (38097) | Grand Forks (38035) | 66.1% |
| Sarles (3870780) | Cavalier (38019) | Towner (38095) | 74.6% |
| Tower City (3879340) | Cass (38017) | Barnes (38003) | 69.5% |
| Wilton (3886580) | McLean (38055) | Burleigh (38015) | 55.4% |

### Requirements
- Allocate multi-county place population to constituent counties using defensible weights
- Project each county's share of the place independently using the existing share-trending engine
- Reaggregate county-level projections back to the place level
- Maintain county-balance consistency across all constituent counties
- Backward compatible: when disabled, behavior is identical to Phase 1

### Challenges
- Historical population data for multi-county places is not disaggregated by county in Census subcounty estimates
- Area-based allocation may not reflect actual population distribution within a place
- County-balance reconciliation becomes more complex with shared places

## Decision

Use TIGER 2020 area-share weights as the default allocation method for distributing multi-county place population across constituent counties.

### Approach

1. **Allocation weights**: Use `area_share` from the multicounty detail CSV (already computed by the crosswalk builder from TIGER shapefiles). Weights are normalized to sum to 1.0 within each place.

2. **Share history distribution**: For each multi-county place, take its historical share entries (which exist only under the primary county) and create synthetic per-county share entries by scaling proportionally to allocation weights:
   - For county `c` with weight `w_c` and primary county weight `w_p`, the synthetic share is `s * (w_c / w_p)`.

3. **Independent projection**: Each county's share-trending engine processes its county-portion of the place independently, including reconciliation with other places in that county.

4. **Reaggregation**: After all counties are projected, sum projected populations across constituent counties to recover a single place-level total.

5. **Configuration**: Feature gated behind `place_projections.multicounty_allocation.enabled` in config. When disabled, behavior is identical to Phase 1 (primary-county-only assignment).

### Why Area-Share

- Census does not publish sub-place-by-county population breakdowns for these small places
- Area-share is the most defensible proxy available from authoritative TIGER data
- For these 7 places, the primary county dominates (55-99% area share), so allocation method sensitivity is low
- `population_share` is supported as an alternative allocation_method in config for future use if sub-place population data becomes available

## Consequences

### Positive
1. More accurate projections for multi-county places, especially those with significant secondary-county shares (Lehr 40%, Wilton 45%, Reynolds 34%)
2. Proper county-balance accounting across jurisdictions
3. Eliminates the primary-county-only simplification from Phase 1
4. Feature-gated: zero risk to existing pipeline when disabled

### Negative
1. Adds complexity to the projection orchestrator (deferred output for multi-county primary entries)
2. Allocation weights introduce a methodological assumption (area = population distribution)
3. Affects only 7 places -- limited practical impact for most users

## Implementation Notes

### Key Functions/Classes
- `multicounty_allocation.py` module with:
  - `load_allocation_weights()`: Load and normalize TIGER area-share weights
  - `split_multicounty_place()`: Fan-out population to county portions
  - `split_multicounty_shares()`: Create synthetic share history for non-primary counties
  - `reaggregate_multicounty_place()`: Fan-in county projections to place total
  - `identify_multicounty_places()`: Detect multi-county places from crosswalk
  - `prepare_multicounty_share_history()`: Augment full share history with synthetic rows
  - `get_multicounty_config()`: Extract config with defaults

### Orchestrator Integration
- Before trending: augment share history and expand crosswalk with secondary county entries
- During county loop: defer output for multicounty primary entries; skip output for secondary entries
- After county loop: reaggregate deferred multicounty places and write combined outputs
- Metadata includes `multicounty: true` and `constituent_counties` list

### Configuration
```yaml
place_projections:
  multicounty_allocation:
    enabled: true
    allocation_method: "area_share"
    multicounty_detail_path: "data/processed/geographic/place_county_crosswalk_2020_multicounty_detail.csv"
```

### Testing Strategy
38 unit tests in `tests/test_data/test_multicounty_allocation.py`:
- Identifying multi-county places from crosswalk (4 tests)
- Loading allocation weights from CSV (5 tests)
- Splitting population across counties (7 tests)
- Splitting share history across counties (6 tests)
- Reaggregating county projections (4 tests)
- Config extraction (3 tests)
- Share history augmentation (4 tests)
- Edge cases and roundtrip invariants (5 tests)

### Documentation
- [x] ADR-058 accepted with implementation details
- [ ] Update `docs/methodology.md` with multi-county handling approach

## Implementation Results

- **Tests**: 38 passed, 0 failed
- **Ruff**: All checks passed (both module and tests)
- **Mypy**: Success, no issues found
- **Existing test regression**: 0 failures (orchestrator: 4 passed, share trending: 14 passed)
- **Coverage**: multicounty_allocation.py at 88.6% (uncovered: file-not-found branches, logger calls)

## References

1. ADR-033: City-Level Projection Methodology (parent methodology)
2. TIGER 2020 shapefiles: `data/interim/geographic/tiger2020/`
3. Multi-county detail: `data/processed/geographic/place_county_crosswalk_2020_multicounty_detail.csv`

## Revision History

- **2026-03-01**: Initial proposal (ADR-058) - Multi-county place splitting methodology
- **2026-03-01**: Accepted with implementation -- area-share allocation, deferred reaggregation

## Related ADRs

- ADR-033: City-Level Projection Methodology (extends place handling)
- ADR-010: Geographic Scope (county/place FIPS relationships)
