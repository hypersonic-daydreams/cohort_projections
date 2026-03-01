# ADR-060: Housing-Unit Method for Place Projections

## Status
Accepted

## Date
2026-03-01

## Context

The cohort projection system produces place-level population projections
using county-constrained share-trending (PP-003).  Share-trending works
well for medium-to-long horizons but rests on assumptions about
historical population share trajectories.  An independent cross-check
using a different methodology strengthens confidence in the near-term
place projections published by the State Data Center.

The housing-unit (HU) method is a well-established short-term projection
technique used by state and local demographers.  It estimates population
as:

    population = housing_units x persons_per_household

ACS 5-year estimates provide both housing-unit counts (Table B25001) and
average household size (Table B25010) for all incorporated places in
North Dakota, covering vintages from 2009 through 2023.

### Requirements
- Provide an independent short-term (5-10 year) cross-check for share-trending.
- Use publicly available Census ACS data.
- Produce place-level population estimates comparable to the share-trending outputs.
- Quantify divergence between the two methods for diagnostic reporting.

### Challenges
- ACS 5-year estimates have wide margins of error for small places.
- PPH has been declining nationally, making the "hold last" assumption
  modestly conservative.
- Housing-unit counts reflect permits and completions, which are forward-looking
  but noisy for small geographies.

## Decision

### Decision 1: Complementary Cross-Check, Not Replacement

**Decision**: The housing-unit method supplements share-trending.  It does
not replace or override the share-trending projections in the production
pipeline.

**Rationale**:
- Share-trending is county-constrained and age-sex detailed; HU produces
  total population only.
- HU is most useful in the near-term (5-10 years) where building permit
  trends are informative.
- Divergence between the two methods flags places requiring analyst review.

### Decision 2: Log-Linear Default Trend

**Decision**: Use log-linear (exponential) trend as the default for
projecting housing-unit counts forward.

**Rationale**:
- Housing growth in growing ND cities (Fargo, Bismarck, Williston) is
  closer to multiplicative than additive.
- Linear trend is available as an alternative via config.
- Log-linear naturally prevents negative projections.

### Decision 3: Hold-Last PPH Default

**Decision**: Default PPH projection is "hold last observed value constant".

**Rationale**:
- National PPH has declined slowly (~0.02/decade).  For a 5-10 year
  horizon, holding constant is a reasonable simplification.
- Linear PPH trending is available as a config option for sensitivity
  analysis.
- PPH is floored at 1.0 to prevent nonsensical values.

### Decision 4: ACS 5-Year Data Source

**Decision**: Use Census ACS 5-year estimates (Tables B25001, B25010)
fetched via the Census API for all available vintages (2009-2023).

**Rationale**:
- ACS 5-year estimates cover all places, including small CDPs.
- Multiple vintages provide enough history for trend fitting.
- The fetch script (`scripts/data/fetch_census_housing_data.py`) follows
  the existing `CensusDataFetcher` patterns.

## Consequences

### Positive
1. Provides an independent cross-check for share-trending projections.
2. Uses a well-established, transparent methodology.
3. Identifies places where share-trending and HU diverge significantly.
4. Modular design: can be enabled or disabled via config.

### Negative
1. Total population only (no age-sex detail).
2. ACS margins of error are large for small places (<500 population).
3. Adds a new data dependency (ACS housing tables).

### Risks and Mitigations

**Risk**: HU projections diverge substantially from share-trending for
rapidly growing places (e.g., Watford City during oil boom).
- **Mitigation**: Cross-validation metrics flag divergences > 10% for
  analyst review.  HU is advisory, not production.

**Risk**: PPH hold-last assumption overstates population in places where
household sizes are shrinking.
- **Mitigation**: Config option to switch to linear PPH trend.

## Implementation Notes

### Key Functions/Classes
- `load_housing_data(config)`: Load ACS housing CSV.
- `trend_housing_units(history, method, years)`: Fit linear or log-linear trend.
- `project_pph(history, method, years)`: Project persons-per-household.
- `project_population_from_hu(hu, pph)`: Multiply HU x PPH.
- `run_housing_unit_projections(config)`: Orchestrate per-place projections.
- `cross_validate_with_share_trending(hu, st)`: Compute divergence metrics.

### Configuration Integration
```yaml
housing_unit_method:
  enabled: true
  housing_data_path: "data/raw/housing/nd_place_housing_units.csv"
  projection_horizon: 10
  projection_years: [2025, 2030, 2035]
  trend_method: "log_linear"
  pph_method: "hold_last"
  min_history_years: 3
```

### Testing Strategy
- 18 unit tests covering loading, trends, PPH, population computation,
  orchestration, edge cases, cross-validation, and config parsing.
- 5 integration tests covering end-to-end pipeline, dry-run, output
  format, config-disabled skipping, and multi-scenario.

### Documentation
- [ ] Update `docs/methodology.md` when HU results are included in publications

## References

1. **Smith, S.K. and Cody, S. (2004)**. "An Evaluation of Population
   Estimates in Florida." *Population Research and Policy Review*.
2. **Census Bureau ACS 5-Year Estimates**: Tables B25001, B25010.
3. **BEBR Housing-Unit Method**: University of Florida Bureau of Economic
   and Business Research methodology documentation.

## Revision History

- **2026-03-01**: Initial version (ADR-060) - Housing-unit method implementation.

## Related ADRs

- ADR-033: City-Level Projections (deferred; HU provides an alternative path)
- ADR-054: State-County Aggregation Reconciliation (HU operates at place level)
- ADR-055: Group Quarters Separation (GQ is not included in HU method)
