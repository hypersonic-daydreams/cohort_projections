# ADR-046: High Growth Scenario via BEBR Convergence Rates

## Status
Accepted

## Date
2026-02-18

## Last Reviewed
2026-02-18

## Scope
Replace broken multiplicative `+15_percent` migration adjustment with scenario-specific convergence rates derived from BEBR high scenario

**Related**: [ADR-037](037-cbo-grounded-scenario-methodology.md) (CBO-grounded scenarios), [ADR-036](036-migration-averaging-methodology.md) (BEBR averaging), [ADR-039](039-international-only-migration-factor.md) (intl-only factor), [ADR-043](043-migration-rate-cap.md) (rate cap)

## Context

### Problem: Multiplicative Scaling of Signed Net Migration Rates

The `high_growth` scenario was configured with `migration: "+15_percent"`, which multiplied all net migration rates by 1.15 inside `apply_migration_scenario()`. Because 45 of 53 North Dakota counties have negative net migration (net out-migration), multiplying by 1.15 made out-migration more negative, producing **lower** population than baseline -- the exact opposite of the scenario's intent.

| Year Offset | Counties with Net Out-Migration | Effect of x1.15 |
|-------------|:-------------------------------:|-----------------|
| 1           | 45 of 53                        | Amplifies out-migration |
| 5           | 43 of 53                        | Amplifies out-migration |
| 10          | 43 of 53                        | Amplifies out-migration |
| 20          | 47 of 53                        | Amplifies out-migration |

The high_growth scenario models "counterfactual continuation of elevated post-2020 immigration trends" (ADR-037). This is an additive concept: more people arrive, regardless of whether the receiving county has positive or negative net migration. A multiplicative scaling of total net migration rates cannot achieve this when most rates are negative.

### Why Not a Multiplicative Factor with intl_share Decomposition?

The restricted_growth scenario uses the `intl_share` mechanism to apply CBO factors only to the international component. Applying a factor > 1.0 for high_growth was considered (Option B), but it fails for the same reason: the `intl_share` decomposition applies the factor proportionally to the total net rate. When the total rate is negative, scaling the "international component" (which is a proportion of a negative number) makes it more negative, not more positive.

### Available Data: BEBR High Scenario Migration Rates

The BEBR multi-period averaging pipeline already produces `migration_rates_pep_high.parquet`, which uses the maximum period mean for each county (the most optimistic historical period). This file has the desirable property of always producing higher migration than baseline for all 53 counties:

| Metric | Baseline | High | Difference |
|--------|:--------:|:----:|:----------:|
| State total net migration | +1,485 | +2,787 | **+1,302** |
| Counties higher than baseline | -- | 53/53 | All |
| Per-county surplus range | -- | -- | +1 to +216 |

The +1,302/year increment is well-grounded by three independent estimates:

| Derivation Method | Annual Increment | Source |
|-------------------|:----------------:|--------|
| CBO Jan 2025 elevated vs long-term, at ND share | ~1,163 | CBO 60875; PEP 2023-2025 |
| ND PEP international surge (50% of excess) | ~1,243 | PEP components |
| BEBR high-vs-baseline rate file difference | 1,302 | `migration_rates_pep_high.parquet` |

## Decision

### Route BEBR High Scenario Rates Through the Convergence Pipeline (Option D)

Generate scenario-specific convergence rates by computing the per-county additive difference between BEBR high and baseline scenarios, converting to a per-cell rate increment, and adding it to all three convergence window averages (recent, medium, longterm) before convergence interpolation.

The implementation has four parts:

**1. Convergence pipeline modification**: The `run_convergence_pipeline()` function accepts a `variant` parameter. When `variant="high"`:
- Load BEBR baseline and high PEP files
- Compute per-county absolute difference (high - baseline net migration)
- Convert to per-cell rate increment using `pop_start` from residual rates as the population reference, distributed uniformly across 36 age-group x sex cells per county
- Add the increment to all three window averages (recent, medium, longterm) before interpolation
- Output to `convergence_rates_by_year_high.parquet` with variant metadata

**2. Engine routing**: A new `convergence_variant` config parameter in the scenario definition tells the projection runner to load `convergence_rates_by_year_{variant}.parquet` instead of the baseline file.

**3. Configuration**: The high_growth scenario uses `migration: "recent_average"` (no in-engine adjustment) with `convergence_variant: "high"` to load the boosted rates.

**4. Dead code removal**: The `+15_percent` migration scenario handler is removed from the migration module and projection runner.

### Configuration

```yaml
high_growth:
  name: "High Growth (Elevated Immigration)"
  description: "Counterfactual: BEBR-optimistic migration rates representing sustained elevated immigration"
  fertility: "+5_percent"
  mortality: "improving"
  migration: "recent_average"        # No in-engine adjustment; scenario baked into convergence rates
  convergence_variant: "high"        # Load convergence_rates_by_year_high.parquet
  active: true
```

### Rate Increment Computation

For each county, the additive rate increment per cell is:

```
county_diff = high_net_migration - baseline_net_migration   (absolute persons)
pop_ref = median pop_start across residual periods          (persons)
n_cells = 36                                                (18 age groups x 2 sexes)
rate_increment = county_diff / pop_ref / n_cells            (annual rate per cell)
```

This increment is added uniformly to all 36 cells for each county, lifting all three window averages before convergence interpolation. The convergence schedule (5-10-5) then interpolates the lifted rates as usual.

### Why Additive, Not Multiplicative

The increment is added to each cell's rate rather than multiplied because:
1. An additive increment guarantees high >= baseline regardless of the sign of the base rate
2. It models the real-world process: additional immigrants arrive, adding to the population
3. 43 of 53 counties have negative baseline rates; a multiplicative approach cannot increase population for these counties

## Consequences

### Positive

1. **Correct directionality**: High growth scenario always produces higher population than baseline, for all 53 counties, all 30 years, all age-sex cells
2. **Empirically grounded**: The increment derives from BEBR multi-period averaging methodology, not arbitrary percentages
3. **County-specific**: Each county's increment reflects its own historical high-vs-baseline migration difference, rather than a uniform statewide factor
4. **No engine code changes**: The projection engine is unchanged; the difference is in the input rates
5. **Consistent with convergence framework**: Both baseline and high scenarios use the same convergence interpolation schedule (5-10-5), differing only in the window average levels
6. **Validated properties**: Verified that high >= baseline for all 57,240 cells (55,869 strictly higher, 1,371 equal at rate cap ceiling)

### Negative

1. **Pipeline complexity**: Generating convergence rates requires running `01b_compute_convergence.py --all-variants` to produce both baseline and high files
2. **Conceptual mixing**: The BEBR "high" (most optimistic historical period) has a different conceptual basis than "CBO-elevated immigration." For oil counties, the most optimistic period (2010-2014) reflects domestic boom migration, not international immigration
3. **Rate cap interaction**: Both baseline and high scenarios are subject to the same rate cap (ADR-043). In 1,371 cells the high rate equals the baseline rate because both hit the cap ceiling

### Files Modified

| File | Change |
|------|--------|
| `cohort_projections/data/process/convergence_interpolation.py` | Added `variant` parameter, `_compute_high_scenario_rate_increment()`, `_lift_window_averages()` |
| `scripts/pipeline/01b_compute_convergence.py` | Added `--variant` and `--all-variants` CLI arguments |
| `scripts/pipeline/02_run_projections.py` | Added `_load_scenario_convergence_rates()`, removed `+15_percent` handlers |
| `cohort_projections/core/migration.py` | Removed `+15_percent` scenario handler |
| `config/projection_config.yaml` | Updated high_growth scenario configuration |
| `scripts/exports/_methodology.py` | Updated methodology text for high_growth description |

### New Files

| File | Description |
|------|-------------|
| `data/processed/migration/convergence_rates_by_year_high.parquet` | High-scenario convergence rates (57,240 rows) |
| `data/processed/migration/convergence_metadata_high.json` | Metadata for high-scenario convergence rates |

### Pipeline Rerun Required

1. **Step 01b**: `python scripts/pipeline/01b_compute_convergence.py --all-variants` (generates both baseline and high convergence rates)
2. **Step 02**: Population projections (loads high convergence rates for high_growth scenario)
3. **Step 03**: Export workbooks

## Implementation Notes

### Key Functions

- `_compute_high_scenario_rate_increment(config)`: Loads BEBR baseline/high PEP files, computes per-county per-cell rate increment
- `_lift_window_averages(df, increment_df)`: Adds increment to a window-average DataFrame by merging on `county_fips` and `age_group`/`sex`
- `_load_scenario_convergence_rates(config, scenario)`: Checks for `convergence_variant` in scenario config and loads the corresponding parquet file

### Schema Mismatch Handling

The BEBR PEP files use single-year ages and race/ethnicity, while the convergence pipeline uses 5-year age groups and sex. The increment computation:
1. Aggregates single-year-age BEBR data to county-level totals
2. Computes the absolute migration difference per county
3. Distributes uniformly across the 36 age-group x sex cells using population-referenced rates

### Rate Cap Consistency

Both baseline and high convergence rates are generated with the same rate cap settings (ADR-043: +/-15% for ages 15-24, +/-8% for others). This ensures the only difference between the two files is the additive increment.

## References

1. **Design Document**: `docs/reviews/2026-02-18-sanity-check-investigations/design-additive-migration-boost.md`
2. **Finding 7**: `docs/reviews/2026-02-18-sanity-check-investigations/finding-7-high-growth-inversion.md`
3. **ADR-037**: CBO-Grounded Scenario Methodology
4. **ADR-039**: International-Only Migration Factor
5. **ADR-043**: Age-Aware Migration Rate Cap
6. **BEBR Methodology**: Smith, Tayman, Swanson (2001), *State and Local Population Projections*

## Revision History

- **2026-02-18**: Initial version -- Replace multiplicative `+15_percent` with BEBR convergence-based high scenario

## Related ADRs

- **ADR-036: Migration Averaging Methodology** -- Defines the BEBR multi-period averaging that produces the baseline and high scenario files
- **ADR-037: CBO-Grounded Scenario Methodology** -- Defines the three-scenario framework (baseline, restricted_growth, high_growth)
- **ADR-039: International-Only Migration Factor** -- Analogous decomposition for the restricted_growth scenario
- **ADR-043: Age-Aware Migration Rate Cap** -- Rate cap applied to both baseline and high convergence rates
