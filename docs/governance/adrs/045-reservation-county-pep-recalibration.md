# ADR-045: Reservation County PEP-Anchored Migration Recalibration

## Status
Accepted

## Date
2026-02-18

## Last Reviewed
2026-02-18

## Scope
Migration rate recalibration for AIAN reservation counties where the residual method systematically overestimates out-migration

**Related**: [ADR-036](036-migration-averaging-methodology.md) (migration averaging), [ADR-040](040-extend-boom-dampening-2015-2020.md) (boom dampening)

## Context

### Problem: Residual Method Overestimates Out-Migration for Reservation Counties

Three AIAN reservation counties are projected to decline 45-47% over 30 years under the baseline scenario. Investigation revealed that the residual migration method produces out-migration estimates 2-3x larger than Census PEP component estimates for these counties:

| County | Median Residual/PEP Ratio | 30-Year Projected Decline |
|--------|:------------------------:|:------------------------:|
| Benson (38005) | 2.00x | -47% |
| Sioux (38085) | 1.42x | -47% |
| Rolette (38079) | 1.45x | -46% |

The divergence arises from two sources:

1. **Census undercounts on tribal lands**: Differential census coverage between decennial censuses inflates the apparent population loss, which the residual method attributes to out-migration.
2. **Statewide survival rates applied to populations with lower life expectancy**: The residual method uses statewide survival rates. On reservations where mortality is higher, excess deaths are misattributed as out-migration because the expected surviving population is too high.

### Sign Reversal Periods

In the 2010-2015 period, PEP shows net in-migration for Sioux (+79) and Rolette (+65), while the residual method shows net out-migration (-81 and -394). This is the most extreme form of divergence, where the residual method is qualitatively wrong about the direction of migration.

### PEP vs Residual: The Bridge Problem

PEP provides reliable county-level net migration totals but has no age-sex detail. The residual method provides age-sex-specific rates needed for cohort-component projection, but its totals are unreliable for reservation counties. A recalibration approach must merge PEP totals with residual age-sex shapes.

## Decision

### Hybrid Option D: PEP-Anchored Residual Shape with Rogers-Castro Fallback

Implement a two-tier recalibration applied after period dampening and male dampening but before period averaging in the residual migration pipeline:

1. **When PEP and residual have the same sign**: Scale all age-sex rates by `k = PEP_total / Residual_total` to preserve the county-specific age-sex shape while anchoring the total to PEP.

2. **When they have different signs (sign reversal)**: Fall back to distributing the PEP total using the Rogers-Castro standard migration age pattern with a 50/50 sex split.

3. **When the residual total is near zero** (|residual| < 10 persons): Fall back to Rogers-Castro, since the shape is dominated by noise.

**Target counties**: Benson (38005), Sioux (38085), Rolette (38079).

**Rationale**:

- Options A and B (pure scaling) fail on sign-reversal periods, which affect 2 of 5 periods for Sioux and Rolette.
- Option C (pure Rogers-Castro) discards county-specific age-sex information unnecessarily when the residual method is directionally correct.
- Option D preserves the most information while handling all edge cases robustly.
- The residual age-sex shape has moderate cross-period stability (median correlation r=0.60), supporting its use when the total direction is correct.

### Configuration

```yaml
residual:
  pep_recalibration:
    enabled: true
    counties: ["38005", "38085", "38079"]  # Benson, Sioux, Rolette
    pep_data_path: "data/processed/pep_county_components_2000_2025.parquet"
    fallback_method: "rogers_castro"
    near_zero_threshold: 10
```

### Why These Three Counties

These three counties were selected because they:
- Contain significant AIAN reservation populations (Standing Rock, Turtle Mountain, Fort Berthold)
- Show the largest PEP-residual divergence (2-3x overestimation)
- Exhibit sign-reversal periods that confirm the residual method is qualitatively wrong
- All project declines exceeding 45% under the uncorrected baseline, which is implausible given historical trajectories of -11% to -15% per 20 years

For comparison, a non-reservation declining county (Pembina, 38067) shows a median Residual/PEP ratio of approximately 0.87, confirming the reservation-specific nature of the bias.

## Consequences

### Positive

1. **Realistic reservation county projections**: Declines reduced from approximately -47% to approximately -23% over 30 years, which is 1.5-2.3x historical rates rather than the implausible 2.7-11x
2. **Preserves county-specific patterns**: When the residual method is directionally correct, the age-sex shape is preserved rather than replaced with a generic model
3. **Handles edge cases**: Sign reversals and near-zero residuals are handled gracefully via Rogers-Castro fallback
4. **Non-target counties unaffected**: The recalibration is applied only to the three specified counties
5. **Audit trail**: Each county-period records which method was used and the scaling factor

### Negative

1. **Introduces asymmetric methodology**: Three counties are treated differently from the other 50, which requires documentation and justification
2. **Discontinuity in averaging**: Periods using Rogers-Castro fallback may have slightly different age-sex shapes than scaled periods, creating minor inconsistency across the 5-period average
3. **PEP data dependency**: The recalibration requires access to PEP components data, adding a data dependency to the residual pipeline

### Expected Impact

| County | Before | After (est.) | Change |
|--------|:------:|:------------:|:------:|
| Benson (38005) | -47% | ~-23% | Decline halved |
| Sioux (38085) | -47% | ~-25% | Decline halved |
| Rolette (38079) | -46% | ~-21% | Decline halved |

### Pipeline Rerun Required

1. **Step 01**: Residual migration computation (to apply recalibration)
2. **Step 01b**: Convergence interpolation (to recompute windows with recalibrated rates)
3. **Step 02**: Population projections
4. **Step 03**: Export workbooks

## Implementation Notes

### Key Files

- `config/projection_config.yaml`: `rates.migration.domestic.residual.pep_recalibration` configuration block
- `cohort_projections/data/process/residual_migration.py`: `apply_pep_recalibration()`, `_rogers_castro_to_age_group_rates()`, `_load_pep_for_recalibration()` functions
- `cohort_projections/data/process/migration_rates.py`: Existing `get_standard_age_migration_pattern()` (reused for Rogers-Castro pattern)

### Pipeline Integration Point

The recalibration is applied in `run_residual_migration_pipeline()` after:
- Period dampening (oil counties)
- Male migration dampening (boom periods)

And before:
- Period averaging
- College-age adjustment

This ensures dampening adjustments are applied first (since they affect all counties), and the PEP recalibration corrects the reservation county totals before rates are averaged across periods.

### PEP Period Convention

For each residual period (start_year, end_year), PEP annual netmig values are summed for years (start_year+1) through end_year inclusive. For example, period (2000, 2005) sums PEP years 2001-2005.

### FIPS Code Matching

The PEP data uses a `geoid` column with 5-digit FIPS codes (e.g., "38005") that matches the residual data's `county_fips` column directly. The PEP `county_fips` column contains only 3-digit codes (e.g., "005") and is not used for matching.

### Testing Strategy

1. **Unit tests**: Scaling factor computation (same sign, different sign, near-zero), Rogers-Castro fallback produces correct totals
2. **Integration test**: Full pipeline with recalibration enabled; verify non-target counties are unaffected
3. **Regression test**: Compare target county rates before and after recalibration

## References

1. **Design Document**: `docs/reviews/2026-02-18-sanity-check-investigations/design-reservation-migration-recalibration.md`
2. **Finding 3**: `docs/reviews/2026-02-18-sanity-check-investigations/finding-3-reservation-county-declines.md`
3. **Census PEP Components**: `data/processed/pep_county_components_2000_2025.parquet`
4. Rogers, A. (1988). Age patterns of elderly migration: An international comparison. *Demography*, 25(3), 355-370.

## Revision History

- **2026-02-18**: Initial version (ADR-045) -- PEP-anchored recalibration for reservation counties

## Related ADRs

- **ADR-036: Migration Averaging Methodology** -- Defines the multi-period averaging framework in which recalibration operates
- **ADR-040: Extend Boom Dampening to 2015-2020** -- Analogous county-specific rate adjustment for oil counties
- **ADR-003: Migration Rate Processing** -- Original migration processing methodology
- **ADR-035: Census PEP Components for Migration Inputs** -- Established PEP as the authoritative migration data source
