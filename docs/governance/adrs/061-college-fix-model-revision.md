# ADR-061: College Fix Model Revision (m2026r1)

## Status
Proposed

## Date
2026-03-04

## Scope
Four coordinated improvements to the m2026 projection model, motivated by Census Bureau "College Fix" research and backtesting accuracy analysis.

## Context

### Problem: Systematic Under-Projection Bias Growing with Horizon Length

Walk-forward validation and sensitivity analysis revealed systematic under-projection bias growing with horizon length. Census Bureau "College Fix" research (UMDI methodology) showed our GQ-based partition and rate-blending approach is solving the right problem with insufficient granularity.

Three specific issues compound to create excessive conservatism in urban/college counties:

1. **College-age smoothing covers only ages 15-24, missing the asymmetric 25-29 departure signal.** The IRS-based migration data captures graduates filing independently but misses incoming freshmen who remain dependents. This creates an asymmetric administrative data artifact that inflates apparent out-migration at ages 25-29 in college counties.

2. **Phase 2 GQ correction subtracts 100% of GQ from historical population snapshots** -- broader than the Census approach. The Census "College Fix" partitions at projection time only; our Phase 2 modifies historical snapshots used for rate computation. Phase 2 is the 3rd-largest sensitivity factor (37,084-person impact on 2050 projection).

3. **The 5-10-5 convergence schedule converges to long-run averages too quickly** for a state with structural growth. ND has consistently grown faster than long-run equilibrium; convergence to historical average prematurely suppresses growth. Convergence schedule was identified as a 3.44 pp state error swing in sensitivity analysis.

### Motivation from Census Bureau Research

The Census Bureau "College Fix" research (UMDI V2022/V2024 methodology) demonstrated that enrollment-based partitioning of college-age populations produces more accurate estimates than administrative-data-based approaches. Key findings relevant to our pipeline:

- The College Fix covers ages 15-19, 20-24, **and** 25-29 because the departure signal at 25-29 is the amplified side of the administrative data asymmetry
- The Census approach does NOT modify historical rate computation -- it partitions at projection time only
- Our Phase 2 (ADR-055) modifies historical snapshots, which is broader than what the Census methodology validates

## Decision

### Decision 1: Extend College-Age Smoothing to Ages 25-29

**Decision**: Add "25-29" to both `college_age.age_groups` and `rate_cap.college_ages` in `projection_config.yaml`.

**Rationale**:
- Census College Fix covers 15-19, 20-24, AND 25-29 because the departure signal at 25-29 is the amplified side of the administrative data asymmetry (IRS captures graduates filing independently but misses incoming freshmen who remain dependents)
- ADR-049's existing smoothing function already accepts `age_groups` as a parameter -- this is a config-only change
- The 25-29 age group is where the college cycle departure signal is strongest, yet it was excluded from the original ADR-049 smoothing

**Implementation**:
```yaml
college_age:
  age_groups: ["15-19", "20-24", "25-29"]  # Extended per ADR-061
rate_cap:
  college_ages: ["15-19", "20-24", "25-29"]  # Extended per ADR-061
```

**Alternatives Considered**:
- Keep smoothing at 15-24 only: Rejected -- leaves the departure-signal gap identified by Census research unaddressed
- Smooth all ages 15-34: Rejected -- too broad, masks real migration patterns in 30-34 age group

### Decision 2: Parameterize GQ Correction Fraction

**Decision**: Add `fraction` parameter (default 1.0) to `subtract_gq_from_populations()` in `residual_migration.py`. Configurable via `gq_correction.fraction` in `projection_config.yaml`.

**Rationale**:
- Census approach does NOT modify historical rate computation -- it partitions at projection time only. Our Phase 2 modifies historical snapshots, which is broader. The fraction parameter allows calibrating the correction intensity.
- Phase 2 is the 3rd-largest sensitivity factor (37,084-person impact on 2050 projection)
- Enables testing Phase 2 at reduced levels (0.5) or disabled (0.0 = Phase 1 only) without code changes

**Implementation**:
```python
def subtract_gq_from_populations(
    populations: pd.DataFrame,
    gq_historical: pd.DataFrame,
    fraction: float = 1.0,  # ADR-061: 1.0 = full Phase 2; 0.5 = half; 0.0 = Phase 1 only
) -> pd.DataFrame:
    ...
```

```yaml
gq_correction:
  fraction: 1.0  # ADR-061: 1.0 = full; 0.5 = half; 0.0 = Phase 1 only
```

**Alternatives Considered**:
- Complete revert of Phase 2: Rejected -- too aggressive; fractional parameter allows calibration instead
- Binary on/off toggle: Rejected -- fractional parameter is strictly more flexible

### Decision 3: Extend Convergence Hold Period

**Decision**: Walk-forward m2026r1 variant uses extended medium hold: years 6-20 at medium rate (vs 6-15 in baseline 5-10-5).

**Rationale**:
- ND has consistently grown faster than long-run equilibrium; convergence to historical average prematurely suppresses growth
- Convergence schedule was identified as a 3.44 pp state error swing in sensitivity analysis
- Walk-forward infrastructure now supports direct method comparison to validate the extended schedule before production deployment

**Implementation**:
```python
def get_convergence_rate_for_year_r1(year: int, ...) -> float:
    """Extended medium hold: years 6-20 at medium rate."""
    ...

def prepare_m2026r1_convergence_rates_annual(...) -> pd.DataFrame:
    """Rate preparation for m2026r1 variant."""
    ...
```

**Alternatives Considered**:
- Keep 5-10-5 schedule unchanged: Rejected -- sensitivity analysis shows this is a major source of under-projection bias
- Fully flat (no convergence): Rejected -- some convergence to long-run is still needed to prevent runaway extrapolation

### Decision 4: Expand College County List Based on Enrollment Data

**Decision**: Expand the `college_age.counties` list in `projection_config.yaml` from 4 counties to 12, using on-campus face-to-face enrollment as a percentage of county population, with a 2.5% threshold.

**Rationale**:
- The original 4-county list (Grand Forks, Cass, Ward, Burleigh) covered only the largest NDUS institutions and missed counties where enrollment-to-population ratios are actually higher
- NDUS Annual Enrollment Report data (2024-2025) provides on-campus face-to-face headcount by institution, which approximates students physically present in each county
- Richland County (NDSCS, 10.6% ratio) and Barnes County (VCSU, 7.3% ratio) were completely excluded despite having higher ratios than Ward (2.9%) and Burleigh (3.8%)
- Two private institutions (University of Mary, University of Jamestown) are included with estimated on-campus enrollment

**Data source**: NDUS Annual Enrollment Reports (AY 2001-2002 through 2024-2025), supplemented by IPEDS/institutional data for private institutions. See `data/raw/enrollment/DATA_SOURCE_NOTES.md`.

**Threshold**: Counties are included if on-campus face-to-face enrollment exceeds 2.5% of the county's 2025 PEP population. Williams County (WSC, 1.5%) is included below threshold because WSC enrollment still produces measurable enrollment turnover artifacts in the 15-24 age migration data; oil-boom dampening (ADR-040) and college-age smoothing address different distortions (economic in-migration vs enrollment turnover) and do not double-count.

**Implementation**:

| County | FIPS | Institution(s) | On-Campus F2F | Population | Ratio |
|--------|:----:|----------------|:-------------:|:----------:|:-----:|
| Grand Forks | 38035 | UND | 9,973 | 74,501 | 13.4% |
| Richland | 38077 | NDSCS | 1,780 | 16,731 | 10.6% |
| Barnes | 38003 | VCSU | 774 | 10,573 | 7.3% |
| Traill | 38097 | MASU | 487 | 7,920 | 6.1% |
| Cass | 38017 | NDSU | 11,084 | 201,794 | 5.5% |
| Bottineau | 38009 | DCB | 342 | 6,284 | 5.4% |
| Stutsman | 38093 | U of Jamestown | ~900 | 21,414 | ~4.2% |
| Burleigh | 38015 | BSC + U of Mary | ~3,937 | 103,251 | 3.8% |
| Stark | 38089 | DSU | 1,012 | 34,013 | 3.0% |
| Ramsey | 38071 | LRSC | 351 | 11,530 | 3.0% |
| Ward | 38101 | MISU | 1,953 | 68,233 | 2.9% |
| Williams | 38105 | WSC | 633 | 41,767 | 1.5% |

**Note on distance education**: Only on-campus face-to-face enrollment is counted. Distance education students may reside anywhere and should not affect local migration rate smoothing. The NDUS reports break down enrollment by delivery mode starting in 2018; earlier reports used different categories (IVN, Internet).

**Note on GQ allocation profiles**: The expanded county list also affects the GQ age allocation profiles in `scripts/data/fetch_census_gq_data.py`. Counties with college dormitories use a college-weighted GQ profile (concentrated in ages 18-29) rather than the general GQ profile. The `COLLEGE_COUNTY_FIPS` set must be updated to match.

**Alternatives Considered**:
- Keep original 4-county list: Rejected -- leaves 7 counties with significant student populations unsmoothed, particularly Richland (10.6%) which has a higher ratio than most original counties
- Use total enrollment instead of on-campus F2F: Rejected -- distance education students may be anywhere; on-campus F2F approximates students physically present in the county
- Include tribal colleges: Deferred -- student population dynamics differ (local population base); may be revisited with better data

## Alternatives Considered

### Alternative 1: Enrollment-Based Partition (Census UMDI Approach)

**Description**: Use ACS PUMS enrollment data to identify enrolled students and partition them separately, matching the Census Bureau's College Fix methodology more precisely.

**Pros**:
- More complete match to Census methodology
- Could identify enrolled students regardless of housing type (on-campus vs off-campus)

**Cons**:
- Requires ACS PUMS enrollment data not currently in the pipeline
- Additional data dependency and processing complexity

**Why Rejected**: Identified as future work. The three changes in this ADR address the most impactful gaps using data already available.

### Alternative 2: Separate ADRs Per Change

**Description**: Document each of the three changes as an independent ADR.

**Pros**:
- Cleaner ADR scope
- Easier to accept/reject independently

**Cons**:
- The three changes share motivation (Census College Fix research + backtesting)
- The changes interact with each other (e.g., extending smoothing to 25-29 interacts with GQ fraction)

**Why Rejected**: The three changes are coordinated improvements motivated by the same research; a single ADR better captures their shared context and interactions.

## Consequences

### Positive

1. **Addresses the departure-signal gap at ages 25-29** identified by Census research. The administrative data asymmetry (IRS captures departures but not arrivals of freshmen dependents) is now smoothed on both legs of the college cycle.
2. **Provides configurable GQ correction intensity** instead of binary on/off. The fraction parameter allows calibration to find the optimal level between full Phase 2 (1.0) and Phase 1 only (0.0).
3. **Walk-forward infrastructure now supports 3+ method comparison** for future iterations. The method registry (`METHOD_DISPATCH`) maps method names to rate-prep and projection functions.
4. **All changes are backward-compatible and can be individually toggled.** Config changes are additive; existing behavior is preserved at default parameter values.
5. **College-age smoothing now covers all ND counties with significant student populations.** The enrollment-data-driven threshold replaces ad hoc selection, making the methodology more systematic and reproducible.

### Negative

1. **Extending smoothing to 25-29 may over-dampen** in counties where 25-29 migration is genuinely high. The statewide blend dilutes county-specific signals at these ages.
2. **Relaxing convergence schedule increases long-horizon sensitivity** to recent period anomalies. If the recent period is atypical (e.g., Bakken boom), the extended hold propagates that signal further.
3. **Additional complexity in the walk-forward validation script.** The method registry and CLI args add configuration surface area.

### Risks and Mitigations

**Risk**: Extended smoothing overcorrects for 25-29 age group.
- **Mitigation**: Walk-forward validation with separate method variant provides direct accuracy comparison before production deployment.

**Risk**: GQ fraction at 0.0 removes legitimate GQ correction signal.
- **Mitigation**: Testing at both 0.0 and 0.5 to find optimal level; full Phase 2 (1.0) remains the default.

**Risk**: College-age smoothing and oil-boom dampening (ADR-040) both affect Williams County ages 25-29, potentially double-dampening.
- **Mitigation**: The two corrections address different distortions: oil dampening reduces oil-driven working-age in-migration magnitude, while college smoothing blends college-age rates with the statewide average to remove enrollment turnover artifacts. Walk-forward validation should empirically assess whether the combined effect over-corrects for Williams. If so, Williams can be removed from the college county list without affecting the oil dampening.

**Risk**: Extended convergence hold propagates recent-period anomalies.
- **Mitigation**: Walk-forward backtesting compares m2026r1 against m2026 and sdc_2024 methods across multiple origin years; production config change requires validation results review (Tier 3 human approval gate).

## Implementation Notes

### Key Files Modified

| File | Change |
|------|--------|
| `config/projection_config.yaml` | `age_groups`, `college_ages`, `gq_correction.fraction` |
| `cohort_projections/data/process/residual_migration.py` | `fraction` parameter on `subtract_gq_from_populations()` |
| `scripts/analysis/walk_forward_validation.py` | Method registry, CLI args, m2026r1 variant |
| `scripts/analysis/sensitivity_analysis.py` | `--run-label` CLI arg |
| `scripts/data/fetch_census_gq_data.py` | `COLLEGE_COUNTY_FIPS` set expanded to 12 counties |

### Key Functions

- `subtract_gq_from_populations(populations, gq_historical, fraction=1.0)`: Now accepts fractional subtraction
- `apply_college_age_adjustment()`: Already supports custom `age_groups` via parameter; config extended to include 25-29
- `prepare_m2026r1_convergence_rates_annual()`: New function for revised variant rate preparation
- `get_convergence_rate_for_year_r1()`: Extended medium hold (years 6-20)
- `_project_annual_core()`: Refactored shared projection loop parameterized by convergence getter
- `METHOD_DISPATCH`: Registry mapping method names to rate-prep and projection functions

### Configuration Integration

```yaml
college_age:
  age_groups: ["15-19", "20-24", "25-29"]  # Extended per ADR-061
  # Counties with on-campus F2F enrollment >2.5% of population (2024-2025 data)
  counties: ["38003", "38009", "38015", "38017", "38035", "38071", "38077", "38089", "38093", "38097", "38101", "38105"]
gq_correction:
  fraction: 1.0  # ADR-061: 1.0 = full; 0.5 = half; 0.0 = Phase 1 only
rate_cap:
  college_ages: ["15-19", "20-24", "25-29"]  # Extended per ADR-061
```

### Testing Strategy

- 5 new unit tests (1 college-age 25-29, 4 GQ fraction tests)
- Walk-forward validation with `--methods sdc_2024 m2026 m2026r1 --run-label m2026r1_final`
- Tier 3 human approval gate: production config changes require validation results review

### Documentation

- [x] Update `docs/methodology.md` -- Sections 5b, 5f, 5h, 5i
- [x] ADR-049 amended: age group extension noted
- [x] ADR-055 amended: fractional GQ correction noted

## References

1. **Census Bureau "College Fix" research**: `docs/reviews/2026-03-04-college-town-estimates-research.pdf`
2. **College Fix implications analysis**: `docs/reviews/2026-03-04-college-fix-research-implications.md`
3. **Projection accuracy analysis**: `docs/reviews/2026-03-04-projection-accuracy-analysis.md`
4. **UMDI V2022/V2024 Population Projections Methodology** (cited in Census Bureau research)

## Revision History

- **2026-03-04**: Added Decision 4 -- expanded college county list from 4 to 12 based on NDUS enrollment data (on-campus F2F / population >2.5% threshold)
- **2026-03-04**: Initial version (ADR-061) -- Proposed; three coordinated improvements motivated by Census College Fix research

## Related ADRs

- **ADR-049**: College-Age Smoothing -- age groups extended to 25-29 by this ADR
- **ADR-055**: Group Quarters Separation -- Phase 2 fraction parameter added by this ADR
- **ADR-043**: Migration Rate Cap -- `college_ages` config extended to 25-29
- **ADR-036**: Migration Averaging Methodology -- convergence schedule adjustment
- **ADR-057**: Rolling-Origin Backtests -- validation framework used for method comparison
