# ADR-037: CBO-Grounded Scenario Methodology

## Status
Accepted

**Amended by**:
- [ADR-039](039-international-only-migration-factor.md): International-only migration factor (CBO factor applies to international migration only, not total)
- [ADR-040](040-extend-boom-dampening-2015-2020.md): Extended Bakken boom dampening to 2015-2020 period

## Date
2026-02-17

## Last Reviewed
2026-02-17

## Scope
Production projection scenarios (replaces arbitrary multipliers)

**Supersedes**: Scenario definitions in [ADR-018](018-immigration-policy-scenario-methodology.md) (retains ADR-018's regression methodology proposal for future work)

## Context

### Problem: Production Scenarios Lack Empirical Basis

The cohort-component projection system for North Dakota (53 counties, 2025-2045) currently defines three growth scenarios using arbitrary multipliers:

| Scenario | Migration Adjustment | Fertility Adjustment |
|----------|---------------------|---------------------|
| High Growth | +25% | +10% |
| Low Growth | -25% | -10% |
| Baseline | No adjustment | No adjustment |

These multipliers have no empirical grounding. They were not derived from any observed data, policy analysis, or external projection. [ADR-018](018-immigration-policy-scenario-methodology.md) explicitly rejected "arbitrary multipliers" as Alternative 1, noting they provide "no empirical basis for ND-specific adjustment." Yet the production configuration ([`config/projection_config.yaml`](../../../config/projection_config.yaml)) still uses them:

```yaml
# Current production config (to be replaced)
high_growth:
  fertility: "+10_percent"
  migration: "+25_percent"

low_growth:
  fertility: "-10_percent"
  migration: "-25_percent"
```

This contradiction between the project's stated methodological standards and its actual configuration undermines the credibility of the scenario analysis.

### Opportunity: CBO Demographic Outlook Provides Empirical Anchor

Two Congressional Budget Office Demographic Outlook reports, published one year apart, provide a natural experiment for scenario construction:

- **CBO January 2025** (Publication 60875): Pre-policy baseline projecting elevated immigration trends
- **CBO January 2026** (Publication 61879): Post-enforcement revision incorporating the effects of 2025 federal immigration policy changes

The ratio of the January 2026 projections to the January 2025 projections yields year-specific adjustment factors that quantify the expected impact of immigration enforcement policy on net migration. These factors are not arbitrary -- they are derived from CBO's own modeling of observed policy actions.

### CBO Net Immigration Data

The following table presents CBO's net immigration projections from both publications and the resulting adjustment factor (Jan 2026 / Jan 2025):

| Year | CBO Jan 2025 (000s) | CBO Jan 2026 (000s) | Adjustment Factor |
|------|---------------------|---------------------|-------------------|
| 2025 | 2,010 | 408 | 0.203 |
| 2026 | 1,563 | 574 | 0.367 |
| 2027 | 1,265 | 694 | 0.549 |
| 2028 | 1,068 | 833 | 0.780 |
| 2029 | 1,070 | 969 | 0.906 |
| 2030 | 1,073 | 1,086 | 1.012 |
| 2031 | 1,075 | 1,082 | 1.007 |
| 2035 | 1,085 | 1,160 | 1.069 |
| 2036+ | ~1,090-1,111 | ~1,209-1,235 | ~1.111 |

**Source**: CBO Publication 60875 (Jan 2025) and CBO Publication 61879 (Jan 2026), net immigration tables.

The data reveals a three-phase pattern:

1. **Enforcement Shock (2025-2027):** Net immigration reduced by 80% to 45% relative to the pre-policy baseline. CBO attributes this to 600,000+ deportations, an estimated 1.9 million "voluntary self-departures," and the Other Foreign Nationals category swinging from +1.1 million to -365,000 in 2025.
2. **Recovery (2028-2030):** Gradual convergence as enforcement effects moderate and legal immigration channels adjust. By 2030, the Jan 2026 projection essentially matches the Jan 2025 projection.
3. **New Steady State (2031+):** The Jan 2026 projection actually exceeds the Jan 2025 projection by approximately 7-11%, reflecting CBO's assessment that long-run legal immigration levels will settle slightly above the pre-policy trajectory.

### CBO Fertility Revision

CBO also revised its Total Fertility Rate (TFR) projections downward between the two publications:

| Year | CBO Jan 2025 TFR | CBO Jan 2026 TFR | Percent Change |
|------|-------------------|-------------------|---------------|
| 2025 | 1.623 | 1.592 | -1.9% |
| 2035 | 1.603 | 1.534 | -4.3% |
| 2045 | 1.598 | 1.530 | -4.3% |

The long-term revision is approximately -4.3%, reflecting updated data on fertility intentions and compositional effects of reduced immigration (immigrant women tend to have higher fertility rates than native-born women).

### Population Impact

Under CBO's January 2026 projections, the U.S. population is 7.4 million (-2.0%) smaller by 2045 than under the January 2025 projections. Immigration is the dominant driver of this difference in the near term (2025-2030), while the fertility revision becomes the dominant driver over the long term as reduced immigrant cohorts reduce both the population base and the aggregate fertility rate.

### External Benchmarks

Additional external data points provide context for the scenario parameters:

- **Goldman Sachs** estimates net immigration at approximately 750,000 per year (down from a pre-pandemic average of roughly 1 million and a 2022-2023 peak of approximately 3 million)
- **CBO Jan 2025** projected 2025 net immigration at 2.01 million, approximately 82% above the historical average of roughly 1.1 million, reflecting elevated post-2020 trends
- CBO projects foreign-born TFR of approximately 1.79 versus native-born TFR of approximately 1.53, indicating that immigration levels affect aggregate fertility

## Decision

**Replace the three current production scenarios (Baseline, High Growth, Low Growth) with three empirically-grounded scenarios derived from CBO Demographic Outlook data.** The new scenarios anchor all migration and fertility adjustments to observed CBO revisions rather than arbitrary percentage multipliers.

### Scenario 1: Baseline (Trend Continuation)

**Migration**: Uses historical PEP data (2000-2024) averaged migration rates with convergence interpolation per [ADR-036](036-migration-averaging-methodology.md). No CBO adjustment applied. This represents continuation of observed historical trends.

**Fertility**: Constant (current observed rates per existing SEER-based methodology).

**Mortality**: Improving (annual improvement factor per existing methodology).

**Rationale**: This is the "what if recent trends continue" scenario. It provides the reference point against which policy-adjusted scenarios are compared. The underlying migration assumptions are already grounded in 24 years of PEP data per [ADR-035](035-migration-data-source-census-pep.md) and [ADR-036](036-migration-averaging-methodology.md).

### Scenario 2: Restricted Growth (CBO Policy-Adjusted)

**Migration**: Apply year-specific adjustment factors derived from the ratio of CBO January 2026 to CBO January 2025 net immigration projections:

| Projection Year | Adjustment Factor | Description |
|----------------|-------------------|-------------|
| 2025 | 0.20 | Enforcement shock: 80% reduction |
| 2026 | 0.37 | Continued enforcement: 63% reduction |
| 2027 | 0.55 | Moderating enforcement: 45% reduction |
| 2028 | 0.78 | Recovery underway: 22% reduction |
| 2029 | 0.91 | Near convergence: 9% reduction |
| 2030+ | 1.00 | Converged: no adjustment (returns to baseline rates) |

After convergence in 2030, the restricted scenario returns to baseline migration rates. There is no permanent penalty. The CBO data does show a slight long-term increase (factors of 1.01-1.11 in 2031+), but this ADR conservatively sets post-convergence factors to 1.00 rather than introducing a permanent upward adjustment that would require additional justification at the subnational level.

**Fertility**: -5% (grounded in CBO's approximately -4.3% TFR revision, rounded to -5% to reflect the compositional effect of reduced immigration on aggregate fertility).

**Mortality**: Improving (same as baseline).

**Rationale**: This scenario quantifies the impact of current federal immigration enforcement policy as modeled by CBO. The time-varying migration factors capture the shock-and-recovery pattern that CBO projects rather than applying a static multiplier. The fertility adjustment reflects CBO's own downward revision, which incorporates the indirect effect of reduced immigration on the fertility rate.

### Scenario 3: High Growth (Pre-Policy Elevated Immigration)

**Migration**: +15% above baseline.

**Fertility**: +5%.

**Mortality**: Improving (same as baseline).

**Rationale**: This scenario represents the counterfactual where elevated post-2020 immigration trends continued without policy intervention. The +15% migration adjustment is grounded in the following:

- CBO January 2025 projected 2025 net immigration at 2.01 million versus a historical average of approximately 1.1 million (82% above average)
- Goldman Sachs estimated a pre-pandemic average of approximately 1 million per year versus the historical IRS/Census baseline of approximately 920,000
- The +15% figure is a conservative estimate anchored to the sustained above-average trend, well below the peak but above the pre-2020 norm

The +5% fertility adjustment reflects the higher aggregate fertility associated with elevated immigration. CBO projects foreign-born TFR of approximately 1.79 versus native-born TFR of approximately 1.53; a larger immigrant population would push the aggregate fertility rate upward.

### Justification of Specific Parameter Values

| Parameter | Value | Empirical Basis |
|-----------|-------|----------------|
| Migration factors (restricted, 2025-2029) | 0.20 to 0.91 | Directly from CBO Jan 2026 / Jan 2025 ratio, year by year |
| Migration post-convergence (restricted) | 1.00 | CBO shows convergence by 2030; conservative (actual CBO ratio is 1.01) |
| Migration (high) | +15% | CBO Jan 2025 projected 82% above historical avg; +15% is a conservative sustained-trend estimate |
| Fertility (restricted) | -5% | CBO's own revision was -4.3%; rounded up to -5% to bracket CBO's uncertainty |
| Fertility (high) | +5% | Symmetric with restricted; grounded in foreign-born vs native-born TFR differential (1.79 vs 1.53) |
| Time-varying migration | Year-indexed schedule | Required because CBO shows a shock-and-recovery pattern, not a permanent shift |

### Why Time-Varying Migration Factors Are Required

A static multiplier (e.g., -50% for all years) would fundamentally misrepresent the CBO data. The CBO projects:

- An 80% reduction in 2025 (enforcement shock)
- Convergence to baseline by 2030 (policy normalization)

Applying a uniform -50% for all years would overstate migration in 2025-2026, understate migration in 2029+, and produce a qualitatively different population trajectory. The time-varying schedule preserves the shape of CBO's projection, which is its most analytically important feature.

## Implementation

### Phase 1: Configuration Schema Update

Add a `time_varying` migration scenario type to `projection_config.yaml` that supports year-indexed adjustment schedules:

```yaml
scenarios:
  baseline:
    name: "Baseline (Trend Continuation)"
    description: "Historical PEP trends with convergence interpolation; no policy adjustment"
    fertility: "constant"
    mortality: "improving"
    migration: "recent_average"
    active: true

  restricted_growth:
    name: "Restricted Growth (CBO Policy-Adjusted)"
    description: "CBO-derived immigration enforcement impact with time-varying migration factors"
    fertility: "-5_percent"
    mortality: "improving"
    migration:
      type: "time_varying"
      schedule:
        2025: 0.20
        2026: 0.37
        2027: 0.55
        2028: 0.78
        2029: 0.91
        # 2030+: implicit 1.00 (no adjustment after last specified year)
      default_factor: 1.00  # Applied to years not in schedule
    active: true

  high_growth:
    name: "High Growth (Pre-Policy Elevated Immigration)"
    description: "Counterfactual continuation of elevated post-2020 immigration"
    fertility: "+5_percent"
    mortality: "improving"
    migration: "+15_percent"
    active: true
```

**Key design decisions**:
- The `time_varying` type uses a simple year-to-factor mapping. Years not listed in `schedule` use `default_factor` (1.00).
- This is intentionally minimal. A more complex interpolation scheme could be added later, but the CBO data provides explicit year-by-year values that do not require interpolation.
- The old `high_growth` and `low_growth` scenario keys are removed entirely. The `zero_migration` and `sdc_2024` scenarios are retained for reference.

### Phase 2: Projection Engine Update

Modify the migration application logic in [`cohort_projections/core/migration.py`](../../../cohort_projections/core/migration.py) to support the `time_varying` scenario type:

```python
def get_migration_factor(scenario_config: dict, projection_year: int) -> float:
    """Return the migration adjustment factor for a given projection year.

    For 'time_varying' scenarios, looks up the year in the schedule.
    For static scenarios (e.g., '+15_percent'), returns the constant factor.
    """
    migration_config = scenario_config.get("migration", "recent_average")

    if isinstance(migration_config, dict) and migration_config.get("type") == "time_varying":
        schedule = migration_config.get("schedule", {})
        default = migration_config.get("default_factor", 1.0)
        return schedule.get(projection_year, default)

    # Existing static multiplier logic
    return parse_static_multiplier(migration_config)
```

### Phase 3: Fertility Module Update

Modify [`cohort_projections/core/fertility.py`](../../../cohort_projections/core/fertility.py) to apply the scenario-specific fertility adjustment factor (e.g., -5% or +5%) to the baseline age-specific fertility rates.

### Phase 4: Validation

1. Run all three scenarios for the full 53-county projection
2. Verify that the restricted scenario shows the expected shock-and-recovery pattern in annual population growth
3. Verify that the restricted scenario converges to the baseline by approximately 2035 (allowing for the compounding effects of the 2025-2029 cohort reductions)
4. Verify that the high scenario produces higher population than the baseline throughout the projection horizon
5. Compare state-level results to CBO's national population impact (-2.0% by 2045) as a reasonableness check

### Phase 5: Documentation and Export

1. Update scenario labels in all output files, reports, and visualizations
2. Include the CBO source citation in methodology sections of exported reports
3. Retain the old scenario definitions in version control history for reproducibility

## Consequences

### Positive

1. **Empirical grounding**: All scenario parameters are traceable to published CBO data, eliminating the "why these numbers?" question
2. **Policy relevance**: The restricted scenario directly addresses the most significant current demographic policy change, making projections timely and useful for stakeholders
3. **Methodological consistency**: Resolves the contradiction between ADR-018's rejection of arbitrary multipliers and the production config's use of them
4. **Time-varying capability**: Introducing year-indexed migration schedules creates infrastructure for future time-varying scenarios (e.g., economic cycle projections)
5. **Defensible range**: The three scenarios bracket a plausible range anchored to CBO's own uncertainty rather than to symmetric round numbers
6. **Transparency**: Stakeholders can independently verify the CBO source data and the derivation of adjustment factors

### Negative

1. **Implementation complexity**: The `time_varying` migration type requires code changes to the projection engine, whereas static multipliers required no engine modifications
2. **CBO dependency**: The restricted scenario is grounded in a specific CBO publication vintage (January 2026). If CBO substantially revises its projections, the scenario parameters may need updating
3. **Asymmetric scenarios**: The restricted and high scenarios are no longer symmetric around the baseline (the restricted has time-varying factors while the high has a static +15%), which may require additional explanation for stakeholders
4. **Loss of simplicity**: "+25% / -25%" is immediately intuitive even if wrong; "0.20x in 2025 converging to 1.00x by 2030" requires more explanation

### Risks and Mitigations

**Risk**: CBO publishes a substantially revised Demographic Outlook in 2027 that changes the adjustment factors.
- **Mitigation**: The configuration schema stores the schedule as explicit data in `projection_config.yaml`, making updates straightforward. The ADR documents the specific CBO publication vintage, so the provenance is clear. Future updates would be a config change, not a code change.

**Risk**: North Dakota's response to national immigration policy differs from the national pattern modeled by CBO.
- **Mitigation**: ADR-018 proposed a regression-based transfer function from national to state-level migration. This remains a planned enhancement. In the interim, the CBO factors are applied uniformly, which is a conservative simplification. North Dakota's international migration is a relatively small share of total migration, limiting the magnitude of any state-specific divergence.

**Risk**: The +15% high-growth migration assumption proves too conservative or too aggressive.
- **Mitigation**: The parameter is stored in configuration, not hard-coded. The empirical basis (CBO Jan 2025 projected 82% above historical average; Goldman Sachs benchmarks) is documented, so future adjustments have a starting point for recalibration.

**Risk**: Stakeholders misinterpret the restricted scenario as a forecast rather than a conditional projection.
- **Mitigation**: All outputs should label the restricted scenario as "CBO Policy-Adjusted" and include a caveat that it represents a specific policy assumption, not a prediction. The scenario description in the config file and all reports should explicitly state: "Represents the impact of current federal immigration enforcement policy as quantified by CBO, conditional on that policy remaining in effect."

## Alternatives Considered

### Alternative 1: Retain Arbitrary Multipliers with Different Values

**Description**: Keep the +/-N% static multiplier approach but choose values that are "less arbitrary" (e.g., +/-15% instead of +/-25%).

**Pros**:
- No code changes required
- Symmetric and easy to explain
- Familiar to stakeholders

**Cons**:
- Still lacks empirical grounding -- "less arbitrary" is not the same as "empirically derived"
- Cannot represent time-varying effects (the shock-and-recovery pattern)
- Already rejected in principle by ADR-018

**Why Rejected**: This alternative treats the symptom (the specific multiplier values) rather than the disease (the lack of empirical basis). Any static multiplier, regardless of its value, fails to capture the temporal structure of the CBO data.

### Alternative 2: Full CBO Year-by-Year Factors for All Years Through 2045

**Description**: Use the CBO Jan 2026 / Jan 2025 ratio for every year through 2045, including the post-convergence period where the ratio exceeds 1.0.

**Pros**:
- Maximum fidelity to CBO data
- Captures the long-term slight increase in immigration (factors of 1.07-1.11 after 2035)

**Cons**:
- The long-term divergence (7-11% above baseline) reflects national trends that may not apply at the subnational level without a transfer function
- Introduces a permanent upward adjustment to the "restricted" scenario, which is counterintuitive
- Adds complexity for marginal analytical value

**Why Rejected**: The shock-and-recovery pattern (2025-2030) is the empirically strongest and most policy-relevant feature of the CBO data. The post-convergence divergence is small enough that setting it to 1.00 is a reasonable simplification, especially in the absence of a state-specific transfer function. This can be revisited when the ADR-018 regression methodology is implemented.

### Alternative 3: ADR-018 Regression-Based Transfer Function

**Description**: Implement the full regression methodology proposed in ADR-018 -- fit a statistical model relating North Dakota migration to national migration, then use CBO national projections as inputs.

**Pros**:
- State-specific adjustment (addresses the "North Dakota may differ from national trends" concern)
- Statistically grounded
- Provides confidence intervals

**Cons**:
- Requires substantial additional analysis (data gathering, model fitting, validation)
- Historical data for North Dakota international migration has limited degrees of freedom
- Delays the replacement of arbitrary multipliers

**Why Rejected**: Not rejected permanently -- deferred. The CBO-grounded approach in this ADR provides an immediate improvement over arbitrary multipliers while the regression methodology is developed. ADR-018's regression proposal is retained as planned future work. When completed, it could refine the adjustment factors in this ADR's framework rather than replacing it.

### Alternative 4: Use Goldman Sachs or Other Private-Sector Forecasts

**Description**: Anchor scenarios to Goldman Sachs, Moody's, or other private-sector immigration forecasts.

**Pros**:
- May incorporate economic modeling that CBO does not
- Multiple sources provide a range

**Cons**:
- Less transparent methodology (proprietary models)
- Harder for stakeholders to verify
- CBO is a nonpartisan, publicly accountable source with legislative mandate

**Why Rejected**: CBO is the standard reference for demographic projections in U.S. government and academic work. Goldman Sachs and other sources are cited as corroborating evidence but are not suitable as primary anchors due to their proprietary methodologies. The CBO data is publicly available, independently verifiable, and carries institutional credibility.

## Implementation Notes

### Key Functions and Modules

- [`cohort_projections/core/migration.py`](../../../cohort_projections/core/migration.py): Add `get_migration_factor()` to support `time_varying` schedule lookup
- [`cohort_projections/core/fertility.py`](../../../cohort_projections/core/fertility.py): Add scenario-specific fertility adjustment factor application
- [`config/projection_config.yaml`](../../../config/projection_config.yaml): Replace `high_growth` and `low_growth` with `restricted_growth` and `high_growth` (new definitions)

### Configuration Integration

The `time_varying` migration type is a new schema element. The config loader must be updated to parse:

```yaml
migration:
  type: "time_varying"
  schedule:
    2025: 0.20
    ...
  default_factor: 1.00
```

When `migration` is a string (e.g., `"+15_percent"`), existing parsing applies. When `migration` is a dict with `type: "time_varying"`, the new schedule-based lookup applies. This maintains backward compatibility.

### Testing Strategy

1. **Unit tests**: Verify `get_migration_factor()` returns correct values for each year in the schedule and the default for out-of-schedule years
2. **Integration tests**: Run a single-county projection under all three scenarios; verify restricted < baseline < high for the 2025-2029 period and restricted approximately equals baseline for 2035+
3. **Regression tests**: Ensure existing baseline projection results are unchanged (the baseline scenario definition is not modified)
4. **Validation tests**: Compare state-aggregate population under the restricted scenario to CBO's -2.0% national impact as a reasonableness check

## References

### CBO Publications
1. **CBO January 2025 Demographic Outlook** (Publication 60875): [`data/projections/CBO/60875-demographic-outlook.pdf`](../../../data/projections/CBO/60875-demographic-outlook.pdf) (data: [`60875-Data.xlsx`](../../../data/projections/CBO/60875-Data.xlsx))
2. **CBO January 2026 Demographic Outlook** (Publication 61879): [`data/projections/CBO/61879-Demographic-Outlook.pdf`](../../../data/projections/CBO/61879-Demographic-Outlook.pdf) (data: [`61879-Data.xlsx`](../../../data/projections/CBO/61879-Data.xlsx))

### Internal Documentation
3. **ADR-018: Immigration Policy Scenario Methodology**: [`018-immigration-policy-scenario-methodology.md`](018-immigration-policy-scenario-methodology.md) -- Proposed regression-based transfer function; rejected arbitrary multipliers; scenario definitions superseded by this ADR
4. **ADR-035: Census PEP Components of Change for Migration Inputs**: [`035-migration-data-source-census-pep.md`](035-migration-data-source-census-pep.md) -- Established PEP as primary migration data source
5. **ADR-036: Migration Averaging Methodology**: [`036-migration-averaging-methodology.md`](036-migration-averaging-methodology.md) -- BEBR multi-period and Census Bureau interpolation methods for baseline migration assumptions
6. **Immigration Policy Research Report**: [`docs/research/2025_immigration_policy_demographic_impact.md`](../../research/2025_immigration_policy_demographic_impact.md) -- Comprehensive analysis of 2025 policy impacts on demographic trends

### Code References
7. **Migration module**: [`cohort_projections/core/migration.py`](../../../cohort_projections/core/migration.py)
8. **Fertility module**: [`cohort_projections/core/fertility.py`](../../../cohort_projections/core/fertility.py)
9. **Projection config**: [`config/projection_config.yaml`](../../../config/projection_config.yaml)

### External Sources
10. **Goldman Sachs**: Net immigration estimates (~750K/yr under current policy; ~1M/yr pre-pandemic average)
11. **U.S. Census Bureau Population Estimates Program**: Historical components of change data (2000-2024)

## Revision History

- **2026-02-17**: Initial version (ADR-037) -- Replace arbitrary scenario multipliers with CBO-grounded scenarios; introduce time-varying migration schedule

## Related ADRs

- **ADR-018: Immigration Policy Scenario Methodology** -- Proposed regression-based approach; scenario definitions superseded by this ADR; regression methodology retained for future work
- **ADR-035: Census PEP Components of Change for Migration Inputs** -- Established the migration data source used by the baseline scenario
- **ADR-036: Migration Averaging Methodology** -- Defines how PEP data is averaged for baseline migration assumptions
- **ADR-034: Census PEP Data Archive** -- Infrastructure for PEP data access
- **ADR-005: Configuration Management Strategy** -- Configuration schema that must accommodate the new `time_varying` type
- **ADR-010: Geographic Scope and Granularity** -- County-level projection scope to which these scenarios apply
- **ADR-039: International-Only Migration Factor** -- Refines CBO migration factor to apply to international migration only; amends this ADR
- **ADR-040: Extend Bakken Boom Dampening to 2015-2020** -- Extends boom dampening period for oil counties; amends this ADR
