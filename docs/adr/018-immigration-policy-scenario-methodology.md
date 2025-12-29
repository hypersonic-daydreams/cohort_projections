# ADR-018: Immigration Policy Scenario Methodology

**Status:** Proposed
**Date:** 2025-12-28
**Context:** SDC 2024 Replication with Updated Data (isolated from production)

---

## Context

The 2025 U.S. immigration policy changes represent a significant structural break in demographic trends. According to the Congressional Budget Office (CBO), net international migration projections have been revised dramatically:

- **January 2025 estimate:** +1.1 million net inflow (Other Foreign Nationals)
- **September 2025 revision:** -290,000 net outflow

This represents a ~127% swing in expectations. To create a meaningful "immigration policy" scenario for North Dakota projections, we need an empirically-grounded methodology rather than arbitrary multipliers.

## Decision

We will develop an immigration policy scenario variant for the SDC 2024 replication that:

1. **Uses historical data** to establish ND's relationship to national migration patterns
2. **Separates international from domestic migration** to isolate the policy-affected component
3. **Derives adjustment factors** from statistical analysis rather than assumptions
4. **Maintains isolation** within the `sdc_2024_replication/` directory

### Methodology Overview

#### Step 1: Gather Components of Population Change Data

Source: Census Bureau Vintage Population Estimates (2000-2024)

Data needed:
- **National level:** Total population change, natural change, net domestic migration, net international migration (annual)
- **North Dakota:** Same components at state level (annual)

Files to obtain:
- `NST-EST2024-ALLDATA.csv` - National/State estimates with components
- Historical vintage files for 2000-2020 period

#### Step 2: Analyze ND International Migration Patterns

Calculate for each year:
```
ND_intl_migration_share = ND_net_intl_migration / US_net_intl_migration
ND_intl_as_pct_of_change = ND_net_intl_migration / ND_total_change
```

Examine:
- Is ND's share of international migration stable over time?
- How does ND's international migration correlate with national trends?
- What is the lag structure (if any)?

#### Step 3: Statistical Model

Fit a regression model:
```
ND_net_intl_migration = β₀ + β₁ × US_net_intl_migration + ε
```

Or more sophisticated:
```
ND_net_intl_migration = β₀ + β₁ × US_net_intl_migration + β₂ × ND_economic_indicator + ε
```

This gives us a **transfer function** from national policy changes to ND impact.

#### Step 4: Apply to Projection Scenarios

Using CBO's revised net international migration estimates:
1. Calculate expected ND international migration under new policy
2. Adjust SDC migration rates proportionally
3. Run projection with modified rates

### Scenario Definitions

| Scenario | Description | Migration Adjustment |
|----------|-------------|---------------------|
| **SDC Original** | SDC 2024 methodology with 2020 data | None (baseline) |
| **SDC Updated** | SDC methodology with 2024 data | Updated base pop + survival |
| **Immigration Policy** | SDC with 2024 data + policy adjustment | Empirically-derived intl migration reduction |

## Data Sources

### Primary
- Census Bureau Vintage Population Estimates (components of change)
  - URL: https://www.census.gov/data/tables/time-series/demo/popest/2020s-state-total.html
  - Files: NST-EST2024-ALLDATA.csv (and historical)

### Supporting
- CBO Demographic Outlook (September 2025 update)
  - URL: https://www.cbo.gov/publication/61735
  - Key data: Revised net immigration projections by year

### Reference
- Google Gemini Deep Research report on 2025 immigration policy impacts
  - Location: `docs/research/2025_immigration_policy_demographic_impact.md`

## Implementation

### Directory Structure

```
sdc_2024_replication/
├── data/                          # Original SDC 2020 data
├── data_updated/                  # 2024 Census + CDC data
├── data_immigration_policy/       # NEW: Policy-adjusted data
│   ├── MANIFEST.md
│   ├── analysis/                  # Statistical analysis outputs
│   │   ├── nd_migration_components.csv
│   │   ├── regression_results.json
│   │   └── adjustment_factors.csv
│   └── rates/                     # Adjusted migration rates
├── scripts/
│   ├── fetch_components_data.py   # NEW: Download Census data
│   ├── analyze_migration.py       # NEW: Statistical analysis
│   ├── prepare_policy_data.py     # NEW: Create adjusted rates
│   └── run_all_variants.py        # Modified: Run 3 variants
└── output/
    └── three_variant_comparison.csv
```

### Key Outputs

1. **Migration Analysis Report**
   - Time series of ND international migration
   - ND share of national international migration
   - Regression model parameters

2. **Adjustment Factors**
   - Policy impact multipliers by projection period
   - Confidence intervals for scenario bounds

3. **Projection Comparison**
   - SDC Original vs Updated vs Immigration Policy
   - Decomposition of differences by component

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Limited historical data for ND international migration | Use available years, acknowledge uncertainty |
| Policy effects may differ from historical relationships | Provide sensitivity analysis with range of multipliers |
| ND may have unique dynamics (Bakken, universities) | Control for ND-specific factors in regression |
| Census methodology changes over time | Document vintage-to-vintage comparisons |

## Success Criteria

1. Statistical model explains >50% of variance in ND international migration
2. Adjustment factors have reasonable confidence intervals
3. Immigration policy scenario produces projections between Updated and Original scenarios
4. Methodology is reproducible and documented

## Alternatives Considered

### Alternative 1: Arbitrary Multiplier
Apply a fixed reduction (e.g., 50%) to migration rates based on national trends.

**Rejected because:** No empirical basis for ND-specific adjustment; doesn't account for ND's unique position in migration patterns.

### Alternative 2: Use CBO National Figures Directly
Scale ND migration by same percentage as CBO's national revision.

**Rejected because:** ND may respond differently than national average; need to establish historical relationship first.

### Alternative 3: Ignore International Migration Separately
Adjust total migration without separating domestic/international.

**Rejected because:** Policy specifically affects international migration; domestic migration may continue or even increase (replacement workers, etc.).

## References

- [Census Bureau: Net International Migration Drives Population Growth](https://www.census.gov/newsroom/press-releases/2024/population-estimates-international-migration.html)
- [CBO: Demographic Outlook Update 2025-2055](https://www.cbo.gov/publication/61735)
- [ADR-017: SDC 2024 Methodology Comparison](017-sdc-2024-methodology-comparison.md)
- [Immigration Policy Research Report](../research/2025_immigration_policy_demographic_impact.md)

---

## Appendix: Key Statistics from Research

From the Gemini Deep Research report:

| Metric | Value | Source |
|--------|-------|--------|
| CBO net immigration revision (2025) | +1.1M to -290K | CBO Sept 2025 |
| Claimed deportations (2025) | 600,000+ | DHS |
| "Voluntary self-deportations" (2025) | 1.9 million | CBO |
| National pop growth (2023-24) | ~1% (3.3M) | Census |
| International migration share of growth | 84% (2.8M) | Census |
| States where intl migration >50% of growth | 38 | Census |
| States that would have lost pop without immigration | 16 | Census |
