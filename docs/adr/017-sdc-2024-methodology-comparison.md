# ADR-017: SDC 2024 Methodology Comparison and Scenario

| Attribute | Value |
|-----------|-------|
| **Status** | Accepted |
| **Date** | 2025-12-28 |
| **Decision Makers** | Project Team |
| **Categories** | Methodology, Scenarios |

## Context

The North Dakota State Data Center (SDC) released official population projections in 2024. As we develop our own 2026 projection release, we needed to:

1. Understand SDC's methodology and data sources
2. Compare our projection approach to theirs
3. Identify the drivers of any divergence
4. Potentially offer an SDC-aligned scenario for users who want continuity

## Decision

We extracted SDC 2024 methodology parameters (fertility, survival, migration rates) from their Excel workbooks and created:

1. A replication using SDC rates with our projection engine
2. A systematic comparison with our Baseline 2026 methodology
3. Documentation of key methodological differences

## Methodology Comparison

### Migration (Primary Divergence Driver)

| Aspect | SDC 2024 | Baseline 2026 |
|--------|----------|---------------|
| **Data Source** | Census residual method (2000-2020) | IRS county-to-county flows (2019-2022) |
| **Period** | 20-year average (4 census periods) | 4-year average (post-Bakken) |
| **Adjustment** | 60% Bakken dampening applied | No adjustment |
| **Direction** | Net IN-migration projected | Net OUT-migration observed |
| **Annual Impact** | +5,000 to +8,700/year | -2,000 to -3,000/year |

### Fertility

| Aspect | SDC 2024 | Baseline 2026 |
|--------|----------|---------------|
| **Data Source** | Blended ND + national rates | SEER single-year data |
| **TFR** | ~1.73 | ~1.73 (similar) |
| **Projection** | Constant rates | Optional decline scenarios |

### Mortality

| Aspect | SDC 2024 | Baseline 2026 |
|--------|----------|---------------|
| **Data Source** | CDC Life Tables (2020) | CDC Life Tables + SEER |
| **Improvement** | None (constant rates) | 0.5% annual improvement option |
| **Age Detail** | 5-year groups | Single-year ages |

## Projection Results

| Year | SDC Official | SDC Replicated | Baseline 2026 | Zero Migration |
|------|-------------|----------------|---------------|----------------|
| 2025 | 797,000 | 797,000 | 797,000 | 797,000 |
| 2030 | 831,543 | 801,187 | 789,516 | 795,914 |
| 2035 | 865,397 | 809,752 | 782,649 | 799,559 |
| 2040 | 890,424 | 815,255 | 771,446 | 796,087 |
| 2045 | 925,101 | 818,273 | 754,882 | 787,199 |

### Key Finding: 170,000 Person Divergence by 2045

The SDC projects ~925,000 population by 2045 (+16% growth) while our Baseline 2026 methodology projects ~755,000 (-5% decline). This ~170,000 person difference is **entirely driven by migration assumptions**.

## Replication Gap Analysis

Our SDC replication projects lower than SDC official results (818K vs 925K).

### Corrected Understanding (Revised 2025-12-28)

**Both SDC and our engine use the cohort-component method.** Our previous characterization of SDC as an "age-group stock model" was incorrect.

From SDC's official report: *"The process used in these projections is a modified version of the cohort survival component method."*

The Excel workbook formulas confirm this:
```
Nat_Grow[age+5] = Population[age] × Survival_Rate[age]
Migration = Nat_Grow × Mig_Rate × period_multiplier
Projection = Nat_Grow + Adjustments + Migration
```

### Root Cause: Multiple Factors

The gap between our replication (818K) and SDC official (925K) has several components:

1. **Period-specific multipliers we didn't capture:**
   - SDC uses varying multipliers: 0.2, 0.6, 0.6, 0.5, 0.7, 0.7 by period
   - We applied constant 0.6 dampening

2. **Workbook vs Published discrepancy:**
   - SDC's own workbook produces ~913K for 2050, not 957K
   - ~44,000 person gap is unexplained (possibly post-workbook adjustments)

3. **Manual adjustments:**
   - SDC applies ~32,000 in "Adjustments" per period
   - Purpose includes college-age corrections, Bakken region tweaks

### Key Insight

The methodological difference is NOT "cohort vs age-group" but rather:

- **Time step:** SDC uses 5-year intervals; we use annual
- **Data inputs:** SDC uses 2000-2020 census residual; we use 2019-2022 IRS
- **Migration direction:** SDC projects net IN-migration; we project net OUT-migration

The ~170,000 person gap is **primarily driven by migration direction**, not model architecture

## Recommendations

### 1. Primary Methodology: Baseline 2026
Continue using IRS-based migration (net out-migration) as our primary projection. Rationale:
- Uses most recent data (2019-2022)
- Reflects post-Bakken demographic reality
- Aligns with observed recent trends

### 2. Alternative Scenario: SDC-Aligned
Offer an optional "High Migration" scenario that uses SDC-style assumptions for users wanting continuity with previous official projections.

### 3. Documentation
Clearly document the methodological differences and their impact in all published materials.

## Files Created

| File | Description |
|------|-------------|
| `data/processed/sdc_2024/fertility_rates_sdc_blended_2024.csv` | Extracted SDC fertility rates |
| `data/processed/sdc_2024/survival_rates_sdc_2024.csv` | Extracted SDC survival rates |
| `data/processed/sdc_2024/migration_rates_sdc_2024.csv` | Extracted SDC migration rates (dampened) |
| `data/processed/sdc_2024/METHODOLOGY_NOTES.md` | Extraction methodology documentation |
| `scripts/projections/run_sdc_2024_comparison.py` | Comparison projection script |
| `data/projections/methodology_comparison/` | Comparison outputs and visualizations |

## Consequences

### Positive
- Clear understanding of why projections differ from SDC
- Ability to offer SDC-aligned scenario if requested
- Transparent methodology comparison for stakeholders
- Extracted rates available for sensitivity analysis

### Negative
- Our projections will differ significantly from previous SDC releases
- May require additional communication to explain divergence
- SDC replication is imperfect (lower than official results)

### Neutral
- Migration remains the critical uncertainty in ND projections
- Both methodologies are defensible given their data sources
- Future work could calibrate to match SDC more precisely if desired

## References

- SDC 2024 Projections: `data/raw/nd_sdc_2024_projections/`
- Methodology comparison doc: `docs/methodology_comparison_sdc_2024.md`
- Comparison visualization: `data/projections/methodology_comparison/comprehensive_methodology_comparison.png`
