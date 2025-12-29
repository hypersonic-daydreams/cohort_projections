# SDC 2024 Methodology Replication

## Purpose

This directory contains a **standalone replication** of the North Dakota State Data Center's 2024 population projection methodology. The goal is to produce projections that are directly comparable to SDC's official 2024 release, while exploring how different data inputs and assumptions affect outcomes.

**This project is completely isolated from the main production projection code.**

## Current Variants

We implement multiple variants of the SDC methodology to understand how different inputs affect projections:

| Variant | Base Year | Data Sources | Key Difference |
|---------|-----------|--------------|----------------|
| **Original** | 2020 | SDC 2024 original data | Baseline replication |
| **Updated** | 2024 | Census 2024 + CDC 2023 | Updated base population and survival rates |
| **Immigration Policy** | 2024 | Updated + CBO policy adjustment | Reduced international migration |

### Projection Results (2050)

| Variant | 2050 Population | vs SDC Official |
|---------|-----------------|-----------------|
| SDC Official | 957,194 | — |
| Original (2020 data) | 971,055 | +1.4% |
| Updated (2024 data) | 1,013,400 | +5.9% |
| Immigration Policy | 944,587 | -1.3% |

See [output/three_variant_comparison.csv](output/three_variant_comparison.csv) for full time series.

---

## Variant Details

### Variant 1: Original (SDC 2020 Data)

Replicates SDC 2024 methodology with their original data sources:
- **Base Population:** 2020 Census
- **Survival Rates:** CDC 2020 Life Tables for ND
- **Fertility Rates:** SDC blended county rates
- **Migration Rates:** 2000-2020 Census residual with 60% Bakken dampening
- **Data Directory:** `data/`

### Variant 2: Updated (2024 Data)

SDC methodology with updated demographic data:
- **Base Population:** Census Vintage 2024 estimates
- **Survival Rates:** CDC National Life Tables 2023
- **Fertility Rates:** SDC original (unchanged)
- **Migration Rates:** SDC original (unchanged)
- **Data Directory:** `data_updated/`
- **Documentation:** [data_updated/MANIFEST.md](data_updated/MANIFEST.md)

### Variant 3: Immigration Policy

Updated data with empirically-derived immigration policy adjustment:
- **Base Population:** Same as Updated variant
- **Survival Rates:** Same as Updated variant
- **Fertility Rates:** Same as Updated variant
- **Migration Rates:** SDC rates × 0.6504 adjustment factor
- **Data Directory:** `../data/processed/immigration/rates/`
- **Raw Data:** `../data/raw/immigration/`
- **Analysis Data:** `../data/processed/immigration/analysis/`
- **Documentation:** [../data/processed/immigration/rates/MANIFEST.md](../data/processed/immigration/rates/MANIFEST.md)

**Methodology:**
1. Downloaded Census Bureau components of population change (2010-2024)
2. Ran regression: ND international migration ~ US international migration
3. Found ND receives 0.18% of US international migration (R² = 0.918)
4. International migration = 31% of ND total migration
5. Applied CBO-derived reduction to international component only

See [ADR-018](../docs/adr/018-immigration-policy-scenario-methodology.md) for full methodology.

---

## Directory Structure

```
sdc_2024_replication/
├── README.md                      # This file
├── METHODOLOGY_SPEC.md            # Detailed SDC methodology documentation
├── DATA_UPDATE_PLAN.md            # Plan for data updates
│
├── data/                          # Variant 1: Original SDC 2020 data
│   ├── base_population_by_county.csv
│   ├── fertility_rates_by_county.csv
│   ├── survival_rates_by_county.csv
│   └── migration_rates_by_county.csv
│
├── data_updated/                  # Variant 2: Updated 2024 data
│   ├── MANIFEST.md
│   ├── base_population_by_county.csv
│   ├── fertility_rates_by_county.csv
│   ├── survival_rates_by_county.csv
│   └── migration_rates_by_county.csv
│
├── # Immigration policy data is now in project-level data/ directories:
│   # ../data/raw/immigration/          # Census Bureau source data
│   #   ├── NST-EST2024-ALLDATA.csv
│   #   ├── NST-EST2020-ALLDATA.csv
│   #   ├── dhs_yearbook/
│   #   ├── nd_immigrant_profile/
│   #   └── ...
│   # ../data/processed/immigration/
│   #   ├── analysis/                  # Statistical analysis outputs
│   #   │   ├── combined_components_of_change.csv
│   #   │   ├── migration_analysis_results.json
│   #   │   └── nd_migration_summary.csv
│   #   └── rates/                     # Adjusted projection inputs
│   #       ├── MANIFEST.md
│   #       ├── migration_rates_by_county.csv  # Adjusted rates
│   #       ├── period_multipliers.json
│   #       └── adjustment_details.json
│
├── scripts/                       # Projection and analysis scripts
│   ├── projection_engine.py       # Core SDC methodology implementation
│   ├── run_both_variants.py       # Run Original + Updated variants
│   ├── run_three_variants.py      # Run all three variants
│   ├── analyze_migration_components.py   # Census data analysis
│   └── prepare_immigration_policy_data.py # Create policy variant data
│
└── output/                        # Projection outputs
    ├── three_variant_comparison.csv      # Main comparison table
    ├── original_state_totals.csv
    ├── updated_state_totals.csv
    └── policy_variant_state_totals.csv
```

---

## Running Projections

### Run All Three Variants

```bash
cd sdc_2024_replication/scripts
python run_three_variants.py
```

This produces `output/three_variant_comparison.csv` with all variants side-by-side.

### Run Original + Updated Only

```bash
python run_both_variants.py
```

### Regenerate Immigration Policy Data

If you need to regenerate the policy variant data:

```bash
# First, ensure analysis data exists
python analyze_migration_components.py

# Then create the adjusted rates
python prepare_immigration_policy_data.py
```

---

## Adding New Variants

This framework is designed to be extensible. To add a new variant:

1. **Create a data directory:** `data_<variant_name>/`

2. **Prepare input files:** Each variant needs these CSV files:
   - `base_population_by_county.csv`
   - `fertility_rates_by_county.csv`
   - `survival_rates_by_county.csv`
   - `migration_rates_by_county.csv`
   - (optional) `adjustment_factors_by_county.csv`

3. **Document the variant:** Create a `MANIFEST.md` in the data directory explaining:
   - Data sources used
   - Methodology differences from baseline
   - Any adjustment factors applied

4. **Update scripts:** Add the new variant to `run_three_variants.py` (or create a new runner)

5. **Document in ADR:** Create an ADR in `docs/adr/` for significant methodology changes

### Planned Future Variants

| Variant | Description | Status |
|---------|-------------|--------|
| High Fertility | SDC + increased TFR scenarios | Planned |
| Mortality Improvement | SDC + CDC mortality decline trends | Planned |
| Alternative Migration | SDC + IRS-based migration rates | Planned |
| Economic Shock | Policy variant + recession adjustment | Planned |

---

## Core Methodology

All variants implement the SDC 2024 cohort-component method:

### Projection Parameters
- **Time Intervals:** 5-year periods (2025, 2030, 2035, 2040, 2045, 2050)
- **Age Groups:** 18 five-year groups (0-4, 5-9, ..., 80-84, 85+)
- **Sex:** Male and Female projected separately
- **Geographic Level:** 53 counties, summed to state total

### Core Formula

```
For each 5-year period:
  1. Births:     Pop[0-4, t+5] = Σ(Female_Pop[age] × Fertility[age]) × 5 × Survival[infant]
  2. Aging:      Nat_Grow[age+5] = Pop[age, t] × Survival[age, sex]
  3. Migration:  Migration = Nat_Grow × Mig_Rate[age,sex,county] × Period_Multiplier
  4. Final:      Pop[age, t+5] = Nat_Grow + Migration + Adjustments
```

### Period Multipliers (SDC Original)

| Period | Multiplier | Rationale |
|--------|------------|-----------|
| 2020-2025 | 0.2 | COVID + post-Bakken adjustment |
| 2025-2030 | 0.6 | Bakken dampening |
| 2030-2035 | 0.6 | Continued dampening |
| 2035-2040 | 0.5 | Conservative estimate |
| 2040-2045 | 0.7 | Return toward historical |
| 2045-2050 | 0.7 | Return toward historical |

See [METHODOLOGY_SPEC.md](METHODOLOGY_SPEC.md) for complete technical documentation.

---

## Isolation from Production Code

This replication is **intentionally separate** from the main `cohort_projections/` package:

| Aspect | SDC Replication | Production Code |
|--------|-----------------|-----------------|
| Location | `sdc_2024_replication/` | `cohort_projections/` |
| Methodology | SDC 2024 exact | Our custom approach |
| Time intervals | 5-year | Configurable |
| Migration | SDC rates + dampening | IRS county-to-county |
| Dependencies | Minimal (pandas, numpy) | Full package |
| Purpose | Comparison & scenarios | Production projections |

**Do not import from or modify the main `cohort_projections/` package when working in this directory.**

---

## Documentation References

| Document | Description |
|----------|-------------|
| [METHODOLOGY_SPEC.md](METHODOLOGY_SPEC.md) | Complete SDC methodology specification |
| [ADR-017](../docs/adr/017-sdc-2024-methodology-comparison.md) | SDC vs Baseline 2026 comparison |
| [ADR-018](../docs/adr/018-immigration-policy-scenario-methodology.md) | Immigration policy scenario methodology |
| [data_updated/MANIFEST.md](data_updated/MANIFEST.md) | Updated variant data sources |
| [Immigration Rates MANIFEST.md](../data/processed/immigration/rates/MANIFEST.md) | Policy variant data sources |
| [Research Report](../docs/research/2025_immigration_policy_demographic_impact.md) | Immigration policy impact analysis |

---

## Status

- [x] Directory structure created
- [x] Base population data prepared (Original + Updated)
- [x] Fertility rates compiled
- [x] Survival ratios calculated (Original + Updated with CDC 2023)
- [x] Migration rates and adjustments defined
- [x] Projection engine implemented
- [x] Original variant complete
- [x] Updated variant complete
- [x] Immigration policy variant complete
- [x] Three-variant comparison complete
- [ ] Additional scenario variants (planned)

---

**Last Updated:** 2025-12-28
