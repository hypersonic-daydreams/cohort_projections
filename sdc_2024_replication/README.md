# SDC 2024 Methodology Replication

## Purpose

This directory contains a **standalone replication** of the North Dakota State Data Center's 2024 population projection methodology. The goal is to produce projections that are directly comparable to SDC's official 2024 release by implementing their exact methodology with updated data sources where available.

**This project is completely isolated from the main production projection code.**

## Methodology Overview

This replication implements the SDC 2024 approach exactly:

### Projection Parameters
- **Base Year**: 2020 (Census)
- **Projection Horizon**: 2025-2045
- **Time Intervals**: 5-year periods (2020-2025, 2025-2030, 2030-2035, 2035-2040, 2040-2045)
- **Geographic Level**: County-level projections (53 counties)
- **Age Groups**: 5-year age groups (0-4, 5-9, ..., 80-84, 85+)
- **Sex**: Male and Female

### Core Components

1. **Fertility Rates**
   - Age-specific fertility rates (ASFR) by 5-year age groups
   - Period-specific rates that can vary across projection intervals

2. **Mortality/Survival**
   - Survival ratios derived from CDC life tables
   - Age and sex-specific survival rates

3. **Migration**
   - Net migration rates by age and sex
   - Period-specific multipliers to adjust for economic/demographic trends
   - Manual county-level adjustments where data indicates special circumstances

4. **Special Adjustments**
   - Oil patch counties (Bakken region) receive dampened migration assumptions
   - Tribal areas may have separate adjustment factors
   - Large employers/developments can trigger manual adjustments

## Data Sources

### Original SDC 2024 Sources (Reference)
- 2020 Census population by age, sex, and county
- CDC WONDER vital statistics (births, deaths)
- IRS SOI migration data (2000-2020)
- State-level adjustments from BLS employment data

### Updated Data for This Replication
- 2020 Census population (same base)
- More recent fertility and mortality data where available
- Updated migration data (IRS SOI through 2022)
- Recent employment/economic indicators

## Directory Structure

```
sdc_2024_replication/
├── README.md           # This file
├── data/               # Input data for replication
│   ├── population/     # Base population by county/age/sex
│   ├── fertility/      # Age-specific fertility rates
│   ├── survival/       # Survival ratios from life tables
│   └── migration/      # Net migration rates and adjustments
├── output/             # Projection outputs
│   ├── county/         # County-level projections
│   ├── state/          # Aggregated state totals
│   └── validation/     # Comparison with SDC official projections
└── scripts/            # Standalone projection scripts
    ├── prepare_data.py # Data preparation
    ├── project.py      # Main projection engine
    └── validate.py     # Compare with SDC 2024 official
```

## Isolation from Production Code

This replication is **intentionally separate** from the main `cohort_projections/` package:

| Aspect | SDC 2024 Replication | Production Code |
|--------|---------------------|-----------------|
| Location | `sdc_2024_replication/` | `cohort_projections/` |
| Methodology | SDC 2024 exact | Our custom approach |
| Time intervals | 5-year | Configurable |
| Migration | Period multipliers + manual | Data-driven |
| Dependencies | Minimal (pandas, numpy) | Full package |
| Purpose | Validation/comparison | Production projections |

**Do not import from or modify the main `cohort_projections/` package when working in this directory.**

## Goals

1. **Exact Methodology Match**: Implement SDC's cohort-component approach precisely as documented in their 2024 release
2. **Updated Data**: Use the most recent available data sources while maintaining methodological consistency
3. **Validation**: Compare our replication outputs to SDC's official 2024 projections to verify implementation accuracy
4. **Divergence Analysis**: Understand exactly where and why projections differ when using updated data

## Reference Materials

The original SDC 2024 methodology and working files are available in:
```
data/raw/nd_sdc_2024_projections/
```

Key reference documents:
- `County_Population_Projections_2023.xlsx` - SDC's projection workbook
- `source_files/writeup/` - Methodology documentation
- `source_files/migration/` - Migration rate calculations

## Status

- [ ] Directory structure created
- [ ] Base population data prepared
- [ ] Fertility rates compiled
- [ ] Survival ratios calculated
- [ ] Migration rates and adjustments defined
- [ ] Projection engine implemented
- [ ] Validation against SDC 2024 complete

---

**Last Updated**: 2024-12-28
