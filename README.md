# North Dakota Population Projections

Cohort component population projections for North Dakota state, counties, and places (2025-2045).

## Overview

This project implements the standard demographic cohort component method to project population for:
- **State level**: North Dakota
- **County level**: All 53 counties
- **Place level**: Incorporated cities and Census-Designated Places

**Projection horizon**: 2025-2045 (20 years)

**Demographic detail**: Age × Sex × Race/Ethnicity cohorts

## Methodology

The cohort component method projects population through:
1. **Base population**: Starting population by demographic cohorts (2025)
2. **Fertility**: Age-specific birth rates by race/ethnicity
3. **Mortality**: Survival rates by age, sex, and race
4. **Migration**: Net migration by cohort (domestic + international)
5. **Aging**: Advance cohorts forward each year

## Project Structure

```
cohort_projections/
├── cohort_projections/     # Main Python package
├── config/                 # Configuration files
├── data/                   # Data directory
├── scripts/                # Executable scripts
├── tests/                  # Test suite
├── notebooks/              # Jupyter notebooks
└── docs/                   # Documentation
```

## Quick Start

### 1. Environment Setup

```bash
# Create micromamba environment
micromamba create -n cohort_proj python=3.11
micromamba activate cohort_proj

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Base Data

```bash
# Fetch Census and demographic data
python scripts/setup/02_download_base_data.py
```

### 3. Run Projections

```bash
# Run complete projection pipeline
python scripts/projections/run_all_projections.py
```

## Data Sources

- **Census Bureau**: Population Estimates Program (PEP), American Community Survey (ACS)
- **SEER**: Demographic rates and population estimates
- **CDC WONDER**: Vital statistics (births, deaths)
- **IRS**: County-to-county migration flows

## Output

Projections are available in multiple formats:
- **Parquet**: Compressed, efficient (primary format)
- **CSV**: For sharing and analysis
- **Excel**: Summary tables and reports

## License

To be determined

## Contact

North Dakota demographic projections project
