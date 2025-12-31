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
└── docs/                   # Documentation
```

## Quick Start

### 1. Environment Setup

```bash
# Clone and enter project
cd ~/workspace/demography/cohort_projections

# Using direnv (recommended - auto-activates on cd)
direnv allow
uv sync

# Or manually
uv sync                          # Creates .venv and installs dependencies
source .venv/bin/activate        # Activate the environment
```

### 2. Fetch Data from Sibling Repositories

```bash
# Copy data from local sibling repositories (popest, ndx-econareas, maps)
python scripts/fetch_data.py

# List available data sources and their status
python scripts/fetch_data.py --list
```

See [ADR-016](./docs/adr/016-raw-data-management-strategy.md) for data management details.

### 3. Run Projections

```bash
# Run complete projection pipeline
python scripts/projections/run_all_projections.py
```

## Multi-Computer Sync (rclone bisync)

This project uses **rclone bisync** to sync data files between development computers via Google Drive. Code is synced via git; data files are synced via rclone.

### Initial Setup (run once per computer)

```bash
./scripts/setup_rclone_bisync.sh
```

### Regular Sync

```bash
./scripts/bisync.sh              # Normal sync
./scripts/bisync.sh --resync     # Force resync (after conflicts)
./scripts/bisync.sh --dry-run    # Preview changes
```

**Important**: Always run bisync after:
- Fetching new data with `fetch_data.py`
- Running projections that create output files
- Before switching to another computer

See [ADR-016](./docs/adr/016-raw-data-management-strategy.md) for the full data management strategy.

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
