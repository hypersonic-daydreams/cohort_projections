# CLAUDE.md

Quick reference for Claude Code. **For detailed guidance, see [AGENTS.md](./AGENTS.md) when available.**

---

## Project Overview

**North Dakota Cohort Component Population Projection System**

| Attribute | Value |
|-----------|-------|
| Purpose | Demographic projections using cohort-component method |
| Projection Horizon | 2025-2045 (20 years) |
| Geographic Levels | State, 53 Counties, Places (cities/CDPs) |
| Demographic Detail | Age x Sex x Race/Ethnicity cohorts |
| Tech Stack | Python 3.12, uv, pandas, pytest, ruff, mypy |

**Data Sources:**
- Census Bureau (PEP, ACS)
- SEER (demographic rates)
- CDC WONDER (vital statistics)
- IRS SOI (migration flows)

---

## Virtual Environment (Critical)

```bash
# Using direnv (recommended - auto-activates on cd)
direnv allow

# Or manually activate
source .venv/bin/activate
```

**Workflow:** Use `uv sync` to install/update dependencies, then run commands normally.

**Never install packages globally.** Add dependencies to `pyproject.toml` and run `uv sync`.

---

## Essential Commands

### Testing

```bash
pytest                          # Run all tests
pytest --cov                    # With coverage
pytest tests/unit/              # Unit tests only
pytest tests/integration/       # Integration tests only
```

### Code Quality

```bash
pre-commit run --all-files      # All quality checks
ruff check src/                 # Linting
ruff check --fix src/           # Auto-fix lint issues
mypy src/                       # Type checking
```

### Data Operations

```bash
# Fetch data from sibling repositories
python scripts/fetch_data.py            # Fetch all available data
python scripts/fetch_data.py --list     # List data sources and status
python scripts/fetch_data.py --dry-run  # Preview without copying

# Sync data between computers (see Data Management section)
./scripts/bisync.sh                     # Normal sync
./scripts/bisync.sh --resync            # Force resync after conflicts
./scripts/bisync.sh --dry-run           # Preview changes
```

### Run Projections

```bash
python scripts/projections/run_all_projections.py
```

---

## Data Management

**Reference:** [ADR-016](docs/adr/016-raw-data-management-strategy.md)

### Strategy Overview

| What | Where | How Synced |
|------|-------|------------|
| Code, configs | Git repository | `git push/pull` |
| Raw data files | `data/raw/` | rclone bisync |
| Processed data | `data/processed/` | rclone bisync |
| Data sources manifest | `config/data_sources.yaml` | Git |

### Workflow

**Computer with sibling repos (primary):**
```bash
python scripts/fetch_data.py    # Copy data from sibling repos
./scripts/bisync.sh             # Sync to Google Drive
```

**Computer without sibling repos:**
```bash
./scripts/bisync.sh             # Sync data from Google Drive
```

### Critical Rules

- **NEVER run `rclone bisync` directly** - Use `./scripts/bisync.sh` wrapper
- **ALWAYS run bisync after** fetching new data or generating outputs
- **ALWAYS run bisync before** switching to another computer
- Data files are in `.gitignore` - they sync via rclone, not git

---

## Decision Framework

| Tier | Scope | Action |
|------|-------|--------|
| **Tier 1** | Implementation details, bug fixes, refactoring | Just do it |
| **Tier 2** | New dependencies, schema changes, new config options | Document in ADR |
| **Tier 3** | Data deletion, security changes, breaking changes | Stop and ask |

### Tier 2 Examples (Require ADR)

- Adding a new Python dependency
- Changing data file formats
- Modifying projection methodology
- New configuration parameters

### Tier 3 Examples (Stop and Ask)

- Deleting raw data files
- Changing authentication/credentials
- Modifying BigQuery project settings
- Breaking API changes

---

## Critical Rules

### NEVER

- Hard-code file paths (use config or `pathlib`)
- Commit data files to git (they belong in `data/` which is gitignored)
- Run `rclone bisync` directly (use `./scripts/bisync.sh`)
- Skip pre-commit hooks with `--no-verify`
- Install packages globally (use virtual environment)
- Commit secrets or credentials
- Modify production code without running related tests afterward

### ALWAYS

- Activate virtual environment before working
- Run tests before proposing changes
- Run `pre-commit run --all-files` before committing
- Use type hints in function signatures
- Document ADRs for Tier 2 decisions
- Run bisync before switching computers
- Update tests when changing function signatures or behavior

---

## Test Workflow for AI Agents

### When Modifying Production Code

1. **Before changing code**: Run `pytest tests/ -v` to establish baseline
2. **After changing code**: Run tests again - failures indicate breaking changes
3. **If tests fail**: Either fix the code OR update the tests (if behavior change is intentional)
4. **Pre-commit enforces this**: Tests run automatically when committing changes to `cohort_projections/`

### When to Update Tests

| Change Type | Test Action |
| ----------- | ----------- |
| Bug fix | Add test that reproduces the bug, then fix |
| New function | Add tests for the new function |
| Changed signature | Update all tests that call the function |
| Changed behavior | Update tests to expect new behavior |
| Removed function | Remove tests for that function |

### Test Commands

```bash
pytest tests/ -v                           # All tests
pytest tests/test_core/ -v                 # Just core module tests
pytest tests/ -k "test_fertility" -v       # Tests matching pattern
pytest tests/ -x                           # Stop on first failure
pytest tests/ --tb=long                    # Detailed tracebacks
```

### Finding Related Tests

When modifying a function, find its tests:

```bash
# Find tests for a specific function
grep -r "function_name" tests/

# Find tests for a module
ls tests/test_core/test_fertility.py      # Tests for core/fertility.py
```

### Test File Mapping

| Production Module | Test File |
| ----------------- | --------- |
| `cohort_projections/core/cohort_component.py` | `tests/test_core/test_cohort_component.py` |
| `cohort_projections/core/fertility.py` | `tests/test_core/test_fertility.py` |
| `cohort_projections/data/process/base_population.py` | `tests/test_data/test_base_population.py` |
| `cohort_projections/output/writers.py` | `tests/test_output/test_writers.py` |

---

## Project Structure

```
cohort_projections/
├── cohort_projections/     # Main Python package
│   ├── core/               # Projection engine
│   ├── data/               # Data fetching and processing
│   ├── geographic/         # Geographic utilities
│   └── output/             # Export and visualization
├── config/                 # Configuration files
│   ├── projection_config.yaml
│   └── data_sources.yaml
├── data/                   # Data directory (gitignored)
│   ├── raw/                # Input data
│   │   └── nd_sdc_2024_projections/  # SDC reference materials (see below)
│   ├── processed/          # Transformed data
│   └── projections/        # Output projections
├── scripts/                # Executable scripts
│   ├── fetch_data.py
│   ├── bisync.sh
│   └── projections/
├── tests/                  # Test suite
├── docs/                   # Documentation
│   └── adr/                # Architecture Decision Records
└── notebooks/              # Jupyter notebooks
```

---

## SDC 2024 Reference Materials

The `data/raw/nd_sdc_2024_projections/` directory contains the ND State Data Center's 2024 population projections and source files. **These are reference materials only** — not used in our production pipeline.

### Purpose

- **Methodology comparison**: Understand how SDC approached projections
- **Validation reference**: Compare our outputs to their official projections
- **Data source examples**: See their fertility, mortality, migration inputs

### Key Files

| File/Directory | Description |
|---------------|-------------|
| `README.md` | SDC methodology summary |
| `County_Population_Projections_2023.xlsx` | Final county projections workbook |
| `sdc_county_projections_summary.csv` | Extracted county totals for comparison |
| `source_files/` | Original SDC working files |
| `source_files/fertility/` | Birth data, female population counts |
| `source_files/life_tables/` | CDC life tables for ND |
| `source_files/migration/` | Migration rate calculations |
| `source_files/writeup/` | Draft methodology documents |

### Documentation

- **[Methodology Comparison](docs/methodology_comparison_sdc_2024.md)**: Detailed analysis of where our methodology aligns with and diverges from SDC 2024

### Key Finding

Our projections diverge dramatically from SDC (~170,000 people by 2045) due to different migration assumptions:

- **SDC**: Uses 2000-2020 data with 60% Bakken dampening → projects net **in-migration**
- **Ours**: Uses 2019-2022 IRS data → shows net **out-migration**

See the methodology comparison document for full analysis.

---

## Key Configuration

**Main config:** `config/projection_config.yaml`

Key sections:
- `project`: Base year, horizon, intervals
- `geography`: State, county, place settings
- `demographics`: Age groups, sex, race/ethnicity categories
- `rates`: Fertility, mortality, migration sources
- `scenarios`: Baseline, high/low growth options
- `output`: Formats, visualizations, reports
- `bigquery`: Cloud data access settings

---

## Documentation References

| Document | Purpose |
|----------|---------|
| [README.md](./README.md) | Project overview and quick start |
| [AGENTS.md](./AGENTS.md) | Detailed agent guidance (when available) |
| [docs/adr/](./docs/adr/) | Architecture Decision Records |
| [docs/adr/016-raw-data-management-strategy.md](./docs/adr/016-raw-data-management-strategy.md) | Data sync strategy |
| [config/projection_config.yaml](./config/projection_config.yaml) | Main configuration |
| [config/data_sources.yaml](./config/data_sources.yaml) | Data source manifest |

### Key ADRs

- **ADR-001**: Fertility rate processing
- **ADR-002**: Survival rate processing
- **ADR-003**: Migration rate processing
- **ADR-004**: Core projection engine architecture
- **ADR-005**: Configuration management
- **ADR-006**: Data pipeline architecture
- **ADR-016**: Raw data management and cross-computer sync

---

## BigQuery Integration

The project can optionally use Google BigQuery for Census/demographic data access.

```yaml
# In projection_config.yaml
bigquery:
  enabled: true
  project_id: "antigravity-sandbox"
  dataset_id: "demographic_data"
```

**Setup:** See [docs/BIGQUERY_SETUP.md](./docs/BIGQUERY_SETUP.md)

---

## Common Workflows

### Starting a New Session

```bash
cd ~/workspace/demography/cohort_projections
direnv allow                     # Auto-activates venv (first time only)
git pull                         # Get latest code
./scripts/bisync.sh              # Get latest data
uv sync                          # Install any new dependencies
```

### After Making Changes

```bash
pytest                           # Run tests
pre-commit run --all-files       # Quality checks
git add . && git commit -m "..."
./scripts/bisync.sh              # Sync data changes
```

### Updating Data

```bash
python scripts/fetch_data.py     # Refresh from sibling repos
./scripts/bisync.sh              # Sync to other computers
```

---

**Last Updated:** 2025-12-29 | **Version:** 1.1.0
