# Troubleshooting Guide

Common errors, solutions, and debugging tips for the cohort projection system.

**Related**: [testing-workflow.md](./testing-workflow.md) | [AGENTS.md](../../AGENTS.md)

---

## Quick Diagnostics

```bash
# Check environment is active
which python  # Should show .venv path

# Run tests to verify system state
pytest tests/ -x -q

# Verify data integrity
python scripts/validate_data.py

# Check configuration
python -c "from cohort_projections.utils import load_projection_config; print(load_projection_config())"
```

---

## Common Errors and Solutions

### Environment Issues

#### "ModuleNotFoundError: No module named 'cohort_projections'"

**Cause**: Virtual environment not activated or package not installed.

**Solution**:
```bash
# Activate environment
source .venv/bin/activate  # or: micromamba activate cohort_proj

# Install package in development mode
uv sync
# or: pip install -e .
```

#### "direnv: error .envrc is blocked"

**Cause**: direnv hasn't allowed the project directory.

**Solution**:
```bash
direnv allow
```

#### Import errors after pulling changes

**Cause**: Dependencies changed in pyproject.toml.

**Solution**:
```bash
uv sync
```

---

### Configuration Errors

#### "FileNotFoundError: Configuration file not found"

**Cause**: Running from wrong directory or config missing.

**Solution**:
```bash
# Ensure you're in project root
cd ~/workspace/demography/cohort_projections

# Check config exists
ls config/projection_config.yaml
```

#### "KeyError" when accessing config values

**Cause**: Config structure changed or key misspelled.

**Solution**:
```python
# Print full config to see structure
from cohort_projections.utils import load_projection_config
import yaml
config = load_projection_config()
print(yaml.dump(config, default_flow_style=False))
```

---

### Data Errors

#### "FileNotFoundError: ... data file not found"

**Cause**: Raw data files not present.

**Solution**:
```bash
# Check what data is available
python scripts/fetch_data.py --list

# Fetch from sibling repositories
python scripts/fetch_data.py

# Or download manually (see data-sources-workflow.md)
```

#### "ValueError: Missing required columns: [...]"

**Cause**: Data file format doesn't match expected schema.

**Solution**:
1. Check the expected columns in `config/data_sources.yaml`
2. Verify column names in your data file (case-sensitive)
3. Census Bureau may have changed column naming - check their documentation

**Common column name changes**:
| Old Name | New Name | Source |
|----------|----------|--------|
| `POPESTIMATE2023` | `POPESTIMATE2024` | Census PEP (annual) |
| `county_fips` | `COUNTY` | Census format varies |

#### "ValueError: projection_df is empty"

**Cause**: Data processing resulted in no rows, usually due to filtering.

**Solution**:
1. Check geographic filters in config (state FIPS = "38")
2. Verify input data contains expected geographies
3. Check for NaN values that might cause filtering

```python
# Debug: Check intermediate data
import pandas as pd
df = pd.read_csv("data/processed/nd_county_population.csv")
print(f"Rows: {len(df)}")
print(f"Counties: {df['county_fips'].nunique()}")
print(df['state_fips'].value_counts())
```

---

### FIPS Code Errors

#### "Cannot determine level for FIPS: ..."

**Cause**: FIPS code has unexpected length or format.

**Solution**:
- State FIPS: 2 digits (e.g., "38")
- County FIPS: 5 digits (e.g., "38101")
- Place FIPS: 7 digits (e.g., "3825700")

```python
# Ensure FIPS codes are strings with proper padding
fips = str(fips).zfill(5)  # For counties
```

#### "County data missing required columns"

**Cause**: Geographic reference data not properly formatted.

**Solution**:
```python
# Check what columns exist
import pandas as pd
df = pd.read_csv("data/raw/geographic/nd_counties.csv")
print(df.columns.tolist())

# Required columns:
# - state_fips (2 digits)
# - county_fips (5 digits)
# - county_name
```

---

### Projection Engine Errors

#### "ValueError: base_population contains negative values"

**Cause**: Data processing error or invalid input data.

**Solution**:
```python
# Find negative values
import pandas as pd
df = pd.read_csv("data/processed/nd_county_population.csv")
negatives = df[df['population'] < 0]
print(negatives)
```

#### "ValueError: Invalid population state at year ..."

**Cause**: Projection produced implausible results (usually extreme migration).

**Solution**:
1. Check migration rate assumptions in config
2. Verify survival rates are <= 1.0
3. Check for outlier values in input data

```yaml
# In projection_config.yaml, adjust:
rates:
  mortality:
    cap_survival_at: 1.0  # Ensure survival <= 100%
  migration:
    domestic:
      smooth_extreme_outliers: true  # Enable smoothing
```

---

### Parallel Processing Errors

#### "pickle.PicklingError: Can't pickle ..."

**Cause**: Parallel processing can't serialize some objects.

**Solution**: Parallel processing is disabled by default for this reason.
```yaml
# In projection_config.yaml:
geographic:
  parallel_processing:
    enabled: false  # Keep disabled
```

---

### BigQuery Errors

#### "google.auth.exceptions.DefaultCredentialsError"

**Cause**: BigQuery credentials not configured.

**Solution**:
```bash
# Set up credentials (see docs/BIGQUERY_SETUP.md)
export GCP_CREDENTIALS_PATH=~/.config/gcloud/cohort-projections-key.json
```

Or disable BigQuery:
```yaml
# In projection_config.yaml:
bigquery:
  enabled: false
```

#### "google.api_core.exceptions.NotFound: 404 Dataset not found"

**Cause**: BigQuery dataset doesn't exist or wrong project ID.

**Solution**:
1. Verify `project_id` in config matches your GCP project
2. Create the dataset if it doesn't exist
3. Check `location` matches your dataset region

---

### Pre-commit Hook Errors

#### "ruff found issues"

**Cause**: Code style or linting violations.

**Solution**:
```bash
# Auto-fix what can be fixed
ruff check --fix cohort_projections/

# See specific issues
ruff check cohort_projections/
```

#### "mypy found issues"

**Cause**: Type annotation errors.

**Solution**:
```bash
# See specific issues
mypy cohort_projections/

# Common fixes:
# - Add type annotations to function parameters
# - Handle Optional types properly
# - Import types from typing module
```

#### "pytest-check failed"

**Cause**: Tests are failing.

**Solution**:
```bash
# Run tests with verbose output
pytest tests/ -v

# Run just failing tests
pytest tests/ --lf -v
```

---

## Debugging Tips

### Enable Verbose Logging

```python
# In your script, at the top:
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or via config:
```yaml
# In projection_config.yaml:
logging:
  level: "DEBUG"
```

### Inspect Intermediate Data

```python
# After each processing step, save and inspect
df.to_csv("debug_step1.csv", index=False)
print(df.describe())
print(df.head())
```

### Use the Interactive Python Shell

```bash
# Start IPython in project context
python -c "from cohort_projections.utils import *; import IPython; IPython.embed()"
```

### Check Git Diff for Recent Changes

```bash
# What changed recently that might have broken things?
git log --oneline -10
git diff HEAD~1

# Revert to last known good state
git stash  # Save current changes
git checkout HEAD~1  # Go back
pytest tests/ -x  # Test
git checkout -  # Come back
git stash pop  # Restore changes
```

---

## When to Ask for Help

### Self-Service First

1. **Read the error message carefully** - Python errors are usually descriptive
2. **Search existing documentation** - Check AGENTS.md, guides, ADRs
3. **Check test files** - Tests often show expected usage patterns
4. **Use `--verbose` flags** - Most scripts have verbose modes

### Escalation Points

| Issue Type | Resource |
|------------|----------|
| Data format questions | Census Bureau documentation |
| Methodology questions | `docs/methodology_*.md` files |
| Architecture decisions | `docs/governance/adrs/` |
| Configuration | `docs/guides/configuration-reference.md` |

### Information to Include When Asking for Help

1. **Full error traceback** (not just the last line)
2. **Command you ran** and current directory
3. **Relevant configuration** settings
4. **Python/package versions**: `python --version && pip freeze | grep cohort`
5. **Recent changes**: `git log --oneline -5`

---

## Error Reference

### Quick Lookup Table

| Error Message Fragment | Likely Cause | Quick Fix |
|------------------------|--------------|-----------|
| "ModuleNotFoundError" | Environment not active | `source .venv/bin/activate` |
| "FileNotFoundError: config" | Wrong directory | `cd ~/workspace/demography/cohort_projections` |
| "FileNotFoundError: data" | Missing data files | `python scripts/fetch_data.py` |
| "Missing required columns" | Data format mismatch | Check column names in source file |
| "negative values" | Data processing error | Check input data for negatives |
| "pickle" | Parallel processing issue | Disable parallel processing |
| "credentials" | BigQuery auth missing | Disable BigQuery or configure auth |
| "ruff found issues" | Code style | `ruff check --fix` |

---

*Last Updated: 2026-02-02*
