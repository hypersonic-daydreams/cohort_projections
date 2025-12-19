# Fertility Rates Processor - Quick Start Guide

## 5-Minute Quick Start

### 1. Basic Usage (One-Line Processing)

```python
from cohort_projections.data.process import process_fertility_rates

# Process SEER fertility data in one call
fertility_rates = process_fertility_rates(
    input_path='data/raw/fertility/seer_asfr_2018_2022.csv',
    year_range=(2018, 2022),
    averaging_period=5
)
```

**Output**: Creates 3 files in `data/processed/fertility/`:
- `fertility_rates.parquet` (primary)
- `fertility_rates.csv` (backup)
- `fertility_rates_metadata.json` (provenance)

### 2. Load Processed Rates

```python
import pandas as pd

# Load for use in projection engine
fertility_rates = pd.read_parquet('data/processed/fertility/fertility_rates.parquet')

# Use in projection
from cohort_projections.core import CohortComponentProjection

projection = CohortComponentProjection(
    base_population=pop_df,
    fertility_rates=fertility_rates,  # <-- Use here
    survival_rates=surv_df,
    migration_rates=mig_df
)
```

### 3. Validate Existing Rates

```python
from cohort_projections.data.process import validate_fertility_rates
import pandas as pd

# Load your fertility rates
df = pd.read_csv('your_fertility_data.csv')

# Validate
result = validate_fertility_rates(df)

print(f"Valid: {result['valid']}")
print(f"Errors: {result['errors']}")
print(f"TFR by race: {result['tfr_by_race']}")
```

## Required Input Format

Your SEER/NVSS data should have these columns:

| Column | Type | Example | Notes |
|--------|------|---------|-------|
| year | int | 2018 | Data year |
| age | int | 25 | Age 15-49 |
| race | str | "White NH" | SEER race code |
| fertility_rate | float | 0.085 | Births per woman |
| population | float | 10000 | Optional: for weighting |

## Expected Output Format

Processed data ready for projection engine:

| Column | Type | Example | Notes |
|--------|------|---------|-------|
| age | int | 25 | Ages 15-49 |
| race_ethnicity | str | "White alone, Non-Hispanic" | Standardized |
| fertility_rate | float | 0.082 | Multi-year average |
| processing_date | str | "2025-12-18" | Metadata |

**Size**: 210 rows (35 ages × 6 races)

## Common SEER Race Codes

| Your Data | Maps To |
|-----------|---------|
| White NH, 1 | White alone, Non-Hispanic |
| Black NH, 2 | Black alone, Non-Hispanic |
| AIAN NH, 3 | AIAN alone, Non-Hispanic |
| Asian NH, API NH, 4 | Asian/PI alone, Non-Hispanic |
| Two+ Races NH, 5 | Two or more races, Non-Hispanic |
| Hispanic, 6 | Hispanic (any race) |

See `SEER_RACE_ETHNICITY_MAP` in `fertility_rates.py` for complete mapping.

## Validation Criteria

### Will FAIL Validation (Error)
- ❌ Negative fertility rates
- ❌ Missing ages 15-49
- ❌ Missing race categories
- ❌ Rates > 0.15 (biological impossibility)

### Will WARN but PASS
- ⚠️ Rates > 0.13 (unusual)
- ⚠️ TFR < 1.0 or > 3.0
- ⚠️ Any zero-filled combinations

## Total Fertility Rate (TFR) Reference

**What it means**: Average children per woman over lifetime

**Typical U.S. values**:
- White NH: 1.6-1.8
- Black NH: 1.7-1.9
- Hispanic: 2.0-2.3
- AIAN NH: 1.7-2.0
- Asian/PI NH: 1.4-1.7
- Two+ Races NH: 1.6-1.9

**Overall U.S.**: ~1.6-1.8 (below replacement of 2.1)

## Step-by-Step Processing

```python
from cohort_projections.data.process.fertility_rates import (
    load_seer_fertility_data,
    harmonize_fertility_race_categories,
    calculate_average_fertility_rates,
    create_fertility_rate_table
)

# Step 1: Load
raw_df = load_seer_fertility_data('seer_data.csv', year_range=(2018, 2022))

# Step 2: Harmonize race codes
harmonized_df = harmonize_fertility_race_categories(raw_df)

# Step 3: Average over 5 years
averaged_df = calculate_average_fertility_rates(harmonized_df, averaging_period=5)

# Step 4: Create complete table
fertility_table = create_fertility_rate_table(averaged_df, validate=True)

# Step 5: Save
fertility_table.to_parquet('output/fertility_rates.parquet')
```

## Troubleshooting

### Problem: "Unmapped race categories found"

**Solution**: Update `SEER_RACE_ETHNICITY_MAP` in `fertility_rates.py`

```python
SEER_RACE_ETHNICITY_MAP = {
    'Your New Code': 'White alone, Non-Hispanic',  # Add this
    # ... existing mappings
}
```

### Problem: "Missing ages 15-49"

**Solution**: Check your input data has all ages
```python
print(df['age'].unique())  # Should show 15-49
```

### Problem: "TFR seems too low/high"

**Check**:
1. Rates should be per woman (not per 1000)
2. Sum rates across ages to calculate TFR
3. Compare to metadata JSON for validation

### Problem: "Validation warnings about zero rates"

**This is OK if**:
- Small populations truly have no births
- Data genuinely missing for that age-race

**Review needed if**:
- Major race categories all zero
- Entire age ranges zero

## Configuration

Edit `config/projection_config.yaml`:

```yaml
rates:
  fertility:
    source: "SEER"
    averaging_period: 5      # Years to average
    apply_to_ages: [15, 49]  # Reproductive age range
```

## Example: Full Workflow

```python
# 1. Process SEER data
from cohort_projections.data.process import process_fertility_rates

fertility_rates = process_fertility_rates(
    input_path='data/raw/fertility/north_dakota_fertility_2018_2022.csv',
    year_range=(2018, 2022),
    averaging_period=5
)

# 2. Review metadata
import json
with open('data/processed/fertility/fertility_rates_metadata.json') as f:
    metadata = json.load(f)

print(f"TFR by race: {metadata['tfr_by_race']}")
print(f"Overall TFR: {metadata['overall_tfr']}")

# 3. Use in projection
from cohort_projections.core import CohortComponentProjection

projection = CohortComponentProjection(
    base_population=base_pop,
    fertility_rates=fertility_rates,
    survival_rates=survival,
    migration_rates=migration
)

results = projection.run_projection(start_year=2025, end_year=2045)
```

## Testing Your Data

Run the example script to test with synthetic data:

```bash
python examples/process_fertility_example.py
```

This creates sample SEER-format data and demonstrates all processing steps.

## Key Files

| File | Purpose |
|------|---------|
| `fertility_rates.py` | Main processor (803 lines) |
| `001-fertility-rate-processing.md` | ADR documenting decisions |
| `process_fertility_example.py` | Working examples |
| `FERTILITY_QUICKSTART.md` | This guide |
| `README.md` | Full documentation |

## Next Steps

1. **Get SEER data**: Download from <https://seer.cancer.gov/popdata/>
2. **Process data**: Use `process_fertility_rates()` function
3. **Validate output**: Check TFR values make sense
4. **Use in projection**: Pass to `CohortComponentProjection`

## Additional Resources

- **Full Documentation**: See `README.md` section on Fertility Rates
- **ADR-001**: Design decisions in `docs/adr/001-fertility-rate-processing.md`
- **Example Script**: `examples/process_fertility_example.py`
- **Core Engine**: `cohort_projections/core/fertility.py`

## Questions?

**Q: What averaging period should I use?**
A: 5 years is standard (balances stability vs recency)

**Q: Can I use NVSS instead of SEER?**
A: Yes, if it has the required columns

**Q: What if I have county-specific rates?**
A: Process each county separately or use state rates for all

**Q: How do I update race categories?**
A: Edit `SEER_RACE_ETHNICITY_MAP` dictionary

**Q: Why zero-fill missing data?**
A: Conservative approach, see ADR-001 for rationale
