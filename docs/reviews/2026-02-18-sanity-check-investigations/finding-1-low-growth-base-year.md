# Finding 1: Low Growth Base Year Mismatch

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-02-18 |
| **Investigator** | Claude Code (Opus 4.6) |
| **Parent Review** | [Projection Output Sanity Check](../2026-02-18-projection-output-sanity-check.md) |
| **Status** | Root cause confirmed; fix requires re-running low_growth scenario |

---

## 1. Summary

The `low_growth` scenario starts at 796,568 in 2025, while `baseline`, `high_growth`, and `restricted_growth` all start at 799,358. The 2,790-person discrepancy (0.35%) is caused by **stale output data**: the `low_growth` projection was generated before the Vintage 2025 data integration and was never re-run afterward. It uses Census Vintage 2024 county populations, while the other three scenarios use Census Vintage 2025.

Additionally, the `low_growth` scenario definition was removed from `projection_config.yaml` during the CBO methodology refactor (commit `9d48bbf`), making it a **deprecated scenario** with no active config entry. The `restricted_growth` scenario replaced it conceptually.

---

## 2. Root Cause Analysis

### 2.1 The Base Population Loading Path

All scenarios share a single code path for loading base population:

1. **`scripts/pipeline/02_run_projections.py`**, line 1165-1166:
   ```python
   base_population_by_geography = {
       fips: load_base_population(config, fips) for fips in fips_to_process
   }
   ```

2. **`02_run_projections.py`**, line 530-541 (delegates to the loader):
   ```python
   def load_base_population(config, fips):
       return load_base_population_for_county(fips, config)
   ```

3. **`cohort_projections/data/load/base_population_loader.py`**, line 280-369:
   - `load_base_population_for_county()` calls `load_county_populations()` (line 332)
   - `load_county_populations()` reads `data/raw/population/nd_county_population.csv` (line 241)
   - Uses a hardcoded column name `pop_col` to select the population vintage (line 256)

There is **no scenario-specific base population selection**. All scenarios load from the same CSV file and the same column. The discrepancy is therefore a **temporal artifact**, not a code logic error.

### 2.2 The Vintage 2025 Code Change

In commit `27904a4` ("feat: integrate Vintage 2025 data, international-only migration factor, and export improvements", 2026-02-18 12:52 CST), the `pop_col` was changed:

**Before** (commit `839d21b` and earlier):
```python
# The file has population_2024 as the most recent
pop_col = "population_2024"
```

**After** (commit `27904a4`):
```python
# The file has population_2025 as the most recent (Vintage 2025)
pop_col = "population_2025"
```

File: `/home/nhaarstad/workspace/demography/cohort_projections/cohort_projections/data/load/base_population_loader.py`, line 255-256.

### 2.3 The Data File Change

The `nd_county_population.csv` was updated on 2026-02-17 at 12:51 CST (file modification time confirmed via `stat`) with Vintage 2025 data from the stcoreview ingestion. The update:

- **Revised** the `population_2024` column from Vintage 2024 values (state total 796,568) to Vintage 2025 revised values (state total 793,387)
- **Added** a new `population_2025` column with Vintage 2025 estimates (state total 799,358)

This revision is standard Census Bureau practice: each new vintage revises all prior year estimates.

### 2.4 Timeline of Events (2026-02-17, all times CST)

| Time | Event | pop_col in code | nd_county_population.csv | Result |
|------|-------|----------------|--------------------------|--------|
| 08:26 | **low_growth run** | `population_2024` | Old (Vintage 2024): pop_2024 = 796,568 | Total: **796,568** |
| ~11:53 | Commit `9d48bbf` (CBO methodology) | `population_2024` | Old | low_growth removed from config |
| ~12:00 | Commits `9d48bbf`, `34b8860` | `population_2024` | Old | Code changes committed |
| 12:51 | Data file updated | `population_2024` (code) | **New** (Vintage 2025): pop_2024 = 793,387, pop_2025 = 799,358 | — |
| ~12:51+ | Code changed (uncommitted) | **`population_2025`** | New | Working directory updated |
| 13:21 | restricted_growth run | `population_2025` | New: pop_2025 = 799,358 | Total: **799,358** |
| 13:22 | high_growth run | `population_2025` | New: pop_2025 = 799,358 | Total: **799,358** |
| 15:30 | baseline run | `population_2025` | New: pop_2025 = 799,358 | Total: **799,358** |

**2026-02-18:**

| Time | Event |
|------|-------|
| 12:52 | Commit `27904a4` (Vintage 2025 integration committed) |

The low_growth scenario was the **only scenario run before both the data file update and the code change**. All other scenarios were run afterward.

### 2.5 Additional Evidence: Projection Horizon Mismatch

The low_growth output files span **2025-2045** (20 years), while baseline spans **2025-2055** (30 years). The `projection_horizon` config was changed from 20 to 30 in commit `27904a4`, confirming the low_growth output was generated under the older configuration.

### 2.6 Additional Issue: Stale `data/processed/base_population.parquet`

A separate processed file (`data/processed/base_population.parquet`, last modified 2026-02-13) also contains the stale Vintage 2024 total of 796,568. This file is **not** used by the projection pipeline (which reads from `nd_county_population.csv` directly), but it **is** referenced by `scripts/data_processing/process_pep_rates.py` for race distribution during migration rate processing. This could propagate stale data into downstream rate files if re-processed.

---

## 3. Verification

### 3.1 County-Level Perfect Match to Vintage 2024

All 53 low_growth county base populations match `POPESTIMATE2024` from `co-est2024-alldata.csv` (Census Vintage 2024) exactly:

```
Mismatches: 0 / 53 counties
```

Sample comparison:

| FIPS | County | co-est2024 (V2024) | low_growth base | baseline base (V2025) |
|------|--------|-------------------:|----------------:|----------------------:|
| 38001 | Adams | 2,141 | 2,141 | 2,275 |
| 38017 | Cass | 200,945 | 200,945 | 201,794 |
| 38015 | Burleigh | 103,107 | 103,107 | 103,251 |
| 38101 | Ward | 68,427 | 68,427 | 68,233 |

### 3.2 All Other Scenarios Match Vintage 2025

| Scenario | 2025 Total | Mismatches vs population_2025 |
|----------|----------:|:------|
| baseline | 799,358 | 0/53 |
| high_growth | 799,358 | 0/53 |
| restricted_growth | 799,358 | 0/53 |
| **low_growth** | **796,568** | **53/53** |

### 3.3 Data Source Comparison

| Source | 2024 Estimate | 2025 Estimate |
|--------|-------------:|-------------:|
| `co-est2024-alldata.csv` (Vintage 2024) | **796,568** | N/A |
| `stcoreview_v2025_nd_parsed.parquet` (Vintage 2025) | 793,387 (revised) | **799,358** |
| `nd_county_population.csv` (current) | 793,387 | **799,358** |
| low_growth output | **796,568** (stale) | — |

---

## 4. Scenario Status

The `low_growth` scenario was removed from `projection_config.yaml` in commit `9d48bbf` (2026-02-17 11:53 CST) during the CBO-grounded scenario methodology refactor. Prior config:

```yaml
low_growth:
  name: "Low Growth Scenario"
  description: "Conservative population growth"
  fertility: "-10_percent"
  mortality: "constant"
  migration: "-25_percent"
  active: false
```

This was replaced conceptually by `restricted_growth` (CBO policy-adjusted scenario with time-varying international migration factor). The `low_growth` output in `data/projections/low_growth/` is an **orphaned artifact** from the previous scenario framework.

The `run_pep_projections.py` script's `PEP_SCENARIO_FILE_MAP` also has no `low_growth` entry:

```python
PEP_SCENARIO_FILE_MAP = {
    "baseline": "baseline",
    "high_growth": "high",
    "restricted_growth": "baseline",
}
```

---

## 5. Fix Required

### 5.1 Immediate Fix

**Option A (Recommended): Delete the stale low_growth output.**

Since `low_growth` is no longer an active scenario and has been replaced by `restricted_growth`, the cleanest fix is to remove the orphaned output:

```bash
rm -rf data/projections/low_growth/
```

Then update any scripts that reference `low_growth`:
- `scripts/generate_visualizations_and_reports.py` (lines 100, 104): remove `low_growth` from the scenario list
- Export scripts that aggregate all scenario directories

**Option B: Re-run low_growth with current data.**

If the team wants to retain a `low_growth` scenario, re-add its definition to `projection_config.yaml` and re-run:

```bash
python scripts/pipeline/02_run_projections.py --counties --scenarios low_growth
```

This would use the current `population_2025` column (799,358) and the 30-year projection horizon.

### 5.2 Secondary Fix: Update `data/processed/base_population.parquet`

Regardless of the low_growth decision, `data/processed/base_population.parquet` should be regenerated to reflect Vintage 2025 data. It currently contains the stale 796,568 total and could propagate incorrect base populations if used for rate processing.

### 5.3 Preventive Measures

To prevent this class of error from recurring:

1. **Add a base-year consistency check to the sanity check pipeline**: After running all scenarios, verify that `population.sum()` for `year == base_year` is identical across all scenario outputs.

2. **Add a data vintage assertion to the projection runner**: Before running projections, log and assert the total base population matches an expected value from the config or a known reference.

3. **Consider adding the scenario list to config validation**: If a scenario directory exists in `data/projections/` but is not defined in `projection_config.yaml`, flag it as potentially stale.

---

## 6. Files Examined

| File | Role in Investigation |
|------|----------------------|
| `/home/nhaarstad/workspace/demography/cohort_projections/scripts/projections/run_pep_projections.py` | Scenario launcher; confirmed no `low_growth` in `PEP_SCENARIO_FILE_MAP` |
| `/home/nhaarstad/workspace/demography/cohort_projections/scripts/pipeline/02_run_projections.py` | Main projection pipeline; traced base population loading (line 530, 1165) |
| `/home/nhaarstad/workspace/demography/cohort_projections/cohort_projections/data/load/base_population_loader.py` | Base population loader; `pop_col` change from `population_2024` to `population_2025` (line 256) |
| `/home/nhaarstad/workspace/demography/cohort_projections/config/projection_config.yaml` | Confirmed no `low_growth` scenario definition; `restricted_growth` replaced it |
| `/home/nhaarstad/workspace/demography/cohort_projections/data/raw/population/nd_county_population.csv` | Current county populations (Vintage 2025: pop_2024=793,387, pop_2025=799,358) |
| `/home/nhaarstad/workspace/demography/cohort_projections/data/raw/population/co-est2024-alldata.csv` | Vintage 2024 Census data; POPESTIMATE2024=796,568 matches low_growth exactly |
| `/home/nhaarstad/workspace/demography/cohort_projections/data/raw/population/stcoreview_v2025_nd_parsed.parquet` | Vintage 2025 stcoreview data; confirmed revised 2024 and new 2025 values |
| `/home/nhaarstad/workspace/demography/cohort_projections/data/processed/base_population.parquet` | Stale processed file (total=796,568); not used by projection pipeline but used by rate processing |
| `/home/nhaarstad/workspace/demography/cohort_projections/data/projections/low_growth/county/*.parquet` | Low_growth output files (all 53 counties, 2025-2045, total=796,568) |
| `/home/nhaarstad/workspace/demography/cohort_projections/data/projections/baseline/county/*.parquet` | Baseline output files (all 53 counties, 2025-2055, total=799,358) |
| `/home/nhaarstad/workspace/demography/cohort_projections/data/projections/low_growth/metadata/*.json` | Run metadata; earliest run 2025-12-28, latest 2026-02-17T14:26 UTC |

### Git History Examined

| Commit | Date | Relevance |
|--------|------|-----------|
| `27904a4` | 2026-02-18 12:52 CST | Changed `pop_col` from `population_2024` to `population_2025`; extended horizon to 30 years |
| `9d48bbf` | 2026-02-17 11:53 CST | Removed `low_growth` from config; added `restricted_growth` |
| `34b8860` | 2026-02-17 12:00 CST | Multi-workbook export (last commit before data file update) |
| `839d21b` | Earlier | Confirmed `pop_col = "population_2024"` in pre-Vintage-2025 code |
