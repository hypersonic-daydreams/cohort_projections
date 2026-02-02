# Scripts Dependencies and Data Flow

This document shows how scripts depend on each other and what data flows between them.

---

## Main Pipeline Data Flow

```
                                    DATA SOURCES
                                         |
                                         v
    +-------------------------------------------------------------------+
    |                        fetch_data.py                               |
    |    Fetches from sibling repos based on config/data_sources.yaml    |
    +-------------------------------------------------------------------+
                                         |
                                         v
                              data/raw/ (CSV files)
                                         |
                                         v
    +-------------------------------------------------------------------+
    |                 [00] prepare_processed_data.py                     |
    |                    CSV -> Parquet conversion                       |
    +-------------------------------------------------------------------+
                                         |
                                         v
                           data/processed/*.parquet
                                         |
                                         v
    +-------------------------------------------------------------------+
    |               [01] process_demographic_data.py                     |
    |         Process fertility, survival, migration rates               |
    +-------------------------------------------------------------------+
                                         |
                                         v
                     data/processed/ (standardized rates)
                                         |
                                         v
    +-------------------------------------------------------------------+
    |                    [02] run_projections.py                         |
    |     Run cohort-component model for state/counties/places           |
    +-------------------------------------------------------------------+
                                         |
                                         v
                    data/projections/{scenario}/*.parquet
                                         |
                                         v
    +-------------------------------------------------------------------+
    |                    [03] export_results.py                          |
    |              Export to CSV/Excel, create packages                  |
    +-------------------------------------------------------------------+
                                         |
                                         v
                        data/exports/ (distribution files)
```

---

## Script Dependencies by Stage

### Stage 0: Data Acquisition

| Script | Inputs | Outputs |
|--------|--------|---------|
| `fetch_data.py` | `config/data_sources.yaml`, sibling repos | `data/raw/**/*.csv` |

**Dependencies:**
- Requires sibling repositories to exist at configured paths
- Reads manifest from `config/data_sources.yaml`

---

### Stage 00: Data Preparation

| Script | Inputs | Outputs |
|--------|--------|---------|
| `00_prepare_processed_data.py` | `data/raw/**/*.csv` | `data/processed/*.parquet` |

**File Conversions:**
```
data/raw/fertility/asfr_processed.csv           -> data/processed/fertility_rates.parquet
data/raw/mortality/survival_rates_processed.csv -> data/processed/survival_rates.parquet
data/raw/migration/nd_migration_processed.csv   -> data/processed/migration_rates.parquet
data/raw/population/nd_county_population.csv    -> data/processed/county_population.parquet
data/raw/population/nd_age_sex_race_distribution.csv -> data/processed/age_sex_race_distribution.parquet
```

**Dependencies:**
- Requires `data/raw/` files from `fetch_data.py`
- Must run BEFORE `01_process_demographic_data.py`

---

### Stage 01: Data Processing

| Script | Inputs | Outputs |
|--------|--------|---------|
| `01_process_demographic_data.py` | `data/processed/*.parquet` | `data/processed/reports/` |

**Processing Components:**
- `--fertility` - Processes fertility rates (SEER -> cohort fertility table)
- `--survival` - Processes survival rates (life tables -> cohort survival table)
- `--migration` - Processes migration rates (IRS flows -> cohort migration table)

**Dependencies:**
- Requires Parquet files from `00_prepare_processed_data.py`
- Uses `config/projection_config.yaml` for settings

---

### Stage 02: Projection Execution

| Script | Inputs | Outputs |
|--------|--------|---------|
| `02_run_projections.py` | Processed rates, base population | `data/projections/{scenario}/` |

**Output Structure:**
```
data/projections/
  baseline/
    state/38_projection.parquet
    county/38XXX_projection.parquet
    place/38XXXXX_projection.parquet
    metadata/projection_run_TIMESTAMP.json
  high_growth/
    ...
  low_growth/
    ...
```

**Dependencies:**
- Requires processed rates from Stage 01
- Uses `config/projection_config.yaml` for scenarios and geographies

---

### Stage 03: Export and Dissemination

| Script | Inputs | Outputs |
|--------|--------|---------|
| `03_export_results.py` | `data/projections/**/*.parquet` | `data/exports/` |

**Output Structure:**
```
data/exports/
  {scenario}/
    {level}/
      csv/*.csv.gz
      excel/*.xlsx
    summaries/*.csv
  data_dictionary.json
  data_dictionary.md
  packages/nd_projections_{level}_YYYYMMDD.zip
  export_report_TIMESTAMP.json
```

**Dependencies:**
- Requires projection results from Stage 02
- Can export subsets with `--state`, `--counties`, `--places`

---

## Auxiliary Script Dependencies

### Validation
```
data/raw/**/*.csv
        |
        v
validate_data.py
        |
        v
data/DATA_VALIDATION_REPORT.md
```

### Visualization
```
data/projections/**/*.parquet
        |
        v
generate_visualizations_and_reports.py
        |
        v
data/output/visualizations/*.png
data/output/reports/*.html, *.md, *.json
```

### Integration Testing
```
data/raw/**/*.csv
        |
        v
run_integration_test.py (uses Cass County, FIPS 38017)
        |
        v
data/processed/integration_test_results.csv
```

### SDC Comparison
```
data/processed/sdc_2024/*.csv
data/processed/*.parquet
        |
        v
projections/run_sdc_2024_comparison.py
        |
        v
data/projections/methodology_comparison/
```

---

## Data Sync Dependencies

```
Local data/           <-- bisync.sh -->           Google Drive
       ^                                                 ^
       |                                                 |
       +-------- setup_rclone_bisync.sh (first run) ----+
```

**Sync workflow:**
1. Run `setup_rclone_bisync.sh` once per machine
2. Run `bisync.sh` before/after work sessions
3. Run `bisync.sh --resync` after conflicts

---

## Configuration Files Used

| Config File | Used By |
|-------------|---------|
| `config/projection_config.yaml` | Pipeline scripts (00-03), projection runners |
| `config/data_sources.yaml` | `fetch_data.py` |
| `~/.config/rclone/cohort_projections-bisync-filter.txt` | `bisync.sh` |

---

## Typical Workflow

1. **New machine setup:**
   ```bash
   python scripts/setup/01_initialize_project.py
   ./scripts/setup_rclone_bisync.sh
   ./scripts/bisync.sh --resync
   python scripts/fetch_data.py
   ```

2. **Run full pipeline:**
   ```bash
   python scripts/pipeline/00_prepare_processed_data.py
   python scripts/pipeline/01_process_demographic_data.py --all
   python scripts/pipeline/02_run_projections.py --all
   python scripts/pipeline/03_export_results.py --all
   ```

3. **After changes, sync:**
   ```bash
   ./scripts/bisync.sh
   ```

---

| Attribute | Value |
|-----------|-------|
| **Last Updated** | 2026-02-02 |
| **Maintainer** | Project Team |
