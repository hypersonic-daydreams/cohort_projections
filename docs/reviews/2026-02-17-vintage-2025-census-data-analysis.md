# Census Bureau Vintage 2025 Data Analysis

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-02-17 |
| **Timestamp** | 2026-02-17T18:40:04Z |
| **Reviewer** | Claude Code (Opus 4.6) |
| **Source File** | `C:\Users\nhaarstad\Downloads\stcoreview_v2025_ND.xlsx` |
| **Classification** | Pre-release / embargoed county-level data; state-level publicly released |
| **Status** | Analysis complete; incorporation recommended |

---

## 1. File Contents

### Identity

Census Bureau Population Estimates Program **Vintage 2025** "State and County Review" file for North Dakota. Internal/advance product sent to state demographers before public county-level release. Single worksheet named `in`, 54 rows (1 state + 53 counties), 249 columns.

### Variables

| Variable Prefix | Description | Age Breakdowns | Year Coverage |
|---|---|---|---|
| `Respop` | Resident population | Total, 0-17, 18-64, 65+ | Census, Base, 2020-2025 |
| `HHpop` | Household population | Total, 0-17, 18-64, 65+ | Census, Base, 2020-2025 |
| `GQpop` | Group quarters population | Total, 0-17, 18-64, 65+ | Census, Base, 2020-2025 |
| `Births` | Annual births | Total only | 2020-2025 |
| `Deaths` | Annual deaths | Total, 0-17, 18-64, 65+ | 2020-2025 |
| `Dommig` | Net domestic migration (count) | Total, 0-17, 18-64, 65+ | 2020-2025 |
| `Dommigrate` | Domestic migration rate | 0-17, 18-64, 65+ | 2020-2025 |
| `Intlmig` | Net international migration (count) | Total, 0-17, 18-64, 65+ | 2020-2025 |
| `Residual` | Residual (unexplained change) | Total, 0-17, 18-64, 65+ | 2020-2025 |
| `Popturning18` | Population aging into 18+ | Total only | 2020-2025 |
| `Popturning65` | Population aging into 65+ | Total only | 2020-2025 |
| `Natrake` | Net age raking factors | 0-17, 18-64, 65+ | 2020-2025 |

### What the file does NOT contain

- No single-year-of-age detail (only 3 broad age groups)
- No sex breakdown (both sexes combined)
- No race/ethnicity breakdown (total population only)
- No 5-year age groups (the projection system uses 18 five-year groups)

---

## 2. Current Data Vintage vs. New Data

### Current: Vintage 2024

- State-level: `data/raw/population/NST-EST2024-ALLDATA.csv` (POPESTIMATE2020 through POPESTIMATE2024)
- County-level: `data/raw/population/co-est2024-alldata.csv`
- County base population: `data/raw/population/nd_county_population.csv` (population_2024 column)
- PEP migration: `data/processed/pep_county_components_2000_2024.parquet` (1,325 rows: 53 counties x 25 years)

### New: Vintage 2025

All existing years (2020-2024) revised; one new year (2025) added.

---

## 3. State-Level Population Revisions

| Year | Vintage 2024 (Current) | Vintage 2025 (New) | Difference | % Change |
|------|------------------------|--------------------|-----------:|----------|
| 2020 | 779,563 | 779,612 | +49 | +0.01% |
| 2021 | 777,966 | 777,977 | +11 | +0.00% |
| 2022 | 781,057 | 780,191 | -866 | -0.11% |
| 2023 | 789,047 | 787,071 | -1,976 | -0.25% |
| 2024 | 796,568 | 793,387 | **-3,181** | **-0.40%** |
| 2025 | N/A | 799,358 | (new) | -- |

**Revisions are significant and directional.** 2024 population revised downward by 3,181 people.

---

## 4. County-Level Population Revisions (Largest for 2024)

| County | Vintage 2024 | Vintage 2025 | Difference |
|--------|-------------|-------------|-----:|
| Cass (Fargo) | 200,945 | 199,271 | -1,674 |
| Burleigh (Bismarck) | 103,107 | 102,447 | -660 |
| Ward (Minot) | 68,427 | 68,186 | -241 |
| Grand Forks | 73,771 | 73,633 | -138 |
| Barnes | 10,798 | 10,673 | -125 |
| Stutsman | 21,546 | 21,445 | -101 |
| Adams | 2,141 | 2,220 | +79 |
| Stark | 33,767 | 33,694 | -73 |
| Morton | 34,194 | 34,261 | +67 |
| Renville | 2,376 | 2,310 | -66 |

Largest downward revisions concentrated in metro counties. Consistent with revised international migration allocation.

---

## 5. Migration Component Revisions (State Level) — THE KEY FINDING

### Domestic Migration

| Year | V2024 | V2025 | Difference |
|------|------:|------:|-----------:|
| 2020 | -438 | -447 | -9 |
| 2021 | -3,966 | -3,878 | +88 |
| 2022 | -2,667 | -2,744 | -77 |
| 2023 | +912 | +930 | +18 |
| 2024 | -291 | -423 | -132 |
| 2025 | N/A | +512 | (new) |

Domestic migration revisions are modest (within ~150 people). Direction unchanged.

### International Migration — SUBSTANTIAL DOWNWARD REVISIONS

| Year | V2024 | V2025 | Difference |
|------|------:|------:|-----------:|
| 2020 | +30 | +30 | 0 |
| 2021 | +453 | +453 | 0 |
| 2022 | +3,287 | +2,554 | **-733** |
| 2023 | +4,269 | +3,158 | **-1,111** |
| 2024 | +5,126 | +4,083 | **-1,043** |
| 2025 | N/A | +2,810 | (new) |

**Cumulative revision for 2022-2024: approximately -2,887 international migrants.** This nearly entirely explains the -3,181 total population revision.

**2025 international migration (+2,810):** A ~31% decline from 2024 (+4,083). This is the first empirical observation of the immigration policy environment that the restricted growth scenario models.

---

## 6. Recommendation: Incorporate This Data

### Benefits

1. **Corrected migration base rates** — The overstated 2022-2024 international migration propagates through all 20 projection years via the convergence interpolation schedule (`recent_period: [2022, 2024]`). A cumulative overstatement of ~2,900 in the base period could result in projected populations thousands too high by 2045.

2. **Actual 2025 base year population** — Config says `base_year: 2025` but system uses 2024 data. The new file provides real July 1, 2025 figures for every county.

3. **One additional migration data year (2025)** — Extends the recent-period window, provides first observation under new immigration policy, reduces influence of anomalous years.

4. **Census Bureau-computed county domestic migration rates** — `Dommigrate` columns provide rates by broad age group (0-17, 18-64, 65+) as computed by Census. Useful as validation benchmark for residual method.

5. **Household vs. group quarters split** — `HHpop` and `GQpop` breakdowns enable future refinements for counties with large GQ populations (military, university, corrections).

6. **Complete demographic accounting** — All components simultaneously enables internal consistency checks.

### Restricted Growth Scenario Calibration

The current schedule assumes a migration factor of 0.20 (80% reduction) for 2025. Actual 2025 international migration declined ~31% from 2024. However, the factor applies to total migration (domestic + international), and domestic migration actually turned positive in 2025. A more careful decomposition is needed, but the empirical data point suggests the 0.20 factor may be more aggressive than reality.

---

## 7. Implementation Plan

### Step 1: Ingest the stcoreview file

Create a loader function to:
- Parse column naming convention (`Dommig_YYYY`, `Intlmig_YYYY`, etc.)
- Handle `'.'` missing values (state-level rates)
- Construct 5-digit FIPS codes from State/County columns
- Output standard schema: `[state_fips, county_fips, county_name, year, netmig, intl_mig, domestic_mig, residual, births, deaths, population]`
- Save to `data/raw/population/stcoreview_v2025_nd_parsed.parquet`

### Step 2: Update county population file

Add `population_2025` column to `data/raw/population/nd_county_population.csv` using `Respop_2025` values. Update `population_2024` with Vintage 2025 revised figures.

### Step 3: Update base population loader

Change `cohort_projections/data/load/base_population_loader.py` line 251 from `pop_col = "population_2024"` to `pop_col = "population_2025"`. Consider using broad age group totals (0-17, 18-64, 65+) as constraints on the proportional age-sex-race allocation.

### Step 4: Update PEP migration components

Update `data/processed/pep_county_components_2000_2024.parquet`:
- Replace 2020-2024 values with Vintage 2025 revisions
- Add 2025 rows for all 53 counties
- Rename to `pep_county_components_2000_2025.parquet`
- Update config reference at `pipeline.data_processing.migration.pep_input`

### Step 5: Residual migration endpoint decision

**Option A (recommended for now):** Keep 2024 as residual migration endpoint. Continue using existing `cc-est2024-agesex-all.parquet` for age-sex detail. Use 2025 PEP components for averaging only, not residual computation. Avoids needing to disaggregate the stcoreview's broad age groups.

**Option B (after public release):** Extend to [2020, 2025] period when `cc-est2025-agesex-all` is publicly released (likely mid-2026). This gives a clean 5-year final period and eliminates the current special-case handling for the 4-year [2020, 2024] period.

### Step 6: Update convergence interpolation config

```yaml
migration:
  interpolation:
    recent_period: [2023, 2025]   # was [2022, 2024]
    medium_period: [2014, 2025]   # was [2014, 2024]
    longterm_period: [2000, 2025] # was [2000, 2024]
```

### Step 7: Update scenario calibration

Use empirical 2025 international migration (+2,810 vs +4,083 in 2024, ~31% decline) to validate/recalibrate restricted growth schedule. Current 2025 factor of 0.20 may be too aggressive.

### Step 8: Update regime analysis

Update `cohort_projections/data/process/pep_regime_analysis.py` line 63:
```python
"recovery": {"start": 2022, "end": 2025, "label": "Recovery"},
```

### Step 9: Update validation benchmarks

```yaml
validation:
  census_benchmark_years: [2020, 2021, 2022, 2023, 2024, 2025]
```

### Step 10: Rerun full pipeline

```bash
python scripts/pipeline/00_prepare_processed_data.py
python scripts/pipeline/01_compute_residual_migration.py
python scripts/pipeline/01b_compute_convergence.py
python scripts/pipeline/02_run_projections.py
python scripts/pipeline/03_export_results.py
```

Then compare old vs new projections across all 3 scenarios and 53 counties.

---

## 8. Data Handling Concerns

### Embargo compliance

- **OK to use** data internally for projection calibration (user confirmed permission)
- **NOT OK to publish** county-level numbers or outputs that would allow back-calculation before Census Bureau public release
- Delay public release of county projections until Census publishes county estimates (likely March-May 2026)
- Alternatively, use for internal methodology validation only, then rerun with public release

### Vintage revision risk

Vintage 2025 is postcensal and will itself be revised. However, this is equally true of V2024. Each successive vintage incorporates better administrative data.

### Format caution

The `stcoreview` format uses different column naming than standard public releases and `'.'` for missing values. Cross-check values against known V2024 data for reasonableness.

### Reproducibility

Document source as "Census Bureau PEP, Vintage 2025, State and County Review File for North Dakota (advance/pre-release)." Re-verify against official release when published.

### Storage

Store in `data/raw/population/` per existing convention. Add to `.gitignore` since it contains embargoed county-level data.
