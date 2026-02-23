# Design: Replace PUMS Race Allocation with Census cc-est2024-alldata

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-02-18 |
| **Investigator** | Claude Code (Opus 4.6) |
| **Parent Review** | [Projection Output Sanity Check](../2026-02-18-projection-output-sanity-check.md) |
| **Related Findings** | [Finding 4-5: Race-Specific Trends](finding-4-5-race-specific-trends.md) |
| **Related ADR** | [ADR-041: Census+PUMS Hybrid Base Population](../../governance/adrs/041-census-pums-hybrid-base-population.md) |
| **Status** | Design Complete -- Ready for Implementation |

---

## 1. Problem Statement

The current base population distribution uses a Census+PUMS hybrid approach (ADR-041): Census cc-est2024-agesex-all provides age x sex proportions, while ACS PUMS provides race allocation within each age-sex cell. The PUMS 1% sample for North Dakota (~12,277 records) is catastrophically insufficient for race cross-tabulation of small groups:

| Group | State Population | PUMS Observations | Populated Cells (of 36) | Critical Defect |
|-------|-----------------|-------------------|------------------------|-----------------|
| Black non-Hispanic | 4,412 | ~44 | 7 | Zero females at reproductive ages (15-49) |
| Hispanic | 16,286 | ~163 | 11 | 45% of population in single cell (F 10-14) |
| Asian/PI non-Hispanic | 13,885 | ~139 | 8 | Sparse across most age groups |
| Two or more races | 26,797 | ~268 | 17 | Gaps in older age groups |

This produces physically impossible projections: the Black population cannot produce births (no fertile females), while the Hispanic population generates +204% growth from an artificial echo boom of a single age cohort.

---

## 2. Data Availability Assessment

### 2.1 What We Need: cc-est2024-alldata

The Census Bureau publishes **CC-EST2024-ALLDATA** as part of the Vintage 2024 Population Estimates. This file contains county-level population estimates by **age group x sex x race x Hispanic origin** for all U.S. counties, covering April 1, 2020 through July 1, 2024.

**Availability status**: The file exists on the Census FTP site but has **NOT been downloaded** to the project's shared-data archive. It was released in June 2025.

| Attribute | Value |
|-----------|-------|
| **File name** | `cc-est2024-alldata.csv` (all states combined) |
| **Per-state files** | `cc-est2024-alldata-{SS}.csv` (51 files, one per state FIPS) |
| **FTP directory** | `https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/counties/asrh/` |
| **File layout** | `https://www2.census.gov/programs-surveys/popest/technical-documentation/file-layouts/2020-2024/CC-EST2024-ALLDATA.pdf` |
| **Data type** | Full-count demographic analysis (not sample-based) |
| **Geographic level** | County |
| **Currently in project** | No -- must be downloaded |

### 2.2 What We Already Have: cc-est2020int-alldata (Proxy for Schema Verification)

The project's shared-data archive already contains the intercensal equivalent: `cc-est2020int-alldata.parquet` (2010-2020 vintage). This file has an **identical schema** to cc-est2024-alldata and was used to verify the column structure and race categories.

**File location**: `/home/nhaarstad/workspace/shared-data/census/popest/parquet/2010-2020/county/cc-est2020int-alldata.parquet`

**Verified contents for North Dakota**:
- 12,084 rows (53 counties x 12 years x 19 age groups)
- Full race x sex cross-tabulation at every age group
- All race groups populated at reproductive ages (see Section 3)

### 2.3 What We Also Have: cc-est2024-agesex-all (Already Downloaded, No Race)

The age-sex-only file is already in the archive (used by ADR-041 for the current hybrid approach):

**File location**: `/home/nhaarstad/workspace/shared-data/census/popest/parquet/2020-2024/county/cc-est2024-agesex-all.parquet`

This file has age x sex detail at county level but **no race breakdown**. It is currently the Census data source in the hybrid approach.

---

## 3. Schema Analysis: cc-est2024-alldata Column Structure

Based on the cc-est2020int-alldata schema (which cc-est2024-alldata shares):

### 3.1 Identifying Columns

| Column | Description |
|--------|-------------|
| `SUMLEV` | Summary level (always "050" for county) |
| `STATE` | State FIPS code (string, "38" for North Dakota) |
| `COUNTY` | County FIPS code (string, 3 digits) |
| `STNAME` | State name |
| `CTYNAME` | County name |
| `YEAR` | Year code (see Section 3.3) |
| `AGEGRP` | Age group code (see Section 3.2) |

### 3.2 Age Group Codes (18 five-year groups + total)

| AGEGRP | Age Range | Maps to Project Age Group |
|--------|-----------|--------------------------|
| 0 | Total (all ages) | -- (validation only) |
| 1 | 0-4 | 0-4 |
| 2 | 5-9 | 5-9 |
| 3 | 10-14 | 10-14 |
| 4 | 15-19 | 15-19 |
| 5 | 20-24 | 20-24 |
| 6 | 25-29 | 25-29 |
| 7 | 30-34 | 30-34 |
| 8 | 35-39 | 35-39 |
| 9 | 40-44 | 40-44 |
| 10 | 45-49 | 45-49 |
| 11 | 50-54 | 50-54 |
| 12 | 55-59 | 55-59 |
| 13 | 60-64 | 60-64 |
| 14 | 65-69 | 65-69 |
| 15 | 70-74 | 70-74 |
| 16 | 75-79 | 75-79 |
| 17 | 80-84 | 80-84 |
| 18 | 85+ | 85+ |

**Match**: All 18 five-year age groups align exactly with the project's `AGE_GROUP_RANGES` in `base_population_loader.py`.

### 3.3 Year Codes (for cc-est2024-alldata)

Based on the intercensal file pattern and Census documentation:

| YEAR Code | Meaning |
|-----------|---------|
| 1 | April 1, 2020 Census population |
| 2 | April 1, 2020 Estimates base |
| 3 | July 1, 2020 estimate |
| 4 | July 1, 2021 estimate |
| 5 | July 1, 2022 estimate |
| 6 | July 1, 2023 estimate |
| 7 | July 1, 2024 estimate |

For the base population distribution, use **YEAR=7** (the July 1, 2024 estimate, the most recent available).

### 3.4 Race x Sex Columns

The file contains population counts for each combination of race group and sex. The columns directly relevant to the project's 6-category scheme:

| Census Column(s) | Project Race Category | Notes |
|-------------------|-----------------------|-------|
| `NHWA_MALE`, `NHWA_FEMALE` | White alone, Non-Hispanic | Direct mapping |
| `NHBA_MALE`, `NHBA_FEMALE` | Black alone, Non-Hispanic | Direct mapping |
| `NHIA_MALE`, `NHIA_FEMALE` | AIAN alone, Non-Hispanic | Direct mapping |
| `NHAA_MALE` + `NHNA_MALE`, `NHAA_FEMALE` + `NHNA_FEMALE` | Asian/PI alone, Non-Hispanic | Combine Asian alone + NHPI alone |
| `NHTOM_MALE`, `NHTOM_FEMALE` | Two or more races, Non-Hispanic | Direct mapping |
| `H_MALE`, `H_FEMALE` | Hispanic (any race) | Direct mapping |

**Additional columns available but not needed** (race-alone categories without Hispanic cross-tabulation, race-in-combination categories):
- `WA_MALE`, `WA_FEMALE` -- White alone (regardless of Hispanic origin)
- `BA_MALE`, `BA_FEMALE` -- Black alone (regardless of Hispanic origin)
- `WAC_MALE`, `WAC_FEMALE` -- White alone or in combination
- (etc.)

### 3.5 Validation Against cc-est2020int-alldata (ND, 2020 Intercensal)

The 2020 intercensal data for North Dakota shows realistic population in all race groups at all age groups, including the groups that are catastrophically distorted in the PUMS-based distribution:

**Black non-Hispanic (ages 15-34, reproductive/young adult):**

| Age Group | Male | Female | Total |
|-----------|------|--------|-------|
| 15-19 | 1,110 | 1,017 | 2,127 |
| 20-24 | 1,503 | 1,213 | 2,716 |
| 25-29 | 1,584 | 1,313 | 2,897 |
| 30-34 | 1,600 | 1,381 | 2,981 |

The current PUMS-based distribution has **zero** Black females at ages 15-34. The Census data shows 3,924 Black females in this range. This is the core fix.

**Hispanic (ages 15-34):**

| Age Group | Male | Female | Total |
|-----------|------|--------|-------|
| 15-19 | 1,661 | 1,547 | 3,208 |
| 20-24 | 2,142 | 1,776 | 3,918 |
| 25-29 | 2,045 | 1,676 | 3,721 |
| 30-34 | 1,624 | 1,274 | 2,898 |

The current PUMS-based distribution concentrates 45% of all Hispanics (8,623 persons) in the female 10-14 cell. The Census data shows a realistic spread across age groups and a normal sex ratio.

---

## 4. Can cc-est2024-alldata Serve as a Direct Replacement?

**Yes.** The file is a near-perfect match for the project's requirements:

| Requirement | Status |
|-------------|--------|
| Age groups match (18 five-year groups, 0-4 through 85+) | Exact match |
| Sex breakdown (male/female) | Exact match |
| Race categories map to project's 6-category scheme | Direct mapping (combine NHAA + NHNA for Asian/PI) |
| Geographic level (county) | Exact match (53 ND counties) |
| Full-count data (not sample) | Yes -- demographic analysis-based estimates |
| Most recent vintage (2024) | Yes -- July 1, 2024 estimates |
| Compatible output format | Requires transformation (see Section 5) |

### 4.1 What This Replaces

The cc-est2024-alldata replaces **both** components of the current hybrid:
- It replaces cc-est2024-agesex-all (for age x sex proportions)
- It replaces PUMS (for race allocation within age-sex cells)

This eliminates the PUMS dependency entirely for the base population distribution. PUMS is no longer needed.

### 4.2 Improvement: County-Level Race Distributions

A bonus of using cc-est2024-alldata: the data is available **at the county level**, meaning each county can have its own race distribution rather than using a single statewide distribution. This matters because North Dakota's racial composition varies significantly by county:
- Reservation counties (Sioux, Rolette, Benson) have AIAN populations of 40-80%
- Urban counties (Cass, Grand Forks) have larger Black, Asian, and Hispanic populations
- Rural western counties are >95% White non-Hispanic

However, implementing county-specific distributions is a larger scope change. The minimum viable approach (Section 6) uses a state-level aggregation, which is a drop-in replacement.

---

## 5. Transformations Required

### 5.1 Transformation Pipeline

To produce the same output format as the current `nd_age_sex_race_distribution.csv`:

```
Step 1: Download cc-est2024-alldata.csv (or cc-est2024-alldata-38.csv for ND only)
Step 2: Filter to STATE='38', YEAR=7 (July 1, 2024 estimate)
Step 3: Filter AGEGRP > 0 (exclude total row)
Step 4: Sum across all 53 counties to get state totals by age group x sex x race
Step 5: Map Census race columns to project's 6 categories:
        - NHWA_MALE/FEMALE -> white_nonhispanic
        - NHBA_MALE/FEMALE -> black_nonhispanic
        - NHIA_MALE/FEMALE -> aian_nonhispanic
        - (NHAA + NHNA)_MALE/FEMALE -> asian_nonhispanic  (combine Asian + NHPI)
        - NHTOM_MALE/FEMALE -> multiracial_nonhispanic
        - H_MALE/FEMALE -> hispanic
Step 6: Compute proportions: proportion = cell_count / state_total
Step 7: Write to CSV with schema: age_group, sex, race_ethnicity, estimated_count, proportion
```

### 5.2 Output File Schema

The output schema matches the existing `nd_age_sex_race_distribution.csv` exactly:

| Column | Type | Example |
|--------|------|---------|
| `age_group` | string | "0-4", "5-9", ..., "85+" |
| `sex` | string | "male", "female" |
| `race_ethnicity` | string | "white_nonhispanic", "black_nonhispanic", etc. |
| `estimated_count` | float | 1,017.0 |
| `proportion` | float | 0.001306 |

### 5.3 Expected Output Size

With 18 age groups x 2 sexes x 6 race categories = **216 rows** (vs. current 115 rows). The increase from 115 to 216 is because every age-sex-race combination will be populated, rather than having zero-count cells omitted as in the current PUMS-based file.

---

## 6. Gaps and Limitations

### 6.1 Vintage Mismatch

The base population uses **Vintage 2025** county totals (from the stcoreview file), but the cc-est2024-alldata provides the **Vintage 2024** race distribution. The total population as of July 1, 2024 in cc-est2024-alldata will not match the project's base year of 2025.

**Mitigation**: This is acceptable because the distribution file provides **proportions**, not absolute counts. The proportions are applied to the Vintage 2025 county totals. A one-year gap in proportions introduces negligible error -- racial composition changes slowly (the 2020-2024 period would show <1% shift in any group's share annually for ND).

### 6.2 State-Level Distribution Applied to Counties

Using a statewide distribution for all counties (current approach, maintained in minimum viable fix) ignores county-level racial variation. A reservation county like Sioux County (80% AIAN) will be allocated population as though it has the state-average ~5% AIAN share.

**Mitigation for minimum viable fix**: None needed -- this limitation already exists in the current system and is unchanged. The county-level enhancement (Section 7.2) addresses this as a future improvement.

### 6.3 No cc-est2025-alldata Yet

When Vintage 2025 county characteristics data is released (expected mid-2026), it will provide a 2025-base distribution that exactly matches the project's base year. Until then, Vintage 2024 is the best available.

### 6.4 NHPI Combined with Asian

The Census data separates Asian alone (`NHAA`) from Native Hawaiian/Pacific Islander alone (`NHNA`). The project combines these into "Asian/PI alone, Non-Hispanic" per its 6-category scheme. This is a deliberate design choice (see ADR-007) that loses the NHPI detail, but it is consistent across the pipeline (fertility rates, survival rates, and migration all use the combined category).

---

## 7. Recommendation

### 7.1 Immediate Action: Download and Build New Distribution (P1)

**Priority**: P1 (blocking publication of race-specific projections).

**Implementation steps**:

1. **Download the data**:
   ```bash
   # Download ND-only file (much smaller than all-states)
   curl -o data/raw/population/cc-est2024-alldata-38.csv \
     "https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/counties/asrh/cc-est2024-alldata-38.csv"
   ```

2. **Create ingestion script** at `scripts/data/build_race_distribution_from_census.py`:
   - Read cc-est2024-alldata-38.csv
   - Filter YEAR=7 (2024 estimate), AGEGRP > 0
   - Sum across counties to get state totals
   - Map race columns to project 6-category scheme
   - Compute proportions
   - Write to `data/raw/population/nd_age_sex_race_distribution.csv`
   - Validate: proportions sum to 1.0, all 216 cells populated, sex ratio ~105

3. **Update catalog** in shared-data archive:
   - Add cc-est2024-alldata-38 to `catalog.yaml`
   - Create metadata JSON
   - Archive raw CSV

4. **Re-run pipeline**: Steps 02 (projections) and 03 (exports)

5. **Create ADR**: Document the switch from Census+PUMS hybrid to Census-only race distribution

6. **Validate results**: Compare Black and Hispanic population trajectories against expectations (Black should show stable-to-growing population; Hispanic should show moderate growth in 50-100% range over 30 years, not 204%)

**Estimated effort**: 2-3 hours including validation.

### 7.2 Future Enhancement: County-Level Race Distributions

Once the state-level replacement is working and validated, consider extending to county-level distributions. This would:
- Replace the single statewide `nd_age_sex_race_distribution.csv` with per-county distributions
- Modify `load_base_population_for_county()` to look up a county-specific distribution
- Significantly improve reservation county projections (Finding 3) by correctly initializing AIAN population shares

This is a larger change requiring modifications to the base population loader architecture.

### 7.3 Future Enhancement: Automated Data Pipeline

The current approach requires manual downloads and script execution. A future improvement would add cc-est2024-alldata to the `download_census_pep.py` script and automate the distribution-building step in the pipeline.

---

## 8. Files Referenced

| File | Purpose |
|------|---------|
| `/home/nhaarstad/workspace/demography/cohort_projections/data/raw/population/nd_age_sex_race_distribution.csv` | Current PUMS-based distribution (115 rows, to be replaced) |
| `/home/nhaarstad/workspace/demography/cohort_projections/cohort_projections/data/load/base_population_loader.py` | Loader that consumes the distribution file |
| `/home/nhaarstad/workspace/demography/cohort_projections/docs/governance/adrs/041-census-pums-hybrid-base-population.md` | Current methodology ADR |
| `/home/nhaarstad/workspace/shared-data/census/popest/parquet/2010-2020/county/cc-est2020int-alldata.parquet` | Intercensal race data (used for schema verification) |
| `/home/nhaarstad/workspace/shared-data/census/popest/parquet/2020-2024/county/cc-est2024-agesex-all.parquet` | Current Census source (age-sex only, no race) |
| `/home/nhaarstad/workspace/shared-data/census/popest/docs/ftp-key-index.json` | FTP index with cc-est2024-alldata download URLs |
| `/home/nhaarstad/workspace/shared-data/census/popest/catalog.yaml` | Data catalog (cc-est2024-alldata not yet listed) |

---

## 9. Appendix: Race Column Reference for cc-est2024-alldata

### Race Alone (regardless of Hispanic origin)

| Prefix | Race Category |
|--------|--------------|
| `WA` | White alone |
| `BA` | Black or African American alone |
| `IA` | American Indian and Alaska Native alone |
| `AA` | Asian alone |
| `NA` | Native Hawaiian and Other Pacific Islander alone |
| `TOM` | Two or More Races |

### Race Alone or in Combination

| Prefix | Race Category |
|--------|--------------|
| `WAC` | White alone or in combination |
| `BAC` | Black alone or in combination |
| `IAC` | AIAN alone or in combination |
| `AAC` | Asian alone or in combination |
| `NAC` | NHPI alone or in combination |

### Non-Hispanic by Race Alone (project uses these)

| Prefix | Race Category | Project Category |
|--------|--------------|-----------------|
| `NHWA` | Non-Hispanic White alone | White alone, Non-Hispanic |
| `NHBA` | Non-Hispanic Black alone | Black alone, Non-Hispanic |
| `NHIA` | Non-Hispanic AIAN alone | AIAN alone, Non-Hispanic |
| `NHAA` | Non-Hispanic Asian alone | Asian/PI alone, Non-Hispanic (combined) |
| `NHNA` | Non-Hispanic NHPI alone | Asian/PI alone, Non-Hispanic (combined) |
| `NHTOM` | Non-Hispanic Two or More | Two or more races, Non-Hispanic |

### Hispanic by Race

| Prefix | Race Category | Project Category |
|--------|--------------|-----------------|
| `H` | Hispanic (total, any race) | Hispanic (any race) |
| `HWA` | Hispanic White alone | (not used separately) |
| `HBA` | Hispanic Black alone | (not used separately) |
| (etc.) | | |

### Suffixes

All race columns use `_MALE` and `_FEMALE` suffixes for sex breakdown.
