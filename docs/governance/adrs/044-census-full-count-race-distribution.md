# ADR-044: Census Full-Count Race Distribution (cc-est2024-alldata)

## Status
Accepted

## Date
2026-02-18

## Last Reviewed
2026-02-18

## Scope
Base population age-sex-race distribution methodology for county projections

## Context

### Problem: PUMS 1% Sample Is Catastrophically Insufficient for Small Race Groups

The Census+PUMS hybrid approach (ADR-041) used ACS PUMS for race allocation within each age-sex cell. While ADR-041 correctly fixed the gross sex ratio error of the pure-PUMS approach, the PUMS 1% sample (~12,277 records for ND) remained catastrophically insufficient for race cross-tabulation of small groups:

| Group | State Population | PUMS Observations | Populated Cells (of 36) | Critical Defect |
|-------|-----------------|-------------------|------------------------|-----------------|
| Black non-Hispanic | 4,412 | ~44 | 7 | Zero females at reproductive ages (15-49) |
| Hispanic | 16,286 | ~163 | 11 | 45% of population in single cell (F 10-14) |
| Asian/PI non-Hispanic | 13,885 | ~139 | 8 | Sparse across most age groups |
| Two or more races | 26,797 | ~268 | 17 | Gaps in older age groups |

This produced physically impossible projections:
- The Black population cannot produce births (no fertile females exist in the distribution)
- The Hispanic population generates +204% growth from an artificial echo boom of a single age cohort
- Only 115 of 216 possible age-sex-race cells were populated (remaining cells had zero count)

### Data Availability: cc-est2024-alldata

The Census Bureau publishes CC-EST2024-ALLDATA as part of the Vintage 2024 Population Estimates. This file contains county-level population estimates by age group x sex x race x Hispanic origin for all U.S. counties. These are **full-count demographic analysis-based estimates** (not sample-based), covering April 1, 2020 through July 1, 2024.

The file provides all the detail needed to replace both the Census age-sex component and the PUMS race component of the hybrid approach, eliminating the PUMS dependency entirely.

### Requirements

- All 216 age-sex-race cells (18 age groups x 2 sexes x 6 races) must be populated
- Black females must be present at reproductive ages (15-49)
- Hispanic population must be distributed realistically across age groups
- Sex ratio must be approximately 105.5 males per 100 females
- Proportions must sum to 1.0
- Output format must be compatible with the existing base population loader

## Decision

### Decision: Replace Census+PUMS Hybrid with Census cc-est2024-alldata Full-Count Data

**Decision**: Use Census Bureau County Characteristics Estimates (cc-est2024-alldata-38.csv) exclusively for the age-sex-race distribution. This replaces both the Census age-sex component (cc-est2024-agesex-all) and the PUMS race allocation component of the ADR-041 hybrid approach.

**Data source**: `data/raw/population/cc-est2024-alldata-38.csv` (North Dakota only, downloaded from Census FTP)

**Ingestion pipeline** (`scripts/data/build_race_distribution_from_census.py`):

1. Read cc-est2024-alldata-38.csv
2. Filter to YEAR=6 (July 1, 2024 estimate, the most recent available)
3. Filter AGEGRP > 0 (exclude total row)
4. Sum across all 53 counties to get state totals by age group x sex x race
5. Map Census race columns to project's 6-category scheme:
   - `NHWA_MALE/FEMALE` -> `white_nonhispanic`
   - `NHBA_MALE/FEMALE` -> `black_nonhispanic`
   - `NHIA_MALE/FEMALE` -> `aian_nonhispanic`
   - `(NHAA + NHNA)_MALE/FEMALE` -> `asian_nonhispanic` (combine Asian + NHPI per ADR-007)
   - `NHTOM_MALE/FEMALE` -> `multiracial_nonhispanic`
   - `H_MALE/H_FEMALE` -> `hispanic`
6. Compute proportions: proportion = cell_count / state_total
7. Write to `data/raw/population/nd_age_sex_race_distribution.csv`

**Output**: 216 rows (18 age groups x 2 sexes x 6 races), same CSV schema as the previous file.

**Rationale**:

- **Full-count data eliminates sampling noise**: The cc-est2024-alldata file is derived from the Census Bureau's demographic analysis program, not a sample. Every age-sex-race cell is populated with the Bureau's best estimate.
- **Single data source is simpler**: The hybrid approach required coordinating two data sources at different levels of the distribution hierarchy. The full-count approach uses a single authoritative source for all dimensions.
- **Critical defects are fixed**: All 216 cells are populated (vs. 115), Black females exist at all reproductive ages, Hispanic population is spread realistically across age groups.

**Alternatives Considered**:

- **Keep hybrid, improve PUMS with 5-year ACS**: The 5-year ACS has a larger sample, but still insufficient for county-level race x age x sex cross-tabulation in a small state. The full-count data is strictly superior.
- **Iterative Proportional Fitting (IPF)**: IPF could reconcile PUMS race shares with Census marginals, but the cc-est2024-alldata already provides the complete joint distribution, making IPF unnecessary.

## Consequences

### Positive

1. **All 216 cells populated**: Every age-sex-race combination has a non-zero population count (was 115 of 216)
2. **Black population can produce births**: 7,600 Black females at reproductive ages across all 7 five-year age groups (was zero)
3. **Hispanic distribution is realistic**: Largest cell is 7.0% of Hispanic total (was 45% concentrated in F 10-14)
4. **Sex ratio correct**: 105.5 males per 100 females (unchanged from ADR-041 fix)
5. **PUMS dependency eliminated**: No longer need ACS PUMS for the base population distribution
6. **Simpler methodology**: Single data source instead of two-source hybrid
7. **Reproducible**: Ingestion script reads a specific Census file with fixed filters

### Negative

1. **Vintage mismatch**: V2024 race proportions are applied to V2025 population totals. This introduces negligible error because racial composition changes slowly (less than 1% shift in any group's share annually). When cc-est2025-alldata is released (expected mid-2026), the proportions can be updated.
2. **State-level distribution**: Uses statewide proportions for all counties (same limitation as before). A future enhancement could use county-specific distributions from the same data source.

### What This Supersedes

This ADR supersedes ADR-041 for race allocation methodology. ADR-041's description of the Census+PUMS hybrid approach remains accurate as historical documentation of the intermediate methodology. The age-sex correction described in ADR-041 (fixing the 119.1 sex ratio) is now achieved through the full-count data rather than the hybrid approach.

## Implementation Notes

### Key Files

| File | Purpose |
|------|---------|
| `data/raw/population/cc-est2024-alldata-38.csv` | Raw Census input (not committed to git) |
| `data/raw/population/nd_age_sex_race_distribution.csv` | Output distribution (not committed to git) |
| `scripts/data/build_race_distribution_from_census.py` | Ingestion script |
| `cohort_projections/data/load/base_population_loader.py` | Loader (unchanged schema, updated docstring) |

### Configuration Integration

No configuration changes required. The distribution file path is resolved by the loader using the project root convention (`data/raw/population/nd_age_sex_race_distribution.csv`). The file schema (age_group, sex, race_ethnicity, estimated_count, proportion) is unchanged.

### Testing Strategy

1. **Ingestion script validation**: Built-in checks for 216 rows, proportion sum, sex ratio, all cells populated, Black females at reproductive ages, Hispanic distribution spread
2. **Loader regression**: Existing `load_state_age_sex_race_distribution()` tests pass unchanged (same schema)
3. **Projection comparison**: Re-run projections and verify Black/Hispanic population trajectories are physically plausible

### Validation Results

| Metric | Old (PUMS-based) | New (Census full-count) |
|--------|-------------------|------------------------|
| Total rows | 115 | 216 |
| Populated cells | 115 / 216 | 216 / 216 |
| Sex ratio | 105.5 (after ADR-041) | 105.5 |
| Black females 15-49 | 0 | 7,600 |
| Hispanic largest cell share | 45% (F 10-14) | 7.0% |
| Race categories | 6 | 6 |

## References

1. **Census Bureau CC-EST2024-ALLDATA**: County Characteristics Resident Population Estimates by Age, Sex, Race, and Hispanic Origin. Vintage 2024. Published June 2025.
   - FTP: `https://www2.census.gov/programs-surveys/popest/datasets/2020-2024/counties/asrh/`
   - Layout: `https://www2.census.gov/programs-surveys/popest/technical-documentation/file-layouts/2020-2024/CC-EST2024-ALLDATA.pdf`
2. **ADR-041**: Census+PUMS Hybrid Base Population (superseded for race allocation)
3. **ADR-007**: Race/Ethnicity Categorization (defines the 6-category scheme)
4. **Design document**: `docs/reviews/2026-02-18-sanity-check-investigations/design-race-data-replacement.md`

## Revision History

- **2026-02-18**: Initial version (ADR-044) -- Replace PUMS race allocation with Census full-count cc-est2024-alldata

## Related ADRs

- **ADR-041: Census+PUMS Hybrid Base Population** -- Superseded for race allocation; ADR-041 fixed the sex ratio but retained PUMS for race, which this ADR replaces
- **ADR-007: Race/Ethnicity Categorization** -- Defines the 6-category race/ethnicity scheme
- **ADR-004: Core Projection Engine** -- Consumes the base population distribution
- **ADR-035: Census PEP Components of Change** -- Related Census data source usage
