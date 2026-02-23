# ADR-053: North Dakota-Specific Fertility and Mortality Rates

## Status
Accepted

## Date
2026-02-20

## Implemented
2026-02-23

## Last Reviewed
2026-02-23

## Scope
Fertility rates (ASFR/TFR) and mortality rates (survival/life tables) used as inputs to the cohort-component projection engine

## Context

### Problem: National Rates Understate ND Fertility by 14-20%

The current projection system uses **national** CDC NCHS age-specific fertility rates (ASFR) as input (ADR-001). The national TFR for 2023 is approximately **1.62** births per woman. However, North Dakota's TFR is substantially higher:

| Year | ND TFR (est.) | National TFR | ND Premium | Source |
|------|---------------|-------------|------------|--------|
| 2020 | 1.93 | 1.64 | +18% | Gardner Policy Institute |
| 2021 | ~1.95 | 1.66 | +17% | Inferred from GFR 66.68/1,000 |
| 2022 | ~1.88 | 1.656 | +14% | Inferred from GFR decline |
| 2023 | ~1.90 | 1.621 | +17% | ND one of 3 states with TFR increase |

North Dakota has consistently ranked **3rd-4th highest** nationally in fertility. The general fertility rate (GFR) in 2023 was 62.0 per 1,000 women aged 15-44, versus the national 54.5 — a ratio of 1.14x. Using national rates systematically underestimates ND births by roughly 1,400-1,600 per year (approximately 8,000 actual births vs ~9,647 actual births in 2023), compounding over the 30-year projection horizon.

### Problem: National Mortality Rates Miss ND-Specific Patterns

The current system uses **national** CDC 2023 life tables by sex and race (ADR-002). North Dakota's overall life expectancy is approximately **0.5 years above** the national average, per NCHS State Life Tables 2022 (NVSR 74-12):

| Measure | ND (2022) | National (2022) | Difference |
|---------|-----------|-----------------|------------|
| Total e0 | 77.93 | 77.46 | **+0.47** |
| Male e0 | 75.37 | 74.76 | **+0.61** |
| Female e0 | 80.76 | 80.24 | **+0.52** |

While ND's aggregate mortality is slightly *better* than national, this masks critical racial disparities:

| Group | ND Pattern vs. National | Implication |
|-------|------------------------|-------------|
| White | Slightly above national White average (dominant population) | National rates slightly overestimate White mortality |
| AIAN | **Dramatically worse** — median age at death 55 (M) / 62 (F) vs. White 77 (M) / 85 (F) | National AIAN rates **overestimate** survival |
| Black, Hispanic, Asian | Small ND populations; limited data | National rates are reasonable proxies |

The aggregate ND ratio is dominated by the White majority (~82% of population), so the ND/national ratio method primarily captures the White mortality advantage. The AIAN disparity remains the most consequential issue. ND's AIAN population (~5.3%) is concentrated in reservation counties (Benson, Sioux, Rolette, Mountrail) where projection accuracy depends on correctly modeling AIAN mortality. A future enhancement (see Implementation Notes) would compute an AIAN-specific ratio from CDC WONDER death counts rather than relying on the aggregate ratio.

### Data Availability

Recent research identified several sources for ND-specific vital rates:

**Fertility:**
| Source | Dimensions | Years | Machine-Readable | Assessment |
|--------|-----------|-------|-----------------|------------|
| CDC WONDER Natality | State × age × race × year | 2007-2024 | Yes (TSV) | **Best source** for ND ASFR by race |
| NVSR Table 8 (Births: Final Data) | State × age | 2023 | No (PDF) | ND age-specific birth rates |
| ND DHHS Vital Event Summary | State, some county | 2009-2024 | No (PDF) | Context/validation |
| ACS Table B13016 | State × age; county (5yr) | 2024 | Yes (CSV) | Cross-check |

**Mortality:**
| Source | Dimensions | Years | Machine-Readable | Assessment |
|--------|-----------|-------|-----------------|------------|
| NCHS State Life Tables (NVSR 74-12) | State × age × sex | **2022** | Yes (Excel) | **Newest ND life tables** — supersedes SDC's 2020 tables |
| CDC WONDER UCD | State × age × sex × race | 2018-2024 | Yes (TSV) | **Best source** for ND death counts by race |
| National life tables (NVSR 74-06) | Age × sex × race | 2023 | Yes (Excel) | Current approach — race detail |
| Census NP2023 | Age × sex × race × year | 2023-2100 | Yes (CSV) | Mortality improvement trajectory |

### Key Constraint: No ND Life Tables by Race

State-specific life tables (NVSR 74-12) provide ND survival rates by age and sex but **not by race**. National life tables (NVSR 74-06) provide survival rates by race but **not by state**. Neither source alone gives ND × race survival rates. A hybrid approach is required.

## Decision

### Decision: Derive ND-Specific Adjustment Factors for Both Fertility and Mortality

Replace national-only rates with ND-adjusted rates using a ratio method that preserves race specificity from national data while incorporating ND-specific levels.

### Part A: ND-Adjusted Fertility Rates

**Data source:** CDC WONDER Natality query — ND-specific births by mother's age group and race/ethnicity, pooled over 2020-2023 (4 years for statistical stability).

**Method:**

1. Query CDC WONDER for ND births by 5-year age group (15-19, 20-24, ..., 45-49) and race/ethnicity (White NH, Black NH, AIAN NH, Asian/PI NH, Two+ NH, Hispanic), 2020-2023 combined.

2. Obtain ND female population by age group and race from Census PEP estimates (already in project data).

3. Compute ND-specific ASFR by age and race:
   ```
   ND_ASFR[age, race] = ND_births[age, race, 2020-2023] / (ND_females[age, race] × 4 years)
   ```

4. For cells with suppressed birth counts (< 10 births in 4 years), use national ASFR as fallback.

5. Expand 5-year age groups to single-year ages using the same within-group weight method as base population (ADR-048).

**Expected outcome:** ND TFR rises from ~1.62 to ~1.85-1.90, producing approximately 1,400-1,600 additional births per year. Race-specific differentials are preserved (ND Hispanic TFR, ND AIAN TFR, etc.).

**Validation:**
- Computed ND TFR should be within 5% of independently reported values (~1.90)
- Total births in 2023 should approximate the actual 9,647

### Part B: ND-Adjusted Mortality Rates

**Data sources:**
- NCHS 2022 State Life Tables (NVSR 74-12) — ND-specific survival by age × sex
- CDC 2023 National Life Tables (NVSR 74-06) — race-specific survival by age × sex (current data)

**Method:**

1. Download ND 2022 state life tables (ND-1/Total, ND-2/Male, ND-3/Female) from the NVSR 74-12 FTP site.

2. Compute ND/national mortality ratio by age and sex:
   ```
   ND_ratio[age, sex] = ND_qx_2022[age, sex] / National_qx_2022[age, sex]
   ```
   Where `qx` is the probability of dying between ages x and x+1.

3. Apply the ratio to the national race-specific life tables:
   ```
   ND_qx[age, sex, race] = National_qx_2023[age, sex, race] × ND_ratio[age, sex]
   ND_survival[age, sex, race] = 1 - ND_qx[age, sex, race]
   ```
   This adjusts all race groups by the ND-specific level while preserving the race differentials from national data.

4. Cap survival rates at [0, 1].

5. For AIAN population specifically, consider an additional adjustment using CDC WONDER ND AIAN death counts (2018-2023) to compute a more accurate ND AIAN mortality ratio, since the aggregate ND ratio understates AIAN-specific excess mortality.

6. Continue applying Lee-Carter mortality improvement (0.5%/yr) on top of the ND-adjusted base rates, since improvement is about future trends, not current levels.

**Expected outcome:** Slightly *higher* survival rates overall (~0.5 year life expectancy increase), reflecting ND's above-average mortality profile driven by the White majority. The AIAN-specific adjustment is minimal with the aggregate ratio method (see note below).

**Validation (actual results, 2026-02-23):**
- ND Male e0 (computed from adjusted 'total'): 75.86 (published: 75.37)
- ND Female e0 (computed from adjusted 'total'): 81.09 (published: 80.76)
- Computed values are within 0.3-0.5 years of published, expected since ratios are derived from 2022 ND tables but applied to 2023 national race-specific tables
- AIAN adjustment from aggregate ratio is minimal (+0.06 to +0.45 years) because the aggregate ND/national ratio is dominated by White population patterns
- A future AIAN-specific adjustment using CDC WONDER ND AIAN death counts would provide more accurate AIAN mortality calibration

### Configuration

```yaml
rates:
  fertility:
    source: "ND_adjusted_CDC"  # was "CDC_NCHS"
    state_adjustment:
      enabled: true
      source: "CDC_WONDER"
      pooling_years: [2020, 2023]
      fallback: "national"  # for suppressed cells
    assumption: "constant"
    apply_to_ages: [15, 49]

  mortality:
    source: "ND_adjusted_CDC"  # was "CDC_life_tables"
    state_life_table:
      enabled: true
      source: "NVSR_74_12"
      year: 2022
    improvement_factor: 0.005
    cap_survival_at: 1.0
```

## Alternatives Considered

### Alternative A: Use SDC's Blended Fertility Rates Directly
The SDC 2024 projections used county-specific fertility rates blended from ND DHHS 2016-2022 data, with TFR ~2.33. **Rejected** because:
- The blending methodology is undocumented and non-reproducible
- The 2016-2022 reference period inflates rates (Bakken boom era)
- No race dimension, which our system requires
- TFR of 2.33 is likely too high (ND's 2023 TFR is ~1.90, not 2.33)

### Alternative B: Apply a Simple Scalar Multiplier to National Rates
Multiply all national ASFR by a constant factor (e.g., 1.17) to match ND's aggregate TFR. **Rejected** because:
- A uniform scalar assumes ND's fertility premium is equal across all ages and races
- In reality, the ND premium varies by age (higher for 25-34, lower for teens) and by race
- CDC WONDER data allows computing age- and race-specific adjustments at no additional complexity cost

### Alternative C: Continue Using National Rates As-Is
Keep the current approach and accept the ~17% fertility undercount. **Rejected** because:
- 1,400-1,600 fewer births per year compounds over 30 years
- Systematically understates the youngest cohorts, affecting future labor force and school enrollment projections
- The fix is straightforward and the data is available

### Alternative D: Use ND DHHS Vital Event Summary PDFs
Extract data from the state's own vital statistics reports. **Deferred** because:
- Reports are PDF-only, requiring manual extraction
- CDC WONDER provides the same underlying data (same death certificates, same birth certificates) in a machine-readable format
- May be useful as a future validation cross-check

### Alternative E: Use 2020 ND State Life Tables (SDC Approach)
The SDC used NVSR 71-02 (2020). **Rejected** in favor of NVSR 74-12 (2022) because:
- 2020 data is distorted by COVID-19 pandemic mortality
- 2022 tables are 2 years more current and reflect post-pandemic mortality recovery
- Both have the same limitation (no race breakdown), so the newer vintage is strictly superior

## Consequences

### Positive
- Fertility projections become grounded in ND-specific data rather than national averages
- Birth counts align with actual ND vital statistics (~9,647/year vs ~8,000/year)
- AIAN mortality more accurately modeled, improving reservation county projections
- Overall mortality level calibrated to ND's actual life expectancy
- Race-specific differentials preserved from national data (the race dimension is not lost)

### Negative
- Adds a data processing step for CDC WONDER queries (manual web interface, not API)
- Small race-group cells may be suppressed in CDC WONDER (fall back to national)
- The ND/national ratio method assumes that ND's deviation from national rates is uniform across races within each age-sex group — an approximation
- State life tables add a dependency on NVSR 74-12 publication cycle

### Risk Mitigation
- Suppressed CDC WONDER cells fall back to national rates (no data gaps)
- CDC WONDER queries can be re-run when new data years become available
- The ratio method is standard demographic practice (used by Census Bureau for state projections, PPL-47)
- Validation against ND DHHS published totals provides a cross-check

## Implementation Notes

### Data Acquisition Steps
1. **CDC WONDER fertility query**: Navigate to https://wonder.cdc.gov/natality-current.html; select State=North Dakota, Group By: Year + Mother's Age 5-Year Groups + Mother's Single Race 6 + Hispanic Origin; years 2020-2023. Export tab-delimited file.
2. **NVSR 74-12 state life tables**: Download ND Excel files from https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/NVSR/74-12/
3. Process both through ingestion scripts → produce adjusted rate files
4. Update config to point to ND-adjusted rate files

### Files Affected
- `config/projection_config.yaml` — fertility and mortality source config
- `data/raw/fertility/` — new ND WONDER export file
- `data/raw/mortality/` — new ND 2022 state life table Excel files
- `cohort_projections/data/process/fertility_rates.py` — ND adjustment logic
- `cohort_projections/data/process/survival_rates.py` — ND ratio adjustment
- `scripts/data/` — new ingestion script for CDC WONDER + state life tables

### Compatibility
- Downstream consumers (convergence, projection engine, export) are unaffected — they receive rate DataFrames in the same schema
- The adjustment is applied during data processing, before rates enter the engine
- All existing tests should continue to pass (tests use fixture rates, not production data)

## Implementation Results (2026-02-23)

### Part A: ND Fertility

- **Script**: `scripts/data/build_nd_fertility_rates.py`
- **Output**: `data/raw/fertility/nd_asfr_processed.csv` (49 rows: 7 age groups x 7 race categories)
- **ND TFR**: 1.863 (vs. national 1.621, ratio 1.15x) — within 2% of target range 1.85-1.90
- **Average annual births**: 9,804 (vs. actual 9,647 in 2023, within 1.6%)
- **Suppressed cells**: 5 cells fell back to national rates (Asian NH 15-19, several 45-49 groups)
- **New category**: `two_or_more_nh` included with real CDC WONDER data (previously absent from national file)

| Race | ND TFR | National TFR | Ratio |
|------|--------|-------------|-------|
| total | 1.863 | 1.621 | 1.15 |
| white_nh | 1.748 | 1.550 | 1.13 |
| black_nh | 2.661 | 1.647 | 1.62 |
| hispanic | 2.401 | 1.866 | 1.29 |
| aian_nh | 2.298 | 1.422 | 1.62 |
| asian_nh | 1.717 | 1.478 | 1.16 |
| two_or_more_nh | 2.347 | N/A | — |

### Part B: ND Mortality

- **Script**: `scripts/data/build_nd_survival_rates.py`
- **Input**: NVSR 74-12 ND state life tables (2022), ND2 (male) and ND3 (female)
- **Output**: `data/processed/survival_rates.parquet` (1,212 rows, ND-adjusted qx)
- **Adjusted**: 1,200 of 1,212 records (12 age-100 records had no matching ND ratio)
- **Ratio caps**: [0.5, 2.0] applied to prevent extreme adjustments at volatile ages
- **Male ratio**: mean=0.9370, range 0.5000-1.5727
- **Female ratio**: mean=0.9372, range 0.5000-2.0000

### Key Finding: ND e0 Above National

Contrary to the original ADR-053 proposal (which assumed ND was ~0.6 years *below* national), the NVSR 74-12 data shows ND is ~0.5 years *above* national. The ratio method correctly adjusts in the right direction regardless — all race-specific survival rates are scaled by the ND/national ratio, producing slightly improved survival across the board.

### Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `scripts/data/build_nd_fertility_rates.py` | Created | CDC WONDER → ND ASFR processing |
| `scripts/data/build_nd_survival_rates.py` | Created | NVSR 74-12 → ND survival adjustment |
| `data/raw/fertility/nd_asfr_processed.csv` | Created | ND ASFR output (49 rows) |
| `config/projection_config.yaml` | Modified | fertility input_file → nd_asfr_processed.csv |
| `scripts/pipeline/00_prepare_processed_data.py` | Modified | fertility source path updated |
| `data/processed/fertility_rates.parquet` | Regenerated | Now contains ND rates |
| `data/processed/survival_rates.parquet` | Overwritten | Now contains ND-adjusted qx |

## References

- CDC WONDER Natality: https://wonder.cdc.gov/natality-current.html
- NCHS State Life Tables 2022 (NVSR 74-12): https://www.cdc.gov/nchs/data/nvsr/nvsr74/nvsr74-12.pdf
- Gardner Policy Institute, State TFR Analysis (2022): https://gardner.utah.edu/news/utahs-fertility-rate-continues-to-drop-now-fourth-highest-in-the-nation/
- ADR-001: Fertility Rate Processing Methodology
- ADR-002: Survival Rate Processing Methodology
- Census Bureau PPL-47: State Population Projections Methodology
- March of Dimes PeriStats, ND Fertility 2013-2023: https://www.marchofdimes.org/peristats/data?reg=38&top=2
- ND DHHS Vital Statistics: https://www.hhs.nd.gov/vital
