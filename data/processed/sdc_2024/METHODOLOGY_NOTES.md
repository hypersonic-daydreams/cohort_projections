# SDC 2024 Migration Rate Extraction - Methodology Notes

## Source Data
- File: Projections_Base_2023.xlsx (Sheet: Mig_Rate)
- Location: data/raw/nd_sdc_2024_projections/source_files/results/

## SDC Methodology
- **Method**: Census Residual
  - Migration = Actual_Pop[t+5] - Expected_Pop[t+5]
  - Where Expected_Pop = Starting_Pop * Survival_Rate + Births

- **Time Periods Averaged**:
  - 2000-2005, 2005-2010, 2010-2015, 2015-2020
  - 4 census periods averaged for stability

- **Bakken Dampening**: 60% factor applied
  - Original rates multiplied by 0.60
  - Rationale: Bakken oil boom (2010-2014) created abnormal in-migration
  - SDC judged this was unsustainable and dampened future projections

## Key Findings

### Net Migration Direction
- **Males**: Net IN-migration (avg +1.5% raw, +0.9% dampened per 5yr period)
- **Females**: Net OUT-migration (avg -0.9% raw, -0.5% dampened per 5yr period)
- **Overall**: Slight net in-migration, primarily driven by young male workers

### Critical Age Groups
- **20-24 Males**: Highest in-migration (+32.8% raw, +19.7% dampened)
- **25-29 Females**: Highest out-migration (-24.3% raw, -14.6% dampened)
- **65+**: Both sexes show out-migration (retirement departure)

## Comparison with Our Projections
- **SDC (2000-2020 data)**: Projects net IN-migration with 60% dampening
- **Our baseline (2019-2022 IRS data)**: Shows net OUT-migration
- **Impact**: ~170,000 person difference by 2045

## Files Created
1. `migration_rates_sdc_2024.csv` - Single-year ages (0-90), dampened rates
2. `migration_rates_sdc_2024_raw.csv` - Single-year ages, undampened rates
3. `migration_rates_sdc_2024_5yr.csv` - Original 5-year age groups
4. `migration_rates_summary.csv` - Summary statistics
