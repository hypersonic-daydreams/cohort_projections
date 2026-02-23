# Finding 2: Oil County Growth Rates Post-Dampening

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-02-18 |
| **Investigator** | Claude Code (Opus 4.6) |
| **Parent Review** | [Projection Output Sanity Check](../2026-02-18-projection-output-sanity-check.md) |
| **Related** | [ADR-040](../../governance/adrs/040-extend-boom-dampening-2015-2020.md), [Bakken Dampening Review](../2026-02-17-bakken-migration-dampening-review.md) |
| **Status** | Confirmed -- ADR-040 dampening insufficient; further action required |

---

## 1. Executive Summary

After implementing ADR-040 (extending boom dampening to the 2015-2020 period), three Bakken-area counties still project implausible 30-year growth rates:

| County | FIPS | 2025 Pop | 2055 Pop | 30-Year Growth | Dampened? |
|--------|------|-------:|-------:|------:|-----------|
| **Billings** | 38007 | 1,071 | 2,010 | **+87.7%** | **No** |
| **McKenzie** | 38053 | 15,192 | 27,921 | **+83.8%** | Yes |
| **Williams** | 38105 | 41,767 | 63,195 | **+51.3%** | Yes |
| Cass (Fargo) | 38017 | 201,794 | 328,393 | +62.7% | N/A |
| Burleigh (Bismarck) | 38015 | 103,251 | 139,449 | +35.1% | N/A |
| State total | | 799,358 | ~1,020,775 | +27.7% | N/A |

McKenzie (+84%) and Billings (+88%) both exceed Cass/Fargo (+63%), which is the state's fastest-growing metro area. This is implausible for rural oil-patch counties whose most recent migration period (2020-2024) shows negative net migration (McKenzie) or statistically noisy positive migration driven by a handful of individuals (Billings).

ADR-040 predicted McKenzie would drop to +40-60%. The actual result of +84% significantly exceeds that expectation. **The root causes are threefold**: (1) the 0.60x dampening factor is insufficient for the magnitude of boom-era rates, (2) Billings County is not on the dampened list despite being in the oil patch, and (3) the convergence schedule's 10-year medium-rate hold amplifies boom-contaminated working-age rates through the fertility multiplier.

---

## 2. Current Dampening Configuration (Post-ADR-040)

From `config/projection_config.yaml` (lines 134-147):

```yaml
dampening:
  enabled: true
  factor: 0.60
  counties:
    - "38105"  # Williams
    - "38053"  # McKenzie
    - "38061"  # Mountrail
    - "38025"  # Dunn
    - "38089"  # Stark
  boom_periods:
    - [2005, 2010]
    - [2010, 2015]
    - [2015, 2020]  # ADR-040
```

Additionally, male migration dampening (0.80x) applies in periods 2005-2010 and 2010-2015 only.

**Billings County (38007) is NOT in the dampened counties list.**

---

## 3. Post-Dampening Migration Rates by Period

All rates below are annual migration rates **after** dampening has been applied (i.e., the rates stored in `data/processed/migration/residual_migration_rates.parquet`, processed 2026-02-17T21:30:04Z).

### 3a. All age-sex cells (unweighted mean)

| Period | McKenzie (38053) | Williams (38105) | Billings (38007) |
|--------|------:|------:|------:|
| 2000-2005 | -0.00529 | -0.00459 | -0.02517 |
| 2005-2010 | +0.00315 (D) | +0.00655 (D) | -0.00071 |
| 2010-2015 | +0.03996 (D+M) | +0.02129 (D+M) | **+0.03013** |
| 2015-2020 | +0.01601 (D) | +0.01180 (D) | -0.01008 |
| 2020-2024 | -0.01827 | -0.02283 | **+0.02386** |

D = period dampening (0.60x); M = male dampening (0.80x additional); Billings has no dampening applied.

### 3b. Working-age rates (ages 20-34, unweighted mean)

| Period | McKenzie | Williams | Billings |
|--------|------:|------:|------:|
| 2000-2005 | +0.01501 | -0.00663 | -0.01708 |
| 2005-2010 | +0.01508 | +0.02355 | -0.00435 |
| 2010-2015 | **+0.08548** | **+0.07007** | **+0.08669** |
| 2015-2020 | +0.05464 | +0.04340 | +0.01466 |
| 2020-2024 | -0.00533 | +0.00128 | +0.05155 |

The dampened 2010-2015 working-age rate for McKenzie is still **+8.5% annually**. This is the rate *after* applying 0.60x (and 0.80x male dampening). The undampened rate was approximately +13.3% annually for males and +11.7% for females.

Billings' 2010-2015 working-age rate of +8.7% is completely undampened. For a county of ~950 people, this was driven by approximately 30-40 individuals.

### 3c. Reconstructed pre-dampening rates (boom periods only)

| Period | County | Sex | Stored Rate | Dampening | Original Rate |
|--------|--------|-----|------:|-----------|------:|
| 2010-2015 | McKenzie | Male | +0.0379 | 0.60 x 0.80 = 0.48x | +0.0789 |
| 2010-2015 | McKenzie | Female | +0.0421 | 0.60x | +0.0701 |
| 2015-2020 | McKenzie | Male | +0.0179 | 0.60x | +0.0298 |
| 2015-2020 | McKenzie | Female | +0.0141 | 0.60x | +0.0235 |
| 2010-2015 | Williams | Male | +0.0223 | 0.48x | +0.0465 |
| 2010-2015 | Williams | Female | +0.0203 | 0.60x | +0.0338 |

---

## 4. Convergence Window Analysis

### 4a. Window-to-period mapping

From `data/processed/migration/convergence_metadata.json`:

| Window | Config Range | Mapped Periods |
|--------|-------------|----------------|
| Recent | [2023, 2025] | (2020, 2024) |
| Medium | [2014, 2025] | (2010, 2015), (2015, 2020), (2020, 2024) |
| Long-term | [2000, 2025] | All 5 periods |

### 4b. Window averages (mean of all age-sex cells)

| Window | McKenzie | Williams | Billings | Cass (Fargo) | Burleigh |
|--------|------:|------:|------:|------:|------:|
| Recent | -0.01827 | -0.02283 | **+0.02386** | +0.00312 | +0.00150 |
| Medium | **+0.01256** | +0.00342 | **+0.01464** | +0.00624 | +0.00590 |
| Long-term | +0.00711 | +0.00245 | +0.00361 | +0.00549 | +0.00537 |

### 4c. Working-age (20-34) window averages

| Window | McKenzie | Williams | Billings |
|--------|------:|------:|------:|
| Recent | -0.00533 | +0.00128 | **+0.05155** |
| Medium | **+0.04493** | **+0.03825** | **+0.05097** |
| Long-term | +0.03297 | +0.02634 | +0.02629 |

McKenzie's medium-window working-age rate of +4.5% per year is held for 10 years (projection years 6-15). For ages 25-29, the medium rate reaches **+8.5% (females) and +6.3% (males)**. These are the rates applied to the population at Williston's and Watford City's prime working ages.

### 4d. Convergence schedule output

Rates at key projection years (mean of all age-sex cells, from `data/processed/migration/convergence_rates_by_year.parquet`):

| Proj Year | Year Offset | Phase | McKenzie | Williams | Billings | Cass |
|-----------|--------:|-------|------:|------:|------:|------:|
| 2026 | 1 | Recent->Medium | -0.01211 | -0.01758 | +0.02201 | +0.00312 |
| 2028 | 3 | Recent->Medium | +0.00023 | -0.00708 | +0.01832 | +0.00468 |
| 2030 | 5 | Medium start | +0.01256 | +0.00342 | +0.01464 | +0.00624 |
| 2035 | 10 | Medium hold | +0.01256 | +0.00342 | +0.01464 | +0.00624 |
| 2040 | 15 | Medium end | +0.01256 | +0.00342 | +0.01464 | +0.00624 |
| 2045 | 20 | Long-term start | +0.00711 | +0.00245 | +0.00361 | +0.00549 |
| 2050 | 25 | Long-term hold | +0.00711 | +0.00245 | +0.00361 | +0.00549 |
| 2055 | 30 | Long-term hold | +0.00711 | +0.00245 | +0.00361 | +0.00549 |

---

## 5. Population-Weighted Rate Analysis (McKenzie)

The unweighted mean migration rate understates the effective growth impact because working-age cohorts (which have the highest positive rates) are disproportionately large in oil counties due to the boom-era population structure.

McKenzie County 2025 base population by age group with medium-hold (year 10) convergence rate:

| Age Group | Pop | % of Total | Migration Rate | Migration Contribution |
|-----------|----:|------:|------:|------:|
| 0-4 | 935 | 6.2% | 0.00000 | 0.0 |
| 5-9 | 1,013 | 6.7% | +0.03796 | +38.5 |
| 10-14 | 990 | 6.5% | +0.03462 | +34.3 |
| 15-19 | 1,096 | 7.2% | +0.01854 | +20.3 |
| **20-24** | **1,231** | **8.1%** | **+0.02497** | **+30.7** |
| **25-29** | **1,054** | **6.9%** | **+0.07354** | **+77.5** |
| **30-34** | **1,078** | **7.1%** | **+0.03627** | **+39.1** |
| 35-39 | 1,032 | 6.8% | +0.02621 | +27.0 |
| 40-44 | 973 | 6.4% | +0.01898 | +18.5 |
| 45-49 | 797 | 5.2% | +0.02337 | +18.6 |
| 50+ | 4,993 | 32.9% | various | -5.0 |
| **Total** | **15,192** | **100%** | **+0.01829 (weighted)** | **+278.1** |

The population-weighted rate (+0.0183) is 46% higher than the unweighted mean (+0.0126) because the age structure is skewed toward young adults who carry the highest positive rates.

The 25-29 age group alone, with 6.9% of the population, contributes 28% of total migration inflow. This age group adds ~78 people per year during the medium hold phase, and those in-migrants immediately begin producing children -- creating a compounding growth engine.

---

## 6. Why ADR-040 Was Insufficient

### 6a. The dampened boom-era rates are still enormous

McKenzie's 2010-2015 period, even after 0.60x dampening (and 0.80x male dampening), still shows a mean annual rate of **+4.0%** across all age-sex cells, with working-age cells at **+8.5%**. For context:
- Cass County (Fargo), the state's fastest-growing metro, has an all-period mean rate of +0.54%
- The state average medium-hold rate is -0.58% (negative)
- A +4% annual rate sustained for even 5 years implies +22% growth from migration alone

The 2010-2015 Bakken boom was an *extreme* statistical outlier. McKenzie County's population doubled from ~6,400 (2010) to ~12,300 (2015). Even at 60% of those rates, the residuals still dominate the multi-period average.

### 6b. ADR-040's expected impact was overestimated

ADR-040 predicted McKenzie would drop from +114% to approximately +40-60%. The actual result is +84%.

The error in estimation: ADR-040 correctly identified that the undampened 2015-2020 period was inflating the medium window. But the ADR focused narrowly on the 2015-2020 period's contribution and did not adequately account for the fact that the dampened 2010-2015 rates are still massive. Even with ADR-040:

- Medium window went from +0.016 (pre-ADR-040) to +0.013 (post-ADR-040)
- That is only a 19% reduction in the medium-window rate
- The medium rate is held for 10 years of the 30-year projection

### 6c. The convergence schedule amplifies boom contamination

The fundamental structural issue is the 5-10-5 convergence schedule:

1. **Years 1-5**: Linear ramp from recent to medium. For McKenzie, this goes from -0.018 to +0.013, crossing zero around year 3. Only 2 years of negative migration.
2. **Years 6-15**: Hold at medium (+0.013). Ten straight years of positive migration built on boom-era rates.
3. **Years 16-20**: Converge to long-term (+0.007). Still positive.
4. **Years 21-30**: Hold at long-term (+0.007). Positive to the end.

Despite the most recent data showing negative migration, the projection only has ~2 years of net out-migration before boom-contaminated averages take over for the remaining 28 years.

### 6d. Fertility multiplier on young adult in-migrants

The cohort-component engine applies migration to each age cohort independently (confirmed in `cohort_projections/core/migration.py`, line 88: `migration_amount = population * migration_rate`). Each year's working-age in-migrants then produce children via the fertility module. There is no rate cap in the migration engine.

For McKenzie at the medium-hold rate:
- Ages 25-29 receive +7.4% annual migration (females at +8.5%)
- Those in-migrants are in peak childbearing years (fertility rate ~0.10)
- Each year's 25-29 female in-migrants produce ~10% more births in subsequent years
- Over 10 years of medium hold, this creates compounding growth

---

## 7. Billings County: The Undampened Outlier

### 7a. Billings is not on the dampened list

Billings County (38007) is absent from the `dampening.counties` list in the config. The five dampened counties are Williams, McKenzie, Mountrail, Dunn, and Stark.

### 7b. Billings shows the same boom-era migration spike

Billings' 2010-2015 migration rate of **+0.030** (annual, undampened) is comparable to the dampened oil counties. Its working-age rate of **+0.087** is the highest of any county examined. This was driven by approximately 102 net migrants into a county of ~850 people.

### 7c. Small-population noise amplifies the problem

Billings County has only ~1,071 people in 2025. Individual-level migration events create enormous percentage rates:

- 2020-2024 ages 20-24 Female: +16.5% annual rate (16 people)
- 2020-2024 ages 40-44 Male: +19.9% annual rate (24 people)
- 2020-2024 ages 80-84 Male: +14.1% annual rate (9 people)

These rates are statistical noise, not demographic trends. But the convergence schedule treats them as signals and projects them forward.

### 7d. Recent period is positive (unlike McKenzie/Williams)

For McKenzie and Williams, the recent (2020-2024) rate is negative, which provides partial self-correction in years 1-5. For Billings, the recent rate is **positive** (+0.024), so the convergence schedule starts high and converges to an even higher medium rate. There is no self-correcting negative phase.

### 7e. Recommendation: Add Billings to the dampened list

Billings County should be added to `dampening.counties` in the config:
- It is geographically located in the Bakken oil patch (between McKenzie and Stark)
- Its 2010-2015 migration spike is boom-driven
- Its small population makes it more vulnerable to rate distortion, not less
- Without dampening, it projects +88% growth -- the highest of all oil counties

However, adding Billings to the dampened list alone will not solve the problem. It would dampen 2005-2010, 2010-2015, and 2015-2020 rates, but the 2020-2024 positive rate driven by small-population noise would remain and would still dominate the recent window.

---

## 8. Root Cause Diagnosis

The aggressive growth rates originate from **three interacting pipeline stages**, not from any single step:

### Stage 1: Residual computation (sufficient but could be stronger)

`cohort_projections/data/process/residual_migration.py` (`apply_period_dampening`, line 388)

The 0.60x dampening reduces boom-era rates but is insufficient for the extreme magnitude of the Bakken boom. McKenzie's 2010-2015 undampened rate was +6.7% (all ages) / +13% (working-age males). At 0.60x, these become +4.0% / +6.3% -- still enormous by any demographic standard.

### Stage 2: Convergence averaging (the primary amplifier)

`cohort_projections/data/process/convergence_interpolation.py` (`compute_period_window_averages`, line 70)

The medium window [2014, 2025] maps to three periods: (2010-2015), (2015-2020), (2020-2024). The boom-era periods, even dampened, dominate the average. The medium rate is then held for 10 years (6-15) of the projection.

For McKenzie, the medium window computation:
- 2010-2015 dampened: +0.040
- 2015-2020 dampened: +0.016
- 2020-2024: -0.018
- **Average: +0.013** -- driven by the 2010-2015 outlier

### Stage 3: Cohort-component engine (no safeguards)

`cohort_projections/core/migration.py` (`apply_migration`, line 88)

Migration is applied as `population * migration_rate` with no cap, ceiling, or plausibility check on the rate. The population-weighted effective rate (+0.018) compounds with fertility, producing growth that far exceeds the simple arithmetic of the mean rate.

---

## 9. Recommendations

### Recommendation A: Reduce dampening factor from 0.60 to 0.40 for 2010-2015

The 2010-2015 period is the primary driver. At 0.40x instead of 0.60x, the medium window for McKenzie would drop from +0.0126 to approximately +0.0074, nearly halving the growth impulse.

**Impact estimate**: McKenzie would drop from +84% to approximately +40-50% over 30 years.

**Tradeoff**: Departs from the SDC 2024 methodology's 0.60x factor. Should be documented as an ADR amendment.

### Recommendation B: Add Billings County (38007) to the dampened list

Config-only change. Add `"38007"` to `dampening.counties`. This addresses the +88% growth projection for a county of 1,071 people.

### Recommendation C: Implement a migration rate cap in the convergence step

Add a plausibility cap to `calculate_age_specific_convergence` (or as a post-processing step): no county-age-sex cell's convergence rate should exceed N times the state average for that age-sex cell, or an absolute cap (e.g., +/- 5% annual for any cell).

**Rationale**: This is a structural safeguard that would prevent any future boom/bust episode from producing implausible projections, regardless of dampening configuration.

**Suggested implementation**: In `convergence_interpolation.py`, after computing the interpolated rate for each year, clip to `max(rate, state_avg * 2.0)` or `clip(-0.05, +0.05)`.

### Recommendation D: Consider narrowing the medium window for oil counties only

Instead of [2014, 2025] (which captures 2010-2015), use [2019, 2025] for oil counties only. This would make the medium window = 2020-2024 only, which shows negative migration for McKenzie/Williams.

**Tradeoff**: Requires conditional logic in the convergence pipeline (per-county window definitions), adding complexity.

### Priority ordering

1. **Recommendation B** (add Billings to dampened list) -- lowest risk, config-only
2. **Recommendation A** (reduce factor to 0.40) -- moderate impact, config-only
3. **Recommendation C** (rate cap) -- structural fix, requires code change
4. **Recommendation D** (narrow medium window) -- largest impact but most complex

---

## 10. Supporting Data Files

| File | Path | Description |
|------|------|-------------|
| Period rates | `data/processed/migration/residual_migration_rates.parquet` | Post-dampening rates by period, county, age, sex |
| Averaged rates | `data/processed/migration/residual_migration_rates_averaged.parquet` | Multi-period averaged rates |
| Convergence rates | `data/processed/migration/convergence_rates_by_year.parquet` | Year-varying rates for projection years 1-30 |
| Residual metadata | `data/processed/migration/residual_migration_metadata.json` | Processing date 2026-02-17, confirms ADR-040 applied |
| Convergence metadata | `data/processed/migration/convergence_metadata.json` | Window mappings, schedule, rate summaries |
| McKenzie projection | `data/projections/baseline/county/nd_county_38053_projection_2025_2055_baseline.parquet` | Full cohort projection output |
| Williams projection | `data/projections/baseline/county/nd_county_38105_projection_2025_2055_baseline.parquet` | Full cohort projection output |
| Billings projection | `data/projections/baseline/county/nd_county_38007_projection_2025_2055_baseline.parquet` | Full cohort projection output |

## 11. Code References

| Component | File | Key Function | Line |
|-----------|------|-------------|------|
| Period dampening | `cohort_projections/data/process/residual_migration.py` | `apply_period_dampening()` | 388 |
| Male dampening | `cohort_projections/data/process/residual_migration.py` | `apply_male_migration_dampening()` | 515 |
| Period averaging | `cohort_projections/data/process/residual_migration.py` | `average_period_rates()` | 558 |
| Window averaging | `cohort_projections/data/process/convergence_interpolation.py` | `compute_period_window_averages()` | 70 |
| Convergence interpolation | `cohort_projections/data/process/convergence_interpolation.py` | `calculate_age_specific_convergence()` | 130 |
| Migration application | `cohort_projections/core/migration.py` | `apply_migration()` | 16 |
| Config | `config/projection_config.yaml` | `rates.migration.domestic.dampening` | 134-147 |
