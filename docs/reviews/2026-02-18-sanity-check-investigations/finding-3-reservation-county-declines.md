# Finding 3: AIAN Reservation County Declines

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-02-18 |
| **Investigator** | Claude Code (Opus 4.6) |
| **Parent Review** | [Projection Output Sanity Check](../2026-02-18-projection-output-sanity-check.md) |
| **Status** | Confirmed -- projected declines are significantly steeper than historical trends; root cause identified as residual method amplification combined with absence of migration dampening mechanisms |

---

## 1. Executive Summary

Three AIAN reservation-area counties in North Dakota are projected to lose 45-47% of their population over 30 years under the baseline scenario. This investigation confirms that these projected declines are **2.7x to 11x steeper than their historical 2000-2020 trajectories**, and identifies two compounding root causes:

1. **Residual method amplification**: The cohort-survival residual method produces out-migration estimates 2-3x larger than PEP component estimates for these counties, likely due to census coverage issues on tribal lands and survival rate calibration mismatches.

2. **Absence of deceleration mechanisms**: The convergence interpolation schedule converges from "recent" to "long-term" rates, but when all historical windows show substantial out-migration, there is no population floor, migration dampening, or asymptotic convergence that would cause the decline to decelerate over time.

**Recommendation**: These projections should not be published without either (a) recalibrating migration rates using PEP component data rather than the residual method, or (b) implementing an out-migration deceleration mechanism for counties that would otherwise decline by more than 30-40% over the projection horizon.

---

## 2. Projected Population Trajectories

### 2.1 Baseline Scenario Results

| County | FIPS | Reservation | Pop 2025 | Pop 2040 | Pop 2055 | 30-Yr Change |
|--------|------|-------------|--------:|--------:|--------:|------------:|
| Benson | 38005 | Spirit Lake | 5,759 | 4,241 | 3,031 | **-47.4%** |
| Sioux | 38085 | Standing Rock (ND) | 3,667 | 2,634 | 1,940 | **-47.1%** |
| Rolette | 38079 | Turtle Mountain | 11,688 | 8,644 | 6,353 | **-45.6%** |

### 2.2 Annualized Decline Rates by Period

| County | 2026-2030 | 2031-2035 | 2036-2040 | 2041-2045 | 2046-2050 | 2051-2055 |
|--------|----------:|----------:|----------:|----------:|----------:|----------:|
| Benson | -1.98%/yr | -1.97%/yr | -2.11%/yr | -2.04%/yr | -2.25%/yr | -2.35%/yr |
| Sioux | -2.37%/yr | -2.10%/yr | -2.07%/yr | -1.93%/yr | -2.01%/yr | -2.11%/yr |
| Rolette | -1.98%/yr | -1.96%/yr | -2.04%/yr | -2.01%/yr | -2.08%/yr | -2.00%/yr |

**Critical observation**: The decline **accelerates** for Benson County (from -2.0%/yr to -2.4%/yr), holds roughly constant for Rolette, and shows only modest deceleration for Sioux. There is no natural deceleration over the 30-year horizon. In Benson's case, the decline actually worsens in the final decade because the shrinking fertile-age population produces fewer births while migration rates remain constant.

### 2.3 Age Structure Erosion (Benson County Example)

| Year | Total Pop | Under 18 | Working Age (18-64) | Senior (65+) | Fertile Women (15-49) |
|------|----------:|---------:|-------------------:|-------------:|---------------------:|
| 2025 | 5,759 | 1,362 (23.7%) | 3,401 (59.0%) | 996 (17.3%) | 1,296 (22.5%) |
| 2035 | 4,719 | 1,030 (21.8%) | 2,831 (60.0%) | 859 (18.2%) | 1,021 (21.6%) |
| 2045 | 3,825 | 746 (19.5%) | 2,342 (61.2%) | 737 (19.3%) | 771 (20.2%) |
| 2055 | 3,031 | 503 (16.6%) | 1,862 (61.4%) | 667 (22.0%) | 538 (17.8%) |

The under-18 population declines from 1,362 to 503 (-63%), and fertile-age women decline from 1,296 to 538 (-58%). This creates a **demographic spiral**: out-migration of young adults reduces the birth cohort, which in turn reduces future natural increase, amplifying the population decline beyond what migration alone would produce.

---

## 3. Comparison with Historical Trends

### 3.1 Historical Population Timeline

| Year | Benson (38005) | Sioux (38085) | Rolette (38079) | Source |
|------|---------------:|-------------:|--------------:|--------|
| 2000 | 6,964 | 4,044 | 13,674 | Census 2000 |
| 2010 | 6,660 | 4,153 | 13,937 | Census 2010 |
| 2020 | 5,959 | 3,893 | 12,188 | Census 2020 (base) |
| 2025 | 5,759 | 3,667 | 11,688 | PEP Vintage 2025 |
| 2055 | 3,031 | 1,940 | 6,353 | **Projected** |

### 3.2 Historical vs. Projected Annual Decline Rates

| County | 2000-2020 Historical | 2025-2055 Projected | Ratio (Proj/Hist) |
|--------|---------------------:|--------------------:|------------------:|
| Benson | -0.78%/yr | -2.12%/yr | **2.7x steeper** |
| Sioux | -0.19%/yr | -2.10%/yr | **11.0x steeper** |
| Rolette | -0.57%/yr | -2.01%/yr | **3.5x steeper** |

**Sioux County** is the most extreme: it actually *grew* slightly from 2000-2010 (4,044 to 4,153, +2.7%) before declining in 2010-2020. Its projected -2.1%/yr decline is 11x steeper than its 20-year historical average. Even accounting for the fact that recent trends may have accelerated, this magnitude of discontinuity is implausible.

### 3.3 Comparison with Other Declining Counties

The three reservation counties form a stark outlier cluster:

| Rank | County | FIPS | 30-Yr Change | Is Reservation? |
|------|--------|------|-------------:|:-----------:|
| 1 | Benson | 38005 | **-47.4%** | Yes |
| 2 | Sioux | 38085 | **-47.1%** | Yes |
| 3 | Rolette | 38079 | **-45.6%** | Yes |
| 4 | Slope | 38087 | -35.5% | No |
| 5 | Pembina | 38067 | -21.7% | No |
| 6 | Walsh | 38099 | -17.3% | No |
| 7 | Bowman | 38011 | -15.8% | No |
| 8 | Foster | 38043 | -14.3% | No |

There is a **24-percentage-point gap** between the 3rd-worst county (Rolette, -45.6%) and the 4th-worst county (Slope, -35.5%). Of 53 counties, only 4 decline by more than 30%, and 3 of those are reservation counties. This gap is far too large to be coincidental and strongly suggests a reservation-specific methodological artifact.

Slope County (38087), the only non-reservation county with severe decline (-35.5%), has only 628 people -- the smallest county in North Dakota -- where small-number volatility naturally produces extreme rates.

---

## 4. Root Cause Analysis

### 4.1 Root Cause #1: Residual Method Amplification

The residual migration computation (`cohort_projections/data/process/residual_migration.py`) derives net migration as:

```
expected_pop = pop_start[age] * survival_rate
net_migration = pop_end[age+5] - expected_pop
migration_rate = annualize(net_migration / expected_pop)
```

Comparing the residual method output against PEP component estimates reveals **systematic amplification** of out-migration for these counties:

#### Benson County (38005) -- Residual vs. PEP by Period

| Period | Residual Net Mig | PEP Net Mig | Ratio |
|--------|----------------:|----------:|------:|
| 2000-2005 | -668 | -334 | 2.00x |
| 2005-2010 | -499 | -492 | 1.01x |
| 2010-2015 | -261 | -83 | 3.15x |
| 2015-2020 | **-1,318** | **-473** | **2.79x** |
| 2020-2024 | -586 | -225 | 2.61x |

#### Rolette County (38079) -- Residual vs. PEP by Period

| Period | Residual Net Mig | PEP Net Mig | Ratio |
|--------|----------------:|----------:|------:|
| 2000-2005 | -1,056 | -727 | 1.45x |
| 2005-2010 | -774 | -765 | 1.01x |
| 2010-2015 | -394 | +65 | -6.06x |
| 2015-2020 | **-3,199** | **-1,088** | **2.94x** |
| 2020-2024 | -1,118 | -438 | 2.55x |

#### Sioux County (38085) -- Residual vs. PEP by Period

| Period | Residual Net Mig | PEP Net Mig | Ratio |
|--------|----------------:|----------:|------:|
| 2000-2005 | -309 | -218 | 1.42x |
| 2005-2010 | -298 | -273 | 1.09x |
| 2010-2015 | -81 | +79 | -1.02x |
| 2015-2020 | **-817** | **-422** | **1.93x** |
| 2020-2024 | -393 | -156 | 2.52x |

**Key finding**: The 2015-2020 period shows the largest discrepancy for all three counties. The residual method reports 2-3x more out-migration than PEP components. This period spans the 2020 Census, which had well-documented undercounts on American Indian reservations.

**Statewide context**: The median residual-to-PEP ratio across all 53 counties for 2015-2020 is 1.16x. The reservation counties at 1.9-2.9x are clear outliers in this distribution.

**Why the residual method amplifies for these counties:**

1. **Census coverage issues on tribal lands**: The 2020 Census Post-Enumeration Survey documented significant undercounts on American Indian reservations nationally. If the 2020 Census base population used in the residual computation is too low, the residual attributes the coverage gap to out-migration.

2. **Survival rate calibration**: The residual method uses a single statewide survival rate table (CDC North Dakota 2020). AIAN populations have lower life expectancy than the statewide average. If survival rates are too high for these populations, the expected surviving population is overstated, and more of the actual population is attributed to out-migration.

3. **Period boundary population jumps**: The residual method uses population snapshots at period boundaries (e.g., PEP 2015 and Census 2020 base). These come from different estimation series with different methodologies, and the discontinuity is absorbed into the residual.

4. **Compounding across windows**: Because the residual overstatement affects the recent (2020-2024), medium (2010-2024), and long-term (2000-2024) windows, the convergence interpolation never encounters a "moderate" rate to converge toward. All three windows are amplified.

### 4.2 Root Cause #2: Absence of Migration Deceleration

The convergence interpolation (`cohort_projections/data/process/convergence_interpolation.py`) follows a Census Bureau 5-10-5 schedule:

| Phase | Years | Rate Used |
|-------|-------|-----------|
| Phase 1 | Yr 1-5 | Linear interpolation: Recent -> Medium |
| Phase 2 | Yr 6-15 | Hold at Medium rate |
| Phase 3 | Yr 16-20 | Linear interpolation: Medium -> Long-term |
| Post-schedule | Yr 21-30 | Hold at Long-term rate |

For the reservation counties, the convergence schedule produces migration rates:

#### Benson County Mean Migration Rates by Convergence Year

| Year | Calendar Year | Mean Rate | Source Window |
|------|-------------|----------:|:------------|
| 1 | 2026 | -0.03073 | Recent |
| 5 | 2030 | -0.02892 | Medium |
| 10 | 2035 | -0.02892 | Medium (holding) |
| 15 | 2040 | -0.02892 | Medium (holding) |
| 20 | 2045 | -0.02520 | Long-term |
| 25 | 2050 | -0.02520 | Long-term (holding) |
| 30 | 2055 | -0.02520 | Long-term (holding) |

The total convergence from year 1 to year 30 is only **0.6 percentage points** (from -3.07% to -2.52%). This is because all three historical windows (recent, medium, long-term) show substantial negative migration for these counties. The convergence schedule is designed to moderate between windows, but when all windows are negative, there is minimal moderation.

**The long-term rate of -2.5%/yr is then applied unchanged for years 20-30**, providing zero deceleration in the final decade. This is the period where the compounding effect is most severe.

The existing system has **no mechanism** for:
- Population floors (preventing counties from declining below some threshold)
- Migration rate dampening as population shrinks
- Asymptotic convergence toward zero migration
- Rate caps for sustained extreme out-migration

---

## 5. Historical PEP Migration Patterns (2000-2025)

### 5.1 Annual PEP Net Migration Data

#### Benson County (Spirit Lake Reservation)

| Period | Total Net Mig | Domestic | International | Avg/Year |
|--------|-------------:|--------:|---------:|--------:|
| 2000-2004 | -334 | -332 | -2 | -67/yr |
| 2005-2009 | -492 | -492 | 0 | -98/yr |
| 2010-2014 | -83 | -92 | +9 | -17/yr |
| 2015-2019 | -473 | -477 | +4 | -95/yr |
| 2020-2024 | -268 | -302 | +34 | -54/yr |
| **2025** | **+14** | **+2** | **+12** | -- |
| **Total** | **-1,636** | | | **-63/yr** |

#### Sioux County (Standing Rock Reservation, ND portion)

| Period | Total Net Mig | Domestic | International | Avg/Year |
|--------|-------------:|--------:|---------:|--------:|
| 2000-2004 | -218 | -216 | -2 | -44/yr |
| 2005-2009 | -273 | -273 | 0 | -55/yr |
| 2010-2014 | +79 | +77 | +2 | +16/yr |
| 2015-2019 | -422 | -434 | +12 | -84/yr |
| 2020-2024 | -156 | -165 | +9 | -31/yr |
| **2025** | **-55** | **-63** | **+8** | -- |
| **Total** | **-1,045** | | | **-40/yr** |

#### Rolette County (Turtle Mountain Reservation)

| Period | Total Net Mig | Domestic | International | Avg/Year |
|--------|-------------:|--------:|---------:|--------:|
| 2000-2004 | -727 | -758 | +31 | -145/yr |
| 2005-2009 | -765 | -821 | +56 | -153/yr |
| 2010-2014 | +65 | +60 | +5 | +13/yr |
| 2015-2019 | -1,088 | -1,092 | +4 | -218/yr |
| 2020-2024 | -492 | -516 | +24 | -98/yr |
| **2025** | **+17** | **+8** | **+9** | -- |
| **Total** | **-2,990** | | | **-115/yr** |

### 5.2 Key Patterns from PEP Data

1. **Episodic, not constant**: Migration is highly variable across periods. The 2010-2014 period shows near-zero or positive migration for all three counties, while 2015-2019 shows a dramatic spike in out-migration.

2. **Recent trend improvement**: The 2020-2024 period shows reduced out-migration compared to 2015-2019, and the single year 2025 shows positive net migration for Benson and Rolette. This recent moderation is not reflected in the projected rates.

3. **International migration is negligible**: These counties have minimal international migration (typically 0-15 people/year). The CBO restricted-growth scenario (which acts on international migration) would have virtually no effect on reservation counties.

4. **Domestic migration dominates**: Over 95% of net migration is domestic, driven by movement to urban areas within and outside North Dakota.

---

## 6. Sensitivity Analysis

### 6.1 Benson County Under Alternative Migration Assumptions

| Scenario | Migration Rate | Pop 2055 | 30-Yr Change |
|----------|-------------:|---------:|------------:|
| **Current projection** | **-2.5%/yr** | **3,031** | **-47.4%** |
| PEP-components rate | -1.0%/yr | ~4,590 | -20.3% |
| Historical 2000-2020 rate | -0.78%/yr | ~4,883 | -15.2% |
| Halved current rate | -1.25%/yr | ~4,279 | -25.7% |
| Pop-proportional dampening | Variable | ~3,594 | -37.6% |

Using PEP component-based migration rates instead of the residual method would reduce the projected decline from -47% to approximately -20%, a dramatic difference. Even halving the current rate produces a more plausible -26% decline.

### 6.2 Decomposition of Projected Decline

The -47.4% decline for Benson County is the combined effect of:

- **Migration**: Compound -2.5%/yr migration applied to a shrinking base accounts for approximately -53% (pure compounding), but natural increase partially offsets this.
- **Natural increase**: Births exceed deaths initially, but the gap narrows as the population ages and the fertile-age female population declines from 1,296 to 538 (-58%). The net natural increase contribution is approximately +5-6 percentage points over 30 years.
- **Demographic spiral**: Out-migration of young adults (ages 20-24 have the highest out-migration rates at -7% to -9%/yr) reduces future births, which further reduces the under-18 cohort, amplifying the total decline.

---

## 7. What the Migration Rates Look Like by Age and Period

### 7.1 Most Negative Age-Sex Rates (5-Period Average)

For Benson County, the age groups with the highest out-migration rates are:

| Age Group | Sex | Mean Rate (5 Periods) | Net Migration/Period |
|-----------|-----|---------------------:|--------------------:|
| 85+ | Female | -12.1%/yr | -41 |
| 85+ | Male | -9.0%/yr | -23 |
| 20-24 | Female | -8.7%/yr | -100 |
| 20-24 | Male | -7.7%/yr | -91 |
| 80-84 | Male | -6.6%/yr | -20 |
| 75-79 | Female | -5.2%/yr | -22 |

The elderly age groups (75+) show high "out-migration" rates, but this likely reflects mortality underestimation rather than actual geographic mobility. When statewide survival rates are applied to populations with lower life expectancy (common in AIAN communities), excess mortality is misattributed to out-migration.

The 20-24 age group out-migration (-7.7% to -8.7%/yr) reflects genuine young-adult departure from reservations for education and employment.

---

## 8. Is This Reservation-Specific or a General Rural Pattern?

**This is clearly reservation-specific.** The evidence:

1. **Magnitude gap**: The three reservation counties decline 45-47%; the next-worst non-reservation county (Slope, pop. 628) declines 35.5%; and the typical rural declining county declines 10-20%.

2. **Residual method bias**: The residual-to-PEP migration ratio for these counties (2-3x) is significantly higher than the statewide median (1.16x). The residual method is systematically more pessimistic for reservation counties.

3. **Convergence rate levels**: The convergence rates for reservation counties (-2.5% to -3.5%) are roughly 2x the rates for comparable non-reservation declining counties (-0.9% to -1.3%).

4. **Period-by-period patterns**: All three reservation counties show a dramatic migration spike in 2015-2020, while comparable non-reservation counties (Pembina, Walsh) show relatively stable rates across periods.

Non-reservation rural counties with declining populations (Pembina, Walsh, Bowman, etc.) have much more moderate projected declines in the -8% to -22% range, consistent with their historical trajectories.

---

## 9. Political and Sensitivity Considerations

### 9.1 Publication Risks

Publishing projections showing reservation populations declining by nearly half carries significant risks:

1. **Narrative of abandonment**: These projections could be interpreted as the state writing off reservation communities, even though they are purely mechanical extrapolations.

2. **Self-fulfilling prophecy**: If policymakers use these projections to reduce investment in reservation-area infrastructure, schools, or health services, the projections could contribute to the very out-migration they forecast.

3. **Tribal sovereignty context**: Population projections for reservation areas have policy implications for tribal governance, federal funding formulas (Indian Health Service, BIA), and tribal enrollment. Publishing steep declines without tribal consultation would be inappropriate.

4. **Methodological defensibility**: If challenged, we cannot credibly defend a -47% projection when:
   - Historical 20-year decline was only -15% (Benson) to -4% (Sioux)
   - PEP components show much lower migration than the residual method
   - The 2020 Census had known reservation undercounts
   - No demographic deceleration mechanism exists in the model

### 9.2 Recommendations for Publication

- **Do not publish the current reservation county projections without revision.**
- Consider presenting these counties with expanded confidence intervals or explicit caveats about the known Census 2020 undercount on tribal lands.
- Engage with tribal data partners (if any existing relationships) before publishing.
- Present a range of scenarios for reservation counties rather than a single-point projection.

---

## 10. Recommendations

### 10.1 Immediate Actions (Before Next Publication)

1. **Recalibrate reservation county migration rates**: Use PEP component-derived migration rates rather than residual-method rates for the three reservation counties. The PEP components directly measure net migration and are not subject to the survival-rate misallocation that inflates the residual method for AIAN populations.

2. **Implement a migration deceleration mechanism**: Add a configuration option that reduces out-migration rates for counties whose projected population would decline beyond a configurable threshold (e.g., 30% decline triggers gradual rate reduction). This mirrors the real-world dynamic where out-migration naturally slows as the "at-risk" population shrinks.

### 10.2 Medium-Term Methodological Improvements

3. **Race-specific survival rates**: The residual method currently uses a single statewide survival table. AIAN populations have lower life expectancy, which causes excess mortality to be misattributed to out-migration. Implementing race-specific survival rates (from CDC WONDER or NVSS) would significantly reduce the residual method bias for reservation counties.

4. **Period weighting adjustment**: The 2015-2020 period appears anomalous for reservation counties (possibly reflecting Census 2020 coverage issues). Consider downweighting or trimming this period from the convergence windows for counties with known census coverage concerns.

5. **PEP-residual hybrid**: Rather than choosing purely between PEP components and the residual method, develop a hybrid that uses PEP component totals as a constraint while retaining the residual method's age-sex allocation. This would anchor the total migration to the more reliable PEP estimate while preserving the demographic detail needed for cohort projection.

### 10.3 Longer-Term Considerations

6. **Tribal consultation**: Before publishing any projections for reservation-area counties, seek input from tribal data offices at Standing Rock, Turtle Mountain, and Spirit Lake regarding the plausibility of the migration assumptions and any planned economic development that could affect future trends.

7. **ADR documentation**: Draft an ADR (ADR-042 or similar) documenting the decision on how to handle reservation county projections, including the technical rationale for any adjustments and the political sensitivity considerations.

---

## 11. Technical Appendix

### 11.1 Config Convergence Settings (from `config/projection_config.yaml`)

```yaml
interpolation:
  method: "census_bureau_convergence"
  recent_period: [2023, 2025]     # Maps to residual period: (2020, 2024)
  medium_period: [2014, 2025]     # Maps to periods: (2010,2015), (2015,2020), (2020,2024)
  longterm_period: [2000, 2025]   # Maps to all 5 periods
  convergence_schedule:
    recent_to_medium_years: 5     # Years 1-5: recent -> medium
    medium_hold_years: 10         # Years 6-15: hold at medium
    medium_to_longterm_years: 5   # Years 16-20: medium -> long-term
```

Post year-20, the long-term rate is held constant through year 30.

### 11.2 Convergence Rate Trajectories

#### Benson County (38005)

| Window | Mean Migration Rate | Source Periods |
|--------|-------------------:|:-------------|
| Recent | -0.03118 | (2020, 2024) |
| Medium | -0.02892 | (2010,2015), (2015,2020), (2020,2024) |
| Long-term | -0.02520 | All 5 periods |

Convergence produces only a 0.6 ppt improvement over 30 years: -3.07% (yr 1) to -2.52% (yr 30).

#### Sioux County (38085)

| Window | Mean Migration Rate | Source Periods |
|--------|-------------------:|:-------------|
| Recent | -0.03510 | (2020, 2024) |
| Medium | -0.02861 | (2010,2015), (2015,2020), (2020,2024) |
| Long-term | -0.02616 | All 5 periods |

Convergence: -3.38% (yr 1) to -2.62% (yr 30), a 0.8 ppt improvement.

#### Rolette County (38079)

| Window | Mean Migration Rate | Source Periods |
|--------|-------------------:|:-------------|
| Recent | -0.03647 | (2020, 2024) |
| Medium | -0.03061 | (2010,2015), (2015,2020), (2020,2024) |
| Long-term | -0.02552 | All 5 periods |

Convergence: -3.53% (yr 1) to -2.55% (yr 30), a 1.0 ppt improvement.

### 11.3 Files Referenced

| File | Purpose |
|------|---------|
| `config/projection_config.yaml` | Projection configuration including convergence settings |
| `cohort_projections/data/process/convergence_interpolation.py` | Convergence interpolation implementation |
| `cohort_projections/data/process/residual_migration.py` | Residual migration rate computation |
| `cohort_projections/core/migration.py` | Migration application in projection engine |
| `cohort_projections/core/cohort_component.py` | Cohort component projection engine |
| `data/processed/pep_county_components_2000_2025.parquet` | PEP components of change data |
| `data/processed/migration/residual_migration_rates.parquet` | Computed residual migration rates |
| `data/processed/migration/convergence_rates_by_year.parquet` | Year-varying convergence rates |
| `data/projections/baseline/county/nd_county_38005_projection_2025_2055_baseline.parquet` | Benson County projection |
| `data/projections/baseline/county/nd_county_38085_projection_2025_2055_baseline.parquet` | Sioux County projection |
| `data/projections/baseline/county/nd_county_38079_projection_2025_2055_baseline.parquet` | Rolette County projection |

---

| Attribute | Value |
|-----------|-------|
| **Last Updated** | 2026-02-18 |
| **Version** | 1.0 |
| **Next ADR** | ADR-042 (reservation county migration methodology, if accepted) |
