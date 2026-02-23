# Finding 6: Baseline Trajectory Plausibility

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-02-18 |
| **Investigator** | Claude Code (Opus 4.6) |
| **Parent Review** | [Projection Output Sanity Check](../2026-02-18-projection-output-sanity-check.md) |
| **Status** | Conditionally Plausible -- Requires Prominent Caveats |

---

## 1. Summary of Finding

The baseline scenario projects North Dakota crossing 1 million population around 2054-2055, reaching 1,005,281 by 2055. This represents +25.8% growth over 30 years (CAGR of +0.77%/yr). The trajectory is **conditionally plausible** as a trend-continuation scenario, but carries significant caveats that must be communicated to stakeholders:

1. The 0.77%/yr growth rate exceeds ND's historical norm outside the 2010-2015 oil boom period.
2. Growth is overwhelmingly driven by sustained international migration at 2020-2025 elevated levels.
3. Nearly 90% of projected state growth is concentrated in just three counties (Cass, Burleigh, Williams).
4. The projection is a mechanical extrapolation of recent trends, not a forecast of the most likely outcome.

---

## 2. Historical Growth Context

### 2.1 ND Decadal Population (Census)

| Period | Start Pop | End Pop | Total Change | CAGR |
|--------|-----------|---------|-------------|------|
| 1930-1940 | 680,845 | 641,935 | -5.7% | -0.59% |
| 1940-1950 | 641,935 | 619,636 | -3.5% | -0.35% |
| 1950-1960 | 619,636 | 632,446 | +2.1% | +0.21% |
| 1960-1970 | 632,446 | 617,761 | -2.3% | -0.24% |
| 1970-1980 | 617,761 | 652,717 | +5.7% | +0.55% |
| 1980-1990 | 652,717 | 638,800 | -2.1% | -0.22% |
| 1990-2000 | 638,800 | 642,200 | +0.5% | +0.05% |
| **2000-2010** | **642,200** | **672,591** | **+4.7%** | **+0.46%** |
| **2010-2020** | **672,591** | **779,094** | **+15.8%** | **+1.48%** |
| **2020-2025** | **779,094** | **799,358** | **+2.6%** | **+0.52%** |

Key observations:
- The 2010-2020 decade (+1.48%/yr) was **historically unprecedented**, driven by the Bakken oil boom (2010-2015).
- Excluding the oil boom decade, ND's growth has typically been 0.0-0.5%/yr.
- The 2020-2025 period (+0.52%/yr) represents a return toward the non-boom norm.
- The projected 0.77%/yr **exceeds every decade except the oil boom decade**.

### 2.2 Actual 2024-2025 Growth Rate

From Census Vintage 2025 estimates:
- 2024 population: 793,387
- 2025 population: 799,358
- Growth: +5,971 (+0.75%)

The most recent single year is close to the projected CAGR, but this is the peak of a post-2020 acceleration driven by an international migration surge that may not persist.

---

## 3. Components of Change Analysis

### 3.1 Historical Migration Components (PEP State Totals)

Source: `data/processed/pep_county_components_2000_2025.parquet`

| Year | Net Migration | International | Domestic | Residual |
|------|-------------|---------------|----------|----------|
| 2000 | -1,456 | +258 | -1,714 | -33 |
| 2001 | -6,140 | +651 | -6,791 | -512 |
| 2002 | -3,797 | +264 | -4,061 | -522 |
| 2003 | -1,962 | -545 | -1,417 | -707 |
| 2004 | +1,964 | +1,025 | +939 | -684 |
| 2005 | -2,861 | +535 | -3,396 | -583 |
| 2006 | -1,088 | +815 | -1,903 | -395 |
| 2007 | -1,570 | +461 | -2,031 | +86 |
| 2008 | -203 | +583 | -786 | +49 |
| 2009 | +1,896 | +521 | +1,375 | +107 |
| 2010 | +1,253 | +436 | +817 | +44 |
| 2011 | +7,286 | +1,070 | +6,216 | +58 |
| 2012 | +11,588 | +1,132 | +10,456 | +420 |
| 2013 | +16,331 | +1,094 | +15,237 | +491 |
| 2014 | +10,275 | +783 | +9,492 | +320 |
| 2015 | +11,421 | +2,067 | +9,354 | +161 |
| 2016 | -4,866 | +1,436 | -6,302 | +38 |
| 2017 | -4,289 | +2,692 | -6,981 | +34 |
| 2018 | -1,295 | +1,063 | -2,358 | +13 |
| 2019 | -316 | +951 | -1,267 | +12 |
| 2020 | -417 | +30 | -447 | +101 |
| 2021 | -3,425 | +453 | -3,878 | -389 |
| 2022 | -190 | +2,554 | -2,744 | +11 |
| 2023 | +4,088 | +3,158 | +930 | -1 |
| 2024 | +3,660 | +4,083 | -423 | -22 |
| 2025 | +3,322 | +2,810 | +512 | +19 |

### 3.2 Period Averages (Annual)

| Period | Net Migration | International | Domestic |
|--------|-------------|---------------|----------|
| 2000-2004 | -2,278 | +331 | -2,609 |
| 2005-2009 | -765 | +583 | -1,348 |
| **2000-2009** | **-1,522** | **+457** | **-1,978** |
| 2010-2014 | +9,347 | +903 | +8,444 |
| 2015-2019 | +131 | +1,642 | -1,511 |
| **2010-2019** | **+4,739** | **+1,272** | **+3,466** |
| **2020-2025** | **+1,173** | **+2,181** | **-1,008** |
| **2023-2025** | **+3,690** | **+3,350** | **+340** |

### 3.3 Critical Structural Shift

The migration composition has undergone a fundamental structural change:

- **2000-2009**: International migration was modest (+457/yr avg), domestic migration was strongly negative (-1,978/yr). ND was losing population to other states.
- **2010-2015**: The oil boom produced massive domestic in-migration (+8,444/yr 2010-2014). International migration was a minor contributor.
- **2016-2022**: Post-boom domestic out-migration resumed (-2,000 to -7,000/yr). International migration grew substantially.
- **2023-2025**: International migration surged to +3,350/yr average, accounting for 91% of net migration (the `intl_share = 0.91` parameter). Domestic migration is near zero or slightly negative.

This means the baseline projection's growth depends almost entirely on sustained elevated international migration -- a phenomenon that is (a) historically very recent, (b) driven by national policy and global conditions outside ND's control, and (c) currently facing potential disruption from federal immigration enforcement.

### 3.4 Natural Increase

From Vintage 2025 stcoreview data:

| Year | Births | Deaths | Natural Increase |
|------|--------|--------|-----------------|
| 2021 | 10,010 | 7,831 | +2,179 |
| 2022 | 9,761 | 7,368 | +2,393 |
| 2023 | 9,729 | 6,936 | +2,793 |
| 2024 | 9,583 | 6,905 | +2,678 |
| 2025 | 9,760 | 7,130 | +2,630 |

Natural increase currently contributes approximately +2,600-2,800/yr. This is positive but insufficient alone to drive the projected growth trajectory -- without migration, the zero-migration scenario shows population peaking near 800,000 and declining by 2040.

---

## 4. Baseline Projection Trajectory

### 4.1 Year-by-Year State Population

Source: Aggregated from 53 county baseline parquet files (`data/projections/baseline/county/*_2025_2055_*.parquet`)

| Year | State Pop | Annual Change | Growth Rate |
|------|----------|--------------|-------------|
| 2025 | 799,358 | -- | -- |
| 2026 | 800,862 | +1,504 | +0.19% |
| 2027 | 804,276 | +3,414 | +0.43% |
| 2028 | 809,391 | +5,115 | +0.64% |
| 2029 | 816,010 | +6,620 | +0.82% |
| 2030 | 823,967 | +7,957 | +0.98% |
| 2031 | 832,342 | +8,375 | +1.02% |
| 2032 | 840,958 | +8,615 | +1.04% |
| 2033 | 849,790 | +8,832 | +1.05% |
| 2034 | 858,796 | +9,006 | +1.06% |
| 2035 | 867,915 | +9,119 | +1.06% |
| 2036 | 877,162 | +9,248 | +1.07% |
| 2037 | 886,387 | +9,224 | +1.05% |
| 2038 | 895,647 | +9,260 | +1.04% |
| 2039 | 904,996 | +9,349 | +1.04% |
| 2040 | 914,480 | +9,483 | +1.05% |
| 2041 | 923,636 | +9,156 | +1.00% |
| 2042 | 932,344 | +8,708 | +0.94% |
| 2043 | 940,597 | +8,253 | +0.89% |
| 2044 | 948,379 | +7,782 | +0.83% |
| 2045 | 955,679 | +7,300 | +0.77% |
| 2046 | 962,919 | +7,240 | +0.76% |
| 2047 | 967,345 | +4,427 | +0.46% |
| 2048 | 971,719 | +4,373 | +0.45% |
| 2049 | 976,034 | +4,316 | +0.44% |
| 2050 | 980,295 | +4,261 | +0.44% |
| 2051 | 984,799 | +4,504 | +0.46% |
| 2052 | 989,535 | +4,736 | +0.48% |
| 2053 | 994,523 | +4,988 | +0.50% |
| 2054 | 999,772 | +5,249 | +0.53% |
| 2055 | 1,005,281 | +5,509 | +0.55% |

### 4.2 Growth Rate Phases

The projection shows three distinct growth phases tied to the convergence interpolation schedule:

| Period | CAGR | Phase |
|--------|------|-------|
| **2025-2030** | **+0.61%** | Years 1-5: Recent-to-medium convergence (ramping up) |
| **2030-2035** | **+1.04%** | Years 6-10: Medium rate hold (peak growth) |
| **2035-2040** | **+1.05%** | Years 11-15: Medium rate hold (sustained peak) |
| **2040-2045** | **+0.89%** | Years 16-20: Medium-to-longterm convergence (slowing) |
| **2045-2050** | **+0.51%** | Years 21-25: Longterm rate hold |
| **2050-2055** | **+0.50%** | Years 26-30: Longterm rate hold |

The peak growth period (2030-2040) sees annual increases of +8,600 to +9,500, which is comparable to the oil boom years of 2011-2015. This is the most questionable phase of the projection.

### 4.3 Implied Net Migration

Using rough decomposition (age-0 as births proxy, estimated deaths from aging population with mortality improvement):

| Period | Avg Annual Change | ~Natural Increase | ~Net Migration |
|--------|------------------|-------------------|----------------|
| 2026-2030 | +4,922 | +1,419 | ~+3,500 |
| 2031-2035 | +8,989 | +2,902 | ~+6,090 |
| 2036-2040 | +9,313 | +3,792 | ~+5,520 |
| 2041-2045 | +8,000 | +4,081 | ~+3,920 |
| 2046-2050 | +4,485 | +3,622 | ~+860 |
| 2051-2055 | +4,997 | +2,789 | ~+2,210 |

The implied net migration in the 2030-2040 peak period (~5,500-6,100/yr) is **significantly higher than the 2020-2025 average of +1,173/yr** (from PEP data) and comparable to the **2023-2025 spike of +3,690/yr**.

---

## 5. Convergence Interpolation Analysis

### 5.1 Configuration

From `config/projection_config.yaml` and `data/processed/migration/convergence_metadata.json`:

| Window | Config Range | Mapped Periods | Description |
|--------|-------------|----------------|-------------|
| Recent | 2023-2025 | 2020-2024 | Most recent PEP period |
| Medium | 2014-2025 | 2010-2015, 2015-2020, 2020-2024 | Includes oil boom and post-boom |
| Longterm | 2000-2025 | All 5 periods (2000-2024) | Full 25-year history |

### 5.2 Convergence Schedule (5-10-5-hold)

- **Years 1-5** (2026-2030): Linear interpolation from Recent to Medium rate
- **Years 6-15** (2031-2040): Hold at Medium rate
- **Years 16-20** (2041-2045): Linear interpolation from Medium to Longterm rate
- **Years 21-30** (2046-2055): Hold at Longterm rate

### 5.3 The Medium Period Problem

The **medium period** (2010-2015, 2015-2020, 2020-2024) includes the oil boom years 2010-2015, when net migration averaged **+9,347/yr**. This significantly elevates the medium-term rate. The projection holds at this elevated medium rate for 10 years (2031-2040), producing the peak growth phase.

Even though boom dampening is applied (0.60 factor for Williams, McKenzie, Mountrail, Dunn, Stark counties in 2005-2015 periods, and now 2015-2020 per ADR-040), the medium period average still reflects:
- The massive boom-era domestic in-migration to urban centers like Cass County, which is NOT dampened
- The structural shift toward elevated international migration in the 2020-2024 period

### 5.4 Migration Rate Summary (State Average)

| Year Offset | Mean Rate | Notes |
|------------|-----------|-------|
| 1 (2026) | -0.00896 | Recent rate (net negative average across all counties) |
| 5 (2030) | -0.00575 | Converged to medium |
| 10 (2035) | -0.00575 | Holding at medium |
| 15 (2040) | -0.00575 | Still holding at medium |
| 20 (2045) | -0.00740 | Converged to longterm |
| 30 (2055) | -0.00740 | Holding at longterm |

Note: The negative average is misleading because **43 of 53 counties have negative migration rates** while the **10 positive-rate counties** (led by Cass, Burleigh, Williams, McKenzie) are much larger in population, so the absolute number of migrants is positive despite the negative average rate.

---

## 6. Geographic Concentration of Growth

### 6.1 Top 3 Counties

| County | 2025 Pop | 2055 Pop | Growth | CAGR | Share of State Growth |
|--------|----------|----------|--------|------|----------------------|
| Cass (Fargo) | 201,794 | 328,393 | +62.7% | +1.64% | 61% |
| Burleigh (Bismarck) | 103,251 | 139,449 | +35.1% | +1.01% | 18% |
| Williams (Williston) | 41,767 | 63,195 | +51.3% | +1.39% | 10% |
| **Top 3 subtotal** | **346,812** | **531,037** | **+53.1%** | -- | **89%** |
| Rest of state (50 counties) | 452,546 | 474,244 | +4.8% | +0.16% | 11% |

Cass County alone accounts for **61% of all projected state growth**. Its projected CAGR of +1.64%/yr for 30 years would make it one of the fastest-growing counties in the northern Great Plains.

### 6.2 Urbanization Implication

The baseline projects ND shifting from 43.4% to 52.8% of population in just three counties. The remaining 50 counties collectively grow only 4.8% over 30 years, meaning most rural ND is essentially stagnant or declining.

---

## 7. External Benchmarks

### 7.1 National CBO Projections

The Congressional Budget Office's January 2026 Demographic Outlook (2026-2056) projects:
- U.S. population: 349 million (2026) to 364 million (2056) -- CAGR of +0.14%
- Growth slowing from +0.3%/yr (2026-2035) to +0.1%/yr (2036-2056)
- Annual deaths exceeding annual births beginning in 2030
- Net immigration as the sole source of population growth after 2030

ND's projected **+0.77%/yr is 5.5x the national projected rate of +0.14%/yr**. While state-level growth can diverge from national trends, sustaining growth 5x the national rate for 30 years requires extraordinary sustained in-migration.

### 7.2 Peer State Growth (2020-2024)

Source: `data/raw/population/NST-EST2024-ALLDATA.csv`

| State | 2020 Pop | 2024 Pop | CAGR |
|-------|----------|----------|------|
| Montana | 1,087,230 | 1,137,233 | +1.13% |
| South Dakota | 887,948 | 924,669 | +1.02% |
| United States | 331,577,720 | 340,110,988 | +0.64% |
| North Dakota | 779,563 | 796,568 | +0.54% |
| Wyoming | 577,681 | 587,618 | +0.43% |
| Minnesota | 5,710,735 | 5,793,151 | +0.36% |

ND's recent actual growth (+0.54%/yr 2020-2024) lags Montana and South Dakota. The baseline projection assumes ND accelerates to 1.0%+/yr through the 2030s, which would make it one of the fastest-growing states in the region -- exceeding even Montana's current pace.

### 7.3 ND State Data Center Official Projections

The ND State Data Center (SDC) 2024 official projections:
- 2030: 831,543
- 2035: 865,397
- 2040: 890,424
- 2045: 925,101
- 2050: ~957,124

Our baseline is slightly lower than the SDC through 2040 (914,480 vs 890,424 -- actually our baseline is higher by 2040), and broadly comparable in the 2045-2050 range. The SDC methodology appears to use similar trend-continuation logic but may have different migration assumptions.

### 7.4 ND International Migration in National Context

| Year | ND Intl Rate (per 1,000) | US Intl Rate (per 1,000) | ND/US Ratio |
|------|--------------------------|--------------------------|-------------|
| 2021 | 0.58 | 1.13 | 0.51 |
| 2022 | 4.22 | 5.10 | 0.83 |
| 2023 | 5.44 | 6.87 | 0.79 |
| 2024 | 6.47 | 8.27 | 0.78 |

ND's international migration rate rose dramatically from 2021 to 2024, tracking the national surge. However, the CBO now projects a **historic decline in net international migration** nationally, from 2.7 million to 1.3 million in the 2024-2025 period. If ND's international migration tracks the national decline, the projection's key growth driver weakens substantially.

---

## 8. Plausibility Assessment

### 8.1 What Makes It Plausible

1. **Recent trend continuation**: The most recent year (2024-2025) showed +0.75% growth, matching the projected 30-year CAGR. The model is not projecting acceleration beyond what has been observed.
2. **Growing base population**: As population grows, the cohort-component method generates more births, which compounds growth.
3. **Convergence interpolation**: The schedule appropriately decelerates growth in the later decades (years 16-30), producing a more moderate long-run rate.
4. **ND economic fundamentals**: Fargo's diversified economy (tech, healthcare, education, agriculture services) supports sustained growth independent of energy cycles.
5. **Consistency with SDC official projections**: The trajectory is broadly comparable to the ND State Data Center's published projections.

### 8.2 What Makes It Questionable

1. **International migration dependence**: The `intl_share = 0.91` parameter means 91% of recent net migration is international. The baseline implicitly assumes this elevated international flow continues for decades. However:
   - National international migration is currently declining sharply (CBO projects -52% drop in 2024-2025)
   - Federal immigration policy is actively restricting flows
   - The 2022-2024 surge was driven by exceptional global displacement events that may not persist

2. **Medium period rate inflation**: The 10-year hold at the medium rate (years 6-15) is significantly elevated by the inclusion of oil boom years (2010-2015) in the medium period. This produces peak annual growth of +9,000-9,500 in the 2030s, comparable to oil boom levels but without a boom driver.

3. **Historically unprecedented sustained rate**: The 0.77%/yr CAGR would be the highest sustained growth rate in ND's modern history. Every previous period of elevated growth (1970s energy, 2010-2015 Bakken) was followed by stagnation or decline.

4. **Extreme geographic concentration**: With 89% of growth in 3 counties and 61% in Cass County alone, the projection is really a bet on Fargo-Moorhead's continued exceptional performance. Any disruption to Fargo's growth trajectory (economic downturn, housing constraints, regional competition) would dramatically affect the state total.

5. **CBO national context**: The national population is projected to grow at only +0.14%/yr through 2055. For ND to grow at 5.5x the national rate requires sustained competitive advantage in attracting and retaining population -- possible but far from assured.

### 8.3 Overall Assessment

**The baseline trajectory to 1 million is a reasonable upper bound for trend-continuation modeling, but should not be presented as a central forecast or most-likely outcome.**

The scenario correctly represents what happens if recent migration patterns persist. However, the recent pattern itself (2022-2025) is anomalous in ND's history, driven by a national international migration surge that is already reversing. The more defensible range for 2055 population is likely **900,000-1,050,000**, with the baseline sitting near the top of that range.

---

## 9. Recommendations for Stakeholder Communication

### 9.1 Framing Recommendations

1. **Do NOT describe the baseline as a "forecast" or "expected" outcome.** Use language like "if recent migration trends continue unchanged" or "trend-continuation scenario."

2. **Always present the baseline alongside the restricted growth scenario** to show the range of plausible outcomes. The restricted growth scenario (~870,000 by 2045) represents a more conservative trajectory that accounts for immigration policy changes.

3. **Highlight the migration dependence explicitly**: "Approximately 90% of recent net migration is international in origin. The baseline assumes this elevated level continues, while the restricted growth scenario models the impact of federal immigration enforcement."

4. **Contextualize with historical volatility**: "ND's migration has historically been highly volatile, swinging from -6,000/yr (2001) to +16,000/yr (2013). The projection smooths this volatility through averaging and convergence."

5. **Note the geographic concentration**: "Growth is heavily concentrated in the Fargo, Bismarck, and Williston metropolitan areas. Most of the state's 50 remaining counties are projected to see minimal growth or continued population decline."

### 9.2 Suggested Presentation Format

Present projections as a **range with scenarios**:

| Year | Restricted Growth | Baseline | High Growth |
|------|------------------|----------|-------------|
| 2030 | 801,067 | 823,967 | 815,534 |
| 2035 | 821,003 | 867,915 | 856,864 |
| 2040 | 846,998 | 914,480 | 904,698 |
| 2045 | 870,525 | 955,679 | 952,387 |

Note: The restricted growth and high growth scenarios have 20-year horizons (to 2045), while the baseline extends to 2055.

### 9.3 Key Talking Points for "1 Million by 2055"

- "Reaching 1 million is plausible under continued strong migration trends, but is not guaranteed."
- "The primary driver is net international migration, which is subject to federal policy and global conditions outside North Dakota's control."
- "If immigration enforcement reduces international flows as the CBO projects, population growth would slow significantly -- our restricted growth scenario shows this alternative path."
- "Growth is heavily concentrated in the Fargo-Moorhead, Bismarck-Mandan, and Williston areas. Most rural counties continue their long-term population decline in all scenarios."
- "The 1-million milestone, if reached, would not occur until the mid-2050s -- roughly a generation from now -- giving ND ample time to plan for either growth or slower-than-expected trends."

### 9.4 Potential Methodology Refinements

1. **Consider shortening the medium period hold**: The 10-year hold at the medium rate (which includes oil boom data) may be overly generous. A 5-year hold (5-5-5-15 schedule with longer longterm hold) could produce a more conservative trajectory.

2. **Sensitivity analysis on intl_share**: Run the baseline with intl_share values of 0.50, 0.75, and 0.91 to show how sensitive the trajectory is to the composition of migration.

3. **Year-by-year component reporting**: Output births, deaths, and net migration as separate columns in the projection output to enable transparent decomposition.

4. **External validation with Fargo-Moorhead Metro COG**: The FM Metro COG has published its own 2050 demographic forecast. Comparing the Cass County projection against their independent analysis would provide valuable validation.

---

## 10. Data Sources

All data extracted from project files:
- PEP components: `data/processed/pep_county_components_2000_2025.parquet`
- County population: `data/raw/population/nd_county_population.csv`
- Census estimates: `data/raw/population/NST-EST2024-ALLDATA.csv`
- Stcoreview: `data/raw/population/stcoreview_v2025_nd_parsed.parquet`
- Baseline projections: `data/projections/baseline/county/*_2025_2055_*.parquet`
- Convergence metadata: `data/processed/migration/convergence_metadata.json`
- Convergence rates: `data/processed/migration/convergence_rates_by_year.parquet`
- Methodology comparison: `data/projections/methodology_comparison/full_methodology_comparison.csv`
- Projection config: `config/projection_config.yaml`
- ADR-039: `docs/governance/adrs/039-international-only-migration-factor.md`

External sources:
- [CBO Demographic Outlook 2026-2056](https://www.cbo.gov/publication/61879)
- [CBO Updated Demographic Outlook 2025-2055](https://www.cbo.gov/publication/61390)
- [Census Bureau Vintage 2025 Population Estimates](https://www.census.gov/newsroom/press-kits/2026/national-state-population-estimates.html)
- [Census Bureau Population Growth Slowdown (2026)](https://www.census.gov/newsroom/press-releases/2026/population-growth-slows.html)
- [ND State Data Center](https://www.commerce.nd.gov/economic-development-finance/state-data-center)
