# Migration Assumption Time Periods in Cohort-Component Projections: Best Practices Review

## Purpose

This document reviews published best practices and the actual methodologies used by major state, federal, and international demography offices for selecting migration assumption base periods in cohort-component population projection models. The review was conducted to evaluate our initial regime-weighted averaging approach (ADR-035) and inform the design of a more defensible methodology (ADR-036).

## Key Finding: Optimal Base Period Length

The most directly relevant empirical research comes from **Smith and Tayman**, the leading scholars on subnational projection methodology. Their study "The relationship between the length of the base period and population forecast errors" (*International Journal of Forecasting*, 2003) found:

> **Increasing the length of the base period up to 10 years improves forecast accuracy, but further increases generally have little additional effect.**

The one important exception: for **long-range forecasts of rapidly growing (or volatile) areas**, longer base periods provided substantially better accuracy. This finding is directly relevant to North Dakota, which experienced dramatic population volatility due to the Bakken oil boom/bust cycle.

Their comprehensive framework in *A Practitioner's Guide to State and Local Population Projections* (Springer, 2013) — the standard reference for state and local demographers — emphasizes that forecast accuracy depends on population size, growth rate, base period length, forecast horizon, and launch year. Chapter 6 covers migration methodology in detail, including data sources, measurement approaches, and assumption formation.

---

## Survey of Major Demography Office Practices

### U.S. Census Bureau

**National Projections (1999–2100)**:
- Net international migration trends based on **~35 years** of historical data (1980–2015) with multiple scenarios (low, main, high, zero immigration).

**State Interim Projections (2004–2030)**:
- Internal migration used **25 years of IRS data** (1975–2000).
- **Convergence/dampening methodology**: ARIMA models for the first 5 projection years, then linear interpolation toward the long-term series mean, then held constant at the mean.
- This is the closest federal precedent to "regime-aware" handling — they explicitly dampen recent volatility toward longer-term averages rather than extrapolating it forward.

**Key takeaway**: Long data series + dampening toward long-term mean. Recent trends inform only the near-term forecast.

### Texas Demographic Center

- Uses **one full intercensal decade** (2010–2020) as the core base period.
- Produces three scenarios: the mid scenario assumes migration patterns "remain similar to the past two decades" (~20 years of context).
- The "1.0 scenario" uses full 2010–2020 rates; the "0.5 scenario" uses half those rates.

**Key takeaway**: 10-year base period, equal weights, scenarios for uncertainty.

### Colorado State Demography Office

- Uses an **economic-demographic linked model** where job growth drives migration rather than extrapolating historical rates.
- Economic forecast establishes labor demand; cohort-survival model indicates labor supply from existing population; the difference determines net migration.
- Out-migration rates based on historical Census/CPS data and held constant; in-migration determined by the economic model.

**Key takeaway**: Migration tied to economic drivers, not historical extrapolation. This inherently handles regime changes.

### Florida BEBR (Bureau of Economic and Business Research)

Arguably the **most sophisticated multi-method approach** for county projections in the United States.

**State-level cohort-component projections**: Domestic migration rates based on **ACS data from 2010–2019** (~10 years), with rates weighted to account for recent changes.

**County-level projections**: Uses **six different projection techniques with varying base periods**:
- 2-year base (2018–2020)
- 5-year base (2015–2020)
- 10-year base (2010–2020)
- 15-year base (2005–2020)
- 20-year base (2000–2020)
- Additional trend-based methods

This produces **eleven separate projections per county**, from which **trimmed averages** are calculated (removing the two highest and two lowest before averaging the remaining nine).

Their explicit rationale:

> "County growth patterns are so volatile that a single technique based on data from a single time period may provide misleading results."

**Key takeaway**: Multiple base periods averaged together, with trimming to remove outlier periods. No single time window dominates.

### Weldon Cooper Center (University of Virginia)

- Uses the **Hamilton-Perry reduced form** of the cohort-component method.
- Cohort Change Ratios (CCRs) calculated from **two successive censuses** (effectively a 10-year base period, 2010–2020).
- Their philosophy: "Carefully constructed, simple methodologies can yield more reliable results than some elaborate and complicated modeling techniques."

**Key takeaway**: 10-year base period via census-to-census comparison, simplicity valued.

### Cornell Program on Applied Demographics (New York)

- Cohort-component model with in- and out-migration rates that **trend toward constant rates**.
- Projects at the state level first, then uses those as controls for county projections (top-down approach).
- Domestic migration flows identified as "the biggest driver of population change in New York and also hardest to project."

**Key takeaway**: Dampening/convergence toward long-term rates, hierarchical state→county control.

### Minnesota State Demographic Center

- **2024–2075 projections** use a shift-share method with Cohort Change Ratios examining historical patterns.
- Projects at five-year increments for all 87 counties.

### Indiana (STATS Indiana)

**2010–2040 projections**: Migration rates averaged from **1990–2000 and 2000–2005** (15 years, equally weighted). Implemented a **convergence method** causing rates to "gradually move toward zero."

**Updated 2025–2060 projections**: Uses 2010–2020 Census data (10-year base). For county-level short-term projections, shift-share rates from 2021–2023 **"with the rates for the most recent year given extra weight."** For 2030–2050, they average **13 different projection models**, removing the two highest and two lowest before averaging the remaining nine.

**Key takeaway**: 10-year primary base, modest recency weighting for short-term, multi-model averaging with trimming for medium-term. Convergence toward zero for long-term.

### Missouri

- Migration data from **2000–2007** (7 years of recent data), building on 1990–2000 rates.
- Applies a **"root function" that progressively decays net migration controls** at each successive projection interval.

**Key takeaway**: Explicit dampening/decay of migration rates over the projection horizon.

---

## International Comparators

### UK Office for National Statistics (ONS)

- **Changed from a 25-year average to a 10-year average** for long-term international migration assumptions, based on advice from their Migration Expert Advisory Panel.
- Uses a 10-year average of international migration (ending mid-2023) for the principal assumption.
- For subnational projections: **5-year averages** of internal migration.
- For the short-term: **linear interpolation** from the most recent estimate to the long-term assumption (rather than jumping directly to the long-run average).

**Key takeaway**: 10-year average as primary, interpolation from recent values to long-term mean for the near-term projection.

### Statistics Canada

Uses a **multi-scenario approach** with six distinct reference periods for interprovincial migration:

| Scenario | Reference Period | Duration |
|----------|-----------------|----------|
| M1 | 2000/2001 to 2024/2025 | 25 years |
| M2 | 2000/2001 to 2012/2013 | 13 years |
| M3 | 2006/2007 to 2010/2011 | 5 years |
| M4 | 2008/2009 to 2016/2017 | 9 years |
| M5 | 2013/2014 to 2021/2022 | 9 years |
| M6 | 2022/2023 to 2024/2025 | 3 years |

**Explicitly uses equal weighting** within each scenario period: "Averages are used so that all of the years can be assigned equal weights, regardless of the population sizes and number of out-migrants."

For the M1 (medium-growth) scenario: migration rates over the first 10 projection years are a **linear interpolation from recent rates (2022–2025 average) toward the long-term average (2000–2025)**, with rates held constant thereafter.

**Key takeaway**: Equal weights within periods, multiple scenarios to span uncertainty, interpolation from recent to long-term.

---

## Synthesis: Common Patterns Across Practitioners

### Base Period Length

| Base Period | Practitioners |
|-------------|---------------|
| **2–5 years** | BEBR (one of several techniques); Indiana (short-term); Statistics Canada M6 |
| **10 years** | UK ONS; Indiana (primary); Weldon Cooper; Texas (primary) |
| **10–15 years** | BEBR (multi-technique); Indiana (combined); Missouri |
| **20–25 years** | BEBR (one technique); Statistics Canada M1; Census Bureau State Projections |

**Consensus: 10 years is the primary base period, with longer series used for context or as alternative scenarios.**

### Weighting Within Periods

- **Equal weighting is the dominant standard.** Statistics Canada is explicit: equal weights regardless of population size. UK ONS uses unweighted averages.
- **Modest recency weighting** is practiced by Indiana (short-term only) and implicitly by Census Bureau ARIMA models.
- **No major office assigns 50%+ weight to a 3-year window** as their primary projection assumption.

### Handling Volatile Periods

Four strategies predominate:

1. **Convergence/dampening** (Census Bureau, Indiana, Missouri, Cornell): Migration rates decay toward zero or toward a long-term mean over the projection horizon.
2. **Economic-demographic linking** (Colorado): Migration driven by economic forecasts, not historical extrapolation.
3. **Multi-method averaging with varying base periods** (Florida BEBR, Indiana 2025–2060): Using multiple base periods simultaneously and averaging results.
4. **Multiple scenarios** (Texas, Statistics Canada, Census Bureau): Low/medium/high rather than a single assumption.

### Recency Bias Risks

The literature consistently warns about recency bias, particularly for volatile areas:

- Smith and Tayman document that projection errors are **positively related to growth rate volatility**. Areas with volatile migration are already the hardest to project — recency bias amplifies this risk.
- The Population Reference Bureau notes that "international migration can be particularly unpredictable and difficult to incorporate into projection assumptions" because migration flows "often result from short-term changes in economic, social, political, or environmental factors."
- For North Dakota specifically, research documents that "the volatility of oil prices and agricultural prices makes it difficult to project migration and population trends" (UND thesis, 2016).

---

## Assessment of Our Initial Regime-Weighted Approach

### What Was Defensible

- Using the full 2000–2024 range (24 years) as the data source is consistent with longer windows used by Census Bureau, Statistics Canada (M1), and BEBR.
- The impulse to account for regime-specific patterns (oil boom/bust) reflects real structural breaks in ND migration.
- Boom dampening (0.60) aligns with SDC 2024 methodology.

### What Was Problematic

1. **50% weight to 3 years is aggressive by any standard.** No major state or federal demography office assigns this much weight to such a short recent period as their primary projection. Even Indiana, which explicitly weights recent years more, does so modestly. Statistics Canada's "most recent 3 years" scenario (M6) is presented as one extreme of six scenarios, not the central case.

2. **"Regime-weighted averaging" is not a recognized standard methodology.** While the underlying concept has precedents, the specific formulation of giving 50% weight to 3 years and 50% to 21 years lacks precedent in published methodology literature.

3. **North Dakota's volatility makes recency bias especially dangerous.** Williams County lost 2,665 people in a single year during the oil downturn. If the 2022–2024 window captures a recovery upswing, giving it 50% weight could substantially overproject migration in oil-affected counties.

4. **Per-year influence is extremely unequal.** Each year in 2022–2024 receives ~16.7% of total influence vs. ~1.4% per year in 2000–2010. This 12:1 ratio is difficult to justify methodologically.

### Effective Per-Year Weights Under Initial Regime Approach

| Regime | Years | Total Weight | Per-Year Weight |
|--------|-------|-------------|-----------------|
| Recovery (2022–2024) | 3 | 50% | 16.7% |
| Bust+COVID (2016–2021) | 6 | 25% | 4.2% |
| Pre-Bakken (2000–2010) | 11 | 15% | 1.4% |
| Boom (2011–2015) | 5 | 10% | 2.0% |

---

## Recommended Approaches

Based on this review, two approaches are recommended for implementation (see ADR-036):

### Primary: BEBR-Style Multi-Period Averaging

Compute equal-weighted migration averages from multiple base periods (5-year, 10-year, 15-year, and full 24-year), generate separate projections from each, and produce a trimmed average. This is the most defensible approach for volatile counties because no single time window dominates.

### Secondary: Census Bureau-Style Interpolation

For the near-term (first 5 projection years), interpolate from the most recent 3-year average toward a 10-year equal-weighted average. For medium-term (years 5–15), use the 10-year average. For long-term (15+ years), allow rates to converge toward the full-period average. This explicitly handles the "launch year" effect documented by Smith and Tayman.

Both methods should produce scenario variants (low, baseline, high) and be compared against each other and against the SDC 2024 projections.

---

## References

### Empirical Research
1. Smith, S.K. & Tayman, J. (2003). "The relationship between the length of the base period and population forecast errors." *International Journal of Forecasting*, 19(3), 413-424. https://pubmed.ncbi.nlm.nih.gov/12155386/
2. Smith, S.K., Tayman, J. & Swanson, D.A. (2013). *A Practitioner's Guide to State and Local Population Projections*. Springer. https://link.springer.com/book/10.1007/978-94-007-7551-0
3. Smith, S.K., Tayman, J. & Swanson, D.A. (2013). "Subnational Population Forecasts: Migration." Chapter 6 in *A Practitioner's Guide*. https://link.springer.com/chapter/10.1007/978-94-007-7551-0_6
4. Rayer, S. & Smith, S.K. (2010). "Factors affecting the accuracy of subcounty population forecasts." *Journal of Planning Education and Research*, 30(4), 413-428.

### State and Federal Methodology Reports
5. U.S. Census Bureau (2000). "Methodology and Assumptions for the Population Projections of the United States: 1999 to 2100." Working Paper No. 38. https://www.census.gov/library/working-papers/2000/demo/POP-twps0038.html
6. U.S. Census Bureau. "State Interim Projections Methodology." https://wonder.cdc.gov/wonder/help/populations/population-projections/methodology.html
7. Texas Demographic Center. "Vintage 2024 Projections." https://demographics.texas.gov/Projections/
8. Colorado State Demography Office. https://demography.dola.colorado.gov/assets/html/aboutSDO.html
9. Florida BEBR. "Projections of Florida Population by County, 2025-2050." University of Florida. https://bebr.ufl.edu/wp-content/uploads/2024/01/projections_2024.pdf
10. Weldon Cooper Center for Public Service. "Virginia Population Projections." https://www.coopercenter.org/virginia-population-projections
11. Cornell Program on Applied Demographics. "County Projections." https://pad.human.cornell.edu/counties/projections.cfm
12. Minnesota State Demographic Center (2024). "2024-2075 Long-Term Population Projections: Methods." https://mn.gov/admin/assets/2024-Minnesota-Long-Term-Population-Projections-Methods%20(1)_tcm36-626201.pdf
13. Indiana STATS. "2010-2040 Population Projections Methodology." https://www.stats.indiana.edu/about/pop_proj_10-40.asp
14. Indiana STATS. "2025-2060 Population Projections Methodology." https://www.stats.indiana.edu/about/pop-projections-2025-60.asp
15. Missouri Budget and Planning. "Population Projections Methodology." https://budplan.oa.mo.gov/demographic-info/population-projections/methodology

### International
16. UK Office for National Statistics (2024). "National population projections: migration assumptions, 2022-based." https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationprojections/methodologies/nationalpopulationprojectionsmigrationassumptions2022based
17. Statistics Canada (2026). "Population Projections for Canada, Provinces and Territories, 2025 to 2073: Technical Report on Methodology and Assumptions." https://www150.statcan.gc.ca/n1/pub/17-20-0003/172000032026002-eng.htm

### North Dakota Specific
18. Population Reference Bureau. "Understanding Population Projections: Assumptions Behind the Numbers." https://www.prb.org/resources/understanding-population-projections-assumptions-behind-the-numbers/
19. Wilson, L. (2016). North Dakota oil boom population dynamics. UND Theses. https://commons.und.edu/theses/1906/
20. Rayer, S. (2008). "Population Forecast Errors: A Primer for Planners." *Journal of Planning Education and Research*, 27(4), 417-430.

---

*Document prepared: 2026-02-12*
*Supports: ADR-036 (Migration Averaging Methodology)*
*Supersedes methodology discussion in: ADR-035 (Census PEP Migration Data Source)*
