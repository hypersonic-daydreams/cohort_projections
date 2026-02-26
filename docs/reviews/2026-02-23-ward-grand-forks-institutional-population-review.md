# Ward and Grand Forks County Institutional Population Review

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-02-23 |
| **Reviewer** | Claude Code (Opus 4.6), prompted by N. Haarstad |
| **Scope** | Investigation of military base and college/university population handling in Ward and Grand Forks counties |
| **Status** | Complete |
| **Related ADRs** | ADR-049 (college-age smoothing), ADR-052 (high-growth migration floor), ADR-043 (rate cap), ADR-046 (high-growth BEBR convergence), ADR-047 (county distributions), ADR-048 (single-year age) |

---

## Executive Summary

Ward County (Minot, FIPS 38101) and Grand Forks County (FIPS 38035) are projected to decline in the baseline scenario (-14.6% and -8.7% over 30 years, respectively) despite hosting significant institutional anchors: two Air Force bases (Minot AFB, Grand Forks AFB), two universities (Minot State, UND), and serving as regional economic centers.

This review finds that:

1. **Military populations are not explicitly handled.** The projection system treats military personnel as part of the general population. There is no group quarters (GQ) distinction, no military population floor, and no force-structure-based adjustment. The Census PEP data that drives our migration rates does not distinguish military from civilian net migration.

2. **The college-age smoothing (ADR-049) is insufficient for Grand Forks.** Grand Forks has the most extreme college cycle in the state: 15-24 in-migration rates of +3-5% annually, followed by a 25-29 graduation exodus of -11% to -13% annually. The net college cycle effect (ages 15-34 combined) is strongly negative (-0.069 annual rate), meaning that UND's enrollment cycle is a net population *drain* in the projection model. The 25-29 out-migration is not covered by the college-age smoothing, which only applies to ages 15-24.

3. **Cass County succeeds because Fargo retains graduates.** Cass has a similar 15-24 in-migration pattern but its 25-29 out-migration rate (-0.044) is less than half of Grand Forks's (-0.119), and its 30-34 rate is slightly positive (+0.003 vs -0.029). Fargo/West Fargo's economy retains young adults; Grand Forks and Minot do not.

4. **Ward County's decline is driven by pervasive negative migration across all age groups**, not just the college cycle. Only 4 of 36 age-sex cells have positive averaged migration rates. The 2020-2024 period is the most negative in 25 years of data, and it dominates the recent convergence window.

5. **The high-growth migration floor (ADR-052) is working as designed.** Ward high growth shows +29.0% and Grand Forks shows +38.4%, providing meaningful planning ranges.

**Key recommendations (prioritized):**
- **R1 (High priority):** Extend college-age smoothing to the 25-29 age group for college counties, addressing the asymmetric college cycle problem.
- **R2 (Medium priority):** Investigate a military population stability adjustment for counties with active military installations.
- **R3 (Low priority):** Consider a 2-period recent convergence window for counties where the most recent period is an outlier.

---

## 1. Current Institutional Population Handling

### 1.1 No Military-Specific Treatment

A comprehensive search of the codebase reveals **no military-specific adjustments** anywhere in the projection pipeline. The relevant files were examined:

- `cohort_projections/data/process/residual_migration.py` -- handles oil-boom dampening, college-age smoothing, reservation county PEP recalibration, and male migration dampening, but contains no references to military bases, group quarters, or institutional populations.
- `cohort_projections/data/process/convergence_interpolation.py` -- implements the convergence schedule with high-growth migration floor (ADR-052), but no military-specific logic.
- `config/projection_config.yaml` -- defines college counties (38035, 38017, 38101, 38015), oil counties, and reservation counties, but no military county classification.
- `cohort_projections/data/load/base_population_loader.py` -- loads total county populations from PEP data and distributes by age-sex-race using county-specific distributions. No distinction between household and group quarters populations.

**Implication:** Military base populations (active duty, dependents in on-base housing) are treated identically to civilian populations. When PEP data shows net out-migration for Ward County, the model cannot distinguish whether this reflects military force reductions, civilian departures, or temporary assignment rotations.

### 1.2 College-Age Smoothing (ADR-049)

The college-age smoothing blends county-specific migration rates with the statewide average using a 50/50 blend factor for ages 15-19 and 20-24 only:

```
smoothed_rate = 0.5 * county_rate + 0.5 * statewide_average
```

This is applied to four college counties: Grand Forks (38035), Cass (38017), Ward (38101), and Burleigh (38015).

**Critical gap:** The smoothing does NOT apply to age 25-29, which is where the "graduation exodus" manifests. For Grand Forks, the raw 25-29 out-migration rate is approximately -11% to -13% annually -- the most extreme rate for any non-infant age group in the county.

### 1.3 Group Quarters Data Availability

The base population data (`data/raw/population/nd_county_population.csv`) contains only total population columns: `population_2024`, `population_2025`, births, deaths, and net migration. There is no group quarters breakdown.

The PEP components data (`data/processed/pep_county_components_2000_2025.parquet`) has columns: `state_fips`, `county_fips`, `state_name`, `county_name`, `year`, `netmig`, `intl_mig`, `domestic_mig`, `residual`, `dataset_id`, `estimate_type`, `revision_status`, `uncertainty_level`, `geoid`, `is_preferred_estimate`. No GQ/institutional population columns are present.

The Census Bureau does publish group quarters data separately (in ACS Table B26001 and the GQ component of PEP), but this data has not been incorporated into the projection pipeline.

---

## 2. Ward County Analysis

### 2.1 Projection Trajectory

| Scenario | 2025 | 2035 | 2045 | 2055 | 30-Year Change |
|----------|-----:|-----:|-----:|-----:|:--------------:|
| Baseline | 68,233 | 65,153 | 62,601 | 58,255 | **-14.6%** |
| Restricted | 68,233 | 62,568 | 59,418 | 54,471 | -20.2% |
| High Growth | 68,233 | 75,620 | 82,679 | 88,031 | **+29.0%** |

### 2.2 Historical Migration Pattern

Ward County has the most volatile migration history among major ND counties, oscillating between significant out-migration and strong in-migration:

| Period | PEP Net Migration | Intl Migration | Domestic Migration | Key Events |
|--------|------------------:|---------------:|-------------------:|------------|
| 2001-2005 | -4,076 | -584 | -3,492 | MAFB drawdown, post-9/11 |
| 2006-2010 | -1,114 | +218 | -1,332 | Slow recovery |
| 2011-2015 | +5,100 | +1,252 | +3,848 | Oil boom + MAFB expansion |
| 2016-2020 | -5,960 | +892 | -6,852 | Oil bust, flood aftermath |
| 2021-2025 | -3,817 | +766 | -4,583 | COVID, continued decline |

The 5-period average net migration is approximately -1,973/year. However, the 2020-2025 period (-3,817) is the second-most-negative period and it dominates the "recent" convergence window.

### 2.3 Migration Rate Profile

Ward County has only **4 of 36 age-sex cells** with positive averaged migration rates:

| Age-Sex Cell | Averaged Rate | Interpretation |
|:-------------|:------------:|:---------------|
| 20-24 Male | +0.025 | Military enlistment/college enrollment |
| 15-19 Male | +0.009 | Youth arriving for college/military |
| 50-54 Male | +0.002 | Small positive (noise) |
| 15-19 Female | +0.000 | Near zero |
| All other cells | -0.001 to -0.053 | Pervasive out-migration |

**Key observation:** Ward's negative migration is not concentrated in the college cycle -- it is pervasive across nearly all age groups. The 85+ group has the most extreme negative rate (-0.053 for both sexes), likely reflecting elderly out-migration to larger medical centers. Ages 25-29 show the next-most-negative rates (-0.037 male, -0.026 female), reflecting both post-college and post-military departures.

### 2.4 Military Demographic Signature

Ward County shows a clear military-base demographic signature in its age-sex structure:

**Age over-representation (relative to state):** Ages 25-35 are over-represented by +0.15% to +0.24% per single year of age. This is the military active-duty age range (plus families). Ages 0-4 are also slightly over-represented (+0.08-0.14%), consistent with military family dependents.

**Male-heavy sex ratio:** Ages 20-25 show elevated male-to-female ratios (1.26-1.39 vs statewide 1.10-1.11), consistent with predominantly male military personnel. The male skew extends through ages 30-44 at a smaller magnitude.

### 2.5 Convergence Rate Analysis

| Year Offset | Baseline Mean Rate | High Mean Rate |
|:-----------:|:------------------:|:--------------:|
| 1 | -0.0196 | 0.0000 |
| 5 | -0.0118 | 0.0000 |
| 10 | -0.0118 | 0.0000 |
| 15 | -0.0118 | 0.0000 |
| 20 | -0.0121 | 0.0000 |

The high-growth floor (ADR-052) lifts Ward's convergence rates exactly to zero, producing growth entirely from natural increase. The baseline rate of approximately -1.2% annual net out-migration accumulates to -14.6% over 30 years.

---

## 3. Grand Forks County Analysis

### 3.1 Projection Trajectory

| Scenario | 2025 | 2035 | 2045 | 2055 | 30-Year Change |
|----------|-----:|-----:|-----:|-----:|:--------------:|
| Baseline | 74,501 | 72,781 | 71,201 | 68,035 | **-8.7%** |
| Restricted | 74,501 | 69,748 | 67,253 | 62,971 | -15.5% |
| High Growth | 74,501 | 83,443 | 92,938 | 103,091 | **+38.4%** |

### 3.2 Historical Migration Pattern

| Period | PEP Net Migration | Intl Migration | Domestic Migration |
|--------|------------------:|---------------:|-------------------:|
| 2001-2005 | -1,201 | +218 | -1,419 |
| 2006-2010 | -2,251 | +502 | -2,753 |
| 2011-2015 | +748 | +943 | -195 |
| 2016-2020 | -3,005 | +530 | -3,535 |
| 2021-2025 | -37 | +2,109 | -2,146 |

**Notable trend:** Grand Forks has seen dramatically increasing international migration (from +218 in 2001-2005 to +2,109 in 2021-2025), driven by UND's growing international student population. However, domestic migration remains persistently negative, and the 2021-2025 period (-37 total) is essentially zero net migration -- a significant improvement over the 2016-2020 period (-3,005). The most recent year (2025) shows +560 net migration.

### 3.3 The College Cycle Problem

Grand Forks has the most extreme "college cycle" in North Dakota -- massive in-migration at ages 15-24 followed by even larger out-migration at ages 25-29:

| Age Group | Male Rate | Female Rate | Interpretation |
|:---------:|:---------:|:-----------:|:---------------|
| 15-19 | +0.045 | +0.044 | Students arriving at UND |
| 20-24 | +0.044 | +0.025 | Continuing enrollment |
| 25-29 | **-0.110** | **-0.128** | Graduation exodus |
| 30-34 | -0.036 | -0.022 | Post-graduation continued departure |

**Net college cycle effect (ages 15-34):** -0.069 annual rate. This means that UND's enrollment cycle, as captured by the residual method, produces a net LOSS of population in the projection model.

The raw (pre-smoothing) 20-24 rates for Grand Forks are approximately +0.10 to +0.17 annually. After 50/50 smoothing, they are reduced to +0.03 to +0.04. But the 25-29 rates, which are NOT smoothed, remain at -0.110 to -0.128. Even after the rate cap clips them to -0.080, the asymmetry is severe: students counted as "in-migrants" at 20-24 are partially smoothed away, but their departure at 25-29 is counted at full value (minus the general rate cap).

### 3.4 University/Military Age Structure

Grand Forks shows the most extreme university-driven age bulge in the state. At ages 20-23, the county has **1.4 to 1.6 percentage points** more population than the state average -- the equivalent of having approximately 4,000-5,000 extra people in this narrow age band. This UND enrollment bulge is much larger than Ward's military/college bulge.

The military signature at Grand Forks AFB is visible but subtler than Ward's: ages 26-30 show elevated male-to-female ratios (1.25-1.33 vs statewide 1.11-1.13), consistent with a smaller military installation (Grand Forks AFB has approximately 1,500 military personnel vs MAFB's approximately 5,200).

### 3.5 College-Age Population Trajectory

Grand Forks's college-age (18-24) population drops from 14,126 (19.0% of total) in 2025 to 8,433 (11.6%) in 2035 in the baseline scenario. This dramatic decline in college-age share is a model artifact: the projection applies negative migration rates to the entire county, reducing the inflow of new students over time. In reality, UND enrollment is driven by university policies, not county-level migration trends.

---

## 4. Comparison with Cass County

Cass County (Fargo, NDSU) is projected at +29.8% baseline growth over 30 years. The contrast with Ward (-14.6%) and Grand Forks (-8.7%) is stark and requires explanation.

### 4.1 Migration Rate Comparison

| Metric | Cass | Ward | Grand Forks | Burleigh |
|:-------|:----:|:----:|:-----------:|:--------:|
| Mean migration rate | -0.0015 | -0.0121 | -0.0147 | +0.0023 |
| Positive cells | 17/36 | 4/36 | 4/36 | 25/36 |
| 15-19 in-migration | +0.026 | +0.005 | +0.045 | +0.010 |
| 20-24 in-migration | +0.035 | +0.012 | +0.034 | -0.019 |
| 25-29 out-migration | -0.044 | -0.032 | **-0.119** | +0.003 |
| 30-34 rate | +0.003 | -0.019 | -0.029 | +0.014 |
| Net college cycle (15-34) | **+0.021** | -0.033 | **-0.069** | +0.008 |

### 4.2 Why Cass Succeeds

1. **Graduate retention:** Cass/Fargo retains its 25-29 population. The 25-29 out-migration rate (-0.044) is less than half of Grand Forks's (-0.119) and less than Ward's (-0.032). Moreover, Cass's 30-34 rate is slightly positive (+0.003), indicating Fargo continues to attract young professionals.

2. **Balanced migration profile:** Cass has 17 positive and 17 negative cells (perfectly balanced), whereas Ward and Grand Forks each have only 4 positive and 30 negative. This means Cass has in-migration across many age groups, not just college ages.

3. **Persistent PEP positive migration:** Cass shows positive PEP net migration in every 5-year period (+4,811 in 2001-2005 up to +11,619 in 2021-2025). Ward shows positive migration in only one period (2011-2015: +5,100, the oil boom), and Grand Forks has been marginally positive in only two periods.

4. **Economic diversification:** Fargo/West Fargo has a diversified economy (healthcare, technology, manufacturing, financial services) that generates persistent employment growth across age groups. Ward (Minot) and Grand Forks are more dependent on institutional anchors (military, university) and lack Fargo's private-sector employment base.

### 4.3 Burleigh County Comparison

Burleigh (Bismarck) is instructive as a fourth college county (U of Mary, Bismarck State) that performs well (+20.0% baseline). Like Cass, Burleigh has positive migration across most age groups (25 of 36 cells), a positive mean migration rate (+0.0023), and does not suffer from the college cycle problem (its 25-29 rate is slightly positive at +0.003). Burleigh is a state capital with diversified government/healthcare employment, similar to Cass's economic profile.

---

## 5. Military Population Assessment

### 5.1 Is Military Population Explicitly Handled?

**No.** There is no military-specific adjustment anywhere in the codebase. The military population at MAFB (approximately 5,200 active duty + approximately 7,000 family members) and Grand Forks AFB (approximately 1,500 active duty + approximately 3,000 family members) is treated as ordinary civilian population. This means:

- Military PCS (permanent change of station) transfers show up as migration
- Force structure changes (base expansions, drawdowns) directly affect migration rates
- The 2020-2024 period may reflect COVID-related operational tempo changes, remote work policies, or hiring freezes that are specific to military operations and not representative of long-term trends

### 5.2 Military Force Structure Considerations

**Minot AFB (Ward County):**
- Hosts the 5th Bomb Wing (B-52H Stratofortress fleet) and the 91st Missile Wing (150 Minuteman III ICBMs)
- These are nuclear deterrence assets with decades-long strategic commitments
- The B-52 fleet is planned to remain operational into the 2040s (replaced by B-21 Raider, which may or may not be based at Minot)
- The ICBM mission is permanent as long as the US maintains a land-based nuclear deterrent
- MAFB has approximately 5,200 active-duty personnel as of 2025

**Grand Forks AFB (Grand Forks County):**
- Hosts the 319th Reconnaissance Wing (RQ-4 Global Hawk UAS)
- Smaller installation (approximately 1,500 active-duty personnel)
- More vulnerable to force-structure changes than MAFB's nuclear mission
- Grand Forks AFB has been at risk of BRAC (Base Realignment and Closure) in past rounds

### 5.3 Implications of No Military Adjustment

The lack of military-specific handling creates two problems:

1. **Downside risk is overstated:** If the 2020-2024 period captured a temporary military population dip (e.g., deployment cycle, hiring freeze), the projection treats this as a permanent trend. For MAFB with its nuclear deterrence mission, a 30-year projected decline of -14.6% implies that Minot would lose approximately 10,000 people -- substantially more than the entire MAFB active-duty population. This is plausible only if the civilian economy also declines.

2. **Force-structure permanence is unrecognized:** The ICBM mission at MAFB is a multi-generational commitment. The US nuclear triad has been maintained since the 1960s. A projection model that allows the host county to decline by 15% over 30 years does not reflect the institutional stability that this mission provides.

### 5.4 What a Military Adjustment Could Look Like

Several options exist:

- **Military population floor:** Hold the military-age male population (or a subset representing active-duty strength) constant at base-year levels. This prevents the projection from "eroding" the military population through negative migration.
- **GQ stabilization:** If group quarters data were incorporated, military barracks population could be held constant while household population is projected normally.
- **Migration rate overlay:** For age-sex cells that correspond to military demographics (males 18-35), replace or blend the county-specific migration rate with a military-specific rate based on historical reenlistment/PCS patterns.

---

## 6. College Population Assessment

### 6.1 Is College-Age Smoothing Sufficient?

**No, particularly for Grand Forks.** ADR-049 itself notes: "The 50/50 blend factor may not be aggressive enough for Grand Forks (UND), which has even more pronounced student migration patterns."

The analysis confirms this concern. The core problem is not the blend factor for ages 15-24 -- it is that **the 25-29 "graduation exodus" is not smoothed at all**.

### 6.2 The Asymmetric College Cycle Problem

The current smoothing architecture creates an asymmetry:

| Age Group | Treatment | GF Rate (smoothed) | GF Rate (raw est.) |
|:---------:|:---------:|:------------------:|:------------------:|
| 15-19 | 50/50 blend | +0.045 | +0.097 |
| 20-24 | 50/50 blend | +0.034 | +0.130 |
| 25-29 | **No smoothing** | -0.119 (raw) | -0.119 (raw) |
| 30-34 | No smoothing | -0.029 (raw) | -0.029 (raw) |

The smoothing correctly recognizes that the 15-24 in-migration is inflated by transient student enrollment. But it does not apply the same logic to the 25-29 out-migration, which is the mirror image of the same phenomenon: students who were counted as "in-migrants" now leave as "out-migrants." The residual method captures both legs of this cycle, but the smoothing only dampens one leg.

The rate cap (ADR-043) partially compensates by clipping the 25-29 rate from -0.119 to -0.080, but this still produces a severe net negative college cycle effect.

### 6.3 Grand Forks vs Cass: The Retention Signal

It is important to distinguish the "transient student" signal from the "real retention" signal. Not all 25-29 out-migration in college counties is students leaving after graduation -- some is genuine brain drain of non-student young adults.

Cass County provides the benchmark:
- Cass has a similarly high 20-24 in-migration (+0.035) but a much lower 25-29 out-migration (-0.044)
- The difference (-0.044 vs -0.119) reflects Fargo's superior graduate retention
- A 50/50 blend of Grand Forks's 25-29 rate with the statewide average (+0.002 male, +0.018 female) would produce rates of approximately -0.054, which is closer to Cass's level

### 6.4 Recommended Smoothing Extension

Extend the college-age smoothing to include the 25-29 age group for the same set of college counties. The same 50/50 blend factor is a reasonable starting point:

```yaml
adjustments:
  college_age:
    enabled: true
    method: "smooth"
    counties: ["38035", "38017", "38101", "38015"]
    age_groups: ["15-19", "20-24", "25-29"]  # Extended from ["15-19", "20-24"]
    blend_factor: 0.5
```

**Expected impact on Grand Forks:**
- 25-29 Male: -0.110 would become approximately -0.054 (50/50 blend with +0.002 statewide)
- 25-29 Female: -0.128 would become approximately -0.055 (50/50 blend with +0.018 statewide)
- Net college cycle effect would improve from -0.069 to approximately -0.027
- Baseline 30-year projection would likely improve from -8.7% to approximately flat or slightly positive

**Risk:** The 25-29 out-migration for Cass County (-0.044) is genuinely lower than Grand Forks because Fargo retains graduates. If we smooth Grand Forks's 25-29 rate to -0.054, we may slightly underestimate the real brain drain. However, the current -0.119 rate (even capped to -0.080 in convergence) is clearly too extreme -- it implies that 34% of all 25-29 year-olds leave the county over a 5-year span, which conflates student departure with genuine population loss.

---

## 7. Data Quality Assessment

### 7.1 Group Quarters Blindness

The most significant data quality limitation is the absence of group quarters (GQ) data in the projection pipeline. The Census Bureau publishes GQ populations separately, including:

- **Institutional GQ:** Correctional facilities, nursing homes, mental health facilities
- **Non-institutional GQ:** College dormitories, military barracks, group homes, religious group quarters

For Ward County, the MAFB barracks population (approximately 1,500-2,000 unaccompanied personnel) is counted in the GQ population. For Grand Forks, UND dormitory residents (approximately 3,000-3,500 students) and GFAFB barracks are in GQ. These populations are inherently transient -- military personnel rotate on 3-4 year PCS cycles, students on 4-6 year enrollment cycles.

When the residual method computes migration rates, these GQ rotations appear as large in/out-migration flows even though the underlying population is relatively stable (the beds remain filled, the institution continues operating). The college-age smoothing (ADR-049) partially addresses this for ages 15-24, but not for military ages or the 25-29 graduation exodus.

### 7.2 PEP Components Data

The PEP data correctly separates net migration into domestic and international components. Ward County's recent international migration has been positive and growing (+766 in 2021-2025), but domestic migration is strongly negative (-4,583). Grand Forks shows a similar pattern with even larger international gains (+2,109) partially offsetting domestic losses (-2,146).

The PEP data does NOT separate military from civilian components of domestic migration. This is a Census Bureau data limitation, not a projection system limitation.

### 7.3 Recent Period Anomaly

Both Ward and Grand Forks show their most negative migration in the 2020-2024 period, which coincides with COVID-19, reduced economic activity, and potentially military-specific disruptions (deployment cycles, recruitment shortfalls). The 2025 PEP estimates show improvement: Ward at -392 net (vs -1,044 in 2022), Grand Forks at +560 net (vs -1,038 in 2021). This suggests the 2020-2024 period may be an outlier, supporting the concern raised in ADR-052 about recent-window sensitivity.

---

## 8. Findings and Recommendations

### Finding 1: Grand Forks's 25-29 Out-Migration Creates an Asymmetric College Cycle Problem
**Severity:** High
**Evidence:** Grand Forks 25-29 averaged rate is -0.119 (male: -0.110, female: -0.128). This is 6x the statewide average for this age group. The rate is the mirror image of the 15-24 in-migration that IS smoothed, creating an asymmetric treatment where student arrivals are dampened but departures are counted at full value.

**Recommendation (R1):** Extend college-age smoothing to include the 25-29 age group. This is a one-line config change plus a rerun of the pipeline. The same 50/50 blend factor and same college county list should be used.

### Finding 2: Military Populations Have No Special Treatment
**Severity:** Medium
**Evidence:** No code, config, or data references to military bases, group quarters, or institutional populations. Ward County's base-year military-age (18-40) population is 24,840 (36.4% of total), declining to 16,857 (28.9%) by 2055 in the baseline -- a loss of 8,000 people from the military-age cohort alone.

**Recommendation (R2):** Investigate a military population stability adjustment. Options include:
- (a) Add Ward (38101) and Grand Forks (38035) to a "military_counties" config list with a PEP-recalibration approach similar to reservation counties (ADR-045)
- (b) Implement a "GQ stabilization" approach that holds a portion of the 18-35 male population constant at base-year levels
- (c) Add military base-specific migration rate overrides for the active-duty age bands

This should be scoped as a new ADR (ADR-055) given its potential impact.

### Finding 3: Ward County Baseline Decline May Be Overly Pessimistic
**Severity:** Medium
**Evidence:** Ward's baseline projects -14.6% decline, diverging 37 percentage points from the SDC 2024 reference (+23%). The 2020-2024 period is anomalously negative compared to the 25-year history. Ward's migration pattern is highly volatile (ranging from -5,960 to +5,100 in 5-year periods), making any single-period recent window unreliable.

**Recommendation (R3):** Consider a wider recent convergence window (2 periods, 2015-2024) for counties with high migration volatility. This was deferred in ADR-052 but warrants revisiting given the magnitude of the divergence from SDC. Alternative: add a "volatility dampening" mechanism that detects when the most recent period deviates more than 1.5 standard deviations from the historical mean and automatically widens the window.

### Finding 4: College-Age Smoothing Blend Factor Is Adequate for Ages 15-24
**Severity:** Low
**Evidence:** The 50/50 blend factor reduces Grand Forks's raw 20-24 rates from approximately +0.130 to +0.034. The smoothed rates are reasonable: they still show positive in-migration (students arrive) but at a rate that reflects a mix of transient students and genuine settlers. More aggressive smoothing (30/70 or 20/80) would push the 20-24 rates near zero or negative, which would be unrealistic for a major university county.

**Recommendation:** No change to the 15-24 blend factor. The current 50/50 blend is appropriate.

### Finding 5: Grand Forks's Recent International Migration Surge Is Not Fully Captured
**Severity:** Low-Medium
**Evidence:** Grand Forks international migration grew from +502 (2006-2010) to +2,109 (2021-2025), driven by UND's international enrollment growth. The 2025 net migration of +560 (with +509 international) suggests a structural shift. The residual method, which averages over five 5-year periods, dilutes this recent trend.

**Recommendation:** Monitor. The convergence schedule already weights the recent period more heavily in years 1-5. If international migration continues at 2023-2025 levels, the medium convergence window will incorporate this signal as new PEP data becomes available.

### Finding 6: The Rate Cap on 25-29 Rates Is Working But Is a Blunt Instrument
**Severity:** Low
**Evidence:** The general rate cap clips Grand Forks 25-29 rates from -0.119 to -0.080 in the convergence pipeline. This is the only protection against the extreme graduation exodus rate. Without the cap, Grand Forks would project much worse.

**Recommendation:** Keep the rate cap as a safety net, but address the root cause with 25-29 smoothing (R1). Once smoothing is applied, the 25-29 rates should fall to approximately -0.054, well within the -0.080 cap, making the cap no longer the binding constraint.

---

## 9. Appendix: Data Tables

### A. Ward County Averaged Migration Rates (All Age-Sex Cells)

| Age Group | Male Rate | Female Rate |
|:---------:|:---------:|:-----------:|
| 0-4 | 0.000000 | 0.000000 |
| 5-9 | -0.017152 | -0.016543 |
| 10-14 | -0.014540 | -0.010717 |
| 15-19 | +0.008925 | +0.000337 |
| 20-24 | +0.025048 | -0.000545 |
| 25-29 | -0.037363 | -0.026206 |
| 30-34 | -0.021626 | -0.015440 |
| 35-39 | -0.010454 | -0.015946 |
| 40-44 | -0.010717 | -0.014776 |
| 45-49 | -0.011917 | -0.001996 |
| 50-54 | +0.002182 | -0.006376 |
| 55-59 | -0.004680 | -0.005903 |
| 60-64 | -0.006293 | -0.009940 |
| 65-69 | -0.010482 | -0.009988 |
| 70-74 | -0.015310 | -0.015913 |
| 75-79 | -0.014969 | -0.009000 |
| 80-84 | -0.017648 | -0.013993 |
| 85+ | -0.053219 | -0.052837 |
| **Mean** | **-0.011679** | **-0.012543** |

### B. Grand Forks County Averaged Migration Rates (All Age-Sex Cells)

| Age Group | Male Rate | Female Rate |
|:---------:|:---------:|:-----------:|
| 0-4 | 0.000000 | 0.000000 |
| 5-9 | -0.016527 | -0.012809 |
| 10-14 | -0.011102 | -0.008147 |
| 15-19 | +0.045193 | +0.043822 |
| 20-24 | +0.043588 | +0.025241 |
| 25-29 | **-0.109987** | **-0.127756** |
| 30-34 | -0.036298 | -0.021981 |
| 35-39 | -0.019385 | -0.025037 |
| 40-44 | -0.015572 | -0.012066 |
| 45-49 | -0.007951 | -0.013384 |
| 50-54 | -0.002404 | -0.003419 |
| 55-59 | -0.008412 | -0.008805 |
| 60-64 | -0.008319 | -0.010466 |
| 65-69 | -0.019936 | -0.014506 |
| 70-74 | -0.011552 | -0.009842 |
| 75-79 | -0.023367 | -0.013853 |
| 80-84 | -0.015992 | -0.000743 |
| 85+ | -0.054284 | -0.041970 |
| **Mean** | **-0.015128** | **-0.014207** |

### C. PEP Net Migration Comparison (5-Year Sums)

| Period | Ward | Grand Forks | Cass | Burleigh |
|:------:|-----:|------------:|-----:|---------:|
| 2001-2005 | -4,076 | -1,201 | +4,811 | +3,046 |
| 2006-2010 | -1,114 | -2,251 | +6,137 | +4,132 |
| 2011-2015 | +5,100 | +748 | +12,890 | +8,450 |
| 2016-2020 | -5,960 | -3,005 | +5,468 | +366 |
| 2021-2025 | -3,817 | -37 | +11,619 | +3,427 |
| **25-Year Total** | **-9,867** | **-5,746** | **+40,925** | **+19,421** |

### D. College Cycle Effect Summary

| County | 15-19 | 20-24 | 25-29 | 30-34 | Net (15-34) |
|:-------|:-----:|:-----:|:-----:|:-----:|:-----------:|
| Grand Forks | +0.045 | +0.034 | **-0.119** | -0.029 | **-0.069** |
| Ward | +0.005 | +0.012 | -0.032 | -0.019 | -0.033 |
| Cass | +0.026 | +0.035 | -0.044 | +0.003 | **+0.021** |
| Burleigh | +0.010 | -0.019 | +0.003 | +0.014 | +0.008 |

### E. Convergence Rates at Medium Hold (Year Offset 10)

| County | Baseline Mean | High Mean | Cells at Rate Cap |
|:-------|:------------:|:---------:|:-----------------:|
| Ward | -0.0118 | 0.0000 | 0 |
| Grand Forks | -0.0104 | +0.0010 | 2 (25-29 M/F at -0.080) |
| Cass | -0.0001 | 0.0000 | 0 |
| Burleigh | +0.0033 | +0.0053 | 0 |
