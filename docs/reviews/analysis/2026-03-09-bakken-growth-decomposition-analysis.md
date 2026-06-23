# Bakken County Growth Decomposition: Natural Increase vs. Migration

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-03-09 |
| **Author** | Claude Code (Opus 4.6) + N. Haarstad |
| **Scope** | Decomposition of projected growth in 6 Bakken oil counties into natural increase vs. net migration components |
| **Status** | Active reference |
| **Related ADRs** | ADR-040, ADR-051 (Rejected), ADR-061 |
| **Related Reviews** | [`2026-02-17-bakken-migration-dampening-review.md`](2026-02-17-bakken-migration-dampening-review.md), [`2026-03-04-projection-accuracy-analysis.md`](2026-03-04-projection-accuracy-analysis.md) |

---

## 1. Summary

This review decomposes the projected 30-year growth of the six Bakken oil counties to determine how much is driven by **demographic momentum** (natural increase from people already living there) versus **continued net in-migration** from boom-era rates persisting in the convergence schedule.

**Key finding**: For the two largest Bakken counties, growth is overwhelmingly driven by natural increase — not ongoing in-migration. Williams County's projected migration is actually *negative* over 30 years. McKenzie County shows modest positive migration (~18% of total growth), but the bulk of its growth (+82%) comes from natural increase driven by its extremely young age structure.

---

## 2. Projected Growth Trajectories (Baseline Scenario)

| County | FIPS | 2025 Pop | 2045 Pop (20yr) | 2055 Pop (30yr) | 20yr % | 30yr % |
|--------|------|----------|-----------------|-----------------|--------|--------|
| **McKenzie** | 38053 | 15,192 | ~22,585 | ~27,647 | +48.7% | +82.0% |
| **Williams** | 38105 | 41,767 | ~55,851 | ~65,393 | +33.7% | +56.6% |
| Stark | 38089 | 34,013 | — | ~42,070 | — | +23.7% |
| Mountrail | 38061 | 9,395 | — | ~9,573 | — | +1.9% |
| Dunn | 38025 | 4,058 | — | ~3,766 | — | -7.2% |
| Billings | 38007 | 1,255 | — | — | — | ~+54% |

For context, the statewide baseline growth is +10.4% over 30 years (799,358 to 882,146). McKenzie and Williams are growing 5-8x faster than the state average.

### SDC 2024 Comparison (20-Year Horizon)

| County | Our 20yr | SDC 20yr | Gap |
|--------|----------|----------|-----|
| McKenzie | +48.7% | +47.1% | +1.6pp |
| Williams | +33.7% | +33.4% | +0.3pp |

On a 20-year apples-to-apples basis, our projections align closely with the SDC 2024 reference. The divergence appears only on the 30-year horizon, where our convergence schedule extends projections beyond SDC's range. (This was the basis for rejecting ADR-051.)

---

## 3. Growth Decomposition: Natural Increase vs. Migration

### Williams County (38105)

| Component | 30-Year Contribution | Share of Growth |
|-----------|---------------------|-----------------|
| **Natural increase** | +25,859 | **109.5%** |
| Net migration | -2,233 | -9.5% |
| **Total growth** | **+23,626** | **100%** |

**Williams' growth is entirely demographic momentum.** Migration is a net drag — the county loses more people to out-migration than it gains. All projected growth comes from births exceeding deaths, driven by the young workforce that settled during the boom and is now in peak childbearing years.

The convergence schedule confirms this: Williams' mean migration rate reaches essentially **zero** (-0.000163/yr) at the medium hold (years 6-15). The projection is not projecting continued in-migration — it's projecting the natural demographic consequences of a young population.

### McKenzie County (38053)

| Component | 30-Year Contribution | Share of Growth |
|-----------|---------------------|-----------------|
| **Natural increase** | +10,215 | **82.0%** |
| Net migration | +2,239 | 18.0% |
| **Total growth** | **+12,455** | **100%** |

McKenzie shows modest positive migration contributing ~18% of growth. The medium-hold migration rate is +0.0068/yr — about 110 people per year on a base of ~16,000. This is a small absolute flow, but it compounds through fertility.

**The critical mechanism is age-specific**: McKenzie's 25-29 age group has a convergence rate of **+5.78%/yr held for 10 years** (years 6-15). These young adults enter peak fertility, producing children who age through the pyramid.

---

## 4. Why Natural Increase Is So High

### The Boom Left Behind a Young Population

The Bakken oil boom (2005-2015) attracted predominantly working-age adults (20-39). These workers settled, formed families, and had children. By 2025, the Bakken counties have an age structure dramatically younger than the state average.

**McKenzie age structure in 2025:**
- Working age (20-44): ~37% of population (vs ~30% statewide)
- Children (0-14): ~25% of population
- Elderly (65+): ~8% (vs ~17% statewide)

This young age structure produces sustained natural increase:
- Estimated ~209 births/year vs ~76 deaths/year = **+133 natural increase/year** (0.9% of population)
- Williams: ~650 natural increase/year (1.5% of population)

### The Compounding Effect

Natural increase is self-reinforcing:
1. Young boom-era settlers (now 30-45) are in peak or late childbearing years
2. Children born 2010-2020 are entering working ages by 2035-2045
3. The growing population base amplifies even modest per-capita natural increase
4. Each 5-year cohort of children adds to the working-age pool, producing the next generation

This is genuine demographic momentum — the same force that drives population growth in developing countries with young age structures. It cannot be "fixed" by dampening migration rates because it's not a migration phenomenon.

---

## 5. The Migration Rate Convergence Schedule

### Period-Level Residual Migration Rates (After Dampening)

| Period | McKenzie | Williams | Dampening Factor |
|--------|----------|----------|-----------------|
| 2000-2005 | -0.006413 | -0.005707 | 1.00 (no dampening) |
| 2005-2010 | +0.002013 | +0.005263 | 0.50 |
| 2010-2015 | +0.026517 | +0.014068 | 0.40 |
| 2015-2020 | +0.012871 | +0.009140 | 0.50 |
| 2020-2025 | -0.018922 | -0.023697 | 1.00 (no dampening) |

### Window Averages (Feed into Convergence)

| County | Recent (2020-25) | Medium (2010-25) | Long-term (all) |
|--------|-----------------|-------------------|-----------------|
| **McKenzie** | -0.0189 | **+0.0068** | +0.0032 |
| **Williams** | -0.0237 | **-0.0002** | -0.0002 |
| Mountrail | -0.0331 | -0.0106 | -0.0068 |
| Dunn | -0.0221 | -0.0087 | -0.0090 |
| Stark | -0.0185 | -0.0026 | -0.0029 |

Williams' medium-window rate is essentially zero — the dampening successfully neutralized boom-era migration. McKenzie's medium rate remains modestly positive at +0.68%/yr.

### Net Migration by Historical Period

| County | 2000-05 | 2005-10 | 2010-15 | 2015-20 | 2020-25 | Total |
|--------|---------|---------|---------|---------|---------|-------|
| McKenzie | -217 | +125 | +1,361 | +1,300 | -900 | +1,669 |
| Williams | -353 | +713 | +2,708 | +2,992 | -3,051 | +3,009 |
| Mountrail | -18 | +225 | +542 | -245 | -891 | -387 |
| Dunn | -250 | +4 | +232 | -232 | -276 | -522 |
| Stark | -641 | +309 | +1,554 | +739 | -1,656 | +305 |

The 2020-2025 period shows strong net out-migration for all counties except Billings. The boom migration has reversed.

---

## 6. The Convergence Rate Trajectory

The 5-10-5 convergence schedule produces this mean migration rate path:

**McKenzie (mean across all age-sex groups):**

| Year | Rate | Phase |
|------|------|-------|
| 1 | -0.0138 | Recent → Medium ramp |
| 2 | -0.0086 | Recent → Medium ramp |
| 3 | -0.0035 | Recent → Medium ramp |
| 4 | +0.0017 | Recent → Medium ramp |
| 5 | +0.0068 | Arriving at medium |
| 6-15 | +0.0068 | **Medium hold (10 years)** |
| 20 | +0.0032 | Long-term |
| 30 | +0.0032 | Long-term |

**Williams (mean across all age-sex groups):**

| Year | Rate | Phase |
|------|------|-------|
| 1 | -0.0190 | Recent → Medium ramp |
| 5 | -0.0002 | Arriving at medium |
| 6-15 | -0.0002 | **Medium hold (10 years)** |
| 20-30 | -0.0002 | Long-term |

Williams' trajectory is clean: negative migration that converges to zero and stays there. Growth is purely natural increase.

McKenzie's trajectory is the source of concern: migration turns positive by year 4 and stays positive at +0.68%/yr for 11 years.

---

## 7. The McKenzie 25-29 Age Group: The Hidden Amplifier

The overall McKenzie mean rate of +0.68%/yr at medium hold obscures extreme age-specific variation:

| Age Group | Year 1 Rate | Medium Hold Rate | Long-term Rate |
|-----------|-------------|-----------------|----------------|
| 15-19 | -0.0070 | +0.0110 | -0.0045 |
| 20-24 | -0.0169 | +0.0151 | -0.0144 |
| **25-29** | **+0.0370** | **+0.0578** | **+0.0521** |
| 30-34 | -0.0145 | +0.0248 | +0.0380 |
| 35-39 | -0.0310 | +0.0156 | +0.0192 |

The 25-29 age group has a convergence rate of **+5.78%/yr held for 10 years**, and even the long-term rate is +5.21%/yr. This means the model projects McKenzie will continue attracting ~50 young adults per year for the next 30 years.

### Is This Plausible?

**Evidence that it might be plausible:**
- Even in the most recent period (2020-2025, post-boom), McKenzie 25-29 shows +1.9% (male) and +4.5% (female) migration rates — still positive
- McKenzie's oil infrastructure is permanent: roads, housing, processing facilities are built
- The county needs replacement workers as the boom-era workforce ages out
- The rates in absolute terms are small: ~50 people per year into a 15,000-person county

**Evidence that it might be too high:**
- The 2020-2025 positive rate may reflect lag from 2019 housing completions, not ongoing attraction
- Oil industry is increasingly automated; fewer workers needed per well
- The compounding through fertility amplifies even small positive rates into large long-term growth
- The 5.78% medium rate is an average of the dampened 2010-2015 rate (+6-8%), the dampened 2015-2020 rate (+6-7%), and the post-boom 2020-2025 rate (+2-4%) — the boom periods still dominate

### The Small-Denominator Problem

McKenzie's 25-29 population in 2020 was only **928 people** (497 male, 431 female). The rates that seem extreme in percentage terms correspond to modest absolute flows:

| Period | 25-29 Net Migration | Starting Pop | Rate |
|--------|--------------------|-------------|------|
| 2010-2015 | +191 | 362 | ~+6-8%/yr |
| 2015-2020 | +347 | 776 | ~+6-7%/yr |
| 2020-2025 | +121 | 928 | ~+2-4%/yr |

Adding 25-50 young adults per year to a county of 15,000 produces a 5-6% age-specific migration rate but represents a trivially small absolute flow. The problem is that these ~50 people/year enter peak fertility (ASFR ~0.129 for ages 25-29) and produce ~6-7 additional births per year, compounding over 30 years.

---

## 8. Summary of Findings by County

| County | 30yr Growth | Primary Driver | Migration Role | Assessment |
|--------|-------------|----------------|---------------|------------|
| **Williams** | +56.6% | Natural increase (110%) | Negative drag | Growth is plausible — purely demographic momentum from a legitimately young population. No migration-related adjustment needed. |
| **McKenzie** | +82.0% | Natural increase (82%) + migration (18%) | Modest positive (+0.68%/yr at medium hold) | Growth is high but mostly demographic momentum. The 25-29 migration rate (+5.78%/yr for 10 years) warrants scrutiny — it assumes continued attraction of ~50 young adults/year. |
| **Stark** | +23.7% | Mixed | Near-zero medium rate (-0.003/yr) | Reasonable. Growth is moderate and migration-neutral. |
| **Mountrail** | +1.9% | Natural increase (offset by out-migration) | Negative (-0.011/yr medium) | Near-stagnation is plausible for a peripheral oil county. |
| **Dunn** | -7.2% | Natural increase insufficient to offset out-migration | Negative (-0.009/yr medium) | Decline is plausible — small, aging peripheral county. |
| **Billings** | ~+54% | Natural increase (~62%) + migration (~38%) | Positive (+0.96%/yr medium hold), driven by 2020-24 spike | Small-denominator amplification: a single-year spike of +54 people (2022) anchors the projection. See Section 9a for full analysis. |

---

## 9. Billings County: The Small-Denominator Case Study

### 9a. Why Billings Deserves Special Attention

Billings County (FIPS 38007, pop. 1,071) projects ~+54% growth — the third-highest percentage among Bakken counties — but from the smallest base by far. This makes it the purest example of the **small-denominator amplification problem** in the projection system.

### 9b. The Anomalous 2020-2024 Period

Billings is **the only Bakken county with positive 2020-2024 migration**:

| County | 2020-2024 Net Migration | Rate |
|--------|------------------------|------|
| Mountrail | -891 | -9.2% |
| Williams | -3,051 | -7.5% |
| Dunn | -276 | -6.9% |
| McKenzie | -900 | -6.2% |
| Stark | -1,656 | -5.0% |
| **Billings** | **+73** | **+7.8%** |

This +73-person inflow is dominated by a single-year spike: **+54 people in 2022** (~5.5% of the county's population in one year). This likely represents a handful of oil-field families, a housing development, or Census estimation noise — but the model treats it as a structural trend.

### 9c. Dampening Mismatch

The current dampening configuration creates a paradox for Billings:

- **Boom periods (2005-2020)**: Dampened at 0.40-0.50x — but Billings had *weak* boom-era rates (2010-2015: +0.012, 2015-2020: -0.006). The dampening barely matters.
- **2020-2024**: Not classified as a boom period — so the anomalous +0.024 rate receives **zero dampening** and anchors the recent convergence window.

The periods that get dampened aren't the ones driving Billings' projection. The period driving the projection doesn't get dampened.

### 9d. Small-Denominator Amplification

With only 1,071 people, the average 5-year age-sex cell contains **~30 people**:

| Cell | Population | Medium Hold Rate | Implied Annual Migrants | Comment |
|------|-----------|-----------------|------------------------|---------|
| 30-34 Female | 23 | +0.078 | 1.8/yr | One or two families |
| 40-44 Male | 35 | +0.080 (capped) | 2.8/yr | Uncapped rate was +0.200 |
| 20-24 Female | 39 | +0.058 | 2.3/yr | Two or three individuals |
| 20-24 Male | 51 | +0.044 | 2.2/yr | A couple of workers |

The rate cap (0.08) provides meaningful protection — without it, the 40-44 Male cell would project at +20%/yr. But rates of 5-8% sustained for 15 years on cells of 20-50 people are statistically unreliable; they represent the arrival (or non-arrival) of **individual families**.

### 9e. Growth Decomposition

| Component | Estimated Contribution | Share |
|-----------|----------------------|-------|
| Net migration (from convergence) | ~218 people | ~38% |
| Natural increase + fertility feedback | ~360 people | ~62% |

Unlike McKenzie and Williams, Billings' migration contribution is substantial (~38%) rather than marginal. This is because the undampened 2020-2024 spike sustains positive migration rates through the medium hold period (years 6-15 at +0.96%/yr), adding ~11 people/year who then enter peak fertility ages.

### 9f. Historical Context Contradicts the Projection

| Metric | Historical (2000-2025) | Projected (2025-2055) |
|--------|----------------------|----------------------|
| Growth rate | +21.6% over 25yr (+0.78%/yr) | +54% over 30yr (+1.45%/yr) |
| Avg annual growth | ~8 people/yr | ~19 people/yr |
| Net migration (total) | -7 people over 25yr | ~+218 over 30yr |

The projection nearly doubles the historical growth rate and converts a county with essentially **zero cumulative net migration** over 25 years into one with sustained positive in-migration. The entire basis for this reversal is a single year (2022).

### 9g. Plausibility Assessment

**+54% growth (1,071 → ~1,649) is likely too high**, but the absolute error is small:

- In absolute terms, the "excess" growth beyond historical trends is roughly **+300 people over 30 years** — about 10 people per year
- This has negligible impact on the state total (799,358) — less than 0.04%
- No SDC reference exists for Billings to calibrate against

**The projection illustrates a fundamental limitation**: cohort-component models treat age-sex-specific migration rates as stable structural parameters, but for a county of 1,071 people, these "rates" are dominated by individual-level decisions (one family moving in or out). Statistical reliability requires a minimum population scale that Billings does not meet.

### 9h. Implications for Publication

Billings should be flagged in accompanying narrative as a county where:
- The projection is subject to extreme small-sample uncertainty
- A single year of anomalous migration (2022: +54 people) materially affects the 30-year outlook
- The absolute numbers are small enough that the percentage growth rate is misleading — +54% sounds dramatic but represents ~19 additional people per year
- Users should treat the percentage figure with appropriate caution for a county of this size

---

## 10. The Key Question: Is McKenzie's +82% Too High?

### Arguments that it's reasonable:
1. **82% of the growth is natural increase**, not assumed migration — the boom left a young population that will produce children regardless of what happens to oil
2. **The 20-year projection (+48.7%) matches SDC (+47.1%)** to within 1.6pp — the divergence only appears at 30 years where SDC has no reference point
3. **Even the undampened 2020-2025 period shows positive 25-29 migration** — suggesting some ongoing structural attraction beyond the boom
4. **The absolute migration volumes are small** — ~110 people/year net on a 16,000-person county

### Arguments that it's too high:
1. **The economic base may not support +82% growth** — future jobs in oil are uncertain, and automation reduces labor needs
2. **The 25-29 medium-hold rate of +5.78%/yr for 10 years** may overstate structural attraction by including boom-era residue
3. **Compounding through fertility amplifies even small biases** — a modest overestimate of migration produces outsized population growth through births
4. **At 30 years, projection uncertainty dominates** — the model is extrapolating far beyond observable data

### The Fundamental Tension

The dampening was designed to reduce boom-era migration rates, and it has largely worked (Williams is at zero migration). But the boom's lasting effect isn't in the migration rates — it's in the **age structure**. The young people who moved during the boom are now producing children, and no amount of migration dampening can change that without also suppressing the (real) natural increase.

Making an arbitrary adjustment to McKenzie's projection would require either:
- **Capping total growth** — which would mean overriding the natural increase calculation, a methodological departure
- **Further suppressing migration** — which would push Williams into unrealistic decline and has already been tested (EXP-E, ADR-051)
- **Adjusting fertility** — which would require McKenzie-specific fertility assumptions, adding complexity and subjectivity

---

## 11. Comparison with Other High-Growth Small Counties

For context, McKenzie's growth rate is not unique among small counties with structural economic advantages:

| County | 2025 Pop | 30yr % | Primary Driver |
|--------|----------|--------|---------------|
| McKenzie | 15,192 | +82.0% | Bakken infrastructure + young age structure |
| Billings | 1,255 | ~+54% | Small-denominator effect |
| Morton (Mandan) | 34,234 | +15.4% | Bismarck metro spillover |
| Cass (Fargo) | 190,710 | +15.8% | University + regional hub |
| Burleigh (Bismarck) | 104,268 | +19.7% | State capital + services |

McKenzie's percentage growth is high but it's growing from a very small base. In absolute terms, +12,455 people over 30 years is about 415 people per year — comparable to what a single new housing subdivision might absorb.

---

## 12. Recommendations

1. **No methodological change is currently justified** for Williams, Stark, Mountrail, or Dunn — their projections are well-calibrated.

2. **McKenzie warrants monitoring but not arbitrary adjustment.** The 20-year projection aligns with SDC, and the growth is primarily natural increase. Making an ad hoc reduction without an evidence-based mechanism would undermine the projection's methodological consistency.

3. **Billings' +54% projection is likely overstated** but the absolute error (~300 people over 30 years) is too small to materially affect the state total. The root cause — a single-year migration spike in 2022 anchoring the convergence schedule — illustrates a structural limitation of cohort-component models at very small population scales. Possible future mitigations include small-county rate smoothing or a minimum-population threshold for convergence rate reliability, but these are not urgent given the negligible state-level impact.

4. **When publishing**: Accompany Bakken county projections with narrative context explaining that:
   - The boom attracted a young workforce that has permanently changed these counties' age structures
   - Projected growth is primarily driven by natural increase (births exceeding deaths), not assumed continued in-migration
   - Williams' projection includes zero net migration — all growth is from the existing population
   - McKenzie assumes modest continued attraction of ~50 young adults per year, consistent with recent (post-boom) data but subject to economic uncertainty
   - Billings' high percentage growth reflects small-denominator amplification — +54% represents only ~19 additional people per year, and users should interpret the figure with appropriate caution
   - 30-year projections carry inherent uncertainty, especially for small counties dependent on a single industry

5. **Future work**: If McKenzie's or Billings' actual 2025-2030 growth departs significantly from projected trajectories, that would provide the evidence basis for adjustment. The walk-forward validation framework is designed for exactly this kind of recalibration.

---

## Appendix A: Convergence Rate Trajectories by Age Group

### McKenzie County (38053)

| Age Group | Year 1 | Year 5 (Medium) | Year 6-15 (Hold) | Year 20-30 (Long) |
|-----------|--------|------------------|-------------------|---------------------|
| 0-4 | 0.000 | 0.000 | 0.000 | 0.000 |
| 15-19 | -0.007 | +0.011 | +0.011 | -0.005 |
| 20-24 | -0.017 | +0.015 | +0.015 | -0.014 |
| **25-29** | **+0.037** | **+0.058** | **+0.058** | **+0.052** |
| 30-34 | -0.015 | +0.025 | +0.025 | +0.038 |
| 35-39 | -0.031 | +0.016 | +0.016 | +0.019 |

### Williams County (38105)

| Age Group | Year 1 | Year 5 (Medium) | Year 6-15 (Hold) | Year 20-30 (Long) |
|-----------|--------|------------------|-------------------|---------------------|
| 0-4 | 0.000 | 0.000 | 0.000 | 0.000 |
| 15-19 | -0.006 | +0.010 | +0.010 | +0.003 |
| 20-24 | +0.011 | +0.033 | +0.033 | +0.008 |
| 25-29 | +0.018 | +0.039 | +0.039 | +0.026 |
| 30-34 | -0.012 | +0.012 | +0.012 | +0.024 |
| 35-39 | -0.022 | +0.009 | +0.009 | +0.010 |

---

## Appendix B: Historical Net Migration by Period

| County | 2000-05 | 2005-10 | 2010-15 | 2015-20 | 2020-25 | 25yr Total |
|--------|---------|---------|---------|---------|---------|------------|
| McKenzie | -217 | +125 | +1,361 | +1,300 | -900 | +1,669 |
| Williams | -353 | +713 | +2,708 | +2,992 | -3,051 | +3,009 |
| Mountrail | -18 | +225 | +542 | -245 | -891 | -387 |
| Dunn | -250 | +4 | +232 | -232 | -276 | -522 |
| Stark | -641 | +309 | +1,554 | +739 | -1,656 | +305 |
| Billings | -114 | -2 | +40 | -4 | +73 | -7 |

---

## Appendix C: Dampening Configuration (Current Production)

```yaml
dampening:
  factor:
    "2005-2010": 0.50
    "2010-2015": 0.40   # Peak boom — most transient
    "2015-2020": 0.50   # Boom-adjacent
    default: 0.60
  counties: [38105, 38053, 38061, 38025, 38089, 38007]
  boom_periods: [[2005,2010], [2010,2015], [2015,2020]]

  male_dampening:
    factor: 0.80
    boom_periods: [[2005,2010], [2010,2015]]
```
