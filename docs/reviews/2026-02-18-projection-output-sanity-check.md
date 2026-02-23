# Projection Output Sanity Check — 2025-2055 Projections

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-02-18 |
| **Timestamp** | 2026-02-18T06:00:00Z |
| **Reviewer** | Claude Code (Opus 4.6), prompted by N. Haarstad |
| **Scope** | Full output review of all four scenario projections (baseline, high_growth, restricted_growth, low_growth) |
| **Data Vintage** | Census PEP Vintage 2025; CBO Jan 2025/2026 |
| **Status** | Findings documented; corrective investigation needed for 7 items |
| **Related ADRs** | [ADR-037](../governance/adrs/037-cbo-grounded-scenario-methodology.md), [ADR-039](../governance/adrs/039-international-only-migration-factor.md), [ADR-040](../governance/adrs/040-extend-boom-dampening-2015-2020.md) |
| **Related Reviews** | [Bakken Dampening Review](2026-02-17-bakken-migration-dampening-review.md), [Vintage 2025 Analysis](2026-02-17-vintage-2025-census-data-analysis.md) |
| **Config Version** | `config/projection_config.yaml` as of commit `27904a4` |

---

## 1. Executive Summary

A comprehensive sanity check was performed on the latest projection outputs in `data/projections/`. The projections are **structurally sound**: no negative populations, no missing data, all 53 counties present across all scenarios, and the cohort-component engine logic verifies correctly. The CBO restricted-growth implementation (ADR-039) is mathematically correct.

However, **seven items warrant further investigation**, ranging from a data-loading inconsistency (Finding 1) to demographic patterns that may reflect methodological artifacts rather than plausible futures (Findings 2-6) and a counterintuitive scenario ordering (Finding 7).

**Overall assessment**: Production-quality data structure; several output magnitudes need review before publication.

---

## 2. State-Level Scenario Results

| Year | Baseline | High Growth | Low Growth | Restricted Growth |
|-----:|--------:|----------:|----------:|-----------------:|
| 2025 | 799,358 | 799,358 | **796,568** | 799,358 |
| 2030 | 823,967 | 815,534 | 795,719 | 801,067 |
| 2035 | 867,915 | 856,864 | 807,174 | 821,003 |
| 2040 | 914,480 | 904,698 | 819,451 | 846,998 |
| 2045 | 955,679 | 952,386 | 827,812 | 870,525 |
| 2055 | 1,005,281 | — | — | — |

30-year (baseline) and 20-year (other scenarios) growth rates:

| Scenario | Period | Absolute Growth | Percent Growth | Avg Annual |
|----------|--------|---------------:|---------------:|-----------:|
| Baseline | 2025-2055 | +205,923 | +25.8% | +0.77% |
| High Growth | 2025-2045 | +153,029 | +19.1% | +0.88% |
| Restricted Growth | 2025-2045 | +71,167 | +8.9% | +0.43% |
| Low Growth | 2025-2045 | +31,244 | +3.9% | +0.19% |

---

## 3. Data Quality Checks Passed

| Check | Result | Details |
|-------|--------|---------|
| Negative populations | **PASS** | Zero records with population < 0 |
| NaN/missing values | **PASS** | Zero missing values across all columns |
| Age range | **PASS** | 0 to 90 (open-ended 90+ group) |
| Sex categories | **PASS** | "Female" and "Male" only |
| Race categories | **PASS** | 6 categories, balanced record counts |
| Year coverage | **PASS** | 2025-2055 (baseline), 2025-2045 (others) |
| County count | **PASS** | 53 counties in all scenarios |
| County aggregation | **PASS** | Sum of 53 counties = state total (within rounding) |
| Single-year growth rates | **PASS** | All within ±5% (no single-year spikes) |
| Sex ratio stability | **PASS** | M:F ≈ 1.055 throughout (consistent with ND) |

---

## 4. Items Requiring Investigation

### Finding 1: Low Growth Base Year Mismatch (Data Bug)

**Severity**: High — affects scenario comparability

The low growth scenario starts at **796,568** in 2025, while all three other scenarios start at **799,358**. This 2,790-person discrepancy (0.35%) means the low growth scenario is not directly comparable to the others at any time point.

**Likely cause**: The low growth scenario may be loading from a different base population file or using a pre-Vintage 2025 data source. All scenarios should share the same 2025 base population.

**Action**: Trace the data loading path for `low_growth` in `scripts/projections/run_all_projections.py` and confirm it references the same base population parquet as the other scenarios.

**Cross-reference**: The canonical base population value of 799,358 was validated in the [Vintage 2025 analysis](2026-02-17-vintage-2025-census-data-analysis.md).

---

### Finding 2: Oil County Growth Rates Remain Aggressive Despite Dampening

**Severity**: Medium — affects credibility of county-level results

Even after implementing [ADR-040](../governance/adrs/040-extend-boom-dampening-2015-2020.md) (extending boom dampening to 2015-2020), the Bakken-area counties still show growth rates far above the state average:

| County | FIPS | 2025 | 2055 | 30-Yr Growth | Prior (Pre-ADR-040) |
|--------|------|-----:|-----:|-----------:|--------------------:|
| McKenzie | 38053 | 15,192 | 27,921 | **+83.8%** | +114.3% |
| Williams | 38105 | 41,767 | 63,195 | **+51.3%** | +73.5% |
| Billings | 38007 | 1,071 | 2,010 | **+87.7%** | N/A |
| **State** | | **799,358** | **1,005,281** | **+25.8%** | +27.7% |

ADR-040 reduced McKenzie from +114% to +84% and Williams from +74% to +51%, which is meaningful improvement. However, +84% still implies sustained ~2%/yr growth for 30 years. As documented in the [Bakken Dampening Review](2026-02-17-bakken-migration-dampening-review.md), the most recent period (2020-2024) shows **negative** net migration for both counties (McKenzie: −893, Williams: −3,033), making sustained high growth questionable.

Billings County (+88%, from 1,071 to 2,010) was not assessed in the prior dampening review. At a population of ~1,000, small absolute migration flows produce outsized percentage growth. Verify whether Billings should be added to the dampened county list, or whether its rates are an artifact of small-number volatility.

**Action**: Review post-dampening migration rates for these counties. Consider whether the dampening factor (0.60) is sufficient, or whether additional mechanisms (rate caps, extended dampening to the convergence step) are needed per the options in the [Bakken Dampening Review](2026-02-17-bakken-migration-dampening-review.md).

---

### Finding 3: AIAN Reservation County Declines Are Very Steep

**Severity**: Medium — affects credibility and political sensitivity

Three reservation-area counties lose nearly half their population over 30 years:

| County | FIPS | 2025 | 2055 | 30-Yr Change | Reservation |
|--------|------|-----:|-----:|-----------:|-------------|
| Benson | 38005 | 5,759 | 3,031 | **−47.4%** | Fort Berthold (partial) / Spirit Lake |
| Sioux | 38085 | 3,667 | 1,940 | **−47.1%** | Standing Rock (ND portion) |
| Rolette | 38079 | 11,688 | 6,353 | **−45.6%** | Turtle Mountain |

A −47% decline implies sustained net out-migration of roughly −2%/yr for 30 years. While these counties have experienced historical out-migration, population decline typically decelerates as the population shrinks — fewer people remain to leave, and fixed-population institutions (tribal governments, schools, health services) provide a floor.

The convergence interpolation schedule (Census Bureau method) ramps from recent rates → medium rates → long-term rates, but if all three windows show negative migration for these counties, there is no built-in floor or asymptotic convergence toward zero.

**Action**: Examine the migration rate trajectories for these three counties across the convergence windows. Determine whether a population floor or out-migration deceleration mechanism is warranted. This is also politically sensitive — projecting near-halving of reservation populations deserves careful review before publication.

---

### Finding 4: Hispanic Population Growth of +204% Seems Aggressive

**Severity**: Low-Medium — plausible but worth validating inputs

| Group | 2025 | 2055 | Growth | Share 2025 | Share 2055 |
|-------|-----:|-----:|-------:|---------:|---------:|
| White, Non-Hispanic | 642,227 | 740,971 | +15.3% | 80.3% | 73.7% |
| AIAN, Non-Hispanic | 91,656 | 140,755 | +53.6% | 11.5% | 14.0% |
| Hispanic (any race) | 19,141 | 58,200 | **+204.1%** | 2.4% | 5.8% |
| Asian/PI, Non-Hispanic | 14,157 | 32,730 | +131.2% | 1.8% | 3.3% |
| Two or More Races | 27,150 | 28,436 | +4.7% | 3.4% | 2.8% |
| Black, Non-Hispanic | 5,027 | 4,189 | −16.7% | 0.6% | 0.4% |

Hispanic population tripling from 19,141 to 58,200 requires sustained high net in-migration for this group. While ND starts from a very small Hispanic base (amplifying percentage growth), the absolute increase of ~39,000 Hispanic residents over 30 years (~1,300/year net) should be cross-checked against:

- Historical PEP components of change for the Hispanic group
- Whether fertility rate differentials by race are being applied (the config uses a single "total" fertility rate for all races — see Section 5, Note 1)
- National trends for Hispanic population growth in non-traditional destination states

**Action**: Spot-check the Hispanic migration rate inputs in the processed data. Verify that the growth is migration-driven (as expected) rather than an artifact of rate misallocation.

---

### Finding 5: Black Population Declining (−16.7%) Is Unusual

**Severity**: Low-Medium — counterintuitive but may reflect ND-specific dynamics

The Black non-Hispanic population declines from 5,027 to 4,189 (−16.7%) over 30 years. Nationally, Black populations are growing. ND's Black population is very small and concentrated in a few urban counties (Cass, Grand Forks, Burleigh), with much of the growth since 2010 driven by refugee resettlement.

Possible explanations:
- **Net out-migration in recent PEP data**: If the 2020-2024 period shows net out-migration for this group, the convergence schedule would project that forward
- **Small-number volatility**: With only ~5,000 people statewide, small absolute migration flows produce large percentage effects
- **Refugee resettlement patterns**: Refugee arrivals are episodic and may not be well-captured by trend extrapolation

**Action**: Examine the Black population migration rates in the processed data. Determine whether the decline reflects genuine recent trends or is an artifact of small-sample estimation in the PEP residual migration computation.

---

### Finding 6: Baseline Reaching 1 Million by 2055

**Severity**: Low — plausible but depends heavily on migration assumptions

The baseline scenario projects ND crossing the 1-million population mark around 2053-2054, representing +25.8% growth over 30 years (+0.77%/yr average).

For context:
- ND grew +16% during 2010-2020 (largely Bakken-driven, an unusual decade)
- ND grew +0.5% during 2000-2010
- The baseline assumes continued net in-migration at 2019-2024 levels, with 91% classified as international (`intl_share = 0.91`)

Crossing 1 million is not implausible if international migration remains elevated, but it represents the upper end of reasonable expectations. The restricted-growth scenario (870K by 2045, ~+8.9%) may be the more defensible central estimate for planning purposes.

**Action**: No immediate action needed, but consider whether framing the baseline as "trend continuation" vs. "most likely" matters for stakeholder communication. This is a presentation question, not a data quality issue.

---

### Finding 7: High Growth Scenario Below Baseline at 2045

**Severity**: Medium — counterintuitive scenario ordering undermines credibility

At 2045, the high growth scenario (952,386) is **lower** than the baseline (955,679). This is counterintuitive — users expect a scenario named "high growth" to always exceed "baseline."

| Year | Baseline | High Growth | Difference |
|-----:|--------:|----------:|----------:|
| 2025 | 799,358 | 799,358 | 0 |
| 2030 | 823,967 | 815,534 | −8,433 |
| 2035 | 867,915 | 856,864 | −11,051 |
| 2040 | 914,480 | 904,698 | −9,782 |
| 2045 | 955,679 | 952,386 | **−3,293** |

The high growth scenario is **below** baseline at every projected time point. Possible explanations:

1. **Different convergence schedules**: The baseline and high growth scenarios may use different convergence windows or period weightings, causing them to diverge in unexpected ways
2. **Migration multiplier interaction**: The +15% migration multiplier in high growth may be applied to a different base rate than baseline, or may interact with the convergence schedule differently
3. **Fertility offset insufficient**: The +5% fertility boost in high growth may not compensate if the base migration rates differ

Regardless of the underlying cause, this ordering will confuse stakeholders and undermine confidence in the scenario framework.

**Action**: This is a **priority investigation item**. Trace the high growth scenario through the pipeline to determine why it produces lower populations than baseline. Either the scenario parameters need adjustment, or the scenario naming needs revision.

---

## 5. Additional Observations (Non-Anomalous)

These items were reviewed and found to be within expected ranges:

### Note 1: Fertility Rates Not Differentiated by Race

The config specifies 6 race/ethnicity categories, but the processed fertility rates in `data/processed/fertility_rates.parquet` are aggregated to "total" (a single ASFR schedule applied to all races). This is a known simplification. Given ND's 80% white population, the impact on total projections is modest, but it means race-specific projections embed an assumption that all groups share the same age-specific fertility — which they do not nationally (e.g., Hispanic TFR is ~1.9 vs. white non-Hispanic ~1.6).

### Note 2: Age Structure Evolution Is Demographically Rational

The age pyramid shifts as expected:

| Age Group | 2025 Share | 2055 Share | Interpretation |
|-----------|--------:|--------:|----------------|
| 0-17 | 23.7% | 20.5% | Youth share declining (constant fertility, aging population) |
| 18-29 | 17.9% | 15.9% | Young adult share stable-declining |
| 30-44 | 20.3% | 22.4% | Working-age growing (migration-driven) |
| 45-64 | 20.8% | 25.6% | Prime working age expanding (cohort aging) |
| 65+ | 17.3% | 15.5% | Elderly share slightly declining (faster working-age growth) |

### Note 3: CBO Restricted Growth Implementation Is Correct

The international-only migration factor (ADR-039) was verified mathematically:

| Year | CBO Factor | intl_share | Effective Factor | Migration Retained |
|------|--------:|-----:|--------:|--------:|
| 2025 | 0.20 | 0.91 | 0.272 | 27.2% |
| 2026 | 0.37 | 0.91 | 0.426 | 42.6% |
| 2027 | 0.55 | 0.91 | 0.590 | 59.0% |
| 2028 | 0.78 | 0.91 | 0.800 | 80.0% |
| 2029 | 0.91 | 0.91 | 0.917 | 91.7% |
| 2030+ | 1.00 | 0.91 | 1.000 | 100% (no adjustment) |

Formula: `effective_factor = 1.0 − intl_share × (1.0 − cbo_factor)`

The progressive ramp-up (72.8% reduction in 2025, tapering to 0% by 2030) correctly models CBO's projected immigration enforcement trajectory. Domestic migration passes through unchanged, as specified in ADR-039.

### Note 4: Mortality Improvement Is Conservative and Appropriate

The 0.5% annual mortality improvement compounds to a 14.1% death rate reduction by year 30. This is conservative relative to historical trends (US age-adjusted mortality declined ~1%/yr over most of the 20th century, though improvement stalled 2010-2019). The cap at 100% survival prevents numerical artifacts.

---

## 6. Summary of Required Actions

| # | Finding | Severity | Action |
|---|---------|----------|--------|
| 1 | Low growth base year mismatch | High | Trace data loading path; fix to use Vintage 2025 base |
| 2 | Oil county growth still aggressive | Medium | Review post-dampening rates; consider stronger dampening |
| 3 | Reservation county declines ~47% | Medium | Examine convergence trajectories; consider population floor |
| 4 | Hispanic +204% growth | Low-Med | Spot-check migration rate inputs for this group |
| 5 | Black population declining | Low-Med | Examine migration rates; check for small-sample artifacts |
| 6 | Baseline reaching 1M by 2055 | Low | Presentation/framing question for stakeholders |
| 7 | High growth < baseline (ordering) | Medium | **Priority**: trace pipeline to find cause of inversion |

---

## 7. Files Examined

| File | Role in Review |
|------|----------------|
| `config/projection_config.yaml` | Scenario parameters, convergence windows, dampening config |
| `cohort_projections/core/cohort_component.py` | Projection engine logic verification |
| `cohort_projections/core/migration.py` | International-only factor implementation |
| `cohort_projections/core/mortality.py` | Survival rate and improvement factor logic |
| `data/projections/baseline/` | Baseline scenario output files |
| `data/projections/high_growth/` | High growth scenario output files |
| `data/projections/restricted_growth/` | Restricted growth scenario output files |
| `data/projections/low_growth/` | Low growth scenario output files |
| `data/processed/base_population.parquet` | Base population verification |
| `data/processed/fertility_rates.parquet` | Fertility rate structure check |
| `data/processed/survival_rates.parquet` | Survival rate validation |
| `scripts/projections/run_all_projections.py` | Scenario execution pipeline |
| `scripts/exports/build_detail_workbooks.py` | Export logic and output structure |
| `scripts/exports/_methodology.py` | Scenario naming and methodology constants |
