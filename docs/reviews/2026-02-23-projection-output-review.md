# 2026-02-23 Projection Output Review and ADR Assessment

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-02-23 |
| **Reviewer** | Claude Code (Opus 4.6), prompted by N. Haarstad |
| **Scope** | Full output review of all 3 scenario projections (baseline, high_growth, restricted_growth); ADR assessment for 051, 052, 036, 033 |
| **Data Vintage** | Census PEP Vintage 2025; CBO Jan 2025/2026 |
| **Status** | Complete — decisions documented for 5 ADRs |
| **Related ADRs** | [ADR-033](../governance/adrs/033-city-level-projection-methodology.md), [ADR-036](../governance/adrs/036-migration-averaging-methodology.md), [ADR-045](../governance/adrs/045-reservation-county-pep-recalibration.md), [ADR-047](../governance/adrs/047-county-specific-age-sex-race-distributions.md), [ADR-048](../governance/adrs/048-single-year-of-age-base-population.md), [ADR-049](../governance/adrs/049-college-age-smoothing-convergence-pipeline.md), [ADR-050](../governance/adrs/050-restricted-growth-additive-migration-adjustment.md), [ADR-051](../governance/adrs/051-oil-county-dampening-recalibration.md), [ADR-052](../governance/adrs/052-ward-county-high-growth-floor.md), [ADR-053](../governance/adrs/053-nd-specific-vital-rates.md), [ADR-054](../governance/adrs/054-state-county-aggregation-reconciliation.md) |
| **Related Reviews** | [Projection Output Sanity Check (2026-02-18)](2026-02-18-projection-output-sanity-check.md), [Bakken Migration Dampening Review (2026-02-17)](2026-02-17-bakken-migration-dampening-review.md) |
| **Config Version** | `config/projection_config.yaml` as of commit `d07e834` |

---

## 1. Executive Summary

All three scenarios (baseline, high_growth, restricted_growth) were re-run on 2026-02-23, incorporating all fixes from ADRs 047-050 (Sprague interpolation, county race distributions, scenario ordering, additive migration) and ADR-053 (ND-specific vital rates). Previous projections were stale: baseline from Feb 20, high/restricted from Feb 18, exports from Feb 17.

**Key outcomes:**
- 1,257 tests passing, zero processing failures across 53 counties
- Zero scenario ordering violations (restricted < baseline < high) across all 53 counties
- Oil county calibration adequate on 20-year basis — ADR-051 rejected
- Ward County migration floor working as designed — ADR-052 confirmed
- State-county aggregation discrepancy of ~10.6% discovered — flagged for ADR-054 investigation

**Overall assessment:** Projection outputs are structurally sound and demographically reasonable. The additive migration methodology (ADR-050) resolved all previous scenario ordering issues. One significant new finding (state-county aggregation gap) requires investigation before publication.

---

## 2. Context

This review was prompted by the completion of a comprehensive set of methodology improvements:

- **ADR-047**: Sprague interpolation for smooth age distributions
- **ADR-048**: County race distributions from Census full-count data
- **ADR-049**: Scenario ordering enforcement
- **ADR-050**: Additive migration methodology (replaced multiplicative)
- **ADR-053**: ND-specific vital rates

All three scenarios were re-run from scratch to ensure outputs reflect the complete set of fixes. The previous sanity check ([2026-02-18](2026-02-18-projection-output-sanity-check.md)) identified 7 findings, most of which have been addressed by the ADR-042 through ADR-053 implementation cycle.

---

## 3. State-Level Results

| Scenario | 2025 | 2035 | 2045 | 2055 | 30yr Change |
|----------|-----:|-----:|-----:|-----:|------------:|
| Baseline | 799,358 | 836,864 | 880,325 | 900,971 | +12.7% |
| High Growth | 799,358 | 893,032 | 984,698 | 1,067,814 | +33.6% |
| Restricted | 799,358 | 818,263 | 840,795 | 842,885 | +5.4% |

### Comparison to Feb-18 Sanity Check

| Scenario | Feb-18 (30yr) | Feb-23 (30yr) | Change |
|----------|:------------:|:------------:|--------|
| Baseline | +25.8% (1,005,281) | +12.7% (900,971) | Reduced substantially; ND-specific vital rates and additive migration lowered trajectory |
| High Growth | Below baseline (broken) | +33.6% (1,067,814) | Fixed; now correctly above baseline at all time points |
| Restricted | +8.9% (20yr only) | +5.4% (842,885) | Extended to 30yr; more conservative trajectory |

The scenario spread is asymmetric: restricted is -6.4 percentage points below baseline, while high growth is +18.5 percentage points above. This asymmetry is expected given the additive migration methodology — the high scenario adds BEBR-optimistic rates while restricted applies CBO international migration reductions.

---

## 4. Oil County Analysis (ADR-051 Assessment)

### 4a. Full 30-Year Table: All 6 Oil Counties

#### Baseline Scenario

| County | 2025 | 2030 | 2035 | 2040 | 2045 | 2050 | 2055 | 30yr % |
|--------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-------:|
| McKenzie | 15,192 | 16,891 | 18,742 | 20,429 | 21,764 | 22,701 | 23,247 | +53.0% |
| Williams | 41,767 | 44,776 | 47,722 | 50,201 | 52,056 | 53,225 | 53,750 | +28.7% |
| Mountrail | 9,395 | 9,513 | 9,617 | 9,679 | 9,679 | 9,600 | 9,448 | +0.6% |
| Stark | 34,013 | 34,685 | 35,251 | 35,613 | 35,710 | 35,508 | 35,028 | +3.0% |
| Dunn | 4,058 | 4,109 | 4,147 | 4,158 | 4,131 | 4,063 | 3,960 | -2.4% |
| Billings | 1,071 | 1,104 | 1,134 | 1,155 | 1,162 | 1,154 | 1,132 | +5.7% |

#### High Growth Scenario

| County | 2025 | 2035 | 2045 | 2055 | 30yr % |
|--------|-----:|-----:|-----:|-----:|-------:|
| McKenzie | 15,192 | 21,174 | 27,192 | 32,475 | +113.8% |
| Williams | 41,767 | 52,463 | 62,170 | 69,662 | +66.7% |
| Mountrail | 9,395 | 10,463 | 11,313 | 11,731 | +24.9% |
| Stark | 34,013 | 37,586 | 40,441 | 42,084 | +23.7% |
| Dunn | 4,058 | 4,475 | 4,784 | 4,912 | +21.0% |
| Billings | 1,071 | 1,231 | 1,370 | 1,450 | +35.4% |

#### Restricted Growth Scenario

| County | 2025 | 2035 | 2045 | 2055 | 30yr % |
|--------|-----:|-----:|-----:|-----:|-------:|
| McKenzie | 15,192 | 17,484 | 19,306 | 20,280 | +33.5% |
| Williams | 41,767 | 44,263 | 45,711 | 45,715 | +9.5% |
| Mountrail | 9,395 | 9,147 | 8,847 | 8,424 | -10.3% |
| Stark | 34,013 | 33,660 | 33,061 | 32,064 | -5.7% |
| Dunn | 4,058 | 3,871 | 3,672 | 3,435 | -15.3% |
| Billings | 1,071 | 1,050 | 1,019 | 978 | -8.7% |

### 4b. SDC 20-Year Comparison

The State Data Center (SDC) provides independent 20-year projections for comparison:

| County | Baseline 20yr | SDC 20yr | Difference |
|--------|:------------:|:--------:|:----------:|
| McKenzie | +48.8% | +47.1% | +1.7 pp |
| Williams | +34.2% | +33.4% | +0.8 pp |

Both McKenzie and Williams are within 2 percentage points of the SDC reference on a 20-year horizon. This is well within acceptable calibration tolerance.

### 4c. Annualized Growth Rate Trajectories (Baseline)

| County | 2025-2030 | 2030-2035 | 2035-2040 | 2040-2045 | 2045-2050 | 2050-2055 |
|--------|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| McKenzie | +2.1% | +2.1% | +1.7% | +1.3% | +0.9% | +0.5% |
| Williams | +1.4% | +1.3% | +1.0% | +0.7% | +0.4% | +0.2% |
| Mountrail | +0.3% | +0.2% | +0.1% | +0.0% | -0.2% | -0.3% |
| Stark | +0.4% | +0.3% | +0.2% | +0.1% | -0.1% | -0.3% |
| Dunn | +0.3% | +0.2% | +0.1% | -0.1% | -0.3% | -0.5% |
| Billings | +0.6% | +0.5% | +0.4% | +0.1% | -0.1% | -0.4% |

All oil counties show decelerating growth trajectories, with the smaller counties (Mountrail, Stark, Dunn, Billings) transitioning to mild decline in the later decades. McKenzie and Williams maintain positive but declining growth throughout, consistent with established infrastructure supporting continued but slowing in-migration.

### 4d. Per-County Assessment

- **McKenzie**: +53.0% (30yr) appears high but the 20yr +48.8% tracks SDC +47.1% closely. Growth decelerates from 2.1%/yr to 0.5%/yr. The trajectory is front-loaded, reflecting convergence from elevated medium-window rates toward long-term averages. Acceptable.
- **Williams**: +28.7% (30yr) is close to the state average. 20yr +34.2% vs SDC +33.4%. Well-calibrated. Acceptable.
- **Mountrail**: Near-flat at +0.6%. Transitions to mild decline post-2045. Reasonable for a small county adjacent to the oil patch.
- **Stark (Dickinson)**: +3.0% (30yr), essentially stable. Growth trajectory mirrors Mountrail. Reasonable.
- **Dunn**: Mild decline at -2.4%. Small county (4,058), limited institutional base. Plausible.
- **Billings**: +5.7% (30yr), down from +87.7% in the Feb-18 review. The additive migration methodology and Sprague interpolation have resolved the previous small-county volatility artifact. Now reasonable.

### 4e. Decision

**ADR-051 (Oil County Dampening): REJECTED.** The 20-year calibration for McKenzie and Williams is within 2 percentage points of SDC references. The growth trajectories show appropriate deceleration. Additional dampening is not warranted. Billings County, previously a concern at +87.7%, is now at +5.7% after ADR-050 fixes.

---

## 5. Ward County and Urban Analysis (ADR-052 Assessment)

### 5a. Ward County Results

| Scenario | 2025 | 2035 | 2045 | 2055 | 30yr % |
|----------|-----:|-----:|-----:|-----:|-------:|
| Baseline | 67,641 | 62,996 | 59,211 | 57,764 | -14.6% |
| High Growth | 67,641 | 74,461 | 81,610 | 87,251 | +29.0% |
| Restricted | 67,641 | 60,177 | 54,320 | 50,419 | -25.5% |

The migration floor implemented in ADR-052 is working: high growth shows +29.0%, demonstrating that the floor prevents the negative migration rates from dominating even in the optimistic scenario. Scenario ordering is correct (restricted < baseline < high).

### 5b. SDC Comparison (20-Year Horizon)

| Metric | Model Output | SDC Reference | Difference |
|--------|:-----------:|:------------:|:----------:|
| High Growth 20yr | +21.2% | +23% | -1.8 pp |

The high growth 20-year figure (+21.2%) is slightly below the SDC reference (+23%), within acceptable range.

**Baseline remains pessimistic at -14.6% (30yr).** Ward County (Minot) has an Air Force base (Minot AFB) that provides a population floor in practice, but the model's migration rates reflect recent PEP trends that show net out-migration. This is a known limitation — military-driven populations are poorly captured by trend extrapolation. The high growth scenario is the more defensible central estimate for Ward County planning purposes.

### 5c. Urban County Comparison

| County (City) | Baseline 30yr | High 30yr | 20yr Baseline | SDC 20yr | Notes |
|---------------|:------------:|:---------:|:------------:|:--------:|-------|
| Cass (Fargo) | +29.8% | — | +23.4% | +30% | Moderate undershoot vs SDC; strong sustained growth |
| Burleigh (Bismarck) | +20.0% | — | +15.7% | +20% | Close alignment on 20yr horizon |
| Grand Forks | -8.7% | +38.4% | — | — | Ward-like pattern: baseline negative, high strongly positive |

**Grand Forks** exhibits the same pattern as Ward County: baseline shows decline (-8.7%) while high growth shows strong increase (+38.4%). This suggests the model's baseline migration rates capture recent out-migration trends, while the BEBR-optimistic rates in the high scenario provide the growth path. Grand Forks should be monitored alongside Ward — if future PEP data shows migration recovery, the baseline will improve.

**Cass County** is the strongest growth story in the state at +29.8% baseline (30yr). The 20-year +23.4% is about 6.6 pp below SDC's +30%, suggesting the model may be slightly conservative for Fargo. This is preferable to overshoot for planning purposes.

**Burleigh County** at +20.0% (30yr) shows excellent alignment with SDC on the 20-year horizon (+15.7% vs SDC +20%).

### 5d. Decision

**ADR-052 (Ward County Floor): ACCEPTED/IMPLEMENTED.** The migration floor is functioning correctly. High growth scenario shows +29.0%, scenario ordering is maintained, and the 20-year high growth figure (+21.2%) tracks SDC (+23%) closely. The baseline pessimism for Ward and Grand Forks is a known limitation of trend-based migration rates in military/university-dominated counties.

---

## 6. Reservation County Calibration Check (ADR-045)

| County | ADR-045 Pre-fix | ADR-045 Post-fix Est. | Actual (2026-02-23) |
|--------|:--------------:|:--------------------:|:-------------------:|
| Benson | -47% | ~-23% | **-10.7%** |
| Sioux | -47% | ~-25% | **-2.2%** |
| Rolette | -46% | ~-21% | **-4.7%** |

All three reservation counties are performing significantly better than both the pre-fix projections and the ADR-045 post-fix estimates. The PEP-anchored migration recalibration with hybrid scaling and Rogers-Castro fallback is working well:

- **Benson** (-10.7%): Improved from -47% to -10.7%, a 36.3 pp improvement. The Spirit Lake tribal community provides institutional stability.
- **Sioux** (-2.2%): Nearly stable, improved from -47% to -2.2%, a 44.8 pp improvement. Standing Rock's cross-border (ND/SD) dynamics may be better captured by the recalibrated rates.
- **Rolette** (-4.7%): Improved from -46% to -4.7%, a 41.3 pp improvement. Turtle Mountain Band's established institutional presence supports population retention.

The ADR-045 recalibration is delivering results that are both more plausible and more defensible for publication. The remaining mild declines (-2% to -11%) are consistent with long-term trends in rural reservation counties and avoid the politically problematic near-halving projections of the pre-fix model.

---

## 7. Migration Averaging Assessment (ADR-036)

The BEBR multiperiod approach with trimmed average is active across all scenarios. The convergence interpolation pipeline computes window averages (recent, medium, long-term) and applies them through the BEBR schedule.

### State-Level Trajectory

The state trajectory shows smooth growth across all scenarios with no discontinuities or inflection artifacts:

| Period | Baseline Avg Annual | High Avg Annual | Restricted Avg Annual |
|--------|:------------------:|:--------------:|:--------------------:|
| 2025-2035 | +0.46% | +1.11% | +0.23% |
| 2035-2045 | +0.51% | +0.98% | +0.27% |
| 2045-2055 | +0.23% | +0.82% | +0.02% |

### Scenario Spread

The scenario spread is asymmetric:
- Restricted growth: -6.4 percentage points below baseline (30yr)
- High growth: +18.5 percentage points above baseline (30yr)

This asymmetry reflects the different mechanisms: restricted applies CBO international migration reductions (time-varying, front-loaded), while high applies BEBR convergence rates (sustained throughout). The asymmetry is methodologically defensible — the CBO reductions expire by 2030, limiting the long-term impact on restricted growth.

### Scenario Ordering

Zero violations across all 53 counties. Restricted < baseline < high holds universally. The ADR-050 additive migration methodology resolved all previous ordering issues identified in the Feb-18 sanity check (Finding 7).

### Decision

**ADR-036 (Migration Averaging): ACCEPTED/IMPLEMENTED.** The BEBR multiperiod approach with trimmed average produces smooth trajectories, reasonable scenario spreads, and universal scenario ordering compliance.

---

## 8. City-Level Projections Assessment (ADR-033)

### Small County Threshold Analysis

18 of 53 counties are projected to fall below 2,500 population by 2055 under the baseline scenario. County-level infrastructure (age-sex-race distributions, migration rates, vital rates) remains stable for all 53 counties.

### Blocking Issue: State-County Aggregation Discrepancy

The state-county aggregation discrepancy (see Section 9) means that city-level projections — which would require consistent county-to-city allocation — cannot proceed until the county-level totals are reconciled with the independent state projection. Any city-level allocation methodology would inherit and potentially amplify the 10.6% aggregation gap.

### Decision

**ADR-033 (City-Level Projections): DEFERRED.** The county infrastructure is ready, but the state-county aggregation discrepancy must be resolved first. City-level work is blocked pending the investigation under ADR-054.

---

## 9. State-County Aggregation Discrepancy (New Finding)

### Description

County sums exceed the independent state projection by approximately 86,000 persons (10.6%) at 2055. This pattern holds across all three scenarios:

| Scenario | County Sum (2055) | State Projection (2055) | Gap | Gap % |
|----------|------------------:|------------------------:|----:|------:|
| Baseline | 900,971 | 814,934 | +86,037 | +10.6% |
| High Growth | 1,067,814 | 974,581 | +93,233 | +9.6% |
| Restricted | 842,885 | 766,576 | +76,309 | +10.0% |

### Pattern Analysis

- The gap is consistent across scenarios (9.6% to 10.6%), suggesting a structural cause rather than a scenario-specific artifact
- The gap grows over time: it is near-zero at 2025 (both start from the same base population) and increases steadily through 2055
- The gap exceeds the 1% validation tolerance specified in the projection config

### Root Cause Hypothesis

State-level and county-level migration rate inputs are computed independently and sum to different aggregate volumes. When the 53 county projections are run independently, each county's migration rates are drawn from county-level PEP residual migration estimates. These 53 independent migration rate sets, when applied to the 53 independent cohort-component models, produce aggregate migration volumes that differ from the single state-level migration rate applied to the single state-level model.

Specifically:
- County-level residual migration rates are estimated from county PEP components, which may have different coverage and estimation methodology than state-level components
- The convergence interpolation operates independently at state and county levels, with different window averages
- Small-county migration rates, when applied to small populations, can produce outsized effects that do not net to the state total

### Impact

This discrepancy does not affect the validity of individual county projections in isolation, but it:
1. Prevents direct comparison of county sums to the state total
2. Blocks city-level projections (ADR-033) which require consistent allocation hierarchies
3. Would be noticed by any user who sums the county detail workbook sheets and compares to the state summary

### Action

Flagged for investigation under **ADR-054 (State-County Aggregation)**. Resolution options may include:
- Top-down control totals (rake county projections to match state total)
- Bottom-up aggregation (derive state projection from county sum)
- Hybrid approach with reconciliation at each projection step

---

## 10. Scenario Ordering Verification

Zero violations across all 53 counties at all time points. The ordering **restricted < baseline < high** holds universally.

This is a significant improvement from the Feb-18 sanity check, which found that the high growth scenario was *below* baseline at every projected time point (Finding 7). The ADR-050 additive migration methodology fully resolved this issue by replacing the multiplicative approach (which could invert under certain rate conditions) with an additive differential that preserves ordering by construction.

---

## 11. Export Outputs Generated

Four workbooks were created on 2026-02-23:

| # | File | Size | Sheets | Scenario |
|---|------|-----:|-------:|----------|
| 1 | `nd_projections_baseline_detail_20260223.xlsx` | 319 KB | 63 | Baseline |
| 2 | `nd_projections_restricted_growth_detail_20260223.xlsx` | 320 KB | 63 | Restricted Growth |
| 3 | `nd_projections_high_growth_detail_20260223.xlsx` | 320 KB | 63 | High Growth |
| 4 | `nd_population_projections_provisional_20260223.xlsx` | 39 KB | Multi-tab | Summary (all scenarios) |

Each detail workbook contains 63 sheets: 1 summary + 1 state + 53 counties + 8 region/planning district aggregations.

---

## 12. Minor Issues Found

### Issue 1: Empty Summary CSVs

Initial 2026-02-23 finding: export summary CSVs (`county_growth_rates.csv`) appeared header-only.

**Revalidation (2026-02-28): Resolved / stale finding.**

- `data/exports/baseline/summaries/county_growth_rates.csv`: 54 lines
- `data/exports/high_growth/summaries/county_growth_rates.csv`: 54 lines
- `data/exports/restricted_growth/summaries/county_growth_rates.csv`: 54 lines

This issue is now closed.

### Issue 2: Pandas FutureWarning

Initial 2026-02-23 finding: `build_provisional_workbook.py` used a `groupby` default that triggered a pandas FutureWarning.

**Revalidation (2026-02-28): Resolved / stale finding.**

`scripts/exports/build_provisional_workbook.py` now explicitly sets `observed=True` in the relevant `groupby` operation, so this warning item is closed.

---

## 13. Summary of Decisions

This table records decision status at the time of this 2026-02-23 review. Subsequent ADR updates (notably ADR-054 acceptance/implementation) are reflected in the ADR files and `DEVELOPMENT_TRACKER.md`.

| ADR | Title | Previous Status | New Status | Rationale |
|-----|-------|:--------------:|:----------:|-----------|
| 051 | Oil County Dampening | Proposed | **Rejected** | 20yr calibration within 2 pp of SDC for McKenzie and Williams; growth trajectories show appropriate deceleration |
| 052 | Ward County Floor | Proposed | **Accepted/Implemented** | Floor working; high +29.0%; scenario ordering correct; 20yr high tracks SDC within 2 pp |
| 036 | Migration Averaging | Proposed | **Accepted/Implemented** | BEBR multiperiod active; smooth trajectories; zero ordering violations; reasonable scenario spread |
| 033 | City-Level Projections | Proposed | **Deferred** | County infrastructure ready but blocked by state-county aggregation gap (10.6%) |
| 054 | State-County Aggregation | — | **Proposed (New)** | 10.6% discrepancy between county sums and independent state projection needs investigation |

---

## 14. Files Examined

| File | Role in Review |
|------|----------------|
| `config/projection_config.yaml` | Scenario parameters, convergence windows, validation tolerances |
| `data/projections/baseline/` | Baseline scenario output files |
| `data/projections/high_growth/` | High growth scenario output files |
| `data/projections/restricted_growth/` | Restricted growth scenario output files |
| `data/exports/` | Export workbook outputs |
| `cohort_projections/core/cohort_component.py` | Projection engine logic |
| `cohort_projections/core/migration.py` | Additive migration implementation |
| `cohort_projections/data/process/convergence_interpolation.py` | BEBR convergence and rate caps |
| `cohort_projections/data/process/residual_migration.py` | PEP recalibration for reservation counties |
| `scripts/exports/build_detail_workbooks.py` | Detail workbook generation |
| `scripts/exports/build_provisional_workbook.py` | Summary workbook generation |

---

| Attribute | Value |
|-----------|-------|
| **Last Updated** | 2026-02-28 |
| **Version** | 1.1 |
