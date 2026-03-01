# 2026-02-28 Backtesting Design and Acceptance Metrics (PP3-S05)

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-02-28 |
| **Reviewer** | Claude (AI Agent) -- requires human review before acceptance |
| **Scope** | PP3-S05 backtest window design, error metrics, and acceptance thresholds for Phase 1 city/place share-trending projections |
| **Status** | Revised -- human decisions incorporated 2026-02-28; final approval at PP3-S07 gate |
| **Related ADR** | ADR-033 |

---

## 1. Purpose

Define the backtesting protocol and quantitative acceptance criteria that Phase 1 place-level projections must satisfy before publication. The design must:

- Validate the share-trending method across all three projected confidence tiers (HIGH, MODERATE, LOWER).
- Produce empirical prediction-interval estimates grounded in actual ND place volatility.
- Provide clear pass/fail thresholds so the PP3-S07 approval gate has an unambiguous evidence base.

## 2. Train/Test Window Design

### 2.1 Primary Window (Required)

| Partition | Years | Duration | Rationale |
|-----------|-------|----------|-----------|
| **Training** | 2000-2014 | 15 years | Captures pre-oil-boom baseline, boom onset (~2010-2014), and gives 15 annual share observations per place -- sufficient for linear and simple nonlinear trend fits. |
| **Test** | 2015-2024 | 10 years | Covers boom plateau, bust/recovery cycle, and COVID-era volatility. Aligns with the ADR-033 Decision 4 specification. |

The primary window is treated as the binding acceptance test. All pass/fail thresholds (Section 4) are evaluated on this window.

### 2.2 Secondary Window (Recommended)

| Partition | Years | Duration | Rationale |
|-----------|-------|----------|-----------|
| **Training** | 2000-2019 | 20 years | Longer history absorbs more structural variation. |
| **Test** | 2020-2024 | 5 years | Tests shorter-horizon accuracy and stability through the COVID period. |

The secondary window is diagnostic only -- it is used to check whether error distributions shift materially with longer training history. If the secondary window shows substantially better accuracy, that is evidence that Phase 1 production runs should train on the full 2000-2024 history rather than a subset.

### 2.3 Expanding/Rolling Windows

Additional rolling-origin experiments (e.g., train on 2000-t, test on t+1 through 2024 for t in 2009..2019) are deferred. For Phase 1 with 355 places and a simple share-trending model, two fixed windows provide adequate diagnostic power without over-engineering the validation apparatus. Rolling-origin evaluation can be added in Phase 2 if the model specification becomes more complex (e.g., regime-switching or structural-break detection).

### 2.4 EXCLUDED Tier Informational Backtesting

Places below 500 population (EXCLUDED tier) will receive informational backtesting in Phase 1. These places are not evaluated against pass/fail thresholds and their results are reported in a separate section from the acceptance evaluation. The purpose is twofold: (1) provide additional signal for threshold calibration by expanding the empirical error distribution, and (2) document the error characteristics of very small places for future reference if tier boundaries are revisited.

## 3. Error Metrics

### 3.1 Per-Place Metrics

For each projected place p and each test year t, compute:

```
APE(p, t) = |projected_pop(p, t) - actual_pop(p, t)| / actual_pop(p, t) * 100

PE(p, t) = (projected_pop(p, t) - actual_pop(p, t)) / actual_pop(p, t) * 100
```

Aggregate across test years for each place:

| Metric | Definition | Purpose |
|--------|-----------|---------|
| **MAPE(p)** | Mean of APE(p, t) across all test years | Average accuracy per place |
| **MedAPE(p)** | Median of APE(p, t) across all test years | Robust accuracy (dampens outlier years) |
| **ME(p)** | Mean of PE(p, t) across all test years | Bias direction (positive = over-projection) |
| **MaxAPE(p)** | Max of APE(p, t) across all test years | Worst-case single-year error |
| **AE_terminal(p)** | Absolute error at final test year (2024) | End-of-horizon accuracy |

### 3.2 Tier-Level Aggregates

For each confidence tier, aggregate the per-place metrics:

| Aggregate | Computation |
|-----------|-------------|
| **Tier MedAPE** | Median of MAPE(p) across all places in the tier |
| **Tier Mean ME** | Mean of ME(p) across all places in the tier |
| **Tier 90th-percentile MAPE** | 90th percentile of MAPE(p) within the tier |
| **Tier MaxAPE** | Maximum MAPE(p) within the tier |

### 3.3 Primary Metric

**MedAPE (Median Absolute Percentage Error)** is the primary acceptance metric at the tier level.

Rationale:
- MAPE is sensitive to extreme outliers in small places where a single-person difference can produce a large percentage error.
- MedAPE gives a robust central-tendency measure of accuracy without requiring outlier trimming.
- Standard practice in subcounty projection evaluation (Wilson 2015, Rayer & Smith 2010).

ME (mean error / bias) is evaluated as a secondary requirement to ensure projections are not systematically biased in one direction.

## 4. Acceptance Thresholds by Tier (DRAFT / PROPOSED)

The thresholds below are **proposals** based on published subcounty projection error benchmarks and the known volatility characteristics of ND places. They require human review and approval at the PP3-S07 gate.

**Benchmark context:** Rayer & Smith (2010) report 10-year MAPEs for subcounty areas in the range of 8-15% for places >10,000 and 15-30% for places 1,000-10,000. Wilson (2015) finds MAPEs of 7-12% at 10-year horizons for areas >10,000 using simple extrapolation methods. ND places are generally smaller and more volatile than national averages, so thresholds are set at the permissive end of published ranges.

| Tier | Population | Tier MedAPE Threshold | 90th-Pctl MAPE Ceiling | Bias (|Mean ME|) Ceiling | Rationale |
|------|------------|----------------------|------------------------|--------------------------|-----------|
| **HIGH** | >10,000 | <=10% | <=20% | <=5% | Largest places have most stable shares; tightest standard. |
| **MODERATE** | 2,500-10,000 | <=15% | <=30% | <=8% | Moderate volatility; relaxed proportionally. |
| **LOWER** | 500-2,500 | <=25% | <=45% | <=12% | High relative volatility from small denominators; generous ceiling. |

**Pass rule:** A tier passes if all three column thresholds are met simultaneously. All three projected tiers must pass for the model to proceed to production.

**Failure protocol:**
1. If one tier fails, investigate per-place outliers -- a small number of structurally unusual places (e.g., oil-boom towns, reservation communities) may be individually flagged and excluded from tier statistics. Each exclusion must be individually justified with documented structural-break rationale (identified break year and cause); no numeric cap on the number of exclusions.
2. If systematic failure occurs (majority of places in a tier exceed thresholds), revisit model specification in PP3-S04 before re-running backtests.
3. Any post-exclusion re-evaluation must be documented and the exclusion list carried forward into production output caveats.

## 5. Reporting Format

### 5.1 Summary Table (One Per Backtest Window)

```
+----------+--------+-----------+----------+---------------+---------+---------+
| Tier     | Places | Tier      | Tier     | 90th-Pctl     | Mean ME | Pass/   |
|          | (n)    | MedAPE    | Mean ME  | MAPE          |         | Fail    |
+----------+--------+-----------+----------+---------------+---------+---------+
| HIGH     |      9 |     X.X%  |   +X.X%  |         X.X%  |  +X.X%  | PASS    |
| MODERATE |      9 |     X.X%  |   +X.X%  |         X.X%  |  +X.X%  | PASS    |
| LOWER    |     72 |     X.X%  |   +X.X%  |         X.X%  |  +X.X%  | PASS    |
+----------+--------+-----------+----------+---------------+---------+---------+
```

### 5.2 Per-Place Detail Table

One row per projected place, sorted by tier then descending MAPE:

```
place_fips | place_name | county_fips | tier | MAPE | MedAPE | ME | MaxAPE | AE_terminal | flag
```

The `flag` column marks places that exceed their tier's 90th-percentile MAPE ceiling (candidate outliers for review).

### 5.3 Outlier Narrative

For any place flagged in the detail table, provide a brief narrative identifying the likely cause (e.g., oil-boom population surge, annexation, institutional population change). This narrative is required for PP3-S07 approval evidence.

### 5.4 Prediction Interval Calibration Table

Using the empirical error distributions from the backtest, report the 80% and 90% prediction interval half-widths by tier:

```
+----------+---------------------+---------------------+
| Tier     | 80% PI Half-Width   | 90% PI Half-Width   |
+----------+---------------------+---------------------+
| HIGH     |               X.X%  |               X.X%  |
| MODERATE |               X.X%  |               X.X%  |
| LOWER    |               X.X%  |               X.X%  |
+----------+---------------------+---------------------+
```

These empirical intervals replace the assumed bands in ADR-033 Decision 2 (+-10%, +-15%, +-25%) if the backtest data supports tighter or wider intervals.

## 6. Edge Cases

### 6.1 Dissolved Places

Bantry city (04740, 2019 pop: 7) and Churchs Ferry city (14140, 2019 pop: 9) are present in the 2000-2019 training data but absent from the 2020-2024 test file.

**Rule:** Include in training-set share calculations for their county (to maintain correct county-share sums in historical years), but exclude from the backtest evaluation universe. They are already flagged `historical_only` per S02/S03 rules and will not appear in production projections.

### 6.2 Places Near Tier Boundaries

A place whose 2024 population is close to a tier threshold (e.g., 9,800 vs. 10,000) could be assigned to a different tier under slight population revision.

**Rule:** Tier assignment uses the 2024 PEP population estimate as published. If a place falls within 5% of a tier boundary (i.e., population in [9,500-10,000] or [2,375-2,500] or [475-500]), flag it in the per-place detail table as `tier_boundary`. No special threshold treatment -- the place is evaluated against the tier it falls into, but the flag alerts reviewers to potential sensitivity.

### 6.3 Structural Breaks in Share Trends

Some places experienced abrupt share changes driven by identifiable external events (e.g., Williston during the 2010-2014 Bakken oil boom, Watford City oil-driven growth, institutional openings/closures).

**Rule:**
1. Run the backtest with the baseline model specification from PP3-S04 first. Do not pre-emptively adjust for structural breaks.
2. If a place fails its tier threshold and the outlier narrative identifies a clear structural break, it may be excluded from tier aggregate statistics. The exclusion must be documented with the break year and cause.
3. Each exclusion requires individually documented justification with identified break year and cause. No numeric cap on exclusions. However, if the cumulative number of exclusions becomes large relative to tier size, reviewers should consider whether the pattern indicates a model specification issue rather than place-level anomalies.

### 6.4 Multi-County Places

Places assigned via `multi_county_primary` in the S03 crosswalk (place population attributed entirely to a single primary county) may show share instability if the place's actual growth is occurring in the secondary county.

**Rule:** Flag all `multi_county_primary` places in the per-place detail table. If their error rates are systematically higher than `single_county` places in the same tier, note this in the outlier narrative for potential Phase 2 refinement (population splitting across counties).

## 7. Dependencies

| Dependency | Source Step | Required Artifact | Status |
|------------|------------|-------------------|--------|
| Place-to-county crosswalk | PP3-S03 | `data/processed/geographic/place_county_crosswalk_2020.csv` | Defined (rules in S03 note), artifact not yet built |
| Model specification | PP3-S04 | Phase 1 model spec note defining trend-fitting approach and share constraints | Pending |
| Historical place population series | PP3-S02 | Assembled long-format place population file (2000-2024) | Data sources verified, assembly script not yet built |
| County population actuals | Existing | `data/processed/pep_county_components_2000_2025.parquet` or equivalent county totals file | Available |
| County projections (for forward runs) | Existing | `data/projections/*/county/*.parquet` | Available (not needed for backtest, but needed for production) |

**Sequencing:** PP3-S04 (model spec) can proceed in parallel with S05 since the backtest design is method-agnostic. However, S04 must be finalized before the backtest can actually be executed, because the trend model to be tested is defined there.

## 8. Resolved Questions (Human Decisions -- 2026-02-28)

1. **Threshold calibration:** **Accepted as proposed.** The 10% / 15% / 25% MedAPE thresholds for HIGH / MODERATE / LOWER tiers are adopted for Phase 1. These thresholds are revisable at the S07 gate after seeing actual backtest error distributions; if empirical results suggest the thresholds are miscalibrated, they can be adjusted before final acceptance.
2. **Exclusion ceiling:** **No cap.** The 10%-of-tier-membership cap is removed. Structural-break exclusions will be handled case-by-case, with each exclusion individually documented (break year, cause, and rationale). Reviewers retain judgment to flag excessive exclusions as a model-specification concern, but no hard numeric ceiling applies.
3. **Secondary window:** **Include it.** The secondary window (train 2000-2019, test 2020-2024) will run in Phase 1 alongside the primary window. The additional diagnostic value justifies the marginal scope increase.
4. **EXCLUDED tier:** **Run informational backtest.** Places below 500 population will be backtested for informational purposes -- their error distributions contribute signal for threshold calibration and future tier-boundary decisions. These places are not evaluated against pass/fail thresholds, and their results are reported in a separate section from the acceptance evaluation (see Section 2.4).

---

| Attribute | Value |
|-----------|-------|
| **Last Updated** | 2026-02-28 |
| **Version** | 1.0 |
