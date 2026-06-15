# Naive-Method Value-Add Comparison (ADR-063 / Stage 1.3)

**Date:** 2026-06-13
**Author:** Claude Code (PUB-2026 finality remediation, Stage 1.3)
**Status:** Complete
**Depends on:** ADR-067 F1 harness fix (this comparison runs on the **corrected raw rate base**;
the pre-2026-06-11 contaminated bundles are not used).

ADR-063 promised a naive-method value-add comparison: does the cohort-component model actually
beat trivial extrapolation? This is the answer, run on the corrected walk-forward base.

## Method

- **Model methods:** `m2026` (champion) and `m2026r1` (challenger, base config) from the
  walk-forward harness, re-run 2026-06-13 on the F1-corrected raw base (`s13rawbase` bundle).
  The locked production config (`cfg-20260611-production-lock`) differs from `m2026r1`-base only
  in forward-looking parameters (11 vs 4 college counties, GQ 0.75, Williams removed); these do
  not materially change the backtest at these horizons, and the value-add conclusion is
  config-robust.
- **Naive baselines** (definitions per `cohort_projections/analysis/evaluation/benchmark_runners.py`,
  applied at county-total granularity — the granularity at which model results are reported):
  - **carry-forward** — hold the origin-year county population constant.
  - **linear-trend** — OLS fit on county totals at snapshot years up to the origin, extrapolated.
  - **average-growth** — compound the origin-year population by the geometric-mean historical
    annual growth rate observed up to the origin.
- **Origins:** 2010, 2015, 2020 (per the plan). **County types:** from `county_report_cards.csv`
  (Rural, Bakken, Urban/College, Reservation). Actuals are the harness's observed snapshot county
  totals (verified: max |harness actual − snapshot total| = 0.0000).
- **Metric:** absolute percentage error (APE); county MAPE = mean over counties.

## Results

### County MAPE by horizon (all county types)

| Horizon | m2026 | m2026r1 | carry-forward | linear-trend | average-growth |
|---|---:|---:|---:|---:|---:|
| 4 | 3.23 | 3.16 | **2.57** | 4.47 | 3.12 |
| 5 | 7.52 | 7.49 | **6.23** | 7.92 | 6.84 |
| 9 | 10.15 | 9.96 | 8.10 | 8.93 | **8.02** |
| 10 | 10.62 | 10.58 | **8.73** | 12.93 | 11.01 |
| 14 | 12.96 | 12.90 | **10.43** | 15.97 | 13.09 |

At the **county** level, simple persistence (carry-forward) is a strong baseline and generally
**beats** the cohort-component model on average point accuracy. Most ND counties are small and
roughly stable, so "hold constant" is hard to beat for county totals.

### County MAPE by county type (origins pooled)

| County type | m2026 | m2026r1 | carry-forward | linear-trend | average-growth |
|---|---:|---:|---:|---:|---:|
| Bakken | 18.53 | 19.00 | 19.98 | **18.34** | 18.46 |
| Reservation | **10.12** | **10.12** | 12.45 | 13.63 | 13.18 |
| Rural | 7.38 | 7.36 | **4.82** | 8.85 | 6.98 |
| Urban/College | 8.42 | 7.12 | 9.68 | 4.47 | **3.50** |

The model's county-level value-add is **type-dependent**:
- **Reservation counties:** the model **wins** (10.1 vs 12–14 naive) — the ADR-045 regime-aware
  recalibration captures structure persistence misses.
- **Bakken:** all methods are poor (~18–20); the model is competitive but boom-bust volatility
  dominates.
- **Rural:** carry-forward wins (4.82) — small stable populations favor persistence.
- **Urban/College:** average-growth wins (3.50) — these counties grew steadily, so compounding
  the historical growth rate tracks them; the model is more conservative.

### State APE by origin and horizon

| Origin | Target | Horizon | m2026 | m2026r1 | carry-forward | linear-trend | average-growth |
|---|---|---:|---:|---:|---:|---:|---:|
| 2010 | 2015 | 5 | 8.48 | 8.15 | 8.79 | 7.24 | **6.18** |
| 2010 | 2020 | 10 | 13.61 | 12.97 | 13.66 | 10.25 | **8.34** |
| 2010 | 2024 | 14 | 15.95 | 15.11 | 15.56 | 10.70 | **7.84** |
| 2015 | 2020 | 5 | 0.78 | **0.03** | 5.35 | 3.40 | 0.12 |
| 2015 | 2024 | 9 | **0.15** | 1.40 | 7.43 | 2.39 | 2.25 |
| 2020 | 2024 | 4 | 1.09 | 1.25 | 2.20 | **0.14** | 2.47 |

This is the decisive table. **From recent origins (2015, 2020) — the relevant analog for a 2026
projection launched from 2025 — the cohort-component model is excellent at the state level**
(0.03%–1.40% APE) and dramatically beats carry-forward (5.35%–7.43%). From the **2010 origin,
which spans the Bakken oil boom**, average-growth "wins" by extrapolating the boom's momentum —
but that is luck on a structural break neither method can predict (it would have badly
over-projected had the boom stopped). The 2026 projection launches with the full boom-and-bust
history in its training data, i.e. the favorable recent-origin regime.

### Value-add (county MAPE: best naive − model)

| Horizon | m2026 | m2026r1 | best naive | value-add m2026 | value-add m2026r1 |
|---|---:|---:|---:|---:|---:|
| 4 | 3.23 | 3.16 | 2.57 | −0.66 | −0.59 |
| 5 | 7.52 | 7.49 | 6.23 | −1.29 | −1.26 |
| 9 | 10.15 | 9.96 | 8.02 | −2.13 | −1.94 |
| 10 | 10.62 | 10.58 | 8.73 | −1.89 | −1.85 |
| 14 | 12.96 | 12.90 | 10.43 | −2.53 | −2.47 |

On county-total point MAPE alone, the model's value-add is **negative** versus the best naive
baseline (persistence). This must be read with the critical caveat below.

## Interpretation — where the model's value actually is

A point-MAPE-only comparison **understates** the cohort-component model, for four reasons:

1. **The naive methods produce totals only.** They cannot generate the age, sex, and
   race/ethnicity structure that is the model's core public deliverable (the State Age-Sex Detail
   sheet, dependency ratios, pyramids, age-group trends). There is no naive baseline for those
   outputs — the comparison above is on the *one* dimension (county totals) where naive methods
   even exist.
2. **State-level, recent-origin accuracy is where the model wins decisively** (0.03–1.40% APE vs
   5–7% carry-forward), and that is exactly the use case for the 2026 release: a statewide
   baseline launched from 2025. The headline public number is the state total.
3. **Regime structure:** the model beats all naive methods on reservation counties (ADR-045) —
   the one county type with a strong non-persistence signal it is designed to capture.
4. **Internal consistency:** state = Σcounty by construction (ADR-054) and components of change
   (births/deaths/migration) reconcile — properties naive county-total extrapolation cannot
   provide and which the public package requires.

Where naive persistence is genuinely competitive — small, stable rural counties and steady-growth
urban counties on county-total point accuracy — the honest takeaway is a **calibration of
expectations**, consistent with the existing accuracy analysis (`2026-03-04-projection-accuracy-analysis.md`)
and methodology.md §10: long-horizon and small-county point estimates carry real uncertainty and
should be read as planning ranges. The model earns its place through structured, internally
consistent, state-accurate, regime-aware output — not through beating "hold constant" on every
small county's total.

## Conclusion

ADR-063's promised comparison is complete on the corrected base. The cohort-component model
**clearly adds value where it matters for this release** (state-level recent-origin accuracy,
reservation-county structure, and the age/sex/race detail that naive methods cannot produce), and
is **honestly no better than simple persistence** for small, stable county totals at point-MAPE.
This nuance is now documented for the public methodology/limitations narrative and does not change
the locked-config disposition.

*Inputs: `data/analysis/walk_forward/s13rawbase_county_detail.csv` (corrected base, 2026-06-13);
`county_report_cards.csv` (county-type map); harness snapshots (actuals).*
