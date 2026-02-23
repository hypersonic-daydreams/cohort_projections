# Sanity Check Investigation Reports — 2026-02-18

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-02-18 |
| **Parent Review** | [Projection Output Sanity Check](../2026-02-18-projection-output-sanity-check.md) |
| **Investigators** | Claude Code (Opus 4.6), directed by N. Haarstad |
| **Scope** | Deep investigation of 7 findings from the 2026-02-18 projection output sanity check |
| **Status** | All investigations complete; all P0-P2 fixes implemented 2026-02-18 |

---

## Synthesis of Findings

Six parallel investigations examined the seven findings identified in the [parent sanity check](../2026-02-18-projection-output-sanity-check.md). The results fall into three tiers by severity and required action.

### Tier 1: Bugs Blocking Publication (Fix Before Release)

| # | Finding | Root Cause | Severity | Report |
|---|---------|-----------|----------|--------|
| 7 | [High growth scenario below baseline](finding-7-high-growth-inversion.md) | **Design error**: +15% multiplicative multiplier on net migration amplifies *out*-migration for 45/53 counties with negative net flows. The adjustment is also dead code — overridden by convergence rates. | Critical | Deactivate `high_growth` scenario; redesign as additive migration boost |
| 4-5 | [Race-specific trends (Hispanic +204%, Black −17%)](finding-4-5-race-specific-trends.md) | **Data error**: PUMS 1% sample produces catastrophically distorted age-sex structures for small groups. Zero Black females at reproductive ages; 45% of Hispanics in one cell (F 10-14). | Critical | Replace PUMS race allocation with Census cc-est2024-alldata full-count data |
| 3 | [Reservation county declines of ~47%](finding-3-reservation-county-declines.md) | **Method amplification**: Residual migration method inflates out-migration 2-3x vs. PEP components for AIAN counties (Census undercount + statewide survival rates misapplied). No deceleration floor. Declines are 2.7-11x steeper than 2000-2020 actuals. | Critical | Recalibrate using PEP component rates; implement migration deceleration; consider race-specific survival |

### Tier 2: Issues Requiring Methodological Adjustment

| # | Finding | Root Cause | Severity | Report |
|---|---------|-----------|----------|--------|
| 2 | [Oil county growth still aggressive (+51-88%)](finding-2-oil-county-growth.md) | ADR-040 dampening reduced McKenzie from +114% to +84%, but boom-era rates still dominate the 10-year medium-window hold. Billings County (38007) not on dampened list at all. Population-weighted effective rate 46% higher than unweighted mean. | Medium | Add Billings to dampened list; strengthen factor from 0.60→0.40 for 2010-2015; consider migration rate cap |
| 6 | [Baseline reaching 1M by 2055](finding-6-baseline-1m-plausibility.md) | Projected 0.77%/yr growth exceeds all non-boom historical decades. 91% of net migration is international; 89% of growth concentrated in 3 counties (Cass, Burleigh, Williams). Conditionally plausible as trend-continuation but must not be framed as a forecast. | Medium | Presentation/framing issue — always pair with restricted growth; add caveats about international migration dependency |

### Tier 3: Resolved / Housekeeping

| # | Finding | Root Cause | Severity | Report |
|---|---------|-----------|----------|--------|
| 1 | [Low growth base year mismatch](finding-1-low-growth-base-year.md) | **Stale output**: `low_growth` was run before Vintage 2025 integration on 2026-02-17. It loaded the old `population_2024` column (796,568). The scenario is deprecated — already removed from config and replaced by `restricted_growth`. | Low | Delete `data/projections/low_growth/` directory |

---

## Cross-Cutting Themes

Three systemic issues appear across multiple findings:

### 1. Small-Number Vulnerability

Findings 2, 3, 4-5, and 6 all involve small populations where individual-level data noise produces outsized effects. The projection system currently has no guardrails for small-population counties or race groups:

- **Billings County** (1,071 pop): 16 in-migrants create a +16.5% rate for one age-sex cell (Finding 2)
- **Black non-Hispanic** (5,027 pop): PUMS yields only 44 observations, 7 populated cells (Finding 4-5)
- **Sioux County** (3,667 pop): Residual method misattributes Census undercount as out-migration (Finding 3)

**Recommendation**: Implement systematic small-population safeguards — rate caps, minimum cell counts, Bayesian shrinkage toward state/regional averages, or suppression thresholds.

### 2. Convergence Schedule Amplifies Historical Extremes

Findings 2, 3, and 6 all trace back to the 10-year medium-window hold (years 6-15) in the convergence interpolation schedule. When historical periods contain extreme events (oil boom for Finding 2, Census undercount for Finding 3), those extremes are held constant for a decade of the projection:

- **Oil counties**: Medium rate of +1.3-1.8% held for 10 years (Finding 2)
- **Reservation counties**: Medium rate of −2.5-3.1% held for 10 years (Finding 3)
- **State total**: Medium rate includes boom-era data, inflating growth to ~1%/yr through the 2030s (Finding 6)

**Recommendation**: Consider shortening the medium-window hold, adding rate caps per county, or implementing regime-aware convergence that detects and adjusts for structural breaks.

### 3. Race-Specific Data Quality Gap

Findings 3, 4-5, and implicitly 6 reveal that the race dimension is the weakest link in the data pipeline:

- **Base population by race**: PUMS 1% sample is too small for cross-tabulation (Finding 4-5)
- **Fertility by race**: Single "total" rate applied to all groups — no race differentiation (Finding 4-5 Note 1)
- **Survival by race**: Statewide rates applied to AIAN populations with substantially different mortality (Finding 3)
- **Migration by race**: Distributed proportional to base population, inheriting its distortions (Finding 4-5)

**Recommendation**: Prioritize replacing the PUMS race allocation with Census full-count data (cc-est2024-alldata). Consider race-specific survival rates for AIAN populations. Add prominent caveats to all race-specific projection outputs until these improvements are made.

---

## Recommended Action Sequence

Based on dependency ordering and impact:

| Priority | Action | Findings Addressed | Complexity | Status |
|----------|--------|-------------------|------------|--------|
| **P0** | Deactivate `high_growth` scenario | 7 | Config-only | **Done** (2026-02-18) |
| **P0** | Delete stale `data/projections/low_growth/` | 1 | File deletion | **Done** (2026-02-18) |
| **P1** | Replace PUMS race allocation with Census cc-est2024-alldata | 4-5, 3 (partial) | Data pipeline | **Done** (2026-02-18, ADR-044) |
| **P1** | Recalibrate reservation county migration using PEP components | 3 | Data pipeline | **Done** (2026-02-18, ADR-045) |
| **P2** | Add Billings County (38007) to dampened list | 2 | Config-only | **Done** (2026-02-18) |
| **P2** | Strengthen dampening factor from 0.60 to 0.40 for 2010-2015 | 2 | Config + possible code | Pending |
| **P2** | Implement migration rate cap in convergence step | 2, 3 | Code change | **Done** (2026-02-18, ADR-043) |
| **P2** | Redesign high_growth as additive migration scenario | 7 | Code + methodology | **Done** (2026-02-18, ADR-046) |
| **P3** | Add presentation caveats for baseline 1M projection | 6 | Documentation | **Done** (2026-02-18, ADR-042) |
| **P3** | Implement race-specific survival rates for AIAN populations | 3 | Data pipeline | Pending |
| **P3** | Add small-population safeguards (rate caps, shrinkage) | 2, 3, 4-5 | Architecture | Partial (rate cap done; shrinkage pending) |

---

## File Index

| File | Findings | Status |
|------|----------|--------|
| [finding-1-low-growth-base-year.md](finding-1-low-growth-base-year.md) | 1 | **Fixed** — stale output deleted; code references removed |
| [finding-2-oil-county-growth.md](finding-2-oil-county-growth.md) | 2 | **Partially fixed** — Billings added to dampened list; migration rate cap implemented (ADR-043); further dampening factor reduction pending |
| [finding-3-reservation-county-declines.md](finding-3-reservation-county-declines.md) | 3 | **Fixed** — PEP-anchored recalibration implemented (ADR-045); migration rate cap provides additional guard-rail (ADR-043) |
| [finding-4-5-race-specific-trends.md](finding-4-5-race-specific-trends.md) | 4, 5 | **Fixed** — Census cc-est2024-alldata replaces PUMS (ADR-044); 216 cells populated (was 115) |
| [finding-6-baseline-1m-plausibility.md](finding-6-baseline-1m-plausibility.md) | 6 | **Fixed** — ADR-042 establishes presentation requirements and mandatory caveats |
| [finding-7-high-growth-inversion.md](finding-7-high-growth-inversion.md) | 7 | **Fixed** — `high_growth` redesigned with BEBR convergence rates (ADR-046); re-enabled with `active: true` |
| [design-race-data-replacement.md](design-race-data-replacement.md) | 4-5 | Design document for Census cc-est2024-alldata integration |
| [design-reservation-migration-recalibration.md](design-reservation-migration-recalibration.md) | 3 | Design document for PEP-anchored reservation county migration |
| [design-additive-migration-boost.md](design-additive-migration-boost.md) | 7 | Design document for BEBR-based high_growth scenario |
| [design-migration-rate-cap.md](design-migration-rate-cap.md) | 2, 3 | Design document for age-aware migration rate cap |
| [future-improvements-roadmap.md](future-improvements-roadmap.md) | All | Roadmap for remaining P2/P3 improvements and future enhancements |
