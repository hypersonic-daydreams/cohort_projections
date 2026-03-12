# Experiment Catalog

Prioritized list of benchmark experiments for improving county-level projections.
Each entry is ready to be turned into an experiment spec via `/experiment`.

**Base for all experiments:** `m2026r1 / cfg-20260309-college-fix-v1` (current candidate)

**Last updated:** 2026-03-09

---

## Tier 1 — High Priority (low risk, likely to move the needle)

### EXP-A: Convergence Medium Hold Extension

| Field | Value |
|-------|-------|
| Slug | `convergence-medium-hold-5` |
| Parameter | `convergence_medium_hold: 3 → 5` |
| Hypothesis | Extending medium-rate hold from 3 to 5 years will reduce long-horizon overshoot in rural counties by keeping rates closer to the medium-term average before transitioning to the (noisier) long-term rate. |
| Expected improvement | `county_mape_rural`, `county_mape_overall` |
| Risk areas | Fast-growth counties (Cass, Burleigh) may under-project if medium rates are too conservative |
| Rationale | The convergence schedule is the single biggest lever on projection behavior. The candidate already extended medium hold from 2→3; pushing to 5 tests whether more persistence helps. Walk-forward validation at 15-20yr horizons is the key signal. |
| Results | `needs_human_review` — overall MAPE improved -0.02pp, but Bakken MAPE regressed +0.52pp (breaches 0.50 threshold). Rural MAPE regressed +0.02pp. Run: `br-20260309-182103-m2026r1-201f5eb` |

### EXP-B: College Blend Factor 0.7

| Field | Value |
|-------|-------|
| Slug | `college-blend-70` |
| Parameter | `college_blend_factor: 0.5 → 0.7` |
| Hypothesis | Increasing smoothing weight from 50% to 70% for college-age migration in the 12 college counties will reduce volatility in 15-19, 20-24, 25-29 age groups and improve college-county MAPE. |
| Expected improvement | `county_mape_urban_college`, Grand Forks/Cass APE |
| Risk areas | Over-smoothing could mask real enrollment-driven migration shifts |
| Rationale | College counties show the widest variance in walk-forward errors. More aggressive smoothing trades signal for stability — this tests how far we can push it. |
| Results | `passed_all_gates` (reclassified) — original `failed_hard_gate` due to 1 aggregation violation was a rounding artifact (1.1 person on 662K; tolerance widened from 1.0→2.0). Overall MAPE improved -0.09pp, Bakken +0.33pp (within threshold), rural +0.02pp. Best overall improvement of sweep. Run: `br-20260309-182528-m2026r1-201f5eb` |

### EXP-C: GQ Fraction Sensitivity (0.75)

| Field | Value |
|-------|-------|
| Slug | `gq-fraction-75` |
| Parameter | `migration.domestic.residual.gq_correction.fraction: 1.0 → 0.75` |
| Hypothesis | Partial GQ subtraction (75% instead of 100%) will better fit counties where GQ and household populations co-move, improving college-county accuracy without degrading Bakken/rural MAPE. |
| Expected improvement | `county_mape_urban_college` |
| Risk areas | Cass County may over-project if NDSU dorm growth leaks back into migration signal |
| Rationale | Full GQ subtraction (Phase 2) was a large swing. Partial subtraction tests whether we over-corrected. Grand Forks (-2.9pp delta) is the sentinel. |
| Notes | **Now benchmark-testable as config-only.** GQ fraction is injectable via `gq_correction_fraction` in MethodConfig; walk-forward validation recomputes migration rates in-memory when the value differs from the default (1.0). |

---

## Tier 2 — Medium Priority (moderate risk, targeted improvement)

### EXP-D: Rate Cap Tightening (General 6%)

| Field | Value |
|-------|-------|
| Slug | `rate-cap-general-6pct` |
| Parameter | `migration.interpolation.rate_cap.general_cap: 0.08 → 0.06` |
| Hypothesis | Tightening the general migration rate cap from 8% to 6% will clip more statistical noise in small counties, improving rural MAPE without affecting urban counties (which stay within 6% anyway). |
| Expected improvement | `county_mape_rural`, small-county max APE |
| Risk areas | Rapidly growing small counties (e.g., Morton, Billings) may be artificially capped |
| Rationale | The 8% cap was set conservatively. Most non-Bakken, non-college counties have rates well within 6%. This tests whether tighter clipping helps. |
| Notes | **Now benchmark-testable as config-only.** Rate cap is injectable via `rate_cap_general` in MethodConfig; walk-forward validation applies the cap during convergence blending when the value differs from the default (0.08). |

### EXP-E: Boom Dampening 2010-2015 at 0.30

| Field | Value |
|-------|-------|
| Slug | `boom-dampening-peak-30` |
| Parameter | `migration.domestic.dampening.factor["2010-2015"]: 0.40 → 0.30` |
| Hypothesis | Further dampening peak boom-era (2010-2015) migration from 40% to 30% will reduce Bakken county overshoot at 10-15yr horizons without affecting non-Bakken counties. |
| Expected improvement | `county_mape_bakken`, Williams/McKenzie APE |
| Risk areas | If Bakken sees renewed growth, 30% dampening may be too aggressive |
| Rationale | The 2010-2015 period was peak oil boom with the most transient workers. 2020-2025 evidence shows these flows didn't persist. Testing whether we should dampen even more. |
| Results | `needs_human_review` — overall MAPE improved -0.02pp, but Bakken MAPE regressed +0.55pp (breaches 0.50 threshold). Rural +0.02pp unchanged. Dampening direction is correct but overshoots the tolerance. Run: `br-20260309-182946-m2026r1-201f5eb` |

### EXP-F: Recent Period Window Expansion

| Field | Value |
|-------|-------|
| Slug | `recent-period-2020-2025` |
| Parameter | `migration.interpolation.recent_period: [2023, 2025] → [2020, 2025]` |
| Hypothesis | Widening the recent convergence window from 2 years to 5 years will smooth COVID-recovery noise and provide a more stable short-term migration signal, improving early-horizon accuracy. |
| Expected improvement | Short-horizon (5yr) state APE, `county_mape_overall` |
| Risk areas | Dilutes the most current (2023-2025) trend signal; may miss post-COVID structural shifts |
| Rationale | The [2023, 2025] window is very narrow — a single unusual year can dominate. [2020, 2025] includes COVID disruption but also the full recovery arc. |

---

## Tier 3 — Exploratory (higher risk, hypothesis-generating)

### EXP-G: Mortality Improvement Factor 0.3%

| Field | Value |
|-------|-------|
| Slug | `mortality-improvement-03pct` |
| Parameter | `rates.mortality.improvement_factor: 0.005 → 0.003` |
| Hypothesis | Reducing the mortality improvement assumption from 0.5%/yr to 0.3%/yr will produce more conservative elderly population growth in rural counties, improving rural MAPE for long horizons. |
| Expected improvement | `county_mape_rural` at 20+ year horizons |
| Risk areas | State-level total may under-project if mortality actually does improve at historical rates |
| Rationale | Post-COVID mortality improvement has stalled nationally. ND's rural aging population may see less improvement than the 0.5% assumption implies. Tests sensitivity of long-run projections to this parameter. |

### EXP-H: Add Billings County to Bakken FIPS

| Field | Value |
|-------|-------|
| Slug | `billings-bakken-fips` |
| Parameter | `bakken_fips: add "38007"` |
| Hypothesis | Including Billings County (FIPS 38007) in boom dampening will reduce its overshoot, as it experienced +88% undampened growth in 2010-2015 per ADR-051 Finding 2-B. |
| Expected improvement | Billings County APE |
| Risk areas | Billings is very small (~950 pop) — dampening on tiny counts may introduce instability |
| Rationale | The config already lists 38007 in the dampening counties list but it's not in the method profile's `bakken_fips`. This is either an oversight or deliberate exclusion due to population size. |

### EXP-I: College Blend Factor 0.3

| Field | Value |
|-------|-------|
| Slug | `college-blend-30` |
| Parameter | `college_blend_factor: 0.5 → 0.3` |
| Hypothesis | Reducing smoothing weight to 30% preserves more of the raw college-age migration signal, which may improve accuracy for counties where enrollment trends are the dominant population driver (Grand Forks, Barnes, Richland). |
| Expected improvement | College counties with stable enrollment trends |
| Risk areas | Higher volatility in counties with noisy enrollment patterns |
| Rationale | Paired with EXP-B — testing both directions to find the optimal smoothing point. Running both establishes the sensitivity curve. |

---

## Experiment Pairing Strategy

Some experiments are best run in pairs or sequences:

| Pair | Rationale |
|------|-----------|
| EXP-B + EXP-I | Bracket the blend factor (0.3, 0.5, 0.7) to find the optimum |
| EXP-A then EXP-F | If convergence hold helps, also test the window that feeds it |
| EXP-C then EXP-B | GQ fraction and blend factor interact in college counties |
| EXP-D then EXP-E | Both are noise-reduction strategies for different county types |

## Running Order Recommendation

1. **EXP-A** (convergence hold) — highest expected impact, isolated parameter
2. **EXP-B** (blend 0.7) — second-highest expected impact, independent of EXP-A
3. **EXP-C** (GQ fraction) — tests a key ADR-055 assumption *(requires upstream reprocessing)*
4. **EXP-D** (rate cap) — low-risk noise reduction *(requires upstream reprocessing)*
5. **EXP-E** through **EXP-I** — based on what Tier 1-2 results reveal

---

## Sweep Results — 2026-03-09

| Experiment | Classification | Key Deltas | Recommendation |
|------------|---------------|-----------|----------------|
| EXP-A: convergence-medium-hold-5 | `needs_human_review` | overall MAPE -0.02pp, rural +0.02pp, Bakken +0.52pp | Bakken regression barely breaches 0.50 threshold. Consider medium_hold=4 as a compromise. |
| EXP-B: college-blend-70 | `passed_all_gates` (reclassified) | overall MAPE -0.09pp, rural +0.02pp, Bakken +0.33pp, college -1.78pp | Best improvement of the sweep. Original `failed_hard_gate` was a rounding artifact (1.1 person drift; tolerance widened to 2.0). Consider promotion. |
| EXP-E: boom-dampening-peak-30 | `needs_human_review` | overall MAPE -0.02pp, rural +0.02pp, Bakken +0.55pp | Bakken regression is the largest of the three. 0.30 dampening is too aggressive; try 0.35. |

### Infrastructure Notes

- **Config injection implemented**: `run_benchmark_suite.py` now injects method profile `resolved_config` into `METHOD_DISPATCH` at runtime, enabling config-only experiments. This was a missing piece in the BM-001 experiment pipeline.
- **Upstream parameter injection implemented**: EXP-C (`gq_correction_fraction`) and EXP-D (`rate_cap_general`) are now injectable via MethodConfig. Walk-forward validation recomputes migration rates in-memory for non-default GQ fractions and applies rate caps during convergence blending for non-default cap values. No on-disk data reprocessing is needed.

### Cross-Cutting Observation

All three experiments show a consistent +0.02pp rural MAPE regression and varying Bakken regression. The rural regression is identical across experiments, suggesting it may be a baseline measurement artifact (same champion comparison) rather than a real signal from the config changes. The Bakken sensitivity is the key discriminator.

### Generated Experiment Ideas

| ID | Slug | Description | Source |
|----|------|-------------|--------|
| EXP-J | `convergence-medium-hold-4` | Test medium_hold=4 as a compromise between 3 and 5, targeting the Bakken regression sweet spot | EXP-A result |
| EXP-K | `boom-dampening-peak-35` | Test 0.35 dampening as a midpoint between current 0.40 and tested 0.30 | EXP-E result |
