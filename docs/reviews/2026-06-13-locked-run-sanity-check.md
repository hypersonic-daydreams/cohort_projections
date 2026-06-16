# Locked-Config Production Run — Dated Sanity Check

> **⚠️ SUPERSEDED (numbers) by ADR-068 (2026-06-15) + its 2026-06-16 survival-horizon amendment.**
> This sanity check reviewed the 2026-06-13 locked run (→ 889,017 @2055). Current figures:
> 799,358 → trough 797,298 (2027) → **898,907 (2055)**, 90+ @2055 **8,172**. The structural
> findings (exact aggregation, components reconcile, dip = CBO lever) still hold; numbers here
> are retained as the historical record.

**Date:** 2026-06-13
**Reviewer:** Claude Code (PUB-2026 finality remediation, Stage 4.1; QA Gate 1b)
**Run under review:** `m2026r1` / `cfg-20260611-production-lock` @ commit `12fa6f9`
(see [final-run-metadata.md](../plans/2026-public-projection-release-handoff/final-run-metadata.md))
**Verdict:** **PASS** — all structural checks exact; all demographic patterns plausible and explained; three divergent counties (Williams, Ward, Grand Forks) dispositioned (see companion framing doc `2026-06-13-divergent-counties-methods-and-framing.md`).

This is the post-ADR-065/066 dated sanity review the rigor review (2026-06-10, blocking item B3) and repo practice require before final numbers. Every number below is computed from the locked run on disk.

---

## 1. Structural integrity (exact checks)

| Check | Result |
|---|---|
| State = sum of 53 counties (ADR-054), all years | **max abs diff 0.0000** |
| Components reconcile to population change: Δpop = births − deaths + net_migration, each year | **residual +0 every year** (GQ constant confirmed) |
| Negative population cells | none |
| Counties collapsing below 2,000 / to zero / negative | none |
| Scenario | `baseline` only (ADR-065); no stale scenario rows |

The components-of-change reconciliation is exact because group quarters are held constant (ADR-055 Phase 1), so GQ drops out of the year-over-year change and the household-population flows fully explain the published total trajectory.

## 2. State trajectory and the 2025–2028 dip

799,358 (2025) → trough **787,382 (2028, −1.50%)** → 792,478 (2030) → 836,767 (2040) → 872,730 (2050) → **889,017 (2055, +11.2%)**.

The dip reverses the recent observed positive trend, and that is **by design**: it is entirely the CBO front-loaded migration ramp (f(2025)=0.20 → 0.91 by 2029). The Stage-1 decomposition confirmed that with the CBO migration adjustment removed, the trajectory rises monotonically with no dip (see [ADR-065 defensibility memo](2026-06-12-adr-065-defensibility-memo.md) §4). Not a defect.

## 3. Components of change vs PEP observed

Observed (Census CO-EST2025, ND, sum of counties) vs projected (locked, household population):

| | Births | Deaths | Natural increase | Net migration |
|---|---:|---:|---:|---:|
| Observed 2023 | 9,729 | 6,936 | +2,793 | +4,088 |
| Observed 2024 | 9,583 | 6,905 | +2,678 | +3,660 |
| Observed 2025 | 9,760 | 7,130 | +2,630 | +3,322 |
| Projected 2026 | 9,060 | 5,084 | +3,976 | −10,545 |
| Projected 2030 | 9,272 | 4,933 | +4,338 | −764 |

Three reconciliations, all expected:

- **Births** (~9,060 vs observed ~9,760): the −5% CBO fertility adjustment plus age-structure shift. In range.
- **Deaths** (~5,000 projected vs ~7,000 observed): **this gap is the group-quarters artifact, not a mortality under-projection.** The engine projects the household population; GQ (~31,000, dominated for mortality by nursing homes) is held constant per ADR-055, so its ~2,000/yr deaths are not in the household flow. Because held-constant GQ implies matching entrants, this does **not** affect the published population trajectory (confirmed by the exact reconciliation in §1). **Action for Stage 5:** any public components table must label deaths as household-basis or add a GQ note, or it will read as inconsistent with PEP's total-deaths figure.
- **Net migration** early years deeply negative (−10,545 in 2026, recovering to −764 by 2030, positive after): the CBO ramp at its most aggressive in the first year. This is the mechanism behind the dip (§2).

## 4. Age structure and sex ratio (state)

| Year | Total | % under 18 | % 18–64 | % 65+ | Median age | Sex ratio (M/100F) |
|---|---:|---:|---:|---:|---:|---:|
| 2025 | 799,358 | 23.6 | 59.5 | 16.9 | 35 | 105.4 |
| 2035 | 814,484 | 22.2 | 60.4 | 17.4 | 36 | 105.7 |
| 2045 | 855,475 | 21.1 | 62.2 | 16.7 | 38 | 105.4 |
| 2055 | 889,017 | 20.4 | 61.0 | 18.5 | 40 | 105.4 |

Plausible: gradual aging (median 35→40, %65+ rising modestly to 18.5%), a slowly shrinking youth share, and a stable male-skewed sex ratio (~105, consistent with ND's energy/agriculture workforce). No age-structure or sex-ratio drift artifacts.

Large/divergent counties (2025 → 2055):

| County | %65+ 2025→2055 | Median age 2025→2055 | Sex ratio 2055 |
|---|---|---|---|
| Cass (Fargo/NDSU) | 13.4 → 16.7 | 33 → 38 | 104 |
| Burleigh (Bismarck) | 19.0 → 23.0 | 38 → 44 | 104 |
| Grand Forks (UND) | 14.9 → 14.9 | 30 → 34 | 104 |
| Ward (Minot/MAFB) | 15.2 → 19.3 | 34 → 41 | 109 |
| Williams (Williston) | 10.8 → 12.3 | 32 → 37 | 110 |

All coherent: each county ages gradually; Williams stays the youngest (lowest %65+, reflecting its energy-era working-age in-migrants and their children); Ward and Williams retain the highest sex ratios (Minot AFB; oil-field workforce). No county shows runaway aging or sex-ratio drift.

## 5. 53-county scan

Growth 2025→2055 spans **−36% to +76%**, a demographically coherent spread:

- **Top growth — all Bakken oil + urban anchor:** McKenzie +76%, Williams +52%, Billings +49%, Cass +32%, Morton +29%. The oil counties' projected net migration is **at or below recent observed** (McKenzie +259/yr proj vs +272 obs; Williams +388 vs +780; Billings +9 vs +10) — the growth is compounding natural increase from very young age structures (e.g. McKenzie 2035 births 210 vs deaths 77), **not** boom-migration extrapolation.
- **Top decline — aging rural:** Nelson −36%, Slope −35%, Walsh −27%. Consistent with long-running rural out-migration and old age structures. Smallest counties (Slope 628→410) decline without collapsing.
- Ward −13% and Grand Forks −4% are the documented divergences from SDC 2024 (see §6).

No county exhibits implausible terminal dynamics.

## 6. Divergent-county disposition (Gate 1b)

The three counties most likely to draw public scrutiny are dispositioned in detail in the companion reference `2026-06-13-divergent-counties-methods-and-framing.md`. Summary:

- **Williams +52%** — backtest-justified removal from college smoothing (ADR-067); brought in line with peer oil counties (McKenzie, Billings); conservative migration. Dominant driver of the +12.5k lock-vs-provisional difference.
- **Ward −13%** and **Grand Forks −4%** — observed 2020–2025 out-migration (Ward) and the disclosed CBO international-migration assumption (Grand Forks ≈ 52% of its decline; ADR-067 F4). Not method artifacts.

## 7. Conclusion

The locked run is structurally exact and demographically coherent. The two features that diverge from naive expectation — the near-term dip and the household-basis deaths — are both explained (CBO ramp; ADR-055 GQ-constant), and the high-growth oil counties are conservative on migration. Cleared for public-artifact construction (Stage 5), with one Stage-5 action: label/annotate deaths in any public components table (§3).
