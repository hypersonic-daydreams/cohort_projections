# Census Bureau "College Fix" Research: Implications for m2026 Model

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-03-04 |
| **Author** | Claude Code (Opus 4.6) |
| **Type** | External research synthesis |
| **Scope** | Census Bureau college-town estimation methodology vs m2026 GQ correction and college-age smoothing |
| **Status** | Active reference |
| **Source Document** | `docs/reviews/2026-03-04-college-town-estimates-research.pdf` |
| **Related Reviews** | [`2026-03-04-projection-accuracy-analysis.md`](2026-03-04-projection-accuracy-analysis.md) |
| **Related ADRs** | [ADR-049](../governance/adrs/049-college-age-smoothing-convergence-pipeline.md), [ADR-055](../governance/adrs/055-group-quarters-separation.md) |
| **Related Code** | `cohort_projections/data/process/residual_migration.py`, `cohort_projections/data/load/base_population_loader.py` |

---

## The Core Insight

The Census/UMDI "College Fix" partitions the population by **enrollment status** (all students, not just dormitory residents), holds the enrolled segment completely static, and applies cohort-component mechanics only to the non-college population. This is a fundamentally different approach than what m2026 does — we partition by **housing type** (GQ vs household) and separately blend rates. These two partial fixes overlap and interact to create the documented double-dampening.

---

## What the Research Validates

- The core premise of ADR-055 (separate GQ from HH) is correct and standard practice
- The "revolving door" / "aging in place" diagnosis is nearly word-for-word what Census found
- Including ages 15-19 in the smoothing is correct

## What the Research Challenges

1. **GQ partition misses most students.** NDSU has ~12,800 enrolled but only ~3,900 in dorms. The 8,900 off-campus students still distort our migration rates.
2. **Rate blending (50/50) is the wrong technique.** Census uses complete exclusion, not blending. Our approach still passes half the distorted signal through.
3. **Ages 25-29 are uncovered.** Census includes 25-29 because the *departure* signal is the amplified side of the administrative data asymmetry. Our ADR-049 stops at 24.
4. **Phase 2's 100% GQ subtraction from historical denominators is too broad.** Census does NOT modify historical rate computation — it partitions at projection time only.

---

## Prioritized Actions

| Priority | Action | Expected Impact |
|----------|--------|-----------------|
| **1** | Investigate enrollment-based partition using ACS PUMS data for college counties | Would replace both Phase 2 and ADR-049 for ages 15-29; addresses root cause |
| **2** | Extend ADR-049 smoothing to ages 25-29 as an interim fix | Low-cost; addresses the asymmetric departure signal |
| **3** | Consider reverting Phase 2 GQ correction (or reducing to fractional) | 37,084-person sensitivity; Census approach doesn't modify historical rates this way |
| **4** | Verify 2020 stcoreview GQ counts against pandemic PCGQR corrections | If 2020 GQ is depressed, the 2020-2024 "growth" includes recovery, not just institutional expansion |
| **5** | Research "aging stayer" retention rates for NDSU/UND from ACS PUMS | Captures student-to-resident transitions that m2026 currently ignores |
| **6** | Deprecate ADR-049 once enrollment partition is in place | Eliminates double-dampening by design |
