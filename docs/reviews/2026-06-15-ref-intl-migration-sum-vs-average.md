# Reference International Migration: Sum-vs-Average Error in the Locked CBO Adjustment

**Date:** 2026-06-15
**Reviewer:** Claude Code (15-agent revalidation workflow `cbo-assumption-revalidation`, then independent re-verification of every load-bearing internal claim against live repo data)
**Scope:** The CBO additive migration adjustment in the locked PUB-2026 baseline (`m2026r1` / `cfg-20260611-production-lock`, commit `12fa6f9`) — specifically whether the near-term statewide decline is well-founded, and whether the `reference_intl_migration` parameter is correctly specified.
**Status:** **Open finding — disposition pending. Numbers currently UNCHANGED (locked run untouched).**
**Related ADRs:** ADR-050 (additive migration adjustment formula), ADR-065 (CBO-adjusted public baseline), ADR-037 (CBO schedule derivation), ADR-067 (divergence investigation + F1 harness fix). A corrective decision, if taken, should be recorded as **ADR-068**.
**Related reviews:** [`2026-06-10-pub-2026-finality-rigor-review.md`](2026-06-10-pub-2026-finality-rigor-review.md) (the dip-reconciliation blocker this finding reopens), [`2026-06-12-adr-065-defensibility-memo.md`](2026-06-12-adr-065-defensibility-memo.md), [`2026-06-13-locked-run-sanity-check.md`](2026-06-13-locked-run-sanity-check.md).

---

## Verdict

The locked baseline's headline near-term **decline** (799,358 in 2025 → trough **787,382 in 2028, −1.50%** → 889,017 in 2055) rests on two facts that, together, make the *depth* of that decline very likely overstated:

1. **The near-term decline is ~100% a product of the CBO migration adjustment**, not North Dakota's underlying cohort dynamics. With the adjustment removed, the state rises monotonically with no dip.
2. **The adjustment's volume input is ~3× too large** due to a sum-vs-average error: `reference_intl_migration = 10,051` is the **3-year sum** of ND international migration (2023–2025), but it is labeled "annual average" and applied **per year**. The true annual average is **3,350**.

The *direction* of the adjustment is sound and current — immigration genuinely collapsed nationally and in ND, and the model already uses CBO's latest (January 2026) vintage. So this is a **magnitude correction, not a case for removing the adjustment**. A first-order estimate is that correcting the numerator turns the −1.50% trough into approximately flat-to-slightly-positive near-term growth.

**This finding does not change any number on its own.** It documents the issue and the evidence so a disposition (correct-and-rerun vs. accept-as-intentional) can be made deliberately.

---

## How this was found

A 15-agent workflow (`cbo-assumption-revalidation`, 2026-06-15) was run to re-validate the CBO assumptions months after they were set, prompted by reasonable nervousness about publishing a near-term decline in a state accustomed to steady growth. Six finders (3 internal repo-archaeology, 3 external web-research) fed a synthesis, an adversarial-verification pass over the load-bearing claims, and a finalizer. The two **internal** load-bearing claims below were then re-reproduced by hand against live repo data before writing this document; the **external** claims are flagged as web-sourced and post-cutoff (the assistant's training cutoff is January 2026).

This document was subsequently reviewed by an independent **GPT-5.5 (xhigh, read-only) agent** (2026-06-15). It confirmed the sum-vs-average defect, the mislabeling, and the per-year application against the repo, and corrected the first-order trough estimate (see the correction note under Finding 2). Its overall verdict: *"The finding is sound enough to act on… the recommended disposition is appropriate: run a non-destructive corrected-numerator sensitivity first, then likely correct and rerun."*

---

## Finding 1 — The near-term decline is ~100% the CBO lever (verified)

Our own one-factor sensitivity runs (`data/projections/sensitivity_20260611/`, `ref` = CBO-on vs `cbo_off` = CBO-off) show that **with the CBO migration adjustment off, the state total rises monotonically every year — no dip**. With it on, the state dips to a 2028 trough. The entire near-term sign of the change flips on this one lever.

> The on-disk sensitivity CSVs are household-only and begin at 2026, so the often-cited "minimum YoY ≈ +1,770" is the 2026→2027 household-only step, **not** the true first increment. Including the 2025 base and constant group quarters, the first GQ-inclusive `cbo_off` increment (2025→2026) is ≈ **+830** — still positive, so the monotonic-rise conclusion holds, but the rise is gentler than the +1,770 figure implies.

Observed reality agrees that nothing is contracting in the base data: ND **grew +0.75% in 2025** (793,387 → 799,358), with positive net migration (+3,322) and domestic migration flipping positive (+512). The dip is therefore a property of the disclosed CBO front-loaded migration assumption, not of ND's demographics.

> **Caveat (provenance):** the on-disk `sensitivity_20260611` decomposition was run on the **provisional 2026-05-27 config**, two days before the 2026-06-13 locked production rerun. The *relative* one-factor deltas are robust to the lock (and `methodology.md` §10.4 asserts the same no-dip behavior for the locked run), but **no `cbo_off` absolute trajectory exists on disk for the locked config**. `methodology.md` line 1327 currently states the decomposition "was completed against the locked run," which is accurate only for the relative deltas, not the absolute trajectories. See Secondary Findings.

## Finding 2 — `reference_intl_migration = 10,051` is a 3-year sum, mislabeled and used as an annual average (verified)

**The value is the sum, not the average.** ND statewide international migration from `data/processed/pep_county_components_2000_2025.parquet` (preferred estimate):

| Year | ND international migration |
|---|---:|
| 2023 | 3,158 |
| 2024 | 4,083 |
| 2025 | 2,810 |
| **Sum (2023–2025)** | **10,051** ← the config value |
| **Mean (2023–2025)** | **3,350.3** ← the true annual average |

**It is labeled "annual average" in three places:**
- `config/projection_config.yaml:272` — `reference_intl_migration: 10051  # Annual international migration (PEP 2023-2025 avg)` (and the deprecated `restricted_growth` alias at line 306).
- `docs/methodology.md:884` — "M_intl = 10,051: average annual international net migration".
- `docs/governance/adrs/050-restricted-growth-additive-migration-adjustment.md:96` — `state_intl_migration = 10,051  # From PEP 2023-2025 annual average` (repeated at 127, 258).

**It is applied per year.** `cohort_projections/core/migration.py:222`:
```python
annual_reduction = ref_intl * (1.0 - factor)  # persons/year not arriving
reduction_rate = annual_reduction / ref_pop   # per-capita rate decrement
```
ADR-050:104 spells out the intended accounting itself: `annual_reduction = 10,051 * (1 - 0.20) = 8,041 persons`. So in 2025 the engine removes **~8,041 net migrants statewide** — more than ND's *entire* realized 2025 net migration (+3,322) and ~3× realized 2025 international migration (+2,810). An 80% haircut applied to the *true* annual flow (~3,350) should remove ~2,680.

The "annual average" labeling, the variable name `annual_reduction`, and the comment "persons/year" are dispositive that the **intent was an annual figure** — i.e., 10,051 is an error (the 3-year sum substituted where the average was meant), not a deliberate 3× severity choice.

### Magnitude of the over-suppression

Statewide "persons/year not arriving" under the current (10,051) vs. corrected (3,350) numerator. The reduction keyed to year *Y* is applied in the *Y → Y+1* projection step ([cohort_component.py:421](../../cohort_projections/core/cohort_component.py#L421) loops `for year in range(start, end)`; the factor is `schedule.get(year)`), so it affects the *Y+1* population:

| Year (Y) | f | 1 − f | Current (×10,051) | Corrected (×3,350) | Over-removed | Feeds level |
|---|---:|---:|---:|---:|---:|---|
| 2025 | 0.20 | 0.80 | 8,041 | 2,680 | 5,361 | 2026 |
| 2026 | 0.37 | 0.63 | 6,332 | 2,111 | 4,221 | 2027 |
| 2027 | 0.55 | 0.45 | 4,523 | 1,508 | 3,015 | **2028 (trough)** |
| 2028 | 0.78 | 0.22 | 2,211 | 737 | 1,474 | 2029 |
| 2029 | 0.91 | 0.09 | 905 | 302 | 603 | 2030 |
| **Σ feeding the 2028 trough (2025–2027 transitions)** | | | **18,896** | **6,299** | **12,597** | |

### First-order estimate of the corrected trajectory

The **2028 trough level reflects the 2025/2026/2027 transitions only** — the 2028-keyed reduction lands on the 2029 level. So the cumulative over-removal baked into the trough is **~12,597**, not the full five-year sum. Adding it back (first-order; ignores second-order survivorship/birth feedback and the per-capita, population-weighted distribution mechanics):

`787,382 (locked 2028 trough) + ~12,597 ≈ 799,980`.

An independent cross-check — scaling the *empirical* provisional `cbo_off` − `ref` gap at 2028 (≈17,988) by the corrected fraction (1 − 3,350/10,051 ≈ ⅔) and anchoring on the locked trough — lands at ≈ **799,374**. Both methods put the corrected 2028 in the **~798.5k–800.0k** range, i.e. **essentially flat against the 2025 base of 799,358**: the visible −1.50% decline collapses to a near-flat plateau, with the residual sign (a shallow dip of a few hundred vs. a slight rise) genuinely uncertain at this precision. The 2055 endpoint rises modestly above 889,017. These are estimates; the precise figures require an actual rerun with the corrected parameter.

> **Correction note (GPT-5.5 xhigh review, 2026-06-15):** an earlier draft of this section added back all four 2025–2028 ramp reductions (~14,072) and reported ≈801,454 / +0.26%. That double-counted the 2028-keyed reduction, which does not affect the 2028 output. The corrected trough is ~799.4k–800.0k (essentially flat), not a slight gain.

## Finding 3 — The direction of suppression is sound and not stale (web-sourced; adversarially verified)

The macro premise behind the suppression has, if anything, strengthened since it was set, so the fix is to the *magnitude*, not to the existence of the adjustment:

- US net international migration collapsed in the Census Vintage 2025 estimates (≈2.7M in 2024 → ≈1.3M in 2025). *[web-sourced, post-cutoff]*
- ND's own realized international migration fell ~31% (4,083 → 2,810). *[verified in repo]*
- The model already uses **CBO's current vintage** — January 2026 Demographic Outlook (Pub 61879, pulled 2026-01-20); the next CBO update is ~September 2026. The newest vintage revised immigration **down**, not up. *[web-sourced, post-cutoff; CBO PDFs returned HTTP 403 to the finders, so exact figures are secondary-summary confidence]*

> The risk is therefore **not** that CBO is now too optimistic; a blanket re-pull is not warranted. The exposure is the trough *depth*, driven by the Finding 2 numerator.

---

## Secondary findings (lower priority; separate from the numerator)

- **Provenance overstatement.** `docs/methodology.md:1327` states the one-factor decomposition "was completed against the locked run." Accurate for the *relative* deltas only; the absolute `cbo_off` trajectory on disk is the provisional 2026-05-27 config. Either rerun `cbo_off` on the locked config or amend the sentence.
- **External phrasing to keep out of public copy.** The "first net-negative US migration in 50+ years" framing conflates a Census *positive* series (1.3M, 321K) with a separate Brookings/Frey 2025 *negative* estimate. Do not place a specific net-negative national figure in public copy without precise sourcing.
- **Williams +51.5% looks too optimistic** given its 2024–25 growth stall (+0.03%), Williston city decline, and flat 2026 oil (low-$50s WTI, low-20s rig count). Ward (−13%) and the rural natural-decline pattern remain well-supported. Framing/sensitivity, not a numerator error.

---

## Recommended disposition (decision is the owner's)

1. **Correct and rerun (recommended if 10,051 is judged an error).** Set `reference_intl_migration` to the true annual average (~3,350), or change the formula to divide the 3-year window by 3; rerun the locked production; regenerate public artifacts and the marketing package; re-execute the release QA gates. Record as ADR-068. Supersedes the current "final/locked" state.
2. **Confirm magnitude first (low-risk intermediate).** Run a single non-destructive sensitivity projection with `reference_intl_migration = 3,350` (f-schedule unchanged) to replace the first-order estimate above with the exact corrected trough and 2055 endpoint, before committing to a full rerun.
3. **Accept as intentional (requires explicit rationale).** If the deeper suppression is judged a defensible conservative choice, keep the numbers but **fix the misleading "annual average" labels** in config / methodology / ADR-050 to state plainly that 10,051 is a 3-year-sum severity multiplier, and hedge the public copy to frame the dip as a disclosed CBO current-policy assumption. This is hard to defend given the labeling evidence.

Regardless of path, the public copy should frame the near-term dip explicitly as the modeled effect of CBO current-policy immigration assumptions (a disclosed scenario lever), not as a forecast that ND's underlying demographics are contracting.

---

## Appendix — independent reproduction

```bash
# Finding 2a: the value is the 3-year SUM, not the average
python -c "import pandas as pd; d=pd.read_parquet('data/processed/pep_county_components_2000_2025.parquet'); \
d=d[d.is_preferred_estimate]; g=d[d.year.isin([2023,2024,2025])].groupby('year').intl_mig.sum(); \
print(g); print('sum', int(g.sum()), 'mean', round(g.mean(),1))"
# -> 2023 3158 / 2024 4083 / 2025 2810 ; sum 10051 ; mean 3350.3

# Finding 2b: the label ("annual average") and per-year application
#   config/projection_config.yaml:272 ; docs/methodology.md:884 ;
#   docs/governance/adrs/050-restricted-growth-additive-migration-adjustment.md:96
#   cohort_projections/core/migration.py:222  (annual_reduction = ref_intl * (1.0 - factor))

# Finding 1: the decline is the CBO lever (cbo_off rises monotonically, no dip)
#   data/projections/sensitivity_20260611/{ref,cbo_off}/baseline/county/*_summary.csv
```
