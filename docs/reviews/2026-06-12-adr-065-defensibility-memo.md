# ADR-065 Defensibility Memo: Accepted-Conservatism Rationale for the CBO-Adjusted Public Baseline

**Date:** 2026-06-12
**Author:** Claude Code (PUB-2026 finality remediation, Stage 2.2)
**Decision:** Tier-3, 2026-06-12 — **affirm both CBO adjustments as-is** (CBO front-loaded
migration ramp and −5% fertility), document the conservatism with quantified magnitudes, and
carry the rationale into the public copy. No further config movement.
**Source evidence:** ADR-067 F4 forward decomposition (verified against the reproduced
2026-05-27 production state); observed Census PEP Vintage 2025 components; provisional baseline
trajectory (final magnitudes confirmed at the Stage-3 locked-config rerun).

---

## 1. Purpose

The 2026-06-10 finality rigor review raised "stacked conservatism (high)" as a concern: the GQ
Phase-2 subtraction, college-smoothing asymmetry, and premature 5-10-5 convergence were said to
combine with the CBO downward adjustments while the one identified corrective (ADR-061 Decision
3) was not deployed. This memo records the disposition of that concern and the defensibility of
the baseline's conservative shape, using the one-factor forward decomposition produced under
ADR-067.

## 2. The conservatism stack, decomposed

One-factor forward effects on the state total, each measured by turning a single lever
off/on against the locked production configuration (ADR-067 F4):

| Lever | Direction | State 2050 | State 2055 | Status |
|---|---|---|---|---|
| CBO front-loaded migration ramp | remove → higher | +22,948 | +23,262 | **Affirmed (this memo)** |
| −5% fertility adjustment | remove → higher | +13,032 | +16,231 | **Affirmed (this memo)** |
| Convergence hold extension (ADR-061 D3) | apply → higher | +9,080 | +11,677 | Evaluated and **rejected** (ADR-061 disposition) |
| GQ correction fraction 1.0 → 0.75 | apply → higher | +3,048 | +3,619 | **Adopted** (already in locked config) |

Two of the rigor review's stacked layers have already moved since it was written:

- **The GQ layer was softened.** The config lock reduced the Phase-2 correction fraction from
  1.0 to 0.75 (ADR-061 D2 calibration via the EXP-C clean-base benchmark), unwinding ~+3k of
  the conservatism at 2050. This is now reflected in the locked baseline.
- **The convergence "under-projection" that D3 was meant to offset was largely a measurement
  artifact.** ADR-067 F1 found that every pre-2026-06-11 walk-forward benchmark ran on
  adjustment-contaminated rate inputs; on clean inputs the method shows mild *over*-projection
  (recent-origin signed bias +0.44, state APE-medium 0.38), not the systematic
  under-projection that motivated D3. With the premise corrected, there is materially less
  conservatism to offset, and D3 (a +11.7k-at-2055 stance change) was correctly rejected for
  this release rather than deployed as a counterweight.

What remains genuinely "stacked" is therefore the two CBO policy adjustments. They are not
modeling artifacts — they are deliberate current-policy assumptions imported from an external
authority (CBO January 2026 demographic outlook), and they are the subject of this memo.

## 3. Why both CBO adjustments are affirmed

1. **External grounding, not internal judgment.** ADR-065 adopts CBO's current-policy
   demographic outlook (laws and policies in place as of 2025-09-30; reduced net immigration
   for 2025–2029) as the public central projection's external benchmark. The migration ramp
   and the −5% fertility revision both trace directly to that outlook. This is a stronger
   provenance than the 2024 SDC predecessor's unquantified judgmental adjustments, and it is
   the product's single most defensible assumption set.
2. **Quantified and bounded.** The decomposition above gives the exact magnitude of each lever,
   so the conservatism is disclosed rather than hidden. The two adjustments together account
   for roughly −36k at 2050 relative to an unadjusted recent-trend path — a transparent,
   attributable figure.
3. **Softening would cost more than it gains.** Dropping −5% fertility would raise 2050 by
   ~13k and reopen ADR-065 days before finalization; softening the migration ramp (up to ~23k)
   would gut the strongest policy-grounded element and force a redesign of public framing
   already built around the CBO narrative. Neither improves defensibility; both add release
   risk.

## 4. Reconciliation of the 2025–2028 dip (QA Gate 1b)

The provisional baseline declines from 799,358 (2025) to a trough of 786,568 (2028, −1.60%)
before recovering to 791,127 (2030) and rising to 876,479 (2055, +9.65%). The rigor review
correctly noted this **reverses the observed recent PEP trend**, where state net migration was
strongly positive and rising in 2023–2025 (+4,088 / +3,660 / +3,322), driven by international
migration (+3,158 / +4,083 / +2,810).

The dip is **entirely the CBO front-loaded migration ramp**, not a defect or a data error.
Direct confirmation from the decomposition runs: with the CBO migration adjustment removed and
all else held at the locked config, the trajectory has **no dip at all** — it rises
monotonically from 799,358 (2025) → 800,188 (2026) → 804,556 (2028) → 812,212 (2030):

| Year | Baseline (CBO ramp on) | CBO migration off |
|---|---|---|
| 2025 | 799,358 | 799,358 |
| 2026 | 792,505 | 800,188 |
| 2027 | 788,267 | 801,958 |
| 2028 | **786,568** (trough) | 804,556 |
| 2029 | 787,825 | 807,959 |
| 2030 | 791,127 | 812,212 |

Mechanism: the additive-reduction schedule retains only f(2025)=0.20 of reference international
migration in 2025 (an ~80% near-term cut), ramping to 0.91 by 2029. CBO's outlook explicitly
reduces 2025–2029 net immigration for federal policy reasons; the public baseline imports that
assumption, so the near-term path steps down from the 2023–2025 international highs and then
recovers as the ramp relaxes and natural increase plus normalized migration resume growth. The
dip is the visible signature of the disclosed assumption, and the public copy must frame it as
such (not as a forecast that ND will shrink).

## 5. Long-horizon framing

The two CBO adjustments are near-term policy assumptions (the migration ramp fully relaxes by
2029; the −5% fertility shift compounds slowly). Their *level* effect persists and grows
modestly with horizon (fertility: −13k at 2050 → −16k at 2055), which is why the gap to an
unadjusted path widens slightly over time. The public caveats (ADR-042) should state that the
baseline is a projection conditioned on current-policy assumptions, that near-term immigration
is assumed to follow CBO's reduced 2025–2029 path, and that realized outcomes will differ if
federal policy or fertility behavior diverges from CBO's outlook. This is the honest framing
the SDC 2024 release lacked.

## 6. Disposition

- **Affirmed:** CBO front-loaded migration ramp; −5% fertility adjustment.
- **Already adopted:** GQ correction fraction 0.75 (one stacked layer trimmed).
- **Rejected for this release:** ADR-061 D3 convergence-hold extension (premise was a harness
  artifact; not needed as an offset).
- **Action items into public copy (Stage 5):** dip-as-CBO-assumption explanation near the
  first statewide exhibit; the four ADR-042 caveats with the quantified conservatism magnitudes
  from §2; long-horizon framing from §5.
- **Final magnitudes:** the trajectory figures here are the provisional (2026-05-27) baseline;
  the locked-config rerun (Stage 3.3) shifts them by the Williams-removal and GQ-0.75 deltas
  (small, ≈ +3–4k at 2050). This memo's *relative* decomposition is robust to that rerun; the
  Stage-4 sanity review will confirm the absolute numbers.

## References

- ADR-065 (CBO-Adjusted Public Baseline); ADR-067 F4 (forward decomposition); ADR-061
  disposition (`docs/reviews/benchmark_decisions/2026-03-09-m2026r1-vs-m2026.md`)
- `docs/reviews/2026-06-10-pub-2026-finality-rigor-review.md` (stacked-conservatism finding;
  blocking item B3 dip reconciliation)
- `docs/plans/2026-public-projection-release-handoff/finality-remediation-plan.md` (Stage 2.2)
- Observed components: `data/processed/pep_county_components_2000_2025.parquet`
- Decomposition outputs: `data/projections/sensitivity_20260611/{ref,cbo_off,fert_off,d3_hold15,gq075_fwd}/`
