# PUB-2026 Release QA — Gate-by-Gate Sign-off

**Date:** 2026-06-13
**Reviewer:** Claude Code (PUB-2026 finality remediation, Stage 5.3)
**Run under QA:** `m2026r1` / `cfg-20260611-production-lock` @ commit `12fa6f9`
(see [final-run-metadata.md](../plans/2026-public-projection-release-handoff/final-run-metadata.md))
**Method:** independent multi-agent audit (one auditor per gate, grounded in the on-disk
artifacts) followed by adversarial re-verification of every objective PASS claim; findings
remediated; gates re-confirmed.
**Verdict:** **PASS for the data/content under our control.** All numeric, structural, and
language gates pass and were adversarially confirmed. The remaining open items are
marketing-layout actions on the *rendered* PDF (which the State Data Center / marketing team
produces) and the final delivery/publication step — explicitly tracked below, not data defects.

Companion: [release-qa-checklist.md](../plans/2026-public-projection-release-handoff/release-qa-checklist.md)
(the control checklist this signs off), [2026-06-13-locked-run-sanity-check.md](2026-06-13-locked-run-sanity-check.md)
(Gate 1b), [final-run-metadata.md](../plans/2026-public-projection-release-handoff/final-run-metadata.md).

---

## Gate results

| Gate | Result | Notes |
|---|---|---|
| 1 — Method & Data Lock | **PASS** | Re-confirmed: locked method/config, commit, config sha256; input-coverage (mortality 2025–2055, 11 college counties, GQ 0.75); state 799,358 → 889,017. |
| 1b — Demographic Plausibility | **PASS** | [Sanity check](2026-06-13-locked-run-sanity-check.md): aggregation exact, components reconcile, dip = CBO, age/sex plausible, 53-county scan coherent, divergent counties dispositioned. |
| 2 — Public Download QA | **PASS** | CSV exactly 1,922 rows (1 scenario × 62 geographies × 31 years); state/region/county only; baseline only; no place rows; no `2025_2045` files; state = Σcounty and region = Σmember-county (max abs diff ≈ 2.3e-10, i.e. floating-point zero on ~800k); workbook uses **State Key Years** (not Scenario Summary); Data Dictionary == CSV's 18 columns; all 9 spec-required sheets present (+6 Chart sheets, allowed); new **State Age-Sex Detail** sheet present and reconciles exactly. Adversarially re-verified. |
| 3 — PDF Content QA (copy) | **PASS (copy)** | `draft-public-pdf-copy.md`: state/region/county only; Baseline (CBO-Adjusted) only; "projection, not a guaranteed outcome" present; four ADR-042 caveats placed at the first statewide exhibit with refreshed values; methodology summary covers base/fertility/mortality/migration/baseline; county callouts (Williams +52%→~63k, Ward −13%→~59k, Cass +32%→~266k, trough ~787k @2028) reconcile to the CSV. Rendered-PDF layout items are Gate 5. |
| 4 — Language QA | **PASS** | ADR-042 banned-language pass on the public-facing body: every hit of forecast/guarantee/certainty/prediction is a **required negation** ("a projection, not a forecast"; "not a guarantee") or the mandated caveat. The only bare uses are in the internal editorial header (first lines), which is layout guidance and does **not** render into the public PDF. `recent_trend_continuation` does not appear in the copy. Terminology (projection/scenario/Baseline (CBO-Adjusted)) consistent. |
| 5 — Accessibility & Design | **PARTIAL → deferred to marketing layout** | See "Open items" below. Contact/download section added to the PDF copy (with PLACEHOLDERs for live URLs); stale chart watermark fixed; remaining color/legibility/reading-order checks belong to the rendered PDF marketing produces. |
| 6 — Final Handoff | **Deliverables build-complete; delivery pending** | Workbook, CSV, chart-ready sheets, and PDF copy are built and correct; decision log (tracker + benchmark decision record + ADR-067) and DEVELOPMENT_TRACKER are current. Actual delivery to marketing and the publication step remain. |

## Adversarial verification

Each objective PASS claim (row count, scenario/level filters, aggregation identity, sheet
list, data-dictionary match, banned-language) was independently re-run from scratch by a
separate agent instructed to refute it. **No claim was overturned.** Two "refutations" were
nuance, not defects: (a) "Baseline (CBO-Adjusted)" appears 5× not 4× (the 5th is the editorial
header line); (b) state = Σcounty max abs diff is 2.3e-10, which exceeds a literal 1e-10
threshold but is floating-point zero (≈1e-15 relative). Both are immaterial.

## Open items (NOT data defects — marketing layout / delivery)

These were surfaced by the audit and are the responsibility of the final-PDF layout and the
delivery step. None affect the locked numbers or the downloadable data.

1. **Contact block & download links.** A "Contact And Downloads" section was added to the PDF
   copy with `[PLACEHOLDER]` markers; marketing must insert the live landing-page URL, the
   Excel/CSV download links, and the current State Data Center contact block before publication.
2. **Final figure accessibility.** The chart-ready PNGs are reference cuts. For the published
   PDF, marketing should ensure non-color-dependent encodings (the bar charts already encode
   sign by axis position as well as color; the line/age charts use markers), legible map
   labels/legends, and a logical reading order. (Gate 5 items 1–4, 6.)
3. **Pre-publication marks.** Chart PNGs and workbook headers carry a "pre-publication draft"
   mark (numbers are final/locked; the mark signals pending public layout). These should be
   removed/replaced at publication. The previously stale "refresh after final production rerun"
   chart watermark was corrected on 2026-06-13.
4. **Delivery & publication (Gate 6).** Deliver the workbook, CSV, and chart-ready tables to
   marketing; review the final rendered PDF against this folder; then publish. The decision log
   (this folder + `docs/reviews/benchmark_decisions/` + ADR-067) and DEVELOPMENT_TRACKER are
   already current.

## Conclusion

The PUB-2026 public dataset and content are **final and locked**, structurally exact,
demographically validated, ADR-042-compliant, and adversarially QA'd. The release is cleared
to hand to marketing for final PDF layout and publication; the only remaining work is the
layout/accessibility polish and the delivery/publication step itemized above.
