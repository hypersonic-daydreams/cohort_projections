# PUB-2026 Finality & Rigor Review — Are the 2026 Projections Ready to Be Final?

**Date:** 2026-06-10
**Reviewer:** Claude Code (multi-agent review: 6 evidence readers, 3 assessors, 17 adversarial verifiers; all 17 material findings upheld, 0 refuted)
**Question:** Are the 2026 projections rigorous enough to be declared final? Is the methodology at least as good as the 2024 SDC projections? Are the method changes appropriate and not detrimental?
**Benchmark standard:** `docs/plans/2026-public-projection-release-handoff/marketing-ready/PUB-2026 Reference - 2024 SDC PDF.pdf` (16 pp., ND Dept of Commerce SDC, 2024-02-06)

---

## Verdict

**Methodology: meets-with-caveats — at least as good as SDC 2024 on 10 of 14 dimensions, clearly better on most.**
**Finality: NOT ready.** The blockers are governance and QA execution, not method quality. Four blocking items must be closed (estimated ~1–2 weeks of focused work), all of which are requirements the repo's *own* governance (SOP-003, release-qa-checklist.md, ADR-042) already imposes.

The 2024 SDC bar is genuinely low: a one-page methodology, unquantified judgmental adjustments (the ~60% migration reduction, college/Bakken/male tweaks), no validation or error metrics, no reproducibility (ADR-017 found SDC's own workbook misses its published 2050 number by ~44K), and a shipped numeric inconsistency (957,124 vs 957,194 for 2050). The 2026 system exceeds this structurally on every methodological axis. What SDC did better is *execution discipline at release*: it shipped a finished, internally-signed-off product with a track-record disclosure ("Note of Caution") and a components-of-change table. That is exactly where PUB-2026 currently falls short.

## SDC 2024 dimension comparison

| Dimension | Verdict | Basis |
|---|---|---|
| Base population data/vintage | **Better** | PEP Vintage 2025 (799,358), GQ separation, Sprague single-year ages vs SDC Census 2020 base, 5-yr groups |
| Projection method core | **Better** | Annual single-year cohort-component, 6 race groups, 53 counties vs 5-yr steps, no race detail |
| Fertility | **Better** | ND ASFR (CDC WONDER 2020–2023, TFR 1.863) vs 2016–2022 blend |
| Mortality | **Better** | ND-adjusted race-specific life tables + 0.5%/yr Lee-Carter improvement vs static 2020 life tables |
| Migration treatment | **Better** | Six-snapshot residual through 2024, GQ-corrected, algorithmic documented adjustments vs four windows ending 2020, judgmental unquantified reductions |
| Scenario construction | **Better** | CBO-grounded explicit adjustment schedule vs single series, narrative what-ifs |
| Geographic granularity | **Better** | State/region/county (+ internal place capability) |
| Horizon | **Better** | 2025–2055 vs 2020–2050 |
| Empirical validation | **Better** | Rolling-origin/walk-forward backtests, benchmark gates, 1,570-test suite vs none |
| Documentation/reproducibility | **Better** | 66 ADRs, methodology.md, versioned config vs one page |
| Special populations (GQ/college/reservation/Bakken) | **Better** | ADR-055/061/045/040 vs implicit |
| Uncertainty communication | **Unclear** | ADR-042 caveats designed but not yet present in any draft public artifact; SDC shipped its caution text |
| Results continuity vs predecessor | **Unclear** | Baseline 2050 = 862,723 vs SDC 957,194 (−9.9%); 2025–2028 dip reverses observed PEP trend; Ward sign flip — none yet explained in any public-facing or comparison doc |
| Release governance execution | **Worse (state, not design)** | 6-gate QA checklist far exceeds SDC's process but is entirely unexecuted |

Current public baseline (2026-05-27 run): 799,358 (2025) → 791,127 (2030, below base; trough 786,568 in 2028) → 862,723 (2050) → 876,479 (2055), +9.6% over the horizon.

## Blocking items (must close before "final")

1. **CF-001/ADR-061 is undispositioned while the production config is an un-benchmarked hybrid.** `config/projection_config.yaml` embeds ADR-061 Decisions 1 (25–29 college smoothing/rate-cap extension) and 4 (12-county college list) but omits Decision 3 (extended convergence hold — the identified fix for the 3.44pp under-projection swing; `medium_hold_years` is still 10). ADR-061 is still Proposed; the SOP-003 decision record (`docs/reviews/benchmark_decisions/2026-03-09-m2026r1-vs-m2026.md`) is Draft with no reviewer or verdict; the champion alias still points at m2026. The as-run config matches neither benchmarked champion nor challenger. **Action:** disposition all four decisions explicitly (including the EXP-B blend-factor question) — promote m2026r1 fully via the alias tooling *or* revert D1/D4 to the frozen champion — then rerun production. Do not publish from the hybrid.
2. **No validation/QA artifact postdates ADR-065/066.** All 24 benchmark bundles and the SDC comparison doc evaluate the pre-May-2026 method on the pre-Vintage-2025 base. `docs/methodology_comparison_sdc_2024.md` still claims a ~60K/−6.5% gap vs SDC and "both models agree North Dakota will grow," while the actual public baseline gap is ~94K/−9.9% @2050 with a 2025–2028 decline. **Action:** refresh the comparison doc; execute QA against the final run.
3. **The release QA checklist — the repo's own definition of final — is entirely unexecuted** (~40 items, 6 gates, all unchecked), and no post-ADR-065/066 sanity review exists despite documented repo practice (cf. `2026-02-18-projection-output-sanity-check.md`, `2026-02-23-post-adr054-sanity-check.md`). The new 2025–2028 statewide dip needs an explicit reconciliation showing it is the intended CBO front-loaded migration schedule (f(2025)=0.20 ramp), not a defect. **Action:** dated sanity review + gate-by-gate checklist execution against the locked run.
4. **ADR-042 mandatory caveats are absent from every draft public artifact** (the three numeric caveats and the projection-not-forecast distinction appear in no draft copy or export generator), the banned-language pass has not run, and draft PDF copy retains superseded three-scenario framing. **Action:** rewrite public copy for baseline-only ADR-065 framing with refreshed caveat values from the final run.

Additionally blocking-adjacent: **Ward County** flips from SDC +23% to −13.6% (state's 4th-largest county; Grand Forks similar at −8.7% per ADR-052) with the only mitigation (ADR-052 floor) confined to the now-inactive `high_growth` scenario. Record a written disposition — corrective ADR or an explicit accepted-divergence rationale with public narrative — before release. This is the single number most likely to draw public challenge.

## Are the method changes appropriate and not detrimental?

**Mostly yes — individually evidence-based, at or above typical state-demography practice** (ADR-036 grounded in a multi-state best-practices survey; ADR-057 adds rolling-origin validation absent from SDC; bug-fix ADRs 049/050/054 recorded Implementation Results; ADR-045 reservation recalibration validated favorably ex post; ADR-051 was properly *rejected* when calibration showed it unnecessary). No change was shown to be accuracy-detrimental. But verified concerns:

- **Stacked conservatism (high):** GQ Phase-2 subtraction (3rd-largest sensitivity, 37,084 persons @2050) + college smoothing asymmetry + premature 5-10-5 convergence — which ADR-061 itself identifies as causing "systematic under-projection bias growing with horizon length" — now have CBO downward adjustments (−5% fertility, additive migration reduction) layered on top, while the one identified corrective (Decision 3) is not deployed. Run a sensitivity decomposition isolating GQ fraction / convergence hold / CBO adjustments on the 2050–2055 totals; deploy D3 or document why it's excluded.
- **Honest framing of "at least as good" (high):** the repo's own walk-forward shows the reimplemented SDC method beating m2026 on *state-level* APE beyond 4 years (10yr 15.24% vs 17.10%) and in all four urban/college counties; m2026 wins county MAPE at all horizons, bias fraction (28% vs 41%), and recent-origin state APE (0.32% @4yr). m2026r1 targets exactly the deficits but was never promoted. Frame superiority claims by county accuracy/bias/recent-origin, and record the state-level horizon weakness in the comparison doc.
- **Williams County triple-treatment (medium):** in the college list below the ADR's own 2.5% enrollment threshold (WSC 1.5%) while boom-dampened 0.40 × male 0.80; ADR-061's flagged double-dampening risk has no recorded resolution.
- **EXP-B undispositioned (medium):** blend 0.7 is the only experiment to pass all gates, monotone improvement toward higher blends; production knowingly retains 0.5 with no recorded decision. EXP-C (GQ fraction calibration) never successfully ran — f=1.0 is an untested default.
- **Stale scenario artifacts (medium):** on-disk `restricted_growth`/`high_growth` outputs (2026-02-26, pre-ADR-066) contradict the "deprecated alias = equivalent to baseline" claim (~25K apart by 2045). Regenerate or quarantine with a README.
- **Optics items (low):** ADR-062 tolerance widening is empirically sound (rounding 3σ ≈ 1.07 persons) but sequenced right after EXP-B failed the old gate — disclose the rationale proactively if the framework is externally described. The migration pipeline's 2020 snapshot sources from the SDC 2024 base product — add one sentence of provenance to methodology.md §5a. ADRs 040/043/045/046 lack Implementation Results sections.

## Non-blocking gaps worth closing (raise rigor above "meets")

1. **Naive-method comparison** promised by ADR-063 (carry-forward, linear trend) never executed on real data — runners exist; feed existing walk-forward results through them. (days)
2. **Sensitivity table for the as-published config** — existing sensitivity evidence (fertility ±25% → 4.07pp; convergence 3.44pp; migration +50% → 2.91pp) all predates ADR-065/066. (days)
3. **Components-of-change outputs** — engine persists population only; SDC published a components table. Cross-check projected 2025–2030 components vs PEP observed. (days)
4. **Public age-sex detail** — current spec publishes four broad bands; SDC published full 5-year age-sex tables. Add a detail sheet. (hours)
5. **Track-record disclosure** — SDC disclosed its 2018-vs-Census-2020 error (~0.7% high); no equivalent planned. A step down in candor from the predecessor. (hours)
6. **methodology.md Limitations section + post-ADR-065 cleanup** — §3.5 still says baseline fertility "constant"; §7.1/§9.1 attribute the CBO reduction to restricted_growth; §5h says 20-year convergence against a 30-year horizon; §9.3 ADR table omits 064/066 and lists 061 as Proposed while documenting its features as production method. (hours)
7. **Add a demographic-plausibility gate to the QA checklist** — current gates check format/scope/language but no launch-trend, age-structure, or sex-ratio review. (hours)

## Recommended path to final

1. Disposition CF-001/ADR-061 (all four decisions + EXP-B) → lock config → rerun production.
2. Dated sanity review of the final run (dip reconciliation, Ward/GF disposition, age-structure check).
3. Refresh `methodology_comparison_sdc_2024.md` + methodology.md fixes; mark ADR-017 superseded.
4. Execute the 6-gate QA checklist; embed ADR-042 caveats; regenerate/quarantine stale scenario outputs.
5. Optional but recommended: items 1–5 from the gaps list, especially the track-record paragraph and components cross-check.

---

*Full multi-agent evidence pack (verbatim agent findings, 26 agents, 17/17 findings upheld under adversarial verification) generated 2026-06-10; this document is the distilled record.*

**Execution plan:** `docs/plans/2026-public-projection-release-handoff/finality-remediation-plan.md` (Stages 0–5, dependency-ordered, opened 2026-06-10).
