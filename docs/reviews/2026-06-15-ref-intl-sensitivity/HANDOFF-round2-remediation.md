# Handoff: Round-2 Review Remediation (PUB-2026 ref-intl / ADR-068)

| Attribute | Value |
|-----------|-------|
| **Created** | 2026-06-16 |
| **For** | The next agent/session picking up PUB-2026 publication prep |
| **Source** | The round-2 GPT-5.5 Pro review: [`gpt55pro_round2_output.md`](./gpt55pro_round2_output.md) (verdict **GO**, no model rerun) |
| **Status of numbers** | ~~FINAL — do not rerun the model.~~ **SUPERSEDED 2026-06-16 — see banner below.** |

> **✅ COMPLETED 2026-06-16, with one material change.** This punch list was worked in full (§A/§B/§C/§D).
> While verifying §D's survival-coverage gate, the "numbers final / no rerun" premise **broke**: the
> operative survival table on disk spanned only **2025–2045** (the engine silently fell back to the
> uncorrected static base for 2047–2055), so the published 886,585 @2055 / 90+ 9,971 were biased. Root
> cause: a test (`test_full_pipeline_produces_valid_output`) was overwriting the *production* survival
> table with a horizon-20 (2025–2045) version on every `pytest` run. With the survival table corrected
> to the full 2025–2055 horizon (user-authorized verification rerun), the **final** numbers are
> **898,907 @2055 (+12.45%)** / 90+ **8,172** (2025–2046 unchanged, incl. the 797,298@2027 trough).
> See [`../2026-06-16-corrected-run-qa-verification.md`](../2026-06-16-corrected-run-qa-verification.md)
> and [`ADR-068`](../../governance/adrs/068-ref-intl-numerator-and-open-ended-survival-correction.md)
> (Amendment). The §A–§D items below were completed against the corrected run; numbers quoted in this
> dated handoff (886,585 / 9,971) are retained as the historical record.

## Start here (orientation)

ADR-068 corrected the two confirmed numeric errors (`reference_intl_migration` 3-yr-sum → annual
mean 3,350.33; open-ended 90+ survival → open-interval `T91/T90`) and a production rerun is **done**.
An independent round-2 GPT-5.5 Pro review of the corrected product (2026-06-16, live+background,
$17.20) **confirmed both fixes are implemented in code/config (not relabeled)** and returned a clear
**GO: proceed to publication artifact regeneration + QA gates; no further model rerun.** Your job is
to clear the **documentation/labeling blockers** it flagged, **verify** a couple of code-robustness
items, **disclose** the accepted limitations, **then** regenerate the public artifacts and run QA.

Read this file + the review output first. Everything below is derived from that review.

## The two rules

1. **Do NOT rerun the model or change the projection numbers.** The review verified the numbers are
   sound. If you think a number is wrong, you are misreading — re-read the review §1–2 first.
2. **Clear the text/labeling blockers (§A) BEFORE regenerating artifacts (§D)** — otherwise you
   regenerate the workbook/CSV/PDF/docx/pyramid twice.

## Authoritative numbers (assert these everywhere)

| Quantity | Value |
|----------|-------|
| 2025 base | **799,358** (= sum of 53 county `POPESTIMATE2025`) |
| Near-term trough | **797,298 in 2027** (−0.26%) |
| 2055 | **886,585.25** (+10.9%) |
| 90+ population @2055 | **9,971** |
| State vs Σ counties | **0.0 residual** (53 unique county files) |
| config sha256(16) | `cca42fb42be76680` |
| Survival table coverage | must span **2025–2055** |

## Source artifacts

- Review output (the punch list, with citations): [`gpt55pro_round2_output.md`](./gpt55pro_round2_output.md)
- Evidence the reviewer audited against: [`round2/`](./round2/) `evidence_*.csv`
- Decision record: [`ADR-068`](../../governance/adrs/068-ref-intl-numerator-and-open-ended-survival-correction.md)
- [`docs/methodology.md`](../../methodology.md), [`config/projection_config.yaml`](../../../config/projection_config.yaml), [`final-run-metadata.md`](../../plans/2026-public-projection-release-handoff/final-run-metadata.md)
- If you want to run another review: [`docs/guides/gpt-5.5-pro-api-reference.md`](../../guides/gpt-5.5-pro-api-reference.md) — **use live+background, NOT batch** (see that doc for the $85 lesson). Runner: [`run_gpt55pro_round2_batch.py`](./run_gpt55pro_round2_batch.py) `--live`.

---

## §A — MUST-FIX before public release (documentation/labeling; no rerun)

**A1. Methodology §4.3 still states the OLD (pre-ADR-068) 90+ formula.** [`methodology.md:469-479`](../../methodology.md#L469-L479)
(and the related sentence at line ~997) give `S(90+) = T_{91} / (T_{90} + L_{90}/2)`, fallback 0.65,
"typical values 0.60–0.70." The **implemented** correction (ADR-068, `apply_open_ended_survival_correction`)
is `T91/T90`, giving ~**0.778 (M) / 0.806 (F)** at 2025. → Rewrite §4.3 to match the operative formula
and values. *Acceptance:* §4.3 formula and quoted values equal what `evidence_survival_85plus.csv`
shows at age 90.

**A2. Fertility provenance mismatch.** Methodology says ASFR is "pooled CDC WONDER 2020–2023," but the
operative table (`evidence_fertility_rates_FULL.csv`) carries `year=2023` only. → Either prove via
metadata that the 2023 label *is* the 2020–2023 pooled estimate, or change the text to match the table.
Truthful provenance only. *No rerun if numbers are intentionally final.*

**A3. Contradictory mortality provenance docs.** config `life_table_year: 2023` vs methodology §4.2
(NVSR 74-12 / 2022) vs `mortality_improvement.py` docstring (ND CDC 2020 / SDC 2024) vs ADR-068
(CDC 2023 `Tx` for the 90+ correction). These can coexist only if **sequenced**: static base table →
NP2023 time-varying operative table → 2023 open-age correction. → Make methodology §4.2 + the config
comment + the docstring tell one consistent, sequenced story.

**A4. GQ components are household-basis — must be labeled.** Projected deaths (~5k/yr) are HH-basis and
exclude GQ turnover; PEP total deaths are ~7k. Unlabeled, users read this as a mortality error. → Any
public components-of-change table (CSV/workbook/PDF) must state births/deaths/net-migration are
household-basis. (methodology §10 already discusses GQ; ensure the public *outputs* carry the label.)

**A5. Purge stale numbers from PUBLIC / current-status text** (replace with corrected 886,585 @2055,
797,298 @2027 trough). Stale values to grep: `889,017`, `787,382`, `876,479`, `882,146`, old troughs.
- **Fix (public-facing / current-status):** [`docs/methodology_comparison_sdc_2024.md`](../../methodology_comparison_sdc_2024.md),
  [`docs/plans/2026-public-projection-release-handoff/marketing-ready/`](../../plans/2026-public-projection-release-handoff/marketing-ready/),
  any marketing-docx markdown source, [`2026-06-13-locked-run-sanity-check.md`](../2026-06-13-locked-run-sanity-check.md),
  [`2026-06-13-release-qa-signoff.md`](../2026-06-13-release-qa-signoff.md).
- **Do NOT rewrite history:** dated ADRs (`055`, `067`) and dated review memos are historical records —
  leave their numbers, optionally add a one-line "superseded by ADR-068" banner. `final-run-metadata.md`
  already has its superseded banner.

---

## §B — VERIFY (possible real bug; check before regenerating)

**B1. (sharpest) N1: county `_summary.csv` may be household-only.** In
[`multi_geography.py:160-189`](../../../cohort_projections/geographic/multi_geography.py#L160-L189):
`projection_results` gets GQ added back (`_add_gq`, lines ~178-182), but `get_projection_summary()`
(line ~189) returns summaries accumulated **inside the engine before GQ was re-added**. So county
**parquet** outputs are GQ-inclusive while county **`_summary.csv`** may be HH-only. → Determine which
**public** exporters consume the summary CSVs vs the parquet detail (start with
[`scripts/exports/build_provisional_workbook.py`](../../../scripts/exports/build_provisional_workbook.py)
and `build_detail_workbooks.py`). If any public artifact reads `_summary.csv`, recompute summaries from
the GQ-inclusive `projection_results`, or point exporters at the parquet. *Acceptance:* every public
geography total equals the GQ-inclusive parquet sum.

**B2. N2: aggregation does not assert exactly 53 unique county files.**
`aggregate_county_results_to_state()` globs `nd_county_*_projection_*.parquet`. A stale/partial rerun
could be swept in. → Add a QA assertion: exactly 53 unique county FIPS, one current file each.

**B3. Aggregation tolerance enforcement.** config is `0.000001`, but
`multi_geography.validate_aggregation()` still has a function-default `tolerance=0.01`. State is built
bottom-up so this isn't a numeric blocker, but → run an explicit **exact** residual check on the
regenerated files (expect 0.0 at 2025 and 2055).

---

## §C — DISCLOSE (accepted for the 2026 vintage; add to the limitations section, do not fix now)

| Item | Disclosure |
|------|-----------|
| `blend_threshold` config key not wired (build uses CLI-default 5000) | State the county distribution file is prebuilt with a 5,000 threshold; wire the config key next vintage. |
| Flat 5yr→1yr ASFR expansion | "ASFR constant within each 5-year maternal age band" (not Sprague/graduated). |
| Newborns not exposed to infant mortality in birth year | Minor (~tens of persons/yr; infant survival ~0.998). Note it. |
| College smoothing updates `migration_rate` not `net_migration` diagnostics | Don't publish post-smoothing `net_migration` diagnostics as final, or recompute them. |
| CBO decrement uniform across age/sex/race, applied after the rate cap | Disclosed simplification; immigrant-age-profile allocation next vintage. |
| **N3: migration is also race-flat** (not just mortality) | Disclose alongside the ADR-068 D3 race-flat-mortality caveat: race outputs use race-specific fertility + base composition, but migration and operative mortality are race-flat. |
| N5: mortality metadata omits `cdc_lifetables_2023_combined.csv` | Add that file + the age-90 target values to `mortality_improvement_metadata.json` for auditability. |

---

## §D — THEN regenerate artifacts + run QA (the deferred publication work)

Only after §A (and §B verification) are clear:

1. **Regenerate against the corrected run:** public Excel workbook, public CSV, public PDF copy, the six
   marketing `.docx` (edit the markdown source + rerun `scripts/exports/build_marketing_docx.py` — never
   hand-edit `.docx`), and the pyramid explorer (`scripts/exports/build_pyramid_explorer.py`).
2. **Re-run the 6 release-QA gates** against the corrected run (prior sign-off:
   [`2026-06-13-release-qa-signoff.md`](../2026-06-13-release-qa-signoff.md) — was against the superseded run).
3. **Final QA assertions:** 53 unique county files; state = Σ counties (0.0); base = 799,358;
   2055 = 886,585.25; 90+ @2055 = 9,971; survival table spans 2025–2055; no stale numbers in public text.

---

## Git / data state at handoff

- **4 doc commits this session, on branch `docs/ref-intl-sum-vs-average-finding`, NOT yet pushed:**
  `e9206dc` (API ref doc + runner), `4fef906` (doc refinements), `7276c53` (streaming section),
  `71ececd` (round-2 review + evidence). → Push to PR #25 (or the user will).
- The 1.4 MB assembled-package blob (`round2_batch_input.jsonl`) is gitignored; evidence CSVs are committed.
- Data dirs `data/projections/sensitivity_refintl_corrected_20260615/` and the round-2 evidence are local;
  `./scripts/bisync.sh` if you need them on another machine.
- Agent memory updated: `pub-2026-ref-intl-sum-vs-average.md` (round-2 verdict + this punch list),
  `gpt-5.5-pro-api-usage.md` (the API lesson).
