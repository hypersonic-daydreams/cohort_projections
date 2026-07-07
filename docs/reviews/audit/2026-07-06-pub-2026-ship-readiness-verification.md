# PUB-2026 Ship-Readiness Verification and Errata Disposition

| Field | Value |
|-------|-------|
| **Date** | 2026-07-06 |
| **Author** | Claude Code (Fable 5), owner-requested session review |
| **Scope** | Owner asked whether the PUB-2026 marketing draft package is ready to ship, whether it reads as the next State Data Center edition after the 2024 report, and whether the 2024 report is the right content baseline. This review re-verified the same-day [draft-materials assessment](2026-07-06-pub-2026-draft-materials-assessment.md) against the release machine's working tree, reviewed the previously unseen 2026-06-18 uncommitted work, and executed the resulting errata fixes. |
| **Verification basis** | Independent pandas recounts against the locked public CSV (`marketing-ready/drafts/PUB-2026 Draft Public Dataset.csv`, 1,922 rows, config sha16 `a6e0bfbc2d70be85`); visual inspection of all draft PNGs and the five new explainer figures; openpyxl inspection of the public workbook; python-docx extraction of the six marketing `.docx`; full read of the 2024 SDC PDF (16 pp.); source reads of `visualizations.py`, `build_public_draft_package.py`, `build_marketing_docx.py`, `build_explainer_figures.py`; git archaeology across `master`, `origin/master`, and the hygiene branch. Five parallel verification agents plus inline spot-checks. |
| **Relationship to prior review** | Companion to [2026-07-06-pub-2026-draft-materials-assessment.md](2026-07-06-pub-2026-draft-materials-assessment.md) (same day, other machine). That review established findings F1–F7 and improvements I1–I9 but could not inspect the gitignored workbook, the reference PDF, or the production parquets (absent on its machine), and never saw the 2026-06-18 uncommitted work (present only in this tree). This review closes those gaps and records dispositions. |
| **Status** | Findings verified; errata fixes executed this session (see §5). Remaining owner items in §6. |

---

## 1. Verdict

**Not ready to ship as reviewed; ready after roughly one day of repo-side work, none of which touches the model or the locked numbers.** The assessment's verdict stands: the package is already substantially better than "the 2024 template with new numbers," and the structure deliberately mirrors the 2024 report. What blocked shipping was (a) two factual errors in public-facing copy, (b) labeling defects in public artifacts, (c) stale front-door documentation, and (d) one strategic gap — the draft nowhere acknowledged the 2024 edition it succeeds, which is precisely the owner's continuity concern.

## 2. Assessment findings re-verified on the release machine

All re-checked independently in this tree; every one **confirmed**:

- **F1 — declining-county count is 36, not 37.** Recount from the locked CSV: 36 counties have 2055 < 2025; Golden Valley flipped to +0.56% (1,808 → 1,818) when the 2026-06-16 survival-horizon amendment lifted 2047–2055 values. Borderline set: Eddy −0.52% (declining), Golden Valley +0.56%, Sheridan +1.86%, McLean +2.09%. The stale "37" appeared as current fact in `draft-public-pdf-copy.md` (×3, incl. required ADR-042 caveat 2), `how-these-projections-work.md`, both narrative `.docx`, and the new `figure-drafts.md` fig5 caption. The production `countys_summary.csv` independently gives 36/17.
- **F2 — the "within about 0.7%" track-record claim is not printable as-is.** It traces to the 2024 PDF's Note of Caution ("793,537, about 5,500 more than the 2020 decennial census found"), whose two halves contradict each other: 793,537 − 779,094 = 14,443 (~1.85%), not ~5,500 (~0.7%). One of the two 2024-printed values is wrong and the repo cannot resolve which. Compounding it, the sentence called 2018 "the State Data Center's previous projection" — the previous edition is 2024 (see §3).
- **F3 — artifact labeling defects confirmed, plus one new:** all six pyramid PNGs label the open-ended 90+ cohort "90-94" (`visualizations.py:219`, no open-ended case); the workbook's two chart-pyramid sheets each emit **two rows labeled "85+"** (the second is the 90+ cohort — 2,882+5,291 = 8,173 ≈ locked 8,172 @2055; `build_public_draft_package.py` bins without `.clip(upper=85)`, unlike the State Age-Sex Detail sheet which clips correctly). **New:** the county pyramid PNGs additionally render float age labels ("0.0-4.0" … "90.0-94.0") because county parquet ages are float dtype and the f-string formats them verbatim.
- **F4 — machine question resolved: this WSL machine is the "primary".** The corrected stage-02 grid exists here in full (state parquet sums re-verified equal to the locked CSV: 799,358 / 797,298 @2027 / 883,225 @2050 / 898,907 @2055) along with the QA'd workbook, reference PDF, and `.docx`. Google Drive holds only the stale 2025_2045 vintage; the other machine holds nothing newer; and rclone/bisync was permanently disabled per IT policy (commit `79ec8e5`) with no sanctioned replacement. **This machine is therefore the sole holder of the release outputs and the only machine that can assemble the packet.** A manual IT-sanctioned backup of `data/projections/baseline/**` and `marketing-ready/**` is the single most urgent action outside this repo (owner will run it).
- **F5 — stale front-door docs confirmed:** the handoff folder's top `README.md` (2026-05-27) still framed the numbers as provisional pending CF-001; `provisional-number-snapshot.md` is a self-declared-stale husk still routed to by the README table; `decision-log.md` ended at 2026-05-27; `finality-remediation-plan.md` cites superseded figures in dated log entries with no ADR-068 pointer.
- **Verified clean:** state headline set, trough year/value, Cass/Williams/Burleigh/Ward/Grand Forks callouts, top-3 share of gross gains 78.1%, CSV row count and scope, and all superseded-number mentions outside the above sit inside explicit ADR-068 "superseded" banners.

## 3. New findings beyond the assessment

### 3a. Series-continuity findings (from a full read of the 2024 PDF against the draft)

The owner's framing — "the next set of projections produced by the State Data Center" — surfaced gaps the structural map alone did not:

1. **"Previous (2018) projection" misidentifies the series.** The accuracy section skipped the 2024 edition entirely, presenting 2018 as the SDC's previous projection. For a successor document, this reads as erasing the predecessor's edition.
2. **Title convention breaks.** 2024: "*2024 North Dakota State Data Center Population Projections of the State, Regions, and Counties*", with a "Prepared February 6, 2024" date line. The 2026 draft cover dropped the vintage year, the SDC-in-title convention, the scope subtitle, and the date line — despite identical scope.
3. **A silently reversed 2024 claim.** 2024 (p. 7): Ward "will move ahead of Grand Forks around 2040 and continue to be the state's third largest county thereafter" (Ward printed at 85,975 @2050). The 2026 baseline projects Ward −12% to ~60,000, never overtaking Grand Forks (~72,000). Nothing bridged the reversal for a 2024 reader.
4. **The 2050 gap needed a public answer:** 957,194 (2024) vs 883,225 (2026 locked) = −73,969 (−7.7%). The 2024 report itself set the in-series precedent of discussing prior vintages (its Note of Caution benchmarks both the 2018 SDC projections and the 2005 Census Bureau projections — the latter a memorable humility anecdote the draft also dropped).
5. Minor dropped conventions catalogued for the record: under-5 age band as its own row, East/West half-of-state framing, named-staff byline and mailing address in the contact block, three-snapshot pyramids (2030/2040/2050) reduced to a 2025-vs-2055 pair (the pair is judged better), "Prepared [date]" cover line.

**Disposition:** items 1–4 fixed this session via the I1 comparison box, the accuracy-section rewrite, and the cover/title fix (§5). Item 5's conventions are deliberate scope choices left to marketing except the date line, which the cover now carries.

### 3b. The 2026-06-18 uncommitted work (first review)

The tree carried never-reviewed work: a prose-polish diff to the public explainer, `figure-drafts.md` (a five-figure selection menu for the explainer), five rendered figure PNGs, and `scripts/exports/build_explainer_figures.py`. Verdict: **coherent, near-finished, worth keeping — committed this session after three fixes.**

- All data-bearing figure numbers verify against the locked run (fig3: 799,358 / 797,298 @2027 / 898,907 @2055 +12%; fig5: 17 grow / 36 decline / top-3 ≈78% of gains; fig4 uses real Williams-vs-Ward 2025 parquet data; hardcoded top-3 FIPS verified correct).
- `build_explainer_figures.py` clips the pyramid top bracket correctly (`np.minimum((age // 5) * 5, 85)`) — the pattern the two defective code paths now adopt.
- Fixes applied at commit: 7 ruff errors (N806 ×4, C408, B905 ×2); geojson source switched from the stale `high_growth` export to the equivalent baseline geojson; fig5 caption/mockup/build-note "37" reconciled to 36 (the doc's own build notes had flagged the discrepancy — "flagged here, not changed" — and the rendered PNG already computed 36 live, so the caption contradicted its own figure); fig5 hard-coded legend "≈ ¾ of gains" aligned with the computed ≈78% wording.
- The explainer diff renames a section to "What the baseline assumes"; `figure-drafts.md` references the new name, so the two land in one commit.

### 3c. Axis re-assessment (downgrades assessment I7)

Inspected at full resolution, `chart_state_baseline_line.png` and fig3 use truncated but honest y-axes: because the same axis must span the +12.45% rise, the −0.26% trough occupies ~2% of plot height and reads as a shallow flattening, not a plunge; fig3 additionally annotates the dip as a feature of the migration assumption. I7 is downgraded from "fix" to a handoff note asking marketing to preserve the annotation (recorded in the marketing-ready README).

## 4. Trajectory reference (locked CSV, verified this session)

| Year | SDC 2024 | 2026 locked baseline | Gap |
|------|---------:|---------------------:|----:|
| 2025 | 796,989 | 799,358 | +2,369 |
| 2030 | 831,543 | 804,657 | −26,886 |
| 2040 | 890,424 | 848,259 | −42,165 |
| 2050 | 957,194 | 883,225 | **−73,969 (−7.7%)** |
| 2055 | — | 898,907 | (new horizon) |

## 5. Dispositions executed this session (all on branch `chore/repo-hygiene-2026-06`)

1. **Merged `origin/master`** into the hygiene branch (one tracker conflict resolved: hygiene relinks + origin's new PRIMARY-RECOVERY / PUB-2026-ERRATA rows both kept); moved the assessment into `docs/reviews/audit/` per the regroup convention and relinked inbound references.
2. **Committed the 2026-06-18 work** with the three fixes above.
3. **F3 code fixes:** `visualizations.py` labels the top bracket "90+" and casts ages to int (fixes the float labels); `build_public_draft_package.py` chart-pyramid binning clips to 85+ matching the detail sheet.
4. **F1 fixed everywhere** (PDF copy ×3, explainer, figure-drafts): "36 of 53", with the caveat phrasing hardened to "36 of 53 — about two-thirds" so a future ±1 flip cannot silently invalidate the rhetoric.
5. **F2 fixed:** the 0.7% sentence replaced with wording sourced to what the 2024 edition actually printed, attributed rather than asserted, plus the model's own in-repo backtest figures; "previous (2018)" corrected to name the 2024 edition as the predecessor.
6. **I1 added:** "How these projections compare with the 2024 edition" box in the PDF copy (short table from §4 + the assumption drivers: PEP Vintage 2025 base, CBO current-policy migration, −5% fertility, county corrections that follow observed data) with the natural-increase-vs-migration mirror-image framing; storyboard updated. Cover restored to the series title convention with vintage year, scope subtitle, and date line.
7. **I2 added:** components-of-change exhibit (state natural increase vs. net migration by period, household-basis deaths labeled per the 2026-06-13 sanity-check note) generated from the persisted components parquet into the draft package + storyboard + PDF copy.
8. **I4 resolved:** the public explainer joins the marketing packet as a seventh generated `.docx` and is listed in the What-To-Send table; its hand-authored numbers join the sync check.
9. **I6 done:** `_check_prose_sync` extended beyond its three tokens to cover the declining-county count, the five county callouts, and the top-3 gains share; release checklist now runs the generator with `--strict`.
10. **F5 done:** front-door README rewritten to the locked state; provisional snapshot and intake brief marked historical; decision log appended (CF-001 disposition, ADR-068 correction, this errata pass); finality-remediation plan given an ADR-068 pointer.
11. **Decisions recorded:** F4-RESYNC consciously deferred for the 2026 release (stale F4 decomposition figures remain caveated in `methodology_comparison_sdc_2024.md` §4.2; re-run would clobber production `data/processed/` for a non-public internal figure); F7 round-2 external re-review deferred (round-3/round-4 reviews of PRs #25–27 already re-verified the corrected numbers from source). Readiness checklist §3 bisync step rewritten to assemble-on-this-machine + manual sanctioned backup; stale "Merge PR #27" box ticked (merged 2026-06-16).
12. **Regenerated the package** (`build_public_draft_package.py`, then `build_marketing_docx.py --strict`) after archiving the pre-regeneration artifacts to `data_archives/marketing-ready-2026-06-16-pre-errata/` (nothing overwritten without a preserved copy; git history additionally preserves all tracked artifacts). Verified post-regeneration: locked CSV byte-identical, pyramid labels correct, single 85+ row per chart-pyramid sheet, strict sync check green.

## 6. Remaining items (owner / marketing)

- **Owner:** manual IT-sanctioned backup of `data/projections/baseline/**` + `marketing-ready/**` (urgent — sole-copy machine); PRIMARY-RECOVERY disposition per runbook (does not gate the handoff); review the I1 comparison-box prose (public-facing, SOP-005 voice — drafted for owner sign-off); fill contact/download placeholders remains with marketing.
- **Marketing (unchanged from checklists):** Gate 5 layout/accessibility, placeholder fill, re-verify every number at rendered-PDF layout, final delivery items (Gate 6).
- **Explicitly not done:** I3 (8-region key-year table) and I5 (AI-tells prose pass on the PDF copy) remain open options; I9 future-vintage ideas unchanged.

---

*Indexed from `DEVELOPMENT_TRACKER.md` (PUB-2026-ERRATA row) and `docs/reviews/README.md`. Companion: [2026-07-06-pub-2026-draft-materials-assessment.md](2026-07-06-pub-2026-draft-materials-assessment.md).*
