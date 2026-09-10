# PUB-2026 Draft-Materials Assessment — 2026 Public Report vs. 2024 SDC Report

| Field | Value |
|-------|-------|
| **Date** | 2026-07-06 |
| **Author** | Claude Code (Fable 5), owner-requested session review |
| **Scope** | Assess the PUB-2026 draft materials for the public 2026 report against the 2024 SDC publication (`data/raw/ND Population Projections.pdf`, 16 pp.): structural alignment, substantive differences, defects, and improvements worth making before marketing handoff. |
| **Verification basis** | Locked public CSV (`marketing-ready/drafts/PUB-2026 Draft Public Dataset.csv`, 1,922 rows, ADR-068-corrected run, config sha16 `a6e0bfbc2d70be85` per `release-readiness-checklist.md` / `final-run-metadata.md` — headline set 799,358 → 797,298 @2027 → 898,907 @2055); independent Python recounts against that CSV; full read of the 2024 PDF; all handoff-folder documents; marketing `.docx` text extraction; draft chart/pyramid PNGs; `build_public_draft_package.py` / `build_marketing_docx.py` / `output/visualizations.py`; QA record (`2026-06-13-release-qa-signoff.md`, `2026-06-16-corrected-run-qa-verification.md`, `2026-06-17-pub-2026-release-readiness-status.md`); the six 2026-06-18 prose reviews; SOP-005. |
| **Status** | Advisory — findings pending owner disposition. No repo files were modified by this review other than this document and the tracker row that indexes it. |
| **Machine context** | Review ran on the machine **without** the bisynced data tree or the gitignored workbook/reference-PDF artifacts (see F4) and without the `sdc_2024_replication` sibling repo. Claims that require production parquets (median age, 91%-international migration share) are flagged as unverified-here rather than checked. |

---

## 1. Verdict

The draft package is already substantially better than "the 2024 template with new numbers." The structure deliberately and sensibly mirrors the 2024 report; the numbers are locked, reproduced from scratch to zero diff, and unusually well-QA'd; and the plain-language methods explainer is a genuine addition the 2024 release had no counterpart for.

Against that strong baseline, this review found:

- **Two factual errors in public-facing copy that survived the QA gates** (F1: stale "37 of 53" declining-county count — the locked CSV says 36; F2: an unsourced and probably ~2.6×-too-favorable "0.7%" track-record claim). Both are hand-authored figures outside the `_check_prose_sync` tripwire's three tokens.
- **Two labeling defects in public artifacts** (F3: pyramid "90-94" top bracket; duplicate "85+" rows in the workbook chart-pyramid sheets).
- **One packaging gap** (F4: the workbook and bundled 2024 PDF are gitignored and absent on this machine; production parquets not yet bisynced here).
- **Stale front-door documentation** in the handoff folder that contradicts the current locked state (F5).
- The two known open gates (F6: F4-RESYNC decision; F7: deferred round-2 external re-review).

And one significant content gap: **the draft never addresses the 2024 projections**, even though the 2050 statewide number came down by ~74,000 (−7.7%) and the 2024 report itself set the precedent of discussing prior vintages in public (§3, I1).

---

## 2. Structural alignment with the 2024 report

The 2024 publication (prepared 2024-02-06; InDesign; 16 letter pages) contains: cover, TOC, executive summary, statewide chart 2020–2050 with "A Note of Caution," a one-page methodology, a components-of-change table, regional map/bars/narrative, demographic-makeup narratives and table, state/regions summary (West–East bars + 8-region table), a two-page 53-county table at 5-year steps, three pages of age-sex composition tables, pyramids for 2030/2040/2050, and a contact back cover.

Mapping to the 2026 storyboard (`pdf-content-outline.md`):

| 2024 report | 2026 plan | Assessment |
|---|---|---|
| Cover, TOC, executive summary | Same (storyboard pp. 1–3) | Direct match |
| Statewide chart + "A Note of Caution" (p. 4) | Line chart 2025–2055 with four ADR-042 caveats placed **at the first statewide exhibit** (pp. 3–5) | Improvement — caution is integrated rather than sequestered |
| Methodology, one dense page (p. 5) | Plain-language method box (p. 11) + standalone explainer (`how-these-projections-work.md`) | Improvement |
| **Components of change table (p. 6)** | **Absent from storyboard** | Regression — see I2 |
| Region map + bars + narrative (p. 7); 8-region key-year table (p. 9) | Map + bars (p. 7); **no region table in the PDF** | Partial regression — see I3 |
| Demographic makeup narratives + table (p. 8) | Age-group trend chart + narrative (p. 9) | Match |
| County table at 5-year steps (pp. 10–11) | County appendix at key years 2025–2055 (pp. 13–16) | Match |
| Age-sex composition tables (pp. 12–14) | Moved to workbook `State Age-Sex Detail` sheet | Right call — that is what the downloads are for |
| Pyramids 2030/2040/2050 (p. 15) | 2025-vs-2055 pyramid comparison (p. 10) | Improvement — a comparison reads better than three snapshots |
| Contact back cover; **no data downloads offered** | Contact + one consolidated Excel + one consolidated CSV | Clear improvement |

Where the 2026 release is already a different class of product:

- Single-year-of-age cohorts (91 ages × 2 sexes × 6 race/ethnicity = 1,092 cohorts per county) vs. 5-year groups; annual output vs. 5-year steps; PEP Vintage 2025 base (ADR-066); bottom-up state totals that reconcile exactly (verified: max abs diff ≈ 2.3e-10).
- Language discipline: the 2024 caution opens with "an attempt to provide the **most likely outcome**" — phrasing ADR-042 now bans. The 2026 copy's projection-not-forecast framing is consistently applied and QA-gated.
- QA apparatus: Gate 1b plausibility review, adversarial sign-off, from-scratch reproduction to 0 diff across all 33,852 grid cells.

**A note on the 2024 PDF's reliability as a source.** It should be treated as a design/pacing reference only (as `2024-sdc-pdf-review.md` already says) — but also **not as a factual source**, because it contains internal inconsistencies:

1. Its executive summary prints **957,124** for 2050 while its own p. 4 chart and pp. 11/14 tables print **957,194**.
2. Its Note of Caution says the 2018 projection put Census 2020 at **793,537**, "about 5,500 more" than the count — but 793,537 − 779,094 = **14,443 (~1.85%)**, so the sentence contradicts itself (see F2).

Both are exactly the class of error the PUB-2026 reconcile-to-downloads and prose-sync gates exist to prevent; they justify re-verifying every number at rendered-PDF layout (already in Gate 3/5 as re-verify-at-layout).

---

## 3. Substantive trajectory difference (computed from the locked CSV)

| Year | SDC 2024 | 2026 locked baseline | Gap |
|------|---------:|---------------------:|----:|
| 2025 | 796,989 | 799,358 | +2,369 |
| 2030 | 831,543 | 804,657 | −26,886 |
| 2035 | 865,397 | 826,051 | −39,346 |
| 2040 | 890,424 | 848,259 | −42,165 |
| 2045 | 925,101 | 866,590 | −58,511 |
| 2050 | 957,194 | 883,225 | **−73,969 (−7.7%)** |
| 2055 | — | 898,907 | (new horizon) |

This matches `docs/methodology_comparison_sdc_2024.md` (internal/technical). County-level: Ward is the largest divergence (2050: 85,975 SDC vs ~61.5k ours); Williams is the only county where 2026 runs **above** SDC 2024.

The public consequence: users of the 2024 numbers — legislators, agencies, media — will ask why 2050 dropped ~74k, why Ward flipped from +23% to −12%, and why the line now dips before growing. The draft explains the dip well but says nothing about the prior edition. There is also a clean narrative symmetry available: the 2024 projection had natural change going **negative** by 2045–2050 (−5,526) with migration (+37,624/period) carrying all growth; the 2026 projection is the mirror image — migration moderated under the CBO path, natural increase carrying the long run.

---

## 4. Findings — fix before handoff

Ordered by public-facing risk.

### F1 — "37 of 53 counties decline" is stale; the locked CSV says **36**

Independent recount from the locked public CSV: **36** counties have 2055 < 2025. Borderline counties: Eddy −0.52% (declining), **Golden Valley +0.56%**, Sheridan +1.86%, McLean +2.09%. Golden Valley (1,808 → 1,818) is almost certainly the county that flipped when the 2026-06-16 survival-horizon amendment lifted 2047–2055 values; "37" appears to be a carryover from the pre-amendment run.

Locations of the stale count (all public-facing):

- `draft-public-pdf-copy.md:43` (executive summary: "Thirty-seven of North Dakota's 53 counties…")
- `draft-public-pdf-copy.md:59` (required caveat 2: "37 of 53 counties are projected to decline")
- `draft-public-pdf-copy.md:184` (Cautions: "while 37 of 53 counties decline")
- `how-these-projections-work.md:161` ("**37 of the 53 counties are projected to decline.**")

Why it escaped: `_check_prose_sync()` (`scripts/exports/build_marketing_docx.py:313`) checks only three tokens — 2055 state total, trough year, trough value. The declining-county count is hand-authored and unguarded.

**Remedy:** correct to 36 in all four places; regenerate the `.docx` set. Given Golden Valley sits ~0.6% from flipping again on any future rerun, consider robust phrasing in at least the caveat ("about two-thirds of counties," or "36 of 53" plus the count added to the sync-check tokens — see I6).

### F2 — The "came within about 0.7% of the 2020 census" track-record claim is unsourced; the printed numbers imply ~1.85%

Locations: `draft-public-pdf-copy.md:100` and `how-these-projections-work.md:175` (the "How far should you trust it?" / "How accurate have past projections been?" sections).

Provenance (searched repo-wide): **no tracked document computes or verifies 0.7%.** The figure inherits the 2024 PDF's own "about 5,500 more" characterization (5,500 / ~788k ≈ 0.7%), but the same 2024 sentence prints the 2018 projection as **793,537**, and 793,537 − 779,094 = **14,443 = ~1.85%**. One of the two 2024-printed values is wrong, and the repo contains nothing that resolves which. The `sdc_2024_replication` folder on this machine holds only scripts; the 2018 publication is not in the repo.

This sits inside a section whose entire purpose is accuracy-about-accuracy, and marketing is bundling the 2024 PDF as the design reference — anyone can do the subtraction.

**Remedy, best first:** (a) verify against the actual 2018 SDC publication and print whatever is true; (b) failing that, restate consistently with the printed 793,537 ("within about 2%"); (c) drop the 2018 anchor and rely on the model's own in-repo, verified backtests. Do not publish 0.7% unverified. Add the final figure to the sync-check/QA token list.

### F3 — Pyramid age-bracket labeling defects in public artifacts

1. **All six pyramid PNGs label the open-ended 90+ cohort "90-94."** `cohort_projections/output/visualizations.py:219` builds labels as `f"{age}-{age + age_group_size - 1}"` with no open-ended case; the engine's top cohort is 90+ (the 2055 bar ≈ 8.2k total matches the locked 90+ = 8,172). Marketing will copy axis labels verbatim.
2. **The workbook chart-pyramid sheets emit two rows both labeled "85+".** `build_public_draft_package.py:960` bins without clipping (`(age // 5) * 5`), then `_label()` (lines 962–965) maps both start=85 and start=90 to "85+", while the groupby/pivot keys on `age_group_start` — producing separate 85–89 and 90+ rows with identical labels. The `State Age-Sex Detail` sheet does it correctly via `.clip(upper=85)` (line 780).

**Remedy:** label the top bin "90+" (or fold 90+ into "85+" to match the detail sheet and the 2024 report's brackets) in both code paths; regenerate PNGs and workbook.

### F4 — This machine cannot assemble the handoff package as-is

`marketing-ready/README.md` lists the workbook and the bundled 2024 PDF as deliverables, but both are **gitignored** (`.gitignore:163` `*.xlsx`; `.gitignore:192` `docs/**/*.pdf`) and absent here; the production parquets under `data/projections/baseline/` have not bisynced to this machine either. This matches the unchecked Section 3 of `release-readiness-checklist.md`.

**Remedy:** `./scripts/bisync.sh`, then `python scripts/exports/build_public_draft_package.py` (it rebuilds the workbook, copies the reference PDF, embeds run/config/commit provenance, and warns if the 2024 PDF is missing), before assembling anything to send. Alternatively assemble from the machine that already has the artifacts.

**Update 2026-07-06 (verified live):** bisync from this machine cannot deliver the outputs — a dry-run converged with "No changes found," and the 08:15 listing snapshots show **Google Drive itself lacks the corrected run** (zero `2025_2055_baseline` / `baseline/state/` entries on either side; both sides hold the same 265 stale `2025_2045` files from December). The June 13/16 production outputs exist only on the primary machine, which has not bisynced since the final rerun. Required order: **primary machine runs `./scripts/bisync.sh` to push**, then this machine pulls, then regenerate the package here — or simply assemble and send the packet from the primary machine, which already holds the QA'd workbook and reference PDF (no regeneration risk). When pushing from the primary, glance at whether the stale `2025_2045` county files were deleted there — if so the deletions will propagate (desirable); if not, the mixed-horizon hazard from the 2026-05-27 decision log persists on all machines.

**Second update, 2026-07-06 (primary-machine incident):** the primary machine subsequently suffered a data-loss incident (~85% recovery odds). Full asset inventory, contingency regeneration runbook, and systemic follow-ups: [`docs/plans/primary-machine-data-loss-regeneration-runbook.md`](../../plans/primary-machine-data-loss-regeneration-runbook.md) (tracker row PRIMARY-RECOVERY). Key finding: all locked stage-01 pipeline inputs are safe on Drive + the WSL machine; only the stage-02 projection grid and the per-machine `data/analysis` evidence archives are primary-only, and the release itself is fully reconstructible with byte-level proof against the git-tracked locked CSV.

### F5 — Stale front-door documentation in the handoff folder

- `docs/plans/2026-public-projection-release-handoff/README.md` still says assets were "regenerated on 2026-05-27," that current numbers are "draft layout values… stale under ADR-065," and that finals await CF-001 — all superseded 2026-06-13/16. It directly contradicts `marketing-ready/README.md` (current, locked-final).
- `provisional-number-snapshot.md` is a "stale — do not use" husk that the front-door README's table still routes marketing to for "Rounded draft numbers for layout."
- `decision-log.md` ends 2026-05-27 (no CF-001 disposition/ADR-068 entries), and `marketing-intake-brief.md` is a historical meeting artifact that reads as current.

**Remedy:** ~20-minute refresh of the front-door README to the locked state; mark the snapshot and intake brief as superseded/historical (or update the snapshot to the locked rounded numbers); optionally append the 06-11/06-13/06-16 decisions to the decision log.

### F6 — F4-RESYNC decision still open (known)

The last repo-side hard gate (`DEVELOPMENT_TRACKER.md` row F4-RESYNC; `release-readiness-checklist.md` §1): either re-run the forward-decomposition CBO-migration lever against the corrected baseline, or record a conscious deferral of the caveated stale figures in `methodology_comparison_sdc_2024.md` §4.2.

### F7 — Deferred round-2 external re-review (known)

The round-2 GPT-5.5 Pro review (verdict GO) predates the survival-horizon amendment; re-review vs. the corrected run was explicitly deferred ("decide later"). Record the decision either way.

### Verified clean (for the record)

Checked independently against the locked CSV and artifacts, all consistent with the public copy: headline set 799,358 / trough 797,298 @2027 (−0.26%) / 898,907 @2055 (+12.45%); Cass 201,794 → 268,723 (+33.2%, "about 33%, roughly 269,000"); Williams 41,767 → 64,234 (+53.8%, "about 54%, roughly 64,000"); Burleigh 103,251 → 119,664 (+15.9%); Ward 68,233 → 59,986 (−12.1%); Grand Forks 74,501 → 72,011 (−3.3%); top-3 share of gross gains 78.1% ("more than three-quarters"); 90+ @2055 ≈ 8,172; CSV = 1,922 rows, baseline-only, state/region/county only; the six marketing `.docx` carry the corrected numbers (superseded figures appear only inside explicit "superseded" banners). Not verifiable on this machine (parquets absent): median age 35 → 40; the "about 91% of 2023–2025 net migration is international" share — both should be on the extended sync/QA token list (I6).

---

## 5. Improvements worth making beyond the template

### Tier 1 — cheap, before handoff

- **I1 — Add a "How these projections compare with the 2024 edition" box** (one paragraph + a two-row 2030/2040/2050 table from §3). Framing: newer base (PEP 2025 vs. Census 2020), CBO current-policy migration adjustment, −5% fertility revision, county corrections that follow observed data (Ward), and the natural-increase-vs-migration mirror image. The 2024 report's own Note of Caution discussed the 2005 Census Bureau projections and the 2018 vintage — this is in-template, not a departure. Highest value per hour on this list.
- **I2 — Restore a components-of-change exhibit** (state natural increase vs. net migration by period). The 2024 report had one (p. 6); the storyboard dropped it; it is the single best explainer of the 2025–2027 dip-then-growth shape, and the data is already persisted (`…_baseline_components.parquet`, Stage 3.1). Honor the 06-13 sanity-check note: label deaths as household-basis in any public components table.
- **I3 — Add a compact 8-region key-year table** to the regional page (2024 p. 9 had one; 8 rows × 7 key years fits easily). While there, tighten the regional prose: four of eight regions decline (Minot −9.7%, Devils Lake −10.4%, Jamestown −10.6%, Grand Forks −7.3%) — "eastern and central-rural parts" does not accurately place Minot, and "roughly flat" undersells −10%. Name the four.
- **I4 — Decide the explainer's placement and put it in the packet.** `how-these-projections-work.md` is the release's differentiator, but it is not among the six generated Word docs, is absent from `marketing-ready/README.md`'s "What To Send" table, and no storyboard page references it. Bound-in methods appendix or linked companion — pick one, extend `build_marketing_docx.py`, and bring its hand-authored numbers under the sync check.
- **I5 — Apply the AI-tells prose pass to the PDF copy.** The six 2026-06-18 reviews scope their guidance to *all* public prose, but only the explainer was tightened (`b690bfe`); `draft-public-pdf-copy.md` still carries the flagged patterns (em-dash density, repeated "not X, it is Y" constructions). The GPT-5.5 review's §5 edit list is ready to execute under SOP-005 constraints (connective prose only; numbers untouched; ADR-042 caveat semantics preserved).
- **I6 — Extend `_check_prose_sync` token coverage** to: declining-county count, the five county callouts, median age, 90+ @2055, the top-3 gains share, and the final track-record figure (post-F2). That converts this review's two escapes into build failures on the next rerun. Consider running with `--strict` in the release checklist.
- **I7 — Handoff note for the statewide chart axis.** The draft PNG's y-axis spans ~795k–900k, which makes the −0.26% trough read as a plunge. Ask marketing for a fuller axis or an explicit "−0.26%" annotation so the visual does not overstate what the copy carefully understates (ADR-042 in spirit).

### Tier 2 — judgment calls

- **I8 — Decide the public CSV's precision deliberately.** It currently ships full-precision floats (fractional people, e.g. `898907.0052746864`), which preserves exact state=Σcounties additivity but will look odd to lay users. Either round to integers and document the ±1 additivity tolerance, or keep as-is with a data-dictionary note explaining why.
- **I9 — Future vintage (not this release):** empirical uncertainty ranges from the backtest archive to give "planning ranges" visual form; race/ethnicity public display (storyboard already conditions this on a simple display); race-specific survival (already deferred, ADR-068 D3).

### What *not* to change

- Do **not** reintroduce scenario comparisons — baseline-only (ADR-065) is right for a public audience.
- Do **not** pull the age-sex detail tables back into the PDF — the workbook sheet is the correct home.
- Do **not** soften the projection-not-forecast discipline — it is the clearest single upgrade over the 2024 language.

---

## 6. Suggested action sequence

1. Bisync + regenerate the package on this machine (F4); confirm readiness-checklist §3.
2. Fix F1/F2/F3; extend the sync check (I6); rerun `build_public_draft_package.py` + `build_marketing_docx.py --strict`.
3. Refresh the stale front-door docs (F5).
4. Record the F4-RESYNC decision (F6) and the re-review decision (F7).
5. Adopt whichever of I1–I5/I7 survive owner review — I1 and I2 are argued hardest here, because together they answer the two questions readers of the 2024 report will actually ask.
6. Assemble the packet; Gate 5/6 (layout, accessibility, placeholders, delivery) proceed on the marketing side per the existing checklists.

---

*Findings F1–F7 and improvements I1–I9 are indexed from `DEVELOPMENT_TRACKER.md` (PUB-2026-ERRATA row) for disposition. Companion records: `release-qa-checklist.md`, `release-readiness-checklist.md`, `2026-06-17-pub-2026-release-readiness-status.md`.*
