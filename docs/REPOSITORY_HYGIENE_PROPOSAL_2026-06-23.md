# Repository Hygiene & Navigability — Proposal

| Attribute | Value |
|-----------|-------|
| **Status** | PROPOSAL — **all 7 decisions recorded 2026-06-23** (see "Decisions" section). Implementation not yet started. |
| **Scope** | `demography/cohort_projections` |
| **Date** | 2026-06-23 |
| **Author** | Claude Code (Opus 4.8) |
| **Method** | 7-area parallel sub-agent audit + independent re-verification of load-bearing facts |
| **Supersedes** | Nothing — complements the historical `docs/REPOSITORY_EVALUATION.md` (2025-12-28) and `docs/REPOSITORY_HYGIENE_IMPLEMENTATION_PLAN.md` (2025-12-31) |

---

## Executive Summary

The repository is fundamentally well-maintained: `.gitignore` is disciplined, only **826 files** are git-tracked, the 2.8 GB of data is correctly excluded, and the PUB-2026 public release already has genuinely good current-vs-archived signaling (the `README_STALE.md` markers and the `docs/plans/README.md` status taxonomy are models worth copying). The friction is **not in *what* is tracked but in *navigation*** — an agent or a newcomer cannot quickly tell which directory is current vs. archived, which entry-point document to trust, or how files are named, without opening many files individually.

The five highest-leverage moves, in priority order:

1. **Retire `REPOSITORY_INVENTORY.md` (291 KB, ~1,960 lines) from the repo root.** It is auto-generated, stale since **2026-01-01**, and actively wrong: it claims **1,919 active files / 0 archived / 0 deprecated** when git actually tracks **826 files** and **204 archived files** exist on disk. It even lists `.mypy_cache`/`.ruff_cache` as "active source code." Its generator (`generate_inventory_docs.py`) **no longer exists**, so the chain is orphaned and cannot self-correct. This is the single biggest trust hazard in the repo. *(Quick win — verified.)*

2. **Adopt one canonical archival convention** and apply it to the three archive zones (`archived/`, `docs/archive/`, `scratch/archive/`). Their boundaries are currently undocumented; an agent cannot distinguish "deprecated output" from "irreplaceable reference data" from "transient scratch." *(Mostly quick win; a few moves are structural.)*

3. **Add a small set of orienting READMEs where the cost of absence is highest:** `data/projections/baseline/` (the *canonical locked public run* currently has **no README** identifying it as such), `data/exports/`, plus a single human-readable `docs/NAVIGATION.md`. *(Quick win.)*

4. **Remove root cruft and decide on the two external symlinks** (`RCLONE_TEST`, `journal_article_output`, `journal_article_versions`). *(Quick win.)*

5. **Write down the naming/metadata conventions that already mostly exist** (kebab for plans/reviews, `NNN-` for ADRs/SOPs, SCREAMING for root governance docs) into one short `docs/naming-conventions.md`, and decide a status-signaling rule. *(Quick win to document; structural only if applied retroactively.)*

Most of the value is in quick wins. The only genuinely structural items needing care are reorganizing the archive zones and `docs/reviews/`, because moving files breaks cross-references.

> **Recommended sequencing:** approve all quick wins (action-table rows 1–13) as one batch now; defer the structural items (rows 14–17) until after the PUB-2026 release lock, so link-breakage cannot disrupt the active public deliverable.

---

## Methodology Note

This proposal synthesizes seven parallel area audits, each run by an independent read-only sub-agent:

1. Root / top-level layout
2. `docs/` documentation tree
3. `data/` outputs & generated deliverables
4. Archive / deprecated / superseded fragmentation
5. Source code, scripts, tests, config layout
6. Cross-cutting naming & metadata conventions
7. Auto-generated inventory / index infrastructure

68 findings were collected; overlapping ones (notably the three archive locations and the stale inventory, which surfaced in 5+ surveys) were deduplicated and merged. Every fact that drives a top-five recommendation was then **independently re-verified against the live repo** before this document was written.

**Evidence corrections applied during synthesis (re-verified):**

- An early survey claimed `archived/` (41 MB) is "tracked in git." **False** — `archived/` is gitignored and tracks **0 files** (112 files on disk only). Nearly all archive content is therefore *untracked*, not committed.
- The **204 archived files** figure checks out exactly: **112** in `archived/` + **13** in `docs/archive/` + **79** in `scratch/archive/`.
- `generate_inventory_docs.py` (referenced by `scripts/intelligence/README.md`) **does not exist** — only `generate_docs_index.py` and `link_documentation.py` remain. The `REPOSITORY_INVENTORY.md` generator chain is orphaned.
- `data/projections/baseline/` (the canonical locked run) has **no README** — confirmed.
- `README_STALE.md` markers exist in exactly three places: `data/projections/high_growth/`, `data/projections/restricted_growth/`, `data/projections/exports/geojson/`.

Items below marked **[verify before acting]** rest on survey evidence not independently re-confirmed.

---

## 1. Current-vs-Archived Separation *(highest-priority concern)*

This is the user's top concern and the area with the widest spread between "done well" and "done badly."

**What already works — keep and propagate:**

- `data/projections/{high_growth,restricted_growth}/README_STALE.md` and `data/projections/exports/geojson/README_STALE.md` correctly mark deprecated outputs with reason (pre-ADR-065/066) and disposition.
- `docs/archive/README.md` is an exemplary archive index (status, purpose, per-file reason, original location).
- `docs/plans/README.md` classifies plans as current / supporting / historical.
- `docs/REPOSITORY_EVALUATION.md` self-marks as a historical snapshot and redirects to `DEVELOPMENT_TRACKER.md`.

**What breaks navigation:**

1. **The canonical current run is not self-identifying on disk.** `data/projections/baseline/` is *the* locked public projection (`m2026r1`, `cfg-20260611-production-lock`) but has **no README** saying so. A newcomer sees `baseline/`, `high_growth/`, `restricted_growth/`, `sensitivity_20260611/`, `sensitivity_refintl_corrected_20260615/`, `CBO/`, `methodology_comparison/`, and `exports/` as flat siblings with no hierarchy signal.
   **Fix:** add `data/projections/baseline/README.md` ("CURRENT LOCKED production run — PUB-2026 public release; run `m2026r1`, config-sha …") and a `data/projections/README.md` index classifying every sibling dir (current / archived-stale / internal what-if / input-reference). Optionally a `current -> baseline` symlink.

2. **Three archive zones with undocumented boundaries** (`archived/`, `docs/archive/`, `scratch/archive/`). They serve genuinely *different* purposes — deprecated data outputs vs. deprecated docs vs. transient evidence — but nothing states that.
   **Fix:** the canonical archival convention in §7A.

3. **Stale outputs in limbo.** `high_growth/` and `restricted_growth/` are marked STALE and "retained pending Stage 3.4," but no disposition was ever recorded. **Decision needed** (Open Question 1): retire, regenerate-under-lock, or formally archive. Until then they remain a confusion surface.

4. **`REPOSITORY_INVENTORY.md` actively misreports archive state** — claims 0 archived; reality is 204 files. See §4.

5. **Public-final vs. provisional exports are intermingled.** `data/exports/` mixes files dated 2026-02-23, 2026-03-01, and 2026-05-27 (e.g. `nd_population_projections_provisional_20260217.xlsx`, `nd_historical_trends_20260302.html`) plus ~25 accumulated `export_report_YYYYMMDD_HHMMSS.json` run logs spanning Dec 2025 → May 2026. Only the latest set aligns with the locked public numbers.
   **Fix:** a `data/exports/README.md` stating which dated set is canonical, and move pre-2026-05-27 exports + old run-report JSONs to `data/exports/archive/`. **[verify before acting]** — confirm the date→run mapping against the `export_report_*.json` files before moving anything.

---

## 2. Directory Structure *(second-priority concern)*

The structure is sound (clean separation: `cohort_projections/` library vs. `scripts/` runners vs. `tests/` vs. `config/`). The issues are *ambiguous-purpose directories* and *parallel trees with no stated relationship*.

- **Ambiguous root directories lack a one-line purpose statement:** `scratch/` (experiments? archive? figures? — all three, undocumented), `examples/` (7 demo scripts, no README), `observatory_wireframe/` (legacy UI mockup, unmentioned anywhere), `libs/` (3 extracted packages — `codebase_catalog`, `evidence_review`, `project_utils` — a second package root alongside `cohort_projections/`, unexplained).
  **Fix:** a one-paragraph README in each. For `observatory_wireframe/`, decide whether it is a live design artifact or belongs in `docs/archive/` (Open Question 4).

- **`data/projections/` and `data/exports/` are parallel scenario trees with no documented mapping.** Which export came from which run is only recoverable by cross-reading `export_report_*.json`.
  **Fix:** state the derivation in both READMEs ("exports are generated from projections via `scripts/exports/…`"). A deeper reorg (`data/baseline/{projections,exports}/`) is possible but **structural — high migration risk** (every export-script path and many docs reference the current layout). Recommend documentation-first; reorg only if the confusion keeps recurring.

- **`docs/reviews/` is a flat dir of 176 tracked `.md` + ~596 untracked artifacts** (772 on disk), mixing dated single reviews, multi-doc review subdirs, and generated HTML — including, for example, **9 timestamped copies of the same `pp3-human-review-package` HTML from a single day (2026-03-01)**.
  **Fix (structural, moderate risk):** group into `docs/reviews/{analysis,benchmarking,audit}/`, add `docs/reviews/README.md`, and gitignore the regenerable timestamped HTML. Moving tracked files breaks links — do a grep-for-references pass first.

- **`scripts/` taxonomy (20 subdirs) is mostly clean** but `data/` vs. `data_processing/` (canonical vs. exploratory) and `analysis/` vs. `backtesting/` vs. `validation/` are not obvious from names alone.
  **Fix:** a "Taxonomy" section in a `scripts/README.md` rather than renaming (renaming breaks imports and docs).

---

## 3. Naming & Metadata Conventions *(third-priority concern)*

Conventions largely *exist* but are *undocumented and unevenly applied*, so they read as inconsistent.

**Observed patterns (good — just unwritten):**

- ADRs/SOPs: strict `NNN-kebab-case.md` (well enforced, 001–068 plus child suffixes like `020a`).
- Reviews/postmortems: `YYYY-MM-DD-kebab.md` (mostly consistent; at least one `YYYYMMDD` outlier).
- Plans/guides/methodology: kebab-case.
- Root governance docs: `SCREAMING_SNAKE` (`AGENTS`, `CLAUDE`, `DEVELOPMENT_TRACKER`).

**Real inconsistencies worth fixing:**

- `docs/` top level splits ~5 SCREAMING / ~5 snake with no rule (`BIGQUERY_SETUP.md` vs. `census_api_usage.md`).
- ADR citation format varies (`ADR-061` vs. `ADR 061`) — fragments search.
- **Status never appears in filenames** — a closed investigation (`population-projection-explosion-bug-investigation-2026-02-13.md`) still looks active; you must consult `plans/README.md` to learn it is done.
- Output files carry dates but no run-id; the locked run is `m2026r1` / a config-sha internally, but that never appears in any filename, so an export cannot be traced to its run from the name alone.

**Recommendation:** adopt the single naming/metadata convention in §7B and write it to `docs/naming-conventions.md`. Prefer *moving* completed plans into an archive over per-file status suffixes (mass suffix-renaming is high-churn and breaks links). The one place a status suffix earns its cost is locked artifacts (e.g. a `-LOCKED` config) — **[verify before acting]** that no tooling globs `config/projection_config.yaml` by exact name before renaming it.

---

## 4. Navigation / Inventory Documents *(fourth-priority concern — the weakest area)*

The auto-generated intelligence layer is the most distrust-inducing part of the repo.

- **`REPOSITORY_INVENTORY.md`** (root, tracked, 291 KB, last generated 2026-01-01): claims 1,919 active / 0 archived / 0 deprecated. Verified reality: **826 tracked, 204 archived**. Lists cache dirs as source code. Too large for an agent to usefully read. Its generator does not exist.
  **Recommendation: remove from the root and from git tracking.** Move to `docs/auto/` *only* if regeneration is ever fixed; otherwise delete. This is the single clearest quick win.

- **`docs/INDEX.md`** (auto-generated, "Do not edit," last committed 2026-02-27): sparse, no generated-date stamp, omits whole subtrees (`governance/`, `guides/`, `research/`); a search for "ADR" returns nothing. It is a code→doc link *sample*, not a navigation index.
  **Recommendation:** replace its navigational role with a hand-maintained `docs/NAVIGATION.md` (table: directory | purpose | key files | status-source pointer), and either retire `INDEX.md` or stamp it `Generated: <date> — advisory only`.

- **No single entry-point hierarchy.** Four large root markdown files compete: `AGENTS.md` (~405 lines), `CLAUDE.md` (~147), `DEVELOPMENT_TRACKER.md` (~83 KB), `REPOSITORY_INVENTORY.md` (~1,960).
  **Recommendation:** declare the hierarchy explicitly in `README.md` — README = orientation; `CLAUDE.md` = Claude quick-ref; `AGENTS.md` = canonical agent instructions; `DEVELOPMENT_TRACKER.md` = canonical current state. Remove `REPOSITORY_INVENTORY.md` from the set.

- **No archive manifest and no work-ID registry.** Five identifier schemes (ADR / SOP / PP / CF / PUB) live in four locations.
  **Recommendation:** a single `docs/work-registry.md` (table: scheme | purpose | count | canonical location) and a `docs/ARCHIVE_MANIFEST.md` enumerating the 204 archived files by location / reason / date. Both are low-risk additions with high navigation payoff.

- **No documented regeneration process.** The generators require PostgreSQL env vars and silently no-op without them, which is why everything froze.
  **Recommendation:** if the indices stay, add a maintenance note and a DB-free fallback; if they go, remove the orphaned `scripts/intelligence/README.md` reference to the missing generator.

---

## 5. Root & Clutter *(fifth-priority — small, fast, low-risk)*

- **`RCLONE_TEST`** (27 bytes, gitignored, rclone-bisync marker) — delete, or document its purpose in one line.
- **`journal_article_output` / `journal_article_versions`** — symlinks into a sibling repo (`../sdc_2024_replication/…`, ~6,000 files); gitignored but they inflate naive file scans and descend out of the repo. **Decide** (Open Question 3): remove if unused (CLAUDE.md already refers to `../sdc_2024_replication/` directly), or document in a one-line note if kept for convenience.
- **On-disk generated dirs** (`htmlcov/` 14 MB, `logs/` 17 MB, `.mypy_cache/`, `.ruff_cache/`, `.pytest_cache/`, `.uv_cache/`, `cohort_projections.egg-info/`) — **all correctly gitignored**; this is purely cosmetic working-tree clutter, not committed clutter.
  **Recommendation:** one line in README/CONTRIBUTING noting they are regenerable and safe to delete. Do **not** over-engineer a `.cache/` relocation unless desired.
- `.gitignore`, `.env` / `.env.example` handling — **already correct, no change.**

---

## 6. Code / Scripts Organization *(lower priority — structure is sound)*

The gap is missing orientation READMEs in execution-heavy directories.

- Missing READMEs where they would help most: `scripts/{data,analysis,exports}/`, `tests/`, `config/`, `examples/`, `libs/`, `cohort_projections/analysis/`, `cohort_projections/data/`.
  **Recommendation:** add brief (3–5 paragraph) READMEs; prioritize `tests/`, `config/`, and `cohort_projections/analysis/`. The CLI-in-`scripts/` vs. implementation-in-`cohort_projections/` split is the most confusing relationship for newcomers and deserves an explicit note.
- The pipeline numeric-prefix + symlink convention (`01_ -> 01a_`) is fine but undocumented at the `scripts/` root. Document it; don't rename.

---

## Prioritized Action Table

| # | Change | Type | Risk | Why it helps navigation |
|---|--------|------|------|--------------------------|
| 1 | Remove `REPOSITORY_INVENTORY.md` from root + git tracking | Quick win | Low | Kills the single most-wrong, distrust-inducing file |
| 2 | Delete `RCLONE_TEST`; decide on `journal_article_*` symlinks | Quick win | Low | Removes root cruft + out-of-repo scan traps |
| 3 | Add `data/projections/baseline/README.md` + `data/projections/README.md` index | Quick win | Low | Makes the canonical locked run self-identifying |
| 4 | Add `data/exports/README.md` (canonical date set + derivation from projections) | Quick win | Low | Resolves public-final vs. provisional ambiguity |
| 5 | Write `ARCHIVE_STRATEGY.md` + a README in each of the 3 archive zones | Quick win | Low | One rule for where archived things live |
| 6 | Create `docs/ARCHIVE_MANIFEST.md` (204 files, by reason/date) | Quick win | Low | Single lookup for "what's archived and why" |
| 7 | Create `docs/NAVIGATION.md` (directory-purpose table) | Quick win | Low | Human-friendly map replacing sparse `INDEX.md` |
| 8 | Create `docs/work-registry.md` (ADR/SOP/PP/CF/PUB) | Quick win | Low | One place to see all work-item namespaces |
| 9 | Write `docs/naming-conventions.md` (rules in §7B) | Quick win | Low | Makes the de-facto conventions explicit |
| 10 | Add orientation READMEs to scripts/tests/config/libs/analysis | Quick win | Low | Orients agents in execution-heavy dirs |
| 11 | Decide & document `high_growth/` / `restricted_growth/` disposition | Quick win (decision) | Low | Removes "limbo" outputs |
| 12 | Fix/retire `docs/INDEX.md`; fix orphaned `scripts/intelligence/README.md` ref | Quick win | Low | Stops pointing agents at a missing generator |
| 13 | Add metadata headers (status/date) to top-level `docs/*.md` | Quick win | Low | At-a-glance currency for each doc |
| 14 | Reorganize `docs/reviews/` into analysis / benchmarking / audit | Structural | Medium | Tames the largest flat doc dir — **breaks links** |
| 15 | Move pre-2026-05-27 exports + old run-reports to `data/exports/archive/` | Structural | Medium | Segregates final from provisional — **verify date→run first** |
| 16 | Consolidate/rename archive zones (e.g. `archived/` → `data_archives/`) | Structural | Medium | One archive hierarchy — **breaks paths/docs** |
| 17 | Rename `data/raw/.../{backup,rpc_archives}` to `*_vault` | Structural | Medium | Signals immutability — **breaks loader paths; verify first** |

Structural items (14–17) should follow a grep-for-references pass and ideally a single commit per move, leaving a redirect note behind.

---

## 7. Proposed Canonical Conventions (adoptable rules)

### A. Archival convention — one scheme, three zones, classified by *content type* not by accident

> **Rule:** archived material is classified by *what it is*, and each type has exactly one home:
>
> 1. **Deprecated/superseded documentation** → `docs/archive/` (git-tracked; keep the existing `README.md` index format: file | archived-date | reason | original-location).
> 2. **Deprecated/superseded data outputs** → `archived/` at root (gitignored; on-disk only). Every subdir gets a `README_STALE.md` with: Status, Reason (ADR ref), Last-valid-date, Disposition (retire / regenerate / keep + date). This generalizes the existing `data/projections/*/README_STALE.md` pattern.
> 3. **Transient working / evidence artifacts** → `scratch/archive/` (gitignored; auto-cleanable; no preservation guarantee).
>
> **Not archives — immutable source vaults** (`data/raw/.../backup/`, `rpc_archives/`) are *reference data, never delete*; rename to `*_vault` and add a `README_SOURCES.md` (source URL, access date, why retained). Keep them out of the archive taxonomy entirely so an agent never treats them as expendable.
>
> A single `ARCHIVE_STRATEGY.md` (repo root or `docs/`) states this rule and links the three zone READMEs and `docs/ARCHIVE_MANIFEST.md`. The lifecycle is: **`README_STALE` marker → `ARCHIVE_MANIFEST` entry → move to the correct zone once disposition is decided.**

### B. Naming & metadata convention

> **Filenames:**
> - Governance/decision docs: `NNN-kebab-title.md` (ADR/SOP), zero-padded 3 digits.
> - Time-stamped docs (reviews, postmortems): `YYYY-MM-DD-kebab-title.md` (ISO; fix the `YYYYMMDD` outliers).
> - Plans / guides / methodology: `kebab-case.md`.
> - Stable root governance docs: `SCREAMING_SNAKE_CASE.md` (`AGENTS`, `CLAUDE`, `DEVELOPMENT_TRACKER`, `README`).
> - `docs/` top level: SCREAMING for stable entry-points, snake_case for technical/methodology support docs — pick one per file deliberately.
> - Output/export files: include a **run identifier**, not just a date — e.g. `nd_..._<date>_m2026r1.parquet`. Document the run-id ↔ config-sha mapping in `data/projections/README.md`.
>
> **Status signaling:** prefer *moving* finished items into the archive over filename suffixes. Reserve suffixes for the rare immutable artifact (`-LOCKED`). Everywhere else, the directory + a metadata header carries status.
>
> **Metadata header** (top of every non-trivial markdown doc; a module docstring block per SOP-002 for code):
> `Status: current | supporting | historical | deprecated | locked` · `Last Updated: YYYY-MM-DD` · `Owner` · `Supersedes / Superseded-By` (when applicable).
>
> **Cross-references:** always `ADR-NNN` (hyphen, no space). Consider a lightweight pre-commit grep to flag `ADR␣NNN`.

---

## Decisions (Recorded 2026-06-23)

All seven open questions have been resolved by the repo owner. Decisions are binding for the implementation phase.

| # | Question | **Decision** |
| --- | --- | --- |
| 1 | `high_growth/` & `restricted_growth/` disposition | **Move both to `archived/`.** (Retire as stale outputs; do not regenerate.) |
| 2 | `REPOSITORY_INVENTORY.md` & the intelligence system | **Invest in fixing it — by refactoring onto the workspace ledger** (see recommendation below), not by repairing the bespoke `cohort_projections_meta` system. |
| 3 | `journal_article_*` symlinks | **Remove the symlinks.** Separately ensure the *most recent* article version is clearly separated from and discoverable apart from all prior versions. |
| 4 | `observatory_wireframe/` | **Archive it.** |
| 5 | Archive-zone path renames | **Do the renames now** (do not defer). |
| 6 | `docs/reviews/` structural regrouping | **Do the regrouping now** (do not defer). |
| 7 | `data/exports` vs. `data/projections` | **Keep them separate for now**, each with a README explaining its difference and purpose. |

### Decision 2 — recommendation: refactor onto the workspace ledger

**Recommendation: yes — retire the bespoke per-repo intelligence system and re-point this repo's inventory at the workspace ledger.** This is the better investment for these grounded reasons:

- **The bespoke system is already dead, not merely stale.** `scripts/intelligence/README.md` references five scripts; **three are gone** (`scan_repository.py`, `archive_manager.py`, `generate_inventory_docs.py`). Nothing can repopulate the local `cohort_projections_meta` Postgres DB or regenerate `REPOSITORY_INVENTORY.md`. Repairing it means rebuilding a scanner + archive-manager + generator from scratch.
- **The workspace ledger already does all of this, better, and already indexes this repo.** `workspace_silver.ledger` (owned by the command-center repo, `src/workspace_command_center/ledger/`) provides exactly the concepts the local system faked: an `asset_status` table, `ledger.v_stale_assets` (assets inactive > 30 days — *directly* drives the current-vs-stale signal), `presence_status` / `integrity_status`, lineage edges, and per-`repo` scoping. It is a living, maintained, workspace-wide system; the per-repo DB is a drifting duplicate.

**Proposed shape of the refactor:**

1. **Retire** the `cohort_projections_meta` DB and the `scripts/intelligence/` system (delete the orphaned scripts and the README's dead references).
2. **Replace** `REPOSITORY_INVENTORY.md` with a thin generated report (`scripts/intelligence/` → a single small script, or a `scripts/maintenance/` entry) that **queries the workspace ledger filtered to `repo = demography/cohort_projections`** — via the `workspace ledger query` CLI / the HTTP API (`http://127.0.0.1:8765`) / a read-only `SELECT` against the `ledger.v_*` views. No re-implemented scanning.
3. Drive the navigation artifact's "current vs. archived" column directly from `ledger.asset_status` + `v_stale_assets`, so it can never again claim "0 archived."

**Caveats to honor (per workspace governance docs):**

- This repo becomes a **consumer** of the ledger, consistent with the workspace ownership model (`AGENTS.md` / root `CLAUDE.md`: per-repo repos read the ledger; they do not own it).
- The ledger is **per-machine** (`docs/REFERENCE-cross-machine-db-state.md`: Postgres DBs are per-machine rebuilds, not bisynced). Any committed markdown snapshot must be stamped `Generated from workspace ledger on <date> — advisory, may be machine-specific`, and the repo must not hard-depend on the ledger service being up.
- The ledger CLI/API was **not reachable on this machine at audit time** (service down, `workspace` not on PATH). Confirm the ledger is running and that it already has fresh `demography/cohort_projections` rows before wiring the generator to it — **[verify before building]**.

> This decision is recorded as **approved in principle**; the only remaining check is confirming the ledger is live and indexing this repo before the generator is wired up.

---

## Implementation Implications of the Decisions

- Decisions **5 and 6** move the structural items (action-table rows 14, 16, 17) from "deferred" to **do-now**. They still carry link/path-breakage risk, so each must follow a grep-for-references pass and land as its own commit with redirect notes. Recommended ordering: do these **before** the PUB-2026 release lock only if the active handoff files are untouched by the moves; otherwise sequence the `docs/reviews/` regroup to avoid churning any file the release references. **[verify before acting]** which `docs/reviews/` files the PUB-2026 handoff links to.
- Decision **1** (`high_growth/`/`restricted_growth/` → `archived/`) should carry their existing `README_STALE.md` content into a manifest entry (per §7A: `README_STALE` → `ARCHIVE_MANIFEST` → move).
- Decision **3** (remove symlinks) pairs with a follow-up in the *sibling* `sdc_2024_replication` repo to make the latest article version discoverable (e.g. a `latest/` pointer or a `VERSIONS.md` index) — that work is out of scope for *this* repo but should be tracked.
- Decision **7** confirms the low-risk path: add `data/projections/README.md` and `data/exports/README.md` (action-table rows 3, 4) and explicitly state in each how the two trees relate.

---

*This document is a proposal with decisions recorded. No files have been moved, renamed, deleted, or created beyond this proposal itself. Implementation will begin only on explicit go-ahead.*
