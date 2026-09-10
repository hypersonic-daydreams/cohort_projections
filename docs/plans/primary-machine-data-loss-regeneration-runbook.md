# Primary-Machine Data-Loss Regeneration Runbook (PUB-2026)

| Field | Value |
|-------|-------|
| **Created** | 2026-07-06, from the live cross-machine verification in [`2026-07-06-pub-2026-draft-materials-assessment.md`](../reviews/audit/2026-07-06-pub-2026-draft-materials-assessment.md) (F4 update) |
| **Situation** | The primary/production machine had a filesystem incident (owner-reported ~85% chance of file recovery). Its `data/` tree holds the only copies of the stage-02 projection outputs and the June benchmark evidence archives. Google Drive and the secondary (WSL) machine are mutually in sync but **do not** have those outputs. |
| **Status** | **Contingency — execute Section 4 only if primary recovery fails.** Sections 2–3 (inventory) and 6–7 are valid reference regardless of outcome. |
| **Locked run** | `m2026r1` / `cfg-20260611-production-lock`, ADR-068-amended full-horizon rerun 2026-06-16; config sha256(16) `a6e0bfbc2d70be85`; state 799,358 (2025) → 797,298 trough @2027 → 898,907 @2055; 90+ @2055 = 8,172. |
| **Companions** | [`final-run-metadata.md`](./2026-public-projection-release-handoff/final-run-metadata.md), [`release-readiness-checklist.md`](./2026-public-projection-release-handoff/release-readiness-checklist.md), [`2026-06-17-pub-2026-release-readiness-status.md`](../reviews/2026-06-17-pub-2026-release-readiness-status.md) |

---

## 1. Verified facts (2026-07-06, from the WSL machine)

These were established live, not inferred — re-verification commands in §8.

1. **Drive and the WSL machine are byte-converged.** `./scripts/bisync.sh --dry-run` completed with "No changes found"; the 08:15 bisync listing snapshots for both sides are identical.
2. **Neither Drive nor the WSL machine has the stage-02 projection outputs.** Zero `*2025_2055_baseline*` files and no `data/projections/baseline/state/` on either side. Both hold the same 265 stale December-vintage `*2025_2045_baseline*` files.
3. **A partial June push did reach Drive.** All locked *pipeline inputs* are on Drive (and therefore also on the WSL machine) — see §2. Why the push was partial is unknown (aborted sync, timing, or the incident itself); it does not matter operationally.
4. **The live git-tracked config is the locked config.** `sha256sum config/projection_config.yaml | cut -c1-16` → `a6e0bfbc2d70be85` on the WSL machine, matching the recorded lock.
5. **The authoritative public numbers are git-tracked.** The locked 1,922-row public CSV (`docs/plans/2026-public-projection-release-handoff/marketing-ready/drafts/PUB-2026 Draft Public Dataset.csv`) was last committed by the ADR-068 amendment itself (`e719518`, 2026-06-16) and reconciles to the locked headline set (independently recounted 2026-07-06).
6. **`data/analysis/` is excluded from the bisync filter entirely** (the filter whitelists only `data/raw|processed|interim|projections|exports|output`). Benchmark history, experiment archives, and observatory artifacts are per-machine and have never had an off-machine copy.

## 2. Tier 0 — safe regardless of recovery (no regeneration ever needed)

**In git (GitHub + every clone):**

- The locked public CSV (authoritative final numbers), all 10 chart/pyramid PNGs, the six marketing `.docx`, `draft-public-pdf-copy.md`, `how-these-projections-work.md`.
- The complete decision/QA record: ADRs (061/065/066/067/068), `final-run-metadata.md`, the 2026-06-13 sign-off, the 2026-06-16 corrected-run QA verification, the 2026-06-17 readiness memo, `methodology.md`, `methodology_comparison_sdc_2024.md`.
- The locked `config/projection_config.yaml` (verifiable by hash) and all pipeline/export code.

**On Google Drive AND the WSL machine (bisync-held, verified present both sides):**

| Locked input | Path | Stamp (UTC) |
|---|---|---|
| Stage 01a residual migration (11 college counties + GQ 0.75) | `data/processed/migration/residual_migration_rates{,_averaged}.parquet`, `residual_migration_metadata.json` | 2026-06-13 23:34 |
| Stage 01b convergence (53 counties) | `data/processed/migration/convergence_rates_by_year{,_high}.parquet`, `convergence_metadata{,_high}.json` | 2026-06-13 22:32 |
| Stage 01c corrected full-horizon survival (2025–2055) | `data/processed/mortality/nd_adjusted_survival_projections.parquet`, `mortality_improvement_metadata.json` | 2026-06-16 17:02 |
| ADR-066 Vintage-2025 base set | `data/processed/{county_population,age_sex_race_distribution,fertility_rates,survival_rates,migration_rates}.parquet` | 2026-05-27 19:56 |
| GQ inputs | `data/processed/gq_county_age_sex_{2025,historical}.parquet` | 2026-03-04 |
| All raw inputs incl. the 2024 SDC reference PDF | `data/raw/**` | various |

These are exactly the three things QA Gate 1's input-coverage check verifies (mortality horizon span, residual-migration metadata, convergence coverage) plus the base set.

## 3. What exists only on the primary machine (at risk)

| Asset | Release-critical? | If unrecovered |
|---|---|---|
| Stage-02 projection grid: 53 county parquets + state parquet + components parquet (`data/projections/baseline/**`, 2025–2055) | **Yes** (needed to rebuild the workbook; CSV alone lacks single-year age-sex detail) | Regenerate — §4 |
| Consolidated public workbook + bundled reference PDF (gitignored; live under `marketing-ready/`) | **Yes** | Regenerated by §4 step 5 |
| June-11 benchmark/walk-forward evidence bundles (ADR-067 authoritative raw-base matrix incl. `rawbase-production-lock`, EXP-B/EXP-C) under `data/analysis/` | No — decisions/metrics/rationale fully documented in git | Accept + record loss (§5) |
| F4 one-factor decomposition runs (cbo_off / fert_off / d3_hold15 / gq075) | No — already stale under ADR-068; F4-RESYNC governs | Nothing extra (§5) |
| Diagnostic/side outputs: `sensitivity_refintl_corrected_20260615` run data, detail workbooks, pyramid explorer | No | Regenerate from §4 outputs on demand |
| Primary's manifest DB / other per-machine state | No — per-machine rebuilds by workspace convention | Rebuild locally if needed |

## 4. Regeneration runbook (execute ONLY if recovery fails)

Runs entirely on the WSL machine (or any clone + bisync pull). Estimated ~1 day including verification. Sandbox-first so nothing touches production paths until proven — mirroring the 2026-06-17 memo's from-scratch reproduction, which matched the locked grid to **0 diff across all 33,852 cells**.

1. **Preconditions.** `git pull`; confirm `sha256sum config/projection_config.yaml | cut -c1-16` = `a6e0bfbc2d70be85`; `./scripts/bisync.sh --dry-run` converges; re-run the Gate-1 input-coverage checks (mortality file spans 2025–2055 / 31 years; `residual_migration_metadata.json` = 11 college counties + GQ 0.75; convergence rates present for all 53 counties). These guard against the 2026-06-01 silent-stale-input failure mode.
2. **Stage 02 into a sandbox.** `scripts/pipeline/02_run_projections.py --counties --state` with output redirected to a sandbox directory (follow the sandboxed-output approach used by the 2026-06-17 verification). Do **not** rerun 01a/01b/01c — their locked outputs are the Tier-0 inputs above; rerunning them only adds drift risk. The ADR-068 recurrence guards (survival-horizon coverage, migration-horizon, 90+ pin) run in-engine and hard-fail on wrong inputs.
3. **Verify against the git-tracked CSV (the canary).** Rebuild the tidy public dataset from the sandbox outputs (`scripts/exports/build_public_draft_package.py` pointed at the sandbox, or the tidy-builder directly) and **byte-compare** against the tracked locked CSV. Expected result: exact match.
   - **Match ⇒ the regenerated grid IS the locked run**; the existing QA record (Gates 1–4) remains valid as-is; promote the sandbox outputs into `data/projections/baseline/`.
   - **Any diff ⇒ STOP.** Do not promote, do not rationalize small deltas. Diagnose (environment, input drift, code drift since `e719518`); a regenerated run that does not reproduce the locked CSV is a new run and would require the full Gate 1–4 re-execution plus a decision about which numbers are public.
4. **Regenerate the components parquet check.** The engine persists components automatically (since `12fa6f9`); confirm `..._baseline_components.parquet` exists and components reconcile to population change (Gate 1b §3 style).
5. **Rebuild the public package.** `python scripts/exports/build_public_draft_package.py` (workbook + reference-PDF copy + CSV/PNG refresh — tracked CSV/PNGs should come out unchanged; treat any git diff on them as a §4.3-style stop signal). Then `python scripts/exports/build_marketing_docx.py --strict` (expected no-op).
6. **Push.** `./scripts/bisync.sh` so Drive finally holds the projections outputs; confirm a second machine can pull them.
7. **Record.** Note the regeneration (date, machine, byte-compare result) in `final-run-metadata.md` and check off the readiness-checklist §3 items.

## 5. Tier-2 dispositions if recovery fails

- **Benchmark evidence bundles (June 11):** recommended disposition is **accept the loss and record it** in `data/analysis/benchmark_history/README` (or equivalent) with a pointer to the git-tracked decision record (ADR-061/067, the 2026-03-09 decision doc, the 2026-06-13 reviews). Re-running the benchmark suite would produce re-derivations, not the archived originals, at significant compute cost; the governance record is already complete without them. Reopen only if SOP-003 archival requirements are judged to demand re-materialization.
- **F4 decomposition runs:** nothing beyond F4-RESYNC. Its "re-run" branch regenerates them fresh against the corrected baseline anyway; its "defer" branch needs no files.
- **Diagnostics/side outputs:** regenerate individually from the §4 grid only when something actually needs them.

## 6. If recovery succeeds (the 85% case)

1. On the primary: verify the recovered `data/projections/baseline/**` still reconciles (spot-check state 2025/2027/2055 against the locked CSV; cheap insurance after a filesystem incident).
2. `./scripts/bisync.sh` from the primary to push. Check whether the stale `2025_2045` county files were deleted there — if so, the deletions propagate to Drive/WSL (desirable); if not, the mixed-horizon hazard from the 2026-05-27 decision log persists on all machines and should be cleaned deliberately.
3. Pull on the WSL machine; everything in §4 becomes unnecessary.

## 7. Systemic follow-ups (regardless of outcome)

- **Close the evidence blind spot:** add `data/analysis/benchmark_history/**` (at minimum the dated bundles plus `index.csv` / `promotion_history.csv`) to the rclone bisync filter, or give the archive an explicit backup path. Single-machine evidence storage — not the projections — was the actual exposure in this incident.
- **Backfill raw `CO-EST2025-ALLDATA`** (county PEP file, ADR-066) to `data/raw/` → Drive. Only its processed product (`county_population.parquet`) made it off the primary; the raw file is a public Census re-download, wanted for provenance completeness.
- Longer term, note that the consolidated workbook is doubly excluded from both sync channels (gitignored `*.xlsx`; `docs/**` excluded from rclone) — by design it is regenerate-per-machine, but that design choice is only safe while its inputs remain multi-homed, which is what this runbook's Tier-0 table now documents.

## 8. Re-verification appendix

How the §1 facts were established (all read-only; safe to repeat):

- Listing snapshots: `~/.cache/rclone/bisync/home_nigel_workspace_demography_cohort_projections..wsdrive_cohort_projections.path{1,2}.lst` — grep for `2025_2055_baseline` (expect >0 after regeneration/push; was 0), `baseline/state`, `2025_2045_baseline` (was 265 each side), and the §2 input paths with their timestamps.
- Live convergence check: `./scripts/bisync.sh --dry-run` (wrapper-only; never raw rclone).
- Config lock: `sha256sum config/projection_config.yaml | cut -c1-16` → `a6e0bfbc2d70be85`.
- CSV provenance: `git log -1 -- "docs/plans/2026-public-projection-release-handoff/marketing-ready/drafts/PUB-2026 Draft Public Dataset.csv"` → `e719518` (2026-06-16 ADR-068 amendment).
