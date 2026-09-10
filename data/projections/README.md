# `data/projections/` — Projection Outputs Index

| Attribute | Value |
|-----------|-------|
| **Status** | current (orientation index) |
| **Last Updated** | 2026-06-23 |
| **Purpose** | Classify every sibling directory here so a reader can tell, at a glance, which outputs are the live public run versus archived, internal, or input-reference material. |
| **Owner** | cohort_projections |

> **Why this file exists.** These sibling directories are flat siblings with no
> hierarchy signal in the directory name alone. This index removes the
> current-vs-archived ambiguity: it states which directory is *the* public run
> and demotes everything else to its correct role.
>
> **Note:** `data/` is gitignored. This README is intentionally committed-in-spirit
> as an on-disk orienting marker; git tracking of these markers is handled
> separately.

---

## The canonical current run

**`baseline/` is the only current, public, production run.** It is the
**PUB-2026 public release baseline**:

- **Run alias:** `m2026r1` (method/config alias `county_champion`)
- **Locked config:** `cfg-20260611-production-lock`
- **Corrected under:** ADR-068 (2026-06-15, amended 2026-06-16) — the
  `reference_intl_migration`, open-ended 90+ survival, and survival-horizon
  truncation corrections. See `baseline/README.md` for the run-identity detail
  and the authoritative `RUN_CONFIG_SHA16`.
- **Base population:** Census PEP Vintage 2025 (state total 799,358 at 2025;
  ADR-066).
- **Geographies:** 53 counties + a bottom-up state total. **State = county sum
  by construction** (ADR-054) — there is no independent top-down state run.

For the full run provenance and the locked state trajectory, see:

- `data/projections/baseline/README.md` (run identity, emphatic "do not confuse")
- `docs/plans/2026-public-projection-release-handoff/final-run-metadata.md`
- `DEVELOPMENT_TRACKER.md`

---

## Directory classification

| Directory | Class | What it is | Use it? |
|-----------|-------|------------|---------|
| `baseline/` | **CURRENT (locked public run)** | The PUB-2026 public release baseline (`m2026r1`, `cfg-20260611-production-lock`, corrected per ADR-068). 53 counties + bottom-up state. | **Yes — this is the public run.** |
| `CBO/` | **INPUT-REFERENCE** | CBO Demographic Outlook source workbooks/PDFs (`60875-*`, `61879-*`). Reference inputs that informed the CBO-adjusted baseline (ADR-065), not projection outputs. | Reference only — never an output. |
| `methodology_comparison/` | **INTERNAL** | Method-comparison tables/figures (baseline vs. SDC 2024 vs. zero-migration, etc.). Diagnostic/analytical, not a public scenario. | Internal analysis only. |
| `sensitivity_20260611/` | **INTERNAL WHAT-IF** | Parameter-sensitivity runs (`cbo_off`, `d3_hold15`, `fert_off`, `gq075_fwd`, `ref`) supporting the production-lock decisions. | Internal what-if only — not public, not comparable to the public baseline. |
| `sensitivity_refintl_corrected_20260615/` | **INTERNAL WHAT-IF** | The `reference_intl_migration` correction control/corrected comparison (ADR-068 evidence: control vs. corrected configs + diff CSVs). | Internal what-if / audit evidence only. |
| `exports/` | **DERIVED (in-tree)** | Geospatial (GeoJSON) export subtree written by the export pipeline. The `geojson/` subtree is marked stale (`README_STALE.md`, predates the 2026-05-27+ regeneration). | Treat as derived/stale; see `data/exports/` for the canonical export surface. |

### Moved out of this tree

- `high_growth/` and `restricted_growth/` were **deprecated stale scenario
  outputs** (2026-02-26 runs, pre-ADR-065/066) and have been **moved to
  `data_archives/projections/`** per Decision 1 of
  `docs/REPOSITORY_HYGIENE_PROPOSAL_2026-06-23.md` (retire as stale; do not
  regenerate). They are **not comparable** to the current baseline — any
  "scenario range" built against the public baseline from those directories
  mixes incompatible method vintages. If you still see them here, the move has
  not yet executed on this machine; treat them as ARCHIVED-STALE regardless and
  read their `README_STALE.md`.

---

## How `data/projections/` relates to `data/exports/`

Per **Decision 7** of the hygiene proposal, `data/projections/` and
`data/exports/` are **kept as separate trees**, each with its own README
explaining the relationship.

- **`data/projections/`** holds the *raw projection engine outputs* — per-county
  and bottom-up state parquet/CSV/components, organized by scenario.
- **`data/exports/`** holds *published deliverables generated FROM*
  `data/projections/` by the export scripts (`scripts/exports/...` and
  `scripts/pipeline/03_export_results.py`) — formatted workbooks, summary CSVs,
  HTML reports, packaged zips, and data dictionaries.

The derivation is **projections → exports** (one-way). Which export came from
which run is recoverable only by cross-reading the `export_report_*.json` run
logs in `data/exports/`; see `data/exports/README.md` for the canonical dated
set and the provisional/superseded ones.
