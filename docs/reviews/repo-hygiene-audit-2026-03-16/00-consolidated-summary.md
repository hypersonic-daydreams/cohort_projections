# Repository Hygiene Audit -- Consolidated Summary

**Date:** 2026-03-16
**Commit:** `0b133d4` (master)
**Previous audit:** 2026-02-26 (`docs/reviews/repo-hygiene-audit/`)

---

## Aggregate Finding Counts

| Report | CRITICAL | WARNING | INFO | Total |
|--------|:--------:|:-------:|:----:|:-----:|
| 01 - Documentation Freshness | 1 | 9 | 17 | 27 |
| 02 - SOP Compliance | 1 | 5 | 7 | 13 |
| 03 - Code Organization | 0 | 7 | 8 | 15 |
| 04 - Config/Data/Git Hygiene | 0 | 5 | 18 | 23 |
| 05 - AI Navigability | 0 | 5 | 9 | 14 |
| **Totals** | **2** | **31** | **59** | **92** |

---

## Overall Health Assessment

**Grade: B+** -- The repository maintains strong fundamentals (zero circular imports, no hard-coded paths, no secrets, zero broken cross-reference links, clean dependency layering) but has accumulated maintenance debt from rapid growth. The codebase grew 148% since the February audit (17.8K to 44K lines), and documentation/indexing has not kept pace.

### What's Excellent
- Zero broken markdown cross-references across 279 files
- Zero circular imports, zero star imports
- No secrets or credentials in tracked files
- Clean data directory hierarchy with per-directory provenance docs (10/11 dirs)
- Reproducible environment (uv.lock + direnv + .env.example)
- AI agent onboarding path (CLAUDE.md -> AGENTS.md -> DEVELOPMENT_TRACKER.md) is best-in-class
- Autonomy tier system prevents unauthorized methodology changes

### What Needs Attention
- Documentation freshness (stale Current Focus, ADR index, test counts)
- 488 tracked files that match .gitignore rules (pre-date the rules)
- Dead code that survived from the February audit
- New duplication (`_normalize_fips()` in 4 modules)
- 15 data processing scripts missing SOP-002 docstrings
- 41 ADRs missing Implementation Results sections

---

## Top 15 Action Items (Prioritized by Impact/Effort)

### Quick Wins (< 30 min each)

| # | Action | Source | Effort |
|---|--------|--------|--------|
| 1 | **Fix ADR README index** -- wrong status counts, 4 mismatched statuses, 2 missing entries (062, 063), stale total (64 vs 67) | 01 CRITICAL | 15 min |
| 2 | **Delete `example_usage.py`** from production package + remove lazy-load helper from `__init__.py` | 02 CRITICAL, 03 | 5 min |
| 3 | **Update AGENTS.md Current Focus** from PP-005 to CF-001 + maintenance mode | 01, 05 | 5 min |
| 4 | **Update CLAUDE.md status preamble** to reflect PP-005 through PP-009 completion | 05 | 5 min |
| 5 | **Resolve pre-commit advice conflict** -- CLAUDE.md recommends `pre-commit run --all-files` but MEMORY.md says don't (3+ min runtime). Add time warning or change to recommend `pytest` directly | 05 | 5 min |
| 6 | **Create `data/raw/housing/DATA_SOURCE_NOTES.md`** for `nd_place_housing_units.csv` | 01, 02 | 10 min |
| 7 | **Fix broken SOP path references** -- `docs/sops/templates/` should be `docs/governance/sops/templates/` in 3 files | 02 | 5 min |
| 8 | **Delete `vital_stats.py`** stub (133 lines, zero imports) | 03 | 2 min |
| 9 | **Remove `pathspec` from core dependencies** -- only used transitively by mypy | 04 | 2 min |
| 10 | **Clean .gitignore duplicates** and add `docs/**/*.pdf` pattern | 04 | 5 min |

### Medium Effort (30 min - 2 hours)

| # | Action | Source | Effort |
|---|--------|--------|--------|
| 11 | **Untrack 488 files matching .gitignore** -- `git rm --cached` for evidence JSONs, SVG assets, backtest results, generated HTML | 04 | 30 min |
| 12 | **Extract shared `_normalize_fips()`** from 4 place-projection modules into a single utility | 03 | 45 min |
| 13 | **Add code architecture overview** to AGENTS.md -- 5-10 line module map closes the biggest discoverability gap | 05 | 20 min |
| 14 | **Backfill Implementation Results for 5 post-SOP-002 ADRs** (057, 059, 060, 062, 063) | 02 | 1-2 hr |
| 15 | **Add SOP-002 docstrings to 7 highest-priority scripts** (`ingest_stcoreview.py`, `ingest_ves_data.py`, + 5 ADR-034 Census PEP scripts) | 02 | 2-3 hr |

### Deferred / Lower Priority

| Action | Source | Notes |
|--------|--------|-------|
| Split oversized modules (migration_rates 1,963L, base_population_loader 1,471L, place_projection_orchestrator 1,786L) | 03 | Significant refactor; schedule separately |
| Move 8 root-level scripts to appropriate subdirectories | 03 | Low value, cosmetic |
| Consolidate `scripts/data/` and `scripts/data_processing/` | 03 | Overlap, low urgency |
| Extract ~500 lines of transform logic from `02_run_projections.py` into `data/transform/` | 03 | High value but medium risk |
| Move heavy optional deps (psycopg2, sqlalchemy, pdfplumber, ocrmypdf) to optional groups | 04 | Breaking change for existing installs |
| `git filter-repo` to prune ~15 MB of historical blobs | 04 | Only matters if clone speed is a concern |
| Backfill Implementation Results for remaining ~36 pre-SOP-002 ADRs | 02 | Large effort, diminishing returns |
| Normalize ADR-015 format to current template | 02 | Cosmetic |
| Bump `docs/methodology.md` version from 1.0 | 01 | Low risk |
| Update MEMORY.md (stale ADR counter, test count context, archive completed refactor memory) | 05 | Agent housekeeping |
| Add `.gitkeep` for `data/interim/`, `data/backtesting/`, `data/exports/` | 04 | Ensures fresh-clone structure |
| Trim AGENTS.md version history table | 05 | Saves 12 lines of context |
| Exclude cache dirs from REPOSITORY_INVENTORY.md | 05 | Currently >256KB, unreadable by agents |

---

## Changes Since February 2026 Audit

### Fixed Since Feb
- `version.py` modernized (reads from pyproject.toml)
- `output/__init__.py` stale `__version__` removed
- `demographic_utils.py` duplication resolved

### Unfixed From Feb
- `example_usage.py` dead code (now CRITICAL)
- `vital_stats.py` stub
- `migration_rates.py` oversized (1,963 lines)
- `base_population_loader.py` oversized (1,471 lines)
- Mixed import styles (relative in core/output, absolute elsewhere)
- Root-level scripts ungrouped
- PostgreSQL Section 11 in AGENTS.md potentially misleading

### New Issues Since Feb
- Package grew 148% (17.8K -> 44K lines); observatory alone is 13.9K lines
- `_normalize_fips()` copied into 4 modules
- 11 modules now exceed 1,000 lines (was 3)
- ADR index fell out of sync
- AGENTS.md Current Focus became stale
- Pre-commit advice in CLAUDE.md contradicts learned MEMORY.md guidance

---

## Detailed Reports

| Report | File |
|--------|------|
| Documentation Freshness | [`01-documentation-freshness.md`](01-documentation-freshness.md) |
| SOP Compliance | [`02-sop-compliance.md`](02-sop-compliance.md) |
| Code Organization | [`03-code-organization.md`](03-code-organization.md) |
| Config/Data/Git Hygiene | [`04-config-data-git-hygiene.md`](04-config-data-git-hygiene.md) |
| AI Navigability | [`05-ai-navigability.md`](05-ai-navigability.md) |

---

| Attribute | Value |
|-----------|-------|
| **Audit Date** | 2026-03-16 |
| **Auditors** | 5 parallel Claude Opus 4.6 agents |
| **Repository** | cohort_projections @ master (`0b133d4`) |
| **Total Findings** | 2 CRITICAL, 31 WARNING, 59 INFO |
| **Previous Audit** | 2026-02-26 |
