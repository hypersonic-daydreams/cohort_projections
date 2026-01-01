# Agent B0a: Journal Article Versioning Strategy

## Document Information

| Field | Value |
|-------|-------|
| Agent ID | B0a |
| Scope | PDF versioning and auditability infrastructure |
| Status | Planning Complete |
| Created | 2026-01-01 |

---

## 1. Current State Assessment

### 1.1 Inventory of Article Versions Found

**Location**: `sdc_2024_replication/scripts/statistical_analysis/journal_article/output/`

| Filename | Size | Date | Pattern |
|----------|------|------|---------|
| `article_draft.pdf` | 494 KB | 2025-12-29 | Base name |
| `article_draft_v2_revised.pdf` | 752 KB | 2025-12-30 | v-number + semantic |
| `article_draft_v3_phase3.pdf` | ~750 KB | 2025-12-30 | v-number + phase |
| `article_draft_v4_lq_fix.pdf` | ~750 KB | 2025-12-31 | v-number + fix reference |
| `article_draft_v5_p305_complete.pdf` | 757 KB | 2025-12-31 | v-number + section + semantic |

**Additional locations with duplicates**:
- `claim_review/v3_phase3/extracted/` - PDF copies for text extraction
- Possible other scattered copies

### 1.2 Current Naming Patterns

Multiple conflicting conventions in use:
- **v-numbers**: v2, v3, v4, v5 (inconsistent increments)
- **Phase labels**: phase3 (from claim review context)
- **Fix references**: lq_fix (from specific bug fix)
- **Section references**: p305 (section/page identifier)
- **Semantic labels**: revised, complete

### 1.3 What Works

- v-numbers provide rough ordering
- Some context embedded in names
- Files are generally findable

### 1.4 What Doesn't Work

- No link to git commits
- No link to data versions
- No indication of review/approval status
- No machine-readable metadata
- Duplicates without tracking
- "complete" and "final" labels meaningless without context
- Cannot answer: "What data version produced this PDF?"

---

## 2. Proposed Versioning Strategy

### 2.1 Naming Convention

**Format**:
```
article-{MAJOR}.{MINOR}.{PATCH}-{STATUS}_{TIMESTAMP}.pdf
```

**Examples**:
```
article-0.1.0-draft_20251229T140000.pdf
article-0.2.0-draft_20251230T093000.pdf
article-0.3.0-review_20251230T160000.pdf
article-0.4.0-draft_20251231T110000.pdf
article-0.5.0-approved_20251231T180000.pdf
article-1.0.0-production_20260115T090000.pdf
```

**Version Number Semantics**:
- **MAJOR**: Significant structural changes, new sections, major rewrites
- **MINOR**: Content additions, methodology updates, new results
- **PATCH**: Typo fixes, formatting, minor corrections

**Status Codes**:
| Status | Meaning |
|--------|---------|
| `draft` | Work in progress, not for review |
| `review` | Ready for internal/external review |
| `approved` | Review complete, approved for next stage |
| `production` | Final version for submission/publication |
| `archived` | Superseded, kept for reference |

### 2.2 Metadata Approach: Sidecar JSON Files

Each PDF gets a companion `.metadata.json` file:

```json
{
  "version": {
    "major": 0,
    "minor": 5,
    "patch": 0,
    "status": "approved",
    "timestamp": "2025-12-31T18:00:00Z"
  },
  "content": {
    "sha256": "abc123...",
    "page_count": 42,
    "word_count": 12500
  },
  "source": {
    "latex_commit": "4d2080b",
    "sections_hash": "def456...",
    "figures_hash": "ghi789..."
  },
  "dependencies": {
    "data_version": "2025-12-31",
    "data_fixes_applied": ["P3.05_LQ_fix", "P3.07_migration_correction"],
    "analysis_modules": {
      "regime_models": "1.2.0",
      "validation": "1.0.3"
    },
    "commits": {
      "data_pipeline": "f28b26f",
      "analysis_scripts": "bb13f85"
    }
  },
  "review": {
    "external_reviewers": ["ChatGPT 5.2 Pro"],
    "review_date": "2026-01-01",
    "approval_authority": "Principal Investigator"
  },
  "claims": {
    "claims_manifest_version": "v3_phase3",
    "total_claims": 452,
    "verified_claims": 389
  },
  "changelog": [
    {
      "version": "0.5.0",
      "date": "2025-12-31",
      "tasks": ["P3.05 complete", "Claim verification 86%"],
      "commits": ["4d2080b", "473d703"]
    }
  ],
  "status_transitions": {
    "draft_to_review": "2025-12-30T16:00:00Z",
    "review_to_approved": "2025-12-31T18:00:00Z"
  }
}
```

### 2.3 Directory Structure

```
sdc_2024_replication/scripts/statistical_analysis/journal_article/
├── output/
│   ├── versions/
│   │   ├── working/                    # Current drafts
│   │   │   ├── article-0.6.0-draft_20260101T100000.pdf
│   │   │   └── article-0.6.0-draft_20260101T100000.metadata.json
│   │   │
│   │   ├── approved/                   # Review-approved versions
│   │   │   ├── article-0.5.0-approved_20251231T180000.pdf
│   │   │   └── article-0.5.0-approved_20251231T180000.metadata.json
│   │   │
│   │   └── production/                 # Submission-ready versions
│   │       └── (empty until submission)
│   │
│   ├── archive/
│   │   ├── 2025_12_phase3_revisions/
│   │   │   ├── ARCHIVE_INFO.md
│   │   │   ├── article-0.1.0-draft_20251229T140000.pdf
│   │   │   └── ... (older versions)
│   │   │
│   │   └── 2026_01_pre_submission/     # Next archive era
│   │
│   ├── VERSIONS.md                     # Human-readable version index
│   ├── CURRENT_VERSION.txt             # Points to current production version
│   └── CHANGELOG.md                    # Full revision history
│
├── source/
│   ├── main.tex
│   ├── sections/
│   └── figures/
│
└── compile.sh                          # Updated to auto-version
```

### 2.4 Tracking "Which Version is Current"

**CURRENT_VERSION.txt**:
```
# Current production version
article-0.5.0-approved_20251231T180000.pdf

# Current working version
article-0.6.0-draft_20260101T100000.pdf
```

**VERSIONS.md**:
```markdown
# Article Version Index

## Production Versions
| Version | Date | Status | Commit | Notes |
|---------|------|--------|--------|-------|
| 0.5.0 | 2025-12-31 | approved | 4d2080b | P3.05 complete, external review passed |

## Working Versions
| Version | Date | Status | Branch | Notes |
|---------|------|--------|--------|-------|
| 0.6.0 | 2026-01-01 | draft | master | Phase B methodology updates |

## Archived Versions
See: `archive/2025_12_phase3_revisions/ARCHIVE_INFO.md`
```

---

## 3. Auditability Infrastructure

### 3.1 Chain of Custody

```
PDF Version
    ↓ (SHA256 in metadata)
Metadata JSON
    ↓ (commits listed)
Git Commits
    ↓ (files changed)
Data/Analysis Versions
    ↓ (test results referenced)
Test Results
```

### 3.2 Query Capabilities

The metadata structure enables queries like:

| Query | How to Answer |
|-------|---------------|
| "What data version for article 0.5.0?" | `metadata.dependencies.data_version` |
| "What fixes applied to article 0.5.0?" | `metadata.dependencies.data_fixes_applied` |
| "What changed between 0.4.0 and 0.5.0?" | Compare `metadata.changelog` entries |
| "What articles use data fix P3.05?" | Search all metadata for fix in `data_fixes_applied` |
| "When was external review?" | `metadata.review.review_date` |

### 3.3 CHANGELOG.md Structure

```markdown
# Article Changelog

## [0.5.0] - 2025-12-31
### Added
- Phase A methodology assessment complete
- External review by ChatGPT 5.2 Pro
- 86% claim verification coverage

### Changed
- P3.05 LQ data fix applied
- Updated regression tables with corrected data

### Dependencies
- Data version: 2025-12-31
- Analysis commit: 4d2080b

### Review
- External reviewer: ChatGPT 5.2 Pro
- Decision: Option C (Hybrid approach) approved

---

## [0.4.0] - 2025-12-31
...
```

---

## 4. Repository Hygiene Recommendations

### 4.1 What Should Be in Git

**YES - Commit these**:
- All `.metadata.json` files (small, critical for auditability)
- `VERSIONS.md`, `CHANGELOG.md`, `CURRENT_VERSION.txt`
- LaTeX source files (`main.tex`, `sections/*.tex`, `references.bib`)
- Figures source (if vector/code-generated)
- Test results (structured JSON)
- Compile scripts

**NO - Exclude from Git**:
- All `.pdf` files (tracked via metadata, stored externally or locally)
- LaTeX intermediates (`.aux`, `.bbl`, `.log`, `.synctex.gz`)
- Large binary figures (store originals elsewhere, commit references)

### 4.2 .gitignore Updates

```gitignore
# Article PDFs (tracked via metadata, not binary)
sdc_2024_replication/scripts/statistical_analysis/journal_article/output/**/*.pdf

# LaTeX intermediates
*.aux
*.bbl
*.blg
*.log
*.out
*.toc
*.synctex.gz
```

### 4.3 Archive Policy

| Age | Action |
|-----|--------|
| Current + 1 prior | Keep in `versions/` |
| 2-4 versions old | Move to `archive/{era}/` |
| > 6 months old | Compress and move to backup storage |

**Archive Era Naming**:
```
2025_12_phase3_revisions/
2026_01_pre_submission/
2026_02_submitted/
2026_03_revision_1/
```

---

## 5. Files to Create/Modify

### 5.1 New Files to Create

| File | Purpose | Priority |
|------|---------|----------|
| `output/versions/.gitkeep` | Create directory structure | High |
| `output/versions/working/.gitkeep` | Working versions folder | High |
| `output/versions/approved/.gitkeep` | Approved versions folder | High |
| `output/versions/production/.gitkeep` | Production versions folder | High |
| `output/archive/.gitkeep` | Archive root | High |
| `output/VERSIONS.md` | Version index | High |
| `output/CURRENT_VERSION.txt` | Current version pointer | High |
| `output/CHANGELOG.md` | Full revision history | High |
| `MANUSCRIPT_VERSIONING.md` | Strategy documentation | Medium |
| `scripts/versioning/create_version.py` | CLI tool to create versions | Medium |
| `scripts/versioning/query_versions.py` | Query metadata | Medium |
| `scripts/versioning/archive_version.py` | Archive old versions | Low |
| `docs/adr/020-manuscript-versioning.md` | ADR for this decision | Medium |

### 5.2 Files to Modify

| File | Change | Priority |
|------|--------|----------|
| `.gitignore` | Add PDF exclusions | High |
| `compile.sh` | Add versioning on compile | Medium |
| `main.tex` | Add git commit/timestamp to preamble | Low |

### 5.3 Migration Tasks

| Task | Description |
|------|-------------|
| Rename existing PDFs | Apply new naming convention |
| Create metadata for existing | Generate `.metadata.json` for each |
| Move old versions to archive | Organize by era |
| Update DEVELOPMENT_TRACKER | Reference new structure |

---

## 6. Implementation Phases

### Phase A: Foundation (Estimated: 2-3 hours)

1. Create directory structure
2. Create `VERSIONS.md`, `CURRENT_VERSION.txt`, `CHANGELOG.md`
3. Update `.gitignore`
4. Write metadata for current version (0.5.0)
5. Document strategy in `MANUSCRIPT_VERSIONING.md`

### Phase B: Migration (Estimated: 1-2 hours)

1. Rename existing PDFs to new convention
2. Create metadata files for each
3. Move old versions to archive
4. Create `ARCHIVE_INFO.md` for first archive era

### Phase C: Tooling (Estimated: 3-4 hours)

1. Create `create_version.py` CLI
2. Integrate with `compile.sh`
3. Create `query_versions.py` for auditability queries
4. Test full workflow

### Phase D: Documentation (Estimated: 1 hour)

1. Create ADR-020 documenting decision
2. Update DEVELOPMENT_TRACKER with new process
3. Update AGENTS.md with versioning guidance

---

## 7. Benefits and Risks

### Benefits

| Benefit | Impact |
|---------|--------|
| Complete auditability | Can trace any PDF to exact data/code versions |
| Clear status tracking | Know which version is current, approved, archived |
| No repository bloat | PDFs excluded from git, only metadata tracked |
| Query capability | Answer "what changed" and "what depends on what" |
| External review integration | Review dates and decisions recorded |

### Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Metadata drift from PDF | SHA256 checksum verification |
| Forgetting to create metadata | Automated via `create_version.py` |
| Lost PDFs | Regular backup to external storage |
| Complex workflow | CLI tools simplify; document in MANUSCRIPT_VERSIONING.md |

---

## 8. Alignment with Existing Patterns

This strategy builds on existing infrastructure:

| Existing Pattern | How We Use It |
|------------------|---------------|
| JSON manifests (`document_metadata.json`) | Same structure for article metadata |
| `REVISION_STATUS.md` tracking | Becomes `CHANGELOG.md` |
| Archive folders (`revision_outputs/archive/`) | Same era-based structure |
| Semantic versioning (`version.py`) | Same MAJOR.MINOR.PATCH scheme |
| Configuration files (`projection_config.yaml`) | Reference pattern for metadata schema |

---

## Summary

This versioning strategy provides:

1. **Clear naming**: `article-{VERSION}-{STATUS}_{TIMESTAMP}.pdf`
2. **Rich metadata**: JSON sidecar files linking to git, data, tests, reviews
3. **Auditability**: Complete chain of custody from PDF to source data
4. **Repository hygiene**: PDFs excluded from git, metadata tracked
5. **Workflow integration**: CLI tools for version creation and queries
6. **Archive strategy**: Era-based folders with retention policy

**Decision Required**: Approve this strategy to proceed with implementation.
