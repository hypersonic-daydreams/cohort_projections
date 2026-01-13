# Claim Review Workspaces

This directory contains **versioned, self-contained workspaces** for claim-level review of journal-article PDFs.

Each workspace is intended to:
- Snapshot a specific PDF into `source/` (so the review target is stable)
- Track every discrete claim in `claims/claims_manifest.jsonl`
- Record verification evidence in `evidence/` (one file per claim ID)
- Progress section-by-section (Abstract → Introduction → Methods → Results → Discussion → …)

## Workspaces
- `v3_phase3/` - Draft claim inventory + argument map for `article_draft_v5_p305_complete.pdf`
- `v4_article-0.8.7-production_20260112_182507/` - Scaffold for claim verification of the production PDF `article-0.8.7-production_20260112_182507.pdf`

## Starting A New Workspace (Pattern)
1. Create `vN_<label>/` and subfolders (`source/`, `claims/`, `evidence/`, …).
2. Copy the target PDF into `source/` and record a SHA-256 in `document_metadata.json`.
3. Define section list + page ranges in `sections.yaml`.
4. Populate `claims/claims_manifest.jsonl` section-by-section and verify claims via `evidence/C####.md`.
