# Claim Review Workspace - v5_p305_complete Draft

Purpose: Track every discrete claim in the v5_p305_complete PDF, assign verification work, and record evidence or revisions.

Canonical data file: `claims/claims_manifest.jsonl`

AI agent quick start: `AI_AGENT_GUIDE.md`

## Structure
- `source/` - Snapshot of the PDF under review
- `document_metadata.json` - Hash and metadata for the snapshot
- `extracted/` - Text extraction outputs with page anchors
- `claims/` - Claim schema, guidelines, and manifest files
- `argument_map/` - Argument mapping schema, guidelines, and graphs
- `citation_audit/` - Citation integrity checks (APA 7th completeness)
- `assignments/` - Agent assignments and batch definitions
- `evidence/` - Evidence notes per claim (`C####.md`)
- `exports/` - Dashboards or status summaries for reporting

## Workflow (High Level)
1. Extract text with page anchors into `extracted/`.
2. Segment text into discrete claims and populate `claims_manifest.jsonl`.
3. Add support-type annotations for each claim (primary and alternative support).
4. Build argument maps and link them to claim IDs.
5. Run citation audit to verify in-text keys and reference list alignment.
6. Assign claims to agents for verification.
7. Record evidence in `evidence/` and update claim statuses.
