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
3. Run QA checks on the manifest to catch structural issues and parsing artifacts.
4. Add support-type annotations for each claim (primary and alternative support).
5. Build argument maps and link them to claim IDs.
6. Run citation audit to verify in-text keys and reference list alignment.
7. Assign claims to agents for verification.
8. Record evidence in `evidence/` and update claim statuses.

## Claim Parsing Tools

### Generalized Section Parser (`claims/build_section_claims.py`)
Regenerates claims for any section from extracted text:

```bash
# Preview claims for a section (dry-run)
python claims/build_section_claims.py --section "Data and Methods" --dry-run

# Update manifest with regenerated claims (preserves claim IDs)
python claims/build_section_claims.py --section Introduction --write

# Available sections: Abstract, Introduction, Data and Methods, Results,
#                     Discussion, Conclusion, Appendix
```

### Introduction Parser (`claims/build_intro_claims.py`)
Original Introduction-specific parser (legacy, superseded by generalized parser):

```bash
python claims/build_intro_claims.py --write
```

## QA Checks

### All Sections
```bash
# Basic QA on all sections
python claims/qa_claims.py --all-sections

# QA with parser comparison (shows manifest vs parser discrepancies)
python claims/qa_claims.py --all-sections --compare-section
```

### Single Section
```bash
# Section-specific QA
python claims/qa_claims.py --section "Results" --compare-section

# Introduction with legacy comparison
python claims/qa_claims.py --section Introduction --expected-pages 3 4 5 --compare-intro
```

### General Manifest Scan
```bash
python claims/qa_claims.py
```
