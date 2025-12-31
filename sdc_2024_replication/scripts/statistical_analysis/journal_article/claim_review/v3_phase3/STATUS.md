# Claim Review Plan and Status - v5_p305_complete Draft

## Objective
Create a complete, claim-level manifest for the v5_p305_complete PDF, map the argument structure, annotate preferred evidence types, and track verification outcomes for systematic revision.

## Workflow Checklist
- [x] Scaffold claim review workspace
- [x] Snapshot PDF and record metadata hash
- [x] Scaffold argument map workspace and schema
- [ ] Extract text with page anchors into `extracted/`
- [ ] Build claim manifest for Abstract + Introduction (pilot granularity check)
- [ ] Build claim manifest for remaining sections
- [ ] Add support-type annotations (primary + alternative) for all claims
- [ ] Build argument map for Abstract + Introduction (pilot structure check)
- [ ] Build argument map for remaining sections
- [ ] Export Graphviz graphs (per-argument and full-paper)
- [ ] Run citation audit (APA 7th completeness) and address missing/orphaned references
- [ ] Assign claims to agents for verification
- [ ] Collect evidence notes in `evidence/` and update statuses
- [ ] Summarize results in `exports/claims_dashboard.md`

## Notes
- Canonical data file: `claims/claims_manifest.jsonl`
- Support annotation question: "What two types of evidence or support would be most appropriate to support this claim?"
  - `support_primary` = most appropriate or robust support
  - `support_alternative` = alternative support if primary is not feasible
