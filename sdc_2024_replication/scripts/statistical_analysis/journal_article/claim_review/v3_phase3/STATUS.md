# Claim Review Plan and Status - v5_p305_complete Draft

## Objective
Create a complete, claim-level manifest for the v5_p305_complete PDF, map the argument structure, annotate preferred evidence types, and track verification outcomes for systematic revision.

## Workflow Checklist
- [x] Scaffold claim review workspace
- [x] Snapshot PDF and record metadata hash
- [x] Scaffold argument map workspace and schema
- [x] Extract text with page anchors into `extracted/`
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

## Session Notes (2025-12-31)
- Text extraction: `mutool draw -F txt -o extracted/page-%d.txt source/article_draft_v5_p305_complete.pdf 1-61` completed; 61 page files present in `extracted/`.
- Draft claim manifest generated: `claims/claims_manifest.jsonl` currently has 1332 claims across sections (Abstract 15, Introduction 49, Data and Methods 227, Results 465, Discussion 165, Conclusion 38, Appendix 373).
- Known issues in the current manifest (needs cleanup):
  - Some claims are still too coarse and should be split (multi-clause sentences, especially in Abstract/Introduction and Results narrative).
  - Line-break hyphenation is partly fixed (common prefixes preserved), but some words still appear broken (e.g., "grav-ity" in a few claims). Extend the hyphenation preserve list or post-clean.
  - Plot/axis labels mostly filtered, but at least one mixed claim remains (e.g., a Discussion claim that includes figure axis text from page 32). Needs additional filtering for figure labels before sentence assembly.
  - Research-question list items are combined with the lead-in sentence; these should be split into separate claims per question.
  - Claim types are heuristic-only; several labels are likely off (e.g., some descriptive claims tagged as methodological/forecast).
- Draft manifest was produced via ad-hoc Python parsing in the shell (not saved to a script). If repeating, re-run extraction + rebuild with a dedicated script or rework the parser in-session.
