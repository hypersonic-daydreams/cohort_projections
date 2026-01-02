# ADR-019: Claim and Argument Mapping Review Process

## Status
Accepted

## Date
2025-12-31

## Last Reviewed
2025-12-31

## Context

We need a systematic way to verify the journal article's factual claims and the logical structure that connects them. A claim-only list can miss missing warrants, weak support chains, or unsupported leaps in reasoning. The review output must be machine-readable for AI agents and usable for human authors.

### Requirements
- Capture every discrete claim with a stable ID and source location.
- Record preferred evidence types for verification work.
- Map argument structure using a consistent framework.
- Keep outputs in an isolated workspace tied to a specific PDF snapshot.
- Support both structured data (JSONL) and human-friendly visualizations.

### Challenges
- Argument mapping is interpretive; we need consistent rules and schemas.
- Claims and argument nodes must stay linked to avoid drift.
- Tooling should avoid new dependencies and remain lightweight.

## Decision

### Decision 1: Claim Manifest as Canonical Record

**Decision**: Use a JSONL claim manifest as the canonical inventory of claims, with support-type annotations and status tracking.

**Rationale**:
- JSONL is easy for AI agents to parse and for humans to review.
- Structured fields enable assignment, verification, and auditing.

**Implementation**:
- `claim_review/v3_phase3/claims/claims_manifest.jsonl`
- `claim_review/v3_phase3/claims/claim_schema.json`

**Alternatives Considered**:
- Spreadsheet-only tracking: rejected due to weak linkage to automation.
- Free-form notes: rejected due to poor traceability.

### Decision 2: Toulmin Argument Map with Graphviz Exports

**Decision**: Map arguments using Toulmin roles (claim, grounds, warrant, backing, qualifier, rebuttal) in JSONL, and generate Graphviz DOT graphs for visualization.

**Rationale**:
- Toulmin provides a clear, practical model for academic reasoning.
- DOT graphs give a human-readable overview without complex tooling.

**Implementation**:
- `claim_review/v3_phase3/argument_map/argument_map.jsonl`
- `claim_review/v3_phase3/argument_map/argument_schema.json`
- `claim_review/v3_phase3/argument_map/generate_argument_graphs.py`

**Alternatives Considered**:
- Full AIF/argumentation frameworks: too heavy for current needs.
- Narrative-only argument summaries: insufficient for verification workflows.

### Decision 3: Citation Integrity + APA 7th Completeness Audit

**Decision**: Add a citation audit step that checks LaTeX in-text citation keys against BibTeX entries, flags missing or uncited references, and evaluates APA 7th metadata completeness per entry.

**Rationale**:
- Ensures every in-text citation resolves to the reference list.
- Identifies orphaned references before submission.
- Supports APA compliance by flagging missing required metadata.

**Implementation**:
- `claim_review/v3_phase3/citation_audit/check_citations.py`
- `claim_review/v3_phase3/citation_audit/README.md`
- `claim_review/v3_phase3/citation_audit/citation_entries.jsonl` (generated)
- `claim_review/v3_phase3/citation_audit/citation_entry_schema.json`

**Alternatives Considered**:
- Manual spot checks: too error-prone for a long reference list.
- PDF-only parsing: weaker linkage to BibTeX keys.

## Consequences

### Positive
1. Clear linkage between claims, evidence needs, and logical structure.
2. Reusable review process for future papers.
3. Supports both AI-driven verification and human oversight.

### Negative
1. Additional upfront effort to map arguments consistently.
2. Requires reviewer calibration to avoid inconsistent role assignments.

### Risks and Mitigations

**Risk**: Inconsistent argument role labeling across reviewers.
- **Mitigation**: Use `argument_guidelines.md` and pilot mapping on initial sections.

## Alternatives Considered

### Alternative 1: Claim-Only Review
**Pros**: Faster to produce.
**Cons**: Misses missing warrants and logical gaps.
**Why Rejected**: Insufficient for logic verification goals.

### Alternative 2: Full Formal Logic Mapping
**Pros**: Precise.
**Cons**: High complexity and tooling overhead.
**Why Rejected**: Overkill for journal article review.

## Implementation Notes

### Key Scripts
- `generate_argument_graphs.py`: Generates DOT graphs from `argument_map.jsonl`.

### Configuration Integration
- No changes to `projection_config.yaml`. The review process is isolated under the claim review workspace.

### Testing Strategy
- Validate JSONL structure against schemas.
- Spot-check DOT outputs visually after generation.

## References

1. `claim_review/v3_phase3/README.md`
2. `claim_review/v3_phase3/argument_map/ARGUMENTATION_METHOD.md`
3. `claim_review/v3_phase3/claims/claim_schema.json`

## Revision History

- **2025-12-31**: Initial version (ADR-019) - Claim and argument mapping review process.
- **2025-12-31**: Add citation integrity + APA 7th completeness audit decision and tooling.

## Related ADRs

- ADR-017: SDC 2024 Methodology Comparison and Scenario
- ADR-018: Immigration Policy Scenario Methodology
