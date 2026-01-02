# Agent B5: ADR-019 Decision Documentation

## Document Information

| Field | Value |
|-------|-------|
| Agent ID | B5 |
| Scope | Update ADR with external review findings and final decision |
| Status | Planning Complete |
| Created | 2026-01-01 |

---

## 1. Current State Assessment

### 1.1 ADR Naming Conflict Discovery

**Critical Finding**: ADR-019 is used twice:

| File | Purpose | Status |
|------|---------|--------|
| `019-argument-mapping-claim-review-process.md` | Claim review methodology | **Accepted** (in README) |
| `019-extended-time-series-methodology-analysis.md` | Time series extension | **Proposed** (not in README) |
| `019a-vintage-methodology-investigation-plan.md` | Sub-agent investigation plan | Investigation plan |

**Resolution Required**: Rename time series ADR to `020-extended-time-series-methodology-analysis.md`

### 1.2 Current State of Time Series ADR

File: `docs/adr/019-extended-time-series-methodology-analysis.md` (to become 020)

**Current Content** (238 lines):
- Status: **PROPOSED**
- Documents investigation question (extending n=15 to n=25)
- Lists validity risks (5 risks)
- Describes Phase 1-3 structure
- Lists 4 decision options (A, B, C, D)
- Defines 5 sub-agent roles

**What's Missing**:
- External review results
- Decision rationale (Option C selected)
- Conditions for defensibility
- Implementation roadmap
- Links to Phase A/B artifacts
- Status update (should become Accepted)

### 1.3 Current State of ADR-019a

File: `docs/adr/019a-vintage-methodology-investigation-plan.md` (382 lines)

**Status**: Remains as investigation plan; findings documented in agent reports.

### 1.4 ADR Format/Template

From `docs/adr/TEMPLATE.md`:
```
# ADR-NNN: Title
## Status
## Date
## Context
## Decision
## Consequences
## Alternatives Considered
## Implementation Notes
## References
## Revision History
## Related ADRs
```

---

## 2. ADR Update Plan

### 2.1 Renaming Plan

| Current | New |
|---------|-----|
| `019-extended-time-series-methodology-analysis.md` | `020-extended-time-series-methodology-analysis.md` |
| `019a-vintage-methodology-investigation-plan.md` | `020a-vintage-methodology-investigation-plan.md` |
| `020-reports/` directory | `020-reports/` directory |

### 2.2 Sections to Add/Modify

| Section | Action | Content |
|---------|--------|---------|
| **Status** | Modify | "PROPOSED" → "**Accepted**" |
| **Date** | Modify | Update to 2026-01-01 |
| **External Review Summary** | ADD NEW | ChatGPT 5.2 Pro findings |
| **Decision** | ADD NEW | Option C selection |
| **Conditions for Defensibility** | ADD NEW | 4 conditions from external review |
| **Implementation Roadmap** | ADD NEW | Reference Phase B plans |
| **Consequences** | ADD NEW | Positive, negative, risks |
| **References** | Expand | Links to Phase A/B artifacts |
| **Revision History** | ADD NEW | Decision timeline |
| **Related ADRs** | ADD NEW | Links to ADR-017, 018, 019 |

### 2.3 Decision Record Structure

```markdown
## Decision

### External Review Summary
On 2026-01-01, ChatGPT 5.2 Pro validated Phase A findings...

### Decision 1: Adopt Option C (Hybrid Approach)

**Decision**: Primary inference on n=15 (2010-2024); extended series for robustness only.

**Rationale**: [6 bullet points from external review]

**Conditions for Defensibility**:
1. Declare three measurement systems honestly
2. Treat vintage transitions as regime boundaries
3. Make identification explicit (model-based correction)
4. Keep primary claims in within-regime variation

**Implementation**: See Phase B agent plans (B1-B6)
```

---

## 3. Decision Documentation Content

### 3.1 Option C Rationale

From external review:
1. Option C dominates on "reviewer defensibility per unit of pain"
2. Option B risks accepting multi-regime as single process without controls
3. Option A is assumption-driven without overlap years
4. Option D is defensible but suboptimal

### 3.2 Alternatives Considered

| Option | Description | Why Rejected |
|--------|-------------|--------------|
| **A** | Corrections | Assumption-driven without overlap years |
| **B** | Caveats only | Easy target in peer review |
| **D** | Maintain n=15 | Power limitations; also contains COVID shock |

### 3.3 Key Quote to Include

> "If you follow this path, you're not 'breaking the rules' of the Census warning - you're acknowledging that the rules exist because measurement changes are real, then doing what good methods papers do: modeling those changes as part of the inferential problem rather than pretending they aren't there."

---

## 4. Files Inventory

### 4.1 Files to Rename

| Current | New |
|---------|-----|
| `docs/adr/019-extended-time-series-methodology-analysis.md` | `020-...` |
| `docs/adr/019a-vintage-methodology-investigation-plan.md` | `020a-...` |
| `docs/adr/020-reports/` | `020-reports/` |

### 4.2 Files to Modify

| File | Changes |
|------|---------|
| `020-extended-time-series-methodology-analysis.md` | Add Decision, update status |
| `020a-vintage-methodology-investigation-plan.md` | Update references |
| `docs/adr/README.md` | Add ADR-020 to index |
| `DEVELOPMENT_TRACKER.md` | Reference ADR-020 |

### 4.3 Cross-References to Update

All references to `020-reports/` in:
- ADR-020 files (after rename)
- All files in `020-reports/`
- `PHASE_B_SUBAGENT_PLANNING.md`
- Agent reports

### 4.4 Links to Supporting Artifacts

| Artifact | Path |
|----------|------|
| Phase A Reports | `docs/adr/020-reports/AGENT_{1,2,3}_REPORT.md` |
| External Review | `docs/adr/020-reports/chatgpt_review_package/chatgpt_response.md` |
| Phase B Plans | `docs/adr/020-reports/phase_b_plans/AGENT_B{1-6}_PLAN.md` |

---

## 5. Content Outline

### 5.1 Section-by-Section for ADR-020 Updates

| Section | Lines | Content |
|---------|-------|---------|
| **Status** | 1 | Change to "Accepted" |
| **Date** | 1 | 2026-01-01 |
| **External Review Summary** (NEW) | 20 | ChatGPT 5.2 Pro findings |
| **Decision 1: Option C** (NEW) | 40 | Decision, rationale, conditions |
| **Alternatives Considered** (EXPAND) | 25 | Options A, B, D rejection reasons |
| **Consequences** (NEW) | 30 | Positive, negative, risks |
| **Implementation Notes** (NEW) | 15 | Reference Phase B plans |
| **References** (EXPAND) | 15 | Links to artifacts |
| **Revision History** (NEW) | 10 | Timeline |
| **Related ADRs** (NEW) | 5 | Links |
| **TOTAL NEW** | ~160 | Added to existing 238 lines |

### 5.2 Key Terminology

| Term | Definition |
|------|------------|
| **Spliced PEP-vintage series** | Time series bridging measurement regimes |
| **Measurement regime** | Period with consistent methodology |
| **Primary inference window** | 2010-2024 (within-regime) |
| **Extended series** | 2000-2024 (cross-regime, robustness only) |
| **Option C (Hybrid)** | Primary n=15, extended for robustness |

### 5.3 README.md Update

```markdown
| [020](020-extended-time-series-methodology-analysis.md) | Extended Time Series Methodology | Accepted | 2026-01-01 | Hybrid approach for 2000-2024 vintage series |
```

---

## 6. Dependencies

### 6.1 What B5 Needs

| Agent | What B5 Needs | Status |
|-------|---------------|--------|
| Phase A | Agent reports | Available |
| External Review | ChatGPT response | Available |
| B0a-B4 | Implementation plans | Available |

### 6.2 What Others Need from B5

| Consumer | What They Need |
|----------|----------------|
| B3 (Journal Article) | Official Option C framing |
| B6 (Testing) | Acceptance criteria |
| Future developers | Decision rationale |

---

## 7. Estimated Complexity

| Component | Complexity |
|-----------|------------|
| File renaming | **LOW** |
| Cross-reference updates | **MEDIUM** |
| Decision section | **MEDIUM** |
| Consequences section | **LOW** |
| README update | **LOW** |
| **Overall** | **MEDIUM** |

---

## 8. Implementation Sequence

1. Rename files: 019 → 020
2. Update cross-references
3. Add Decision section
4. Add Consequences section
5. Update References
6. Add Revision History
7. Update README.md
8. Verify links

---

## 9. Risks and Blockers

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Broken cross-references | MEDIUM | Systematic find-replace |
| Missing artifacts | LOW | All exist |

---

## Summary

This plan provides:

1. **Discovery of ADR naming conflict** with resolution (rename to 020)
2. **Complete ADR update strategy** for Option C decision
3. **Section-by-section outline** with content summary
4. **File inventory** for renaming and modification
5. **Terminology guide** for consistency
6. **Dependency mapping** with other agents

**Key Finding**: ADR-019 is used for two purposes. Time series investigation should be ADR-020.

**Decision Required**: Approve renaming to ADR-020.
