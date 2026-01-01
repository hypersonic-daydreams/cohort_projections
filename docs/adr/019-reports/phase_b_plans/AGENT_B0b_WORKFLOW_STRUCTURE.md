# Agent B0b: Sprint/Wave Workflow Structure

## Document Information

| Field | Value |
|-------|-------|
| Agent ID | B0b |
| Scope | Repository organization and iterative development workflow |
| Status | Planning Complete |
| Created | 2026-01-01 |

---

## 1. Current State Assessment

### 1.1 How Work is Currently Organized

The project demonstrates a sophisticated **multi-phase, agent-driven workflow model**:

**Current Phase Structure**:
- **Phase A** (Completed 2025-12-31): Validity risk assessment using parallel agents
- **Phase B** (Starting 2026-01-01): Implementation with planning layer first
- **Sub-phases**: v3_phase3 claim review workspace within journal article work

**Session-Based Organization**:
- Work organized by date in DEVELOPMENT_TRACKER.md
- Each session has focus areas and accomplishments
- Related work grouped by thematic waves (Wave 1-5 pattern from data pipeline)

**Key Organizing Principles Already in Use**:
1. Parallel agent structure for independent tasks
2. Artifact standardization (REPORT_TEMPLATE.md)
3. External review integration (ChatGPT 5.2 Pro)
4. Planning before implementation

### 1.2 Existing Sprint-Like Patterns

**Pattern 1: Phased Investigations**
- ADR-019a specifies Phase A/B structure
- Clear gates between phases
- External review as quality gate

**Pattern 2: Wave-Based Execution**
- 2025-12-28 data pipeline used 5 sequential waves
- Waves are sequential with internal parallelization

**Pattern 3: Version-Based Iterations**
- Journal article: v5_p305_complete
- Claim review workspace: v3_phase3
- Separate versioning tracks for documents vs. workspaces

**Pattern 4: Sub-Agent Task Batches**
- PHASE_B_SUBAGENT_PLANNING.md defines 6 agents (B1-B6)
- Planning documents before implementation
- Synthesis step after completion

### 1.3 What Tracking Exists

| Tracker | Location | Purpose |
|---------|----------|---------|
| DEVELOPMENT_TRACKER.md | Project root | Overall status, session logs |
| STATUS.md files | Within workspaces | Phase-specific progress |
| ADRs | `docs/adr/` | Decision records |
| Planning documents | `docs/adr/019-reports/` | Phase planning |

### 1.4 Pain Points

1. **Ambiguous sprint boundaries** - No formal sprint naming
2. **Scattered status** - Multiple tracking locations
3. **No cross-sprint visibility** - Hard to see artifact ownership
4. **Dependency tracking** - No formal mechanism
5. **Archive strategy** - Unclear what happens when phases complete

---

## 2. Proposed Sprint/Wave Structure

### 2.1 Terminology

| Term | Definition | Example |
|------|------------|---------|
| **Phase** | Meta-level grouping (research stage) | Phase A, Phase B |
| **Sprint** | Bounded unit of work (1-4 weeks) | SPRINT_METHODOLOGY_A_001 |
| **Wave** | Subdivision within sprint (thematic/sequential) | WAVE_1_DATA_FETCH |
| **Workspace** | Dedicated directory for sprint artifacts | `docs/adr/019-reports/PHASE_A/` |

### 2.2 Naming Conventions

**Sprint Naming**:
```
SPRINT_{CONTEXT}_{PHASE}_{SEQUENCE}

Examples:
SPRINT_METHODOLOGY_A_001     # Phase A methodology validation
SPRINT_ANALYSIS_B_001        # Phase B implementation
SPRINT_ARTICLE_REVIEW_001    # Article revision cycle
```

**Wave Naming**:
```
{SPRINT_ID}:WAVE_{NUMBER}_{DESCRIPTOR}

Examples:
SPRINT_ANALYSIS_B_001:WAVE_1_INFRASTRUCTURE
SPRINT_ANALYSIS_B_001:WAVE_2_MODELING
```

**Agent Naming**:
```
AGENT_{PHASE}{NUMBER}_{DESCRIPTOR}

Examples:
AGENT_A1_CENSUS_METHODOLOGY
AGENT_B1_STATISTICAL_MODELING
AGENT_B0a_VERSIONING_ANALYSIS
```

### 2.3 Documentation Requirements Per Sprint

Every sprint must produce:

| Document | Purpose | When Created |
|----------|---------|--------------|
| `{SPRINT_ID}_PLAN.md` | Objectives, waves, assignments | Sprint start |
| `{SPRINT_ID}_STATUS.md` | Progress tracking, blockers | Updated throughout |
| `{SPRINT_ID}_ARTIFACTS.md` | Output inventory with locations | Sprint end |
| `{SPRINT_ID}_RETROSPECTIVE.md` | Lessons learned | Sprint end |

### 2.4 Relationship to ADRs

| Sprint Type | ADR Relationship |
|-------------|------------------|
| Decision-bearing | Creates/updates ADR |
| Implementation | References existing ADR |
| Research | May spawn new ADR if decision needed |

---

## 3. Repository Organization Recommendations

### 3.1 Proposed Directory Structure

```
docs/adr/019-reports/
├── SHARED/                               # Templates and specifications
│   ├── REPORT_TEMPLATE.md
│   ├── ARTIFACT_SPECIFICATIONS.md
│   ├── CHATGPT_BRIEFING_TEMPLATE.md
│   └── SPRINT_PLANNING_TEMPLATE.md       # NEW
│
├── PHASE_A/                              # Completed phase
│   ├── PHASE_METADATA.md                 # Phase overview
│   ├── SPRINT_METHODOLOGY_A_001/         # Sprint workspace
│   │   ├── SPRINT_PLAN.md
│   │   ├── SPRINT_STATUS.md
│   │   ├── SPRINT_ARTIFACTS.md
│   │   ├── SPRINT_RETROSPECTIVE.md
│   │   ├── agents/                       # Agent outputs
│   │   │   ├── AGENT_1_REPORT.md
│   │   │   └── agent1_*.{csv,json}
│   │   ├── synthesis/                    # Synthesis outputs
│   │   │   └── synthesis_*.{csv,json}
│   │   └── external_review/              # External review
│   │       ├── CHATGPT_BRIEFING.md
│   │       └── CHATGPT_RESPONSE.md
│   └── PHASE_SUMMARY.md                  # After phase complete
│
├── PHASE_B/
│   ├── PHASE_METADATA.md
│   ├── PLANNING/                         # Planning phase (current)
│   │   ├── AGENT_B0a_*.md
│   │   ├── AGENT_B1_PLAN.md
│   │   └── ...
│   ├── SPRINT_ANALYSIS_B_001/            # First execution sprint
│   │   └── ... (to be created)
│   └── SPRINT_ANALYSIS_B_002/            # Subsequent sprints
│
└── ARCHIVE/                              # Completed phases
    ├── INDEX.md
    └── PHASE_A_20260101/                 # Archived with date
```

### 3.2 Sprint-to-Article Mapping

| Article Version | Sprint(s) | Relationship |
|-----------------|-----------|--------------|
| v5_p305_complete | SPRINT_METHODOLOGY_A_001 | Analysis validated this version |
| v6 (future) | SPRINT_ANALYSIS_B_* | Will incorporate Phase B results |

**Key Principle**: When article version changes significantly, create new sprint for that revision cycle.

### 3.3 Archive Strategy

| Trigger | Action |
|---------|--------|
| Phase complete | Create `PHASE_SUMMARY.md`, move to `ARCHIVE/` |
| Sprint complete | Create `SPRINT_RETROSPECTIVE.md`, keep in phase folder |
| > 6 months old | Compress and move to backup |

---

## 4. Progress Tracking System

### 4.1 Multi-Level Tracking

```
DEVELOPMENT_TRACKER.md          (Project-wide: phases, active sprints)
    ↓
PHASE_METADATA.md               (Phase-level: sprints within phase)
    ↓
SPRINT_STATUS.md                (Sprint-level: waves, daily progress)
    ↓
AGENT_ASSIGNMENTS.md            (For multi-agent sprints)
```

### 4.2 DEVELOPMENT_TRACKER.md Updates

Add new section:

```markdown
## Current Sprint Status

| Sprint | Phase | Status | Progress | Notes |
|--------|-------|--------|----------|-------|
| SPRINT_ANALYSIS_B_PLANNING | B | Active | 30% | Planning agents running |
| SPRINT_ANALYSIS_B_001 | B | Pending | 0% | Awaits planning completion |

## Active Phase
**Phase B**: Correction Methods Implementation
- See: `docs/adr/019-reports/PHASE_B/PHASE_METADATA.md`

## Completed Phases
- **Phase A**: Validity Assessment (2025-12-31)
  - See: `docs/adr/019-reports/ARCHIVE/PHASE_A_20260101/PHASE_SUMMARY.md`
```

### 4.3 Sub-Agent Tracking Template

```markdown
# AGENT_ASSIGNMENTS.md

## Agent Roster
| Agent | Scope | Deliverable | Status |
|-------|-------|-------------|--------|
| B0a | Versioning | AGENT_B0a_VERSIONING_ANALYSIS.md | ✓ Complete |
| B0b | Workflow | AGENT_B0b_WORKFLOW_STRUCTURE.md | ✓ Complete |
| B1 | Statistical Modeling | AGENT_B1_PLAN.md | ⧐ Pending |
| B2 | Multi-State Analysis | AGENT_B2_PLAN.md | ⧐ Pending |
| ... | ... | ... | ... |

## Dependency Graph
B0a, B0b (infrastructure) → B1-B6 (content planning) → Synthesis
```

### 4.4 External Review Integration

Track in sprint status:

```markdown
## External Review Gate

**Required**: Yes (end of Planning phase)
**Materials Due**: After all B1-B6 plans complete
**Review Package**: `docs/adr/019-reports/PHASE_B/external_review/`

### Checklist
- [ ] All agent plans complete
- [ ] Synthesis document created
- [ ] Review package assembled
- [ ] External review initiated
- [ ] Response received
- [ ] Decision incorporated
```

---

## 5. Sprint Lifecycle

### 5.1 Initiation Checklist

**Before Sprint Start**:
- [ ] Create sprint directory structure
- [ ] Write SPRINT_PLAN.md with objectives, waves, success criteria
- [ ] Create SPRINT_STATUS.md template
- [ ] Define AGENT_ASSIGNMENTS.md (if multi-agent)
- [ ] Update DEVELOPMENT_TRACKER.md
- [ ] Get stakeholder approval

### 5.2 In-Progress Management

**Daily/Regular Updates**:
- [ ] Update SPRINT_STATUS.md
- [ ] Log session notes with timestamp
- [ ] Flag blockers immediately

**Mid-Sprint Review**:
- [ ] Assess progress vs. plan
- [ ] Adjust scope if needed
- [ ] Confirm dependencies on track

### 5.3 Completion Criteria

**Technical**:
- [ ] All wave deliverables finished
- [ ] Code passes quality checks
- [ ] Tests passing
- [ ] Artifacts documented

**Documentation**:
- [ ] SPRINT_STATUS.md final summary
- [ ] Session logs complete
- [ ] Limitations documented

**Quality Gates**:
- [ ] External review (if required)
- [ ] Synthesis complete (if multi-agent)
- [ ] All blockers resolved

### 5.4 Retrospective

Create `SPRINT_RETROSPECTIVE.md`:

```markdown
# {SPRINT_ID} Retrospective

## Objectives - Achieved?
- [x] Objective 1: YES
- [ ] Objective 2: PARTIAL (reason)

## What Went Well
1. ...

## What Could Improve
1. ...

## Learnings
- ...

## Metrics
- Planned days: X
- Actual days: Y
- Deliverables: N planned, M delivered

## Next Sprint Dependencies
- Sprint X needs artifact Y from this sprint
```

---

## 6. Files to Create/Modify

### 6.1 New Files (Immediate)

| File | Purpose | Priority |
|------|---------|----------|
| `docs/adr/019-reports/SHARED/SPRINT_PLANNING_TEMPLATE.md` | Template for sprints | High |
| `docs/adr/019-reports/PHASE_A/PHASE_METADATA.md` | Phase A overview | High |
| `docs/adr/019-reports/PHASE_A/PHASE_SUMMARY.md` | Phase A retrospective | High |
| `docs/adr/019-reports/PHASE_B/PHASE_METADATA.md` | Phase B overview | High |
| `docs/adr/019-reports/PHASE_B/PLANNING/AGENT_ASSIGNMENTS.md` | Track planning agents | High |

### 6.2 Files to Modify

| File | Change |
|------|--------|
| `DEVELOPMENT_TRACKER.md` | Add sprint status section |
| `docs/adr/019-reports/PHASE_B_SUBAGENT_PLANNING.md` | Align with sprint structure |

### 6.3 Reorganization Tasks

| Task | Description |
|------|-------------|
| Create SHARED/ folder | Move templates there |
| Create PHASE_A/ structure | Organize existing Phase A outputs |
| Rename planning folder | `phase_b_plans/` → `PHASE_B/PLANNING/` |

---

## 7. Implementation Roadmap

### Phase 1: Lightweight Adoption (This Sprint)

1. Create PHASE_METADATA.md for Phase A and B
2. Add sprint status section to DEVELOPMENT_TRACKER.md
3. Use sprint structure for Phase B planning coordination
4. **Effort**: 1-2 hours

### Phase 2: Full Structure (Phase B Execution)

1. Create full sprint directories for execution sprints
2. Use SPRINT_PLAN.md and SPRINT_STATUS.md templates
3. Test workflow through first execution sprint
4. **Effort**: 1-2 hours per sprint

### Phase 3: Retrospective (After Phase B)

1. Create PHASE_SUMMARY.md
2. Archive Phase A properly
3. Document lessons learned
4. Refine templates based on experience

---

## 8. Key Principles

### Why This Structure?

1. **Scales**: Works for single-person and multi-agent work
2. **Traceable**: Clear hierarchy (Phase → Sprint → Wave → Agent)
3. **Flexible**: Works alongside existing ADRs and tracker
4. **Retrospective-Friendly**: Built-in archival
5. **Decision-Aware**: External review integration points

### What We're NOT Doing

- No velocity metrics (not a software sprint)
- No story points (research hard to estimate)
- No formal "backlog" (planning docs replace this)
- No ceremonies beyond what's useful

### Compatibility

This structure:
- Works alongside existing DEVELOPMENT_TRACKER.md
- Complements ADR system (sprints reference ADRs)
- Enhances rather than replaces current workflows

---

## Summary

This workflow structure provides:

1. **Clear terminology**: Phase → Sprint → Wave → Agent hierarchy
2. **Consistent naming**: `SPRINT_{CONTEXT}_{PHASE}_{SEQ}` convention
3. **Required documentation**: Plan, Status, Artifacts, Retrospective per sprint
4. **Multi-level tracking**: Project → Phase → Sprint → Agent visibility
5. **Archive strategy**: Era-based folders with retention policy
6. **External review integration**: Built-in gates and tracking

**Decision Required**: Approve this structure to proceed with Phase B organization.
