# Planning Synthesis Template

Use this template when creating `PLANNING_SYNTHESIS.md` for Phase B implementation planning.

---

## Template

```markdown
# ADR-0XX Phase B: Planning Synthesis

## Document Information

| Field | Value |
|-------|-------|
| ADR | 0XX |
| Title | [Investigation Title] |
| Status | Planning / Approved / In Progress / Complete |
| Created | YYYY-MM-DD |
| Last Updated | YYYY-MM-DD |

---

## 1. Executive Summary

[2-3 sentence summary of what Phase B will accomplish]

---

## 2. Phase A Findings Summary

### Key Discoveries
1. [Finding 1]
2. [Finding 2]
3. [Finding 3]

### Implications for Implementation
- [Implication 1]
- [Implication 2]

---

## 3. Implementation Phases

### Phase 1: Infrastructure (B0)

| Agent | Scope | Priority | Dependencies |
|-------|-------|----------|--------------|
| B0a | [Scope] | HIGH | None |
| B0b | [Scope] | HIGH | B0a |

### Phase 2: Core Implementation (B1, B2)

| Agent | Scope | Priority | Dependencies |
|-------|-------|----------|--------------|
| B1 | [Scope] | HIGH | B0 |
| B2 | [Scope] | HIGH | B0, B1 (partial) |

### Phase 3: Extensions and Documentation (B3, B4, B5)

| Agent | Scope | Priority | Dependencies |
|-------|-------|----------|--------------|
| B3 | [Scope] | MEDIUM | B1, B2 |
| B4 | [Scope] | MEDIUM | B2 |
| B5 | [Scope] | MEDIUM | All |

### Phase 4: Validation (B6)

| Agent | Scope | Priority | Dependencies |
|-------|-------|----------|--------------|
| B6 | Test suite | HIGH | B1, B2, B4 |

---

## 4. Agent Summary

| Agent | Title | Primary Deliverable |
|-------|-------|---------------------|
| B0a | [Title] | [Deliverable] |
| B0b | [Title] | [Deliverable] |
| B1 | [Title] | [Deliverable] |
| B2 | [Title] | [Deliverable] |
| B3 | [Title] | [Deliverable] |
| B4 | [Title] | [Deliverable] |
| B5 | [Title] | [Deliverable] |
| B6 | [Title] | [Deliverable] |

---

## 5. File Inventory

### New Files to Create

| File | Agent | Purpose |
|------|-------|---------|
| `path/to/file1.py` | B1 | [Purpose] |
| `path/to/file2.py` | B2 | [Purpose] |

### Files to Modify

| File | Agent | Modification |
|------|-------|--------------|
| `path/to/existing.py` | B1 | [Change description] |

---

## 6. Dependency Graph

```
B0a ─────┐
         ├──► B1 ──────┬──► B3 ──────┐
B0b ─────┘             │             │
                       ├──► B5 ◄─────┤
         ┌─────────────┘             │
         │                           │
         └──► B2 ──────┬──► B4 ──────┤
                       │             │
                       └─────────────┴──► B6
```

---

## 7. Success Criteria

### Minimum Viable Implementation
- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

### Full Implementation
- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

---

## 8. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| [Risk 1] | Low/Med/High | Low/Med/High | [Mitigation] |
| [Risk 2] | Low/Med/High | Low/Med/High | [Mitigation] |

---

## 9. Execution Plan

### Recommended Sequence
1. Phase 1 agents (B0a, B0b) - can run in parallel
2. Phase 2 agents (B1, B2) - run sequentially
3. Phase 3 agents (B3, B4, B5) - can run in parallel after Phase 2
4. Phase 4 agent (B6) - final validation

### Review Gates
- [ ] After Phase 1: Human review of infrastructure
- [ ] After Phase 2: Human review of core implementation
- [ ] After Phase 4: Final validation before merge

---

## 10. Individual Agent Plans

See `phase_b_plans/` directory:
- [AGENT_B0_PLAN.md](./phase_b_plans/AGENT_B0_PLAN.md)
- [AGENT_B1_PLAN.md](./phase_b_plans/AGENT_B1_PLAN.md)
- [AGENT_B2_PLAN.md](./phase_b_plans/AGENT_B2_PLAN.md)
- [AGENT_B3_PLAN.md](./phase_b_plans/AGENT_B3_PLAN.md)
- [AGENT_B4_PLAN.md](./phase_b_plans/AGENT_B4_PLAN.md)
- [AGENT_B5_PLAN.md](./phase_b_plans/AGENT_B5_PLAN.md)
- [AGENT_B6_PLAN.md](./phase_b_plans/AGENT_B6_PLAN.md)

---

*Last Updated: YYYY-MM-DD*
```

---

## Agent Plan Template

Create individual plans in `phase_b_plans/AGENT_BN_PLAN.md`:

```markdown
# Agent BN: [Title]

## Document Information

| Field | Value |
|-------|-------|
| Agent ID | BN |
| Scope | [Brief scope description] |
| Status | Planning Complete / In Progress / Complete |
| Created | YYYY-MM-DD |

---

## 1. Current State Assessment

### Existing Infrastructure
| Component | Location | Relevance |
|-----------|----------|-----------|
| [Component] | `path/to/file` | HIGH/MEDIUM/LOW |

### Gaps Identified
- Gap 1
- Gap 2

---

## 2. Implementation Plan

### Files to Create

| File | Purpose |
|------|---------|
| `path/file.py` | [Purpose] |

### Files to Modify

| File | Modification |
|------|--------------|
| `path/existing.py` | [Change] |

---

## 3. Code Structure

### Key Functions

```python
def function_name(
    param1: Type1,
    param2: Type2,
) -> ReturnType:
    """Brief description."""
```

---

## 4. Dependencies

| Agent | What BN Needs | Status |
|-------|---------------|--------|
| B1 | [Requirement] | Available/Pending |

| Agent | What They Need from BN |
|-------|------------------------|
| B3 | [Deliverable] |

---

## 5. Estimated Complexity

| Component | Complexity | Justification |
|-----------|------------|---------------|
| [Component] | LOW/MEDIUM/HIGH | [Reason] |
| **Overall** | LOW/MEDIUM/HIGH | |

---

## 6. Implementation Sequence

1. Step 1
2. Step 2
3. Step 3

---

*Decision Required: Approve to proceed with BN implementation.*
```

---

*Template Version: 1.0*
