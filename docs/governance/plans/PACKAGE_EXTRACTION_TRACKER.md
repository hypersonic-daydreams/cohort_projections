# Package Extraction Tracker

**Plan**: [PACKAGE_EXTRACTION_PLAN.md](./PACKAGE_EXTRACTION_PLAN.md)
**Started**: 2026-01-02
**Status**: Not Started

---

## Progress Summary

| Phase | Package | Status | Progress |
|-------|---------|--------|----------|
| 1a | project_utils | ⬜ Not Started | 0/4 agents |
| 1b | evidence_review | ⬜ Not Started | 0/6 agents |
| 2 | codebase_catalog | ⬜ Not Started | 0/5 agents |
| 3 | Workspace Docs | ⬜ Not Started | 0/2 agents |

---

## Phase 1a: project_utils

| Agent | Task | Status | Notes |
|-------|------|--------|-------|
| 1a-1 | Scaffold repository | ⬜ Pending | |
| 1a-2 | Migrate code | ⬜ Pending | |
| 1a-3 | Tests + documentation | ⬜ Pending | |
| 1a-INT | Integration verification | ⬜ Pending | |

### Validation Gate
- [ ] `uv sync` works
- [ ] `uv run pytest` passes
- [ ] Package importable
- [ ] cohort_projections tests pass
- [ ] Pre-commit hooks pass

---

## Phase 1b: evidence_review

| Agent | Task | Status | Notes |
|-------|------|--------|-------|
| 1b-1 | Scaffold repository | ⬜ Pending | |
| 1b-2 | Migrate citations | ⬜ Pending | Can run parallel with 1b-3, 1b-4 |
| 1b-3 | Migrate claims | ⬜ Pending | Can run parallel with 1b-2, 1b-4 |
| 1b-4 | Migrate argumentation | ⬜ Pending | Can run parallel with 1b-2, 1b-3 |
| 1b-5 | Integration + docs | ⬜ Pending | |
| 1b-INT | Integration verification | ⬜ Pending | |

### Validation Gate
- [ ] `uv sync` works
- [ ] `uv run pytest` passes
- [ ] CLI commands work
- [ ] cohort_projections tests pass
- [ ] Pre-commit hooks pass

---

## Phase 2: codebase_catalog

| Agent | Task | Status | Notes |
|-------|------|--------|-------|
| 2-1 | Scaffold repository | ⬜ Pending | |
| 2-2 | Migrate scanner/inventory | ⬜ Pending | Can run parallel with 2-3 |
| 2-3 | Migrate hooks | ⬜ Pending | Can run parallel with 2-2 |
| 2-4 | Integration + docs | ⬜ Pending | |
| 2-INT | Integration verification | ⬜ Pending | |

### Validation Gate
- [ ] `uv sync` works
- [ ] `uv run pytest` passes
- [ ] CLI commands work
- [ ] Pre-commit hooks work with new package
- [ ] cohort_projections tests pass

---

## Phase 3: Workspace Documentation

| Agent | Task | Status | Notes |
|-------|------|--------|-------|
| 3-1 | Update REPOSITORY_INVENTORY.md | ⬜ Pending | |
| 3-2 | Final cleanup | ⬜ Pending | Remove old source files |

### Final Validation
- [ ] All packages installed in cohort_projections
- [ ] Full test suite passes
- [ ] Pre-commit hooks pass
- [ ] No references to old paths
- [ ] REPOSITORY_INVENTORY.md updated
- [ ] All GitHub repos created

---

## Blockers

| Issue | Phase | Status | Resolution |
|-------|-------|--------|------------|
| (none) | | | |

---

## Completed Tasks Log

| Timestamp | Agent | Task | Duration | Notes |
|-----------|-------|------|----------|-------|
| | | | | |

---

## Status Legend

- ⬜ Pending - Not started
- 🔄 In Progress - Currently being worked on
- ✅ Complete - Finished and verified
- ❌ Blocked - Cannot proceed
- ⚠️ Issues - Completed with issues to address

---

*Last updated: 2026-01-02*
