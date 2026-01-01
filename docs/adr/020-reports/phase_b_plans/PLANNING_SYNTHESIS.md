# Phase B Planning Synthesis

## Document Information

| Field | Value |
|-------|-------|
| Created | 2026-01-01 |
| Status | Planning Complete |
| Agents Completed | B0a, B0b, B1, B2, B3, B4, B5, B6 |

---

## Executive Summary

All 8 Phase B planning agents have completed their analyses. This document synthesizes key findings and provides an implementation roadmap.

### Key Decisions Required

1. **ADR Renaming**: Rename ADR-019 (time series) to ADR-020 to resolve naming conflict
2. **PyMC Dependency**: Approve adding PyMC/arviz for Bayesian VAR (B4)
3. **Implementation Order**: Approve sequential implementation plan

### Critical Discoveries

| Agent | Discovery |
|-------|-----------|
| **B2** | All 50-state data already exists locally - no Census API needed |
| **B4** | PyMC is NOT installed - blocker for full Bayesian VAR |
| **B5** | ADR-019 naming conflict - two ADRs share the same number |

---

## Agent Summary

| Agent | Scope | Complexity | Key Output |
|-------|-------|------------|------------|
| B0a | PDF Versioning | LOW | Semantic versioning + metadata sidecars |
| B0b | Workflow Structure | LOW | Phase → Sprint → Wave hierarchy |
| B1 | Regime-Aware Modeling | MEDIUM | 4 statistical specifications |
| B2 | Multi-State Placebo | MEDIUM | 50-state panel analysis |
| B3 | Journal Article | MEDIUM | Data Comparability subsection |
| B4 | Bayesian/Panel VAR | MEDIUM | Minnesota prior + panel extensions |
| B5 | ADR Documentation | MEDIUM | Decision record + rename |
| B6 | Test Infrastructure | MEDIUM | 16+ new test files |

---

## Implementation Roadmap

### Phase 1: Infrastructure (B0a, B0b, B5)

**Priority**: HIGH - Must complete before other work

| Task | Agent | Description |
|------|-------|-------------|
| Rename ADR-019 → ADR-020 | B5 | Resolve naming conflict |
| Create versioning structure | B0a | `output/versions/{working,approved,production}/` |
| Create workflow templates | B0b | PHASE_METADATA.md, sprint templates |
| Update DEVELOPMENT_TRACKER | B0b | Add sprint status section |

### Phase 2: Core Implementation (B1, B2)

**Priority**: HIGH - Statistical foundation

| Task | Agent | Description |
|------|-------|-------------|
| Create regime-aware module | B1 | `module_regime_aware/` with 6 files |
| Create multi-state module | B2 | `module_B2_multistate_placebo/` |
| Generate 50-state panel | B2 | `all_states_migration_panel.csv` |
| Run sensitivity suite | B1 | All 4 specifications |

### Phase 3: Extensions (B3, B4)

**Priority**: MEDIUM - Depends on B1/B2 outputs

| Task | Agent | Description |
|------|-------|-------------|
| Add Data Comparability section | B3 | ~150 lines in `02_data_methods.tex` |
| Add robustness tables | B3 | Sensitivity results in article |
| Install PyMC/arviz | B4 | New dependencies |
| Implement Bayesian VAR | B4 | Minnesota prior + estimation |

### Phase 4: Quality Assurance (B6)

**Priority**: MEDIUM - Parallel with Phase 3

| Task | Agent | Description |
|------|-------|-------------|
| Create test fixtures | B6 | `tests/conftest.py` |
| Add B1 tests | B6 | `test_regime_aware.py` |
| Add B2 tests | B6 | `test_multistate_placebo.py` |
| Add B3 validation | B6 | `test_numeric_claims.py` |

---

## Dependency Graph

```
                   ┌─────────────────────────────────────────┐
                   │  Phase 1: INFRASTRUCTURE                │
                   │  B0a, B0b, B5 (can run in parallel)     │
                   └─────────────────────────────────────────┘
                                      │
                                      ▼
                   ┌─────────────────────────────────────────┐
                   │  Phase 2: CORE IMPLEMENTATION           │
                   │  B1 → B2 (B2 uses B1 regime specs)      │
                   └─────────────────────────────────────────┘
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
    ┌───────────────────────────┐           ┌───────────────────────────┐
    │  Phase 3A: ARTICLE        │           │  Phase 3B: EXTENSIONS     │
    │  B3 (uses B1, B2 outputs) │           │  B4 (uses B2 panel)       │
    └───────────────────────────┘           └───────────────────────────┘
                 │                                         │
                 └────────────────────┬────────────────────┘
                                      ▼
                   ┌─────────────────────────────────────────┐
                   │  Phase 4: QUALITY ASSURANCE             │
                   │  B6 (tests all implementations)         │
                   └─────────────────────────────────────────┘
```

---

## File Count Summary

| Category | New Files | Modified Files |
|----------|-----------|----------------|
| Statistical Modules | 15 | 2 |
| Article/LaTeX | 6 | 5 |
| ADR/Documentation | 8 | 4 |
| Versioning Infrastructure | 8 | 2 |
| Tests | 16 | 1 |
| **Total** | **53** | **14** |

---

## Risk Assessment

### High Risks

| Risk | Agent | Mitigation |
|------|-------|------------|
| PyMC installation fails | B4 | Fallback to conjugate prior |
| Cross-references break after rename | B5 | Systematic find-replace |
| Small n limits statistical power | B1 | Minnesota prior shrinkage |

### Medium Risks

| Risk | Agent | Mitigation |
|------|-------|------------|
| B3 numeric claim mismatch | B3 | Automated validation tests |
| geopandas not available for maps | B2 | Make choropleth optional |
| LaTeX compilation issues | B3 | Standard debugging |

---

## Blockers

| Blocker | Status | Resolution |
|---------|--------|------------|
| ADR naming conflict | **NEEDS DECISION** | Rename to 020 |
| PyMC not installed | **NEEDS DECISION** | Add to requirements |
| B2 data availability | **RESOLVED** | Data exists locally |

---

## Success Criteria

Before implementation is complete:

- [ ] All 8 agent plans executed
- [ ] ADR-020 accepted with Option C decision
- [ ] Regime-aware models produce sensitivity table
- [ ] 50-state placebo analysis complete with ND percentile
- [ ] Journal article has Data Comparability subsection
- [ ] Bayesian/Panel recommendation documented
- [ ] All new functions have unit tests
- [ ] No broken cross-references

---

## Next Steps

1. **User Decision**: Approve ADR rename (019 → 020)
2. **User Decision**: Approve PyMC dependency addition
3. **Implementation**: Begin Phase 1 (infrastructure)
4. **Review Gate**: External review after Phase 2 complete

---

*End of Planning Synthesis*
