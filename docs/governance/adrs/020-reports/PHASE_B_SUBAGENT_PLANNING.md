# Phase B Sub-Agent Planning Document

## Document Information

| Field | Value |
|-------|-------|
| Created | 2026-01-01 |
| Last Updated | 2026-01-01 |
| Based On | ChatGPT 5.2 Pro External Review Response |
| Status | **ALL PLANNING COMPLETE** - Ready for Implementation |
| Decision | **Option C (Hybrid Approach) Confirmed** |

---

## Executive Summary

ChatGPT 5.2 Pro has validated our Phase A findings and provided a clear path forward:

1. **Option C (Hybrid)** is the recommended approach
2. The time series extension is **conditionally defensible** as a "spliced, multi-instrument series"
3. A **targeted, bounded Phase B** should implement specific robustness specifications
4. The paper needs explicit "comparability disclosure" infrastructure

This document assigns planning tasks to sub-agents. Each agent will **create a detailed implementation plan** identifying all files to be modified, new files to be created, and the specific changes required. **No implementation occurs until all plans are reviewed and approved.**

---

## User Inputs (Resolved)

The following questions have been answered by the user:

| Question | Answer |
|----------|--------|
| **Journal Article Location** | `sdc_2024_replication/scripts/statistical_analysis/journal_article/output/article_draft_v5_p305_complete.pdf` |
| **Scope Boundaries** | All 8 agents should run (B0a, B0b, B1-B6) |
| **Execution Mode** | **Sequential** (not parallel) - to ensure unified approach |
| **Data Collection Appetite** | Yes, proceed with all-50-states analysis; use Census API or Google BigQuery |
| **Bayesian/Panel Priority** | Mandatory but exploratory - decide incorporation after seeing outputs |
| **Timeline** | No external deadlines; focus on steady progress |

### Additional User Concerns (Now Addressed)

1. **PDF Versioning**: User concerned about `article_draft_v5_p305_complete.pdf` naming. Need rigorous versioning with auditability while avoiding repository bloat.
   - **Resolution**: Agent B0a created versioning strategy with semantic versioning, metadata sidecars, and archive structure

2. **Iterative Workflow Structure**: User concerned about managing multiple waves/sprints of work without losing track.
   - **Resolution**: Agent B0b created sprint/wave workflow structure with clear terminology and tracking system

---

## Key Recommendations from External Review

### Confirmed Decisions

| Decision | Rationale |
|----------|-----------|
| Use Option C (Hybrid) | Primary inference on n=15/within-regime; extended series for robustness |
| Treat series as "spliced multi-instrument" | Acknowledge three measurement regimes explicitly |
| Model regime boundaries, don't ignore them | Vintage/regime indicators, piecewise trends |
| Treat 2020 as intervention/shock | Don't let COVID contaminate diagnostics |
| Keep corrections lightweight | Vintage dummies, not heavy splicing |

### Required Deliverables

1. **Data Comparability Subsection** - Main text, not footnote
2. **Regime-Aware Modeling Layer** - Vintage dummies, piecewise trends, intervention terms
3. **Robustness Output Suite** - Multiple specifications clearly reported
4. **Many-States Placebo Check** - Expanded cross-state comparison
5. **Panel/Bayesian Extensions** - Consider multi-state approaches for VAR/cointegration
6. **Clear Naming Convention** - "Spliced PEP-vintage series" terminology

---

## Sub-Agent Planning Assignments

Each sub-agent will produce a **planning document** that includes:
- Complete inventory of files to be touched
- Specific changes required for each file
- New files to be created (if any)
- Dependencies on other agents' work
- Potential risks or blockers
- Estimated complexity (low/medium/high)

---

### Agent B0a: PDF Versioning Strategy ✓ COMPLETE

**Scope**: Establish rigorous versioning and auditability for journal article PDFs

**Status**: **COMPLETE** - See [AGENT_B0a_VERSIONING_ANALYSIS.md](phase_b_plans/AGENT_B0a_VERSIONING_ANALYSIS.md)

**Key Recommendations**:
- Naming convention: `article-{MAJOR}.{MINOR}.{PATCH}-{STATUS}_{TIMESTAMP}.pdf`
- Sidecar `.metadata.json` files linking to git commits, data versions, tests
- Directory structure: `versions/{working,approved,production}/` + `archive/{era}/`
- Exclude PDFs from git; track via metadata
- Chain of custody: PDF → metadata → git → data → tests

**Files to Create**:
- `output/versions/` directory structure
- `output/VERSIONS.md`, `CURRENT_VERSION.txt`, `CHANGELOG.md`
- `MANUSCRIPT_VERSIONING.md` strategy document
- `docs/adr/020-manuscript-versioning.md` (ADR)
- `scripts/versioning/create_version.py` (optional tooling)

**Dependencies**: None (infrastructure agent)

---

### Agent B0b: Sprint/Wave Workflow Structure ✓ COMPLETE

**Scope**: Establish repository organization for iterative development sprints

**Status**: **COMPLETE** - See [AGENT_B0b_WORKFLOW_STRUCTURE.md](phase_b_plans/AGENT_B0b_WORKFLOW_STRUCTURE.md)

**Key Recommendations**:
- Terminology: Phase → Sprint → Wave → Agent hierarchy
- Naming: `SPRINT_{CONTEXT}_{PHASE}_{SEQUENCE}`
- Required docs per sprint: PLAN, STATUS, ARTIFACTS, RETROSPECTIVE
- Multi-level tracking: DEVELOPMENT_TRACKER → PHASE_METADATA → SPRINT_STATUS
- Archive strategy: Era-based folders with retention policy

**Files to Create**:
- `docs/adr/020-reports/SHARED/SPRINT_PLANNING_TEMPLATE.md`
- `docs/adr/020-reports/PHASE_A/PHASE_METADATA.md`
- `docs/adr/020-reports/PHASE_B/PHASE_METADATA.md`
- Updates to `DEVELOPMENT_TRACKER.md`

**Dependencies**: None (infrastructure agent)

---

### Agent B1: Statistical Modeling Infrastructure

**Scope**: Implement regime-aware modeling specifications

**Planning Tasks**:
1. Audit existing statistical analysis scripts to identify where regime-aware models should be added
2. Plan implementation of the following specifications on n=25 series:
   - Vintage dummies (2000s vs 2010s vs 2020s)
   - Piecewise trends (allow slopes to differ by regime)
   - Intervention/outlier term for 2020 (COVID shock)
   - Heteroskedasticity-robust inference (or regime-specific error variance)
3. Plan sensitivity analysis framework:
   - n=15 baseline
   - n=25 with vintage controls
   - Excluding 2020
   - Excluding 2000-2009
4. Plan variance-stabilizing transformations:
   - Per-capita rates (per 1,000 population)
   - Log transformation considerations (handle negative 2003 value)

**Key Questions to Answer**:
- Which existing scripts perform time series analysis?
- What statistical packages are currently used (statsmodels, etc.)?
- Where should regime-aware models be added vs. new scripts created?
- How should the 2003 negative value be handled in transformations?

**Files to Investigate**:
- `sdc_2024_replication/scripts/statistical_analysis/` (all analysis scripts)
- `cohort_projections/` (core projection code)
- `config/projection_config.yaml` (configuration)
- Existing test result files

**Deliverable**: `docs/adr/020-reports/phase_b_plans/AGENT_B1_PLAN.md`

---

### Agent B2: Multi-State Placebo Analysis

**Scope**: Scale up cross-state comparison to all 50 states

**Planning Tasks**:
1. Audit existing data collection scripts to determine if all-state data is already available
2. If not available, plan data collection expansion:
   - Census API queries for all states across all vintages
   - Data storage and processing pipeline changes
3. Plan statistical analysis:
   - Distribution of 2000s→2010s shifts across all states
   - Identify where ND falls in national distribution
   - Compare oil-adjacent states (ND, SD, MT, WY, TX, OK) as a group
   - Quantify "methodology effect" by looking at states with no economic shock
4. Plan visualization outputs:
   - Histogram of state-level shifts
   - ND position highlighted
   - Regional clustering analysis

**Key Questions to Answer**:
- Do we already have migration data for all states?
- What is the structure of existing Census API code?
- How much additional data collection is required?
- Should this be a separate analysis script or integrated?

**Files to Investigate**:
- `data/raw/` and `data/processed/` (existing data inventory)
- `scripts/fetch_data.py` (data collection)
- `cohort_projections/data/census_api.py` (API interface)
- `sdc_2024_replication/scripts/data_collection/` (replication data scripts)

**Deliverable**: `docs/adr/020-reports/phase_b_plans/AGENT_B2_PLAN.md`

---

### Agent B3: Journal Article Methodology Section

**Scope**: Plan "Data Comparability" disclosure infrastructure

**Planning Tasks**:
1. Locate current methods/data section in journal article draft
2. Plan new "Data Comparability" subsection with:
   - Explanation of PEP vintage system
   - Three-regime methodology differences (Residual→ROYA→ROYA+DHS)
   - Census Bureau warning and why we proceed anyway
   - Naming convention: "spliced PEP-vintage series"
3. Plan explicit analysis strategy disclosure:
   - Primary inference window (2010-2024)
   - Extended series role (robustness/diagnostics)
   - Statement: "No substantive interpretation based solely on level difference between 2000s and 2010s"
4. Plan robustness reporting structure:
   - How to present multiple specifications
   - Table/figure format recommendations
   - Appendix vs. main text decisions

**Key Questions to Answer**:
- Where is the current journal article draft?
- What is the existing structure of the methods section?
- What length/format constraints exist for target journal?
- How should supplementary materials be organized?

**Files to Investigate**:
- Journal article draft (location TBD)
- `sdc_2024_replication/` (replication package structure)
- Existing methodology documentation

**Deliverable**: `docs/adr/020-reports/phase_b_plans/AGENT_B3_PLAN.md`

---

### Agent B4: Panel/Bayesian VAR Extensions

**Scope**: Plan alternative statistical approaches for small-n limitations

**Planning Tasks**:
1. Research and plan panel data approaches:
   - Multi-state panel (rather than single ND series)
   - Fixed effects / random effects considerations
   - Cross-sectional dependence handling
2. Plan Bayesian/shrinkage VAR implementations:
   - Minnesota prior VAR specifications
   - Hierarchical models pooling information across states
   - Package evaluation (PyMC, Stan, arviz, etc.)
3. Assess feasibility within project scope:
   - Which approaches are most valuable for publication?
   - Which are tractable given current codebase?
   - What new dependencies would be required?

**Key Questions to Answer**:
- What VAR/cointegration analyses are currently attempted?
- What Bayesian infrastructure exists in the project?
- Is multi-state panel analysis already possible with current data?
- What would be the minimum viable panel approach?

**Files to Investigate**:
- Existing time series analysis code
- `pyproject.toml` (current dependencies)
- Statistical analysis scripts

**Deliverable**: `docs/adr/020-reports/phase_b_plans/AGENT_B4_PLAN.md`

---

### Agent B5: ADR-019 Decision Documentation

**Scope**: Update ADR-019 with external review findings and final decision

**Planning Tasks**:
1. Plan ADR-019 updates to reflect:
   - External review summary
   - Confirmed decision (Option C)
   - Conditions for defensibility
   - Implementation roadmap
2. Plan status updates to related ADRs
3. Plan integration with DEVELOPMENT_TRACKER.md
4. Plan archival of external review materials

**Key Questions to Answer**:
- What ADR format/structure is used in this project?
- Are there related ADRs that need cross-referencing?
- How should the ChatGPT response be formally incorporated?

**Files to Investigate**:
- `docs/adr/019-extended-time-series-methodology-analysis.md`
- `docs/adr/` (other ADRs for format reference)
- `DEVELOPMENT_TRACKER.md`

**Deliverable**: `docs/adr/020-reports/phase_b_plans/AGENT_B5_PLAN.md`

---

### Agent B6: Test Suite and Validation Infrastructure

**Scope**: Plan tests and validation for new statistical code

**Planning Tasks**:
1. Audit existing test coverage for statistical analysis code
2. Plan tests for new regime-aware models:
   - Unit tests for new functions
   - Integration tests for analysis pipelines
   - Regression tests for reproducibility
3. Plan validation checks:
   - Ensure results match Phase A findings
   - Cross-validate regime classifications
   - Verify robustness outputs are consistent
4. Plan CI/CD updates if needed

**Key Questions to Answer**:
- What is current test coverage for statistical code?
- What testing frameworks are used?
- Where should new tests be located?

**Files to Investigate**:
- `tests/` (existing test structure)
- `pytest.ini` or `pyproject.toml` (test configuration)
- `.github/workflows/` (CI/CD if present)

**Deliverable**: `docs/adr/020-reports/phase_b_plans/AGENT_B6_PLAN.md`

---

## Planning Workflow

### Phase B-Planning Execution Order

**Note**: User requested **sequential execution** (not parallel) to ensure unified approach across all agents.

```
┌─────────────────────────────────────────────────────────────────┐
│              INFRASTRUCTURE PLANNING ✓ COMPLETE                 │
├─────────────────────────────────────────────────────────────────┤
│  B0a: PDF Versioning Strategy         ✓ COMPLETE                │
│  B0b: Sprint/Wave Workflow Structure  ✓ COMPLETE                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              CONTENT PLANNING ✓ ALL COMPLETE                    │
├─────────────────────────────────────────────────────────────────┤
│  Step 1: B1 Statistical Modeling      ✓ COMPLETE                │
│      ↓                                                          │
│  Step 2: B2 Multi-State Analysis      ✓ COMPLETE                │
│      ↓                                                          │
│  Step 3: B3 Journal Article           ✓ COMPLETE                │
│      ↓                                                          │
│  Step 4: B4 Panel/Bayesian            ✓ COMPLETE                │
│      ↓                                                          │
│  Step 5: B5 ADR Documentation         ✓ COMPLETE                │
│      ↓                                                          │
│  Step 6: B6 Test Infrastructure       ✓ COMPLETE                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     SYNTHESIS & REVIEW ✓ COMPLETE               │
├─────────────────────────────────────────────────────────────────┤
│  1. Consolidate all agent plans                 ✓               │
│  2. Identify file conflicts/dependencies        ✓               │
│  3. Create FILE_CHANGE_MANIFEST.md              ✓               │
│  4. Sequence implementation order               ✓               │
│  5. User review and approval                    ⧐ PENDING       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     IMPLEMENTATION PHASE                        │
│                (Only after plan approval)                       │
└─────────────────────────────────────────────────────────────────┘
```

### Execution Status

| Agent | Scope | Status | Deliverable |
|-------|-------|--------|-------------|
| B0a | PDF Versioning | ✓ Complete | [AGENT_B0a_VERSIONING_ANALYSIS.md](phase_b_plans/AGENT_B0a_VERSIONING_ANALYSIS.md) |
| B0b | Workflow Structure | ✓ Complete | [AGENT_B0b_WORKFLOW_STRUCTURE.md](phase_b_plans/AGENT_B0b_WORKFLOW_STRUCTURE.md) |
| B1 | Statistical Modeling | ✓ Complete | [AGENT_B1_PLAN.md](phase_b_plans/AGENT_B1_PLAN.md) |
| B2 | Multi-State Analysis | ✓ Complete | [AGENT_B2_PLAN.md](phase_b_plans/AGENT_B2_PLAN.md) |
| B3 | Journal Article | ✓ Complete | [AGENT_B3_PLAN.md](phase_b_plans/AGENT_B3_PLAN.md) |
| B4 | Panel/Bayesian | ✓ Complete | [AGENT_B4_PLAN.md](phase_b_plans/AGENT_B4_PLAN.md) |
| B5 | ADR Documentation | ✓ Complete | [AGENT_B5_PLAN.md](phase_b_plans/AGENT_B5_PLAN.md) |
| B6 | Test Infrastructure | ✓ Complete | [AGENT_B6_PLAN.md](phase_b_plans/AGENT_B6_PLAN.md) |

### Synthesis Documents

| Document | Status | Purpose |
|----------|--------|---------|
| [PLANNING_SYNTHESIS.md](phase_b_plans/PLANNING_SYNTHESIS.md) | ✓ Complete | Implementation roadmap |
| [FILE_CHANGE_MANIFEST.md](phase_b_plans/FILE_CHANGE_MANIFEST.md) | ✓ Complete | Complete file inventory |

### Data Source Information (for Agent B2)

User confirmed the following data source options:
- **Census API**: Existing infrastructure in `cohort_projections/data/census_api.py`
- **Google BigQuery**: Available as alternative (user can provide config if needed)

### Planning Agent Instructions

Each planning agent should:

1. **Inventory** all files in their scope
2. **Read** key files to understand current structure
3. **Document** specific planned changes (not implement)
4. **Identify** dependencies on other agents
5. **Flag** risks, blockers, or questions
6. **Estimate** complexity and effort
7. **Output** a structured planning document

### Output Directory Structure

```
docs/adr/020-reports/phase_b_plans/
├── AGENT_B0a_VERSIONING_ANALYSIS.md   ✓ Infrastructure: PDF versioning
├── AGENT_B0b_WORKFLOW_STRUCTURE.md    ✓ Infrastructure: Sprint workflow
├── AGENT_B1_PLAN.md                   ⧐ Statistical modeling plan
├── AGENT_B2_PLAN.md                   ⧐ Multi-state analysis plan
├── AGENT_B3_PLAN.md                   ⧐ Journal article plan
├── AGENT_B4_PLAN.md                   ⧐ Panel/Bayesian plan
├── AGENT_B5_PLAN.md                   ⧐ ADR documentation plan
├── AGENT_B6_PLAN.md                   ⧐ Test infrastructure plan
├── PLANNING_SYNTHESIS.md              ⧐ Consolidated implementation roadmap
└── FILE_CHANGE_MANIFEST.md            ⧐ Complete list of all file changes
```

---

## Success Criteria for Planning Phase

Before proceeding to implementation, we must have:

- [x] Infrastructure agents complete (B0a, B0b)
- [x] All 6 content agent plans complete (B1-B6)
- [x] Complete inventory of files to be modified (see FILE_CHANGE_MANIFEST.md)
- [x] No unresolved conflicts between agents (none found)
- [x] Clear implementation sequence (see PLANNING_SYNTHESIS.md)
- [x] Risk assessment complete
- [ ] User approval of overall plan
- [ ] Decision: Approve ADR rename (019 → 020)
- [ ] Decision: Approve PyMC dependency addition

---

## Questions for User Before Proceeding

**Status: ✓ ALL RESOLVED**

| Question | Status | Answer |
|----------|--------|--------|
| Journal Article Location | ✓ Resolved | `sdc_2024_replication/scripts/statistical_analysis/journal_article/output/article_draft_v5_p305_complete.pdf` |
| Scope Boundaries | ✓ Resolved | All 8 agents (B0a, B0b, B1-B6) |
| Execution Mode | ✓ Resolved | Sequential (not parallel) |
| Data Collection Appetite | ✓ Resolved | Yes, expand to all 50 states; Census API or BigQuery |
| Bayesian/Panel Priority | ✓ Resolved | Mandatory but exploratory |
| Timeline | ✓ Resolved | No deadlines; focus on steady progress |
| PDF Versioning Concern | ✓ Resolved | Agent B0a created strategy |
| Workflow Structure Concern | ✓ Resolved | Agent B0b created structure |

---

## Appendix: Key Findings Summary from External Review

### What ChatGPT 5.2 Pro Validated

| Finding | Validation |
|---------|------------|
| Census warning is real | Confirmed - treat as design constraint |
| Level shift at 2009-2010 | Reframed - decade difference, not discrete jump |
| 29:1 variance ratio | Largely scale effect + regime changes |
| Cross-state patterns | Supportive of real signal but not conclusive |
| Option C is best | Confirmed with detailed rationale |

### What ChatGPT 5.2 Pro Added

1. **Benchmarking/closure effects** - distinct from methodology change
2. **Scale effects explain variance** - relative variability more relevant
3. **2010 method change may alter composition** - not just totals
4. **2021-2024 has unique state-allocation uncertainty** - DHS adjustment caveat
5. **Intercensal series check** - potential additional validation

### Key Quote

> "If you follow this path, you're not 'breaking the rules' of the Census warning—you're acknowledging that the rules exist because measurement changes are real, then doing what good methods papers do: modeling those changes as part of the inferential problem rather than pretending they aren't there."

---

*End of Planning Document*
