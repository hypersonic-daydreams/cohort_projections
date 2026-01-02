# Agent B3: Journal Article Methodology Section

## Document Information

| Field | Value |
|-------|-------|
| Agent ID | B3 |
| Scope | Data Comparability disclosure and methodology updates for journal article |
| Status | Planning Complete |
| Created | 2026-01-01 |

---

## 1. Current State Assessment

### 1.1 Article Structure

The journal article is located at `sdc_2024_replication/scripts/statistical_analysis/journal_article/` with:

| File | Section | Content |
|------|---------|---------|
| `main.tex` | Master | Title, abstract, includes all sections |
| `sections/01_introduction.tex` | Section 1 | Research questions, contributions |
| `sections/02_data_methods.tex` | Section 2 | Data sources, methodology (9 modules) |
| `sections/03_results.tex` | Section 3 | Empirical findings |
| `sections/04_discussion.tex` | Section 4 | Interpretation, policy implications |
| `sections/05_conclusion.tex` | Section 5 | Summary, future research |
| `sections/06_appendix.tex` | Appendix | Full tables, robustness checks |

### 1.2 Current Data/Methods Section Content

The current `02_data_methods.tex` (401 lines) contains:

**Section 2.1 Data Sources** (~88 lines):
- **Subsection 2.1.1**: Estimand & Measurement definition
- **Subsubsection**: Census Bureau PEP
- **Subsubsection**: DHS LPR Data
- **Subsubsection**: ACS Foreign-Born Population
- **Subsubsection**: Refugee Processing Center

**Current PEP Description** (lines 94-96):
> "The U.S. Census Bureau's Population Estimates Program (PEP) provides annual estimates... This analysis employs the vintage 2024 estimates covering 2010--2024... Key advantages include consistent methodology across years and comprehensive geographic coverage."

### 1.3 Current Gaps Identified

1. **No "Data Comparability" subsection exists** - Article assumes single-vintage analysis (2010-2024 only)
2. **No vintage definition** - "Vintage" is mentioned only in passing
3. **No discussion of methodology differences** across measurement regimes
4. **No Census Bureau warning** about combining vintages
5. **No explicit framing** as "spliced multi-instrument series"
6. **Limited 2020 mention** - Brief note about "measurement artifacts" but no elaboration

### 1.4 Article Extension Context

**Current state**: Article uses n=15 (2010-2024).
**Proposed**: Extend to n=25 (2000-2024), requiring Data Comparability infrastructure.

---

## 2. Data Comparability Subsection Plan

### 2.1 Proposed Location

Insert new subsection **2.1.5 "Data Comparability and Vintage Structure"** after current data source descriptions (after line ~115 in `02_data_methods.tex`).

### 2.2 Section Outline

```latex
\subsection{Data Comparability and Vintage Structure}
\label{subsec:data_comparability}

\subsubsection{PEP Vintage System}
- What "vintage" means in Census Bureau context
- Each vintage supersedes previous estimates
- Census Bureau guidance on cross-vintage comparison

\subsubsection{Measurement Regime Transitions}
- Regime 1: 2000-2009 (Vintage 2009) - Residual method
- Regime 2: 2010-2019 (Vintage 2020) - ROYA method
- Regime 3: 2020-2024 (Vintage 2024) - ROYA + DHS adjustment
- Table: Methodology comparison matrix

\subsubsection{Implications for Extended Time Series}
- Census Bureau warning and our response
- Why we proceed: statistical necessity + transparent sensitivity design
- Naming convention: "spliced PEP-vintage series"

\subsubsection{Analysis Strategy}
- Primary inference: 2010-2024 (within single regime)
- Extended series role: robustness and diagnostics only
- Statement: No substantive interpretation from cross-vintage level differences
```

### 2.3 Key Prose Elements

**Element 1: Census Bureau Warning (quoted verbatim)**

Include direct quote:
> "Data from separate vintages should not be combined. Due to periodic methodological updates... year-to-year comparisons in the estimates should only be done within the same vintage."

**Element 2: Methodology Differences Matrix**

| Aspect | Vintage 2009 | Vintage 2020 | Vintage 2024 |
|--------|--------------|--------------|--------------|
| **Foreign-born estimation** | Residual from decennial | ROYA via ACS | ROYA + DHS adjustment |
| **State allocation** | Census + INS admin data | 3-year pooled ACS | ACS + humanitarian adjustment |
| **Benchmark** | 2000 Census | 2010 Census + 2020 blend | 2020 Census + DHS data |

**Element 3: Analysis Strategy Statement**

> "Primary inferential claims are anchored in the 2010-2024 window (within-regime variation); the extended 2000-2024 series is employed exclusively for robustness diagnostics and exploratory time-series specifications that explicitly model regime transitions. No substantive interpretation is based solely on the level difference between the 2000s and 2010s decades."

### 2.4 Citations Needed

| Citation | Purpose | Source |
|----------|---------|--------|
| Census Bureau PEP methodology | Official methodology page | AGENT_1_REPORT |
| Census Working Paper POP-twps0051 | 1990-2000 NIM methodology | AGENT_1_REPORT |
| Census Working Paper POP-twps0097 | 2010 DA methodology | AGENT_1_REPORT |
| Census Random Samplings Blog 2024 | DHS adjustment announcement | AGENT_1_REPORT |
| Cornell PAD Vintage 2024 Analysis | Independent assessment | AGENT_1_REPORT |

---

## 3. Methodology Updates Plan

### 3.1 Updates to Existing Section 2.1

**Change 1: Modify PEP subsection (lines 92-97)**

Add caveat: "...consistent methodology across years *within the vintage 2024 estimates*"
Add forward reference: "See Section~\ref{subsec:data_comparability} for discussion of cross-vintage comparability"

**Change 2: Update abstract and introduction**

If scope changes to n=25:
- Abstract mentions "spliced PEP-vintage series spanning 2000-2024"
- Introduction acknowledges methodological challenge as contribution

### 3.2 Integration of Option C (Hybrid Approach)

**Implementation locations**:

1. **Data Comparability Subsection** (new Section 2.1.5)
   - Frame as Option C: "hybrid approach balancing inferential rigor with empirical scope"

2. **Time Series Methods Subsection** (Section 2.3)
   - Add regime-aware specifications
   - Reference B1 outputs

3. **Results Section** (Section 3)
   - Present n=15 results as primary
   - Present n=25 robustness results in parallel

4. **Discussion/Limitations** (Section 4.4)
   - Acknowledge methodological constraint

### 3.3 Language for "Spliced Multi-Instrument Series"

**Consistent terminology**:
- First use: "spliced PEP-vintage series" (defined in Section 2.1.5)
- Subsequent: "extended series," "vintage-bridged series"
- Avoid: "combined vintages," "pooled data" (these suggest homogeneity)

---

## 4. Robustness Reporting Plan

### 4.1 Table Structure for Sensitivity Results

**Main Text Table: "Sensitivity Analysis Summary"**

| Specification | n | Trend Coef | SE | 95% CI | Key Result |
|--------------|---|------------|-----|--------|------------|
| Primary (2010-2024) | 15 | X.XX | X.XX | [X.XX, X.XX] | Baseline |
| Extended + vintage dummies | 25 | X.XX | X.XX | [X.XX, X.XX] | Stable/Changed |
| Excluding 2020 | 24 | X.XX | X.XX | [X.XX, X.XX] | COVID impact |
| Excluding 2000-2009 | 15 | X.XX | X.XX | [X.XX, X.XX] | Post-methodology |

**Appendix Table: "Full Robustness Specifications"**
- Complete coefficient tables
- Diagnostic statistics
- Model selection criteria

### 4.2 Figure Requirements

**New Figure: "Coefficient Stability Across Specifications"**
- Forest plot showing key coefficients with 95% CIs
- Location: Section 3 or Appendix

**New Figure: "Extended Series with Vintage Boundaries"**
- Time series with vertical lines at vintage transitions
- Shaded regions for measurement regimes
- Location: Section 3 or Data Comparability

### 4.3 Robustness Discussion Location

**Main Text** (Section 3.X "Robustness and Sensitivity"):
- 2-3 paragraphs summarizing findings
- Key question: "Do substantive conclusions change?"

**Appendix** (new Section A.X):
- Complete regression tables
- Detailed diagnostics

---

## 5. Files Inventory

### 5.1 Files to Modify

| File | Change Type | Priority |
|------|-------------|----------|
| `sections/02_data_methods.tex` | Add Data Comparability subsection (~150 lines); modify PEP description | **HIGH** |
| `sections/03_results.tex` | Add robustness table and figure references (~50 lines) | **HIGH** |
| `sections/04_discussion.tex` | Add methodology limitations paragraph (~30 lines) | **MEDIUM** |
| `sections/06_appendix.tex` | Add extended robustness section (~100 lines) | **MEDIUM** |
| `main.tex` | Update abstract if scope changes to n=25 | **CONDITIONAL** |
| `references.bib` | Add Census Bureau citations (~30-50 lines) | **HIGH** |

### 5.2 New Files/Sections to Create

| Item | Location | Content |
|------|----------|---------|
| Data Comparability subsection | Within `02_data_methods.tex` | ~150 lines LaTeX |
| Methodology comparison table | Within `02_data_methods.tex` | ~30 lines |
| Sensitivity results table | Within `03_results.tex` | ~40 lines |
| Appendix robustness section | Within `06_appendix.tex` | ~100 lines |
| Vintage boundary figure | `figures/fig_vintage_boundaries.pdf` | Python-generated |
| Coefficient stability figure | `figures/fig_coefficient_stability.pdf` | Python-generated |

---

## 6. Draft Content Outlines

### 6.1 Data Comparability Subsection (~150 lines)

| Component | Lines | Content |
|-----------|-------|---------|
| Opening paragraph | 10 | PEP as forecast target; vintage challenges |
| PEP Vintage System | 15 | Define vintage; revision process |
| Census Bureau Warning | 10 | Block quote; source citation |
| Methodology Differences | 40 | Prose + comparison table |
| Implications | 20 | Why this matters; identification challenges |
| Analysis Strategy | 20 | Option C; inference vs. sensitivity |
| Naming Convention | 10 | "Spliced PEP-vintage series" definition |
| Closing | 10 | Transition to subsequent sections |

### 6.2 Key Terminology Glossary

| Term | Definition | Usage |
|------|------------|-------|
| **Vintage** | Annual PEP release superseding prior estimates | "Vintage 2024 estimates" |
| **Measurement regime** | Period with consistent methodology | "The 2010-2019 regime uses ROYA method" |
| **Spliced series** | Time series bridging methodology changes | "The spliced PEP-vintage series spans 2000-2024" |
| **Primary inference window** | 2010-2024 (within-vintage) | "Primary conclusions are based on the 2010-2024 window" |
| **Extended series** | 2000-2024 (cross-vintage) | "The extended series is used for robustness only" |

---

## 7. Dependencies

### 7.1 What B3 Needs from B1

| Deliverable | Purpose | Format |
|-------------|---------|--------|
| Sensitivity results table | Robustness reporting | CSV/JSON |
| Coefficient estimates | Populate tables | JSON |
| Regime-aware diagnostics | Appendix content | JSON |
| Model specification LaTeX | Equations | LaTeX snippets |

### 7.2 What B3 Needs from B2

| Deliverable | Purpose | Format |
|-------------|---------|--------|
| ND percentile in distribution | "Real vs. artifact" discussion | Scalar |
| Oil state comparison | Evidence for discussion | Summary stats |

### 7.3 What Other Agents Need from B3

| Agent | What They Need |
|-------|----------------|
| B5 (ADR) | Methodology framing for ADR-019 |
| B6 (Testing) | Numeric claims for validation |

---

## 8. Estimated Complexity

| Component | Complexity | Justification |
|-----------|------------|---------------|
| Data Comparability subsection | **MEDIUM** | New content requiring careful academic prose |
| Methodology table | **LOW** | Structured from AGENT_1_REPORT |
| PEP modifications | **LOW** | Minor text additions |
| Robustness table | **MEDIUM** | Depends on B1 outputs |
| New figures | **MEDIUM** | Python generation + LaTeX integration |
| Appendix expansion | **LOW** | Follows existing patterns |
| Citations | **LOW** | From AGENT_1_REPORT |
| **Overall** | **MEDIUM** | Complexity is in B1/B2 coordination |

---

## 9. Implementation Sequence

1. Wait for B1 statistical outputs
2. Draft Data Comparability subsection (standalone)
3. Generate figures (vintage boundaries, coefficient stability)
4. Add citations to `references.bib`
5. Modify `02_data_methods.tex`
6. Add robustness table to `03_results.tex`
7. Expand `06_appendix.tex`
8. Update abstract/intro (if n=25 confirmed)
9. Compile and verify

---

## 10. Risks and Blockers

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| B1 outputs delayed | MEDIUM | HIGH | Draft prose without final numbers |
| Scope remains n=15 | LOW | MEDIUM | Minimal changes needed |
| LaTeX compilation errors | LOW | LOW | Standard debugging |

---

## Summary

This plan provides:

1. **Current state assessment** of article structure and methodology content
2. **Detailed Data Comparability subsection plan** with outline and citations
3. **Methodology update strategy** integrating Option C throughout
4. **Robustness reporting framework** with table and figure specifications
5. **Complete file inventory** of modifications
6. **Draft content outlines** with line estimates
7. **Dependency mapping** to B1 and B2 agents

**Key Finding**: Current article claims "consistent methodology across years" which requires significant qualification if extended to n=25.

**Decision Required**: Approve this plan to proceed with B3 implementation.
