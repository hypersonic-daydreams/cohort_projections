# Revision Plan v0.8.5

**Goal:** Address critique points from v0.8.0 review and produce an improved v0.8.5 draft.

## Strategy

1.  **Phase 1: Triage Fixes (Critical Logic & Arithmetic)**
    - Focus on the 5 items identified as "Triage Edits" to ensure internal consistency and accuracy.
    - *Files:* `02_data_methods.tex`, `03_results.tex`, `04_discussion.tex`.

2.  **Phase 2: Methodological Tightening**
    - Clarify justifications for Gravity/Panel models and Intro ranking.
    - *Files:* `01_introduction.tex`, `02_data_methods.tex`.

3.  **Phase 3: Final Output**
    - Verify with pipeline script.
    - Compile v0.8.5 PDF.

## Roadmap

### 1. Triage Fixes (High Priority)
- [ ] **Major 1 (Vintage):** Explicitly state data provenance in Section 2.
    - *Action:* Clarify if 2010-2019 are Vintage 2020 or 2024. (Likely Vintage 2024 back-series for consistency, but need to check scripts).
- [ ] **Major 3 (DiD Logic):** Pick one internal logic (likely "Upper Bound/Overstatement") and align text.
    - *Action:* Edit Discussion/Results to remove "attenuation" claim if arguing for upper bound.
- [ ] **Major 7 (Scenarios):** List all 5 scenarios in Methods. Justify 0.65x (Policy) and 8% (CBO).
    - *Action:* Update Section 2.12 (Forecast Scenarios).
- [ ] **Major 4 (ITS Module):** Reframe as "National System-Level Diagnostic".
    - *Action:* Move or rename ITS section context; ensure it's not confused with ND-specific policy effect.
- [ ] **Major 9 (Arithmetic):** Fix "52 states" to "51 accounting for DC".
    - *Action:* Grep and fix in `03_results.tex`.

### 2. Narrative & Methodology (Medium Priority)
- [ ] **Major 2 (Intro):** Add "Module Ranking" text to Introduction.
- [ ] **Major 5 (Panel):** Rename "determinants" to "decomposition" if covariates are absent.
- [ ] **Major 6 (Gravity):** Explicitly justify omitting distance (allocation model).
- [ ] **Major 8 (Uncertainty):** Add "Conservative Envelope" interpretation to Results/Conclusion.

### 3. Polish (Low Priority)
- [ ] **Minor 7:** Add "Practical Implications" paragraph to Discussion.

## Verification
- [ ] Run `verify_pipeline.py` (ensure no accidental data changes).
- [ ] Manual review of diffs.

## Output Log
- [ ] PDF Generated: `output/drafts/2026_immigration_policy_demographic_impact_v0.8.5.pdf`
