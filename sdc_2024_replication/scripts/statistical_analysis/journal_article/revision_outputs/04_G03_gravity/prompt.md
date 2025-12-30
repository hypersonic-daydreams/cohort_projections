# G03: Gravity Model and Network Analysis Review

## Context

You are a senior demography/migration economist reviewing the gravity model and network analysis sections of an academic paper on forecasting international migration to North Dakota.

A prior reviewer (ChatGPT 5.2 Pro) identified several concerns with the gravity model analysis (see Section 5 of the critique). The key issues are:

### Major Concerns

1. **Cross-Section Limitations**: The gravity model uses only FY2023 DHS LPR admissions data - a single cross-section. The paper sometimes uses language like "causal network effect" which cannot be defended from cross-sectional data alone.

2. **Implausibly Small Standard Errors**: The reported SEs on diaspora elasticity (~0.001-0.002 for PPML) are suspiciously tiny. With state-country data and many zeros, reviewers will immediately ask about:
   - Robust variance estimation
   - Clustering (by state? by country?)
   - Overdispersion adjustment
   - Measurement error in ACS diaspora stock estimates

3. **Missing Distance Variable**: The full gravity specification appears to omit distance while discussing it conceptually. Distance variation across destinations should be included or explicitly justified if excluded.

4. **Identification Limbo**: The analysis is caught between "prediction tool" and "causal mechanism" - journals punish this ambiguity.

### Data Available

The analysis uses:
- **Flow variable**: DHS LPR admissions by state and country of origin (FY2023)
- **Diaspora stock**: ACS foreign-born by origin country (2023, with MOE)
- **Destination mass**: Total foreign-born by state (ACS 2023)
- **Origin mass**: National total foreign-born by origin country (ACS 2023)

---

## Files Attached

1. **01_critique.md** - Full review from ChatGPT 5.2 Pro (see Section 5 for gravity-specific issues)
2. **02_results_gravity.json** - Gravity model estimation results (3 specifications)
3. **03_results_network.json** - Full gravity/network analysis output including ND-specific effects
4. **04_results_effects.json** - Network effects summary (cross-sectional vs panel elasticities)
5. **05_data_lpr.csv** - DHS LPR admissions data (state x country, FY2023)
6. **06_script_gravity.py** - Python script implementing the gravity model analysis

---

## Task Instructions

Please analyze the gravity model specification and provide recommendations for revision.

### Task 1: PPML Specification Review

Evaluate the current PPML gravity model specifications:

1. **Model 1** (Simple): `Flow ~ log(Diaspora_Stock)`
2. **Model 2** (Full): `Flow ~ log(Diaspora) + log(Origin_Mass) + log(Dest_Mass)`
3. **Model 3** (State FE): `Flow ~ log(Diaspora) + State_FE`

For each, assess:
- Is the specification theoretically appropriate for a gravity model?
- What is missing that should be included?
- How should distance be handled (or justified if excluded)?
- Is the PPML estimator appropriate (vs OLS, negative binomial, zero-inflated)?

Provide the correct gravity model specification in LaTeX notation.

### Task 2: Identification Analysis

Address the "causal vs predictive" ambiguity:

1. What can and cannot be claimed from a single cross-section?
2. How should the paper reframe "network effects" language?
3. What would be needed for causal identification?
   - Panel data over time?
   - Instrumental variables?
   - Natural experiments?
4. Provide specific suggested language revisions (before/after examples).

### Task 3: Standard Error Analysis

Investigate why SEs appear implausibly small:

1. Review the current inference approach in the script
2. Identify potential issues:
   - Is clustering implemented? At what level should it be?
   - Is the variance estimator robust?
   - Is overdispersion accounted for?
   - How should ACS measurement error be handled?
3. Provide recommendations for correct inference

### Task 4: Revised Specification

Provide a complete revised gravity model specification that:

1. Correctly specifies the PPML model
2. Includes appropriate controls (or justifies exclusions)
3. Uses defensible standard error computation
4. Clearly separates predictive vs causal claims
5. Acknowledges cross-sectional limitations

---

## OUTPUT FORMAT

**IMPORTANT: In your response text (before the file links), please include a brief executive summary (3-5 paragraphs) covering:**
1. The core issue with the current gravity specification and how to fix it
2. Why the standard errors appear implausibly small and the recommended solution
3. How to reframe "network effects" language (causal → predictive)
4. Top 3 priority changes for the revision

**Then produce THREE downloadable files:**

### File 1: `G03_gravity_specification.md`

Complete revised gravity model specification including:
- Theoretical gravity model equation (LaTeX)
- Empirical specification (LaTeX)
- Variable definitions
- Estimator choice justification
- Recommended controls

### File 2: `G03_se_analysis.md`

Standard error analysis including:
- Diagnosis of current SE issues
- Recommended clustering structure
- Overdispersion assessment
- Measurement error considerations
- Corrected inference approach

### File 3: `G03_recommendations.md`

Concrete revision recommendations including:
- Language changes (before/after examples)
- Specification changes
- New analyses needed (if any)
- Integration with paper narrative
- Acknowledgment of limitations

---

## Key Reminders

- This is a **single cross-section (FY2023 only)** - temporal variation is not available for this analysis
- The paper's goal is **forecasting** for state planning, not establishing causal mechanisms
- North Dakota is a **small state** context - network effects may operate differently than in gateway states
- The network elasticity estimate (~0.10 for full model) is notably weaker than typical migration literature values
