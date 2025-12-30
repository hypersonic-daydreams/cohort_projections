# G04: Causal Inference Deep Dive - CRITICAL SCM DECISION SESSION

## Overview

This is the **CRITICAL** session for evaluating and deciding the fate of the Synthetic Control Method (SCM) analysis. The Travel Ban is a **national policy shock** affecting all states simultaneously - this creates a fundamental identification problem that must be resolved with a DEFINITIVE decision.

**Your primary task: Make the definitive keep/drop decision for SCM and specify alternative approaches.**

## Files Attached

1. **01_critique.md** - Full reviewer critique highlighting SCM identification problems (Section 6b specifically)
2. **02_results_causal.json** - Complete causal inference module results (DiD, SCM, Bartik)
3. **03_results_did.json** - Detailed DiD estimates for Travel Ban and COVID
4. **04_results_scm.json** - Synthetic control results with donor weights and time series
5. **05_data_event_study.csv** - Event study coefficients for dynamic treatment effects
6. **06_data_refugee.csv** - Refugee arrivals filtered to ND and regional peers (MN, SD, MT)
7. **07_data_panel.csv** - State-level panel data with international migration components
8. **08_script_causal.py** - Full implementation of Module 7 causal inference analysis

## The Core Problem

The reviewer critique (Section 6b) identifies a **fundamental design problem** with our SCM:

> "A basic synthetic control requires untreated donor units. A national policy shock like the Travel Ban affects all states. So 'synthetic ND' from other states is not a clean counterfactual unless you very explicitly define **treatment intensity** (e.g., ND is 'high exposure' because of its pre-2017 composition, donors are 'low exposure'). Even then, standard SCM needs adaptation."

**The question is NOT whether SCM produces numbers - it does. The question is whether those numbers have a valid causal interpretation.**

## Your Tasks

### Task 1: DiD Assessment (Priority: HIGH)

Review the Travel Ban DiD analysis and assess:

1. **Parallel trends validity**: Pre-trend test p=0.18, event study shows pre-trend coefficients
2. **Standard error specification**: HC3 vs clustering by nationality - which is appropriate?
3. **Log(arrivals+1) functional form**: Is PPML with FE preferred for count data with zeros?
4. **Sample construction**: 7 treated countries, 119 control countries, 16 pre + 2 post periods

Provide specific recommendations for strengthening the DiD identification.

### Task 2: SCM Decision (Priority: CRITICAL)

**This is THE critical decision of this session.** Make a definitive recommendation:

**Option A: DROP SCM entirely**
- Rationale: No valid untreated donors exist for a national policy
- Action: Remove SCM from paper, acknowledge limitation honestly

**Option B: REFRAME as exposure-intensity design**
- Rationale: ND's high pre-ban refugee share creates differential exposure
- Action: Reframe as "generalized SCM" or "interactive fixed effects"
- Required: Explicit treatment intensity definition, different inference

**Option C: KEEP as descriptive only**
- Rationale: SCM still shows counterfactual trajectory (even if causally questionable)
- Action: Relabel as "descriptive comparison" not "causal estimate"

For whichever option you recommend:
- Provide specific justification
- Address what the paper should say about the limitation
- Specify any required methodological changes

### Task 3: Bartik/Shift-Share Specification (Priority: MEDIUM)

The Bartik instrument shows F=22.5 (strong first stage). Assess:

1. **Base period selection**: Using 2010 - is this truly predetermined?
2. **Inference approach**: Are we using appropriate shift-share standard errors?
3. **Interpretation**: Coefficient of 4.36 needs units/interpretation clarification
4. **Modern concerns**: Goldsmith-Pinkham et al. (2020) vs Borusyak et al. (2022) approach

### Task 4: Triangulation Assessment

Given your assessments above, evaluate:

1. Do DiD and SCM tell a consistent story?
2. Does Bartik instrument provide independent corroboration?
3. What is the strongest causal claim we can make about the Travel Ban effect?
4. What caveats/limitations must be stated?

## Key Data Points for Your Analysis

### DiD Travel Ban Results
- ATT: -1.38 (SE=0.48), p=0.004
- Percentage effect: -74.9% reduction in arrivals from banned countries
- Pre-trend test: p=0.18 (parallel trends supported)
- R-squared: 0.83

### SCM Results
- Treated unit: North Dakota
- Treatment year: 2017
- Pre-treatment RMSPE: 0.02 (excellent fit)
- RMSPE ratio: 48.6 (suggests large post-treatment divergence)
- Key donor weights: Wyoming (42%), Vermont (25%), Rhode Island (20%)

### Bartik Results
- First-stage F: 22.5 (strong instrument)
- Coefficient: 4.36
- Sample: 528 obs, 48 states, 11 years

## SCM Decision Framework

**Arguments for DROPPING SCM:**
1. No state is truly "untreated" by a national executive order
2. Wyoming (42% weight) has minimal refugees - not comparable to ND's refugee-driven migration
3. Donor states may have different migration composition entirely
4. Post-treatment divergence could reflect differential recovery, not treatment effect

**Arguments for KEEPING (with modifications):**
1. Pre-treatment fit is excellent (RMSPE=0.02)
2. Differential exposure IS the identifying variation (ND's refugee share was 3x national average)
3. RMSPE ratio of 48.6 is unusually large
4. Method is increasingly accepted with exposure-intensity framing

**What the paper CANNOT claim:**
- "ND would have had X migration without the Travel Ban" (no valid counterfactual)
- Precise causal magnitude from SCM alone

**What the paper CAN claim:**
- ND's migration diverged substantially from synthetic comparator
- Combined with DiD, evidence suggests Travel Ban had large ND-specific impact

## OUTPUT FORMAT

**IMPORTANT: In your response text (before the file links), please include a brief executive summary (3-5 paragraphs) covering:**
1. Your definitive SCM decision and rationale
2. Key DiD and Bartik findings
3. The strongest causal claims the paper can make
4. Top 3 priority actions

**Then provide your detailed analysis as four downloadable files:**

### 1. scm_decision.md (THE CRITICAL OUTPUT)
Structure:
```markdown
# SCM Decision: [KEEP/DROP/REFRAME]

## Executive Summary
[2-3 sentence definitive recommendation]

## Identification Problem Analysis
[Detailed assessment of the "no untreated donors" critique]

## Recommended Action
[Specific steps to implement the decision]

## Paper Language
[Exact text for how to present/caveat this in the paper]

## Residual Concerns
[What limitations remain even after implementing recommendation]
```

### 2. did_specification.md
```markdown
# DiD Specification Recommendations

## Current Specification Assessment
[Evaluation of existing model]

## Recommended Changes
1. Standard errors: [recommendation]
2. Functional form: [recommendation]
3. Sample: [recommendation]

## Sensitivity Checks Required
[List of robustness checks to add]

## Event Study Improvements
[Specific recommendations for event study presentation]
```

### 3. bartik_specification.md
```markdown
# Bartik/Shift-Share Specification

## Identification Strategy Assessment
[Validity of shift-share in this context]

## Modern Inference Approach
[Which papers to cite, which SE approach]

## Interpretation Guidance
[How to interpret the 4.36 coefficient]

## Required Additions
[What needs to be added to paper]
```

### 4. recommendations.md
```markdown
# Integrated Causal Inference Recommendations

## Triangulation Assessment
[How the three methods relate]

## Strongest Defensible Claims
[What the paper can assert about Travel Ban effects]

## Required Caveats
[Limitations that MUST appear in the paper]

## Revision Priority List
1. [Highest priority change]
2. [Second priority]
...
```

## Critical Reminders

1. **Be definitive on SCM** - the paper cannot proceed with ambiguity on this
2. **The goal is publishability** - recommendations should make causal claims defensible to top-tier reviewers
3. **Acknowledge limitations honestly** - this strengthens rather than weakens the paper
4. **Provide actionable guidance** - specific text/code/methods, not vague suggestions

## Context

This paper argues that small-state international migration forecasting requires understanding policy sensitivity. The Travel Ban is our key policy event. If we cannot make defensible causal claims about its effect, we need to reframe what the paper is contributing.

The DiD design (nationality-level variation within ND) is likely our strongest identification. SCM was meant to provide triangulation but may be creating more problems than it solves.

Make the call.
