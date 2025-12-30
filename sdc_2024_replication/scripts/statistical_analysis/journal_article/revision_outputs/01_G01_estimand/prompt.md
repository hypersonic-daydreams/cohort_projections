# G01: Estimand & Identification Strategy Review

## Context

You are reviewing an academic paper titled **"Forecasting International Migration to North Dakota: A Multi-Method Empirical Analysis"** that is being revised for submission to a top-tier demographic methods journal.

The paper uses a nine-module statistical analysis pipeline to examine international migration patterns to North Dakota and develop projection scenarios. A previous review identified several major issues that need to be addressed, with the top priority being: **clarifying and committing to a single forecasting target (estimand)**.

## Your Role

You are acting as a methodological consultant helping the authors strengthen the paper's identification strategy and estimand clarity. Your task is to:

1. Define the forecast target clearly
2. Map data sources to that target
3. Assess identification assumptions
4. Recommend whether to include/exclude Structural Causal Model (SCM) framework
5. Provide specific actionable recommendations

## Attached Files

Please review the following files (numbered for reference):

1. **01_critique.md** - Previous reviewer critique identifying key issues (focus on Sections 1, 2, 6 regarding estimand and identification)

2. **02_report.md** - Full statistical analysis report with findings from all nine modules

3. **03_results_combined.json** - Combined JSON results from all analysis modules (summary statistics, time series, panel, gravity, causal inference, duration analysis, scenario modeling)

4. **04_data_nd_migration.csv** - Primary migration data series (2010-2024) with ND and US international migration counts

5. **05_latex_data_methods.tex** - Data and Methods section from the LaTeX article

6. **06_latex_results.tex** - Results section from the LaTeX article

---

## Task Instructions

### Part 1: Estimand Definition

The reviewer critique states:
> "You move among at least four 'international migration' objects: (1) PEP net international migration, (2) Refugee arrivals (RPC), (3) LPR admissions (DHS), (4) Foreign-born stock (ACS). A top-tier paper needs a clean statement like: 'The forecast target is PEP net international migration to North Dakota (calendar year); the other sources are used to...'"

**Your task:**
- Propose a clear, formal estimand statement
- Define the primary forecast target variable
- Explain how each secondary data source relates to the estimand
- Address calendar year vs. fiscal year harmonization
- Identify any measurement error or conceptual issues

### Part 2: Source Mapping Matrix

Create a source mapping table that shows:
- Each data source (PEP, DHS LPR, ACS Foreign-Born, RPC Refugee)
- What it measures (flow vs. stock, net vs. gross, population type)
- Time period and geographic coverage
- Role in the analysis (primary target vs. predictor vs. validation)
- Limitations and caveats

### Part 3: Identification Assessment

For each major causal claim in the paper, assess:

1. **Travel Ban DiD (Module 7)**
   - Are parallel trends credible?
   - Is the treatment definition appropriate (2017 vs 2018)?
   - What are threats to identification?

2. **Synthetic Control for ND (Module 7)**
   - Reviewer concern: "A national policy shock affects all states" - is SCM appropriate here?
   - What is the treatment being studied?
   - Recommend keep, modify, or drop

3. **Gravity Model Network Effects (Module 5)**
   - Reviewer concern: "Cross-section limits" and "diaspora stock endogeneity"
   - Is the network elasticity estimate causal or associational?
   - What language adjustments are needed?

4. **Bartik Shift-Share (Module 7)**
   - Are shares predetermined?
   - Is instrument strength adequate (F=22.46)?
   - What inference procedure is appropriate?

### Part 4: SCM Framework Decision

The original paper does not use a formal Structural Causal Model (SCM) / DAG framework. Consider:

- Would adding an SCM/DAG clarify identification assumptions?
- What would be the key nodes and edges?
- Would it help or hurt given the paper's forecasting (not causal) focus?
- Recommendation: Include or exclude SCM, with justification

### Part 5: Specific Recommendations

Provide a prioritized list of:
1. Must-fix issues before resubmission
2. Should-fix issues that strengthen the paper
3. Could-fix issues for completeness
4. Changes to avoid (that would harm the paper)

---

## OUTPUT FORMAT

Please produce three downloadable files:

### File 1: G01_recommendations.md
A markdown file containing:
- Executive summary (1 paragraph)
- Prioritized recommendations list (numbered)
- Specific text suggestions for the estimand statement
- Language fixes for causal claims

### File 2: G01_specifications.md
A markdown file containing:
- Formal estimand definition (mathematical notation acceptable)
- Identification assumptions for each causal analysis
- Recommended modifications to each module
- SCM decision with justification

### File 3: G01_source_mapping.md
A markdown file containing:
- Data source mapping table (Markdown table format)
- Calendar/fiscal year harmonization strategy
- Recommended changes to Data section of paper

---

## Additional Context

**Paper's Core Contribution:** The paper aims to demonstrate that rigorous statistical analysis is feasible even with small samples (n=15 years), and that uncertainty quantification is more valuable than point forecasts for small-state migration projection.

**Target Journal:** Top-tier demographic methods journal (e.g., Demography, Population Studies)

**Key Tension:** The paper wants to make both (a) forecasting claims and (b) causal claims about policy effects. The reviewer is pushing the authors to be clearer about which is primary and to ensure causal claims have proper identification.

Please begin your analysis by reading the attached files, then produce the three output files.
