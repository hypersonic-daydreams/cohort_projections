# Article Draft: Forecasting International Migration to North Dakota

## Article Information

**Title:** Forecasting International Migration to North Dakota: A Multi-Method Empirical Analysis

**Author:** [Author Name] (placeholder - to be completed)

**Affiliation:** [Institutional Affiliation] (placeholder - to be completed)

**Date:** December 2025

## Document Statistics

| Metric | Count |
|--------|-------|
| Word count (estimated) | ~12,500 |
| Page count (estimated) | ~35-40 pages |
| Figures | 9 main + 3 appendix = 12 total |
| Tables | 7 main + 8 appendix = 15 total |
| Equations | 18 |
| References | 72 |
| Appendix sections | 5 |

## Abstract Summary

This paper presents the first comprehensive multi-method analysis of international migration to North Dakota, deploying nine analytical modules spanning descriptive statistics, time series econometrics, panel regression, gravity models, machine learning, causal inference, and duration analysis.

**Key Findings:**
- Coefficient of Variation: 82.5% (high volatility)
- Travel Ban effect: ~75% reduction in refugee arrivals (p = 0.032)
- 2045 projection: 9,056 median [95% PI: 3,570-14,491]
- Structural breaks: 2020, 2021 (COVID-19 pandemic)
- Diaspora elasticity: 0.14 (modest diaspora associations)

## File Manifest

### Main Document
- `main.tex` - Master document (compiles all sections)
- `preamble.tex` - LaTeX packages and custom commands

### Section Files (`sections/`)
1. `01_introduction.tex` - Introduction and research questions
2. `02_data_methods.tex` - Data sources and methodology
3. `03_results.tex` - Empirical findings (9 analytical modules)
4. `04_discussion.tex` - Interpretation and policy implications
5. `05_conclusion.tex` - Summary and future research
6. `06_appendix.tex` - Full regression tables and robustness checks

### Support Files
- `references.bib` - Bibliography (72 entries)
- `figures/figure_captions.tex` - Figure captions with labels
- `compile.sh` - Compilation script

### Output
- `output/article_draft.pdf` - Compiled PDF (when available)
- `output/README.md` - This file

## Compilation Instructions

### Option 1: Local LaTeX Installation

**Requirements:**
- TeX Live, MiKTeX, or MacTeX
- pdflatex and bibtex

**Commands:**
```bash
cd /home/nigel/cohort_projections/sdc_2024_replication/scripts/statistical_analysis/journal_article

# Full compilation (recommended)
./compile.sh

# Or manual compilation
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Option 2: Overleaf

1. Create a new project at https://www.overleaf.com/
2. Upload all `.tex` and `.bib` files
3. Create `sections/` and `figures/` directories
4. Upload section files to `sections/`
5. Upload figure files to `figures/`
6. Compile using the Overleaf interface

### Required LaTeX Packages

The document requires these packages (available in standard TeX distributions):

- **Mathematics:** amsmath, amssymb, amsthm
- **Graphics:** graphicx, float, subcaption
- **Tables:** booktabs, array, multirow, threeparttable, tabularx
- **Citations:** natbib
- **Formatting:** hyperref, enumitem, xcolor, fancyhdr, titlesec
- **Layout:** geometry, setspace

## Figure Status

**Main Figures (all generated):**
- `analysis_pipeline.pdf` - Analytical pipeline diagram (available)
- `fig_01_timeseries.pdf` - Time series of international migration (available)
- `fig_02_concentration.pdf` - Location quotient bar chart (available)
- `fig_03_acf_pacf.pdf` - ACF/PACF diagnostic plots (available)
- `fig_04_structural_breaks.pdf` - Structural break analysis (available)
- `fig_05_gravity.pdf` - Gravity model coefficients (available)
- `fig_06_event_study.pdf` - DiD event study estimates (available)
- `fig_07_survival.pdf` - Kaplan-Meier survival curves (available)
- `fig_08_scenarios.pdf` - Projection scenario fan chart (available)

**Appendix Figures (not yet generated - optional):**
- `fig_app_state_distribution.pdf` - State migration distribution
- `fig_app_residuals.pdf` - ARIMA residual diagnostics
- `fig_app_schoenfeld.pdf` - Cox model Schoenfeld residuals

Note: The appendix figures can be commented out in `sections/06_appendix.tex` if not needed for initial compilation.

## Data Sources

| Source | Period | Description |
|--------|--------|-------------|
| Census PEP | 2010-2024 | Net international migration |
| DHS LPR | FY 2023 | LPR admissions by country |
| ACS | 2009-2023 | Foreign-born population stock |
| Refugee Processing Center | 2002-2020 | Refugee arrival counts |

## Keywords

international migration, forecasting, North Dakota, causal inference, demographic projection, refugee resettlement, Great Plains

## JEL Codes

- J11 - Demographic Trends, Macroeconomic Effects, and Forecasts
- J61 - Geographic Labor Mobility; Immigrant Workers
- C22 - Time-Series Models
- C53 - Forecasting and Prediction Methods
- R23 - Regional Migration; Regional Labor Markets; Population

## Contact

[To be completed with author contact information]

---

*Generated: December 29, 2025*
