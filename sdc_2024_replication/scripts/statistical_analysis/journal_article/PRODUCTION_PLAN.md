# Journal Article Production Plan

## Forecasting International Migration to North Dakota: A Multi-Method Empirical Analysis

**Target Style:** Tier-1 peer-reviewed journal (e.g., *Demography*, *Population and Development Review*, *Journal of Regional Science*, *International Migration Review*)

**Output Format:** LaTeX → PDF (publication-ready)

**Working Directory:** `sdc_2024_replication/scripts/statistical_analysis/journal_article/`

---

## Article Structure (IMRaD Format)

### 1. Front Matter
- Title
- Abstract (250-300 words)
- Keywords (5-7)
- JEL Codes

### 2. Introduction (~1,500 words)
- Research motivation: Why study ND international migration?
- Gap in literature: Limited work on small-state migration forecasting
- Research questions
- Contribution statement
- Article roadmap

### 3. Literature Review (~2,000 words)
- International migration theory (gravity, network, push-pull)
- State-level migration forecasting methods
- Refugee resettlement patterns in the US
- Great Plains demographic dynamics

### 4. Data and Methods (~2,500 words)
- Data sources (Census PEP, DHS LPR, ACS, Refugee arrivals)
- Study period and geographic scope
- Methodological framework
  - Descriptive statistics
  - Time series analysis (unit roots, ARIMA, VAR)
  - Panel data models
  - Gravity and network models
  - Machine learning approaches
  - Causal inference methods
  - Duration analysis
  - Scenario construction

### 5. Results (~3,500 words)
- Descriptive patterns and concentration analysis
- Time series properties and structural breaks
- Panel regression findings
- Gravity model estimates
- Causal effects of policy shocks
- Duration and lifecycle of migration waves
- Forecast scenarios and uncertainty

### 6. Discussion (~1,500 words)
- Interpretation of key findings
- Comparison with prior literature
- Policy implications
- Limitations and caveats

### 7. Conclusion (~500 words)
- Summary of contributions
- Future research directions

### 8. Back Matter
- References (50-80 citations)
- Appendices (technical details, robustness checks)
- Tables (integrated or appendix)
- Figures (6-10 publication-quality)

---

## Sub-Agent Workflow (Sequential)

### Phase 1: Foundation

#### Sub-Agent 1: Literature Review & References
**Input:** Methodology names from analysis modules
**Output:** `references.bib`, `literature_notes.md`
**Tasks:**
1. Web search for foundational references for each method:
   - HP filter (Hodrick-Prescott 1997; Ravn-Uhlig 2002)
   - Unit root tests (Dickey-Fuller 1979; Phillips-Perron 1988)
   - Structural breaks (Chow 1960; Bai-Perron 1998)
   - ARIMA (Box-Jenkins 1970)
   - VAR/Cointegration (Johansen 1988; Engle-Granger 1987)
   - Panel data (Hausman 1978; Baltagi 2013)
   - Gravity models (Anderson-Van Wincoop 2003; Santos Silva-Tenreyro 2006 PPML)
   - Network effects in migration (Massey 1990; Beine et al. 2011)
   - Machine learning (Breiman 2001; Tibshirani 1996)
   - DiD (Card-Krueger 1994; Angrist-Pischke 2009)
   - Synthetic Control (Abadie et al. 2010)
   - Bartik instruments (Bartik 1991; Goldsmith-Pinkham et al. 2020)
   - Cox PH/Duration (Cox 1972; Kalbfleisch-Prentice 2002)
   - Monte Carlo simulation (Metropolis-Ulam 1949; Robert-Casella 2004)
2. Search for ND-specific demographic literature
3. Search for refugee resettlement literature
4. Compile BibTeX file with proper formatting

---

### Phase 2: Section Drafting

#### Sub-Agent 2: Introduction
**Input:** `STATISTICAL_ANALYSIS_REPORT.md`, literature notes
**Output:** `sections/01_introduction.tex`
**Tasks:**
1. Craft compelling opening hook about ND's unique migration context
2. Establish research gap
3. State research questions clearly
4. Articulate contributions
5. Provide article roadmap

#### Sub-Agent 3: Data and Methods
**Input:** Module JSON files, methodology descriptions
**Output:** `sections/02_data_methods.tex`
**Tasks:**
1. Describe each data source with proper citations
2. Explain variable construction
3. Present methodological framework hierarchically
4. Include equations for key models (formatted in LaTeX)
5. Justify methodological choices

#### Sub-Agent 4: Results
**Input:** All JSON results, existing figures
**Output:** `sections/03_results.tex`
**Tasks:**
1. Present findings systematically by module group
2. Create properly formatted tables
3. Reference figures appropriately
4. Report statistical significance correctly
5. Maintain objective, results-focused tone

#### Sub-Agent 5: Discussion and Conclusion
**Input:** Results section, literature notes
**Output:** `sections/04_discussion.tex`, `sections/05_conclusion.tex`
**Tasks:**
1. Interpret findings in context of literature
2. Draw policy implications
3. Acknowledge limitations honestly
4. Suggest future research
5. Write compelling conclusion

---

### Phase 3: Visual Assets

#### Sub-Agent 6: Publication-Quality Figures
**Input:** Existing figures, JSON results
**Output:** `figures/` (new publication-ready versions)
**Tasks:**
1. Select 6-10 most impactful figures
2. Recreate with publication styling:
   - Consistent fonts (sans-serif, readable)
   - Proper axis labels with units
   - Figure captions below
   - Color-blind friendly palettes
   - 300 DPI minimum
   - PDF and PNG versions
3. Create any missing figures needed for narrative
4. Generate figure list with captions

---

### Phase 4: Integration & Polish

#### Sub-Agent 7: Integration and Refinement
**Input:** All section drafts, figures
**Output:** `main.tex` (integrated), refined section files
**Tasks:**
1. Combine sections into cohesive narrative
2. Ensure consistent terminology throughout
3. Check cross-references
4. Verify table/figure numbering
5. Smooth transitions between sections
6. Eliminate redundancy
7. Strengthen argument flow

#### Sub-Agent 8: Final Polish & PDF Generation
**Input:** Integrated manuscript
**Output:** `article_draft.pdf`, `abstract.txt`
**Tasks:**
1. Write/refine abstract (250-300 words)
2. Add keywords and JEL codes
3. Final proofreading pass
4. Check citation formatting
5. Generate PDF using pdflatex
6. Create supplementary materials file if needed
7. Final quality check

---

## File Structure

```
journal_article/
├── PRODUCTION_PLAN.md          # This file
├── main.tex                    # Master document
├── preamble.tex               # LaTeX preamble and packages
├── references.bib             # Bibliography
├── literature_notes.md        # Reference annotations
├── sections/
│   ├── 01_introduction.tex
│   ├── 02_data_methods.tex
│   ├── 03_results.tex
│   ├── 04_discussion.tex
│   ├── 05_conclusion.tex
│   └── 06_appendix.tex
├── figures/
│   ├── fig_01_*.pdf
│   ├── fig_02_*.pdf
│   └── ...
├── tables/
│   └── (generated inline or separate)
└── output/
    ├── article_draft.pdf
    └── supplementary.pdf
```

---

## Style Guidelines

### Writing Style
- Active voice preferred
- Third person (no "we believe", use "this analysis demonstrates")
- Precise quantitative language
- Avoid jargon without definition
- Clear topic sentences

### Statistical Reporting
- Report p-values to 3 decimal places (p < 0.001 for smaller)
- Report coefficients with standard errors in parentheses
- Include 95% confidence intervals where applicable
- Note sample sizes for all analyses
- Acknowledge limitations of small-n analysis

### Citation Style
- Author-date format (e.g., "Smith and Jones (2020) find...")
- Use "et al." for 3+ authors
- Cite seminal works for methodologies
- Include recent empirical applications

### Figure Standards
- Font size: minimum 8pt
- Line width: minimum 1pt
- Color: distinguish-able in grayscale
- Resolution: 300 DPI minimum
- Format: PDF preferred, PNG backup

---

## Quality Checklist

### Before Each Sub-Agent Completes:
- [ ] Consistent with prior sections
- [ ] All claims supported by data or citations
- [ ] Figures/tables properly referenced
- [ ] No placeholder text remaining
- [ ] LaTeX compiles without errors

### Final Checklist:
- [ ] Abstract accurately summarizes paper
- [ ] Introduction clearly states contributions
- [ ] Methods reproducible from description
- [ ] Results tables/figures publication-ready
- [ ] Discussion addresses limitations
- [ ] All references cited and formatted
- [ ] Page count appropriate (20-30 pages)
- [ ] PDF renders correctly

---

## Key Data Points to Highlight

From `STATISTICAL_ANALYSIS_REPORT.md`:

1. **Headline Statistics:**
   - ND receives 0.17% of US international migration (vs 0.23% of population)
   - CV = 82.5% for annual flows (high volatility)
   - 2020 COVID shock: 99% decline in ND international migration

2. **Structural Findings:**
   - Two significant structural breaks: 2020 (COVID), 2021 (recovery)
   - Series is I(1) requiring differencing
   - ARIMA(0,1,0) - random walk with high uncertainty

3. **Causal Estimates:**
   - Travel Ban reduced affected-country migration by ~75%
   - Difference-in-differences significant at p < 0.05

4. **Forecast Range:**
   - 2045 median projection: 8,672 persons
   - 95% CI: 3,183 to 14,104 persons
   - Wide range reflects structural uncertainty

---

## Notes for Sub-Agents

1. **Non-destructive:** Create all new files in `journal_article/` directory
2. **Self-contained:** Each section should be a complete `.tex` file that can be `\input{}` into main
3. **Figures:** Copy and enhance existing figures; do not modify originals
4. **References:** Use BibTeX keys consistently across sections
5. **Tables:** Use `booktabs` package styling for professional appearance
6. **Equations:** Number important equations for reference

---

*Last Updated: December 29, 2025*
