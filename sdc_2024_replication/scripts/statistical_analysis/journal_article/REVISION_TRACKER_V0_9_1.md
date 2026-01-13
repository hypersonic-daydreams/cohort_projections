# Article v0.9.1 Revision Tracker (from v0.9 production)

## Metadata

| Field | Value |
|---|---|
| Base article version | v0.9 (production) |
| Base PDF artifact | `sdc_2024_replication/scripts/statistical_analysis/journal_article/output/versions/production/article-0.9-production_20260112_211009/article-0.9-production_20260112_211009.pdf` |
| Base build timestamp | `2026-01-12T21:10:09.867152+00:00` |
| Base git commit | `124682f1fc26ddb96958dfee84aeb211b4b187fd` |
| Tracker created | `2026-01-13` |
| Goal | Apply all paper-quality improvements surfaced by AI audits + internal consistency checks, producing a *new* v0.9.1 artifact (do not edit v0.9 outputs). |
| Non-goals (unless explicitly approved) | Methodology changes that alter results; breaking output formats; changes to raw data in `data/raw/`. |

## Inputs Reviewed (External/AI Feedback)

Primary (v0.9-specific equation/consistency audit):
- `chatgpt_feedback_on_v0.9.md`
- `formula_audit_article-0.9-production_20260112_205726.md` (equation cross-check + practical “log(0)” notes)

Background (pre-v0.9 revision critiques; used as a sanity checklist for “have we regressed?”):
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/revision_outputs/06_resubmission_review/outputs/ChatGPT_5-2-Pro_revision_critique.md`
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/revision_outputs/06_resubmission_review/outputs/Gemini_3_Pro_DeepThink_revision_critique.md`

## Priority Definitions

- **P0**: Correctness / internal inconsistency / “referee trap” → must fix before building v0.9.1.
- **P1**: Clarity / notation / defensiveness → low risk, aim to include in v0.9.1.
- **P2**: Polish → include if time, avoid scope creep.

## High-ROI “Do This First” (from ChatGPT 5.2 Pro)

1. Fix **Figure 8 vs Table 14** inconsistency (numbers do not match).
2. Align **ADF deterministic terms** (Eq. (4) vs Appendix Table 19 “constant only”).
3. Specify **PPML log(0) handling** for `Stock_{od}` (and other logged covariates if relevant).
4. Add one sentence clarifying **DiD percent interpretation under `ln(y+1)`**.
5. Optional polish: Bai–Perron “break in mean” wording; HHI scaling; shift-share `g`; policy multiplier sign; define ADR once.

---

# Tracker Checklist (v0.9.1)

## P0 — Correctness & Consistency

- [x] **P0-A01** Reconcile **Figure 8 vs Table 14** duration statistics (medians, n’s, log-rank).
- [x] **P0-A02** Resolve **ADF specification mismatch**: Eq. (4) includes a trend, Table 19 reports “constant only”.
- [x] **P0-A03** Document and align **PPML logged covariate zero-handling** (e.g., `ln(Stock_{od}+1)`), and ensure Eq. (9) matches implementation.
- [x] **P0-A04** Reconcile **Figure 9 vs Table 15** Monte Carlo scenario uncertainty summary (draw count, 2045 medians, PI/envelope) across plot annotation, caption, and table/text.

## P1 — Clarity / Notation / Reviewer-Proofing

- [x] **P1-B01** Clarify **Bai–Perron** equation as *breaks in mean* (constant-only model) or switch equation to regression-break form if used.
- [x] **P1-B02** Add explicit **`ln(y+1)` interpretation** sentence for DiD/Event Study percent effects.
- [x] **P1-B03** Add explicit **HHI scaling** sentence (`s_i ∈ [0,1]`; HHI range 0–10,000 with ×10,000 scaling).
- [x] **P1-B04** Clarify **shift-share “g”** as level change vs growth rate; define `t_0` unambiguously.
- [x] **P1-B05** Clarify **policy multiplier sign convention** (is `Δ_hum` a reduction magnitude in [0,1] or a signed shock?).
- [x] **P1-B06** Define **ADR** acronym once (“Analysis Decision Record”) and decide whether ADR citations belong in main text or appendix/replication materials.

## P2 — Optional Polish

- [x] **P2-C01** Add a short note on **Elastic Net scaling convention** (constants absorbed into `λ`; align with scikit-learn conventions).
- [x] **P2-C02** Expand/clarify **K-means objective** shorthand (`min_{C, μ}`) or define centroids explicitly.
- [x] **P2-C03** (Optional) Use conditional mean notation **`E[M_{od} | X_{od}]`** in Eq. (9) for technical cleanliness.
- [x] **P2-C04** Verify **VAR notation** is unambiguously vector/matrix (bold vectors/matrices or distinct symbols) in the rendered PDF.

## Regression/Consistency “No-Regrets” Checks (verify v0.9.1 didn’t regress)

- [x] **V01** Run a quick scan for figure/table numeric mismatches (captions vs plot annotations vs tables) beyond the known Figure 8/Table 14 issue.
- [x] **V02** Verify the abstract’s Travel Ban language stays explicitly **non-causal** given mixed pre-trends and few treated clusters (keep “policy-associated divergence” framing).

---

# Task Details

## P0-A01 — Figure 8 vs Table 14 mismatch (Duration section)

**What’s wrong (confirmed in v0.9 PDF):**
- **Table 14 (PDF p34)**: Q4 median = **6.0 years**; `n=506`; log-rank **χ² = 633.0**, `p < 10^{-136}`.
- **Figure 8 (PDF p44)** plot legend: Q4 median = **4.0 years**; `n=234`; plot textbox shows log-rank = **278.7**, `p < 0.001***`.
- **Figure 8 caption** repeats the **Table 14** stats (median 6 years; χ² = 633.0), so caption and plot disagree too.

**Likely cause (code-level pointer):**
- Figure 8 in the PDF is produced from `sdc_2024_replication/scripts/statistical_analysis/journal_article/figures/fig_07_survival.pdf`.
- That figure is generated by `sdc_2024_replication/scripts/statistical_analysis/journal_article/create_publication_figures.py` (`create_figure_07_survival`), which reads:
  - `sdc_2024_replication/scripts/statistical_analysis/results/module_8_duration_analysis.json` (or a tagged variant),
  - then draws *approximate* curves from summary stats (medians + overall life table).
- Table 14 is typeset from LaTeX (`sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/03_results.tex`) and appears to be out of sync with whatever is in the duration results JSON used by the figure script.

**Decision point (must choose for v0.9.1):**
- **Canonical target**: should Figure 8 visualize the *same* sample/definition used for Table 14?
  - If yes (recommended), regenerate the duration outputs and the figure so the plot legend `n` and medians match the Table 14 quartiles (which sum to 2,057 in the narrative).
  - If no (subset is intentional), explicitly label the figure subset in the caption and update the duration narrative and/or Table 14 to match the subset.

**Planned fix (recommended path):**
1. Identify the canonical duration dataset/filters used for Table 14, and ensure the duration module output reflects it.
2. Regenerate `module_8_duration_analysis*.json` and regenerate `fig_07_survival.pdf` from the same output.
3. Update Table 14, Figure 8 caption, and the surrounding duration narrative so:
   - medians and `n` are consistent everywhere,
   - log-rank stats/p-values are consistent everywhere.

**Implemented (v0.9.1 work):**
- Updated `create_publication_figures.py` to default Figure 8/`fig_07_survival.pdf` to the `__P0` duration outputs when no tag is specified, and to display log-rank as $\chi^2$ with a power-of-ten p-value threshold matching Table 14.
- Regenerated `sdc_2024_replication/scripts/statistical_analysis/journal_article/figures/fig_07_survival.pdf` and `fig_07_survival.png`.

**Files expected to change when implementing v0.9.1:**
- `sdc_2024_replication/scripts/statistical_analysis/module_8_duration_analysis.py` (only if the mismatch is caused by parameterization/filters/bug)
- `sdc_2024_replication/scripts/statistical_analysis/results/module_8_duration_analysis.json` (regenerated artifact)
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/create_publication_figures.py` (if needed to plot the *actual* KM curves or to ensure it reads the intended tagged output)
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/figures/fig_07_survival.pdf` (regenerated)
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/03_results.tex` (Table 14 + narrative)
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/figures/figure_captions.tex` (Figure 8 caption)

**Acceptance criteria (how we know it’s fixed):**
- After rebuilding v0.9.1, `pdftotext` shows:
  - Table 14’s medians and quartile `n` match Figure 8 legend medians and `n`.
  - Figure 8 caption log-rank stat and p-value match what is plotted and what is reported in the Table 14 block and narrative sentence.
  - There is no remaining “Table says X, plot says Y” contradiction in the duration section.

---

## P0-A02 — ADF equation vs Table 19 (“constant only”) mismatch

**What’s wrong (confirmed in v0.9 PDF):**
- Eq. (4) in Methods includes a deterministic trend term: `α + β t`.
- Appendix Table 19 (PDF p61) labels ADF (and PP/KPSS) specifications as **“Constant only”**.

**What the implementation does (strong evidence):**
- `sdc_2024_replication/scripts/statistical_analysis/module_2_1_1_unit_root_tests.py` runs ADF with `regression="c"` (constant only), consistent with Table 19.

**Implemented (v0.9.1 work):**
- Updated Eq. (4) in `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/02_data_methods.tex` to the constant-only ADF regression (no deterministic trend) and added an explicit sentence that the reported ADF/PP results use constant-only deterministic terms, aligning the Methods text with Appendix Table 19.

**Files expected to change when implementing v0.9.1:**
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/02_data_methods.tex` (Eq. (4) + 1 sentence)
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/06_appendix.tex` (Table 19 notes only if we want to add a clarifying phrase; table already says “Constant only”)

**Acceptance criteria:**
- A technical reader no longer sees Eq. (4) contradicting Table 19’s specification label.

---

## P0-A03 — PPML gravity-style equation needs explicit log(0) handling

**What’s wrong:**
- Eq. (9) in Methods uses `ln Stock_{od}`. If any diaspora stocks are zero, `ln(0)` is undefined.
- PPML handles zero outcomes (flows) well, but not undefined regressors.

**Implementation hint (likely current behavior):**
- Several gravity-prep scripts (e.g., `sdc_2024_replication/scripts/statistical_analysis/distance_analysis.py`) create `log_diaspora = log(diaspora_stock + 1)` and similarly use `+1` for origin and destination totals.

**Implemented (v0.9.1 work):**
- Updated the PPML gravity equation (Eq. 9; `\label{eq:gravity}`) to use $\ln(x+1)$ for logged mass terms (diaspora stock, origin total, destination total) and added an explicit sentence documenting the log1p rule, aligning the paper with the implemented preprocessing.
- Added a matching $\ln(x+1)$ note to the PPML results table notes (main text and appendix) so replicators can infer the exact transformation without reading code.

**Files expected to change when implementing v0.9.1:**
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/02_data_methods.tex` (Eq. (9) + 1 sentence)
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/03_results.tex` (PPML table notes)
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/06_appendix.tex` (appendix PPML table notes)

**Acceptance criteria:**
- The equation is no longer ambiguous/incorrect when `Stock_{od}=0`, and a replicator can infer the exact transformation from the paper text.

---

## P0-A04 — Figure 9 vs Table 15 mismatch (Scenario Monte Carlo summary)

**What’s wrong (confirmed in v0.9 PDF and current `main.pdf`):**
- **Table 15** (`\label{tab:scenarios}`) reports Monte Carlo results from **25,000** draws: baseline median **8,720**, wave-adjusted median **9,119**, and 95\% envelope **[3,407, 14,806]**.
- **Figure 9** (`figures/fig_08_scenarios.pdf`) plot textbox reports **Monte Carlo: n = 1,000**, baseline median **8,639**, wave-adjusted median **8,983**.
- **Figure 9 caption** says uncertainty bands come from **1,000 Monte Carlo simulations**, contradicting the Table 15 note that the PI is from **25,000** draws.

**Likely cause (code-level pointer):**
- Figure 9 textbox values are pulled from `sdc_2024_replication/scripts/statistical_analysis/results/module_9_scenario_modeling.json` (`results.monte_carlo.*`), which currently has `"n_draws": 1000`.
- Table 15 and the abstract/narrative appear to be written for a 25,000-draw run (or were updated independently of the saved Module 9 JSON).

**Decision point (must choose for v0.9.1):**
- Set a canonical Monte Carlo draw count and make the plot textbox, Figure 9 caption, Table 15 note, and all referenced numbers (abstract/introduction/discussion) match.

**Planned fix (recommended path):**
1. Treat the 25,000-draw results as canonical (as currently described in Table 15 and the main text).
2. Regenerate Module 9 results (`module_9_scenario_modeling.json`) with `--n-draws 25000` (fixed seed), then regenerate `fig_08_scenarios.*` so the textbox and uncertainty bands match Table 15.
3. If performance or determinism is a concern, alternatively remove the Figure 9 textbox medians (and/or the draw-count mention in the caption) to avoid “Table says X, plot says Y” contradictions.

**Implemented (v0.9.1 work):**
- Re-ran Module 9 with `--n-draws 25000 --seed 42 --n-jobs 1 --chunk-size 1000 --duration-tag P0` so the saved Monte Carlo results match the Table 15 / narrative values (baseline median 8,720; wave-adjusted median 9,119; 95\% envelope [3,407, 14,806]).
- Regenerated `figures/fig_08_scenarios.pdf` and `figures/fig_08_scenarios.png` so the plot textbox reports `Monte Carlo: n = 25,000` and matching 2045 medians.
- Updated the Figure 9 caption to reference 25,000 Monte Carlo simulations and recompiled `main.pdf`; `pdftotext` confirms the caption and textbox are now consistent with Table 15 and the surrounding text.
- Ran a **10-seed stability sweep** (seeds 42--51; 25,000 draws each; `duration-tag P0`) with outputs isolated under `sdc_2024_replication/scripts/statistical_analysis/results/seed_sweeps/20260113T180922Z/`. The 2045 medians and 95\% endpoints vary only modestly across seeds (tens of migrants for medians; <$200$ for interval endpoints), supporting `seed=42` as a stable canonical seed for the reported numbers. Summary: `sdc_2024_replication/scripts/statistical_analysis/results/seed_sweeps/20260113T180922Z/module_9_seed_sweep_summary.csv`.
  - Decision for manuscript integration: **Option A** (robustness note only). Keep `seed=42` as the canonical reproducible artifact used for Figure 9/Table 15 values, and document the multi-seed stability check in Methods (no numeric changes).

**Files expected to change when implementing v0.9.1:**
- `sdc_2024_replication/scripts/statistical_analysis/module_9_scenario_modeling.py` (only if we change defaults; otherwise rerun with CLI flags)
- `sdc_2024_replication/scripts/statistical_analysis/results/module_9_scenario_modeling.json` (regenerated artifact)
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/figures/fig_08_scenarios.pdf` (regenerated)
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/figures/fig_08_scenarios.png` (regenerated)
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/figures/figure_captions.tex` (Figure 9 caption draw-count language)
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/03_results.tex` (Table 15 note + narrative if numbers change)
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/main.tex` (abstract if numbers change)
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/01_introduction.tex` and `sections/04_discussion.tex` (if they retain the 95\% PI numbers)

**Acceptance criteria (how we know it’s fixed):**
- After rebuilding the PDF, `pdftotext` shows Figure 9’s textbox, Figure 9 caption, and Table 15/text all reference the same draw count and report matching 2045 medians / 95\% intervals.

---

## P1-B01 — Bai–Perron equation is “breaks in the mean” (clarify)

**Issue:**
- Eq. (6) is written as minimizing deviations from segment means (piecewise-constant mean shifts). Many readers expect Bai–Perron in regression-with-breaks form.

**Planned fix:**
- Add a short clarifying phrase near Eq. (6): “We apply Bai–Perron to a constant-only model (breaks in the mean)…”
- If the actual implementation is regression-with-breaks, update the displayed equation accordingly.

**Implemented (v0.9.1 work):**
- Clarified that the Bai--Perron block is applied as a constant-only mean-shift change-point detector (breaks in the mean level of $y_t$), matching the implementation (ruptures/PELT on the level series). No change to results, only notation/reader-proofing.

**Files changed (v0.9.1 work):**
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/02_data_methods.tex` (structural break text around `\\label{eq:bai_perron}`)

---

## P1-B02 — DiD percent interpretation under `ln(y+1)`

**Issue:**
- Results and captions interpret `δ` via `1 − exp(δ)` as a percent reduction. With `ln(y+1)`, that is formally a percent change in `(y+1)`.

**Planned fix:**
- Add one sentence (Methods and/or Results/Figure caption) clarifying that `exp(δ)` is multiplicative on `(arrivals+1)` and approximates percent change in arrivals when arrivals are not tiny.

**Implemented (v0.9.1 work):**
- Added an explicit interpretation sentence after the DiD equation in Methods and mirrored the clarification in (i) the event-study figure caption and (ii) the DiD results table notes/narrative so the percent-effect language is reviewer-proof under $\ln(y+1)$.

**Files changed (v0.9.1 work):**
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/02_data_methods.tex` (DiD equation block; `\\label{eq:did}`)
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/03_results.tex` (DiD percent-effect note + narrative)
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/figures/figure_captions.tex` (event-study figure caption)

---

## P1-B03 — HHI scaling clarity

**Issue:**
- HHI formula uses `× 10,000`, but readers can stumble if it’s not explicit that `s_i` are fractions in `[0,1]`.

**Planned fix:**
- Add one explicit clause: `s_i` is a share in `[0,1]`, so HHI ranges `0–10,000` under the scaling.

**Implemented (v0.9.1 work):**
- Added an explicit clause in Methods that $s_i \in [0,1]$ and that the $\times 10{,}000$ scaling expresses HHI on the conventional $0$--$10{,}000$ scale; also aligned the Appendix variable-definition row to match.

**Files changed (v0.9.1 work):**
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/02_data_methods.tex` (HHI paragraph; `\\label{eq:hhi}`)
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/06_appendix.tex` (Appendix variable definitions)

---

## P1-B04 — Shift-share “g” definition (level change vs growth rate)

**Issue:**
- `g` can be read as a growth rate; text says “change”. A picky referee may ask whether `g` is a level change (`Δ`) or a percent change.

**Planned fix:**
- Either rename in the equation (e.g., `ΔM`) or add one explicit sentence: “`g` denotes a level change (not a growth rate) …” (or vice versa).
- Ensure `t_0` is defined clearly as the baseline year for weights.

**Implemented (v0.9.1 work):**
- Defined $t_0$ explicitly ($t_0 = 2010$) and clarified that $g$ is a leave-one-out national *level change* (persons) between $t_0$ and $t$ (not a growth rate); also clarified the share term to match the implementation (state share of national origin-specific arrivals at $t_0$) and aligned the Appendix variable-definition row.

**Files changed (v0.9.1 work):**
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/02_data_methods.tex` (shift-share subsection; `\\label{eq:bartik}`)
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/06_appendix.tex` (Appendix variable definitions)

---

## P1-B05 — Policy multiplier sign convention (`Δ_hum`)

**Issue:**
- The calibration uses `Δ_hum ≈ 0.75` as “75% reduction”, but the sign convention isn’t stated.

**Planned fix:**
- Add one sentence defining `Δ_hum ∈ [0,1]` as a *reduction magnitude* (fractional decrease), or explicitly allow signed shocks and update wording accordingly.

**Implemented (v0.9.1 work):**
- Defined $\Delta_{\mathrm{hum}} \in [0,1]$ as a reduction magnitude (e.g., $\Delta_{\mathrm{hum}} = 0.75$ denotes a 75\% decrease in the humanitarian channel), removing ambiguity about sign conventions in the calibration callout.

**Files changed (v0.9.1 work):**
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/02_data_methods.tex` (policy multiplier calibration callout)

---

## P1-B06 — Define ADR once (and decide placement)

**Issue:**
- “ADR-0xx” appears in the paper without defining ADR; many readers will not know it stands for “Analysis Decision Record”.

**Planned fix:**
- Define on first use: “Analysis Decision Record (ADR)”.
- Decide whether ADR references should remain in the main text or be moved to appendix/replication materials (depends on target journal norms).

**Implemented (v0.9.1 work):**
- Defined \emph{Analysis Decision Records} (ADRs) on first use and explicitly stated that ADRs live in the replication materials; retained in-text ADR references (e.g., ADR-025) as pointers to those replication notes (no relocation into the PDF appendix).

**Files changed (v0.9.1 work):**
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/02_data_methods.tex` (first definition + placement statement)

---

## P2-C01 — Elastic Net scaling convention

**Issue:**
- Elastic Net objective has multiple equivalent scalings across texts/software (e.g., `1/(2n)` in scikit-learn).

**Planned fix:**
- Add one short sentence noting constants can be absorbed into `λ`, and optionally mention alignment with scikit-learn’s convention.

**Likely file:**
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/02_data_methods.tex` (Elastic Net paragraph)

**Implemented (v0.9.1 work):**
- Added a one-sentence scaling note immediately after Eq.~\ref{eq:elastic_net} stating that constant loss scalings can be absorbed into $\lambda$, and that the implementation aligns with scikit-learn's $(1/2n)$ convention.

**Files changed (v0.9.1 work):**
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/02_data_methods.tex` (Elastic Net paragraph around Eq.~\ref{eq:elastic_net})

---

## P2-C02 — K-means objective shorthand

**Issue:**
- Equation writes minimization over partitions only; some readers expect `min_{C, μ}`.

**Planned fix:**
- Add one phrase “min over assignments and centroids” or define `μ_k` explicitly as the within-cluster mean.

**Likely file:**
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/02_data_methods.tex` (K-means paragraph)

**Implemented (v0.9.1 work):**
- Updated Eq.~\ref{eq:kmeans} to explicitly minimize over both the partition $\mathcal{C}$ and the centroid set $\{\boldsymbol{\mu}_k\}_{k=1}^{K}$ (not just over assignments), matching the standard K-means objective.

**Files changed (v0.9.1 work):**
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/02_data_methods.tex` (K-means equation block; Eq.~\ref{eq:kmeans})

---

## P2-C03 — Conditional mean notation in PPML gravity equation

**Issue:**
- Eq.~\ref{eq:gravity} writes $E[M_{od}]$ rather than the conditional mean $E[M_{od}\mid X_{od}]$, which is the object PPML targets.

**Planned fix:**
- Replace $E[M_{od}]$ with $E[M_{od}\mid X_{od}]$ and define $X_{od}$ succinctly in the accompanying text.

**Implemented (v0.9.1 work):**
- Updated Eq.~\ref{eq:gravity} to use $E[M_{od}\mid X_{od}]$ and added a brief definition of $X_{od}$ as the right-hand-side covariate set in the equation description.

**Files changed (v0.9.1 work):**
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/02_data_methods.tex` (PPML gravity equation block; Eq.~\ref{eq:gravity})

## P2-C04 — VAR vector/matrix notation (notation polish)

**Issue:**
- Even when the LaTeX uses `\mathbf{y}_t` and `\mathbf{A}_i`, PDF rendering can make boldface subtle; readers may misread the VAR as scalar if the same `y_t` symbol is used elsewhere.

**Planned fix (only if needed):**
- Ensure vectors/matrices are clearly distinguished (e.g., bold vectors/matrices and/or use a distinct symbol such as `\mathbf{Y}_t` for the vector).

**Likely file:**
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/02_data_methods.tex` (VAR equation block)

**Implemented (v0.9.1 work):**
- Tightened the VAR definition to reduce scalar-vs-vector ambiguity by (i) using an explicit transpose operator $(\cdot)^\top$, (ii) stating $\mathbf{y}_t$ is a $2\\times 1$ vector, and (iii) adding a short parenthetical that bold symbols denote vectors/matrices. Recompiled the PDF and confirmed the rendered Methods text reflects these clarifications.

**Files changed (v0.9.1 work):**
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/02_data_methods.tex` (VAR equation description around Eq.~\ref{eq:var})

---

## V01 — Scan for figure/table numeric mismatches (beyond Figure 8/Table 14)

**Method (reproducible):**
- Used `pdftotext` to extract embedded figure text/annotations from `sdc_2024_replication/scripts/statistical_analysis/journal_article/main.pdf` and the individual figure PDFs in `sdc_2024_replication/scripts/statistical_analysis/journal_article/figures/`.
- Cross-checked caption numbers (`figures/figure_captions.tex` and inline figure captions in `sections/*.tex`) against (a) plot textbox/legend annotations and (b) the corresponding table blocks in `sections/03_results.tex` and `sections/06_appendix.tex`.
- Focused on “reviewer trap” overlays: test statistics and summary numbers (`F`, `W`, `\chi^2`, `p`, medians, and `n`).

**Results:**
- No mismatches found for:
  - Structural break diagnostics (Figure 5) between the plot textbox and Table `\ref{tab:structural_breaks}`.
  - Travel Ban event-study diagnostics (Figure 7) between the plot textbox, Figure 7 caption, and Table `\ref{tab:causal_effects}` (ATT and 75.2\% effect).
  - Appendix diagnostic overlays (Shapiro--Wilk; Schoenfeld global test) between plot text and captions.
- One mismatch found (logged as **P0-A04**, now resolved): Figure 9 scenario plot textbox originally reported 1,000-draw medians (8,639 / 8,983) while Table 15 and the surrounding text reported 25,000-draw medians (8,720 / 9,119) and envelope [3,407, 14,806].

---

## V02 — Abstract Travel Ban language: explicitly non-causal

**Issue:**
- The abstract’s Travel Ban sentence used causal-adjacent phrasing (e.g., “policy effects”) and a definitive verb (“declined”) even though the paper’s own diagnostics emphasize mixed pre-trends and sensitivity with only seven treated nationalities.

**Planned fix:**
- Rewrite the abstract’s Travel Ban language to (i) avoid causal phrasing and (ii) explicitly frame the DiD result as descriptive \emph{policy-associated divergence} given mixed pre-trends and few treated clusters.

**Implemented (v0.9.1 work):**
- Updated the abstract to use “policy-associated treated--control divergence” framing, explicitly note mixed pre-trends and the seven treated nationalities, and avoid “policy effects” language.
- Recompiled the LaTeX (`compile.sh --quick --no-copy`) and verified via `pdftotext` that the rendered abstract preserves the non-causal “policy-associated divergence” interpretation.

**Files changed (v0.9.1 work):**
- `sdc_2024_replication/scripts/statistical_analysis/journal_article/main.tex` (abstract Travel Ban sentence and methods list)

**Acceptance criteria:**
- In `pdftotext` output from `main.pdf`, the abstract uses “policy-associated divergence” phrasing and explicitly disclaims interpretation as a precise causal effect magnitude.

---

# Implementation Workflow (for when we execute v0.9.1)

## Build prerequisites (do before running any Python)

- Activate the environment (preferred): `direnv allow` then `uv sync`
- Or: `source .venv/bin/activate` (from repo root) before running any Python scripts

## Suggested execution order

1. **Fix P0-A01** (duration mismatch) first, because it may require re-running Module 8 outputs and regenerating Figure 8.
2. Apply text/notation fixes (P0-A02, P0-A03, P1/P2 items) in LaTeX sources.
3. Regenerate any impacted figures via `create_publication_figures.py`.
4. Compile the LaTeX (`sdc_2024_replication/scripts/statistical_analysis/journal_article/compile.sh`).
5. Produce a **new** versioned artifact (v0.9.1) via `build_versioned_artifacts.py` and verify metadata.

## Verification checklist for v0.9.1 artifact

- [ ] `pdftotext` scan: duration section has consistent medians, `n`, and log-rank stats across Table 14, Figure 8, caption, and narrative.
- [ ] `pdftotext` scan: Table 19 deterministic terms match the displayed ADF equation and the described ADF implementation.
- [ ] `pdftotext` scan: no remaining `ln(Stock_{od})` ambiguity; paper states the exact log transform.
- [ ] Optional: quick scan for other “caption vs plot annotation” numerical mismatches (W/Q/F/χ² in diagnostics figures).

---

# Session Log (append as work proceeds)

| Date | Change IDs | Notes | Artifact path (if built) |
|---|---|---|---|
| 2026-01-13 | (planning) | Tracker created from AI audits; P0-A01 + A02 + A03 identified as main blockers | n/a |
| 2026-01-13 | P0-A01 | Figure/Table duration stats reconciled by defaulting survival figure to `module_8_duration_analysis__P0.json` and updating plot log-rank annotation; regenerated `fig_07_survival.*` | n/a |
| 2026-01-13 | P1-B01--B06 | Implemented notation/clarity fixes: Bai--Perron mean-shift phrasing; explicit $\ln(y+1)$ percent interpretation in Methods/Results/captions; HHI scaling; shift-share $g$ and $t_0$; policy multiplier sign convention; define ADR and state ADRs live in replication materials; updated Appendix variable definitions; compiled LaTeX successfully. | `sdc_2024_replication/scripts/statistical_analysis/journal_article/main.pdf` |
| 2026-01-13 | P2-C01--P2-C04 | Implemented optional polish: Elastic Net scaling note; K-means objective minimized over assignments and centroids; PPML conditional mean notation $E[M_{od}\\mid X_{od}]$; VAR vector/matrix notation tightened with explicit $2\\times 1$ vector statement and $(\\cdot)^\\top$; recompiled LaTeX for verification. | `sdc_2024_replication/scripts/statistical_analysis/journal_article/main.pdf` |
| 2026-01-13 | V01 | Ran `pdftotext` scan across `main.pdf` + embedded figure PDFs to check caption/plot/table numeric consistency; found one remaining mismatch in Figure 9 vs Table 15 Monte Carlo summary (1,000-draw textbox medians vs 25,000-draw table/text) and logged it as P0-A04. | `sdc_2024_replication/scripts/statistical_analysis/journal_article/main.pdf` |
| 2026-01-13 | P0-A04 | Re-ran Module 9 Monte Carlo with 25,000 draws (`--duration-tag P0`) and regenerated Figure 9 scenario plot so the textbox/caption match Table 15 and the narrative; recompiled and verified via `pdftotext`. | `sdc_2024_replication/scripts/statistical_analysis/journal_article/main.pdf` |
| 2026-01-13 | P0-A04 | Multi-seed Monte Carlo stability check: ran seeds 42--51 (25,000 draws each; `duration-tag P0`) with outputs isolated under `sdc_2024_replication/scripts/statistical_analysis/results/seed_sweeps/20260113T180922Z/`; confirmed reported 2045 medians/interval endpoints are stable across seeds. | `sdc_2024_replication/scripts/statistical_analysis/results/seed_sweeps/20260113T180922Z/module_9_seed_sweep_summary.csv` |
| 2026-01-13 | V02 | Tightened abstract language around the Travel Ban DiD to be explicitly non-causal (“policy-associated divergence”) given mixed pre-trends and only seven treated nationalities; recompiled and confirmed via `pdftotext`. | `sdc_2024_replication/scripts/statistical_analysis/journal_article/main.pdf` |
