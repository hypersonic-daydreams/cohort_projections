---
title: "v0.8.7 Remaining Tasks Tracker (Post v0.8.6 Critique)"
date_created: 2026-01-12
target_revision: "v0.8.7"
source_critique: "sdc_2024_replication/revisions/v0.8.7/critique_chatgpt_5_2_pro_v0.8.6.md"
input_document: "sdc_2024_replication/scripts/statistical_analysis/journal_article/output/versions/production/article-0.8.6-production_20260107_202832/article-0.8.6-production_20260107_202832.pdf"
status: "active"
description: "Plan + progress tracker for resolving inconsistencies and tightening methods/framing in the article without expanding scope or redoing the full analysis."
---

# v0.8.7 Remaining Tasks Tracker (Post v0.8.6 Critique)

## Context and Goals
This tracker coordinates edits prompted by the ChatGPT 5.2 Pro critique of the v0.8.6 production article. The goal is to finalize the paper by eliminating internal inconsistencies and avoiding any obvious misuse or misstatement of methods.

Primary reference documents:
- Critique (verbatim prompt + response): `sdc_2024_replication/revisions/v0.8.7/critique_chatgpt_5_2_pro_v0.8.6.md`
- Reviewed PDF (v0.8.6 production artifact): `sdc_2024_replication/scripts/statistical_analysis/journal_article/output/versions/production/article-0.8.6-production_20260107_202832/article-0.8.6-production_20260107_202832.pdf`

**Scope stance (explicit):**
- Prefer *text, consistency, and framing* fixes over new analyses.
- Only revisit results/parameters if we cannot justify the current choices with transparent calibration and/or sensitivity checks that reuse existing artifacts.
- Avoid “methodology blow-up”: no major new modules or re-estimation unless required to prevent incorrect claims.

## Decisions (Locked Unless Reopened)
- **Travel Ban language**: rewrite to **policy-associated divergence only** (no “upper bound” / “lower bound” claims), because parallel trends fails in the full pre-period and causal magnitude is uncertain.
- **Model averaging**: remove model-averaging discussion; treat VAR (and any multivariate comparisons) as **diagnostic only**, not part of the scenario engine.
- **Immigration Policy multiplier (0.65×)**: default plan is **retain** the multiplier and add **calibration transparency** (share-based logic + sensitivity range). Revisit the value only if the calibration cannot support it.

## Definition of Done
- No contradictory causal/bounds statements for Travel Ban effects anywhere in the text.
- No “model averaging” claims remain; scenario mechanics match narrative.
- Scenario endpoints and referenced numbers match across tables, text, captions, and figures.
- Captions match the actual plotted content.
- The 0.65 multiplier is justified via a transparent calibration note and sensitivity range (even if approximate).
- A fresh PDF compiles successfully from the repo (see “Build/Verify”).

---

## Workstream 1 — Consistency and Version Drift (High Priority)
- [x] Fix scenario number drift in Discussion: `Immigration Policy` endpoint should match Table 15 (currently 3,893 vs 4,581).
  - Target files:
    - `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/04_discussion.tex`
    - `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/03_results.tex` (confirm canonical number)
- [x] Fix concentration Figure caption mismatch (“Egypt, India …” vs plotted origins).
  - Target files:
    - `sdc_2024_replication/scripts/statistical_analysis/journal_article/figures/figure_captions.tex`
    - `sdc_2024_replication/scripts/statistical_analysis/journal_article/create_publication_figures.py` (caption generator)
- [x] Run a quick “number drift” audit for scenario endpoints (2,517; 4,581; 7,048; 19,318) across sections/captions.
  - Output: short checklist of occurrences + confirmation they align.

## Workstream 2 — Travel Ban Framing Cleanup (High Priority)
Goal: remove bound claims and align identification language with what the diagnostics actually support.

- [x] Replace all “upper bound / lower bound / conservative lower bound” statements with consistent wording:
  - “policy-associated divergence” / “descriptive policy sensitivity evidence”
  - “causal magnitude uncertain due to violated parallel trends in full pre-period”
  - “restricted pre-period results are sensitivity/robustness checks”
- [x] Ensure Introduction, Results, and Discussion tell the *same* story.
  - Target files:
    - `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/01_introduction.tex`
    - `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/03_results.tex`
    - `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/04_discussion.tex`
    - `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/02_data_methods.tex`
- [x] Verify the figure caption for the event study does not imply causal certainty beyond diagnostics.
  - Target file: `sdc_2024_replication/scripts/statistical_analysis/journal_article/figures/figure_captions.tex`

## Workstream 3 — Remove Model Averaging Narrative (Medium Priority)
Goal: keep VAR as a diagnostic (if retained), but remove all ensemble/AIC-weight claims and associated equations.

- [x] Remove the model-averaging equation and any narrative describing scenario “ensemble” weights.
  - Target file: `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/02_data_methods.tex`
- [x] Remove results text that reports AIC weights (VAR weight = 1.000) and any implication it affects scenarios.
  - Target file: `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/03_results.tex`
- [x] Update the module-to-scenario integration table to remove “AIC model weights / model ensemble” as a scenario input.
  - Target file: `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/02_data_methods.tex`

## Workstream 4 — Panel/Hausman Reframing (Medium Priority)
Goal: avoid implying a meaningful FE vs RE selection when the reported panel is essentially a two-way decomposition.

- [x] Reframe Methods to describe the panel block as *benchmarking / two-way effects decomposition* unless we truly include time-varying covariates.
  - Target file: `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/02_data_methods.tex`
- [x] Update Results text to avoid “supporting random effects as preferred specification” based on a near-zero Hausman statistic.
  - Target file: `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/03_results.tex`
- [x] Ensure Appendix panel table notes match the framing (what the panel is doing and what it is *not* doing).
  - Target file: `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/06_appendix.tex`

## Workstream 5 — Immigration Policy Multiplier Calibration (Medium Priority)
Goal: keep multiplier stable unless indefensible; add transparent logic and a sensitivity range.

- [x] Add a short calibration box for the 0.65 multiplier:
  - Define an approximate mapping from refugee/humanitarian shock to net \PEP\ total: e.g., multiplier as a function of an assumed humanitarian share.
  - Report a plausible range for the share and show implied multipliers (sensitivity).
  - Keep language “scenario design choice” rather than “identified causal effect”.
  - Target file: `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/02_data_methods.tex`
- [x] (If needed) compute the share range using existing series (no new data pulls):
  - Compare RPC refugees (and/or DHS LPR totals as a rough bound) to \PEP\ net international migration to motivate a plausible interval.
  - Store any intermediate numbers/notes in this tracker under “Session Log”.

## Build/Verify
- [x] Compile the LaTeX document after edits:
  - `sdc_2024_replication/scripts/statistical_analysis/journal_article/compile.sh --clean`
- [x] Ensure the compiled PDF matches the updated numbers/captions in the sections above.
- [ ] (Optional) If we touch Python scripts (e.g., caption generator), run:
  - `uv run python -m ruff check sdc_2024_replication/scripts/statistical_analysis/journal_article/create_publication_figures.py`

---

## Session Log (Append-only)
Use this to record what changed, why, and where.

### 2026-01-12
- Decisions locked: Travel Ban = policy-associated divergence only; remove model-averaging narrative; add calibration transparency for 0.65 multiplier.
- Tracker created.
- Workstream 1 completed:
  - Scenario endpoints audited for drift (2,517; 4,581; 7,048; 19,318): all occurrences in `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/03_results.tex`, `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/04_discussion.tex`, and `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/05_conclusion.tex` align after updating the Discussion `Immigration Policy` endpoint to 4,581.
  - Concentration figure caption updated to match the plotted high-LQ origins (removed Egypt/India reference; emphasizes Liberia/Ivory Coast/Somalia/Tanzania).
- Workstream 2 completed (Travel Ban framing cleanup):
  - Removed contradictory “upper/lower bound” causal language and standardized on “policy-associated divergence” + “descriptive policy-sensitivity evidence” because the full pre-period event-study diagnostic rejects parallel trends (causal magnitude not identified in that specification).
  - Harmonized Travel Ban identification wording across abstract and main sections to prevent mixed causal interpretations (updated `sdc_2024_replication/scripts/statistical_analysis/journal_article/main.tex`, `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/01_introduction.tex`, `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/02_data_methods.tex`, `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/03_results.tex`, `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/04_discussion.tex`, and `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/05_conclusion.tex`).
  - Updated event-study caption to avoid implying causal certainty beyond the reported diagnostics (`sdc_2024_replication/scripts/statistical_analysis/journal_article/figures/figure_captions.tex`).
  - Consistency fixes while editing: standardized reported DiD conventional $p = 0.031$ across narrative sections to match the Results section; corrected restricted-window bootstrap/RI $p$ values in prose to match the displayed table.
- Build/Verify: `sdc_2024_replication/scripts/statistical_analysis/journal_article/compile.sh --clean` succeeds (PDF rebuild completes without errors).
- Workstream 3 completed (remove model-averaging narrative):
  - Removed the model-averaging equation and AIC-weight narrative from `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/02_data_methods.tex` and replaced it with a direct statement that scenario paths are not built via information-criterion weights (justification: the scenario construction logic in `sdc_2024_replication/scripts/statistical_analysis/module_9_scenario_modeling.py` defines scenario trajectories without AIC-weighted forecast combinations).
  - Updated the Module-to-Scenario integration table and surrounding text in `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/02_data_methods.tex` so the VAR is described as diagnostic-only and not a scenario input (justification: standalone long-horizon VAR forecasting would require future US migration values, which are unavailable at forecast time).
  - Removed Results text reporting AIC and $R^2$ ``weights'' in `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/03_results.tex` to eliminate any implication that scenario paths are constructed via model-weighting (justification: the reported scenarios are rule-based and already fully defined by their stated assumptions).
  - Removed remaining Appendix references to model-averaging weights in `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/06_appendix.tex` to prevent the Appendix from reintroducing a narrative removed from the main text (justification: `sdc_2024_replication/scripts/statistical_analysis/journal_article/main.tex` includes the Appendix in the compiled PDF).
- Workstream 4 completed (Panel/Hausman reframing):
  - Reframed the panel block as an intercept-only benchmarking two-way effects decomposition (state and year effects; no time-varying covariates) in `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/02_data_methods.tex` and removed language implying FE vs.\ RE is a substantive ``preferred specification'' choice (justification: the implemented panel model in `sdc_2024_replication/scripts/statistical_analysis/module_3_1_panel_data.py` is intercept-only; a near-zero Hausman statistic in this setting is not evidence for preferring RE).
  - Updated Results text in `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/03_results.tex` to treat the near-zero Hausman statistic as a diagnostic of equivalence in the intercept-only benchmarking setting, not as support for random effects as preferred.
  - Updated Appendix panel table caption/notes in `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/06_appendix.tex` to explicitly state what the panel block is doing (benchmarking decomposition) and what it is not doing (estimating causal covariate effects), aligning Appendix framing with the revised Methods/Results.
- Build/Verify: `sdc_2024_replication/scripts/statistical_analysis/journal_article/compile.sh --clean` succeeds after Workstream 4 edits (PDF rebuild completes without errors).
- Workstream 5 completed (Immigration Policy multiplier calibration transparency):
  - Added a boxed calibration note to the Scenario Construction section in `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/02_data_methods.tex` that (i) defines the share-based mapping $m = 1 - s\Delta$, (ii) reports a data-motivated share interval, (iii) shows implied multiplier sensitivity, and (iv) explicitly states the multiplier is a scenario design choice (not an identified causal effect on the \PEP\ total).
  - Computed the share interval using existing processed series (no new data pulls) and recorded the intermediate figures here for reproducibility:
    - Travel Ban DiD reference magnitude (Module 7 output): $\Delta_{\\mathrm{hum}} \\approx 0.7516$ (a $-75.2\\%$ policy-associated divergence in refugee arrivals from affected origins), from `sdc_2024_replication/scripts/statistical_analysis/results/module_7_did_estimates.json`.
    - ND 2010--2016 calibration window (sources: `data/processed/immigration/analysis/nd_migration_summary.csv`, `data/processed/immigration/analysis/refugee_arrivals_by_state_nationality.csv`, `data/processed/immigration/analysis/dhs_lpr_by_state_time_states_only.csv`):
      - \PEP\ net international migration (sum): 9,023
      - Refugee arrivals (RPC, sum): 6,946
      - LPR admissions (DHS, sum): 8,934
      - Refugee share among measured durable inflows (Refugees + LPR): sum-share 0.437; annual range 0.383--0.492
      - Implied multiplier range via $m = 1 - s\\Delta$: $m \\in [0.630, 0.712]$
      - Chosen multiplier check: $m = 0.65$ implies $s = (1-m)/\\Delta \\approx 0.466$, which falls within the 2010--2016 share range.
    - Justification for using Refugees/(Refugees+LPR) as the bounded proxy for $s$: PEP is a net-flow accounting quantity and RPC/DHS series are gross FY inflows; comparing refugees and LPR in the same time base provides a conservative, unit-consistent share proxy to translate a humanitarian-channel shock into an order-of-magnitude adjustment for the \PEP\ total.
  - Added a reproducible helper script to recompute these figures: `sdc_2024_replication/scripts/statistical_analysis/journal_article/revision_scripts/calibrate_immigration_policy_multiplier.py` (runs locally against existing processed outputs and reads the Travel Ban DiD effect from Module 7 results).
  - Build/Verify for this revision: `sdc_2024_replication/scripts/statistical_analysis/journal_article/compile.sh --clean` succeeds; `pdftotext` confirms the calibration note (and the 0.383--0.492 share range / 0.630--0.712 implied multipliers) appears in the compiled PDF.
  - Conclusion/decision on whether to change $0.65$: we \textbf{retain} $m = 0.65$ because it is \textit{inside} the implied sensitivity interval $[0.630, 0.712]$ under the documented share-based mapping and corresponds to an implied humanitarian share $s \\approx 0.466$ that lies within the 2010--2016 observed Refugees/(Refugees+LPR) range (0.383--0.492). If we wanted a more ``centered'' value under this proxy, the sum-share (0.437) would imply $m \\approx 1 - 0.437\\cdot 0.7516 \\approx 0.67$, but this is a modest shift and (given the net-vs-gross and FY-vs-estimate-year mismatches) does not dominate the ``keep multiplier stable unless indefensible'' goal.
  - Plain-language summary: the calibration does \textit{not} indicate $0.65$ is indefensible; it falls inside the implied sensitivity range (0.63--0.71). We therefore keep $0.65$ for v0.8.7; a more ``centered'' value under the same proxy would be roughly 0.67, but the difference is modest and the mapping is intentionally approximate (PEP net vs.\ gross inflows; FY vs.\ estimate-year timing).
