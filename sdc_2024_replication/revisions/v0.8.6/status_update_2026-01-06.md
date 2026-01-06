---
title: "v0.8.6 Status Update Memo"
date: 2026-01-06
context: "v0.8.6 critique implementation"
status: "update"
---

# v0.8.6 Status Update Memo (2026-01-06)

## Summary
- Travel Ban DiD/event-study updated to enforce nationality-unit hygiene (drops pseudo-nationalities like `Total`) and adds an explicitly-labeled FY2024 regime-dynamics extension (supplemental-only) with an appendix figure.
- Scenario pipeline made reproducible end-to-end: added the missing `Immigration Policy` scenario to Module 9 and aligned Figure 8 + results text with the generated parquet scenario paths.

## Key Decisions (ADRs)
- ADR-027: `docs/governance/adrs/027-travel-ban-extended-dynamics-supplement.md`
  - Keep FY2002--FY2019 as the primary causal Travel Ban estimand (avoid post-2020 confounding and post-rescission invalidity).
  - Add FY2024 extended event-study + regime-block summaries as supplemental descriptive outputs.
  - Exclude pseudo-nationalities from DiD/event-study units.

## Outputs Updated / Added
- Module 7:
  - `sdc_2024_replication/scripts/statistical_analysis/results/module_7_event_study_extended.parquet`
  - `sdc_2024_replication/scripts/statistical_analysis/results/module_7_travel_ban_regime_dynamics.json`
  - Appendix figure: `sdc_2024_replication/scripts/statistical_analysis/journal_article/figures/fig_app_event_study_extended.pdf`
- Module 9:
  - `sdc_2024_replication/scripts/statistical_analysis/results/module_9_scenario_projections.parquet` now includes `immigration_policy`.

## Paper Updates
- Results and robustness numbers refreshed to match regenerated outputs:
  - `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/03_results.tex`
  - Appendix reference added in `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/06_appendix.tex`

## Performance Note
- The original `module_7_robustness.py` bootstrap/permutation workflow is slow on this machine.
  - Use `sdc_2024_replication/scripts/statistical_analysis/module_7_robustness_fast.py` to regenerate `module_7_robustness.json` in minutes.

## Tests
- `uv run pytest tests/ -q` passes (new unit tests added for Module 7 helpers).
