# Phase B File Change Manifest

## Document Information

| Field | Value |
|-------|-------|
| Created | 2026-01-01 |
| Purpose | Complete inventory of all files to create/modify in Phase B |
| Total New Files | 53 |
| Total Modified Files | 14 |

---

## Files to Create

### B0a: PDF Versioning Infrastructure

| File | Purpose |
|------|---------|
| `sdc_2024_replication/scripts/statistical_analysis/journal_article/output/versions/.gitkeep` | Directory structure |
| `sdc_2024_replication/scripts/statistical_analysis/journal_article/output/versions/working/.gitkeep` | Working versions |
| `sdc_2024_replication/scripts/statistical_analysis/journal_article/output/versions/approved/.gitkeep` | Approved versions |
| `sdc_2024_replication/scripts/statistical_analysis/journal_article/output/versions/production/.gitkeep` | Production versions |
| `sdc_2024_replication/scripts/statistical_analysis/journal_article/output/archive/.gitkeep` | Archive root |
| `sdc_2024_replication/scripts/statistical_analysis/journal_article/output/VERSIONS.md` | Version index |
| `sdc_2024_replication/scripts/statistical_analysis/journal_article/output/CURRENT_VERSION.txt` | Current pointer |
| `sdc_2024_replication/scripts/statistical_analysis/journal_article/output/CHANGELOG.md` | Revision history |

### B0b: Workflow Structure

| File | Purpose |
|------|---------|
| `docs/adr/020-reports/SHARED/SPRINT_PLANNING_TEMPLATE.md` | Sprint template |
| `docs/adr/020-reports/PHASE_A/PHASE_METADATA.md` | Phase A overview |
| `docs/adr/020-reports/PHASE_B/PHASE_METADATA.md` | Phase B overview |
| `docs/adr/020-reports/PHASE_B/PLANNING/AGENT_ASSIGNMENTS.md` | Agent tracking |

### B1: Regime-Aware Statistical Modeling

| File | Purpose |
|------|---------|
| `sdc_2024_replication/scripts/statistical_analysis/module_regime_aware/__init__.py` | Module init |
| `sdc_2024_replication/scripts/statistical_analysis/module_regime_aware/regime_definitions.py` | Regime definitions |
| `sdc_2024_replication/scripts/statistical_analysis/module_regime_aware/vintage_dummies.py` | Dummy creation |
| `sdc_2024_replication/scripts/statistical_analysis/module_regime_aware/piecewise_trends.py` | Piecewise trends |
| `sdc_2024_replication/scripts/statistical_analysis/module_regime_aware/covid_intervention.py` | COVID modeling |
| `sdc_2024_replication/scripts/statistical_analysis/module_regime_aware/robust_inference.py` | Robust SE/WLS |
| `sdc_2024_replication/scripts/statistical_analysis/module_regime_aware/sensitivity_suite.py` | Sensitivity runner |
| `sdc_2024_replication/scripts/statistical_analysis/module_B1_regime_aware_models.py` | Main script |
| `sdc_2024_replication/scripts/statistical_analysis/results/module_B1_regime_aware_models.json` | Results |
| `sdc_2024_replication/scripts/statistical_analysis/results/module_B1_sensitivity_summary.csv` | Sensitivity table |

### B2: Multi-State Placebo Analysis

| File | Purpose |
|------|---------|
| `sdc_2024_replication/scripts/statistical_analysis/module_B2_multistate_placebo/__init__.py` | Module init |
| `sdc_2024_replication/scripts/statistical_analysis/module_B2_multistate_placebo/data_loader.py` | Data loading |
| `sdc_2024_replication/scripts/statistical_analysis/module_B2_multistate_placebo/regime_shift_calculator.py` | Shift stats |
| `sdc_2024_replication/scripts/statistical_analysis/module_B2_multistate_placebo/oil_state_hypothesis.py` | Oil state test |
| `sdc_2024_replication/scripts/statistical_analysis/module_B2_multistate_placebo.py` | Main script |
| `sdc_2024_replication/scripts/statistical_analysis/results/module_B2_multistate_placebo.json` | Results |
| `sdc_2024_replication/scripts/statistical_analysis/results/module_B2_state_shift_rankings.csv` | Rankings |
| `sdc_2024_replication/scripts/statistical_analysis/results/module_B2_oil_states_analysis.csv` | Oil analysis |
| `data/processed/immigration/analysis/all_states_migration_panel.csv` | 50-state panel |
| `data/processed/immigration/analysis/state_regime_shift_summary.csv` | Shift summary |

### B3: Journal Article (Content within existing files)

| File | Purpose |
|------|---------|
| `sdc_2024_replication/scripts/statistical_analysis/journal_article/figures/fig_vintage_boundaries.pdf` | New figure |
| `sdc_2024_replication/scripts/statistical_analysis/journal_article/figures/fig_coefficient_stability.pdf` | New figure |

### B4: Bayesian/Panel Extensions

| File | Purpose |
|------|---------|
| `sdc_2024_replication/scripts/statistical_analysis/module_B4_bayesian_panel/__init__.py` | Module init |
| `sdc_2024_replication/scripts/statistical_analysis/module_B4_bayesian_panel/minnesota_prior.py` | Prior spec |
| `sdc_2024_replication/scripts/statistical_analysis/module_B4_bayesian_panel/bayesian_var.py` | BVAR estimation |
| `sdc_2024_replication/scripts/statistical_analysis/module_B4_bayesian_panel/panel_var.py` | Panel VAR |
| `sdc_2024_replication/scripts/statistical_analysis/module_B4_bayesian_panel/model_comparison.py` | Comparison |
| `sdc_2024_replication/scripts/statistical_analysis/module_B4_bayesian_panel/shrinkage_diagnostics.py` | Prior sensitivity |
| `sdc_2024_replication/scripts/statistical_analysis/module_B4_bayesian_panel_var.py` | Main script |
| `sdc_2024_replication/scripts/statistical_analysis/results/module_B4_bayesian_var.json` | BVAR results |
| `sdc_2024_replication/scripts/statistical_analysis/results/module_B4_panel_var.json` | Panel results |
| `sdc_2024_replication/scripts/statistical_analysis/results/module_B4_comparison.json` | Comparison |
| `sdc_2024_replication/scripts/statistical_analysis/results/module_B4_recommendation.json` | Recommendation |

### B6: Test Infrastructure

| File | Purpose |
|------|---------|
| `tests/conftest.py` | Shared fixtures |
| `tests/test_statistical/__init__.py` | Package init |
| `tests/test_statistical/conftest.py` | Statistical fixtures |
| `tests/test_statistical/test_regime_aware.py` | B1 tests |
| `tests/test_statistical/test_multistate_placebo.py` | B2 tests |
| `tests/test_statistical/test_sensitivity_suite.py` | Sensitivity tests |
| `tests/test_statistical/test_bayesian_var.py` | B4 tests |
| `tests/test_article/__init__.py` | Package init |
| `tests/test_article/test_latex_compilation.py` | LaTeX tests |
| `tests/test_article/test_numeric_claims.py` | Claim validation |
| `tests/test_docs/__init__.py` | Package init |
| `tests/test_docs/test_adr_links.py` | ADR link tests |
| `tests/fixtures/sample_nd_migration.csv` | Test data |
| `tests/fixtures/expected_regime_output.json` | Expected output |

---

## Files to Modify

### B0a: Versioning

| File | Change |
|------|--------|
| `.gitignore` | Add PDF exclusions |

### B0b: Workflow

| File | Change |
|------|--------|
| `DEVELOPMENT_TRACKER.md` | Add sprint status section |

### B1/B2: Statistical Analysis

| File | Change |
|------|--------|
| `sdc_2024_replication/scripts/statistical_analysis/SUBAGENT_COORDINATION.md` | Add B1, B2 documentation |

### B3: Journal Article

| File | Change |
|------|--------|
| `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/02_data_methods.tex` | Add Data Comparability (~150 lines) |
| `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/03_results.tex` | Add robustness table (~50 lines) |
| `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/04_discussion.tex` | Add limitations (~30 lines) |
| `sdc_2024_replication/scripts/statistical_analysis/journal_article/sections/06_appendix.tex` | Add robustness section (~100 lines) |
| `sdc_2024_replication/scripts/statistical_analysis/journal_article/references.bib` | Add Census citations (~30 lines) |

### B4: Bayesian/Panel

| File | Change |
|------|--------|
| `sdc_2024_replication/scripts/statistical_analysis/requirements_statistical.txt` | Add PyMC, arviz |

### B5: ADR Documentation (After Rename)

| File | Change |
|------|--------|
| `docs/adr/020-extended-time-series-methodology-analysis.md` | Add Decision section, update status |
| `docs/adr/020a-vintage-methodology-investigation-plan.md` | Update references |
| `docs/adr/README.md` | Add ADR-020 to index |

### B6: Testing

| File | Change |
|------|--------|
| `pyproject.toml` | Add new pytest markers |

---

## Files to Rename (B5)

| Current | New |
|---------|-----|
| `docs/adr/019-extended-time-series-methodology-analysis.md` | `docs/adr/020-extended-time-series-methodology-analysis.md` |
| `docs/adr/019a-vintage-methodology-investigation-plan.md` | `docs/adr/020a-vintage-methodology-investigation-plan.md` |
| `docs/adr/020-reports/` | `docs/adr/020-reports/` |

---

## Cross-Reference Updates Required

After renaming to 020, update all occurrences of `020-reports/` in:

- All files in `docs/adr/020-reports/` (after rename)
- `PHASE_B_SUBAGENT_PLANNING.md`
- `AGENT_*_REPORT.md` files
- `AGENT_B*_PLAN.md` files
- This manifest (internal references)

---

## Dependency Order for Creation

1. **First**: Directory structures (B0a, B0b)
2. **Second**: ADR rename (B5)
3. **Third**: Statistical modules (B1, B2)
4. **Fourth**: Article updates (B3)
5. **Fifth**: Bayesian extensions (B4)
6. **Sixth**: Tests (B6)

---

## Conflict Check

No file conflicts detected between agents. Each agent modifies distinct files.

---

*End of File Change Manifest*
