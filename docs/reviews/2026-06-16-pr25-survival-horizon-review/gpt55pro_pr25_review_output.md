GO-WITH-FIXES — I believe the 898,907 full-horizon result is directionally and internally sound, but the PR should not merge/publish until the recurrence guards are made hard-fail and the aggregation/FIPS assertions are completed.

## Direct answers

1. **Root cause:** Yes. A 2025–2045 operative survival table plus engine fallback explains the first affected output year being **2047**: the 2046→2047 step first needs `survival[2046]`. If the controlled rerun truly changed only appended survival years and the 2025–2045 overlap is byte-identical, the evidence is logically conclusive; an alternative explanation would need another input that changes only starting in 2047.
2. **Numbers:** **898,907 @2055** is present in the regenerated public CSV. **90+ = 8,172** is plausible and self-consistent but not directly auditable from the diff because the public CSV only has 85+. The direction is coherent: non-90 population rises about **+14.1k**, while 90+ falls **−1.8k**, netting the **+12.3k** total increase.
3. **Recurrence prevention:** The known clobbering test is fixed, but prevention is incomplete. The survival guard is warning-only, missing-table fallback is still only INFO, and state aggregation does not enforce exactly 53 current county files.
4. **Collateral/completeness:** The guard’s 2047 off-by-one reasoning is correct. Recomputing `_summary.csv` from GQ-inclusive `projection_results` is the right fix and does not obviously double-count. The 2025–2045 byte-identity and final 90+ detail still require live parquet QA.

## Fix-verification table — survival-horizon amendment

| Claim | Evidence in diff | Verdict |
|---|---|---|
| Test overwrote production survival table with 2025–2045 horizon | Old hard-coded production output replaced by injectable `output_dir` in `cohort_projections/data/process/mortality_improvement.py:400-441`; failing test used `projection_horizon: 20` and now passes `output_dir=tmp_path` in `tests/test_data/test_mortality_improvement.py:421-441` | **Verified for known test** |
| Horizon default fixed from stale 20 to 30 | `mortality_improvement.py:448-451` | **Fixed** |
| Open-age correction provenance recorded | Life-table source and `.attrs["open_ended_correction"]` metadata added at `mortality_improvement.py:488-550` | **Fixed in code; actual regenerated JSON not in diff** |
| Coverage guard warns on incomplete operative survival table | `scripts/pipeline/02_run_projections.py:518-543` computes required years and warns on missing years | **Partial** — warning-only; missing file still falls through |
| Integration tests no longer hard-code 2045 | `tests/test_integration/test_census_method_validation.py:223-235` derives expected years from config; `:545-558` derives dict keys from actual table | **Fixed** |
| State aggregation glob excludes `_components` | New glob `nd_county_*_projection_*_{scenario}.parquet` at `scripts/pipeline/02_run_projections.py:1141-1148` | **Likely fixed** |
| State aggregation asserts one file per FIPS | Duplicate-FIPS detection at `02_run_projections.py:1155-1174` | **Partial** — no exactly-53 or horizon assertion |
| County `_summary.csv` recomputed GQ-inclusive | `cohort_projections/geographic/multi_geography.py:185-203` recomputes summaries from post-GQ `projection_results` | **Looks correct** |
| Final 898,907 public total | State 2055 row in `PUB-2026 Draft Public Dataset.csv` shows `898907.0052746864`; QA doc repeats 898,907 at `docs/reviews/2026-06-16-corrected-run-qa-verification.md:19-29` | **Verified from CSV** |
| Final 90+ = 8,172 | QA/ADR docs assert it (`docs/reviews/2026-06-16-corrected-run-qa-verification.md:19-29`; ADR-068 amendment) | **Plausible but not directly verifiable from included public CSV** |

## Findings

### [major] Survival coverage guard still allows production fallback
`02_run_projections.py:518-543` only logs a warning when years are missing, and the no-operative-table path still logs “No mortality improvement data found” at the following `else`. For PUB-2026, either condition should be fatal unless explicitly opting into static survival.

**Action:** Raise `RuntimeError` for missing/incomplete operative survival in production/public baseline runs; add an explicit `allow_static_survival` override for tests/experiments.

### [major] Aggregation assertion does not enforce 53 current county files
`02_run_projections.py:1141-1175` catches duplicate FIPS, but a 52-county partial run or a complete set of stale 2025–2045 county files would still pass duplicate detection.

**Action:** Parse filenames with a regex and require: exactly 53 expected ND county FIPS, one file each, scenario matches, and filename/data years match `base_year..end_year`.

### [major] Programmatic mortality pipeline still defaults to production writes
`run_mortality_improvement_pipeline(..., output_dir=None)` still writes to `data/processed/mortality` by default (`mortality_improvement.py:434-441`). The known test is fixed, but the same bug can recur if any future test calls the function without `output_dir`.

**Action:** Require `output_dir` for the callable, or add `allow_production_write=True` only in the CLI wrapper. At minimum, fail if `PYTEST_CURRENT_TEST` is set and `output_dir is None`.

### [minor] Stale review evidence can confuse the amended horizon story
The committed round-2 evidence files named `evidence_survival_table_FULL.csv` / `evidence_survival_85plus.csv` stop at 2045 (e.g. `evidence_survival_85plus.csv:1-673`), because they predate the amendment. Also `docs/reviews/README.md:23` still describes the ref-intl finding as “Open… disposition pending.”

**Action:** Add explicit superseded banners or regenerate/rename the evidence files so “FULL” does not imply 2025–2055.

### [minor] Guard is intentionally stricter than engine needs
The engine needs survival keys for step years `base_year..end_year-1`; the guard requires `base_year..end_year` (`02_run_projections.py:529-533`). This is acceptable if the operative table is defined as 2025–2055 inclusive, but document that it is a publication convention, not engine necessity.

**Action:** Keep as-is with clearer wording, or hard-fail on `base..end-1` and separately QA-check inclusive table metadata.

### [minor] Sensitivity configs still contain production-ish output paths
The committed sensitivity configs isolate `pipeline.projection.output_dir`, but retain unrelated defaults such as `pipeline.data_processing.output_dir: data/processed`, `housing_unit_method.output_dir: data/projections`, and export paths. They are not used by the documented reproduction command, but they are a sibling class of “shared-state write” risk if reused broadly.

**Action:** Override all output roots in sensitivity/review configs to a sandbox folder, or add comments that only stage 02 is safe to run.

## Other review notes

- **Root-cause logic:** 2025–2046 identical and first divergence at 2047 is exactly what a missing `survival[2046]` produces. That is strong causal evidence.
- **Number direction:** The total increase despite lower 90+ is coherent: full NP2023 survival improves many working/older cohorts, while the open-interval 90+ correction reduces only the terminal group.
- **GQ-inclusive summaries:** The `multi_geography.py` summary recompute is the correct place to fix household-only summaries; components remain household-basis, which docs now disclose.
- **No other changed pytest path visibly writes production data.** The only explicit test clobber path in this diff is the mortality pipeline test, now redirected to `tmp_path`.

## What I could not verify from the diff alone

- The actual regenerated parquet `data/processed/mortality/nd_adjusted_survival_projections.parquet` spans 2025–2055 after merge and after full pytest.
- The claimed byte-identical 2025–2045 survival overlap.
- Direct detail-output computation of **90+ @2055 = 8,172**.
- Live state = Σ county residual and exactly 53 current county parquet files.
- Regenerated `mortality_improvement_metadata.json` contains the new open-age provenance.
- Binary/exported artifacts excluded from the diff match the CSV and QA memo.