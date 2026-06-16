… you are an expert demographer + research software engineer performing an INDEPENDENT, adversarial
code review of a real pull request. Your job is to find what is wrong or unverified — not to rubber-stamp.

# What you are reviewing

This is **PR #25** of a North Dakota cohort-component population-projection repo (commit `fcb0432`,
branch `docs/ref-intl-sum-vs-average-finding`, base `master`). The PR is the **ADR-068 model
correction** for the 2026 public release ("PUB-2026"). The **full PR text diff** (`git diff
master...HEAD`, binaries excluded) is appended verbatim after this brief — read it as the ground truth
and **chase anything that looks off, including issues this brief does not flag.** This brief only
orients you; it does not bound you.

The PR bundles three layers of ADR-068 work; the **focus of THIS review is the third (newest) layer**:

1. **(Already independently reviewed — GO)** The original ADR-068 fixes: the CBO international-migration
   numerator was a 3-year SUM (10,051) mislabeled "annual average" and applied per year; corrected to the
   true annual mean **3,350.33**. And the open-ended 90+ survival used the 85+ group rate (~0.885 M /
   0.914 F) as terminal retention; corrected to the open-interval survivorship ratio **T₉₁/T₉₀** (~0.778 M
   / 0.806 F). A prior independent GPT-5.5 Pro review confirmed both are implemented in code/config (not
   relabeled) and returned GO. You do **not** need to re-litigate these unless the diff shows them broken.
2. Round-2 remediation docs (methodology provenance, disclosures, labeling) and regenerated public
   artifacts.
3. **★ THE 2026-06-16 SURVIVAL-HORIZON AMENDMENT ★ — your primary target.** See below.

# Primary target: the survival-horizon amendment

**The claim being made (verify it; try to refute it):** the 2026-06-15 "corrected" production run was
itself biased for 2047–2055 because the operative survival table on disk spanned only **2025–2045**, and
the engine silently falls back to a different (uncorrected, race-specific static-base) survival table for
any projection year absent from the operative table. So for the steps 2047→2055, that run used neither the
NP2023 trajectory nor the ADR-068 90+ correction. Regenerating the survival table to the full 2025–2055
horizon and re-running the projection changed the headline:

| Metric | 2026-06-15 (survival truncated) | 2026-06-16 (full-horizon, claimed final) |
|---|---:|---:|
| 2055 total | 886,585 (+10.91%) | **898,907 (+12.45%)** |
| 90+ @2055 | 9,971 | **8,172** |
| 2050 | 877,818 | 883,225 |
| 2025 base / 2027 trough | 799,358 / 797,298 | unchanged |

**Claimed root cause:** a TEST was overwriting the *production* survival table. The mortality pipeline
(`run_mortality_improvement_pipeline`) hard-coded its output to the production path, and
`tests/test_data/test_mortality_improvement.py::test_full_pipeline_produces_valid_output` called it with
`projection_horizon=20` and no output dir — so every `pytest` run rewrote the production survival table to
a 2025–2045 horizon. Two integration tests had even hard-coded 2045 as their expected horizon.

**Claimed fixes (in this diff):**
- `cohort_projections/data/process/mortality_improvement.py`: inject an `output_dir` (production default);
  bump the stale horizon default 20→30; record open-age correction provenance in metadata (N5).
- `tests/test_data/test_mortality_improvement.py`: pass `output_dir=tmp_path` so tests never touch production.
- `scripts/pipeline/02_run_projections.py`: a **coverage guard** in `load_demographic_rates` that warns when
  the operative survival table does not span `base_year..end_year`; plus a tightened state-aggregation glob
  (`*_{scenario}.parquet`, excluding `_components`) + a one-file-per-FIPS assertion.
- `cohort_projections/geographic/multi_geography.py`: recompute the per-county `_summary.csv` from the
  GQ-inclusive results (they were household-only).
- `tests/test_integration/test_census_method_validation.py`: two horizon assertions now derive from config /
  the actual table instead of hard-coding 2045.
- `docs/governance/adrs/068-...md` (Amendment) and `docs/methodology.md` (§4.3, §4.6, §10.x) updated;
  `docs/reviews/2026-06-16-corrected-run-qa-verification.md` records the QA.

# Engine context you need

- The cohort engine (`cohort_projections/core/cohort_component.py::_get_survival_rates`) returns the
  year-specific operative survival table if the calendar year is present, else falls back to
  `self.survival_rates` (the static race-specific base). The in-engine mortality-improvement factor is
  zeroed ONLY when the year is found in the operative table — so fallback years also get improvement
  re-applied from a 2023 anchor. The operative table is built by `mortality_improvement.py`
  (Census NP2023 national survival ratios × an ND adjustment factor; race-flat — a disclosed limitation).
- State is built bottom-up: state = Σ 53 counties (ADR-054). GQ is held constant at 2025 and re-added
  after the household-only projection (ADR-055), so components of change are household-basis.

# Anti-noise notes (do NOT raise these as defects)

- The diff contains an **unrelated** reusable reference doc, `docs/guides/gpt-5.5-pro-api-reference.md`
  (about calling GPT-5.5 Pro / LLM reasoning compute). It has nothing to do with the demography model;
  skim it, do not review it.
- Binaries (regenerated `.docx`/`.png`/`.xlsx`/`.pdf` marketing artifacts) are excluded from the diff;
  their numbers live in the included `PUB-2026 Draft Public Dataset.csv`.
- "10,051" appears in `docs/methodology_comparison_sdc_2024.md` as a *coincidental* 2020 SDC figure and is
  explained elsewhere as the (now-corrected) 3-year sum — neither is a live error.

# What I need from you (answer each explicitly)

1. **Root-cause correctness.** Is the diagnosis right — that the truncated survival table + silent
   static-base fallback explains the 2047–2055 bias? Is the offered evidence (2025–2046 identical across
   the two runs; divergence begins exactly at 2047, the first step needing `survival[2046]`) logically
   conclusive, or is there an alternative explanation?
2. **Number soundness.** Are **898,907 @2055 (+12.45%)** and **90+ @2055 = 8,172** sound as the
   full-horizon result? Sanity-check the direction/magnitude of the change: total *rose* +12,322 (claim:
   NP2023 carries higher survival than the static base at working/old ages) while the 90+ pool *fell*
   −1,799 (claim: the open-interval correction now applies through 2055). Do those two move in
   self-consistent directions, given lower 90+ survival but higher mid-age survival?
3. **Recurrence prevention — and is it complete?** Do the `output_dir` injection + the coverage guard
   actually prevent recurrence? **Critically: search the diff for ANY OTHER test or code path that writes
   to a production data path** (not just survival — fertility, migration, convergence, projection outputs,
   exports) via a hard-coded path or a pipeline call without a redirected output dir. This class of bug
   (a test mutating shared production state) is the real concern; find any siblings.
4. **Collateral & completeness.** Does regenerating the survival table affect the previously-vetted
   ref_intl / 90+ results? (The PR claims the 2025–2045 overlap is byte-identical — survival values
   unchanged, only 2046–2055 added.) Are the guard's bounds correct (off-by-one on `base_year..end_year`
   vs. the step `year→year+1`)? Is the `_summary.csv` GQ-inclusive recompute correct and free of
   double-counting? Flag any other correctness, reproducibility, or methodology issue you see in the diff,
   including ones not mentioned in this brief.

## Output format

Open with a one-line verdict: **GO** / **GO-WITH-FIXES** / **NO-GO** for merging this PR and publishing the
898,907 numbers. Then:
- **Fix-verification table** for the survival-horizon amendment (claim → evidence in diff → verdict).
- **Findings**, each tagged severity **[blocker] / [major] / [minor] / [nit]**, with concrete
  `file:line`-level references into the diff and a recommended action.
- A short **"what I could not verify from the diff alone"** list (e.g., anything needing the actual parquet
  outputs or a live run), so QA can close it.

Be concrete and cite the diff. If you believe a claimed number is wrong, show the reasoning. Default to
skepticism.
