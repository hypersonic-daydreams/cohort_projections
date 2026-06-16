# Handoff: PR #25 Review-Hardening (GPT-5.5 Pro GO-WITH-FIXES)

> ✅ **IMPLEMENTED 2026-06-16.** All three majors (M1 hard-fail survival guard + `allow_static_survival`
> opt-out; M2 filename-horizon match + exactly-53 county-set equality; M3 pytest production-write block)
> and all three minors (N1 evidence banners + README; N2 guard range-convention comment; N3 sensitivity-config
> warnings) are done, plus +7 regression tests proving each guard fires. Full suite **2,267 passed, 5 skipped**;
> no projection number moved (production survival still spans 2025–2055; state-2055 still 898,907; both guards
> proven silent on real production data). Majors committed `c1e921a`; minors + doc wording in the follow-up
> commit. ADR-068 amendment carries the recurrence-hardening note. This file is retained as the completed
> work order / audit record.

| Attribute | Value |
|-----------|-------|
| **Created** | 2026-06-16 |
| **For** | The next agent/session hardening PR #25 before merge/publication |
| **Source** | The PR-framed GPT-5.5 Pro review: [`gpt55pro_pr25_review_output.md`](./gpt55pro_pr25_review_output.md) (verdict **GO-WITH-FIXES**) |
| **PR** | #25 (`docs/ref-intl-sum-vs-average-finding`), reviewed at commit `fcb0432`, base `master` |
| **Status of numbers** | **FINAL & VERIFIED SOUND — do not change them.** This is a **code-hardening** pass, not a numeric one. |

## Start here (orientation)

PR #25 carries the full ADR-068 correction, including the **2026-06-16 survival-horizon amendment**
(the operative survival table had been truncated to 2025–2045 by a test that overwrote production data;
the engine then silently fell back to the uncorrected static-base survival for 2047–2055). A PR-framed
**GPT-5.5 Pro** review (live+background, `xhigh`, 599k input tokens, $23.33) returned **GO-WITH-FIXES**:

- **Confirmed sound** (do NOT revisit): the root-cause diagnosis (2025–2046 identical / first divergence
  at 2047 = a missing `survival[2046]`), the corrected numbers, and the GQ-inclusive `_summary.csv`
  recompute. The corrected totals are **898,907 @2055 (+12.45%)** and **90+ @2055 = 8,172**.
- **Remaining work = recurrence-hardening only.** The fixes below make the anti-regression guards
  *hard* (the original bug survived precisely because a guard was only a warning). **None of them should
  change any projection number** — they fire only on bad states; in the good state they must be silent.

Read this file + the review output first. Everything below is derived from that review.

## The two rules

1. **Do NOT change the projection numbers.** 898,907 / 8,172 are verified by the review AND by local QA.
   If a "fix" would move a number, you have broken something — stop and re-check. The guards must be
   no-ops on the current good state.
2. **The theme is "a test/process must never mutate shared production data."** Every item below is a
   variant of that. Implement defensively and prove the production path still works.

## Authoritative state (assert unchanged after your work)

| Quantity | Value |
|----------|-------|
| 2025 base | 799,358 |
| 2027 trough | 797,298 (−0.26%) |
| 2050 | 883,225 |
| 2055 | **898,907.01 (+12.45%)** |
| 90+ @2055 | **8,172** |
| State vs Σ counties | 0.0 (1e-10) residual |
| Operative survival coverage | 2025–2055 (31 yrs) |
| config sha256(16) | `a6e0bfbc2d70be85` |
| Reviewed commit | `fcb0432` |

## Already verified locally — DO NOT redo (the review's "couldn't verify from the diff" list)

These were closed during the 2026-06-16 session (the reviewer just couldn't see them from the diff):
- Production `nd_adjusted_survival_projections.parquet` spans **2025–2055 after a full pytest run**.
- The 2025–2045 survival overlap is **byte-identical** (max abs diff 0.0) to the pre-amendment table.
- **90+ @2055 = 8,172.40** (from the regenerated baseline parquet).
- State = Σ county residual **~1e-10**; **53 unique county FIPS**, one current file each.
- `mortality_improvement_metadata.json` carries the open-age provenance (N5).

---

## §1 — MUST-FIX before merge/publish (recurrence-hardening; no numeric change)

**M1 — Make the survival-coverage guard HARD-FAIL for production/public runs.** Currently
[`load_demographic_rates`](../../../scripts/pipeline/02_run_projections.py#L387) only **logs a warning**
when the operative survival table doesn't span `base_year..end_year` (~line 533–545), and the
no-operative-table branch only logs INFO (~line 546). A warning is exactly what let the original bug
through. → Raise `RuntimeError` on a missing/incomplete operative survival table in the production /
public baseline path, with an explicit opt-out (e.g. `allow_static_survival=True` param or a config
flag) for tests/experiments that intentionally run without it.
- *Watch out:* confirm the **production path does NOT trip** (its survival spans 2025–2055 → guard
  passes silently). Confirm the experiment/sensitivity harness and the synthetic unit tests either don't
  reach this path or set the opt-out — otherwise you'll break them. *Acceptance:* production stage-02
  runs unchanged (898,907); a deliberately-truncated table raises; tests pass.

**M2 — Enforce exactly 53 *current* county files in state aggregation.**
[`aggregate_county_results_to_state`](../../../scripts/pipeline/02_run_projections.py#L1110) (~line
1141–1175) catches duplicate FIPS but would still accept a 52-county partial run, OR a *complete set of
stale `2025_2045` county files* — the same truncation class as the bug. → Parse each filename's
`base_end` years and require they match the run's `base_year..end_year` (kills stale-horizon files); and
assert the FIPS set equals the expected ND county set (53 for `counties.mode: all`).
- *Watch out:* don't break legitimate subset runs (`mode: list`/`threshold`) — assert 53 only for
  `mode: all`, or assert against the run's expected county set. *Acceptance:* a stray `*_2025_2045_*`
  county file or a missing county fails aggregation loudly; the current 53-file production set passes.

**M3 — Stop the mortality pipeline from defaulting to a production write under test.**
[`run_mortality_improvement_pipeline`](../../../cohort_projections/data/process/mortality_improvement.py#L386)
now accepts `output_dir` (production default at ~line 438), and the known offending test passes
`tmp_path`. But a *future* test that omits `output_dir` would silently re-clobber production. → Add a
belt-and-suspenders: if `output_dir is None` **and** `os.environ.get("PYTEST_CURRENT_TEST")` is set,
raise (force tests to pass a temp dir). *Acceptance:* calling the pipeline with no `output_dir` under
pytest raises; the production CLI (`scripts/pipeline/01c_compute_mortality_improvement.py`) still writes
production normally.

## §2 — SHOULD-FIX (minor; quick)

- **N1 — Stale review evidence implies the wrong horizon.** `round2/evidence_survival_table_FULL.csv`
  and `evidence_survival_85plus.csv` stop at **2045** (they predate the amendment), and the name "FULL"
  is misleading. Also [`docs/reviews/README.md`](../README.md) still lists the ref-intl finding as
  "Open… disposition pending." → Add superseded banners (or regenerate/rename), and update the README
  status line to point at ADR-068 + the 2026-06-16 amendment.
- **N2 — Document the guard's inclusive-range convention.** The engine needs survival for step years
  `base..end-1`; the guard requires `base..end` (inclusive). Acceptable, but add a one-line comment that
  this is a publication convention (full inclusive table), not an engine necessity.
- **N3 — Sensitivity/review configs retain production output roots.** The committed
  `data/projections/sensitivity_*/config_*.yaml` isolate `pipeline.projection.output_dir` but keep
  `data_processing.output_dir: data/processed`, `housing_unit_method.output_dir: data/projections`, etc.
  Not used by the documented repro command, but a sibling "shared-state write" risk. → Override those to
  a sandbox dir, or add a comment that only stage 02 is safe to run from those configs.

## §3 — After implementing

1. **Run `pytest`** (`-k "not test_residual_computation_single_period"`). Expect **2,260 passed, 5
   skipped**. THE key regression check: **after the full run, production survival must still span
   2025–2055** (`python -c "import pandas as pd; print(pd.read_parquet('data/processed/mortality/nd_adjusted_survival_projections.parquet').year.max())"` → 2055).
2. **Confirm no number moved:** state 2055 still 898,907.01; production stage-02 path runs without the
   new guard tripping.
3. **Docs:** if M1 changes guard behavior from warn→raise, update the guard's comment + methodology §4.6
   / ADR-068 amendment wording ("warns" → "hard-fails") and the QA verification note.
4. **Commit** on `docs/ref-intl-sum-vs-average-finding` (PR #25); push.

## Optional: a confirming re-review

The hardening diff is small and mechanical, so a re-review is optional. If wanted, the runner is ready:
[`run_pr_review.py`](./run_pr_review.py) regenerates the `master...HEAD` diff + [`review_brief.md`](./review_brief.md)
and submits live+background (`--send`; **never batch** — see [the API guide](../../guides/gpt-5.5-pro-api-reference.md)).
Default is a dry run. Key is loaded via `direnv exec .` (project `.env`); never printed.

## Git / data state at handoff

- PR #25 pushed at commit `fcb0432` (the reviewed state). The review artifacts (`review_brief.md`,
  `run_pr_review.py`, `gpt55pro_pr25_review_output.md`, this handoff) are committed; the 1.4 MB
  `pr25_review_input.txt` blob is gitignored (regenerable via the runner).
- Numbers/data are final; the regenerated survival + projection + public artifacts are on disk and
  gitignored — `./scripts/bisync.sh` to sync to another machine.
- Agent memory updated: `pub-2026-ref-intl-sum-vs-average.md` (GO-WITH-FIXES + this punch list).
