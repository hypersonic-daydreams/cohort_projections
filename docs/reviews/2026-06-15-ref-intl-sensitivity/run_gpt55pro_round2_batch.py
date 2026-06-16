#!/usr/bin/env python3
"""Round-2 GPT-5.5 Pro review of the CORRECTED ND projection — via the OpenAI **Batch API**.

Round 1 (see gpt55pro_review_output.md) confirmed ~15 findings, including the two we then fixed
under ADR-068 (the reference_intl_migration 3-year-sum error and the open-ended 90+ survival
plateau) plus several it could not verify because only CODE — not the processed DATA — was supplied.

Round 2 sends the reviewer the CORRECTED product end-to-end, the previously-missing build code, and
curated DATA exports (docs/reviews/2026-06-15-ref-intl-sensitivity/round2/evidence_*.csv), and asks it
to (a) verify the two fixes are correctly implemented, (b) re-assess the still-open findings as
publication blockers or not, and (c) give a go/no-go to proceed to the publication/QA step.

Batch API (~50% cheaper than live, async up to 24h):
    python run_gpt55pro_round2_batch.py                 # assemble + write JSONL + cost estimate (NO call)
    python run_gpt55pro_round2_batch.py --send          # upload JSONL, create batch, poll, write output
    python run_gpt55pro_round2_batch.py --recover BATCH  # fetch results for an existing batch id

Stdlib only (urllib). Key read from env (OPENAI_API_KEY), never printed.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
R2 = "docs/reviews/2026-06-15-ref-intl-sensitivity/round2"

# ----------------------------------------------------------------------------------------------------
# Curated review surface. Files are re-read from disk at run time, so every doc/config/ADR reflects the
# CURRENT corrected (ADR-068) state. Order = the order the reviewer reads them.
# ----------------------------------------------------------------------------------------------------
FILES = [
    # === the disposition of round 1, read FIRST ===
    "docs/governance/adrs/068-ref-intl-numerator-and-open-ended-survival-correction.md",
    "docs/reviews/2026-06-15-ref-intl-sensitivity/gpt55pro_review_output.md",  # your own round-1 review
    # === master methodology + parameters (corrected) ===
    "docs/methodology.md",
    "config/projection_config.yaml",
    # === core engine + per-component calculation code ===
    "cohort_projections/core/cohort_component.py",
    "cohort_projections/core/fertility.py",
    "cohort_projections/core/mortality.py",
    "cohort_projections/core/migration.py",
    "cohort_projections/data/load/base_population_loader.py",
    "cohort_projections/data/process/base_population.py",
    "cohort_projections/data/process/fertility_rates.py",
    "cohort_projections/data/process/survival_rates.py",
    "cohort_projections/data/process/mortality_improvement.py",
    "cohort_projections/data/process/residual_migration.py",
    "cohort_projections/data/process/convergence_interpolation.py",
    "cohort_projections/data/process/migration_rates.py",
    # === NEW build code round 1 asked to see ===
    "cohort_projections/utils/demographic_utils.py",            # sprague_graduate() (5yr->1yr)
    "scripts/data/build_race_distribution_from_census.py",      # county dist build + (any) blend
    "cohort_projections/data/load/census_age_sex_population.py",  # county dist loader
    "scripts/data/fetch_census_gq_data.py",                     # GQ source/allocation build
    "cohort_projections/geographic/multi_geography.py",         # ADR-054 bottom-up aggregation
    "scripts/pipeline/01c_compute_mortality_improvement.py",    # mortality runner -> engine survival
    "scripts/pipeline/02_run_projections.py",                   # scenario-application ORDER (CBO vs cap)
    # === load-bearing decisions ===
    "docs/governance/adrs/037-cbo-grounded-scenario-methodology.md",
    "docs/governance/adrs/040-extend-boom-dampening-2015-2020.md",
    "docs/governance/adrs/041-census-pums-hybrid-base-population.md",
    "docs/governance/adrs/045-reservation-county-pep-recalibration.md",
    "docs/governance/adrs/050-restricted-growth-additive-migration-adjustment.md",
    "docs/governance/adrs/054-state-county-aggregation-reconciliation.md",
    "docs/governance/adrs/055-group-quarters-separation.md",
    "docs/governance/adrs/061-college-fix-model-revision.md",
    "docs/governance/adrs/065-cbo-adjusted-public-baseline.md",
    "docs/governance/adrs/066-vintage-2025-pep-base-population-refresh.md",
    "docs/governance/adrs/067-ward-grand-forks-divergence-investigation.md",
    # === the migration finding + the sensitivity that isolated it ===
    "docs/reviews/2026-06-15-ref-intl-migration-sum-vs-average.md",
    "docs/reviews/2026-06-15-ref-intl-sensitivity/README.md",
    "docs/reviews/2026-06-15-ref-intl-sensitivity/comparison_state_trajectory.csv",
    "docs/reviews/2026-06-15-ref-intl-sensitivity/comparison_county_2055.csv",
    # === NEW curated DATA evidence (round 1 could not verify these) ===
    f"{R2}/evidence_county_base_pop_2025.csv",          # 53 counties sum -> 799,358
    f"{R2}/evidence_pep_state_intl_2023_2025.csv",      # 3,158/4,083/2,810 -> sum 10,051 / mean 3,350.33
    f"{R2}/evidence_pep_reservation_counties.csv",      # ADR-045 Benson/Sioux/Rolette components
    f"{R2}/evidence_fertility_rates_FULL.csv",          # ASFR (5yr groups, per-1,000)
    f"{R2}/evidence_fertility_TFR.csv",                 # TFR by race, before/after -5% cut
    f"{R2}/evidence_survival_85plus.csv",               # open-90+ fix focus (all years, ages 85+)
    f"{R2}/evidence_survival_table_FULL.csv",           # full survival table — NO race column (race-flat)
    f"{R2}/evidence_county_distribution_SAMPLE.csv",    # county age-sex-race proportions (3-county sample)
    f"{R2}/evidence_gq_2025_by_county.csv",             # GQ held-constant county totals (state 30,876)
    f"{R2}/evidence_gq_2025_SAMPLE.csv",                # GQ age/sex allocation (2-county sample)
    f"{R2}/evidence_gq_historical_by_year.csv",         # GQ backward-constant construction by year
    # === validation evidence + run metadata ===
    "docs/reviews/2026-06-13-locked-run-sanity-check.md",
    "docs/reviews/2026-06-13-naive-method-value-add.md",
    "docs/reviews/2026-06-13-divergent-counties-methods-and-framing.md",
    "docs/methodology_comparison_sdc_2024.md",
    "docs/plans/2026-public-projection-release-handoff/final-run-metadata.md",
]

BRIEF = """You are an independent senior demographic-methods reviewer. This is a ROUND-2 follow-up
review for a public North Dakota cohort-component population projection. Your own round-1 review is
included below (FILE: .../gpt55pro_review_output.md) — read it; this round builds directly on it.

================================================================================
WHAT CHANGED SINCE YOUR ROUND-1 REVIEW
================================================================================
Round 1 you confirmed ~15 findings. We acted on the two highest-severity NUMERIC errors and disclosed
a third, under decision record ADR-068 (included, read it first), then did a CORRECTED PRODUCTION
RERUN. This package is the CORRECTED product, PLUS the build code and the processed DATA you said you
could not verify in round 1. Round 1 you were asked to ADJUDICATE an open finding; round 2 you are
asked to VERIFY the fixes were implemented correctly and to decide what still blocks publication.

--------------------------------------------------------------------------------
DISPOSITION OF YOUR ROUND-1 FINDINGS  (status -> where to verify in this package)
--------------------------------------------------------------------------------
FIXED (verify the implementation) — the FIRST TWO are the headline NUMERIC corrections under ADR-068;
the remaining three are lower-severity config-hygiene fixes made in the same rerun:
  * reference_intl_migration = 10,051 (3-yr SUM applied per year) -> set to the annual mean 3,350.33.
      Verify: config/projection_config.yaml (baseline + restricted alias) + core/migration.py;
      corroborate the numerator in evidence_pep_state_intl_2023_2025.csv (3,158+4,083+2,810=10,051; mean 3,350.33).
  * Open-ended 90+ survival plateau (~0.885 M / 0.914 F held flat at the 85+ rate) -> corrected to the
      open-interval survivorship ratio T91/T90 (~0.778 M / 0.806 F), NP2023 improvement trajectory preserved.
      Verify: mortality_improvement.py::apply_open_ended_survival_correction (called in build); and the
      operative table evidence_survival_85plus.csv / evidence_survival_table_FULL.csv at age==90 by sex/year.
  * sex_ratio_male 0.51 was a hard-coded default -> now an explicit config key at rates.fertility.sex_ratio_male
      (config line 140; read by core/fertility.py via config['rates']['fertility']['sex_ratio_male']).
  * aggregation_tolerance 0.01 (too loose) -> 0.000001.
  * place projections were config-enabled but out of scope -> place_projections.enabled=false; "place" removed
      from aggregation_levels.
CAVEATED, NOT FIXED (confirm the disclosure is adequate, not that it is rebuilt):
  * Race-flat mortality: the operative survival table is built from Census NP2023 all-race (no race column)
      and broadcast across races. ADR-068 D3 discloses this as a 2026-vintage limitation; a race-specific
      rebuild is deferred to the next vintage. Confirm: evidence_survival_table_FULL.csv HAS NO race column.
STILL OPEN (you flagged these; they were OUTSIDE the approved ADR-068 fix scope — re-assess each as a
publication BLOCKER vs acceptable-with-disclosure):
  * Small-county blend_threshold (5000): blending DOES happen, but at BUILD time in
      build_race_distribution_from_census.py (alpha = min(county_total / 5000, 1.0), blends county vs statewide
      proportions and re-normalizes) — driven by a CLI-default constant, NOT the config key
      demographics.county_distributions.blend_threshold, and base_population_loader.py does not apply it. Assess
      this config-key-vs-code disconnect (is the documented config knob actually wired to the build?).
  * Fertility provenance/expansion: config comment now says "pooled CDC WONDER 2020-2023", but the processed
      ASFR table (evidence_fertility_rates_FULL.csv) carries year=2023 only and 5-year age groups. There is NO
      expansion in fertility_rates.py / core/fertility.py, BUT a FLAT 5yr->single-year expansion DOES exist in
      scripts/pipeline/02_run_projections.py::_transform_fertility_rates -> _expand_age_groups_to_single_years
      (broadcasts each group's rate to every single year in the band — a flat copy, NOT Sprague graduation).
      Assess BOTH: (i) the doc-vs-data provenance mismatch (pooled-2020-2023 label vs year=2023-only table), and
      (ii) whether flat-broadcast (vs graduated) single-year ASFR is acceptable.
  * Newborns not exposed to infant mortality in the birth year (births computed pre-migration, concatenated after).
  * College smoothing updates migration_rate but not net_migration -> saved diagnostics inconsistent post-smoothing.
  * Contradictory mortality-provenance docs (config life_table_year 2023 vs methodology NVSR 2022 vs docstring).
  * GQ hold-constant makes published components of change household-basis (deaths look low) -> labeling risk.
  * CBO decrement is uniform across age/sex/race and applied AFTER the rate cap (can push rates past the cap on
      the negative side).

--------------------------------------------------------------------------------
CURRENT AUTHORITATIVE NUMBERS  (the corrected production run, ADR-068, config sha cca42fb42be76680)
--------------------------------------------------------------------------------
  2025 = 799,358  ->  trough 797,298 in 2027 (-0.26%)  ->  886,585 in 2055 (+10.9%).  90+ pop @2055 = 9,971.

RECONCILIATION (corrected run, independently verified for this package): state = sum of the 53 county
projections EXACTLY — residual 0.0 at both 2025 (799,358.0) and 2055 (886,585.25); the published county
outputs are GQ-INCLUSIVE. The held-constant GQ INPUT (evidence_gq_2025_by_county.csv) sums to 30,876 and is
the basis for the "published components of change are household-basis" labeling caveat (a STILL-OPEN item) — it
does NOT create a state-vs-county break. NOTE: the sensitivity folder's README cites a "-30,463.87 = state -
Sigma-counties = the GQ population" residual; that is a SEPARATE household-basis comparison from the OLD
sensitivity bundle, NOT the corrected-run state-vs-county check (which is 0.0). Do not expect the 30,876 GQ
input to equal that -30,463.87, and do not read the two as a reconciliation contradiction.

CRITICAL — three DIFFERENT number sets appear in this package; do not treat them as contradictions:
  (1) OLD LOCKED, now SUPERSEDED:  787,382 @2028 (-1.50% trough), 889,017 @2055.  The pre-fix run. It appears
      as clearly-marked historical/superseded text (e.g. the superseded table in final-run-metadata.md, and
      inside your own round-1 review). NOT current.
  (2) ref_intl-ONLY SENSITIVITY:   797,911 @2026, 904,692 @2055.  This isolated the MIGRATION fix ALONE
      (numerator only, 90+ untouched). It appears in the sensitivity README + comparison CSVs. It was the
      evidentiary control, NOT the production number.
      !! NAMING TRAP: the included sensitivity files LABEL this set-(2) value "corrected" — the column
      `corrected_run` (= 904692.25 @2055) in comparison_state_trajectory.csv, the "Corrected (3,350.33)" column
      in the sensitivity README, and the file config_corrected_refintl.yaml. That "corrected" means
      numerator-corrected-ONLY (set 2), NOT the production rerun. The authoritative production "corrected"
      numbers (set 3) live in final-run-metadata.md and ADR-068, NOT in the sensitivity folder's CSV/README.
      Also: the sensitivity README and your own round-1 review were written BEFORE the disposition and still
      phrase the fix as an OPEN/undecided call or a future "create ADR-068 / rerun" to-do. That framing is now
      superseded — ADR-068 is accepted and the rerun is COMPLETE. Read those files as historical context.
  (3) CORRECTED PRODUCTION (authoritative):  797,298 @2027, 886,585 @2055.  The ACTUAL rerun, applying BOTH
      fixes. It is LOWER at 2055 than the sensitivity's 904,692 because the 90+ open-ended survival fix REDUCES
      old-age survivorship: the two fixes nearly offset on the horizon (ref_intl ~ +15.7k, 90+ ~ -18.1k @2055).
      Derivation you can check: set(2) 904,692 - set(1) 889,017 = +15,675 (ref_intl effect; both numbers are in
      the sensitivity README/CSV); set(2) 904,692 - set(3) 886,585 = 18,107 (the 90+ effect; set-3 886,585 is in
      final-run-metadata.md / ADR-068). NO single file isolates the 90+ delta on its own — confirm it via that
      subtraction. This is expected arithmetic, not an inconsistency. Treat set (3) as authoritative everywhere.

  BLANKET RULE FOR STALE NUMBERS: several included docs PREDATE ADR-068 and still show pre-fix totals as if
  current, WITHOUT a superseded banner — specifically 889,017 @2055 / 787,382 @2028 in
  docs/methodology_comparison_sdc_2024.md and docs/reviews/2026-06-13-locked-run-sanity-check.md, plus the
  even-earlier provisional 876,479 (ADR-067 F4, labeled "Reference (production baseline)" = the 2026-05-27
  pre-lock draft) and 882,146 (ADR-055, a Feb-2026 intermediate). Treat ANY state-2055 total other than 886,585,
  and ANY near-term trough other than 797,298 @2027, that you encounter in an included doc as HISTORICAL /
  stale doc-text. Such stale text is a known publication-text-freeze item (already on our list, and you may note
  it as a recommendation) — it is NOT evidence the corrected model is internally inconsistent. Do not raise it
  as a model contradiction. (Coincidental: "10,051" also appears in methodology_comparison_sdc_2024.md as a 2020
  total-births figure — unrelated to the migration numerator.)

--------------------------------------------------------------------------------
NEW EVIDENCE IN THIS PACKAGE (closes round-1 "could not verify" gaps)
--------------------------------------------------------------------------------
  base pop -> evidence_county_base_pop_2025.csv (53 counties -> 799,358) ; numerator -> evidence_pep_state_intl_2023_2025.csv ;
  reservation recalibration -> evidence_pep_reservation_counties.csv ; fertility -> evidence_fertility_rates_FULL.csv +
  evidence_fertility_TFR.csv ; survival/90+/race-flat -> evidence_survival_85plus.csv + evidence_survival_table_FULL.csv ;
  county distribution + blend -> build_race_distribution_from_census.py + evidence_county_distribution_SAMPLE.csv ;
  GQ -> fetch_census_gq_data.py + evidence_gq_2025_by_county.csv + evidence_gq_2025_SAMPLE.csv + evidence_gq_historical_by_year.csv ;
  graduation -> demographic_utils.py::sprague_graduate ; aggregation -> multi_geography.py ;
  mortality runner / engine survival expansion -> 01c_compute_mortality_improvement.py ;
  scenario ORDER (CBO vs rate cap) -> 02_run_projections.py + core/migration.py.

--------------------------------------------------------------------------------
SCOPE — what is DELIBERATELY NOT in this package (do not flag as missing/error)
--------------------------------------------------------------------------------
The public-facing artifacts (the public PDF, public Excel workbook, public CSV, marketing .docx, and the
interactive pyramid explorer) and the 6 release-QA gates have NOT yet been regenerated against the corrected
run. That regeneration + QA pass is the DELIBERATE post-review publication step, intentionally sequenced
AFTER this review so it is run once against numbers you have blessed. Your job here is the METHODOLOGY,
ENGINE, and CORRECTED NUMBERS — not the rendered public presentation. Absence of those artifacts is by design.

================================================================================
YOUR CHARGE (ROUND 2)
================================================================================
Audit the corrected projection END TO END (base population & vintage -> fertility -> mortality -> migration
residual -> convergence -> CBO scenario adjustment -> special populations [college / Bakken / GQ / reservation]
-> engine & order-of-operations -> bottom-up state aggregation -> outputs). Specifically:

  1. FIX VERIFICATION. For each FIXED item above, confirm from the code + data that the fix is correctly and
     completely implemented (not just relabeled). Quote the file + exact value/line. Flag any fix that is
     partial, wrong, or inconsistent with the docs.
  2. NUMBER CONSISTENCY. Confirm the corrected set (3) is internally consistent: state = sum of 53 counties;
     the 799,358 base; the +15.7k / -18.1k offset story; 90+ @2055 = 9,971. Flag any reconciliation gap.
  3. STILL-OPEN ITEMS. For each STILL-OPEN finding, give a severity (blocker / major / minor) FOR PUBLICATION
     and a clear verdict: must-fix-before-release, or acceptable-with-disclosure (and if so, what disclosure).
  4. NEW ISSUES. With the build code + data now in hand, surface anything NEW you could not see in round 1.
  5. GO / NO-GO. A single clear recommendation: are the corrected numbers sound enough to PROCEED to the
     publication/QA-gate step? If NO-GO, the minimal blocker list to clear first.

Be specific, quantitative, and adversarial. Cite the material (file + value). If something you need is still
not in the provided text, say so explicitly rather than inferring.

================================================================================
BEGIN REVIEW PACKAGE
================================================================================
"""

MODEL = "gpt-5.5-pro"
# Batch API ~50% off live pricing (live is $30/$180 per M in/out).
IN_PRICE, OUT_PRICE = 15.0, 90.0
REASONING_EFFORT = "xhigh"  # Batch commits to ONE effort (no synchronous fallback). Highest = xhigh.
MAX_OUTPUT = None           # no cap — reasoning bills as output; a cap can truncate the answer.
ENDPOINT = "/v1/responses"
CUSTOM_ID = "nd-projection-round2-corrected"
API = "https://api.openai.com"


def assemble() -> str:
    parts = [BRIEF]
    for rel in FILES:
        path = os.path.join(REPO, rel)
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                content = fh.read()
        except FileNotFoundError:
            print(f"!! MISSING (skipped): {rel}")
            continue
        bar = "=" * 80
        parts.append(f"\n\n{bar}\n=== FILE: {rel} ===\n{bar}\n\n{content}")
    return "\n".join(parts)


def _post_json(url: str, key: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


def _get_json(url: str, key: str) -> dict:
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


def _upload_jsonl(path: str, key: str) -> str:
    """Multipart upload of the batch JSONL via the Files API (purpose=batch). Stdlib only."""
    boundary = f"----batch{uuid.uuid4().hex}"
    with open(path, "rb") as fh:
        filedata = fh.read()
    pre = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"purpose\"\r\n\r\nbatch\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"{os.path.basename(path)}\"\r\nContent-Type: application/jsonl\r\n\r\n"
    ).encode("utf-8")
    body = pre + filedata + f"\r\n--{boundary}--\r\n".encode("utf-8")
    req = urllib.request.Request(
        f"{API}/v1/files", data=body,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.load(resp)["id"]


def _extract_text(response_body: dict) -> str:
    text = response_body.get("output_text")
    if text:
        return text
    chunks = []
    for item in response_body.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text":
                chunks.append(c.get("text", ""))
    return "\n".join(chunks)


def _download_results(batch: dict, key: str) -> None:
    out_id = batch.get("output_file_id")
    err_id = batch.get("error_file_id")
    if err_id:
        try:
            err = _get_json(f"{API}/v1/files/{err_id}/content", key)
            print("error_file:", json.dumps(err)[:2000])
        except Exception as e:  # noqa: BLE001
            print("error_file fetch failed:", e)
    if not out_id:
        print("no output_file_id on batch; status=", batch.get("status"))
        return
    req = urllib.request.Request(f"{API}/v1/files/{out_id}/content",
                                 headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        raw = resp.read().decode("utf-8", "replace")
    text, usage = "", {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        rb = (rec.get("response") or {}).get("body") or {}
        text = _extract_text(rb)
        usage = rb.get("usage", {})
    out_path = os.path.join(HERE, "gpt55pro_round2_output.md")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text or "(empty response)")
    print("review written:", out_path)
    it, ot = usage.get("input_tokens"), usage.get("output_tokens")
    print("usage:", json.dumps(usage))
    if it and ot:
        print(f"ACTUAL batch cost = ${it/1e6*IN_PRICE + ot/1e6*OUT_PRICE:.2f} "
              f"(input {it:,} @ ${IN_PRICE:.0f}/M, output {ot:,} @ ${OUT_PRICE:.0f}/M)")


LIVE_IN_PRICE, LIVE_OUT_PRICE = 30.0, 180.0  # live Responses API $/M (no batch discount)


def _write_output(text: str, usage: dict, in_price: float, out_price: float) -> None:
    out_path = os.path.join(HERE, "gpt55pro_round2_output.md")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text or "(empty response)")
    print("review written:", out_path)
    it, ot = usage.get("input_tokens"), usage.get("output_tokens")
    rt = (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
    print("usage:", json.dumps(usage))
    if rt is not None:
        print(f"  reasoning tokens: {rt:,} of {ot:,} output")
    if it and ot:
        print(f"ACTUAL cost = ${it/1e6*in_price + ot/1e6*out_price:.2f} "
              f"(input {it:,} @ ${in_price:.0f}/M, output {ot:,} @ ${out_price:.0f}/M)")


def _recover_response(rid: str, key: str) -> None:
    data = _get_json(f"{API}/v1/responses/{rid}", key)
    print(f"response {rid}: status={data.get('status')}")
    if data.get("status") == "completed":
        _write_output(_extract_text(data), data.get("usage", {}), LIVE_IN_PRICE, LIVE_OUT_PRICE)
    elif data.get("status") in ("failed", "cancelled", "incomplete"):
        print("terminal non-success:",
              json.dumps(data.get("error") or data.get("incomplete_details") or {})[:800])
    else:
        print("not complete yet; re-run --recover-response later.")


def run_live(no_poll: bool) -> None:
    """Synchronous Responses API (background mode + poll). ~2x batch cost, returns in ~1-2h."""
    pkg = assemble()
    est_in = len(pkg) // 4
    in_cost = est_in / 1e6 * LIVE_IN_PRICE
    print(f"package: {len(pkg):,} chars ~={est_in:,} input tokens (est)")
    print(f"files: {sum(os.path.exists(os.path.join(REPO, f)) for f in FILES)}/{len(FILES)} present")
    print(f"LIVE est input ${in_cost:.2f} + output ~40k @ ${LIVE_OUT_PRICE:.0f}/M -> "
          f"+${40000/1e6*LIVE_OUT_PRICE:.2f} (~${in_cost + 40000/1e6*LIVE_OUT_PRICE:.2f} typical total)")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("ERROR: OPENAI_API_KEY not set — aborting, NO call made.")
        sys.exit(2)
    created, chosen = None, None
    for eff in ["xhigh", "high", None]:
        body = {"model": MODEL, "input": pkg, "background": True}
        if eff:
            body["reasoning"] = {"effort": eff}
        try:
            print(f"creating background response (effort={eff or 'default'})...")
            created = _post_json(f"{API}/v1/responses", key, body)
            chosen = eff or "default"
            break
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")
            if e.code == 400 and eff and ("effort" in msg.lower() or "reasoning" in msg.lower()):
                print(f"effort={eff} rejected (400); lowering...")
                continue
            print("create FAILED", e.code, msg[:1500])
            sys.exit(3)
    rid = created.get("id")
    print(f"created response id={rid} status={created.get('status')} effort={chosen}")
    print(f"  recover later:  python {os.path.basename(__file__)} --recover-response {rid}")
    if no_poll:
        print("--no-poll set; response is generating in the background, not polling.")
        return
    waited, deadline = 0, 10800  # 3h
    data, status = created, created.get("status")
    while status in ("queued", "in_progress"):
        time.sleep(20)
        waited += 20
        if waited > deadline:
            print(f"poll timeout after {waited}s; recover via --recover-response {rid}")
            return
        try:
            data = _get_json(f"{API}/v1/responses/{rid}", key)
        except urllib.error.HTTPError as e:
            print("poll HTTPError", e.code, e.read().decode("utf-8", "replace")[:800])
            break
        status = data.get("status")
        print(f"  [{waited}s] status={status}")
    print(f"final status={status}")
    if status == "completed":
        _write_output(_extract_text(data), data.get("usage", {}), LIVE_IN_PRICE, LIVE_OUT_PRICE)
    else:
        print("non-success:", json.dumps(data.get("error") or data.get("incomplete_details") or {})[:800])


def main() -> None:
    # --live: synchronous Responses API instead of batch (returns in ~1-2h, ~2x cost).
    if "--recover-response" in sys.argv:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            print("ERROR: OPENAI_API_KEY not set.")
            sys.exit(2)
        _recover_response(sys.argv[sys.argv.index("--recover-response") + 1], key)
        return
    if "--live" in sys.argv:
        run_live("--no-poll" in sys.argv)
        return

    # --recover <batch_id>: fetch results for an already-submitted batch.
    if "--recover" in sys.argv:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            print("ERROR: OPENAI_API_KEY not set.")
            sys.exit(2)
        bid = sys.argv[sys.argv.index("--recover") + 1]
        batch = _get_json(f"{API}/v1/batches/{bid}", key)
        print(f"batch {bid}: status={batch.get('status')} counts={batch.get('request_counts')}")
        if batch.get("status") == "completed":
            _download_results(batch, key)
        else:
            print("not complete yet; re-run --recover later.")
        return

    pkg = assemble()
    body = {"model": MODEL, "input": pkg, "reasoning": {"effort": REASONING_EFFORT}}
    if MAX_OUTPUT:
        body["max_output_tokens"] = MAX_OUTPUT
    request_line = {"custom_id": CUSTOM_ID, "method": "POST", "url": ENDPOINT, "body": body}
    jsonl_path = os.path.join(HERE, "round2_batch_input.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(request_line) + "\n")

    est_in = len(pkg) // 4
    in_cost = est_in / 1e6 * IN_PRICE
    typ = 40000
    print(f"package: {len(pkg):,} chars  ~={est_in:,} input tokens (est)")
    print(f"files: {sum(os.path.exists(os.path.join(REPO, f)) for f in FILES)}/{len(FILES)} present")
    print(f"reasoning effort: {REASONING_EFFORT}   output cap: "
          + (f"{MAX_OUTPUT:,}" if MAX_OUTPUT else "NONE"))
    print(f"est BATCH input cost (${IN_PRICE:.0f}/M) = ${in_cost:.2f}")
    print(f"+ output ~{typ//1000}k @ ${OUT_PRICE:.0f}/M -> +${typ/1e6*OUT_PRICE:.2f} "
          f"(~${in_cost + typ/1e6*OUT_PRICE:.2f} typical total, BATCH; live would be ~2x)")
    print(f"JSONL written: {jsonl_path}")

    if "--send" not in sys.argv:
        print("\nDRY RUN — no API call. Re-run with --send (needs OPENAI_API_KEY) to submit the batch.")
        return

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("\nERROR: OPENAI_API_KEY not set — aborting, NO batch created.")
        sys.exit(2)

    print("\nuploading JSONL to Files API (purpose=batch)...")
    file_id = _upload_jsonl(jsonl_path, key)
    print("input_file_id:", file_id)

    print("creating batch...")
    try:
        batch = _post_json(f"{API}/v1/batches", key, {
            "input_file_id": file_id, "endpoint": ENDPOINT, "completion_window": "24h",
            "metadata": {"purpose": "nd-projection-round2-corrected-review"}})
    except urllib.error.HTTPError as e:
        msg = e.read().decode("utf-8", "replace")
        print(f"batch create FAILED ({e.code}): {msg[:1500]}")
        if "endpoint" in msg.lower() or "url" in msg.lower():
            print("NOTE: /v1/responses may not be batch-supported for this account; "
                  "reshape body to /v1/chat/completions or fall back to the live runner.")
        sys.exit(3)
    bid = batch.get("id")
    print(f"batch created: id={bid} status={batch.get('status')}")
    print(f"  recover later with:  python {os.path.basename(__file__)} --recover {bid}")

    if "--no-poll" in sys.argv:
        print("--no-poll set; batch submitted, not polling. Fetch with --recover when ready.")
        return

    # Poll (generous; batch can take minutes to 24h).
    waited, deadline, interval = 0, 24 * 3600, 60
    while batch.get("status") in ("validating", "in_progress", "finalizing"):
        time.sleep(interval)
        waited += interval
        try:
            batch = _get_json(f"{API}/v1/batches/{bid}", key)
        except urllib.error.HTTPError as e:
            print("poll HTTPError", e.code, e.read().decode("utf-8", "replace")[:800])
            break
        print(f"  [{waited//60}m] status={batch.get('status')} counts={batch.get('request_counts')}")
        if waited > deadline:
            print(f"poll deadline reached; recover later with --recover {bid}")
            return
    print(f"final status={batch.get('status')}")
    if batch.get("status") == "completed":
        _download_results(batch, key)
    else:
        print(f"batch not completed (status={batch.get('status')}); inspect with --recover {bid}")


if __name__ == "__main__":
    main()
