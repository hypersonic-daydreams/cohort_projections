#!/usr/bin/env python3
"""Assemble the ND-projection review package and (optionally) send it to GPT-5.5 Pro.

Usage:
    python run_gpt55pro_review.py            # assemble + write package + print cost estimate (NO API call)
    python run_gpt55pro_review.py --send     # also call gpt-5.5-pro (needs OPENAI_API_KEY in env)

Uses only the Python standard library (urllib) — no openai SDK needed. The key is read from the
environment and never printed. Package -> /tmp; review output -> this folder (git-tracked).
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

# Curated review surface: the full assumption/calculation chain, start to finish.
FILES = [
    # --- master methodology + parameters ---
    "docs/methodology.md",
    "config/projection_config.yaml",
    # --- core engine + per-component calculation code ---
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
    # --- load-bearing decisions ---
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
    # --- the open finding + the locked-config sensitivity run (central review target) ---
    "docs/reviews/2026-06-15-ref-intl-migration-sum-vs-average.md",
    "docs/reviews/2026-06-15-ref-intl-sensitivity/README.md",
    "docs/reviews/2026-06-15-ref-intl-sensitivity/comparison_state_trajectory.csv",
    "docs/reviews/2026-06-15-ref-intl-sensitivity/comparison_county_2055.csv",
    # --- validation evidence ---
    "docs/reviews/2026-06-13-locked-run-sanity-check.md",
    "docs/reviews/2026-06-13-naive-method-value-add.md",
    "docs/reviews/2026-06-13-divergent-counties-methods-and-framing.md",
    "docs/methodology_comparison_sdc_2024.md",
    "docs/plans/2026-public-projection-release-handoff/final-run-metadata.md",
]

BRIEF = """You are an independent senior demographic-methods reviewer. Audit this North Dakota
cohort-component population projection END TO END — every assumption and every calculation — and
report what is sound, what is wrong, and what is risky. This is for a public government release, so
hold it to a high standard.

WHAT YOU HAVE (full text below, in order): the master methodology, the locked production config, the
core engine + per-component calculation code (base population, fertility, mortality, migration residual
+ convergence), the load-bearing ADRs, an OPEN FINDING plus a locked-config SENSITIVITY run that tests
it, and the validation evidence. Everything you need to check the chain is here; if something you need
is NOT in the provided text, say so explicitly rather than inferring.

REVIEW THE CHAIN IN ORDER:
  base population & vintage -> fertility -> mortality -> migration (residual computation -> convergence
  interpolation) -> CBO scenario adjustment -> special populations (college / Bakken / group quarters /
  reservation) -> projection engine & order-of-operations -> bottom-up state aggregation -> outputs.

CENTRAL REVIEW TARGET — the open finding: `reference_intl_migration = 10,051` in the config is the
3-YEAR SUM of ND international migration (2023+2024+2025 = 3,158+4,083+2,810), but it is labeled
"annual average" and applied PER YEAR in the additive CBO migration reduction (migration.py). The true
annual average is 3,350.33 — so the near-term suppression appears ~3x too large. A locked-config
sensitivity run (only the numerator corrected to 3,350.33; a control run with the numerator unchanged
reproduced the published locked trajectory to the person, max diff 0.0000) turns the published −1.50%
trough at 2028 into a −0.18% blip at 2026, and lifts 2055 from 889,017 to 904,692. Assess: (a) is the
sum-vs-average diagnosis correct from the config + migration.py? (b) is the corrected run the right
basis? (c) should the locked numbers be published while this is undispositioned?

ALSO SCRUTINIZE (our own inventory flagged these; confirm or dismiss each): the base population using
a Vintage-2024 age/sex/race STRUCTURE scaled to Vintage-2025 TOTALS; the mortality "two-track" design
(whether a race-flattened national track overrides race-specific life tables for all years) and the
~0.885 survival plateau at 90+; the fertility flat −5% cut and any code-vs-doc gap in the 5-year→
single-year expansion and the hard-coded 0.51 sex ratio; the migration adjustment ORDER and any
compounding of oil×male dampening; the GQ hold-constant; and any stale/contradictory numbers.

AUTHORITATIVE NUMBERS: the locked run `m2026r1` (799,358 in 2025 → 787,382 in 2028 → 889,017 in 2055)
and the corrected sensitivity (above) are authoritative. IGNORE pre-lock figures (e.g. 876,479 / 882,146)
and known-stale doc numbers if you encounter them.

DELIVERABLE:
1. A per-link findings table: for each stage, the key assumptions/calculations, whether they check out
   against the code, and any error/risk (quote the specific file + the exact value or line you evaluated).
2. A prioritized issue list with severity (blocker / major / minor) and a recommended fix for each.
3. A clear verdict on the open finding (is 10,051 an error; is correct-and-rerun the right disposition).
4. A go / no-go recommendation on publishing the locked numbers as-is, with your reasoning.
Be specific, quantitative, and adversarial. Cite the material. Flag anything you could not verify.

================================================================================
BEGIN REVIEW PACKAGE
================================================================================
"""

MODEL = "gpt-5.5-pro"
IN_PRICE, OUT_PRICE = 30.0, 180.0  # $/M tokens
# No output cap: reasoning tokens are billed AS output, and max_output_tokens bounds
# reasoning + answer combined — a cap can exhaust on reasoning and truncate the answer
# (status="incomplete") while still billing for it. Billing is per ACTUAL token, so
# omitting the cap costs nothing extra unless the model genuinely needs the tokens.
MAX_OUTPUT = None
# Highest reasoning first; fall back if the API rejects the effort value for this model.
REASONING_EFFORTS = ["xhigh", "high", None]


def assemble() -> str:
    parts = [BRIEF]
    for rel in FILES:
        path = os.path.join(REPO, rel)
        with open(path, encoding="utf-8", errors="replace") as fh:
            content = fh.read()
        bar = "=" * 80
        parts.append(f"\n\n{bar}\n=== FILE: {rel} ===\n{bar}\n\n{content}")
    return "\n".join(parts)


def main() -> None:
    pkg = assemble()
    pkg_path = "/tmp/gpt55pro_review_package.md"
    with open(pkg_path, "w", encoding="utf-8") as fh:
        fh.write(pkg)
    est_in = len(pkg) // 4  # chars/4 approximation
    in_cost = est_in / 1e6 * IN_PRICE
    print(f"package: {len(pkg):,} chars  ~={est_in:,} input tokens (est)")
    print(f"files: {len(FILES)}")
    print(f"est input cost ({MODEL} ${IN_PRICE:.0f}/M) = ${in_cost:.2f}")
    typ = 35000  # rough "thorough review" output (reasoning + answer); billed at ACTUAL usage
    print(f"+ output (reasoning+answer @ ${OUT_PRICE:.0f}/M, billed per actual token): "
          f"~{typ // 1000}k typical -> +${typ / 1e6 * OUT_PRICE:.2f} (~${in_cost + typ / 1e6 * OUT_PRICE:.2f} typical total)")
    print("  output cap: " + (f"{MAX_OUTPUT:,} tokens" if MAX_OUTPUT else "NONE (model decides; reasoning counts as output)"))
    print(f"package written: {pkg_path}")

    if "--send" not in sys.argv:
        print("\nDRY RUN — no API call. Add OPENAI_API_KEY to the env and re-run with --send.")
        return

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("\nERROR: OPENAI_API_KEY not set — aborting, NO billed call made.")
        sys.exit(2)

    auth = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    # 1) Create a BACKGROUND response (survives long generations / dropped connections),
    #    trying the highest reasoning effort first and falling back if rejected.
    created = None
    chosen = None
    for eff in REASONING_EFFORTS:
        body = {"model": MODEL, "input": pkg, "background": True}
        if MAX_OUTPUT:
            body["max_output_tokens"] = MAX_OUTPUT
        if eff:
            body["reasoning"] = {"effort": eff}
        req = urllib.request.Request(
            "https://api.openai.com/v1/responses", data=json.dumps(body).encode("utf-8"), headers=auth
        )
        label = eff or "default"
        print(f"\ncreating background response on {MODEL} (reasoning effort={label})...")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                created = json.load(resp)
            chosen = label
            break
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")
            if e.code == 400 and eff is not None and ("effort" in msg.lower() or "reasoning" in msg.lower()):
                print(f"effort={label} rejected (400); retrying lower...")
                continue
            print("HTTPError (create)", e.code, msg[:3000])
            sys.exit(3)
    if not created:
        print("create failed for all effort levels.")
        sys.exit(4)

    rid = created.get("id")
    status = created.get("status")
    print(f"created id={rid} status={status} effort={chosen}")

    # 2) Poll until terminal.
    waited, deadline = 0, 3600
    data = created
    while status in ("queued", "in_progress"):
        time.sleep(20)
        waited += 20
        if waited > deadline:
            print(f"poll timeout after {waited}s; id={rid} still running (recover later via GET /v1/responses/{rid}).")
            sys.exit(5)
        g = urllib.request.Request(f"https://api.openai.com/v1/responses/{rid}",
                                   headers={"Authorization": f"Bearer {key}"})
        try:
            with urllib.request.urlopen(g, timeout=120) as resp:
                data = json.load(resp)
        except urllib.error.HTTPError as e:
            print("HTTPError (poll)", e.code, e.read().decode("utf-8", "replace")[:1500])
            sys.exit(6)
        status = data.get("status")
        print(f"  [{waited}s] status={status}")
    print(f"final status={status} (effort={chosen})")

    text = data.get("output_text")
    if not text:
        chunks = []
        for item in data.get("output", []):
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    chunks.append(c.get("text", ""))
        text = "\n".join(chunks)
    out_path = os.path.join(HERE, "gpt55pro_review_output.md")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text or "(empty response)")

    usage = data.get("usage", {})
    it = usage.get("input_tokens")
    ot = usage.get("output_tokens")
    print("usage:", json.dumps(usage))
    if it and ot:
        print(f"ACTUAL cost = ${it/1e6*IN_PRICE + ot/1e6*OUT_PRICE:.2f} "
              f"(input {it:,} @ ${IN_PRICE:.0f}/M, output {ot:,} @ ${OUT_PRICE:.0f}/M)")
    print("review written:", out_path)


if __name__ == "__main__":
    main()
