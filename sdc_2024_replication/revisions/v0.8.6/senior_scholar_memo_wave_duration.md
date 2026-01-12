---
title: "Senior Scholar Memo (Verbatim): Wave Duration Refit, Right-Censoring, and Hazard Ratios"
date_created: 2026-01-07
status: "active"
related_tracker: "sdc_2024_replication/revisions/v0.8.6/remaining_tasks_v0.8.6.md"
related_response: "sdc_2024_replication/revisions/v0.8.6/response_to_critique.md"
scope: "Module 8 duration analysis refit (FY2021–FY2024 extension); right-censoring; hazard ratios; Monte Carlo inputs"
source: "Codex conversation (role: senior scholar in demography + statistical modeling)"
---

# Senior Scholar Memo (Verbatim): Wave Duration Refit, Right-Censoring, and Hazard Ratios

## Context
This memo records (verbatim) the “senior scholar” guidance produced for the v0.8.6 remaining task:

- `Refit wave duration metrics; reassess right-censoring and hazard ratios.`

The guidance below is intended to steer decisions about whether to (a) refit only with the extended FY2021–FY2024 refugee series, (b) revise wave definitions / censoring rules, and (c) propagate any resulting changes into Module 9’s Monte Carlo wave machinery.

## Verbatim Guidance
**A senior scholar’s framing (demography + statistical modeling)**
- Treat “wave duration” as a **measurement model** for a latent process (program surges), not a natural law of migration; your goal is **stable, interpretable, reproducible** duration summaries that feed the Monte Carlo engine without quietly redefining “wave.”
- Separate “new data” effects from “new definitions” effects. If you change both at once, you can’t credibly say you “reassessed right-censoring.”
- Make one **pre-declared primary specification** and a small set of **sensitivity specifications** (gap tolerance, baseline window, minimum absolute arrivals). Put the sensitivities in an appendix; don’t chase the “best-looking” hazard ratios.

**How they’d answer the next steps we listed**
- **1) Refit with FY2024 endpoint (no definition changes): Yes, but lock comparability.** Rerun Module 8 to regenerate `sdc_2024_replication/scripts/statistical_analysis/results/module_8_duration_analysis.json` and `.../module_8_hazard_model.json`, *using the exact same wave-definition inputs as the prior run*. They would explicitly flag that `module_8_duration_analysis.py`’s “baseline = first half of sample” logic means the baseline window changes when the endpoint extends—so you must either (a) fix the baseline window, or (b) run a controlled comparison that holds it fixed to the earlier implied window.
- **2) Compare headline metrics: Absolutely, and do it as a delta table.** Show (old vs new): censoring rate, KM median duration, KM survival at 2/3/5 years, Cox concordance, and the key hazard ratios (especially `log_intensity`, `peak_arrivals`, “early/late” timing, region dummies). The scholar will want to see whether changes are “small but expected” (reduced censoring) or “structural” (sign flips / big shifts).
- **3) Audit “censored vs paused”: Yes—because it’s the substantive critique.** Identify waves that were censored at FY2020 in the old run and see whether they (a) end by FY2024, or (b) continue, or (c) dip below threshold and then reappear (pause/resumption). That distinction drives whether we need taxonomy changes.
- **4) Decide on taxonomy/censoring changes: Only after (3), and only as sensitivity first.** Examples: allowing a 1-year gap below threshold (treat “pause” as continuous), or defining termination as “below threshold for ≥k years”. If you want this, we should run it as a sensitivity analysis first and then decide which definition becomes primary.
- **5) Propagate to scenarios: Yes, but isolate the effect.** Since Module 9’s wave machinery loads `module_8_hazard_model.json`/`module_8_duration_analysis.json` (via `sdc_2024_replication/scripts/statistical_analysis/module_08_duration/wave_registry.py`), they’d rerun Module 9 only after Module 8 is finalized, and they’d record “scenario deltas attributable to wave-model update” (not mixed with other changes).

**How they’d answer the two decision questions**
- **Minimal refit vs taxonomy change?** Primary = **minimal refit** (extend endpoint, same wave definition), plus **taxonomy sensitivity** in appendix. Promote taxonomy changes only if the “paused wave” problem is empirically large and substantively consequential.
- **Baseline fixed vs drifting with sample length?** Baseline should be **fixed and concept-driven**, not an artifact of sample length. At minimum, you need a “hold baseline constant” run so you can attribute differences to reduced right-censoring. If you pick a new baseline window for substantive reasons, treat that as a separate specification choice and report it transparently.

If you want, I can propose a concrete “spec grid” (primary + 2–3 sensitivities) that matches your paper narrative and keeps the scope tight enough to get this task checked off in `sdc_2024_replication/revisions/v0.8.6/remaining_tasks_v0.8.6.md`.
