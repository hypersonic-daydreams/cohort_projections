# PUB-2026 Finality Remediation Plan

**Date opened:** 2026-06-10
**Source:** `docs/reviews/2026-06-10-pub-2026-finality-rigor-review.md` (blocking items B1–B4, Ward disposition W, risk actions R1–R8, gaps G1–G7)
**Goal:** Close every blocking and non-blocking finding in dependency order, ending with an executed release QA checklist and final public numbers.

Ordering principles: (1) nothing numbers-dependent before the config lock + rerun; (2) the config lock needs evidence, so compute starts first; (3) engine instrumentation precedes the final rerun so production runs once; (4) the QA checklist is amended before it is executed; (5) doc-only items with no dependencies run in parallel with Stage 1 compute.

---

## Stage 0 — Parallel quick wins (doc-only, no dependencies; ~half day)

- [x] **0.1 (G6, R7)** methodology.md cleanup: fix §3.5 "constant" baseline fertility; fix §7.1/§9.1 restricted-growth attribution of the CBO additive reduction (broken cross-reference); fix §5h 20-vs-30-year convergence language; add ADR-064/065/066 to the §9.3 ADR table; add one-sentence provenance for the SDC-sourced 2020 migration snapshot in §5a. Add a skeleton "10. Limitations" section (final sensitivity magnitudes slot in at Stage 4).
- [x] **0.2 (R6, R8)** ADR housekeeping: add Implementation Results sections to ADR-040 (cite ADR-051's calibration result inside it), ADR-043, ADR-045 (record ex-post Benson −10.7%/Sioux −2.2%/Rolette −4.7% vs estimates), ADR-046; mark ADR-017 Superseded; add the rounding-artifact rationale + pre/post evidence note to ADR-062 for proactive disclosure.
- [x] **0.3 (R5a)** Quarantine stale `restricted_growth`/`high_growth` output directories (2026-02-26 vintage, pre-ADR-066) with a README pointing to the current baseline; full regeneration deferred to Stage 3 (optional).
- [x] **0.4 (G7)** Add "Gate 1b: Demographic Plausibility" to `release-qa-checklist.md`: final-run sanity review linked; 2025–2030 trajectory reconciled to PEP observed components; age/sex structure reviewed; largest county divergences vs SDC 2024 (Ward, Grand Forks) dispositioned.

**Stage 0 completed 2026-06-10** (uncommitted; verified by per-workstream adversarial review). Notes beyond the planned scope:

- Quarantine also covered a third stale tree found in the sweep: `data/projections/exports/geojson/` (GeoJSON county/place exports for all three scenario subtrees, built 2026-03-02 from the 2026-02-26-vintage projections — including a stale `baseline/` subtree). `data/projections/methodology_comparison/` (2025-12-28 SDC-replication artifacts) was noted but left alone; its refresh is already covered by Stage 4.3.
- Consumer scan: no active script reads the quarantined dirs unguarded — all references are explicit `--scenarios` CLI opt-ins or gated by `scenarios: ["baseline"]` in `config/projection_config.yaml`.
- `docs/governance/adrs/README.md` reconciled: missing ADR-056 row added to the index; status-summary and footer counts corrected (70 ADRs: 59 accepted / 2 proposed / 1 rejected / 4 superseded / 4 child).
- The three `README_STALE.md` markers inside `data/projections/` are not gitignored (only data file extensions are); a blanket `git add .` will stage them — intended, they are documentation markers.

## Stage 1 — Evidence generation (compute-heavy; launch first, overlap with Stage 0; ~1–2 days)

- [x] **1.1 (R1, G2)** Sensitivity decomposition on the forward projection — **done 2026-06-11**. One-factor forward runs (ref/cbo_off/fert_off/d3_hold15/gq075) recorded in ADR-067 F4 and the ADR-065 defensibility memo. GQ 0.75 was feasible config-only (EXP-C ran). Reference run double-verified production reproducibility (exact match after the mortality-file fix).
- [x] **1.2 (B1 input, R2, R3)** Benchmark the disposition candidates — **done 2026-06-11, but reshaped by ADR-067**. The original walk-forward bundles were found to run on adjustment-contaminated rate inputs (ADR-067 F1); the **authoritative evidence is the 2026-06-11 raw-base matrix** (college-fix / blend-70 / blend-100 / williams-out / gq-075 / production-lock vs the true champion). Williams-in/out and blend variants are segmented there.
- [ ] **1.3 (G1)** Naive-method value-add comparison promised by ADR-063: run carry-forward / linear-trend / average-growth benchmark runners against existing walk-forward results (origins 2010/2015/2020), stratified by county type and horizon; record in `docs/reviews/`. **Note (ADR-067):** must run on the corrected raw rate base, not the contaminated one.

## Stage 2 — Decision gate (Tier-3 / human verdict; ~half day after Stage 1 results)

- [x] **2.1 (B1, R2, R3)** **USER DECISION — done 2026-06-11.** ADR-061 dispositioned: D1 accept, D2 accept (GQ 0.75), D3 reject for release, D4 accept minus Williams (ADR-067); EXP-B blend 0.7 rejected for production (state-accuracy cost). Decision record Approved; ADR-061 status Accepted; `m2026r1`/`cfg-20260611-production-lock` promoted to `county_champion`. `projection_config.yaml` matches the promoted profile (verified).
- [x] **2.2 (R1, R4)** **Done 2026-06-12** — `docs/reviews/2026-06-12-adr-065-defensibility-memo.md`: accepted-conservatism rationale with quantified magnitudes; both CBO adjustments affirmed (Tier-3); GQ 0.75 already adopted; D3 rejected (premise was a harness artifact). EXP-C ran, so no f=1.0 blocker memo needed.

## Stage 3 — Produce the final run (~half day + run time)

- [ ] **3.1 (G3a)** Components-of-change instrumentation **before** the rerun: persist annual state-level births/deaths/net-migration per run (engine change in `cohort_projections/core/cohort_component.py` or a post-hoc derivation script); tests pass.
- [ ] **3.2 (G4a)** Add "State Age-Sex Detail" sheet (5-year groups × sex, key years 2025–2055) to the public workbook export; update `public-download-spec.md` + data dictionary.
- [ ] **3.3 (B1 completion)** Production rerun under the locked config; record run metadata (run ID, method, config hash) for QA Gate 1.
- [ ] **3.4 (R5b)** Regenerate alias/inactive scenario outputs under the locked config, or formally retire them (extend the 0.3 quarantine to permanent with rationale).

## Stage 4 — Validate the final run (~1 day)

- [ ] **4.1 (B3a, G3b)** Dated sanity review of the final run in `docs/reviews/`: 2025–2030 trajectory vs PEP Vintage 2025 observed components with explicit reconciliation of the early dip to the CBO ramp (f(2025)=0.20); projected 2025–2030 components vs PEP observed; age-structure/sex-ratio plausibility; 53-county scan; scenario-independence/aggregation checks.
- [ ] **4.2 (W)** **USER DECISION (choice of rationale):** Ward + Grand Forks written disposition — corrective ADR (e.g., evaluate ADR-055 Phase 2 × college-smoothing double-dampening) **or** accepted-divergence rationale with public-facing narrative (MAFB/MISU anchors, 2020–2025 observed decline). Do not rely on the inactive high_growth floor.
- [ ] **4.3 (B2)** Refresh `docs/methodology_comparison_sdc_2024.md` against final numbers: real SDC gaps, trajectory-shape change, honest framing of state-level long-horizon APE weakness vs county/bias/recent-origin strengths; finalize the methodology.md Limitations numbers (G6 completion).

## Stage 5 — Public artifacts and QA execution (~1–2 days)

- [ ] **5.1 (B4, G5)** Rewrite public PDF copy: baseline-only ADR-065 framing; the four ADR-042 caveats with refreshed values near the first statewide exhibit; dip explanation; Ward narrative from 4.2; "How accurate have past projections been?" track-record paragraph (SDC 2018-vs-Census-2020 error + plain-language backtest error ranges by horizon).
- [ ] **5.2** Rebuild the public draft package from clean staging (consolidated workbook incl. new detail sheet + chart sheets; CSV = 1,922 rows; no stale-horizon files), embedding run/method/config identifiers.
- [ ] **5.3 (B3b)** Execute the full release QA checklist gate-by-gate (incl. new Gate 1b); run the ADR-042 banned-language pass; record sign-off. **End state: numbers are final.**

---

**Decision gates owned by the user:** 2.1 (ADR-061 verdict + EXP-B + Williams), 4.2 (Ward rationale choice). Everything else is agent-executable.
**Estimated total:** ~5–7 working days, dominated by Stage 1 compute and Stage 5 artifact work.
