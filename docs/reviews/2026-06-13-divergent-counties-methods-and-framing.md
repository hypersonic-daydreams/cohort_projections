# Divergent Counties — Methods, Analysis, and Public Framing Reference

**Date:** 2026-06-13
**Purpose:** Durable internal reference for the three counties whose 2026 locked-config
projections most diverge from naive expectation or from the 2024 SDC predecessor —
**Williams (+52%), Ward (−13%), Grand Forks (−4%)**. Captures, for each: what the method
does and why (methods), the supporting evidence (analysis), and how to talk about it in
public materials (discussion/framing). Written so the public PDF/Q&A copy (PUB-2026 Stage 5)
can be drafted from it later without re-deriving the reasoning.
**Status:** User-signed-off 2026-06-13 (the Williams disposition and the locked numbers).
**Companions:** [ADR-067](../governance/adrs/067-ward-grand-forks-divergence-investigation.md)
(investigation + evidence), [2026-06-13-locked-run-sanity-check.md](2026-06-13-locked-run-sanity-check.md)
(run validation), [final-run-metadata.md](../plans/2026-public-projection-release-handoff/final-run-metadata.md)
(run provenance), [ADR-065 defensibility memo](2026-06-12-adr-065-defensibility-memo.md) (CBO assumptions).

---

## 0. Why these three, and the one mechanism they share

All three turn on **college-age migration smoothing** (ADR-049/061) and/or the **CBO international-migration assumption** (ADR-065). College-age smoothing blends a county's ages-15–29 net-migration rates 50% toward the statewide average, to strip *enrollment-driven* churn (students arriving/leaving) out of counties dominated by a university. The risk: applying it to a county whose young-adult migration is **not** enrollment-driven suppresses real economic dynamics. That risk is exactly what played out at Williams, and the same lens clarifies Ward and Grand Forks.

A key reframing from the 53-county scan (sanity check §5): **the high-growth counties are all Bakken oil — McKenzie +76%, Williams +52%, Billings +49% — and their projected net migration is at or below recent observed.** Their growth is compounding natural increase from very young (energy-era) age structures, not boom-migration extrapolation. Williams belongs to this group; the 2026 change simply stopped holding it apart from its peers.

---

## 1. Williams County (+52%; 41,767 → 63,295)

### Methods — what changed
ADR-061 Decision 4 (2026-03) added Williams to the college-smoothing list (Williston State College). ADR-067 (2026-06) **removed it** for the locked config. This reverts Williams to the *original champion model's* treatment — the champion never smoothed Williams; D4 was the addition, now undone. Williams still receives oil-boom dampening and male dampening (ADR-040); only the college brake is removed.

### Analysis — why removal is correct
- **Backtest:** on the clean (ADR-067) walk-forward, removing Williams from smoothing **improved** its historical error: county MAPE 23.38 → 22.48 (−0.90pp). The model predicts Williams' actual past *better* without the smoothing.
- **Mechanism:** WSC is only **1.5%** of county population — below the ADR's own 2.5% inclusion threshold. Williams' ages-15–29 in-migration is driven by **oil-field employment**, not enrollment. Smoothing it toward the (much lower) state average mis-classified economic migration as enrollment noise and averaged it away. ADR-061 had itself flagged a double-dampening risk for Williams (oil dampening × college smoothing); ADR-067 confirmed and resolved it.
- **Forward plausibility:** Williams' projected net migration settles to ≈ **+388/yr — about half its recent observed +780/yr (2023–2025)**. The +52% is therefore **conservative on migration**; it is driven mainly by natural increase from a young age structure (2035: births 541 vs deaths 196). Williams is consistent with its peer oil counties McKenzie (+76%, never smoothed) and Billings (+49%).
- **Magnitude:** Williams is the dominant driver of the locked run coming in +12,538 above the provisional baseline at 2055 (+8,978 of it is Williams alone; ADR-067 F4 / final-run-metadata).

### Discussion / public framing
- **The honest headline:** Williams is the state's second-fastest-growing county (after McKenzie), continuing — at a *moderated* pace — the energy-era growth of the 2010s.
- **Pre-empt the obvious challenge** ("aren't you just extrapolating the oil boom?"): No. The projection assumes Williams' net in-migration runs at roughly **half** its recent observed rate; most of the projected growth is births exceeding deaths in an unusually young population, not new arrivals.
- **Tie to method quality, not a thumb on the scale:** the 2026 model corrected an earlier over-smoothing that had been suppressing Williams; the correction makes Williams *more* accurate against its own history and consistent with comparable oil counties.

## 2. Ward County (−13%; the genuine sign flip vs SDC 2024 +23%)

### Methods
No special 2026 suppression beyond the standard pipeline. Ward (Minot, Minot AFB, Minot State) receives the same residual-migration + convergence treatment as other counties; MISU (2.9% of population) keeps Ward in the college-smoothing list, but ADR-067 confirmed that is not a material suppressor here (the GQ/smoothing levers move Ward only modestly; F4).

### Analysis
- **Observed data drives it:** Ward net migration was negative every year 2020–2025 (−129, −947, −1,044, −951, −483, −392; cumulative ≈ −3,950). The projection continues an observed decline, it does not invent one.
- **ADR-067 F4 forward decomposition:** all method/assumption levers combined move Ward only ≈ +3,660 at 2055 against a ~9,250 projected decline — i.e. the **majority of Ward's decline survives every method variant**. It is signal, not artifact.
- The SDC 2024 +23% predates the observed 2020–2025 reversal and the Vintage-2025 base.

### Discussion / public framing
- **This is the number most likely to be challenged** (state's 4th-largest county; SDC said +23%). Lead with the observed data: Ward has seen net out-migration in every year since 2020.
- Note the institutional anchors (Minot AFB, MISU, regional hub) are **held constant** (GQ), so the decline is in the household/working-age population, not a base-closure assumption.
- Frame against the predecessor honestly: the 2024 SDC figure was made before the post-2020 reversal was observable in the data.

## 3. Grand Forks County (−4%; vs SDC growth)

### Methods
Standard pipeline; UND (13.4% of population) makes Grand Forks a genuine college county where smoothing is appropriate and retained.

### Analysis
- **ADR-067 F4 attribution:** Grand Forks' decline is ≈ **52% the disclosed CBO international-migration assumption** (+1,827 of the gap if CBO were off) and ≈ **41% the long-run convergence stance** (+1,456). Grand Forks turned positive in 2023–2025 *on international migration* (+684/+376/+560), exactly the component the CBO current-policy assumption deliberately reduces for 2025–2029.
- Therefore Grand Forks' modest decline is **assumption-driven, not county-rate artifact** — and the assumption is the disclosed, externally-grounded CBO outlook (ADR-065).

### Discussion / public framing
- Frame as a consequence of the **stated CBO immigration assumption**, which the whole baseline shares — not a Grand Forks-specific judgment. If a reader disagrees with the federal-immigration assumption, point them to the assumptions section; Grand Forks is simply where it shows most because UND-area recent growth was international.
- The decline is shallow (−4%) and concentrated after 2045; near-term Grand Forks is roughly flat.

---

## 4. Cross-cutting framing guidance for Stage 5

1. **Anchor every divergence in observed data or a disclosed assumption** — never in "the model says so." Williams: conservative migration + young structure. Ward: observed 2020–2025 out-migration. Grand Forks: the stated CBO assumption.
2. **Present the oil counties as a coherent group** (McKenzie, Williams, Billings) so Williams doesn't read as a lone anomaly.
3. **Pre-empt the two predictable challenges** in the Q&A/caveats: "extrapolating the oil boom?" (no — migration is at/below observed) and "why does Ward decline when SDC said growth?" (post-2020 observed reversal + newer base).
4. **Components table caveat** (from sanity check §3): published deaths are household-basis (~5,000/yr) and will look low versus PEP's ~7,000 total; label or annotate for the held-constant GQ population, or users will flag an apparent mortality error.
5. **Method-quality message, applied consistently:** the 2026 system corrected an over-smoothing at Williams and incorporates observed post-2020 reversals at Ward — both make it *more* faithful to the data than the predecessor, in opposite directions. That symmetry is the credibility story.
